//! In-process llama.cpp chat backend. A single worker thread owns the model and
//! runs the decode/sample loop, streaming each accepted token back to the UI
//! through a `Channel`. The UI submits a request (the full conversation + model
//! path + sampling params) and drains tokens once per frame.
//!
//! For simplicity and correctness, generation builds the whole templated prompt
//! and runs it through a freshly-created context each turn (the expensive model
//! load is cached). KV-cache reuse is a later optimization.

const std = @import("std");
const channel = @import("../channel.zig");
const jinja = @import("../jinja.zig");

pub const c = @cImport({
    @cInclude("llama.h");
});

// std's Mutex/Condition moved under the new std.Io model (they need an `Io`
// handle). For a plain blocking worker thread, libc pthreads are simpler and
// already available.
const pt = @cImport({
    @cInclude("pthread.h");
    @cInclude("time.h");
});

fn nowMs() i64 {
    var ts: pt.struct_timespec = undefined;
    _ = pt.clock_gettime(pt.CLOCK_MONOTONIC, &ts);
    return @as(i64, @intCast(ts.tv_sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.tv_nsec)), 1_000_000);
}

pub const Role = enum { system, user, assistant };

/// A conversation message owned by a request (heap-duplicated by `submit`).
pub const ReqMessage = struct {
    role: Role,
    content: []u8,
};

pub const Params = struct {
    temperature: f32 = 0.7,
    top_k: i32 = 40,
    top_p: f32 = 0.95,
    n_ctx: u32 = 4096,
    n_threads: i32 = 4,
    use_gpu: bool = true,
    seed: u32 = 0xFFFFFFFF, // LLAMA_DEFAULT_SEED
};

pub const Event = union(enum) {
    token: []u8, // a piece of text (UI frees after appending)
    done: struct { tokens: u64, ms: u64 },
    err: []u8, // error message (UI frees after surfacing)
};

const Request = struct {
    model_path: []u8,
    messages: []ReqMessage,
    params: Params,
    /// When true, `messages[0].content` is a raw prompt fed verbatim to the
    /// tokenizer (no chat template) — used by the HTTP `/v1/completions` endpoint.
    raw: bool = false,
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    events: channel.Channel(Event),
    job: channel.JobState = .{},

    // Worker thread + request signaling (pthread primitives, initialized in
    // `start` once the Backend is at its final address).
    thread: ?std.Thread = null,
    mutex: pt.pthread_mutex_t = undefined,
    cond: pt.pthread_cond_t = undefined,
    sync_ready: bool = false,
    shutdown: bool = false,
    has_request: bool = false,
    request: ?Request = null,

    // Owned by the worker thread once running.
    model: ?*c.llama_model = null,
    loaded_path: ?[]u8 = null,

    /// Set true once a model is loaded; read by the UI to drive the status dot.
    model_ready: std.atomic.Value(bool) = .init(false),

    pub fn init(gpa: std.mem.Allocator) Backend {
        return .{ .gpa = gpa, .events = channel.Channel(Event).init(gpa) };
    }

    /// Start the worker thread and initialize the llama backend (idempotent).
    /// pthread primitives are initialized here (not in `init`) so they live at
    /// the Backend's final address.
    pub fn start(self: *Backend) !void {
        if (self.thread != null) return;
        _ = pt.pthread_mutex_init(&self.mutex, null);
        _ = pt.pthread_cond_init(&self.cond, null);
        self.sync_ready = true;
        c.llama_backend_init();
        c.llama_log_set(quietLog, null);
        self.thread = try std.Thread.spawn(.{}, workerMain, .{self});
    }

    pub fn deinit(self: *Backend) void {
        if (self.thread) |th| {
            _ = pt.pthread_mutex_lock(&self.mutex);
            self.shutdown = true;
            _ = pt.pthread_cond_signal(&self.cond);
            _ = pt.pthread_mutex_unlock(&self.mutex);
            th.join();
        }
        if (self.sync_ready) {
            _ = pt.pthread_mutex_destroy(&self.mutex);
            _ = pt.pthread_cond_destroy(&self.cond);
        }
        if (self.model) |m| c.llama_model_free(m);
        if (self.loaded_path) |p| self.gpa.free(p);
        self.freePendingRequest();
        self.events.deinit();
    }

    fn freePendingRequest(self: *Backend) void {
        if (self.request) |req| {
            self.gpa.free(req.model_path);
            for (req.messages) |m| self.gpa.free(m.content);
            self.gpa.free(req.messages);
            self.request = null;
        }
    }

    /// True while a generation is in flight (for `busyCheck`).
    pub fn isBusy(self: *Backend) bool {
        return self.job.isRunning();
    }

    pub fn cancel(self: *Backend) void {
        self.job.requestCancel();
    }

    /// Free the cached model to release its memory, without stopping the worker.
    /// Safe to call from the UI thread: it takes the worker mutex and bails if a
    /// generation is in flight (the worker only touches `model` while a job runs
    /// or while parked in `cond_wait` holding the mutex, so holding it here means
    /// the worker can't load/free concurrently). The next submit reloads on demand.
    pub fn unload(self: *Backend) void {
        if (self.thread == null) return;
        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        if (self.job.isRunning()) return;
        if (self.model) |m| {
            c.llama_model_free(m);
            self.model = null;
        }
        if (self.loaded_path) |p| {
            self.gpa.free(p);
            self.loaded_path = null;
        }
        self.model_ready.store(false, .release);
    }

    /// Submit a generation request. `messages` and `model_path` are copied, so
    /// the caller keeps ownership of its originals. Replaces any queued request.
    pub fn submit(self: *Backend, model_path: []const u8, messages: []const ReqMessage, params: Params) !void {
        const mp = try self.gpa.dupe(u8, model_path);
        errdefer self.gpa.free(mp);
        const msgs = try self.gpa.alloc(ReqMessage, messages.len);
        errdefer self.gpa.free(msgs);
        var filled: usize = 0;
        errdefer for (msgs[0..filled]) |m| self.gpa.free(m.content);
        for (messages, 0..) |m, i| {
            msgs[i] = .{ .role = m.role, .content = try self.gpa.dupe(u8, m.content) };
            filled = i + 1;
        }

        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        // Drop any previously-queued (not yet started) request.
        if (self.request) |old| {
            self.gpa.free(old.model_path);
            for (old.messages) |m| self.gpa.free(m.content);
            self.gpa.free(old.messages);
        }
        self.request = .{ .model_path = mp, .messages = msgs, .params = params };
        self.has_request = true;
        // Mark running now so the UI's busyCheck wakes the loop immediately.
        self.job.beginJob();
        _ = pt.pthread_cond_signal(&self.cond);
    }

    /// Like `submit`, but `prompt` is fed to the tokenizer verbatim (no chat
    /// template) — for the OpenAI `/v1/completions` endpoint.
    pub fn submitRaw(self: *Backend, model_path: []const u8, prompt: []const u8, params: Params) !void {
        const mp = try self.gpa.dupe(u8, model_path);
        errdefer self.gpa.free(mp);
        const msgs = try self.gpa.alloc(ReqMessage, 1);
        errdefer self.gpa.free(msgs);
        msgs[0] = .{ .role = .user, .content = try self.gpa.dupe(u8, prompt) };

        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        if (self.request) |old| {
            self.gpa.free(old.model_path);
            for (old.messages) |m| self.gpa.free(m.content);
            self.gpa.free(old.messages);
        }
        self.request = .{ .model_path = mp, .messages = msgs, .params = params, .raw = true };
        self.has_request = true;
        self.job.beginJob();
        _ = pt.pthread_cond_signal(&self.cond);
    }

    pub const EmbedResult = struct { vectors: [][]f32, n_embd: usize };

    /// Compute L2-normalized embeddings for each text. Runs SYNCHRONOUSLY on the
    /// caller's thread — the HTTP server calls this while holding its request
    /// lock, so the worker thread is idle and `self.model` access is exclusive.
    /// Vectors are allocated in `a` (the caller's arena).
    pub fn embed(self: *Backend, a: std.mem.Allocator, model_path: []const u8, params: Params, texts: []const []const u8) !EmbedResult {
        if (!self.ensureModel(model_path, params)) {
            self.drainDiscard();
            return error.ModelLoadFailed;
        }
        const model = self.model.?;
        const vocab = c.llama_model_get_vocab(model);
        const n_embd: usize = @intCast(c.llama_model_n_embd(model));

        var cparams = c.llama_context_default_params();
        cparams.n_ctx = params.n_ctx;
        cparams.n_threads = params.n_threads;
        cparams.n_threads_batch = params.n_threads;
        cparams.embeddings = true;
        cparams.pooling_type = c.LLAMA_POOLING_TYPE_MEAN; // one vector per input
        const ctx = c.llama_init_from_model(model, cparams) orelse return error.ContextFailed;
        defer c.llama_free(ctx);
        const memory = c.llama_get_memory(ctx);
        const n_batch: usize = @intCast(c.llama_n_batch(ctx));

        const vectors = try a.alloc([]f32, texts.len);
        for (texts, 0..) |text, i| {
            const tokens = try self.tokenize(vocab, text);
            defer self.gpa.free(tokens);
            if (tokens.len == 0) return error.EmptyInput;
            // A pooled-embedding batch must be <= n_batch; truncate over-long
            // inputs (embed the leading window) rather than abort in llama_decode.
            const use_len: usize = @min(tokens.len, n_batch);
            c.llama_memory_clear(memory, true); // fresh sequence per input
            const batch = c.llama_batch_get_one(@constCast(tokens.ptr), @intCast(use_len));
            if (c.llama_decode(ctx, batch) != 0) return error.DecodeFailed;
            const emb = c.llama_get_embeddings_seq(ctx, 0);
            if (emb == null) return error.NoEmbeddings;
            const vec = try a.alloc(f32, n_embd);
            var norm: f32 = 0;
            for (0..n_embd) |k| {
                vec[k] = emb[k];
                norm += emb[k] * emb[k];
            }
            if (norm > 0) {
                const inv = 1.0 / @sqrt(norm);
                for (vec) |*v| v.* *= inv;
            }
            vectors[i] = vec;
        }
        return .{ .vectors = vectors, .n_embd = n_embd };
    }

    /// Drop and free any pending events (used to clear a stray error after a
    /// failed synchronous op so it doesn't pollute a later request's drain).
    fn drainDiscard(self: *Backend) void {
        var tmp: std.ArrayList(Event) = .empty;
        defer tmp.deinit(self.gpa);
        self.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .token => |t| self.gpa.free(t),
            .err => |e| self.gpa.free(e),
            .done => {},
        };
    }

    // --- worker thread ----------------------------------------------------

    fn workerMain(self: *Backend) void {
        while (true) {
            _ = pt.pthread_mutex_lock(&self.mutex);
            while (!self.has_request and !self.shutdown)
                _ = pt.pthread_cond_wait(&self.cond, &self.mutex);
            if (self.shutdown) {
                _ = pt.pthread_mutex_unlock(&self.mutex);
                return;
            }
            const req = self.request.?;
            self.request = null;
            self.has_request = false;
            _ = pt.pthread_mutex_unlock(&self.mutex);

            self.process(req);

            self.gpa.free(req.model_path);
            for (req.messages) |m| self.gpa.free(m.content);
            self.gpa.free(req.messages);
            self.job.endJob();
        }
    }

    fn emitErr(self: *Backend, comptime f: []const u8, args: anytype) void {
        const msg = std.fmt.allocPrint(self.gpa, f, args) catch return;
        self.events.push(.{ .err = msg });
    }

    fn ensureModel(self: *Backend, path: []const u8, params: Params) bool {
        if (self.loaded_path) |lp| {
            if (std.mem.eql(u8, lp, path) and self.model != null) return true;
        }
        // (Re)load.
        if (self.model) |m| {
            c.llama_model_free(m);
            self.model = null;
            self.model_ready.store(false, .release);
        }
        if (self.loaded_path) |lp| {
            self.gpa.free(lp);
            self.loaded_path = null;
        }
        const path_z = self.gpa.dupeZ(u8, path) catch return false;
        defer self.gpa.free(path_z);

        var mparams = c.llama_model_default_params();
        mparams.n_gpu_layers = if (params.use_gpu) 999 else 0;
        const model = c.llama_model_load_from_file(path_z.ptr, mparams);
        if (model == null) {
            self.emitErr("failed to load model: {s}", .{path});
            return false;
        }
        self.model = model;
        self.loaded_path = self.gpa.dupe(u8, path) catch null;
        self.model_ready.store(true, .release);
        return true;
    }

    fn process(self: *Backend, req: Request) void {
        if (!self.ensureModel(req.model_path, req.params)) return;
        const model = self.model.?;
        const vocab = c.llama_model_get_vocab(model);

        // Raw mode (/v1/completions): tokenize the prompt verbatim. Otherwise
        // build the templated prompt from the conversation.
        const prompt = if (req.raw)
            (self.gpa.dupe(u8, if (req.messages.len > 0) req.messages[0].content else "") catch {
                self.emitErr("out of memory", .{});
                return;
            })
        else
            (self.buildPrompt(model, req.messages) catch {
                self.emitErr("failed to format prompt", .{});
                return;
            });
        defer self.gpa.free(prompt);

        // Tokenize (with special tokens, since the template embeds them).
        const tokens = self.tokenize(vocab, prompt) catch {
            self.emitErr("tokenization failed", .{});
            return;
        };
        defer self.gpa.free(tokens);
        if (tokens.len == 0) {
            self.emitErr("empty prompt after tokenization", .{});
            return;
        }

        // Fresh context for this turn.
        var cparams = c.llama_context_default_params();
        cparams.n_ctx = req.params.n_ctx;
        cparams.n_threads = req.params.n_threads;
        cparams.n_threads_batch = req.params.n_threads;
        const ctx = c.llama_init_from_model(model, cparams) orelse {
            self.emitErr("failed to create context", .{});
            return;
        };
        defer c.llama_free(ctx);

        // Sampler chain: top_k -> top_p -> temp -> dist.
        const smpl = c.llama_sampler_chain_init(c.llama_sampler_chain_default_params()) orelse {
            self.emitErr("failed to create sampler", .{});
            return;
        };
        defer c.llama_sampler_free(smpl);
        // Repetition penalty FIRST — without it, models (especially on an
        // off-distribution prompt) collapse into "x_x_x" loops. last_n=64, 1.1x.
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_penalties(64, 1.1, 0.0, 0.0));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_top_k(req.params.top_k));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_top_p(req.params.top_p, 1));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_temp(req.params.temperature));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_dist(req.params.seed));

        const n_ctx = c.llama_n_ctx(ctx);
        const n_batch: usize = @intCast(c.llama_n_batch(ctx));
        const gen_start = nowMs();

        // A prompt longer than the context can't fit even after decoding — bail
        // with a clear message instead of overflowing the KV cache.
        if (tokens.len >= n_ctx) {
            self.emitErr("prompt is {d} tokens but the context is only {d}; shorten the conversation or disable agent mode", .{ tokens.len, n_ctx });
            return;
        }

        // Decode the prompt in n_batch-sized chunks: llama_decode asserts a single
        // batch is <= n_batch (default 2048), which long agent/tool prompts exceed.
        var fed: usize = 0;
        while (fed < tokens.len) {
            if (self.job.cancelRequested()) {
                self.events.push(.{ .done = .{ .tokens = 0, .ms = 0 } });
                return;
            }
            const chunk = @min(n_batch, tokens.len - fed);
            const pbatch = c.llama_batch_get_one(@constCast(tokens.ptr + fed), @intCast(chunk));
            if (c.llama_decode(ctx, pbatch) != 0) {
                self.emitErr("decode failed (prompt)", .{});
                return;
            }
            fed += chunk;
        }

        // If the chat template opened a reasoning block at the end of the prompt
        // (e.g. Qwen3 appends `<think>`, gemma a thought channel), the model's
        // output starts INSIDE that block. Emit the open tag first so the
        // downstream parser treats the leading output as reasoning, not answer.
        if (!req.raw) {
            if (promptOpensThink(prompt)) |open| {
                if (self.gpa.dupe(u8, open)) |d| self.events.push(.{ .token = d }) else |_| {}
            }
        }

        // Streaming loop (see llama.cpp simple.cpp): sample from the last logits,
        // emit, then feed the new token back via batch_get_one.
        var cur_token: c.llama_token = 0;
        var produced: u64 = 0;
        var n_past: u32 = @intCast(tokens.len);

        while (!self.job.cancelRequested()) {
            cur_token = c.llama_sampler_sample(smpl, ctx, -1);
            if (c.llama_vocab_is_eog(vocab, cur_token)) break;

            var piece: [256]u8 = undefined;
            const n = c.llama_token_to_piece(vocab, cur_token, &piece, piece.len, 0, true);
            if (n > 0) {
                const dup = self.gpa.dupe(u8, piece[0..@intCast(n)]) catch break;
                self.events.push(.{ .token = dup });
            }
            produced += 1;
            n_past += 1;
            if (n_past >= n_ctx) break;

            const tbatch = c.llama_batch_get_one(&cur_token, 1);
            if (c.llama_decode(ctx, tbatch) != 0) break;
        }

        const elapsed: u64 = @intCast(@max(0, nowMs() - gen_start));
        self.events.push(.{ .done = .{ .tokens = produced, .ms = elapsed } });
    }

    fn tokenize(self: *Backend, vocab: ?*const c.llama_vocab, text: []const u8) ![]c.llama_token {
        // First call with a generous bound; negative return = need more space.
        const cap: i32 = @intCast(text.len + 16);
        var buf = try self.gpa.alloc(c.llama_token, @intCast(cap));
        errdefer self.gpa.free(buf);
        const n = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), buf.ptr, cap, true, true);
        if (n < 0) {
            self.gpa.free(buf);
            const need: usize = @intCast(-n);
            buf = try self.gpa.alloc(c.llama_token, need);
            const n2 = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), buf.ptr, @intCast(need), true, true);
            if (n2 < 0) return error.TokenizeFailed;
            return buf[0..@intCast(n2)];
        }
        return buf[0..@intCast(n)];
    }

    fn buildPrompt(self: *Backend, model: ?*const c.llama_model, messages: []const ReqMessage) ![]u8 {
        const tmpl = c.llama_model_chat_template(model, null); // the GGUF's Jinja, or null
        if (tmpl != null) {
            const vocab = c.llama_model_get_vocab(model);
            var arena = std.heap.ArenaAllocator.init(self.gpa);
            defer arena.deinit();
            const merged = mergeSystemIntoUser(arena.allocator(), messages);

            // Primary: render the model's ACTUAL chat_template with our Jinja
            // engine, so any model (incl. ones llama.cpp's matcher doesn't know,
            // like gemma-4) gets its correct prompt format.
            if (try self.renderJinja(tmpl, messages, vocab)) |p| return p;
            if (merged) |mm| if (try self.renderJinja(tmpl, mm, vocab)) |p| return p;

            // Secondary: llama.cpp's hardcoded template matcher.
            if (try self.applyTemplate(tmpl, messages)) |p| return p;
            if (merged) |mm| if (try self.applyTemplate(tmpl, mm)) |p| return p;
        }
        return self.fallbackPrompt(messages);
    }

    /// Render the GGUF's Jinja `tmpl` with `messages`. We supply `eos_token` (used
    /// by templates to close turns) but leave `bos_token` empty and let the
    /// tokenizer add BOS (`add_special=true`), avoiding double-BOS. Returns the
    /// prompt (owned), or null if rendering failed (then the caller falls back).
    fn renderJinja(self: *Backend, tmpl: [*c]const u8, messages: []const ReqMessage, vocab: ?*const c.llama_vocab) !?[]u8 {
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const msgs_json = try buildMessagesJson(a, messages);
        var eos_buf: [128]u8 = undefined;
        const eos = tokenText(vocab, c.llama_vocab_eos(vocab), &eos_buf);
        var extra: std.Io.Writer.Allocating = .init(a);
        try extra.writer.writeAll("{\"bos_token\":\"\",\"eos_token\":");
        try jsonStr(&extra.writer, eos);
        try extra.writer.writeAll("}");
        return jinja.renderChat(self.gpa, std.mem.span(tmpl), msgs_json, null, extra.written(), true);
    }

    /// Apply the chat template to `messages`. Returns the rendered prompt (owned),
    /// or null if the template rejected the message set (n < 0).
    fn applyTemplate(self: *Backend, tmpl: [*c]const u8, messages: []const ReqMessage) !?[]u8 {
        const chat = try self.gpa.alloc(c.llama_chat_message, messages.len);
        defer self.gpa.free(chat);
        var zbufs = try self.gpa.alloc([]u8, messages.len);
        defer {
            for (zbufs) |zb| self.gpa.free(zb);
            self.gpa.free(zbufs);
        }
        for (messages, 0..) |m, i| {
            const content_z = try self.gpa.dupeZ(u8, m.content);
            zbufs[i] = content_z;
            chat[i] = .{ .role = roleStr(m.role), .content = content_z.ptr };
        }
        var cap: i32 = 0;
        for (messages) |m| cap += @intCast(m.content.len + 32);
        var out = try self.gpa.alloc(u8, @intCast(@max(cap, 256)));
        errdefer self.gpa.free(out);
        var n = c.llama_chat_apply_template(tmpl, chat.ptr, messages.len, true, out.ptr, @intCast(out.len));
        if (n > @as(i32, @intCast(out.len))) {
            self.gpa.free(out);
            out = try self.gpa.alloc(u8, @intCast(n));
            n = c.llama_chat_apply_template(tmpl, chat.ptr, messages.len, true, out.ptr, @intCast(out.len));
        }
        if (n < 0) {
            self.gpa.free(out);
            return null;
        }
        return self.gpa.realloc(out, @intCast(n)) catch out[0..@intCast(n)];
    }

    /// Build a copy of `messages` with all `system` content folded into the first
    /// user turn (gemma-style models reject a system role). Returns null if there
    /// was no system message. Everything is allocated in `a`.
    fn mergeSystemIntoUser(a: std.mem.Allocator, messages: []const ReqMessage) ?[]ReqMessage {
        var sys: std.ArrayList(u8) = .empty;
        var has_sys = false;
        for (messages) |m| {
            if (m.role != .system) continue;
            has_sys = true;
            if (sys.items.len > 0) sys.appendSlice(a, "\n\n") catch return null;
            sys.appendSlice(a, m.content) catch return null;
        }
        if (!has_sys) return null;

        var out: std.ArrayList(ReqMessage) = .empty;
        var injected = false;
        for (messages) |m| {
            if (m.role == .system) continue;
            if (!injected and m.role == .user) {
                const merged = std.fmt.allocPrint(a, "{s}\n\n{s}", .{ sys.items, m.content }) catch return null;
                out.append(a, .{ .role = .user, .content = merged }) catch return null;
                injected = true;
            } else {
                out.append(a, .{ .role = m.role, .content = m.content }) catch return null;
            }
        }
        if (!injected) out.insert(a, 0, .{ .role = .user, .content = sys.items }) catch return null;
        return out.items;
    }

    fn fallbackPrompt(self: *Backend, messages: []const ReqMessage) ![]u8 {
        var list: std.ArrayList(u8) = .empty;
        errdefer list.deinit(self.gpa);
        for (messages) |m| {
            try list.appendSlice(self.gpa, roleLabel(m.role));
            try list.appendSlice(self.gpa, ": ");
            try list.appendSlice(self.gpa, m.content);
            try list.append(self.gpa, '\n');
        }
        try list.appendSlice(self.gpa, "assistant: ");
        return list.toOwnedSlice(self.gpa);
    }
};

/// Drop ggml/llama INFO/DEBUG chatter (e.g. Metal pipeline compilation); keep
/// warnings and errors on stderr.
fn quietLog(level: c.ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    if (level >= c.GGML_LOG_LEVEL_WARN) {
        std.debug.print("{s}", .{text});
    }
}

fn roleStr(r: Role) [*:0]const u8 {
    return switch (r) {
        .system => "system",
        .user => "user",
        .assistant => "assistant",
    };
}

fn roleLabel(r: Role) []const u8 {
    return switch (r) {
        .system => "system",
        .user => "user",
        .assistant => "assistant",
    };
}

/// Serialize messages to a JSON array `[{"role","content"},…]` for the Jinja engine.
fn buildMessagesJson(a: std.mem.Allocator, messages: []const ReqMessage) ![]u8 {
    var b: std.Io.Writer.Allocating = .init(a);
    const w = &b.writer;
    try w.writeByte('[');
    for (messages, 0..) |m, i| {
        if (i > 0) try w.writeByte(',');
        try w.print("{{\"role\":\"{s}\",\"content\":", .{roleLabel(m.role)});
        try jsonStr(w, m.content);
        try w.writeByte('}');
    }
    try w.writeByte(']');
    return b.written();
}

/// Write `s` as a JSON string literal (with quotes), escaping per RFC 8259.
fn jsonStr(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |ch| switch (ch) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => if (ch < 0x20) try w.print("\\u{x:0>4}", .{ch}) else try w.writeByte(ch),
    };
    try w.writeByte('"');
}

/// If `prompt` ends inside an unclosed reasoning block (the template opened one
/// for the generation turn), return the open tag; else null.
fn promptOpensThink(prompt: []const u8) ?[]const u8 {
    const pairs = [_]struct { open: []const u8, close: []const u8 }{
        .{ .open = "<think>", .close = "</think>" },
        .{ .open = "<|channel>thought", .close = "<channel|>" },
    };
    for (pairs) |p| {
        const lo = std.mem.lastIndexOf(u8, prompt, p.open) orelse continue;
        const lc = std.mem.lastIndexOf(u8, prompt, p.close);
        if (lc == null or lc.? < lo) return p.open;
    }
    return null;
}

/// Render a single token to its text (special tokens included). Empty on failure.
fn tokenText(vocab: ?*const c.llama_vocab, token: c.llama_token, buf: []u8) []const u8 {
    if (token < 0) return "";
    const n = c.llama_token_to_piece(vocab, token, buf.ptr, @intCast(buf.len), 0, true);
    if (n <= 0) return "";
    return buf[0..@intCast(n)];
}
