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

        // Build the templated prompt from the conversation.
        const prompt = self.buildPrompt(model, req.messages) catch {
            self.emitErr("failed to format prompt", .{});
            return;
        };
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
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_top_k(req.params.top_k));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_top_p(req.params.top_p, 1));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_temp(req.params.temperature));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_dist(req.params.seed));

        const n_ctx = c.llama_n_ctx(ctx);
        const gen_start = nowMs();

        // Canonical streaming loop (see llama.cpp simple.cpp): decode the prompt
        // batch, sample, emit, then feed the new token back via batch_get_one.
        var batch = c.llama_batch_get_one(@constCast(tokens.ptr), @intCast(tokens.len));
        var cur_token: c.llama_token = 0;
        var produced: u64 = 0;
        var n_past: u32 = @intCast(tokens.len);

        while (!self.job.cancelRequested()) {
            const rc = c.llama_decode(ctx, batch);
            if (rc != 0) {
                if (produced == 0) self.emitErr("decode failed (rc={d})", .{rc});
                break;
            }
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

            batch = c.llama_batch_get_one(&cur_token, 1);
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
        // Convert to llama_chat_message[] with null-terminated role/content.
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
            chat[i] = .{
                .role = roleStr(m.role),
                .content = content_z.ptr,
            };
        }

        const tmpl = c.llama_model_chat_template(model, null); // may be null

        // Apply template; grow the buffer if needed.
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
            // No template available — fall back to a plain concatenation.
            self.gpa.free(out);
            return self.fallbackPrompt(messages);
        }
        return self.gpa.realloc(out, @intCast(n)) catch out[0..@intCast(n)];
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
