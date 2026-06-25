//! A tiny OpenAI-compatible HTTP server exposing the chat model over
//! `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`,
//! and `GET /health`. Bound to localhost by default — "runs on-device".
//!
//! The Zig-0.16 socket plumbing (the `Conn` reader/writer over `std.Io.net`,
//! the poll-with-timeout accept loop) follows the pattern in mlx-serve's
//! `src/server.zig` (../mlx-serve, MIT). The wire format and routing are our own.
//!
//! Design: the server owns its OWN `llama.Backend` (a separate, lazily-loaded
//! model instance — so it never clobbers the GUI's chat backend or its event
//! channel). Requests are serialized through that single backend via `req_lock`;
//! concurrent clients queue. The model path comes from a caller-supplied callback
//! (the GUI's currently-selected chat model) so this module needn't import the app.

const std = @import("std");
const llama = @import("../backends/llama.zig");
const channel = @import("../channel.zig");
const agent = @import("../agent.zig");
const chat_parser = @import("chat_parser.zig");
const Io = std.Io;

/// The port the OpenAI-compatible server listens on (referenced by the tray).
pub const port: u16 = 8080;

pub const Config = struct {
    /// Bound to 0.0.0.0 (all interfaces) so other devices on the LAN can reach
    /// it. No auth — intended for trusted local networks.
    host: []const u8 = "0.0.0.0",
    port: u16 = port,
    n_ctx: u32 = 4096,
    n_threads: i32 = 8,
    /// Default cap on generated tokens when a request omits `max_tokens`.
    default_max_tokens: i64 = 2048,
};

/// Resolve the model path to serve. `out` is scratch; returns the path slice or
/// null when no chat model is selected. Implemented by the app over its state.
pub const ModelResolver = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, out: []u8) ?[]const u8,
};

/// Heap-boxed networking context (stable address: `threaded.io()` captures it).
const Net = struct {
    threaded: Io.Threaded,
    fn io(self: *Net) Io {
        return self.threaded.io();
    }
};

/// A connection's buffered reader/writer over a `std.Io.net.Stream` (Zig 0.16).
const Conn = struct {
    stream: std.Io.net.Stream,
    io: Io,
    read_buf: [16 * 1024]u8 = undefined,
    write_buf: [16 * 1024]u8 = undefined,
    read_state: std.Io.net.Stream.Reader = undefined,
    write_state: std.Io.net.Stream.Writer = undefined,

    fn init(c: *Conn, stream: std.Io.net.Stream, io: Io) void {
        c.stream = stream;
        c.io = io;
        c.read_state = stream.reader(io, &c.read_buf);
        c.write_state = stream.writer(io, &c.write_buf);
    }
    fn reader(c: *Conn) *Io.Reader {
        return &c.read_state.interface;
    }
    fn writer(c: *Conn) *Io.Writer {
        return &c.write_state.interface;
    }
    /// Read up to buf.len bytes (short read; 0 on EOF).
    fn read(c: *Conn, buf: []u8) !usize {
        var bufs: [1][]u8 = .{buf};
        return c.reader().readVec(&bufs) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => |e| e,
        };
    }
    fn writeAll(c: *Conn, data: []const u8) !void {
        try c.writer().writeAll(data);
    }
    fn flush(c: *Conn) !void {
        try c.writer().flush();
    }
    fn close(c: *Conn) void {
        c.flush() catch {};
        c.stream.close(c.io);
    }
};

pub const Server = struct {
    gpa: std.mem.Allocator,
    cfg: Config,
    resolver: ModelResolver,
    backend: llama.Backend,
    req_lock: channel.SpinLock = .{}, // serialize requests to the single backend
    shutdown: std.atomic.Value(bool) = .init(false),
    id_counter: std.atomic.Value(u64) = .init(0),
    thread: ?std.Thread = null,
    net: ?*Net = null,

    pub fn init(gpa: std.mem.Allocator, cfg: Config, resolver: ModelResolver) Server {
        return .{ .gpa = gpa, .cfg = cfg, .resolver = resolver, .backend = llama.Backend.init(gpa) };
    }

    pub fn start(self: *Server) !void {
        if (self.thread != null) return;
        const n = try self.gpa.create(Net);
        n.threaded = Io.Threaded.init(self.gpa, .{});
        self.net = n;
        self.thread = try std.Thread.spawn(.{}, listenLoop, .{self});
    }

    pub fn deinit(self: *Server) void {
        self.shutdown.store(true, .release);
        if (self.thread) |th| th.join(); // accept loop polls the flag every 1s
        self.backend.deinit();
        if (self.net) |n| {
            n.threaded.deinit();
            self.gpa.destroy(n);
        }
    }

    fn listenLoop(self: *Server) void {
        const io = self.net.?.io();
        const ip: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = self.cfg.port } };
        var server = ip.listen(io, .{ .reuse_address = true }) catch |e| {
            std.debug.print("api: failed to bind {s}:{d}: {s}\n", .{ self.cfg.host, self.cfg.port, @errorName(e) });
            return;
        };
        defer server.deinit(io);
        std.debug.print("api: OpenAI-compatible server on http://{s}:{d}/v1\n", .{ self.cfg.host, self.cfg.port });

        var poll_fds = [_]std.posix.pollfd{.{ .fd = server.socket.handle, .events = std.posix.POLL.IN, .revents = 0 }};
        while (!self.shutdown.load(.acquire)) {
            const r = std.posix.poll(&poll_fds, 1000) catch break;
            if (r == 0) continue; // timeout → re-check shutdown
            if (self.shutdown.load(.acquire)) break;
            const stream = server.accept(io) catch continue;
            var conn: Conn = undefined;
            conn.init(stream, io);
            self.handleConn(&conn) catch {};
            conn.close();
        }
    }

    fn handleConn(self: *Server, conn: *Conn) !void {
        var arena_state = std.heap.ArenaAllocator.init(self.gpa);
        defer arena_state.deinit();
        const a = arena_state.allocator();

        // --- read request (headers then body by Content-Length) ---
        var hdr: [16 * 1024]u8 = undefined;
        var total: usize = 0;
        var head_end: usize = 0;
        var content_len: usize = 0;
        while (total < hdr.len) {
            const n = conn.read(hdr[total..]) catch 0;
            if (n == 0) break;
            total += n;
            if (std.mem.indexOf(u8, hdr[0..total], "\r\n\r\n")) |he| {
                head_end = he + 4;
                content_len = findContentLength(hdr[0..he]);
                break;
            }
        }
        if (head_end == 0) return;

        const first_line_end = std.mem.indexOf(u8, hdr[0..head_end], "\r\n") orelse return;
        var it = std.mem.splitScalar(u8, hdr[0..first_line_end], ' ');
        const method = it.next() orelse return;
        const path = it.next() orelse return;

        // Read the full body (header bytes already read may include some of it).
        const total_size = head_end + content_len;
        const body = if (content_len > 0) blk: {
            const buf = a.alloc(u8, content_len) catch return;
            const have = total - head_end;
            if (have > 0) @memcpy(buf[0..have], hdr[head_end..total]);
            var got = have;
            while (got < content_len) {
                const n = conn.read(buf[got..]) catch break;
                if (n == 0) break;
                got += n;
            }
            break :blk buf[0..got];
        } else "";
        _ = total_size;

        // --- route ---
        if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path, "/health")) {
            try conn.writeAll("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}");
            try conn.flush();
            return;
        }
        if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path, "/v1/models")) {
            return self.handleModels(conn, a);
        }
        if (std.mem.eql(u8, method, "POST") and std.mem.startsWith(u8, path, "/v1/chat/completions")) {
            return self.handleChat(conn, a, body);
        }
        if (std.mem.eql(u8, method, "POST") and std.mem.startsWith(u8, path, "/v1/completions")) {
            return self.handleCompletions(conn, a, body);
        }
        if (std.mem.eql(u8, method, "POST") and std.mem.startsWith(u8, path, "/v1/embeddings")) {
            return self.handleEmbeddings(conn, a, body);
        }
        try sendError(conn, "404 Not Found", "not_found", "Unknown endpoint");
    }

    fn handleModels(self: *Server, conn: *Conn, a: std.mem.Allocator) !void {
        var path_buf: [1024]u8 = undefined;
        const model = self.resolver.func(self.resolver.ctx, &path_buf);
        // Status fields mirror mlx-serve: `loaded`/`state`/`bytes_resident` so the
        // GUI can poll this for its model-loaded indicator.
        const loaded = self.backend.model_ready.load(.acquire);
        var bytes: u64 = 0;
        if (model) |m| {
            if (loaded) {
                if (Io.Dir.cwd().statFile(conn.io, m, .{})) |st| bytes = st.size else |_| {}
            }
        }
        var body: Io.Writer.Allocating = .init(a);
        const w = &body.writer;
        try w.writeAll("{\"object\":\"list\",\"data\":[");
        if (model) |m| {
            try w.writeAll("{\"id\":");
            try writeJsonStr(w, modelId(m));
            try w.print(",\"object\":\"model\",\"owned_by\":\"zig-ai\",\"loaded\":{s},\"state\":\"{s}\",\"bytes_resident\":{d}}}", .{
                if (loaded) "true" else "false",
                if (loaded) "ready" else "available",
                bytes,
            });
        }
        try w.writeAll("]}");
        try sendJson(conn, "200 OK", body.written());
    }

    /// Legacy `/v1/completions`: a raw `prompt` (no chat template) → `text_completion`.
    fn handleCompletions(self: *Server, conn: *Conn, a: std.mem.Allocator, body: []const u8) !void {
        const CompReq = struct {
            model: ?[]const u8 = null,
            prompt: []const u8 = "",
            stream: bool = false,
            temperature: ?f32 = null,
            top_p: ?f32 = null,
            top_k: ?i32 = null,
            max_tokens: ?i64 = null,
        };
        const parsed = std.json.parseFromSlice(CompReq, a, body, .{ .ignore_unknown_fields = true }) catch {
            return sendError(conn, "400 Bad Request", "invalid_request_error", "Invalid JSON body (prompt must be a string)");
        };
        defer parsed.deinit();
        const req = parsed.value;

        var path_buf: [1024]u8 = undefined;
        const model_path = self.resolver.func(self.resolver.ctx, &path_buf) orelse {
            return sendError(conn, "503 Service Unavailable", "model_not_found", "No chat model selected in the app");
        };

        var params: llama.Params = .{};
        if (req.temperature) |t| params.temperature = t;
        if (req.top_p) |t| params.top_p = t;
        if (req.top_k) |t| params.top_k = t;
        params.n_ctx = self.cfg.n_ctx;
        params.n_threads = self.cfg.n_threads;
        const max_tokens: i64 = req.max_tokens orelse self.cfg.default_max_tokens;
        const id = self.id_counter.fetchAdd(1, .monotonic);
        const created = Io.Timestamp.now(conn.io, .real).toSeconds();

        self.req_lock.lock();
        defer self.req_lock.unlock();
        self.backend.start() catch {
            return sendError(conn, "500 Internal Server Error", "server_error", "could not start inference backend");
        };
        self.backend.submitRaw(model_path, req.prompt, params) catch {
            return sendError(conn, "500 Internal Server Error", "server_error", "could not submit request");
        };

        const r = self.collect(conn, a, max_tokens);
        if (r.err) |e| return sendError(conn, "500 Internal Server Error", "server_error", e);

        if (req.stream) {
            try conn.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
            var b: Io.Writer.Allocating = .init(a);
            const w = &b.writer;
            try compChunk(w, id, created, model_path, r.text, null);
            try sendSse(conn, b.written());
            var b2: Io.Writer.Allocating = .init(a);
            try compChunk(&b2.writer, id, created, model_path, "", r.finish);
            try sendSse(conn, b2.written());
            try conn.writeAll("data: [DONE]\n\n");
            try conn.flush();
            return;
        }
        var body_w: Io.Writer.Allocating = .init(a);
        const w = &body_w.writer;
        try w.print("{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{ id, created });
        try writeJsonStr(w, modelId(model_path));
        try w.writeAll(",\"choices\":[{\"index\":0,\"text\":");
        try writeJsonStr(w, r.text);
        try w.print(",\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{ r.finish, r.n_tok, r.n_tok });
        try sendJson(conn, "200 OK", body_w.written());
    }

    /// `/v1/embeddings`: `input` (string or array) → L2-normalized vectors.
    fn handleEmbeddings(self: *Server, conn: *Conn, a: std.mem.Allocator, body: []const u8) !void {
        const EmbReq = struct { model: ?[]const u8 = null, input: std.json.Value = .null };
        const parsed = std.json.parseFromSlice(EmbReq, a, body, .{ .ignore_unknown_fields = true }) catch {
            return sendError(conn, "400 Bad Request", "invalid_request_error", "Invalid JSON body");
        };
        defer parsed.deinit();

        // `input` may be a string or an array of strings.
        var texts: std.ArrayList([]const u8) = .empty;
        switch (parsed.value.input) {
            .string => |s| texts.append(a, s) catch {},
            .array => |arr| for (arr.items) |it| {
                if (it == .string) texts.append(a, it.string) catch {};
            },
            else => {},
        }
        if (texts.items.len == 0) {
            return sendError(conn, "400 Bad Request", "invalid_request_error", "`input` must be a string or array of strings");
        }

        var path_buf: [1024]u8 = undefined;
        const model_path = self.resolver.func(self.resolver.ctx, &path_buf) orelse {
            return sendError(conn, "503 Service Unavailable", "model_not_found", "No model selected in the app");
        };
        var params: llama.Params = .{};
        params.n_ctx = self.cfg.n_ctx;
        params.n_threads = self.cfg.n_threads;

        self.req_lock.lock();
        defer self.req_lock.unlock();
        const res = self.backend.embed(a, model_path, params, texts.items) catch {
            return sendError(conn, "500 Internal Server Error", "server_error", "embedding failed (model may not support embeddings)");
        };

        var body_w: Io.Writer.Allocating = .init(a);
        const w = &body_w.writer;
        try w.writeAll("{\"object\":\"list\",\"data\":[");
        for (res.vectors, 0..) |vec, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("{{\"object\":\"embedding\",\"index\":{d},\"embedding\":[", .{i});
            for (vec, 0..) |val, k| {
                if (k > 0) try w.writeAll(",");
                try w.print("{d}", .{val});
            }
            try w.writeAll("]}");
        }
        try w.writeAll("],\"model\":");
        try writeJsonStr(w, modelId(model_path));
        try w.writeAll(",\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}");
        try sendJson(conn, "200 OK", body_w.written());
    }

    const ToolFn = struct { name: []const u8 = "", description: ?[]const u8 = null, parameters: ?std.json.Value = null };
    const ToolDef = struct { type: ?[]const u8 = null, function: ToolFn = .{} };
    const ToolCallIn = struct {
        id: ?[]const u8 = null,
        function: struct { name: []const u8 = "", arguments: ?[]const u8 = null } = .{},
    };
    const InMsg = struct {
        role: []const u8 = "user",
        content: ?[]const u8 = null,
        tool_calls: ?[]const ToolCallIn = null,
        tool_call_id: ?[]const u8 = null,
    };
    const ChatReq = struct {
        model: ?[]const u8 = null,
        stream: bool = false,
        temperature: ?f32 = null,
        top_p: ?f32 = null,
        top_k: ?i32 = null,
        max_tokens: ?i64 = null,
        n_ctx: ?u32 = null, // non-standard: GUI passes its context-size setting
        n_threads: ?i32 = null,
        messages: []const InMsg = &.{},
        tools: ?[]const ToolDef = null,
    };

    fn handleChat(self: *Server, conn: *Conn, a: std.mem.Allocator, body: []const u8) !void {
        const parsed = std.json.parseFromSlice(ChatReq, a, body, .{ .ignore_unknown_fields = true }) catch {
            return sendError(conn, "400 Bad Request", "invalid_request_error", "Invalid JSON body");
        };
        defer parsed.deinit();
        const req = parsed.value;
        if (req.messages.len == 0) {
            return sendError(conn, "400 Bad Request", "invalid_request_error", "`messages` is required");
        }

        var path_buf: [1024]u8 = undefined;
        const model_path = self.resolver.func(self.resolver.ctx, &path_buf) orelse {
            return sendError(conn, "503 Service Unavailable", "model_not_found", "No chat model selected in the app");
        };

        const has_tools = req.tools != null and req.tools.?.len > 0;

        // Build the conversation. Content slices live in the arena until `submit`
        // dupes them. Tool-calling reuses agent.zig's `<tool_call>` protocol:
        //  - tools[] → a system message advertising them in that format
        //  - assistant tool_calls → reconstructed `<tool_call>` text (so the model
        //    sees its own prior call)
        //  - role:"tool" results → a "Tool result:" user message (ReAct-style)
        var msgs = std.ArrayList(llama.ReqMessage).empty;
        if (has_tools) {
            const tools_sys = buildToolsText(a, req.tools.?) catch "";
            if (tools_sys.len > 0) msgs.append(a, .{ .role = .system, .content = @constCast(tools_sys) }) catch {};
        }
        for (req.messages) |m| {
            if (std.mem.eql(u8, m.role, "tool")) {
                const txt = std.fmt.allocPrint(a, "Tool result:\n{s}", .{m.content orelse ""}) catch continue;
                msgs.append(a, .{ .role = .user, .content = txt }) catch {};
            } else if (m.tool_calls) |tcs| {
                var b: Io.Writer.Allocating = .init(a);
                const w = &b.writer;
                if (m.content) |c| w.writeAll(c) catch {};
                for (tcs) |tc| {
                    w.writeAll(agent.open_tag) catch {};
                    w.writeAll("{\"name\":") catch {};
                    writeJsonStr(w, tc.function.name) catch {};
                    w.writeAll(",\"arguments\":") catch {};
                    w.writeAll(tc.function.arguments orelse "{}") catch {};
                    w.writeAll("}") catch {};
                    w.writeAll(agent.close_tag) catch {};
                }
                msgs.append(a, .{ .role = .assistant, .content = @constCast(b.written()) }) catch {};
            } else if (m.content) |content| {
                msgs.append(a, .{ .role = mapRole(m.role), .content = @constCast(content) }) catch {};
            }
        }
        if (msgs.items.len == 0) {
            return sendError(conn, "400 Bad Request", "invalid_request_error", "no usable message content");
        }

        var params: llama.Params = .{};
        if (req.temperature) |t| params.temperature = t;
        if (req.top_p) |t| params.top_p = t;
        if (req.top_k) |t| params.top_k = t;
        params.n_ctx = req.n_ctx orelse self.cfg.n_ctx;
        params.n_threads = req.n_threads orelse self.cfg.n_threads;
        const max_tokens: i64 = req.max_tokens orelse self.cfg.default_max_tokens;

        const id = self.id_counter.fetchAdd(1, .monotonic);
        const created = Io.Timestamp.now(conn.io, .real).toSeconds();

        // Serialize: one inference at a time through our single backend.
        self.req_lock.lock();
        defer self.req_lock.unlock();
        self.backend.start() catch {
            return sendError(conn, "500 Internal Server Error", "server_error", "could not start inference backend");
        };
        self.backend.submit(model_path, msgs.items, params) catch {
            return sendError(conn, "500 Internal Server Error", "server_error", "could not submit request");
        };

        // One path: the chat parser splits reasoning / content / tool-call from
        // the stream (see chat_parser.zig) so nothing leaks regardless of model.
        if (req.stream) {
            try self.streamReply(conn, a, id, created, model_path, max_tokens, has_tools);
        } else {
            try self.blockReply(conn, a, id, created, model_path, max_tokens, has_tools);
        }
    }

    const Agg = struct { text: []const u8, n_tok: u64, finish: []const u8, err: ?[]const u8 };

    /// Drain the backend to completion, accumulating the generated text. Honors
    /// `max_tokens` by cancelling once the cap is hit.
    fn collect(self: *Server, conn: *Conn, a: std.mem.Allocator, max_tokens: i64) Agg {
        var text: std.ArrayList(u8) = .empty;
        var n_tok: u64 = 0;
        var finish: []const u8 = "stop";
        var err_msg: ?[]const u8 = null;
        var tmp: std.ArrayList(llama.Event) = .empty;
        defer tmp.deinit(self.gpa);
        loop: while (true) {
            self.backend.events.drain(&tmp);
            if (tmp.items.len == 0) {
                Io.sleep(conn.io, Io.Duration.fromMilliseconds(5), .awake) catch {};
                continue;
            }
            for (tmp.items) |ev| switch (ev) {
                .token => |t| {
                    text.appendSlice(a, t) catch {};
                    self.gpa.free(t);
                    n_tok += 1;
                    if (n_tok >= @as(u64, @intCast(@max(max_tokens, 1)))) {
                        finish = "length";
                        self.backend.cancel();
                    }
                },
                .done => break :loop,
                .err => |e| {
                    err_msg = a.dupe(u8, e) catch "error";
                    self.gpa.free(e);
                    break :loop;
                },
            };
            tmp.clearRetainingCapacity();
        }
        return .{ .text = text.items, .n_tok = n_tok, .finish = finish, .err = err_msg };
    }

    /// Stream a chat reply: drive the chat parser over the token stream, emitting
    /// `reasoning_content` and `content` deltas, then a `tool_calls` delta if one
    /// was produced. The single source of truth for what reaches the client.
    fn streamReply(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8, max_tokens: i64, has_tools: bool) !void {
        try conn.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
        try self.sendChunkRole(conn, a, id, created, model);

        var parser = chat_parser.Parser.init(self.gpa);
        defer parser.deinit();
        var reasoning: std.ArrayList(u8) = .empty;
        defer reasoning.deinit(self.gpa);
        var content: std.ArrayList(u8) = .empty;
        defer content.deinit(self.gpa);
        var sent_r: usize = 0;
        var sent_c: usize = 0;
        var n_tok: u64 = 0;
        var finish: []const u8 = "stop";
        var client_gone = false;
        var tmp: std.ArrayList(llama.Event) = .empty;
        defer tmp.deinit(self.gpa);
        loop: while (true) {
            self.backend.events.drain(&tmp);
            if (tmp.items.len == 0) {
                Io.sleep(conn.io, Io.Duration.fromMilliseconds(5), .awake) catch {};
                continue;
            }
            for (tmp.items) |ev| switch (ev) {
                .token => |t| {
                    parser.feed(t, &reasoning, &content);
                    self.gpa.free(t);
                    if (!client_gone) self.flushDeltas(conn, a, id, created, model, reasoning.items, content.items, &sent_r, &sent_c) catch {
                        client_gone = true;
                        self.backend.cancel();
                    };
                    n_tok += 1;
                    if (n_tok >= @as(u64, @intCast(@max(max_tokens, 1)))) {
                        finish = "length";
                        self.backend.cancel();
                    }
                },
                .done => break :loop,
                .err => break :loop,
            };
            tmp.clearRetainingCapacity();
        }
        const tc = parser.finish(a, &reasoning, &content);
        if (client_gone) return;
        self.flushDeltas(conn, a, id, created, model, reasoning.items, content.items, &sent_r, &sent_c) catch return;
        if (has_tools and tc != null) {
            try self.sendChunkToolCall(conn, a, id, created, model, tc.?);
            finish = "tool_calls";
        }
        try self.sendChunkFinish(conn, a, id, created, model, finish);
        try conn.writeAll("data: [DONE]\n\n");
        try conn.flush();
    }

    /// Send any not-yet-sent reasoning/content tail as SSE chunks, advancing the
    /// sent cursors. Reasoning is emitted before content (it comes first).
    fn flushDeltas(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8, reasoning: []const u8, content: []const u8, sent_r: *usize, sent_c: *usize) !void {
        if (reasoning.len > sent_r.*) {
            try self.sendChunkReasoning(conn, a, id, created, model, reasoning[sent_r.*..]);
            sent_r.* = reasoning.len;
        }
        if (content.len > sent_c.*) {
            try self.sendChunkDelta(conn, a, id, created, model, content[sent_c.*..]);
            sent_c.* = content.len;
        }
    }

    /// Non-streaming chat reply: aggregate via the parser, then emit one
    /// `chat.completion` with `content`, optional `reasoning_content`, and
    /// optional `tool_calls`.
    fn blockReply(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8, max_tokens: i64, has_tools: bool) !void {
        var parser = chat_parser.Parser.init(self.gpa);
        defer parser.deinit();
        var reasoning: std.ArrayList(u8) = .empty;
        defer reasoning.deinit(self.gpa);
        var content: std.ArrayList(u8) = .empty;
        defer content.deinit(self.gpa);
        var n_tok: u64 = 0;
        var finish: []const u8 = "stop";
        var err_msg: ?[]const u8 = null;
        var tmp: std.ArrayList(llama.Event) = .empty;
        defer tmp.deinit(self.gpa);
        loop: while (true) {
            self.backend.events.drain(&tmp);
            if (tmp.items.len == 0) {
                Io.sleep(conn.io, Io.Duration.fromMilliseconds(5), .awake) catch {};
                continue;
            }
            for (tmp.items) |ev| switch (ev) {
                .token => |t| {
                    parser.feed(t, &reasoning, &content);
                    self.gpa.free(t);
                    n_tok += 1;
                    if (n_tok >= @as(u64, @intCast(@max(max_tokens, 1)))) {
                        finish = "length";
                        self.backend.cancel();
                    }
                },
                .done => break :loop,
                .err => |e| {
                    err_msg = a.dupe(u8, e) catch "error";
                    self.gpa.free(e);
                    break :loop;
                },
            };
            tmp.clearRetainingCapacity();
        }
        const tc = parser.finish(a, &reasoning, &content);
        if (err_msg) |e| return sendError(conn, "500 Internal Server Error", "server_error", e);

        var body: Io.Writer.Allocating = .init(a);
        const w = &body.writer;
        try w.print("{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{ id, created });
        try writeJsonStr(w, modelId(model));
        try w.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (has_tools and tc != null and content.items.len == 0) {
            try w.writeAll("null");
        } else {
            try writeJsonStr(w, content.items);
        }
        if (reasoning.items.len > 0) {
            try w.writeAll(",\"reasoning_content\":");
            try writeJsonStr(w, reasoning.items);
        }
        if (has_tools and tc != null) {
            try w.writeAll(",\"tool_calls\":[{\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":");
            try writeJsonStr(w, tc.?.name);
            try w.writeAll(",\"arguments\":");
            try writeJsonStr(w, tc.?.args_json);
            try w.writeAll("}}]");
            finish = "tool_calls";
        }
        try w.print("}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{ finish, n_tok, n_tok });
        try sendJson(conn, "200 OK", body.written());
    }

    /// Build a system message advertising `tools` in agent.zig's `<tool_call>` format.
    fn buildToolsText(a: std.mem.Allocator, tools: []const ToolDef) ![]const u8 {
        var b: Io.Writer.Allocating = .init(a);
        const w = &b.writer;
        try w.writeAll("# Tools\n\nYou can call a tool. To call one, reply with ONLY a tool call, in this EXACT format (the `name` field is required and must be one of the tool names below):\n\n");
        try w.writeAll(agent.open_tag);
        try w.writeAll("{\"name\": \"<tool_name>\", \"arguments\": {<args>}}");
        try w.writeAll(agent.close_tag);
        try w.writeAll("\n\n## Available tools\n\n");
        for (tools) |t| {
            if (t.function.name.len == 0) continue;
            try w.writeAll("- `");
            try w.writeAll(t.function.name);
            try w.writeAll("`");
            if (t.function.description) |d| {
                try w.writeAll(" — ");
                for (d) |ch| try w.writeByte(if (ch == '\n' or ch == '\r') ' ' else ch);
            }
            if (t.function.parameters) |p| {
                const schema = std.json.Stringify.valueAlloc(a, p, .{}) catch "";
                if (schema.len > 0 and !std.mem.eql(u8, schema, "null")) {
                    try w.writeAll("\n    parameters: ");
                    try w.writeAll(schema);
                }
            }
            try w.writeByte('\n');
        }
        // A concrete example using the first real tool name nudges models that
        // otherwise drop the `name` field (e.g. gemma) into the right shape.
        if (tools.len > 0 and tools[0].function.name.len > 0) {
            try w.writeAll("\nExample — to call `");
            try w.writeAll(tools[0].function.name);
            try w.writeAll("`:\n");
            try w.writeAll(agent.open_tag);
            try w.writeAll("{\"name\": \"");
            try w.writeAll(tools[0].function.name);
            try w.writeAll("\", \"arguments\": {}}");
            try w.writeAll(agent.close_tag);
            try w.writeAll("\n");
        }
        try w.writeAll("\nDo not invent tools that are not listed. Always include the \"name\" field.\n");
        return b.written();
    }

    fn sendChunkReasoning(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8, piece: []const u8) !void {
        _ = self;
        var b: Io.Writer.Allocating = .init(a);
        const w = &b.writer;
        try chunkPrefix(w, id, created, model);
        try w.writeAll("\"delta\":{\"reasoning_content\":");
        try writeJsonStr(w, piece);
        try w.writeAll("},\"finish_reason\":null}]}");
        try sendSse(conn, b.written());
    }
    fn sendChunkToolCall(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8, tc: agent.ToolCall) !void {
        _ = self;
        var b: Io.Writer.Allocating = .init(a);
        const w = &b.writer;
        try chunkPrefix(w, id, created, model);
        try w.writeAll("\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":");
        try writeJsonStr(w, tc.name);
        try w.writeAll(",\"arguments\":");
        try writeJsonStr(w, tc.args_json);
        try w.writeAll("}}]},\"finish_reason\":null}]}");
        try sendSse(conn, b.written());
    }

    fn sendChunkRole(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8) !void {
        _ = self;
        var b: Io.Writer.Allocating = .init(a);
        const w = &b.writer;
        try chunkPrefix(w, id, created, model);
        try w.writeAll("\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}");
        try sendSse(conn, b.written());
    }
    fn sendChunkDelta(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8, piece: []const u8) !void {
        _ = self;
        var b: Io.Writer.Allocating = .init(a);
        const w = &b.writer;
        try chunkPrefix(w, id, created, model);
        try w.writeAll("\"delta\":{\"content\":");
        try writeJsonStr(w, piece);
        try w.writeAll("},\"finish_reason\":null}]}");
        try sendSse(conn, b.written());
    }
    fn sendChunkFinish(self: *Server, conn: *Conn, a: std.mem.Allocator, id: u64, created: i64, model: []const u8, finish: []const u8) !void {
        _ = self;
        var b: Io.Writer.Allocating = .init(a);
        const w = &b.writer;
        try chunkPrefix(w, id, created, model);
        try w.print("\"delta\":{{}},\"finish_reason\":\"{s}\"}}]}}", .{finish});
        try sendSse(conn, b.written());
    }
};

fn chunkPrefix(w: *Io.Writer, id: u64, created: i64, model: []const u8) !void {
    try w.print("{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{ id, created });
    try writeJsonStr(w, modelId(model));
    try w.writeAll(",\"choices\":[{\"index\":0,");
}

/// One `text_completion` SSE chunk: a `text` delta when `finish` is null, else a
/// terminating chunk carrying the finish reason.
fn compChunk(w: *Io.Writer, id: u64, created: i64, model: []const u8, text: []const u8, finish: ?[]const u8) !void {
    try w.print("{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{ id, created });
    try writeJsonStr(w, modelId(model));
    try w.writeAll(",\"choices\":[{\"index\":0,\"text\":");
    try writeJsonStr(w, text);
    if (finish) |f| {
        try w.print(",\"finish_reason\":\"{s}\"}}]}}", .{f});
    } else {
        try w.writeAll(",\"finish_reason\":null}]}");
    }
}

fn sendSse(conn: *Conn, chunk: []const u8) !void {
    try conn.writeAll("data: ");
    try conn.writeAll(chunk);
    try conn.writeAll("\n\n");
    try conn.flush();
}

fn sendJson(conn: *Conn, status: []const u8, body: []const u8) !void {
    var hdr: [256]u8 = undefined;
    const h = try std.fmt.bufPrint(&hdr, "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ status, body.len });
    try conn.writeAll(h);
    try conn.writeAll(body);
    try conn.flush();
}

fn sendError(conn: *Conn, status: []const u8, kind: []const u8, msg: []const u8) !void {
    // All call sites pass short, JSON-safe literals, so no escaping is needed.
    var buf: [1024]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\"}}}}", .{ msg, kind }) catch return;
    sendJson(conn, status, body) catch {};
}

/// Use just the file basename (without extension) as the OpenAI model id.
fn modelId(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| return base[0..dot];
    return base;
}

fn mapRole(role: []const u8) llama.Role {
    if (std.mem.eql(u8, role, "system") or std.mem.eql(u8, role, "developer")) return .system;
    if (std.mem.eql(u8, role, "assistant")) return .assistant;
    return .user; // user, tool, anything else
}

fn findContentLength(headers: []const u8) usize {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = std.mem.trim(u8, line[0..colon], " ");
        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            const val = std.mem.trim(u8, line[colon + 1 ..], " ");
            return std.fmt.parseInt(usize, val, 10) catch 0;
        }
    }
    return 0;
}

/// Write `s` as a JSON string literal (with surrounding quotes), escaping per RFC 8259.
fn writeJsonStr(w: *Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |ch| switch (ch) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x08 => try w.writeAll("\\b"),
        0x0C => try w.writeAll("\\f"),
        else => if (ch < 0x20) try w.print("\\u{x:0>4}", .{ch}) else try w.writeByte(ch),
    };
    try w.writeByte('"');
}
