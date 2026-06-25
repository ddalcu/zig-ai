//! Chat backend that talks to the in-process OpenAI server over HTTP/SSE
//! (localhost). This is the GUI's only chat path — there's no separate in-process
//! inference for chat, so the server's chat engine (reasoning/tool-call/leak
//! handling) is the single source of truth. Mirrors `llama.Backend`'s shape (a
//! worker thread + an event channel drained once per frame) so the UI loop is
//! unchanged: it just consumes `content`/`reasoning`/`tool` deltas instead of
//! raw tokens.

const std = @import("std");
const channel = @import("../channel.zig");
const Io = std.Io;

const pt = @cImport({
    @cInclude("pthread.h");
    @cInclude("time.h");
});

fn nowMs() i64 {
    var ts: pt.struct_timespec = undefined;
    _ = pt.clock_gettime(pt.CLOCK_MONOTONIC, &ts);
    return @as(i64, @intCast(ts.tv_sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.tv_nsec)), 1_000_000);
}

pub const ToolCall = struct { name: []u8, args: []u8 };

pub const Event = union(enum) {
    content: []u8, // answer delta (UI frees)
    reasoning: []u8, // chain-of-thought delta (UI frees)
    tool: ToolCall, // a tool call the server detected (UI frees name+args)
    done: struct { ms: u64, tool: bool },
    err: []u8, // error message (UI frees)
};

const Request = struct { body: []u8 };

pub const Backend = struct {
    gpa: std.mem.Allocator,
    events: channel.Channel(Event),
    job: channel.JobState = .{},
    port: u16 = 8080,

    thread: ?std.Thread = null,
    mutex: pt.pthread_mutex_t = undefined,
    cond: pt.pthread_cond_t = undefined,
    sync_ready: bool = false,
    shutdown: bool = false,
    has_request: bool = false,
    request: ?Request = null,

    net: ?*Net = null,

    // Server model status, refreshed by a background poll of `/v1/models`.
    status_lock: channel.SpinLock = .{},
    st_loaded: bool = false,
    st_bytes: u64 = 0,
    poll_stop: std.atomic.Value(bool) = .init(false),
    poll_thread: ?std.Thread = null,

    pub const Status = struct { loaded: bool, bytes: u64 };

    const Net = struct {
        threaded: Io.Threaded,
        fn io(n: *Net) Io {
            return n.threaded.io();
        }
    };

    pub fn init(gpa: std.mem.Allocator) Backend {
        return .{ .gpa = gpa, .events = channel.Channel(Event).init(gpa) };
    }

    pub fn start(self: *Backend) !void {
        if (self.thread != null) return;
        _ = pt.pthread_mutex_init(&self.mutex, null);
        _ = pt.pthread_cond_init(&self.cond, null);
        self.sync_ready = true;
        const n = try self.gpa.create(Net);
        n.threaded = Io.Threaded.init(self.gpa, .{});
        self.net = n;
        self.thread = try std.Thread.spawn(.{}, workerMain, .{self});
        self.poll_thread = std.Thread.spawn(.{}, pollLoop, .{self}) catch null;
    }

    pub fn status(self: *Backend) Status {
        self.status_lock.lock();
        defer self.status_lock.unlock();
        return .{ .loaded = self.st_loaded, .bytes = self.st_bytes };
    }

    fn setStatus(self: *Backend, loaded: bool, bytes: u64) void {
        self.status_lock.lock();
        defer self.status_lock.unlock();
        self.st_loaded = loaded;
        self.st_bytes = bytes;
    }

    /// Poll `/v1/models` (~every 1.5s) so the UI can show the server's model state.
    fn pollLoop(self: *Backend) void {
        const io = self.net.?.io();
        while (!self.poll_stop.load(.acquire)) {
            self.pollOnce(io);
            var slept: u64 = 0;
            while (slept < 1500 and !self.poll_stop.load(.acquire)) : (slept += 250)
                Io.sleep(io, Io.Duration.fromMilliseconds(250), .awake) catch {};
        }
    }

    fn pollOnce(self: *Backend, io: Io) void {
        const ip: Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = self.port } };
        var stream = ip.connect(io, .{ .mode = .stream }) catch return self.setStatus(false, 0);
        defer stream.close(io);
        var wbuf: [1024]u8 = undefined;
        var rbuf: [16 * 1024]u8 = undefined;
        var ws = stream.writer(io, &wbuf);
        var rs = stream.reader(io, &rbuf);
        (&ws.interface).writeAll("GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n") catch return self.setStatus(false, 0);
        (&ws.interface).flush() catch {};
        const r = &rs.interface;
        var recv: std.ArrayList(u8) = .empty;
        defer recv.deinit(self.gpa);
        var tmp: [8 * 1024]u8 = undefined;
        while (recv.items.len < 256 * 1024) {
            var bufs: [1][]u8 = .{&tmp};
            const n = r.readVec(&bufs) catch 0;
            if (n == 0) break;
            recv.appendSlice(self.gpa, tmp[0..n]) catch break;
        }
        const he = std.mem.indexOf(u8, recv.items, "\r\n\r\n") orelse return self.setStatus(false, 0);
        const Data = struct { loaded: bool = false, bytes_resident: u64 = 0 };
        const Resp = struct { data: []const Data = &.{} };
        const parsed = std.json.parseFromSlice(Resp, self.gpa, recv.items[he + 4 ..], .{ .ignore_unknown_fields = true }) catch return self.setStatus(false, 0);
        defer parsed.deinit();
        if (parsed.value.data.len > 0) self.setStatus(parsed.value.data[0].loaded, parsed.value.data[0].bytes_resident) else self.setStatus(false, 0);
    }

    pub fn deinit(self: *Backend) void {
        self.poll_stop.store(true, .release);
        if (self.poll_thread) |pth| pth.join();
        if (self.thread) |th| {
            _ = pt.pthread_mutex_lock(&self.mutex);
            self.shutdown = true;
            self.job.requestCancel();
            _ = pt.pthread_cond_signal(&self.cond);
            _ = pt.pthread_mutex_unlock(&self.mutex);
            th.join();
        }
        if (self.sync_ready) {
            _ = pt.pthread_mutex_destroy(&self.mutex);
            _ = pt.pthread_cond_destroy(&self.cond);
        }
        if (self.net) |n| {
            n.threaded.deinit();
            self.gpa.destroy(n);
        }
        if (self.request) |r| self.gpa.free(r.body);
        self.events.deinit();
    }

    pub fn isBusy(self: *Backend) bool {
        return self.job.isRunning();
    }

    pub fn cancel(self: *Backend) void {
        self.job.requestCancel();
    }

    /// Submit a pre-built OpenAI `/v1/chat/completions` request body (must set
    /// `"stream": true`). The worker streams the SSE reply back as events.
    pub fn submit(self: *Backend, body: []const u8) !void {
        const b = try self.gpa.dupe(u8, body);
        errdefer self.gpa.free(b);
        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        if (self.request) |old| self.gpa.free(old.body);
        self.request = .{ .body = b };
        self.has_request = true;
        self.job.beginJob();
        _ = pt.pthread_cond_signal(&self.cond);
    }

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
            self.gpa.free(req.body);
            self.job.endJob();
        }
    }

    fn emitErr(self: *Backend, comptime f: []const u8, args: anytype) void {
        const msg = std.fmt.allocPrint(self.gpa, f, args) catch return;
        self.events.push(.{ .err = msg });
    }

    fn process(self: *Backend, req: Request) void {
        const io = self.net.?.io();
        const start_ms = nowMs();
        const ip: Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = self.port } };
        var stream = ip.connect(io, .{ .mode = .stream }) catch {
            self.emitErr("could not reach the local model server on port {d}", .{self.port});
            return;
        };
        defer stream.close(io);
        var wbuf: [16 * 1024]u8 = undefined;
        var rbuf: [16 * 1024]u8 = undefined;
        var ws = stream.writer(io, &wbuf);
        var rs = stream.reader(io, &rbuf);
        const w = &ws.interface;

        w.print("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{req.body.len}) catch {
            self.emitErr("failed to send request", .{});
            return;
        };
        w.writeAll(req.body) catch {};
        w.flush() catch {};

        const r = &rs.interface;
        var recv: std.ArrayList(u8) = .empty;
        defer recv.deinit(self.gpa);
        var headers_done = false;
        var status_ok = true;
        var finished_tool = false;
        var tmp: [8 * 1024]u8 = undefined;

        read: while (true) {
            if (self.job.cancelRequested()) break;
            var bufs: [1][]u8 = .{&tmp};
            const n = r.readVec(&bufs) catch |e| switch (e) {
                error.EndOfStream => 0,
                else => 0,
            };
            if (n == 0) break;
            recv.appendSlice(self.gpa, tmp[0..n]) catch break;

            if (!headers_done) {
                if (std.mem.indexOf(u8, recv.items, "\r\n\r\n")) |he| {
                    const head = recv.items[0..he];
                    status_ok = std.mem.indexOf(u8, head, " 200 ") != null;
                    // drop the header bytes
                    const rest = recv.items[he + 4 ..];
                    std.mem.copyForwards(u8, recv.items[0..rest.len], rest);
                    recv.items.len = rest.len;
                    headers_done = true;
                } else continue;
            }

            // Process complete SSE events (payload terminated by a blank line).
            while (std.mem.indexOf(u8, recv.items, "\n\n")) |ei| {
                const event = recv.items[0..ei];
                const consume = ei + 2;
                // handle the event line(s): we only emit `data:` payloads
                if (std.mem.startsWith(u8, std.mem.trimStart(u8, event, " \t\r\n"), "data:")) {
                    const after = std.mem.trimStart(u8, event, " \t\r\n")["data:".len..];
                    const payload = std.mem.trim(u8, after, " \t\r\n");
                    if (std.mem.eql(u8, payload, "[DONE]")) {
                        // shift + stop
                        const rest = recv.items[consume..];
                        std.mem.copyForwards(u8, recv.items[0..rest.len], rest);
                        recv.items.len = rest.len;
                        break :read;
                    }
                    if (!status_ok) {
                        self.emitErr("server error: {s}", .{payload});
                    } else {
                        if (self.handleChunk(payload)) finished_tool = true;
                    }
                }
                const rest = recv.items[consume..];
                std.mem.copyForwards(u8, recv.items[0..rest.len], rest);
                recv.items.len = rest.len;
            }
        }

        const ms: u64 = @intCast(@max(0, nowMs() - start_ms));
        self.events.push(.{ .done = .{ .ms = ms, .tool = finished_tool } });
    }

    /// Parse one SSE chunk JSON and emit its deltas. Returns true if this chunk's
    /// finish_reason is "tool_calls".
    fn handleChunk(self: *Backend, payload: []const u8) bool {
        const FnCall = struct { name: ?[]const u8 = null, arguments: ?[]const u8 = null };
        const TCall = struct { function: FnCall = .{} };
        const Delta = struct {
            content: ?[]const u8 = null,
            reasoning_content: ?[]const u8 = null,
            tool_calls: ?[]const TCall = null,
        };
        const Choice = struct { delta: Delta = .{}, finish_reason: ?[]const u8 = null };
        const Chunk = struct { choices: []const Choice = &.{} };

        const parsed = std.json.parseFromSlice(Chunk, self.gpa, payload, .{ .ignore_unknown_fields = true }) catch return false;
        defer parsed.deinit();
        if (parsed.value.choices.len == 0) return false;
        const ch = parsed.value.choices[0];
        if (ch.delta.reasoning_content) |rc| {
            if (rc.len > 0) self.events.push(.{ .reasoning = self.gpa.dupe(u8, rc) catch return false });
        }
        if (ch.delta.content) |c| {
            if (c.len > 0) self.events.push(.{ .content = self.gpa.dupe(u8, c) catch return false });
        }
        if (ch.delta.tool_calls) |tcs| {
            if (tcs.len > 0) {
                const f = tcs[0].function;
                const name = self.gpa.dupe(u8, f.name orelse "") catch return false;
                const args = self.gpa.dupe(u8, f.arguments orelse "{}") catch {
                    self.gpa.free(name);
                    return false;
                };
                self.events.push(.{ .tool = .{ .name = name, .args = args } });
            }
        }
        if (ch.finish_reason) |fr| return std.mem.eql(u8, fr, "tool_calls");
        return false;
    }
};
