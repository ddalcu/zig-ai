//! Windows: capture the process's stdout/stderr into the in-app Logs view.
//!
//! The app is a GUI-subsystem binary (no console window), so anything written to
//! the standard streams — our own `std.debug.print`, zigui's `std.log`, and the
//! C backends' (ggml/llama) C stdio — would otherwise be discarded. At startup
//! we point both standard handles at an anonymous pipe; a reader thread feeds
//! complete lines into a small mutex-protected sink, which the UI drains into the
//! shared `LogRing` once per frame (so the ring stays single-threaded).
//!
//! No-op on non-Windows, where output goes to the terminal as usual.

const std = @import("std");
const builtin = @import("builtin");
const LogRing = @import("state.zig").LogRing;

const is_windows = builtin.os.tag == .windows;
const windows = std.os.windows;

// ggml's own logger — the source of the "ggml_cuda_init" / "load_backend" lines.
// Those come from the (MSVC-built) backend DLLs' C runtime, which may not honor a
// redirected std handle, so we route them through ggml's callback instead.
const ggml = @cImport({
    @cInclude("ggml.h");
});

const STD_OUTPUT_HANDLE: windows.DWORD = @bitCast(@as(i32, -11));
const STD_ERROR_HANDLE: windows.DWORD = @bitCast(@as(i32, -12));
const ATTACH_PARENT_PROCESS: windows.DWORD = 0xFFFFFFFF;
const GENERIC_WRITE: windows.DWORD = 0x40000000;
const FILE_SHARE_WRITE: windows.DWORD = 0x00000002;
const OPEN_EXISTING: windows.DWORD = 3;

extern "kernel32" fn CreatePipe(hReadPipe: *windows.HANDLE, hWritePipe: *windows.HANDLE, lpPipeAttributes: ?*anyopaque, nSize: windows.DWORD) callconv(.winapi) c_int;
extern "kernel32" fn SetStdHandle(nStdHandle: windows.DWORD, hHandle: windows.HANDLE) callconv(.winapi) c_int;
extern "kernel32" fn ReadFile(hFile: windows.HANDLE, lpBuffer: [*]u8, nNumberOfBytesToRead: windows.DWORD, lpNumberOfBytesRead: *windows.DWORD, lpOverlapped: ?*anyopaque) callconv(.winapi) c_int;
extern "kernel32" fn AttachConsole(dwProcessId: windows.DWORD) callconv(.winapi) c_int;
extern "kernel32" fn CreateFileW(lpFileName: [*:0]const u16, dwDesiredAccess: windows.DWORD, dwShareMode: windows.DWORD, lpSecurityAttributes: ?*anyopaque, dwCreationDisposition: windows.DWORD, dwFlagsAndAttributes: windows.DWORD, hTemplateFile: ?windows.HANDLE) callconv(.winapi) windows.HANDLE;

// std.Thread.Mutex was removed in 0.16 (it needs an Io handle now). The sink's
// critical sections are tiny (append/move a pointer list) and contention is just
// one reader thread vs the UI thread once per frame, so a test-and-set spinlock
// is more than enough and dependency-free.
var sink_lock: std.atomic.Value(bool) = .init(false);
var sink: std.ArrayList([]u8) = .empty;
var sink_gpa: std.mem.Allocator = undefined;
var started = false;

fn lock() void {
    while (sink_lock.swap(true, .acquire)) std.atomic.spinLoopHint();
}
fn unlock() void {
    sink_lock.store(false, .release);
}

fn pushLine(text: []const u8) void {
    var line = text;
    if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
    const dup = sink_gpa.dupe(u8, line) catch return;
    lock();
    defer unlock();
    sink.append(sink_gpa, dup) catch sink_gpa.free(dup);
}

fn readerMain(rd: windows.HANDLE) void {
    var buf: [4096]u8 = undefined;
    var acc: std.ArrayList(u8) = .empty;
    defer acc.deinit(sink_gpa);
    while (true) {
        var n: windows.DWORD = 0;
        if (ReadFile(rd, &buf, buf.len, &n, null) == 0 or n == 0) break;
        var rest: []const u8 = buf[0..n];
        while (std.mem.indexOfScalar(u8, rest, '\n')) |nl| {
            acc.appendSlice(sink_gpa, rest[0..nl]) catch {};
            pushLine(acc.items);
            acc.clearRetainingCapacity();
            rest = rest[nl + 1 ..];
        }
        acc.appendSlice(sink_gpa, rest) catch {};
        if (acc.items.len > 8192) { // very long unterminated line — flush it
            pushLine(acc.items);
            acc.clearRetainingCapacity();
        }
    }
}

/// Redirect stdout/stderr to a pipe and start the reader thread. Call once,
/// before anything prints. No-op on non-Windows or on failure (output is simply
/// left uncaptured rather than breaking the app).
pub fn start(gpa: std.mem.Allocator) void {
    if (!is_windows or started) return;
    sink_gpa = gpa;
    var rd: windows.HANDLE = undefined;
    var wr: windows.HANDLE = undefined;
    if (CreatePipe(&rd, &wr, null, 0) == 0) return;
    _ = SetStdHandle(STD_OUTPUT_HANDLE, wr);
    _ = SetStdHandle(STD_ERROR_HANDLE, wr);
    const t = std.Thread.spawn(.{}, readerMain, .{rd}) catch return;
    t.detach();
    started = true;
}

fn ggmlLog(level: ggml.enum_ggml_log_level, text: [*c]const u8, ud: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = ud;
    if (!started or text == null) return;
    var s: []const u8 = std.mem.span(text);
    while (s.len > 0 and (s[s.len - 1] == '\n' or s[s.len - 1] == '\r')) s = s[0 .. s.len - 1];
    if (s.len > 0) pushLine(s);
}

/// Route ggml's logging into the same sink. Call after `start()`. ggml_log_set
/// replaces the default stderr handler, so these lines go only to the Logs view
/// (no console duplication). Captures the cuda_init/load_backend/etc. output the
/// std-handle redirect can't reliably reach across the DLLs' C runtime.
pub fn captureGgmlLogs() void {
    if (!is_windows or !started) return;
    ggml.ggml_log_set(ggmlLog, null);
}

/// Move captured lines into the LogRing. Call on the UI thread (per frame).
pub fn drain(ring: *LogRing) void {
    if (!is_windows or !started) return;
    lock();
    defer unlock();
    for (sink.items) |l| {
        ring.append(l);
        sink_gpa.free(l);
    }
    sink.clearRetainingCapacity();
}

/// For CLI/headless runs: attach to the launching terminal's console (if any) so
/// stdout/stderr are visible there, since the GUI subsystem provides no console.
/// No-op / harmless when there is no parent console.
pub fn attachParentConsole() void {
    if (!is_windows) return;
    if (AttachConsole(ATTACH_PARENT_PROCESS) == 0) return;
    // Point the std handles at the attached console.
    const conout = std.unicode.utf8ToUtf16LeStringLiteral("CONOUT$");
    const h = CreateFileW(conout, GENERIC_WRITE, FILE_SHARE_WRITE, null, OPEN_EXISTING, 0, null);
    if (h != windows.INVALID_HANDLE_VALUE) {
        _ = SetStdHandle(STD_OUTPUT_HANDLE, h);
        _ = SetStdHandle(STD_ERROR_HANDLE, h);
    }
}
