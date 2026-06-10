//! Built-in agent tools — a small set of local, cross-platform tools the agent
//! can use even with no MCP server configured (mirrors the built-ins in the
//! mlx-serve agent: read/write/list/search files + run a shell command).
//!
//! These are advertised alongside MCP tools and dispatched through the same
//! async path: the MCP `Manager` worker recognizes a built-in name and calls
//! `execute` here (on its own thread, with its own `Io`), so file/shell work
//! never blocks the UI. All file I/O goes through `std.Io`; the shell tool uses
//! the portable `std.process.spawn` (sh on POSIX, cmd on Windows).

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// A built-in tool definition advertised to the model.
pub const Spec = struct {
    name: []const u8,
    description: []const u8,
    schema: []const u8, // JSON-schema text (matches the MCP inputSchema shape)
};

pub const specs = [_]Spec{
    .{
        .name = "read_file",
        .description = "Read a UTF-8 text file and return its contents.",
        .schema =
        \\{"type":"object","properties":{"path":{"type":"string","description":"File path (absolute or relative)"}},"required":["path"]}
        ,
    },
    .{
        .name = "write_file",
        .description = "Create or overwrite a text file with the given content.",
        .schema =
        \\{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}
        ,
    },
    .{
        .name = "list_files",
        .description = "List the entries of a directory (default: current directory).",
        .schema =
        \\{"type":"object","properties":{"path":{"type":"string","description":"Directory path"}}}
        ,
    },
    .{
        .name = "search_files",
        .description = "Search a file or the files in a directory for a substring; returns matching lines as path:line: text.",
        .schema =
        \\{"type":"object","properties":{"path":{"type":"string"},"query":{"type":"string"}},"required":["path","query"]}
        ,
    },
    .{
        .name = "run_shell",
        .description = "Run a shell command and return its combined stdout/stderr (output is capped).",
        .schema =
        \\{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
        ,
    },
};

pub fn isBuiltin(name: []const u8) bool {
    for (specs) |s| if (std.mem.eql(u8, s.name, name)) return true;
    return false;
}

const max_read: usize = 64 * 1024;
const max_out: usize = 16 * 1024;

/// Execute a built-in tool. Returns an owned result string (the caller frees).
/// Internal failures are returned as `Error: …` text rather than as errors, so
/// the agent can read and recover from them.
pub fn execute(gpa: std.mem.Allocator, io: Io, name: []const u8, args_json: []const u8) []u8 {
    return run(gpa, io, name, args_json) catch |e|
        std.fmt.allocPrint(gpa, "Error: {s}", .{@errorName(e)}) catch gpa.dupe(u8, "Error") catch &.{};
}

fn run(gpa: std.mem.Allocator, io: Io, name: []const u8, args_json: []const u8) ![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    const args: std.json.Value = std.json.parseFromSliceLeaky(std.json.Value, a, args_json, .{}) catch .{ .object = std.json.ObjectMap.init(a, &.{}, &.{}) catch unreachable };

    if (std.mem.eql(u8, name, "read_file")) {
        const path = strArg(args, "path") orelse return gpa.dupe(u8, "Error: missing \"path\"");
        const data = Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(max_read)) catch |e|
            return std.fmt.allocPrint(gpa, "Error reading {s}: {s}", .{ path, @errorName(e) });
        return data;
    }
    if (std.mem.eql(u8, name, "write_file")) {
        const path = strArg(args, "path") orelse return gpa.dupe(u8, "Error: missing \"path\"");
        const content = strArg(args, "content") orelse "";
        Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = content }) catch |e|
            return std.fmt.allocPrint(gpa, "Error writing {s}: {s}", .{ path, @errorName(e) });
        return std.fmt.allocPrint(gpa, "Wrote {d} bytes to {s}", .{ content.len, path });
    }
    if (std.mem.eql(u8, name, "list_files")) {
        const path = strArg(args, "path") orelse ".";
        return listFiles(gpa, io, path);
    }
    if (std.mem.eql(u8, name, "search_files")) {
        const path = strArg(args, "path") orelse return gpa.dupe(u8, "Error: missing \"path\"");
        const query = strArg(args, "query") orelse return gpa.dupe(u8, "Error: missing \"query\"");
        return searchFiles(gpa, io, path, query);
    }
    if (std.mem.eql(u8, name, "run_shell")) {
        const cmd = strArg(args, "command") orelse return gpa.dupe(u8, "Error: missing \"command\"");
        return runShell(gpa, io, cmd);
    }
    return std.fmt.allocPrint(gpa, "Error: unknown built-in tool {s}", .{name});
}

fn strArg(v: std.json.Value, key: []const u8) ?[]const u8 {
    if (v != .object) return null;
    const got = v.object.get(key) orelse return null;
    return switch (got) {
        .string => |s| s,
        else => null,
    };
}

fn listFiles(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]u8 {
    var dir = Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |e|
        return std.fmt.allocPrint(gpa, "Error opening {s}: {s}", .{ path, @errorName(e) });
    defer dir.close(io);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var it = dir.iterate();
    var n: usize = 0;
    while (it.next(io) catch null) |entry| {
        const suffix = if (entry.kind == .directory) "/" else "";
        try out.print(gpa, "{s}{s}\n", .{ entry.name, suffix });
        n += 1;
        if (out.items.len > max_out) {
            try out.appendSlice(gpa, "… (truncated)\n");
            break;
        }
    }
    if (n == 0) try out.appendSlice(gpa, "(empty directory)\n");
    return out.toOwnedSlice(gpa);
}

fn searchFiles(gpa: std.mem.Allocator, io: Io, path: []const u8, query: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);

    // A directory: scan its top-level files. Otherwise treat `path` as a file.
    if (Io.Dir.cwd().openDir(io, path, .{ .iterate = true })) |*d| {
        var dir = d.*;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .file) continue;
            const full = try std.fs.path.join(gpa, &.{ path, entry.name });
            defer gpa.free(full);
            try searchOne(gpa, io, &out, full, query);
            if (out.items.len > max_out) break;
        }
    } else |_| {
        try searchOne(gpa, io, &out, path, query);
    }
    if (out.items.len == 0) try out.print(gpa, "No matches for \"{s}\".", .{query});
    return out.toOwnedSlice(gpa);
}

fn searchOne(gpa: std.mem.Allocator, io: Io, out: *std.ArrayList(u8), file_path: []const u8, query: []const u8) !void {
    const data = Io.Dir.cwd().readFileAlloc(io, file_path, gpa, .limited(max_read)) catch return;
    defer gpa.free(data);
    var line_no: usize = 1;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| : (line_no += 1) {
        if (std.mem.indexOf(u8, line, query) != null) {
            try out.print(gpa, "{s}:{d}: {s}\n", .{ file_path, line_no, std.mem.trim(u8, line, " \t\r") });
            if (out.items.len > max_out) return;
        }
    }
}

fn runShell(gpa: std.mem.Allocator, io: Io, command: []const u8) ![]u8 {
    const argv: []const []const u8 = switch (builtin.os.tag) {
        .windows => &.{ "cmd.exe", "/c", command },
        else => &.{ "/bin/sh", "-c", command },
    };
    var child = std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .pipe,
    }) catch |e| return std.fmt.allocPrint(gpa, "Error launching shell: {s}", .{@errorName(e)});

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    // Read stdout then stderr (bounded). Reading before wait avoids a pipe-full
    // deadlock for normal-sized output.
    if (child.stdout) |f| try drain(gpa, io, &out, f);
    if (child.stderr) |f| try drain(gpa, io, &out, f);
    _ = child.wait(io) catch {};

    if (out.items.len == 0) return gpa.dupe(u8, "(no output)");
    return out.toOwnedSlice(gpa);
}

fn drain(gpa: std.mem.Allocator, io: Io, out: *std.ArrayList(u8), f: Io.File) !void {
    var tmp: [4096]u8 = undefined;
    while (out.items.len <= max_out) {
        const n = f.readStreaming(io, &.{tmp[0..]}) catch break;
        if (n == 0) break;
        try out.appendSlice(gpa, tmp[0..n]);
    }
    if (out.items.len > max_out) try out.appendSlice(gpa, "\n… (truncated)");
}

test "isBuiltin recognizes the tool set" {
    try std.testing.expect(isBuiltin("read_file"));
    try std.testing.expect(isBuiltin("run_shell"));
    try std.testing.expect(!isBuiltin("definitely_not_a_tool"));
}
