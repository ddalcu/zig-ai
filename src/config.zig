//! Cross-platform locations and defaults for the two user-editable config files
//! that drive agent mode:
//!   * `system-prompt.md` — the system prompt prepended to every chat/agent turn.
//!   * `mcp.json`         — the MCP server registry (preset + custom servers).
//!
//! Both live next to `settings.json` in the OS-appropriate per-user config dir
//! (see settings_store.zig for the layout). All I/O goes through the new
//! `std.Io.Dir` API via a short-lived `Io.Threaded`, like the other backends.
//!
//! Files are read on demand (the editor screen loads them into a text buffer)
//! and written when the user saves. On first run, `ensureDefaults` materializes
//! a starter `system-prompt.md` so the user has something to edit.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

const app_dir = "zig-ai";
pub const system_prompt_file = "system-prompt.md";
pub const mcp_file = "mcp.json";
const max_bytes = 1 << 20;

/// The starter system prompt written on first run. Tuned for small local models
/// driven through llama.cpp's plain chat template (no native tool-calling), so
/// the tool-call protocol is spelled out explicitly. The live tool list is
/// appended at generation time (see agent.zig), so this file is just the
/// persona + rules the user is free to edit.
pub const default_system_prompt =
    \\You are a helpful, capable assistant running fully on-device.
    \\Be concise and direct. Answer the user's question without unnecessary preamble.
    \\
    \\This is your persona and house style — edit it freely. When agent mode is on,
    \\the app appends the list of available tools and how to call them below.
    \\
;

/// Read the user's home directory from the environment: HOME on POSIX,
/// USERPROFILE on Windows (WTF-16 → WTF-8). Caller owns the returned slice.
/// (`Environ.getPosix` doesn't compile for Windows targets, so the branch must
/// be comptime — if/else, not an early return.)
pub fn homeDirAlloc(gpa: std.mem.Allocator, environ: std.process.Environ) ?[]u8 {
    if (builtin.os.tag == .windows) {
        const key = comptime std.unicode.wtf8ToWtf16LeStringLiteral("USERPROFILE");
        const w = std.process.Environ.getWindows(environ, key) orelse return null;
        return std.unicode.wtf16LeToWtf8Alloc(gpa, w) catch null;
    } else {
        const h = std.process.Environ.getPosix(environ, "HOME") orelse return null;
        return gpa.dupe(u8, h) catch null;
    }
}

/// Build the per-user config directory path (caller owns it). Mirrors
/// settings_store: macOS → Application Support, Windows → AppData\Roaming,
/// else → ~/.config. Null if there's no home to anchor it to.
pub fn dirAlloc(gpa: std.mem.Allocator, home: []const u8) ?[]u8 {
    const parts: []const []const u8 = switch (builtin.os.tag) {
        .macos => &.{ home, "Library", "Application Support", app_dir },
        .windows => &.{ home, "AppData", "Roaming", app_dir },
        else => &.{ home, ".config", app_dir },
    };
    return std.fs.path.join(gpa, parts) catch null;
}

/// The app's own cross-platform models directory: `<config dir>/models`. This is
/// where the downloader saves, and it's scanned alongside the legacy `.mlx-serve`
/// / `.lmstudio` folders. Caller owns the result.
pub fn modelsDirAlloc(gpa: std.mem.Allocator, home: []const u8) ?[]u8 {
    const dir = dirAlloc(gpa, home) orelse return null;
    defer gpa.free(dir);
    return std.fs.path.join(gpa, &.{ dir, "models" }) catch null;
}

/// Full path to a config file within the config dir (caller owns it).
pub fn pathAlloc(gpa: std.mem.Allocator, home: []const u8, name: []const u8) ?[]u8 {
    const dir = dirAlloc(gpa, home) orelse return null;
    defer gpa.free(dir);
    return std.fs.path.join(gpa, &.{ dir, name }) catch null;
}

/// Read a config file's bytes (caller owns). Null on any error (missing/unreadable).
pub fn read(gpa: std.mem.Allocator, home: []const u8, name: []const u8) ?[]u8 {
    const path = pathAlloc(gpa, home, name) orelse return null;
    defer gpa.free(path);
    var threaded = Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    return Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(max_bytes)) catch null;
}

/// Write a config file, creating the config dir if needed. Returns false on error.
pub fn write(gpa: std.mem.Allocator, home: []const u8, name: []const u8, data: []const u8) bool {
    const dir = dirAlloc(gpa, home) orelse return false;
    defer gpa.free(dir);
    const path = std.fs.path.join(gpa, &.{ dir, name }) catch return false;
    defer gpa.free(path);
    var threaded = Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    Io.Dir.cwd().createDirPath(io, dir) catch {};
    Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data }) catch return false;
    return true;
}

/// Load the system prompt, falling back to the built-in default if the file is
/// missing or empty (caller owns the returned slice).
pub fn loadSystemPrompt(gpa: std.mem.Allocator, home: []const u8) []u8 {
    if (read(gpa, home, system_prompt_file)) |bytes| {
        if (std.mem.trim(u8, bytes, " \t\r\n").len > 0) return bytes;
        gpa.free(bytes);
    }
    return gpa.dupe(u8, default_system_prompt) catch gpa.dupe(u8, "") catch unreachable;
}

/// On first run, materialize a starter `system-prompt.md` so the editor has
/// something to show and the file exists for power users to edit externally.
/// No-op if it already exists. mcp.json is intentionally NOT created until the
/// user enables a server (an empty registry is the right default).
pub fn ensureDefaults(gpa: std.mem.Allocator, home: []const u8) void {
    const path = pathAlloc(gpa, home, system_prompt_file) orelse return;
    defer gpa.free(path);
    var threaded = Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    // Probe for existence by attempting a read; only write the default if absent.
    if (Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(16)) catch null) |b| {
        gpa.free(b);
        return;
    }
    _ = write(gpa, home, system_prompt_file, default_system_prompt);
}
