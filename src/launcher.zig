//! Launch third-party CLI coding agents (opencode, pi) pointed at this app's
//! local OpenAI-compatible server. Cross-platform:
//!   * writes the CLI's provider config (opencode: a temp file referenced via
//!     `OPENCODE_CONFIG`; pi: `~/.pi/agent/models.json`),
//!   * writes a launch script that prepends common install dirs to PATH, sets
//!     env, `cd`s into the chosen folder, and runs the CLI,
//!   * opens it in a new terminal — Terminal.app (macOS `open`), a detected
//!     terminal emulator (Linux), or a new console (Windows `cmd /c start`).
//!
//! Paths, env-var syntax, script extension and the launch command all differ per
//! OS; each is handled below.

const std = @import("std");
const builtin = @import("builtin");

const Io = std.Io;
const is_windows = builtin.os.tag == .windows;
const is_macos = builtin.os.tag == .macos;

pub const Cli = enum {
    opencode,
    pi,

    pub fn binary(self: Cli) []const u8 {
        return switch (self) {
            .opencode => "opencode",
            .pi => "pi",
        };
    }
    pub fn display(self: Cli) []const u8 {
        return switch (self) {
            .opencode => "opencode",
            .pi => "pi",
        };
    }
};

/// Provider id registered in the CLI configs (must be a safe identifier).
const provider = "zigai";

/// Launch `cli` in `folder`, configured against `base_url`
/// (e.g. "http://127.0.0.1:8080") serving `model`. Returns an owned error
/// message on failure, or null on success. Runs synchronously (writes a couple
/// of small files + spawns a process); never blocks on the launched terminal.
pub fn launch(gpa: std.mem.Allocator, environ: std.process.Environ, home: ?[]const u8, cli: Cli, base_url: []const u8, model_in: []const u8, folder: []const u8) ?[]u8 {
    var threaded = Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const model = sanitizeModel(gpa, model_in) orelse return err(gpa, "out of memory");
    defer gpa.free(model);

    const tmp = tmpDir(gpa, environ) orelse return err(gpa, "no temp directory");
    defer gpa.free(tmp);

    // 1. Write the CLI provider config. opencode points at a temp file via env;
    //    pi reads a fixed path under HOME.
    var opencode_cfg: ?[]u8 = null;
    defer if (opencode_cfg) |p| gpa.free(p);
    switch (cli) {
        .opencode => {
            const path = std.fs.path.join(gpa, &.{ tmp, "zig-ai-opencode.json" }) catch return err(gpa, "oom");
            const cfg = std.fmt.allocPrint(gpa, opencode_cfg_tmpl, .{ base_url, model, model }) catch {
                gpa.free(path);
                return err(gpa, "oom");
            };
            defer gpa.free(cfg);
            writeFile(io, path, cfg, false) catch {
                gpa.free(path);
                return err(gpa, "could not write opencode config");
            };
            opencode_cfg = path;
        },
        .pi => {
            const h = home orelse return err(gpa, "no home directory");
            const dir = std.fs.path.join(gpa, &.{ h, ".pi", "agent" }) catch return err(gpa, "oom");
            defer gpa.free(dir);
            Io.Dir.cwd().createDirPath(io, dir) catch {};
            const path = std.fs.path.join(gpa, &.{ dir, "models.json" }) catch return err(gpa, "oom");
            defer gpa.free(path);
            const cfg = std.fmt.allocPrint(gpa, pi_cfg_tmpl, .{ base_url, model, model }) catch return err(gpa, "oom");
            defer gpa.free(cfg);
            writeFile(io, path, cfg, false) catch return err(gpa, "could not write pi config");
        },
    }

    // 2. Build the launch script (OS-specific env syntax + CLI invocation).
    var s: std.ArrayList(u8) = .empty;
    defer s.deinit(gpa);
    buildScript(gpa, &s, cli, model, folder, opencode_cfg) catch return err(gpa, "oom");

    const ext: []const u8 = if (is_windows) "cmd" else if (is_macos) "command" else "sh";
    const name = std.fmt.allocPrint(gpa, "zig-ai-launch-{s}.{s}", .{ cli.binary(), ext }) catch return err(gpa, "oom");
    defer gpa.free(name);
    const script_path = std.fs.path.join(gpa, &.{ tmp, name }) catch return err(gpa, "oom");
    defer gpa.free(script_path);
    // Executable bit only matters where we `open`/exec the file directly (macOS);
    // Linux runs it via `sh <script>` and Windows via `cmd`, so no +x needed.
    writeFile(io, script_path, s.items, is_macos) catch return err(gpa, "could not write launch script");

    // 3. Open it in a new terminal.
    return openTerminal(gpa, io, environ, script_path);
}

/// Assemble the shell/batch body for `cli`.
fn buildScript(gpa: std.mem.Allocator, s: *std.ArrayList(u8), cli: Cli, model: []const u8, folder: []const u8, opencode_cfg: ?[]const u8) !void {
    if (is_windows) {
        try s.appendSlice(gpa, "@echo off\r\n");
        try s.appendSlice(gpa, "set \"PATH=%USERPROFILE%\\.local\\bin;%APPDATA%\\npm;%USERPROFILE%\\.bun\\bin;%PATH%\"\r\n");
        if (opencode_cfg) |cfg| {
            try s.appendSlice(gpa, "set \"OPENCODE_CONFIG=");
            try s.appendSlice(gpa, cfg);
            try s.appendSlice(gpa, "\"\r\n");
        }
        try s.appendSlice(gpa, "cd /d \"");
        try s.appendSlice(gpa, folder);
        try s.appendSlice(gpa, "\"\r\n");
        try appendCliCmd(gpa, s, cli, model, "\r\n");
        return;
    }

    // POSIX (macOS / Linux). A login shell + common install dirs prepended to
    // PATH so npm/bun/curl-installed CLIs are found regardless of how the GUI
    // app was launched.
    try s.appendSlice(gpa, if (is_macos) "#!/bin/zsh -l\n" else "#!/bin/sh\n");
    try s.appendSlice(gpa, "export PATH=\"$HOME/.local/bin:$HOME/.opencode/bin:$HOME/.bun/bin:$HOME/.deno/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH\"\n");
    if (opencode_cfg) |cfg| {
        try s.appendSlice(gpa, "export OPENCODE_CONFIG='");
        try s.appendSlice(gpa, cfg);
        try s.appendSlice(gpa, "'\n");
    }
    try s.appendSlice(gpa, "cd '");
    try s.appendSlice(gpa, folder);
    try s.appendSlice(gpa, "' || exit 1\n");
    try s.appendSlice(gpa, "exec ");
    try appendCliCmd(gpa, s, cli, model, "\n");
}

fn appendCliCmd(gpa: std.mem.Allocator, s: *std.ArrayList(u8), cli: Cli, model: []const u8, eol: []const u8) !void {
    switch (cli) {
        .opencode => {
            try s.appendSlice(gpa, "opencode --model ");
            try s.appendSlice(gpa, provider);
            try s.append(gpa, '/');
            try s.appendSlice(gpa, model);
        },
        .pi => {
            try s.appendSlice(gpa, "pi --provider ");
            try s.appendSlice(gpa, provider);
            try s.appendSlice(gpa, " --model ");
            try s.appendSlice(gpa, model);
        },
    }
    try s.appendSlice(gpa, eol);
}

/// Open `script_path` in a new terminal window. Returns an owned error on
/// failure (e.g. no terminal emulator found on Linux).
fn openTerminal(gpa: std.mem.Allocator, io: Io, environ: std.process.Environ, script_path: []const u8) ?[]u8 {
    if (is_macos) {
        spawnDetached(io, &.{ "open", script_path }) catch return err(gpa, "could not launch Terminal");
        return null;
    }
    if (is_windows) {
        // `start "" cmd /k <script>` opens a new console and keeps it open.
        spawnDetached(io, &.{ "cmd.exe", "/c", "start", "", "cmd", "/k", script_path }) catch
            return err(gpa, "could not open a console");
        return null;
    }
    // Linux: find an installed terminal emulator and run the script in it.
    const Term = struct { bin: []const u8, flag: ?[]const u8 };
    const terminals = [_]Term{
        .{ .bin = "x-terminal-emulator", .flag = "-e" },
        .{ .bin = "gnome-terminal", .flag = "--" },
        .{ .bin = "konsole", .flag = "-e" },
        .{ .bin = "xfce4-terminal", .flag = "-e" },
        .{ .bin = "kitty", .flag = null },
        .{ .bin = "alacritty", .flag = "-e" },
        .{ .bin = "xterm", .flag = "-e" },
    };
    for (terminals) |t| {
        const full = findInPath(gpa, io, environ, t.bin) orelse continue;
        defer gpa.free(full);
        const argv: []const []const u8 = if (t.flag) |f|
            &.{ full, f, "sh", script_path }
        else
            &.{ full, "sh", script_path };
        spawnDetached(io, argv) catch continue;
        return null;
    }
    return err(gpa, "no terminal emulator found (install gnome-terminal, konsole, or xterm)");
}

/// Spawn a process and do NOT wait for it (the launcher runs on the UI thread,
/// and some terminals block until closed). stdio is detached.
fn spawnDetached(io: Io, argv: []const []const u8) !void {
    var child = try std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    });
    _ = &child; // intentionally not waited on
}

/// Resolve `name` to an absolute path by scanning `$PATH` plus a few standard
/// dirs. POSIX only (used for Linux terminal detection).
fn findInPath(gpa: std.mem.Allocator, io: Io, environ: std.process.Environ, name: []const u8) ?[]u8 {
    const std_dirs = [_][]const u8{ "/usr/bin", "/bin", "/usr/local/bin" };
    const path_env = envAlloc(gpa, environ, "PATH", "Path");
    defer if (path_env) |p| gpa.free(p);

    // $PATH dirs first, then the standard fallbacks.
    if (path_env) |pe| {
        var it = std.mem.splitScalar(u8, pe, ':');
        while (it.next()) |dir| {
            if (dir.len == 0) continue;
            if (probe(gpa, io, dir, name)) |full| return full;
        }
    }
    for (std_dirs) |dir| {
        if (probe(gpa, io, dir, name)) |full| return full;
    }
    return null;
}

fn probe(gpa: std.mem.Allocator, io: Io, dir: []const u8, name: []const u8) ?[]u8 {
    const full = std.fs.path.join(gpa, &.{ dir, name }) catch return null;
    Io.Dir.cwd().access(io, full, .{}) catch {
        gpa.free(full);
        return null;
    };
    return full;
}

fn writeFile(io: Io, path: []const u8, data: []const u8, executable: bool) !void {
    var f = try Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, data);
    if (executable and !is_windows) f.setPermissions(io, Io.File.Permissions.fromMode(0o755)) catch {};
}

/// Read an env var as owned UTF-8. POSIX uses `posix_name`; Windows reads
/// `win_name` (WTF-16 → WTF-8). The branch is comptime so each target only
/// compiles the API valid for it.
fn envAlloc(gpa: std.mem.Allocator, environ: std.process.Environ, comptime posix_name: []const u8, comptime win_name: []const u8) ?[]u8 {
    if (is_windows) {
        const key = comptime std.unicode.wtf8ToWtf16LeStringLiteral(win_name);
        const w = std.process.Environ.getWindows(environ, key) orelse return null;
        return std.unicode.wtf16LeToWtf8Alloc(gpa, w) catch null;
    } else {
        const v = std.process.Environ.getPosix(environ, posix_name) orelse return null;
        return gpa.dupe(u8, v) catch null;
    }
}

fn tmpDir(gpa: std.mem.Allocator, environ: std.process.Environ) ?[]u8 {
    if (is_windows) {
        if (envAlloc(gpa, environ, "TEMP", "TEMP")) |v| {
            if (v.len > 0) return v;
            gpa.free(v);
        }
        if (envAlloc(gpa, environ, "TMP", "TMP")) |v| {
            if (v.len > 0) return v;
            gpa.free(v);
        }
    } else if (envAlloc(gpa, environ, "TMPDIR", "TMPDIR")) |v| {
        if (v.len > 0) return v;
        gpa.free(v);
    }
    return gpa.dupe(u8, if (is_windows) "." else "/tmp") catch null;
}

/// Keep only chars valid in a CLI/config model id; others become '-'.
fn sanitizeModel(gpa: std.mem.Allocator, name: []const u8) ?[]u8 {
    if (name.len == 0) return gpa.dupe(u8, "local") catch null;
    const out = gpa.alloc(u8, name.len) catch return null;
    for (name, 0..) |ch, i| {
        out[i] = switch (ch) {
            'A'...'Z', 'a'...'z', '0'...'9', '.', '_', '-' => ch,
            else => '-',
        };
    }
    return out;
}

fn err(gpa: std.mem.Allocator, msg: []const u8) []u8 {
    return gpa.dupe(u8, msg) catch @constCast("launch failed");
}

const opencode_cfg_tmpl =
    \\{{
    \\  "$schema": "https://opencode.ai/config.json",
    \\  "provider": {{
    \\    "zigai": {{
    \\      "npm": "@ai-sdk/openai-compatible",
    \\      "name": "zig-ai (local)",
    \\      "options": {{ "baseURL": "{s}/v1" }},
    \\      "models": {{ "{s}": {{ "name": "{s} (zig-ai)" }} }}
    \\    }}
    \\  }}
    \\}}
;

const pi_cfg_tmpl =
    \\{{
    \\  "providers": {{
    \\    "zigai": {{
    \\      "baseUrl": "{s}/v1",
    \\      "api": "openai-completions",
    \\      "apiKey": "zig-ai",
    \\      "compat": {{ "supportsDeveloperRole": false, "supportsReasoningEffort": false, "maxTokensField": "max_tokens", "thinkingFormat": "qwen" }},
    \\      "models": [ {{"id": "{s}", "name": "zig-ai {s}", "input": ["text"], "contextWindow": 32768, "maxTokens": 8192, "reasoning": true}} ]
    \\    }}
    \\  }}
    \\}}
;

test "buildScript: opencode sets env, cd, and command (posix)" {
    if (is_windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var s: std.ArrayList(u8) = .empty;
    defer s.deinit(a);
    try buildScript(a, &s, .opencode, "qwen3", "/work dir", "/tmp/cfg.json");
    try std.testing.expect(std.mem.indexOf(u8, s.items, "OPENCODE_CONFIG='/tmp/cfg.json'") != null);
    try std.testing.expect(std.mem.indexOf(u8, s.items, "cd '/work dir'") != null);
    try std.testing.expect(std.mem.indexOf(u8, s.items, "opencode --model zigai/qwen3") != null);
}

test "buildScript: pi has no OPENCODE_CONFIG (posix)" {
    if (is_windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var s: std.ArrayList(u8) = .empty;
    defer s.deinit(a);
    try buildScript(a, &s, .pi, "m", "/w", null);
    try std.testing.expect(std.mem.indexOf(u8, s.items, "pi --provider zigai --model m") != null);
    try std.testing.expect(std.mem.indexOf(u8, s.items, "OPENCODE_CONFIG") == null);
}

test "sanitizeModel strips unsafe chars" {
    const a = std.testing.allocator;
    const m = sanitizeModel(a, "Llama 3.2/Q4:x").?;
    defer a.free(m);
    try std.testing.expectEqualStrings("Llama-3.2-Q4-x", m);
}

test "writeFile writes the data (runtime Io)" {
    if (is_windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var threaded = Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const path = "/tmp/zig-ai-launcher-selftest.sh";
    try writeFile(io, path, "#!/bin/sh\necho hi\n", true);
    const data = try Io.Dir.cwd().readFileAlloc(io, path, a, .limited(128));
    defer a.free(data);
    try std.testing.expect(std.mem.indexOf(u8, data, "echo hi") != null);
    Io.Dir.cwd().deleteFile(io, path) catch {};
}

test "config templates embed the base URL" {
    const a = std.testing.allocator;
    const oc = try std.fmt.allocPrint(a, opencode_cfg_tmpl, .{ "http://127.0.0.1:8080", "m", "m" });
    defer a.free(oc);
    try std.testing.expect(std.mem.indexOf(u8, oc, "\"baseURL\": \"http://127.0.0.1:8080/v1\"") != null);
    const pc = try std.fmt.allocPrint(a, pi_cfg_tmpl, .{ "http://127.0.0.1:8080", "m", "m" });
    defer a.free(pc);
    try std.testing.expect(std.mem.indexOf(u8, pc, "\"baseUrl\": \"http://127.0.0.1:8080/v1\"") != null);
}
