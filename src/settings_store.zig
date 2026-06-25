//! Cross-platform persistence for user settings, stored as a small JSON file
//! under the OS-appropriate per-user config directory:
//!   * macOS   → ~/Library/Application Support/zig-ai/settings.json
//!   * Windows → %USERPROFILE%\AppData\Roaming\zig-ai\settings.json
//!   * Linux   → ~/.config/zig-ai/settings.json
//!
//! `getAppDataDir` was removed when std's filesystem moved under `std.Io`, so we
//! derive the directory from `home` ourselves and do all I/O through the new
//! `std.Io.Dir` API (a short-lived `Io.Threaded`, like the downloader backend).
//!
//! `load` runs once at startup; `maybeSave` is called every frame and writes
//! only when a tracked value actually changed, so it's a cheap no-op otherwise.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const st_mod = @import("state.zig");
const AppState = st_mod.AppState;

const app_dir = "zig-ai";
const file_name = "settings.json";
const max_bytes = 1 << 20; // a settings file this large is corrupt; ignore it

/// The on-disk shape. Field names are the JSON keys and the defaults mirror
/// `AppState.init`, so a missing or older file degrades to sane values.
const Persisted = struct {
    theme_pref: i64 = @intFromEnum(st_mod.ThemePref.system),
    theme_family: i64 = 0,
    threads: i64 = 4,
    use_gpu: bool = true,
    agent_mode: bool = false,
    chat_temp: f32 = 0.7,
    chat_top_p: f32 = 0.95,
    chat_top_k: i64 = 40,
    chat_n_ctx: i64 = 16384,
    chat_model: []const u8 = "", // path of the last-used chat model (re-selected on startup)
    model_dirs: []const []const u8 = &.{},
};

/// A cheap fingerprint of the persisted fields, used to skip writes when nothing
/// changed. The last value written is remembered so `maybeSave` can compare.
const Sig = struct { theme: i64, family: i64, threads: i64, gpu: bool, agent: bool, temp: f32, top_p: f32, top_k: i64, nctx: i64, model: u64, dirs: u64 };
var last_sig: ?Sig = null;

fn hashDirs(dirs: []const []const u8) u64 {
    var h = std.hash.Wyhash.init(0);
    for (dirs) |d| {
        h.update(d);
        h.update("\x00");
    }
    return h.final();
}

fn signature(st: *AppState) Sig {
    return .{
        .theme = st.theme_pref.get(),
        .family = st.theme_family.get(),
        .threads = st.threads.get(),
        .gpu = st.use_gpu.get(),
        .agent = st.agent_mode.get(),
        .temp = st.chat_temp.get(),
        .top_p = st.chat_top_p.get(),
        .top_k = st.chat_top_k.get(),
        .nctx = st.chat_n_ctx.get(),
        .model = std.hash.Wyhash.hash(0, chatModelPath(st)),
        .dirs = hashDirs(st.model_dirs.items),
    };
}

/// Path of the currently-selected chat model (empty if none).
fn chatModelPath(st: *AppState) []const u8 {
    const m = st.selectedModel(st.sel_llm.get()) orelse return "";
    return m.path;
}

/// Build the config-directory path for the current OS (caller owns it). Null if
/// there's no home directory to anchor it to.
fn configDirAlloc(gpa: std.mem.Allocator, home: []const u8) ?[]u8 {
    const parts: []const []const u8 = switch (builtin.os.tag) {
        .macos => &.{ home, "Library", "Application Support", app_dir },
        .windows => &.{ home, "AppData", "Roaming", app_dir },
        else => &.{ home, ".config", app_dir },
    };
    return std.fs.path.join(gpa, parts) catch null;
}

/// Load persisted settings into `st`. Silent on any error (missing file, parse
/// failure, …) — the app simply keeps its in-code defaults. Also primes the
/// change-tracking signature so the first `maybeSave` doesn't rewrite an
/// unchanged file.
pub fn load(st: *AppState) void {
    const gpa = st.gpa;
    const home = st.home orelse return;
    const dir = configDirAlloc(gpa, home) orelse return;
    defer gpa.free(dir);
    const path = std.fs.path.join(gpa, &.{ dir, file_name }) catch return;
    defer gpa.free(path);

    var threaded = Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // No file yet (first run) or unreadable: leave `last_sig` null so the first
    // `maybeSave` materializes the file from the current defaults.
    const data = Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(max_bytes)) catch return;
    defer gpa.free(data);

    const parsed = std.json.parseFromSlice(Persisted, gpa, data, .{
        .ignore_unknown_fields = true,
    }) catch return;
    defer parsed.deinit();
    const p = parsed.value;

    st.theme_pref.set(p.theme_pref);
    st.theme_family.set(p.theme_family);
    st.threads.set(p.threads);
    st.use_gpu.set(p.use_gpu);
    st.agent_mode.set(p.agent_mode);
    st.chat_temp.set(p.chat_temp);
    st.chat_top_p.set(p.chat_top_p);
    st.chat_top_k.set(p.chat_top_k);
    st.chat_n_ctx.set(p.chat_n_ctx);
    // Stash the last chat model's path; `resolveStartupChatModel` selects it once
    // the model list is scanned (load runs before the scan).
    if (p.chat_model.len > 0) {
        if (st.startup_chat_model) |old| gpa.free(old);
        st.startup_chat_model = gpa.dupe(u8, p.chat_model) catch null;
    }

    // Replace the added-folder list with the persisted one (owned copies).
    for (st.model_dirs.items) |d| gpa.free(d);
    st.model_dirs.clearRetainingCapacity();
    for (p.model_dirs) |d| {
        const dup = gpa.dupe(u8, d) catch continue;
        st.model_dirs.append(gpa, dup) catch gpa.free(dup);
    }

    // Successful load: treat these values as the persisted baseline.
    last_sig = signature(st);
}

/// Treat the current state as already-persisted (re-prime change tracking)
/// without writing. Call this after startup applies a transient override — e.g.
/// `--dark` — that shouldn't be written back to the user's saved preference.
pub fn markSaved(st: *AppState) void {
    last_sig = signature(st);
}

/// Write current settings to disk only if a tracked value changed since the last
/// save/load. Cheap enough to call every frame; silent on error.
pub fn maybeSave(st: *AppState) void {
    const sig = signature(st);
    if (last_sig) |l| {
        if (std.meta.eql(l, sig)) return;
    }
    save(st);
    last_sig = sig;
}

fn save(st: *AppState) void {
    const gpa = st.gpa;
    const home = st.home orelse return;
    const dir = configDirAlloc(gpa, home) orelse return;
    defer gpa.free(dir);
    const path = std.fs.path.join(gpa, &.{ dir, file_name }) catch return;
    defer gpa.free(path);

    const persisted: Persisted = .{
        .theme_pref = st.theme_pref.get(),
        .theme_family = st.theme_family.get(),
        .threads = st.threads.get(),
        .use_gpu = st.use_gpu.get(),
        .agent_mode = st.agent_mode.get(),
        .chat_temp = st.chat_temp.get(),
        .chat_top_p = st.chat_top_p.get(),
        .chat_top_k = st.chat_top_k.get(),
        .chat_n_ctx = st.chat_n_ctx.get(),
        .chat_model = chatModelPath(st),
        .model_dirs = st.model_dirs.items,
    };
    const bytes = std.json.Stringify.valueAlloc(gpa, persisted, .{ .whitespace = .indent_2 }) catch return;
    defer gpa.free(bytes);

    var threaded = Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().createDirPath(io, dir) catch {};
    Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes }) catch {};
}
