//! zig-ai — a native desktop app (built on zigui) that runs local AI models
//! in-process: chat (llama.cpp), image (stable-diffusion.cpp), and TTS
//! (qwen3-tts.cpp). This is the entry point: it parses args, builds the
//! AppState, scans for local models, installs the tray, and runs the UI.

const std = @import("std");
const builtin = @import("builtin");
const zigui = @import("zigui");
const app = @import("zigui_app");

const st_mod = @import("state.zig");
const AppState = st_mod.AppState;
const settings_store = @import("settings_store.zig");
const shell = @import("ui/shell.zig");
const widgets = @import("ui/widgets.zig");
const api = @import("server/api.zig");
const launcher = @import("launcher.zig");

// libc stdio for the headless BMP writer (--screenshot), matching the llm-chat
// example's dev tooling.
const cstdio = @cImport({
    @cInclude("stdio.h");
});

// BSD sockets for discovering the LAN IP shown in the tray (POSIX only; Windows
// falls back to "localhost"). std.posix dropped raw sockets and std.Io.net has
// no getsockname, so we go straight to libc.
const csock = if (builtin.os.tag == .windows) struct {} else @cImport({
    @cInclude("sys/socket.h");
    @cInclude("netinet/in.h");
    @cInclude("unistd.h");
});

// --- system tray -------------------------------------------------------------

const tray_icon_px: u32 = 22;
var g_tray: ?app.Tray = null;
var g_app: ?*AppState = null;

// Retained tray entries, mutated each frame by `refreshTray` (the menu is
// OS-drawn and not part of the per-frame view tree).
var g_e_status: ?app.TrayEntry = null;
var g_e_chat: ?app.TrayEntry = null;
var g_e_image: ?app.TrayEntry = null;
var g_e_video: ?app.TrayEntry = null;
var g_e_audio: ?app.TrayEntry = null;
var g_e_ram: ?app.TrayEntry = null;
var g_e_unload: ?app.TrayEntry = null;
var g_e_hideclose: ?app.TrayEntry = null;

/// High-level tray state, used to pick the icon color and to skip re-rendering
/// the icon surface unless it actually changed.
const TrayStatus = enum { idle, loaded, busy };
var g_last_status: ?TrayStatus = null;

fn statusIconRGBA(gpa: std.mem.Allocator, size: u32, color: zigui.Color) ![]u8 {
    var canvas = zigui.Canvas.init(gpa);
    defer canvas.deinit();
    const s: f32 = @floatFromInt(size);
    try canvas.fillCircle(.{ .x = s / 2, .y = s / 2 }, s / 2 - 2, color);
    var fb = try zigui.Framebuffer.init(gpa, size, size);
    defer fb.deinit();
    fb.clear(zigui.Color.transparent);
    try zigui.raster.render(gpa, &fb, canvas.commands.items);
    return fb.toRgba8Alloc(gpa);
}

fn gotoChat(st: *AppState) void {
    st.screen.set(@intFromEnum(st_mod.Screen.chat));
    app.showWindow();
}
fn gotoImage(st: *AppState) void {
    st.screen.set(@intFromEnum(st_mod.Screen.image));
    app.showWindow();
}
fn gotoVideo(st: *AppState) void {
    st.screen.set(@intFromEnum(st_mod.Screen.video));
    app.showWindow();
}
fn gotoAudio(st: *AppState) void {
    st.screen.set(@intFromEnum(st_mod.Screen.audio));
    app.showWindow();
}

fn unloadAllCb(st: *AppState) void {
    st.unloadAll();
}

/// Flip the close-to-tray behavior. SDL also toggles the checkbox's own visual
/// on click; `refreshTray` re-syncs the checkmark to the app state each frame so
/// the two never drift.
fn toggleHideClose() void {
    app.setHideOnClose(!app.hideOnClose());
}

// --- CLI coding-agent launcher (tray) ----------------------------------------
// A tray click sets the pending CLI and opens a native folder picker; the
// result is collected in `refreshTray` (main thread) and handed to the launcher.
var g_pending_cli: ?launcher.Cli = null;

fn startCliLaunch(cli: launcher.Cli) void {
    g_pending_cli = cli;
    app.showWindow(); // give the folder dialog a parent + bring the app forward
    if (!app.openFolderDialog(null)) g_pending_cli = null; // a dialog was already open
}
fn launchOpencode() void {
    startCliLaunch(.opencode);
}
fn launchPi() void {
    startCliLaunch(.pi);
}

/// Collect a finished folder-pick and launch the pending CLI. Called once per
/// frame from `refreshTray` (the dialog's completion pushes an event that wakes
/// the loop, so this runs promptly even when otherwise idle).
fn pumpCliLauncher(a: *AppState) void {
    const cli = g_pending_cli orelse return;
    switch (app.takeFileDialogResult(a.gpa)) {
        .none => return,
        .canceled => g_pending_cli = null,
        .picked => |folder| {
            defer a.gpa.free(folder);
            g_pending_cli = null;
            var url_buf: [64]u8 = undefined;
            const base = std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}", .{api.port}) catch "http://127.0.0.1:8080";
            const sel = a.selectedModel(a.sel_llm.get());
            const model = if (sel) |m| m.name else "local";
            if (launcher.launch(a.gpa, a.environ, a.home, cli, base, model, folder)) |emsg| {
                defer a.gpa.free(emsg);
                std.debug.print("launcher: {s} failed: {s}\n", .{ cli.display(), emsg });
            }
        },
    }
}

/// Keep the event loop awake (~60fps) while any backend is generating, so the
/// per-frame channel drain runs and the transcript fills in live.
fn busyCheck() bool {
    const a = g_app orelse return false;
    // Keep the loop awake while any backend works, and while a finished video
    // clip is on-screen so its frames can play back.
    const playing_video = a.vid_result != null and
        a.screen.get() == @intFromEnum(st_mod.Screen.video);
    // Stay awake while a transient "Copied" confirmation is on screen so it
    // can revert without waiting for the next user interaction.
    const copy_feedback = a.copied_until_ms > app.c.SDL_GetTicks();
    // Stay awake while an agent tool call is in flight (or its result is queued)
    // so the agent loop advances without needing a user event.
    const agent_active = a.agent_busy or a.mcp_mgr.events.len() > 0;
    // Stay awake while the mic is recording so pumpAudio drains the capture
    // stream and the elapsed-time readout ticks.
    return a.chat.isBusy() or a.sd.isBusy() or a.video.isBusy() or a.tts.isBusy() or
        a.downloader.isBusy() or playing_video or copy_feedback or agent_active or
        a.tts_recording;
}

/// The app's live theme, queried once per frame by the run loop. `shell.body`
/// keeps `widgets.active` synced to the dark-mode preference, so returning it
/// here makes the Settings toggle apply without a restart.
fn themeProvider() zigui.Theme {
    return widgets.active;
}

/// Model resolver for the HTTP API server: returns the GUI's selected chat model
/// path (thread-safe). Called from the server thread.
fn apiResolveModel(ctx: *anyopaque, out: []u8) ?[]const u8 {
    const st: *AppState = @ptrCast(@alignCast(ctx));
    return st.apiModelPath(out);
}

/// Best-effort primary LAN IPv4 — the address the OS would use to reach the
/// internet — via a connect-less UDP socket (no packets sent). Null if it can't
/// be determined (e.g. Windows, or no network).
fn localIp(buf: []u8) ?[]const u8 {
    if (builtin.os.tag == .windows) return null;
    const fd = csock.socket(csock.AF_INET, csock.SOCK_DGRAM, 0);
    if (fd < 0) return null;
    defer _ = csock.close(fd);
    var dest = std.mem.zeroes(csock.struct_sockaddr_in);
    dest.sin_family = @intCast(csock.AF_INET);
    dest.sin_port = std.mem.nativeToBig(u16, 53);
    dest.sin_addr.s_addr = std.mem.nativeToBig(u32, 0x08080808); // 8.8.8.8
    if (csock.connect(fd, @ptrCast(&dest), @sizeOf(csock.struct_sockaddr_in)) != 0) return null;
    var local = std.mem.zeroes(csock.struct_sockaddr_in);
    var len: csock.socklen_t = @sizeOf(csock.struct_sockaddr_in);
    if (csock.getsockname(fd, @ptrCast(&local), &len) != 0) return null;
    const octets: [4]u8 = @bitCast(local.sin_addr.s_addr);
    return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{ octets[0], octets[1], octets[2], octets[3] }) catch null;
}

/// Persistent storage for the tray's API-server label (the tray references it).
var g_api_url_buf: [96]u8 = undefined;

/// The OpenAI-API URL to advertise in the tray, using the LAN IP when available.
fn apiTrayLabel() [:0]const u8 {
    var ip_buf: [16]u8 = undefined;
    const ip = localIp(&ip_buf) orelse "localhost";
    return std.fmt.bufPrintZ(&g_api_url_buf, "API: http://{s}:{d}/v1", .{ ip, api.port }) catch "API: listening";
}

fn setupTray(gpa: std.mem.Allocator, st: *AppState) void {
    const px = statusIconRGBA(gpa, tray_icon_px, widgets.t().colors.tertiary_label) catch return;
    defer gpa.free(px);
    const created = app.Tray.create(gpa, px, @intCast(tray_icon_px), @intCast(tray_icon_px), "zig-ai") catch |e| {
        std.debug.print("tray: SDL_CreateTray failed: {s} — {s}\n", .{ @errorName(e), app.c.SDL_GetError() });
        return;
    };
    g_tray = created;
    const m = g_tray.?.menu();

    // Live status block (disabled label rows, updated by refreshTray).
    g_e_status = m.addLabel("zig-ai — idle");
    m.addSeparator();
    g_e_chat = m.addLabel("Chat: —");
    g_e_image = m.addLabel("Image: —");
    g_e_video = m.addLabel("Video: —");
    g_e_audio = m.addLabel("Audio: —");
    g_e_ram = m.addLabel("RAM: —");
    m.addSeparator();
    // OpenAI-compatible HTTP server address (static — bound to 0.0.0.0).
    _ = m.addLabel(apiTrayLabel());
    m.addSeparator();

    // Actions.
    _ = m.addItem("Open zig-ai", zigui.action(app.showWindow));
    if (m.addSubmenu("Go to")) |go| {
        _ = go.addItem("Chat", zigui.actionCtx(AppState, st, gotoChat));
        _ = go.addItem("Image", zigui.actionCtx(AppState, st, gotoImage));
        _ = go.addItem("Video", zigui.actionCtx(AppState, st, gotoVideo));
        _ = go.addItem("Audio", zigui.actionCtx(AppState, st, gotoAudio));
    }
    // Launch a CLI coding agent in a chosen folder, pointed at our local server.
    if (m.addSubmenu("Launch coding agent")) |code| {
        _ = code.addItem("opencode", zigui.action(launchOpencode));
        _ = code.addItem("pi", zigui.action(launchPi));
    }
    g_e_unload = m.addItem("Unload all models", zigui.actionCtx(AppState, st, unloadAllCb));
    g_e_hideclose = m.addCheckItem("Keep running when window closes", app.hideOnClose(), zigui.action(toggleHideClose));
    m.addSeparator();
    _ = m.addItem("Quit", zigui.action(app.quit));
}

/// Whether any backend currently has a model resident in memory.
fn anyModelReady(a: *AppState) bool {
    return a.chat.status().loaded or a.sd.model_ready.load(.acquire) or
        a.video.model_ready.load(.acquire) or a.tts.model_ready.load(.acquire);
}

/// Format one per-backend status row into `buf`. Falls back to a plain label if
/// formatting overflows (model names are short in practice).
fn rowLabel(buf: []u8, label: []const u8, ready: bool, busy: bool, name: ?[]const u8) [:0]const u8 {
    const n = name orelse "model";
    const r = if (ready and busy)
        std.fmt.bufPrintZ(buf, "{s}: {s} · generating", .{ label, n })
    else if (ready)
        std.fmt.bufPrintZ(buf, "{s}: {s}", .{ label, n })
    else if (busy)
        std.fmt.bufPrintZ(buf, "{s}: loading…", .{label})
    else
        std.fmt.bufPrintZ(buf, "{s}: —", .{label});
    return r catch "—";
}

/// Per-frame tray refresh: recompute the status header, per-backend rows, RAM
/// proxy and action states, and recolor the icon when the high-level status
/// changes. Registered as the zigui frame hook, so it runs on the main thread.
fn refreshTray() void {
    const a = g_app orelse return;

    pumpCliLauncher(a);

    const llm_busy = a.chat.isBusy();
    const sd_busy = a.sd.isBusy();
    const vid_busy = a.video.isBusy();
    const tts_busy = a.tts.isBusy();
    const any_busy = llm_busy or sd_busy or vid_busy or tts_busy;
    const ready = anyModelReady(a);

    // Status header + icon color.
    const status: TrayStatus = if (any_busy) .busy else if (ready) .loaded else .idle;
    if (g_e_status) |e| {
        const text = switch (status) {
            .busy => blk: {
                const what = if (llm_busy) "chat" else if (sd_busy) "image" else if (vid_busy) "video" else "audio";
                break :blk std.fmt.bufPrintZ(&g_hdr_buf, "zig-ai — generating ({s})", .{what}) catch "zig-ai — generating";
            },
            .loaded => "zig-ai — model loaded",
            .idle => "zig-ai — idle",
        };
        e.setLabel(text);
    }
    if (g_last_status == null or g_last_status.? != status) {
        const color = switch (status) {
            .idle => widgets.t().colors.tertiary_label,
            .loaded => widgets.green(),
            .busy => widgets.t().colors.accent,
        };
        if (statusIconRGBA(a.gpa, tray_icon_px, color)) |px| {
            defer a.gpa.free(px);
            if (g_tray) |*tr| tr.setIcon(px, @intCast(tray_icon_px), @intCast(tray_icon_px));
        } else |_| {}
        g_last_status = status;
    }

    // Per-backend rows.
    var buf: [256]u8 = undefined;
    if (g_e_chat) |e| e.setLabel(rowLabel(&buf, "Chat", a.chat.status().loaded, llm_busy, a.loaded_llm));
    if (g_e_image) |e| e.setLabel(rowLabel(&buf, "Image", a.sd.model_ready.load(.acquire), sd_busy, a.loaded_sd));
    if (g_e_video) |e| e.setLabel(rowLabel(&buf, "Video", a.video.model_ready.load(.acquire), vid_busy, a.loaded_video));
    if (g_e_audio) |e| e.setLabel(rowLabel(&buf, "Audio", a.tts.model_ready.load(.acquire), tts_busy, a.loaded_tts));

    // RAM proxy.
    if (g_e_ram) |e| {
        const bytes = a.loadedBytes();
        const text = if (bytes == 0)
            std.fmt.bufPrintZ(&buf, "RAM: —", .{}) catch "RAM: —"
        else blk: {
            var hb: [32]u8 = undefined;
            break :blk std.fmt.bufPrintZ(&buf, "RAM: {s} (models)", .{models.humanSize(&hb, bytes)}) catch "RAM: —";
        };
        e.setLabel(text);
    }

    // Action states.
    if (g_e_unload) |e| e.setEnabled(ready);
    if (g_e_hideclose) |e| e.setChecked(app.hideOnClose());
}

// Dedicated buffer for the header label (kept out of the per-call stack buffer so
// the "busy" branch can return a slice into it).
var g_hdr_buf: [96]u8 = undefined;

// --- headless screenshot (dev tool) ------------------------------------------

fn renderScreenshot(gpa: std.mem.Allocator, st: *AppState, path: [:0]const u8) !void {
    // Bring up SDL's video subsystem so a `.system` theme can be resolved from
    // the OS (SDL_GetSystemTheme needs it initialized); best-effort, since a
    // headless host without a display still renders fine in light mode.
    const sdl_up = app.c.SDL_Init(app.c.SDL_INIT_VIDEO);
    defer if (sdl_up) app.c.SDL_Quit();

    const wpx: u32 = 1000;
    const hpx: u32 = 700;
    var font = zigui.Font.default();
    var emoji_font = zigui.Font.emoji();
    font.face.fallback = &emoji_font.face;
    var cache = zigui.GlyphCache.init(gpa, &font.face);
    defer cache.deinit();
    var icon_font = zigui.Font.icons();
    var icon_cache = zigui.GlyphCache.init(gpa, &icon_font.face);
    defer icon_cache.deinit();
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    zigui.beginBuild(arena);
    const root = shell.body(st);
    zigui.endBuild();

    var hits: std.ArrayList(zigui.HitRegion) = .empty;
    var overlays: std.ArrayList(zigui.OverlayReq) = .empty;
    var scrolls: std.ArrayList(zigui.ScrollRegion) = .empty;
    var ctx = zigui.Context.initFull(widgets.t(), &cache, arena, &hits, &overlays, null);
    ctx.scroll_regions = &scrolls;
    ctx.icon_cache = &icon_cache;

    var canvas = zigui.Canvas.init(arena);
    const full = zigui.Rect{ .x = 0, .y = 0, .width = @floatFromInt(wpx), .height = @floatFromInt(hpx) };
    try canvas.fillRect(full, widgets.t().colors.window_background);
    try zigui.render(&ctx, root, full, &canvas);

    var fb = try zigui.Framebuffer.init(gpa, wpx, hpx);
    defer fb.deinit();
    fb.clear(widgets.t().colors.window_background);
    try zigui.raster.render(gpa, &fb, canvas.commands.items);
    writeBmp(path, &fb);
}

fn writeBmp(path: [:0]const u8, fb: *const zigui.Framebuffer) void {
    const wpx = fb.width;
    const hpx = fb.height;
    const stride = wpx * 3 + (4 - (wpx * 3) % 4) % 4;
    const img_size = stride * hpx;
    const f = cstdio.fopen(path.ptr, "wb") orelse return;
    defer _ = cstdio.fclose(f);

    var hdr = [_]u8{0} ** 54;
    hdr[0] = 'B';
    hdr[1] = 'M';
    std.mem.writeInt(u32, hdr[2..6], @intCast(54 + img_size), .little);
    std.mem.writeInt(u32, hdr[10..14], 54, .little);
    std.mem.writeInt(u32, hdr[14..18], 40, .little);
    std.mem.writeInt(i32, hdr[18..22], @intCast(wpx), .little);
    std.mem.writeInt(i32, hdr[22..26], @intCast(hpx), .little);
    std.mem.writeInt(u16, hdr[26..28], 1, .little);
    std.mem.writeInt(u16, hdr[28..30], 24, .little);
    std.mem.writeInt(u32, hdr[34..38], @intCast(img_size), .little);
    _ = cstdio.fwrite(&hdr, 1, 54, f);

    const row = std.heap.page_allocator.alloc(u8, stride) catch return;
    defer std.heap.page_allocator.free(row);
    @memset(row, 0);
    var y: u32 = hpx;
    while (y > 0) {
        y -= 1; // BMP rows are bottom-up
        var x: u32 = 0;
        while (x < wpx) : (x += 1) {
            const px = fb.at(x, y).toRgba8();
            row[x * 3 + 0] = px.b;
            row[x * 3 + 1] = px.g;
            row[x * 3 + 2] = px.r;
        }
        _ = cstdio.fwrite(row.ptr, 1, stride, f);
    }
}

// --- headless chat smoke test ------------------------------------------------

const llama = @import("backends/llama.zig");

fn runChatSmoke(gpa: std.mem.Allocator, model_path: []const u8, prompt: []const u8) !void {
    var be = llama.Backend.init(gpa);
    defer be.deinit();
    try be.start();
    const msgs = [_]llama.ReqMessage{.{ .role = .user, .content = @constCast(prompt) }};
    try be.submit(model_path, &msgs, .{ .use_gpu = true });

    var done = false;
    while (!done) {
        var tmp: std.ArrayList(llama.Event) = .empty;
        defer tmp.deinit(gpa);
        be.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .token => |s| {
                std.debug.print("{s}", .{s});
                gpa.free(s);
            },
            .done => |d| {
                std.debug.print("\n[done: {d} tokens, {d} ms]\n", .{ d.tokens, d.ms });
                done = true;
            },
            .err => |e| {
                std.debug.print("[error: {s}]\n", .{e});
                gpa.free(e);
                done = true;
            },
        };
        if (!done and tmp.items.len == 0) {
            // Avoid a hot spin while the worker loads the model / decodes.
            app.c.SDL_Delay(20);
        }
    }
}

const sd = @import("backends/sd.zig");
const tts = @import("backends/tts.zig");
const video = @import("backends/video.zig");
const downloader = @import("backends/downloader.zig");
const models = @import("models.zig");
const mcp = @import("mcp.zig");
const config = @import("config.zig");

/// Headless MCP smoke test: ensure config, optionally add a preset, bring up the
/// runtime from `~/…/zig-ai/mcp.json`, and print the discovered servers + tools.
/// Spawns the real server subprocesses (needs `npx`), so run it deliberately.
fn runMcpSmoke(gpa: std.mem.Allocator, environ: std.process.Environ, add_preset: ?[]const u8) !void {
    const home = config.homeDirAlloc(gpa, environ) orelse {
        std.debug.print("mcp-smoke: no HOME\n", .{});
        return;
    };
    defer gpa.free(home);
    config.ensureDefaults(gpa, home);
    if (add_preset) |id| {
        if (mcp.presetById(id)) |p| {
            _ = mcp.addPreset(gpa, home, p);
            std.debug.print("mcp-smoke: added preset '{s}' to mcp.json\n", .{id});
        } else std.debug.print("mcp-smoke: no such preset '{s}'\n", .{id});
    }

    var mgr = mcp.Manager.init(gpa);
    defer mgr.deinit();
    mgr.setContext(home, environ);
    try mgr.start();

    std.debug.print("mcp-smoke: starting servers… (waiting up to ~20s)\n", .{});
    var waited: u32 = 0;
    var got_result = false;
    var called = false;
    while (waited < 20_000) : (waited += 100) {
        var tmp: std.ArrayList(mcp.Event) = .empty;
        defer tmp.deinit(gpa);
        mgr.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .log => |s| {
                std.debug.print("  {s}\n", .{s});
                gpa.free(s);
            },
            .result => |r| {
                std.debug.print("mcp-smoke: tool result (ok={}): {s}\n", .{ r.ok, r.text });
                gpa.free(r.text);
                got_result = true;
            },
            .changed => {},
        };
        // Fire one tools/call to prove the round-trip. Prefer an MCP server tool
        // (one containing "__"); otherwise exercise a built-in (run_shell).
        if (!called and mgr.toolCount() > 0) {
            called = true;
            var arena0 = std.heap.ArenaAllocator.init(gpa);
            defer arena0.deinit();
            const t0 = mgr.toolListAlloc(arena0.allocator());
            var target: []const u8 = "run_shell";
            var args: []const u8 = "{\"command\":\"echo zig-ai-mcp-ok\"}";
            for (t0) |t| if (std.mem.indexOf(u8, t.qualified, "__") != null) {
                target = t.qualified;
                args = "{\"text\":\"hello mcp\",\"a\":2,\"b\":3}";
                break;
            };
            std.debug.print("mcp-smoke: calling {s}…\n", .{target});
            mgr.callAsync(1, target, args);
        }
        if (got_result) break;
        app.c.SDL_Delay(100);
    }

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const status = mgr.statusAlloc(a);
    std.debug.print("\nmcp-smoke: {d} server(s):\n", .{status.len});
    for (status) |s| std.debug.print("  [{s}] {s} — {d} tool(s){s}{s}\n", .{
        @tagName(s.state), s.name, s.tools,
        if (s.msg != null) " — " else "",
        if (s.msg) |m| m else "",
    });
    const tools = mgr.toolListAlloc(a);
    std.debug.print("mcp-smoke: {d} tool(s) total:\n", .{tools.len});
    for (tools) |t| std.debug.print("  {s}\n", .{t.qualified});
}

/// Headless network smoke for the HF downloader: search `query`, then list the
/// first result's files (quants + auto-included support files). Validates TLS,
/// JSON parsing and both API endpoints without downloading gigabytes.
fn runDlSmoke(gpa: std.mem.Allocator, query: []const u8) !void {
    var be = downloader.Backend.init(gpa);
    defer be.deinit();

    std.debug.print("dl-smoke: searching HF for \"{s}\"…\n", .{query});
    be.search(query, null);

    var repo_id: ?[]u8 = null;
    defer if (repo_id) |r| gpa.free(r);
    var spins: usize = 0;
    poll: while (spins < 2_000_000_000) : (spins += 1) {
        var tmp: std.ArrayList(downloader.Event) = .empty;
        defer tmp.deinit(gpa);
        be.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .results => |repos| {
                std.debug.print("dl-smoke: {d} GGUF repos\n", .{repos.len});
                for (repos[0..@min(repos.len, 5)]) |r|
                    std.debug.print("  [{s}] {s}  ↓{d}\n", .{ r.kind.label(), r.id, r.downloads });
                if (repos.len > 0) repo_id = gpa.dupe(u8, repos[0].id) catch null;
                downloader.freeRepos(gpa, repos);
                break :poll;
            },
            .err => |e| {
                std.debug.print("dl-smoke ERROR: {s}\n", .{e});
                gpa.free(e);
                return;
            },
            else => {},
        };
        std.Thread.yield() catch {};
    }

    const rid = repo_id orelse {
        std.debug.print("dl-smoke: no results\n", .{});
        return;
    };
    std.debug.print("dl-smoke: listing files of {s}…\n", .{rid});
    be.listFiles(rid);
    spins = 0;
    while (spins < 2_000_000_000) : (spins += 1) {
        var tmp: std.ArrayList(downloader.Event) = .empty;
        defer tmp.deinit(gpa);
        be.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .files => |f| {
                std.debug.print("dl-smoke: {d} files\n", .{f.items.len});
                for (f.items) |it| {
                    var sbuf: [32]u8 = undefined;
                    std.debug.print("  {s:<6} {s}  {s}\n", .{
                        if (it.is_quant) "quant" else "supp.",
                        it.path,
                        models.humanSize(&sbuf, it.size),
                    });
                }
                downloader.freeFiles(gpa, f);
                return;
            },
            .err => |e| {
                std.debug.print("dl-smoke ERROR: {s}\n", .{e});
                gpa.free(e);
                return;
            },
            else => {},
        };
        std.Thread.yield() catch {};
    }
}

fn runTtsSmoke(gpa: std.mem.Allocator, model_dir: []const u8, text: []const u8, ref_wav: ?[]const u8) !void {
    var be = tts.Backend.init(gpa);
    defer be.deinit();
    try be.start();
    const ref: tts.Ref = if (ref_wav) |p| .{ .file = p } else .none;
    try be.submit(model_dir, text, ref, .{});
    var done = false;
    while (!done) {
        var tmp: std.ArrayList(tts.Event) = .empty;
        defer tmp.deinit(gpa);
        be.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .audio => |a| {
                std.debug.print("[audio: {d} samples @ {d} Hz = {d:.2}s]\n", .{ a.samples.len, a.sample_rate, @as(f32, @floatFromInt(a.samples.len)) / @as(f32, @floatFromInt(a.sample_rate)) });
                gpa.free(a.samples);
                done = true;
            },
            .err => |e| {
                std.debug.print("[error: {s}]\n", .{e});
                gpa.free(e);
                done = true;
            },
        };
        if (!done and tmp.items.len == 0) app.c.SDL_Delay(50);
    }
}

fn writePpm(path: []const u8, img: zigui.canvas.Image) void {
    const gpa = std.heap.page_allocator;
    const cpath = gpa.dupeZ(u8, path) catch return;
    defer gpa.free(cpath);
    const f = cstdio.fopen(cpath.ptr, "wb") orelse return;
    defer _ = cstdio.fclose(f);
    var hdr: [64]u8 = undefined;
    const hs = std.fmt.bufPrint(&hdr, "P6\n{d} {d}\n255\n", .{ img.width, img.height }) catch return;
    _ = cstdio.fwrite(hs.ptr, 1, hs.len, f);
    const n: usize = @as(usize, img.width) * @as(usize, img.height);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        _ = cstdio.fwrite(img.pixels.ptr + i * 4, 1, 3, f); // RGB (drop alpha)
    }
}

fn runImageSmoke(gpa: std.mem.Allocator, spec: sd.ModelSpec, prompt: []const u8, out: []const u8) !void {
    var be = sd.Backend.init(gpa);
    defer be.deinit();
    try be.start();
    try be.submit(spec, prompt, "", .{ .steps = 20, .width = 512, .height = 512 });
    var done = false;
    while (!done) {
        var tmp: std.ArrayList(sd.Event) = .empty;
        defer tmp.deinit(gpa);
        be.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .progress => |p| std.debug.print("\rstep {d}/{d}   ", .{ p.step, p.total }),
            .image => |img| {
                std.debug.print("\n[image {d}x{d}] -> {s}\n", .{ img.width, img.height, out });
                writePpm(out, img);
                gpa.free(@constCast(img.pixels));
                done = true;
            },
            .err => |e| {
                std.debug.print("[error: {s}]\n", .{e});
                gpa.free(e);
                done = true;
            },
        };
        if (!done and tmp.items.len == 0) app.c.SDL_Delay(50);
    }
}

/// Headless video generation (Wan or LTX). Writes each decoded frame to
/// `<out>-NNN.ppm` (the `--out` value with its extension stripped as prefix).
fn runVideoSmoke(
    gpa: std.mem.Allocator,
    spec: video.ModelSpec,
    prompt: []const u8,
    out: []const u8,
    params: video.Params,
) !void {
    var be = video.Backend.init(gpa);
    defer be.deinit();
    try be.start();
    try be.submit(spec, prompt, "", params);

    const prefix = out[0 .. std.mem.lastIndexOfScalar(u8, out, '.') orelse out.len];
    var done = false;
    while (!done) {
        var tmp: std.ArrayList(video.Event) = .empty;
        defer tmp.deinit(gpa);
        be.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .progress => |p| std.debug.print("\rstep {d}/{d}   ", .{ p.step, p.total }),
            .frames => |f| {
                std.debug.print("\n[{d} frames @ {d} fps]\n", .{ f.images.len, f.fps });
                for (f.images, 0..) |img, i| {
                    var buf: [512]u8 = undefined;
                    const path = std.fmt.bufPrint(&buf, "{s}-{d:0>3}.ppm", .{ prefix, i }) catch continue;
                    writePpm(path, img);
                    std.debug.print("  wrote {s} ({d}x{d})\n", .{ path, img.width, img.height });
                    gpa.free(@constCast(img.pixels));
                }
                gpa.free(f.images);
                done = true;
            },
            .err => |e| {
                std.debug.print("[error: {s}]\n", .{e});
                gpa.free(e);
                done = true;
            },
        };
        if (!done and tmp.items.len == 0) app.c.SDL_Delay(50);
    }
}

// --- entry -------------------------------------------------------------------

pub fn main(init: std.process.Init.Minimal) !void {
    // smp_allocator (fast, thread-safe) rather than page_allocator — the latter
    // does an mmap/munmap syscall per allocation, which is costly for the many
    // small allocations streaming + the immediate-mode UI make every frame.
    const gpa = std.heap.smp_allocator;

    var screenshot_path: ?[:0]const u8 = null;
    // `--dark` forces dark mode regardless of the persisted/OS preference (handy
    // for screenshots and headless runs where there's no OS signal).
    var force_dark = false;
    var start_screen: st_mod.Screen = .chat;
    var chat_smoke: ?[]const u8 = null;
    var image_smoke: ?[]const u8 = null;
    var tts_smoke: ?[]const u8 = null;
    var tts_dir: ?[]const u8 = null;
    // Optional voice-clone reference for --tts-smoke (WAV, any sample rate).
    var ref_wav: ?[]const u8 = null;
    var video_smoke: ?[]const u8 = null;
    var dl_smoke: ?[]const u8 = null;
    var mcp_smoke = false;
    var mcp_add: ?[]const u8 = null;
    var diffusion_path: ?[]const u8 = null;
    var vae_path: ?[]const u8 = null;
    var t5xxl_path: ?[]const u8 = null;
    var llm_path: ?[]const u8 = null;
    var audio_vae_path: ?[]const u8 = null;
    var connectors_path: ?[]const u8 = null;
    var vparams: video.Params = .{ .steps = 8, .width = 256, .height = 256, .frames = 5, .n_threads = 10 };
    var out_path: []const u8 = "/tmp/zigai_out.ppm";
    var model_path: ?[]const u8 = null;
    var mock = false;

    var arg_it = try std.process.Args.iterateAllocator(init.args, gpa);
    defer arg_it.deinit();
    _ = arg_it.next(); // argv[0]
    while (arg_it.next()) |a| {
        if (std.mem.eql(u8, a, "--screenshot")) {
            if (arg_it.next()) |p| screenshot_path = p;
        } else if (std.mem.eql(u8, a, "--dark")) {
            force_dark = true;
        } else if (std.mem.eql(u8, a, "--mock")) {
            mock = true;
        } else if (std.mem.eql(u8, a, "--chat-smoke")) {
            if (arg_it.next()) |p| chat_smoke = p;
        } else if (std.mem.eql(u8, a, "--image-smoke")) {
            if (arg_it.next()) |p| image_smoke = p;
        } else if (std.mem.eql(u8, a, "--tts-smoke")) {
            if (arg_it.next()) |p| tts_smoke = p;
        } else if (std.mem.eql(u8, a, "--tts-dir")) {
            if (arg_it.next()) |p| tts_dir = p;
        } else if (std.mem.eql(u8, a, "--ref-wav")) {
            if (arg_it.next()) |p| ref_wav = p;
        } else if (std.mem.eql(u8, a, "--video-smoke")) {
            if (arg_it.next()) |p| video_smoke = p;
        } else if (std.mem.eql(u8, a, "--dl-smoke")) {
            if (arg_it.next()) |p| dl_smoke = p;
        } else if (std.mem.eql(u8, a, "--mcp-smoke")) {
            mcp_smoke = true;
        } else if (std.mem.eql(u8, a, "--mcp-add")) {
            if (arg_it.next()) |p| mcp_add = p;
        } else if (std.mem.eql(u8, a, "--diffusion")) {
            if (arg_it.next()) |p| diffusion_path = p;
        } else if (std.mem.eql(u8, a, "--vae")) {
            if (arg_it.next()) |p| vae_path = p;
        } else if (std.mem.eql(u8, a, "--t5xxl")) {
            if (arg_it.next()) |p| t5xxl_path = p;
        } else if (std.mem.eql(u8, a, "--llm")) {
            if (arg_it.next()) |p| llm_path = p;
        } else if (std.mem.eql(u8, a, "--audio-vae")) {
            if (arg_it.next()) |p| audio_vae_path = p;
        } else if (std.mem.eql(u8, a, "--connectors")) {
            if (arg_it.next()) |p| connectors_path = p;
        } else if (std.mem.eql(u8, a, "--vwidth")) {
            if (arg_it.next()) |p| vparams.width = std.fmt.parseInt(i32, p, 10) catch vparams.width;
        } else if (std.mem.eql(u8, a, "--vheight")) {
            if (arg_it.next()) |p| vparams.height = std.fmt.parseInt(i32, p, 10) catch vparams.height;
        } else if (std.mem.eql(u8, a, "--vframes")) {
            if (arg_it.next()) |p| vparams.frames = std.fmt.parseInt(i32, p, 10) catch vparams.frames;
        } else if (std.mem.eql(u8, a, "--vsteps")) {
            if (arg_it.next()) |p| vparams.steps = std.fmt.parseInt(i32, p, 10) catch vparams.steps;
        } else if (std.mem.eql(u8, a, "--out")) {
            if (arg_it.next()) |p| out_path = p;
        } else if (std.mem.eql(u8, a, "--model")) {
            if (arg_it.next()) |p| model_path = p;
        } else if (std.mem.eql(u8, a, "--screen")) {
            if (arg_it.next()) |s| {
                if (std.mem.eql(u8, s, "chat")) start_screen = .chat;
                if (std.mem.eql(u8, s, "image")) start_screen = .image;
                if (std.mem.eql(u8, s, "video")) start_screen = .video;
                if (std.mem.eql(u8, s, "audio")) start_screen = .audio;
                if (std.mem.eql(u8, s, "models")) start_screen = .models;
                if (std.mem.eql(u8, s, "logs")) start_screen = .logs;
                if (std.mem.eql(u8, s, "settings")) start_screen = .settings;
                if (std.mem.eql(u8, s, "mcp")) start_screen = .mcp;
                if (std.mem.eql(u8, s, "editor")) start_screen = .editor;
            }
        }
    }

    if (dl_smoke) |query| {
        try runDlSmoke(gpa, query);
        return;
    }
    if (mcp_smoke) {
        try runMcpSmoke(gpa, init.environ, mcp_add);
        return;
    }
    if (chat_smoke) |prompt| {
        const mp = model_path orelse {
            std.debug.print("--chat-smoke requires --model <path.gguf>\n", .{});
            return;
        };
        try runChatSmoke(gpa, mp, prompt);
        return;
    }
    if (image_smoke) |prompt| {
        // Single-file checkpoint via --model, or a split FLUX model via
        // --diffusion <flux.gguf> --vae <vae> --llm <qwen.gguf> (FLUX.2) or
        // --t5xxl <t5> (FLUX.1).
        const spec: sd.ModelSpec = if (diffusion_path) |d|
            .{ .diffusion = d, .vae = vae_path, .t5xxl = t5xxl_path, .llm = llm_path }
        else if (model_path) |mp|
            .{ .model = mp }
        else {
            std.debug.print("--image-smoke requires --model <single-file> OR --diffusion <flux.gguf> --vae <vae> --llm/--t5xxl <encoder>\n", .{});
            return;
        };
        try runImageSmoke(gpa, spec, prompt, out_path);
        return;
    }
    if (tts_smoke) |text| {
        const dir = tts_dir orelse {
            std.debug.print("--tts-smoke requires --tts-dir <model folder>\n", .{});
            return;
        };
        try runTtsSmoke(gpa, dir, text, ref_wav);
        return;
    }
    if (video_smoke) |prompt| {
        const diff = diffusion_path orelse {
            std.debug.print("--video-smoke requires --diffusion <model.gguf> --vae <vae>\n", .{});
            return;
        };
        const vae = vae_path orelse {
            std.debug.print("--video-smoke requires --vae <video_vae.safetensors>\n", .{});
            return;
        };
        if (t5xxl_path == null and llm_path == null) {
            std.debug.print("--video-smoke requires a text encoder: --t5xxl <umt5.gguf> (Wan) or --llm <gemma.gguf> --audio-vae <..> --connectors <..> (LTX)\n", .{});
            return;
        }
        try runVideoSmoke(gpa, .{
            .diffusion = diff,
            .vae = vae,
            .t5xxl = t5xxl_path,
            .llm = llm_path,
            .audio_vae = audio_vae_path,
            .connectors = connectors_path,
        }, prompt, out_path, vparams);
        return;
    }

    var st = AppState.init(gpa);
    defer st.deinit();
    st.home = config.homeDirAlloc(gpa, init.environ);
    // Restore persisted settings (theme, threads, GPU, added model folders), then
    // let an explicit `--dark` override the stored/OS theme for this run.
    settings_store.load(&st);
    if (force_dark) {
        st.theme_pref.set(@intFromEnum(st_mod.ThemePref.dark));
        settings_store.markSaved(&st); // a CLI override is for this run only
    }
    st.screen.set(@intFromEnum(start_screen));
    if (start_screen == .editor) st.openEditor(.system_prompt); // seed the buffer
    if (start_screen == .mcp and mock) st.openMcpConfig(1); // demo the config form
    st.rescanModels();
    st.resolveStartupChatModel(); // re-select the chat model used last session

    // Seed the starting theme; for `.system` this queries the OS. shell.body
    // re-resolves this every frame, so it also tracks live OS theme changes.
    const theme = if (st_mod.effectiveDark(@enumFromInt(st.theme_pref.get())))
        zigui.macos.dark
    else
        zigui.default_theme;
    widgets.active = theme;

    if (mock) {
        const um = st_mod.ChatMessage.create(gpa, .user) catch unreachable;
        um.setText("Can you rewrite the chat app in Zig?") catch {};
        st.messages.append(gpa, um) catch {};
        const am = st_mod.ChatMessage.create(gpa, .assistant) catch unreachable;
        am.setText("Absolutely. We reuse zigui's layout engine, stream tokens from an in-process llama.cpp worker over a thread-safe channel, and word-wrap each bubble to the pane.") catch {};
        am.tokens = 34;
        am.tps = 48.2;
        st.messages.append(gpa, am) catch {};
        st.llm_loaded = true;

        // Select the first chat model so the header pickers show a real name.
        for (st.model_list.items.items, 0..) |m, i| {
            if (m.kind == .text) {
                st.sel_llm.set(@intCast(i));
                break;
            }
        }

        // Seed the HuggingFace downloader: a few results + an open quant popover,
        // so the Download tab renders fully in headless screenshots.
        st.models_tab.set(4);
        const ids = [_][]const u8{ "Qwen/Qwen3-TTS-GGUF", "unsloth/FLUX.2-klein-4B-GGUF", "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF" };
        const kinds = [_]models.Kind{ .tts, .image, .text };
        const dls = [_]i64{ 5322, 163017, 736566 };
        const lks = [_]i64{ 120, 1500, 980 };
        const smin = [_]u64{ 325_400_000, 2_400_000_000, 800_000_000 };
        const smax = [_]u64{ 3_600_000_000, 80_000_000_000, 1_300_000_000 };
        const mod = [_]i64{ 1_733_000_000, 1_705_276_800, 1_718_000_000 };
        for (ids, kinds, dls, lks, smin, smax, mod) |id, k, d, l, mn, mx, lm| {
            const idd = gpa.dupe(u8, id) catch continue;
            st.dl_results.append(gpa, .{
                .id = idd,
                .downloads = d,
                .likes = l,
                .kind = k,
                .size_min = mn,
                .size_max = mx,
                .size_loaded = true,
                .last_modified = lm,
            }) catch gpa.free(idd);
        }
        st.dl_filepick_idx = 0;
        const fs = gpa.alloc(st_mod.RepoFile, 3) catch unreachable;
        fs[0] = .{ .path = gpa.dupe(u8, "qwen3-tts-0.6b-f16.gguf") catch unreachable, .size = 1_700_000_000, .is_quant = true };
        fs[1] = .{ .path = gpa.dupe(u8, "qwen3-tts-1.7b-f16.gguf") catch unreachable, .size = 3_600_000_000, .is_quant = true };
        fs[2] = .{ .path = gpa.dupe(u8, "qwen3-tts-tokenizer-f16.gguf") catch unreachable, .size = 325_400_000, .is_quant = false };
        st.dl_files = .{ .repo_id = gpa.dupe(u8, ids[0]) catch unreachable, .items = fs };
    }

    if (screenshot_path) |path| {
        renderScreenshot(gpa, &st, path) catch |e|
            std.debug.print("screenshot failed: {s}\n", .{@errorName(e)});
        return;
    }

    // Load the user-editable config (system prompt) and bring up the MCP runtime.
    // Done only in the interactive path (after the smoke/screenshot early-returns)
    // so headless runs never spawn server subprocesses.
    st.loadConfig(init.environ);

    // Local OpenAI-compatible HTTP server (localhost), serving the selected chat
    // model over /v1/chat/completions. Lazily loads its own model on first request.
    var chat_api = api.Server.init(gpa, .{}, .{ .ctx = &st, .func = &apiResolveModel });
    chat_api.start() catch |e| std.debug.print("api: server failed to start: {s}\n", .{@errorName(e)});
    defer chat_api.deinit();
    // The GUI talks to that server over HTTP for chat (single chat engine).
    st.chat.port = api.port;
    st.chat.start() catch |e| std.debug.print("chat client failed to start: {s}\n", .{@errorName(e)});

    g_app = &st;
    app.setBusyCheck(&busyCheck);
    // Live light/dark switching: the loop queries this each frame; shell.body
    // keeps widgets.active synced to st.dark just before it builds the tree.
    app.setThemeProvider(&themeProvider);
    setupTray(gpa, &st);
    app.setFrameHook(&refreshTray);
    defer if (g_tray) |*tr| tr.deinit();

    try app.run(gpa, AppState, &st, .{
        .title = "zig-ai",
        .width = 1000,
        .height = 700,
        .min_width = 850,
        .min_height = 450,
        .theme = theme,
        .hide_on_close = true,
    }, shell.body);
}
