//! The app shell: a `NavigationSplitView` whose sidebar switches between the
//! eight screens and whose detail pane renders the selected one. `body` is the
//! root view function passed to `app.run`; it also pumps every backend channel
//! once per frame (no-op until backends are wired in later phases).

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;
const Screen = st_mod.Screen;
const settings_store = @import("../settings_store.zig");

const chat = @import("chat.zig");
const image = @import("image.zig");
const video = @import("video.zig");
const audio = @import("audio.zig");
const model_browser = @import("model_browser.zig");
const settings = @import("settings.zig");
const logs = @import("logs.zig");
const mcp_view = @import("mcp_view.zig");
const editor = @import("editor.zig");

// --- sidebar navigation rows -------------------------------------------------

const NavCtx = struct { st: *AppState, screen: i64 };

fn navCtx(st: *AppState, screen: Screen) *NavCtx {
    const cx = st.frame_arena.allocator().create(NavCtx) catch unreachable;
    cx.* = .{ .st = st, .screen = @intFromEnum(screen) };
    return cx;
}

fn onSelectScreen(p: ?*anyopaque) void {
    const cx: *NavCtx = @ptrCast(@alignCast(p.?));
    cx.st.screen.set(cx.screen);
}

const NavItem = struct { screen: Screen, label: []const u8, icon: zigui.IconName };

const nav_items = [_]NavItem{
    .{ .screen = .chat, .label = "Chat", .icon = .message_circle },
    .{ .screen = .image, .label = "Image", .icon = .image },
    .{ .screen = .video, .label = "Video", .icon = .film },
    .{ .screen = .audio, .label = "Audio", .icon = .audio_lines },
    .{ .screen = .models, .label = "Models", .icon = .boxes },
    .{ .screen = .mcp, .label = "MCP", .icon = .cpu },
    .{ .screen = .logs, .label = "Logs", .icon = .scroll_text },
    .{ .screen = .settings, .label = "Settings", .icon = .settings },
};

fn navRow(st: *AppState, item: NavItem, active: bool) zigui.View {
    const th = w.t();
    const fg = if (active) th.colors.on_accent else th.colors.label;
    const icon_col = if (active) th.colors.on_accent else th.colors.secondary_label;
    var row = zigui.HStack(.{
        zigui.Icon(item.icon, 17, icon_col),
        zigui.Text(item.label).font(.subheadline).foreground(fg),
        zigui.Spacer(),
    })
        .spacing(11)
        .paddingInsets(.{ .top = 8, .leading = 10, .bottom = 8, .trailing = 8 })
        .cornerRadius(8)
        .frameMaxWidth()
        .onTap(.{ .ctx = navCtx(st, item.screen), .func = onSelectScreen });
    if (active) row = row.background(th.colors.accent) else row = row.hoverFill(w.hoverTint());
    return row;
}

fn sidebar(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();
    const current = st.screen.get();

    var rows: std.ArrayList(zigui.View) = .empty;

    // App header: brand mark + name + model-loaded status dot.
    const dot = if (st.llm_loaded) w.green() else th.colors.tertiary_label;
    rows.append(fa, zigui.HStack(.{
        zigui.Icon(.boxes, 20, th.colors.accent),
        zigui.Text("zig-ai").font(.headline),
        zigui.Spacer(),
        w.statusDot(dot),
    }).spacing(8).frameMaxWidth().paddingInsets(.{ .top = 8, .leading = 10, .bottom = 10, .trailing = 10 })) catch {};

    for (nav_items) |item| {
        rows.append(fa, navRow(st, item, current == @intFromEnum(item.screen))) catch {};
    }

    rows.append(fa, zigui.Spacer()) catch {};

    // Compute indicator: memory (VRAM / unified / RAM) on top, then which ggml
    // backend is actually in use. Hidden until the device registry has loaded.
    const ai = st.acceleratorInfo();
    if (ai.ok) {
        const giB: f64 = 1024.0 * 1024.0 * 1024.0;
        const mem_kind = if (!ai.is_gpu) "RAM" else if (ai.unified) "Unified" else "VRAM";
        const mem_text = std.fmt.allocPrint(fa, "{s} {d:.1} / {d:.1} GB", .{
            mem_kind,
            @as(f64, @floatFromInt(ai.mem_used)) / giB,
            @as(f64, @floatFromInt(ai.mem_total)) / giB,
        }) catch "";
        rows.append(fa, zigui.HStack(.{
            zigui.Icon(.hard_drive, 14, th.colors.tertiary_label),
            zigui.Text(mem_text).font(.caption2).foreground(th.colors.tertiary_label),
            zigui.Spacer(),
        }).spacing(7).frameMaxWidth().paddingInsets(.{ .top = 4, .leading = 10, .bottom = 0, .trailing = 8 })) catch {};

        const accel_text = std.fmt.allocPrint(fa, "Running on {s}", .{ai.label}) catch "";
        const accel_col = if (ai.is_gpu) w.green() else th.colors.tertiary_label;
        rows.append(fa, zigui.HStack(.{
            zigui.Icon(.zap, 14, accel_col),
            zigui.Text(accel_text).font(.caption2).foreground(th.colors.tertiary_label),
            zigui.Spacer(),
        }).spacing(7).frameMaxWidth().paddingInsets(.{ .top = 2, .leading = 10, .bottom = 0, .trailing = 8 })) catch {};
    }

    // Trust footer: a quiet, persistent reminder that nothing leaves the device.
    rows.append(fa, zigui.HStack(.{
        zigui.Icon(.shield_check, 14, w.green()),
        zigui.Text("Private · runs on-device")
            .font(.caption2)
            .foreground(th.colors.tertiary_label),
        zigui.Spacer(),
    }).spacing(7).frameMaxWidth().paddingInsets(.{ .top = 4, .leading = 10, .bottom = 4, .trailing = 8 })) catch {};

    return zigui.VStack(rows.items)
        .spacing(2)
        .paddingInsets(.{ .top = 10, .leading = 8, .bottom = 10, .trailing = 8 })
        .frameMaxWidth()
        .frameMaxHeight();
}

// --- per-frame backend pump (filled in by later phases) ----------------------

fn pump(st: *AppState) void {
    st.pumpChat();
    st.pumpImage();
    st.pumpVideo();
    st.pumpAudio();
    st.pumpDownloader();
    st.pumpMcp();
}

// --- root --------------------------------------------------------------------

pub fn body(st: *AppState) zigui.View {
    // Keep the active theme in sync with the theme preference each frame so the
    // Settings picker applies live (no restart). For the `.system` choice this
    // re-queries the OS, so a live OS light/dark switch is followed too — the
    // SDL_EVENT_SYSTEM_THEME_CHANGED event wakes the loop to rebuild. The app's
    // theme provider reads `w.active` after this body builds — see main's
    // setThemeProvider.
    const pref: st_mod.ThemePref = @enumFromInt(st.theme_pref.get());
    const scheme: zigui.theme.ColorScheme = if (st_mod.effectiveDark(pref)) .dark else .light;
    const families = zigui.theme_registry.all;
    const fam_idx: usize = @intCast(std.math.clamp(st.theme_family.get(), 0, @as(i64, families.len - 1)));
    w.active = zigui.themeForScheme(families[fam_idx], scheme);
    // Persist settings when one of the tracked values changes (cheap no-op
    // otherwise; only writes on an actual edit to theme/threads/GPU/folders).
    settings_store.maybeSave(st);
    _ = st.frame_arena.reset(.retain_capacity);
    pump(st);

    const detail = switch (@as(Screen, @enumFromInt(st.screen.get()))) {
        .chat => chat.view(st),
        .image => image.view(st),
        .video => video.view(st),
        .audio => audio.view(st),
        .models => model_browser.view(st),
        .logs => logs.view(st),
        .settings => settings.view(st),
        .mcp => mcp_view.view(st),
        .editor => editor.view(st),
    };

    return zigui.NavigationSplitView(
        sidebar(st),
        detail.padding(16).frameMaxWidth().frameMaxHeight(),
        w.t().colors.secondary_background,
    )
        .alert(st.alert_present.binding(), alertCard(st))
        .alert(st.delete_present.binding(), deleteCard(st));
}

fn onConfirmDelete(st: *AppState) void {
    st.confirmDeleteModel();
}
fn onCancelDelete(st: *AppState) void {
    st.cancelDeleteModel();
}

/// Confirmation for deleting a local model: names the model + the exact path that
/// will be removed, with a destructive Delete and a Cancel.
fn deleteCard(st: *AppState) zigui.View {
    const th = w.t();
    var sbuf: [32]u8 = undefined;
    const size_str = @import("../models.zig").humanSize(&sbuf, st.deleteModelBytes());
    const what = if (st.delete_is_folder)
        w.fmt("Permanently removes this folder and its {d} file(s) ({s}):\n{s}", .{ st.deleteModelFiles(), size_str, st.deleteModelPath() })
    else
        w.fmt("Permanently removes this file ({s}):\n{s}", .{ size_str, st.deleteModelPath() });
    return zigui.VStack(.{
        zigui.Text(w.fmt("Delete \"{s}\"?", .{st.deleteModelName()})).font(.headline).frameMaxWidth(),
        zigui.WrappedText(what)
            .font(.caption)
            .foreground(th.colors.secondary_label)
            .frameMaxWidth(),
        zigui.HStack(.{
            zigui.Spacer(),
            w.secondaryButton(.close, "Cancel", zigui.actionCtx(AppState, st, onCancelDelete)),
            w.tintedButton(.trash, "Delete", th.colors.destructive, zigui.actionCtx(AppState, st, onConfirmDelete)),
        }).spacing(8).frameMaxWidth(),
    }).spacing(12).padding(16).frameMaxWidth();
}

fn onDismissAlert(st: *AppState) void {
    st.alert_present.set(false);
}

/// Content of the modal error alert (message + OK). The scrim also dismisses.
fn alertCard(st: *AppState) zigui.View {
    const th = w.t();
    return zigui.VStack(.{
        zigui.Text(st.alert_title).font(.headline).frameMaxWidth(),
        zigui.WrappedText(st.alertText())
            .font(.callout)
            .foreground(th.colors.secondary_label)
            .frameMaxWidth(),
        zigui.components.ButtonRoled("OK", .normal, zigui.actionCtx(AppState, st, onDismissAlert))
            .frameMaxWidth(),
    }).spacing(12).padding(16).frameMaxWidth();
}
