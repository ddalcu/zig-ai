//! Shared UI helpers and the app-wide active theme. The theme is chosen once at
//! startup (light/dark) and read here by every screen, mirroring how the zigui
//! examples reference `zigui.default_theme` — except we make it switchable.

const std = @import("std");
const zigui = @import("zigui");
const st_mod = @import("../state.zig");
const models = @import("../models.zig");
const AppState = st_mod.AppState;

/// The active theme. Set by `main` before `app.run`; widgets read it via `t()`.
pub var active: zigui.Theme = zigui.default_theme;

pub fn t() zigui.Theme {
    return active;
}

pub fn green() zigui.Color {
    return zigui.Color.fromRgb8(52, 199, 89);
}

/// A subtle, theme-aware highlight for the row/item the cursor is hovering.
/// Pass to `.hoverFill(...)` on tappable rows (menu items, list rows).
pub fn hoverTint() zigui.Color {
    return t().colors.accent.withAlpha(0.10);
}
pub fn orange() zigui.Color {
    return zigui.Color.fromRgb8(255, 149, 0);
}

/// Format into the per-frame build arena (valid until the next rebuild). Mirrors
/// `zigui.components.fmt` but lets callers keep using it directly.
pub const fmt = zigui.components.fmt;

/// A grouped "card" container: padded, filled, rounded, hairline-bordered.
pub fn card(content: zigui.View) zigui.View {
    const th = t();
    return content
        .padding(14)
        .background(th.colors.control_background)
        .cornerRadius(th.metrics.corner_radius)
        .border(th.colors.separator, th.metrics.hairline)
        .frameMaxWidth();
}

/// A small, left-aligned section header. (A bare `.frameMaxWidth()` Text centers;
/// the trailing Spacer pins it to the leading edge.)
pub fn sectionHeader(title: []const u8) zigui.View {
    return zigui.HStack(.{
        zigui.Text(title).font(.subheadline).foreground(t().colors.secondary_label),
        zigui.Spacer(),
    }).frameMaxWidth();
}

/// A label + trailing control row (left text, spacer, right control).
pub fn settingRow(title: []const u8, control: zigui.View) zigui.View {
    return zigui.HStack(.{
        zigui.Text(title),
        zigui.Spacer(),
        control,
    }).frameMaxWidth();
}

/// A filled status dot of the given color.
pub fn statusDot(color: zigui.Color) zigui.View {
    return zigui.Circle(color).frame(10, 10);
}

/// A centered empty-state: a large muted glyph over a title and subtitle,
/// filling the available space. The icon gives each blank screen a clear,
/// modern identity instead of a bare line of text.
pub fn emptyState(icon: zigui.IconName, title: []const u8, subtitle: []const u8) zigui.View {
    const th = t();
    return zigui.VStack(.{
        zigui.Spacer(),
        zigui.Icon(icon, 40, th.colors.tertiary_label).frameMaxWidth(),
        zigui.Text(title).font(.title3).foreground(th.colors.secondary_label).frameMaxWidth(),
        zigui.Text(subtitle).font(.callout).foreground(th.colors.tertiary_label).frameMaxWidth(),
        zigui.Spacer(),
    }).spacing(8).frameMaxWidth().frameMaxHeight();
}

/// A prominent primary action: a filled accent pill with a leading icon. Use
/// for the main call-to-action on a screen (Generate, Search, Download).
pub fn primaryButton(icon: zigui.IconName, label: []const u8, cb: zigui.Callback) zigui.View {
    const th = t();
    return zigui.HStack(.{
        zigui.Icon(icon, 15, th.colors.on_accent),
        zigui.Text(label).font(.callout).foreground(th.colors.on_accent),
    }).spacing(7)
        .paddingInsets(.{ .top = 8, .leading = 14, .bottom = 8, .trailing = 16 })
        .background(th.colors.accent)
        .cornerRadius(8)
        .onTap(cb);
}

/// A full-width primary action with centered icon + label. For the main button
/// at the bottom of a form column (Generate, Synthesize).
pub fn primaryButtonWide(icon: zigui.IconName, label: []const u8, cb: zigui.Callback) zigui.View {
    const th = t();
    return zigui.HStack(.{
        zigui.Spacer(),
        zigui.Icon(icon, 15, th.colors.on_accent),
        zigui.Text(label).font(.callout).foreground(th.colors.on_accent),
        zigui.Spacer(),
    }).spacing(7)
        .paddingInsets(.{ .top = 9, .leading = 14, .bottom = 9, .trailing = 14 })
        .background(th.colors.accent)
        .cornerRadius(8)
        .frameMaxWidth()
        .onTap(cb);
}

/// A secondary action: a bordered, control-filled pill with a leading icon.
/// `tint` colors both the icon and label (defaults to the accent color).
pub fn secondaryButton(icon: zigui.IconName, label: []const u8, cb: zigui.Callback) zigui.View {
    return tintedButton(icon, label, t().colors.accent, cb);
}

pub fn tintedButton(icon: zigui.IconName, label: []const u8, tint: zigui.Color, cb: zigui.Callback) zigui.View {
    const th = t();
    return zigui.HStack(.{
        zigui.Icon(icon, 14, tint),
        zigui.Text(label).font(.callout).foreground(tint),
    }).spacing(6)
        .paddingInsets(.{ .top = 7, .leading = 11, .bottom = 7, .trailing = 13 })
        .background(th.colors.control_background)
        .cornerRadius(8)
        .border(th.colors.separator, th.metrics.hairline)
        .onTap(cb);
}

/// A screen's top bar: large title on the left, optional trailing view.
pub fn header(title: []const u8, trailing: zigui.View) zigui.View {
    return zigui.HStack(.{
        zigui.Text(title).font(.title),
        zigui.Spacer(),
        trailing,
    }).frameMaxWidth();
}

// --- header model-switcher ---------------------------------------------------

const PickCtx = struct { st: *AppState, index: usize };

fn pickCtx(st: *AppState, index: usize) *PickCtx {
    const cx = st.frame_arena.allocator().create(PickCtx) catch unreachable;
    cx.* = .{ .st = st, .index = index };
    return cx;
}

fn onPickModel(p: ?*anyopaque) void {
    const cx: *PickCtx = @ptrCast(@alignCast(p.?));
    cx.st.setSelected(cx.index);
    cx.st.model_picker_open.set(false);
}

fn onTogglePicker(st: *AppState) void {
    st.model_picker_open.set(!st.model_picker_open.get());
}

fn selectionFor(st: *AppState, kind: models.Kind) i64 {
    return switch (kind) {
        .text => st.sel_llm.get(),
        .image => st.sel_sd.get(),
        .video => st.sel_video.get(),
        .tts => st.sel_tts.get(),
    };
}

fn kindIcon(kind: models.Kind) zigui.IconName {
    return switch (kind) {
        .text => .message_circle,
        .image => .image,
        .video => .film,
        .tts => .audio_lines,
    };
}

/// A header control showing the selected model for `kind` as a tappable pill
/// (icon · name · chevron) with a popover to switch it without leaving the
/// screen. Lists every local model of that kind.
pub fn modelPicker(st: *AppState, kind: models.Kind) zigui.View {
    const th = t();
    const fa = st.frame_arena.allocator();
    const sel = selectionFor(st, kind);
    const selected = st.selectedModel(sel);

    const name = if (selected) |m| m.name else "Select model";
    const name_col = if (selected != null) th.colors.label else th.colors.secondary_label;
    const trigger = zigui.HStack(.{
        zigui.Icon(kindIcon(kind), 14, th.colors.secondary_label),
        zigui.Text(name).font(.callout).foreground(name_col),
        zigui.Icon(.chevron_down, 14, th.colors.tertiary_label),
    }).spacing(7)
        .paddingInsets(.{ .top = 6, .leading = 11, .bottom = 6, .trailing = 9 })
        .background(th.colors.control_background)
        .cornerRadius(8)
        .border(th.colors.separator, th.metrics.hairline)
        .onTap(.{ .ctx = st, .func = struct {
            fn f(p: ?*anyopaque) void {
                onTogglePicker(@ptrCast(@alignCast(p.?)));
            }
        }.f });

    // Build the popover's model list (only models matching `kind`).
    var rows: std.ArrayList(zigui.View) = .empty;
    var shown: usize = 0;
    for (st.model_list.items.items, 0..) |m, i| {
        if (m.kind != kind) continue;
        const is_sel = sel >= 0 and @as(usize, @intCast(sel)) == i;
        var sbuf: [32]u8 = undefined;
        const check = if (is_sel) zigui.Icon(.check, 14, t().colors.accent) else zigui.Icon(.check, 14, zigui.Color.transparent);
        const row = zigui.HStack(.{
            check,
            zigui.VStack(.{
                zigui.Text(m.name).font(.subheadline),
                zigui.Text(fmt("{s}", .{models.humanSize(&sbuf, m.size)})).font(.caption2).foreground(th.colors.tertiary_label),
            }).spacing(1).alignment(zigui.Alignment.leading),
            zigui.Spacer(),
        }).spacing(8).frameMaxWidth()
            .paddingInsets(.{ .top = 5, .leading = 6, .bottom = 5, .trailing = 6 })
            .cornerRadius(6)
            .hoverFill(hoverTint())
            .onTap(.{ .ctx = pickCtx(st, i), .func = onPickModel });
        rows.append(fa, row) catch {};
        shown += 1;
    }
    if (shown == 0) {
        rows.append(fa, zigui.Text("No models — use the Models › Download tab.")
            .font(.caption).foreground(th.colors.secondary_label).padding(6)) catch {};
    }

    const content = card(zigui.ScrollViewState(&st.model_picker_scroll, zigui.VStack(rows.items).spacing(2).frameMaxWidth())
        .frameWidth(300).frameMaxHeight()).frameWidth(320);

    return trigger.popover(st.model_picker_open.binding(), content);
}
