//! Model Browser: lists every discovered local GGUF with its kind and size, and
//! lets the user pick which model each task (chat / image / tts) should use.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const models = @import("../models.zig");
const AppState = st_mod.AppState;

const RowCtx = struct { st: *AppState, index: usize };

fn rowCtx(st: *AppState, index: usize) *RowCtx {
    const cx = st.frame_arena.allocator().create(RowCtx) catch unreachable;
    cx.* = .{ .st = st, .index = index };
    return cx;
}

fn onUse(p: ?*anyopaque) void {
    const cx: *RowCtx = @ptrCast(@alignCast(p.?));
    const idx: i64 = @intCast(cx.index);
    const m = cx.st.model_list.items.items[cx.index];
    switch (m.kind) {
        .text => cx.st.sel_llm.set(idx),
        .image => cx.st.sel_sd.set(idx),
        .video => cx.st.sel_video.set(idx),
        .tts => cx.st.sel_tts.set(idx),
    }
}

fn onRescan(st: *AppState) void {
    st.rescanModels();
}

fn onDelete(p: ?*anyopaque) void {
    const cx: *RowCtx = @ptrCast(@alignCast(p.?));
    cx.st.requestDeleteModel(cx.index);
}

fn isSelected(st: *AppState, index: usize, kind: models.Kind) bool {
    const sel: i64 = switch (kind) {
        .text => st.sel_llm.get(),
        .image => st.sel_sd.get(),
        .video => st.sel_video.get(),
        .tts => st.sel_tts.get(),
    };
    return sel >= 0 and @as(usize, @intCast(sel)) == index;
}

/// The icon representing each model kind, shared by badges and empty states.
pub fn kindIcon(kind: models.Kind) zigui.IconName {
    return switch (kind) {
        .text => .message_circle,
        .image => .image,
        .video => .film,
        .tts => .audio_lines,
    };
}

pub fn kindColor(kind: models.Kind) zigui.Color {
    return switch (kind) {
        .text => w.t().colors.accent,
        .image => zigui.Color.fromRgb8(175, 82, 222),
        .video => zigui.Color.fromRgb8(255, 45, 85),
        .tts => zigui.Color.fromRgb8(255, 149, 0),
    };
}

pub fn kindBadge(kind: models.Kind) zigui.View {
    const th = w.t();
    return zigui.HStack(.{
        zigui.Icon(kindIcon(kind), 11, th.colors.on_accent),
        zigui.Text(kind.label()).font(.caption2).foreground(th.colors.on_accent),
    }).spacing(4)
        .paddingInsets(.{ .top = 2, .leading = 6, .bottom = 2, .trailing = 7 })
        .background(kindColor(kind))
        .cornerRadius(6);
}

fn modelRow(st: *AppState, index: usize, m: models.ModelInfo) zigui.View {
    const th = w.t();
    var size_buf: [32]u8 = undefined;
    const size_str = models.humanSize(&size_buf, m.size);
    const selected = isSelected(st, index, m.kind);

    const use_btn = if (selected)
        zigui.HStack(.{
            zigui.Icon(.check, 14, w.green()),
            zigui.Text("In use").font(.caption).foreground(w.green()),
        }).spacing(4)
    else
        zigui.HStack(.{
            zigui.Text("Use").font(.caption).foreground(th.colors.accent),
        }).onTap(.{ .ctx = rowCtx(st, index), .func = onUse });

    // A trash icon to delete the model — only for models we own (the app's own
    // folder). Models from LM Studio / mlx-serve / custom folders are read-only.
    const del_btn: zigui.View = if (m.source.owned())
        zigui.Icon(.trash, 15, th.colors.tertiary_label)
            .onTap(.{ .ctx = rowCtx(st, index), .func = onDelete })
    else
        zigui.Icon(.lock, 13, th.colors.tertiary_label.withAlpha(0.5));

    return zigui.HStack(.{
        kindBadge(m.kind),
        zigui.VStack(.{
            zigui.Text(m.name).font(.subheadline),
            zigui.Text(w.fmt("{s} · {s}", .{ size_str, m.kind.label() }))
                .font(.caption2)
                .foreground(th.colors.tertiary_label),
        }).spacing(2),
        zigui.Spacer(),
        sourceBadge(m.source),
        use_btn,
        del_btn,
    }).spacing(10).frameMaxWidth();
}

/// A small, muted pill showing where a model came from (zig-ai / LM Studio /
/// mlx-serve / custom). zig-ai models are tinted with the accent to stand out as
/// the ones the app manages.
fn sourceBadge(source: models.Source) zigui.View {
    const th = w.t();
    const owned = source.owned();
    const fg = if (owned) th.colors.accent else th.colors.secondary_label;
    return zigui.Text(source.label())
        .font(.caption2)
        .foreground(fg)
        .paddingInsets(.{ .top = 2, .leading = 7, .bottom = 2, .trailing = 7 })
        .background(fg.withAlpha(0.12))
        .cornerRadius(6);
}

/// The local-models list filtered to a single `kind`. Reused by tabs 0..3.
fn localList(st: *AppState, kind: models.Kind) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();
    const items = st.model_list.items.items;

    var rows: std.ArrayList(zigui.View) = .empty;
    var shown: usize = 0;
    for (items, 0..) |m, i| {
        if (m.kind != kind) continue;
        if (shown > 0) rows.append(fa, zigui.Divider()) catch {};
        rows.append(fa, modelRow(st, i, m)) catch {};
        shown += 1;
    }

    if (shown == 0) {
        return w.card(w.emptyState(
            kindIcon(kind),
            w.fmt("No {s} models found", .{kind.label()}),
            "Add a folder in Settings and Rescan, or use the Download tab.",
        )).frameMaxHeight();
    }

    const list = zigui.ScrollViewState(&st.models_scroll, zigui.VStack(rows.items).spacing(8).frameMaxWidth())
        .frameMaxWidth()
        .frameMaxHeight();

    const footer = zigui.Text(w.fmt("{d} {s} · {d} models total", .{
        shown, kind.label(), st.model_list.items.items.len,
    })).font(.caption).foreground(th.colors.tertiary_label);

    return zigui.VStack(.{ w.card(list).frameMaxHeight(), footer })
        .spacing(8).frameMaxWidth().frameMaxHeight();
}

/// Map a tab index (0..3) to its model Kind.
fn tabKind(tab: i64) models.Kind {
    return switch (tab) {
        0 => .text,
        1 => .image,
        2 => .video,
        else => .tts,
    };
}

pub fn view(st: *AppState) zigui.View {
    const tab = st.models_tab.get();

    const tabs = zigui.Picker(st.models_tab.binding(), &[_][]const u8{
        "Chat", "Image", "Video", "TTS", "Download",
    }).frameMaxWidth();

    const body = if (tab == 4)
        @import("downloader.zig").view(st)
    else
        localList(st, tabKind(tab));

    return zigui.VStack(.{
        w.header("Models", w.secondaryButton(.refresh, "Rescan", zigui.actionCtx(AppState, st, onRescan))),
        tabs,
        body,
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
