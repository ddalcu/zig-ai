//! HuggingFace downloader tab (lives inside the Models screen). Search HF for
//! GGUF repos, filter by category, pick a quant from a popover, and download it
//! with live progress + cancel into the scanned models folder.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const mb = @import("model_browser.zig");
const st_mod = @import("../state.zig");
const models = @import("../models.zig");
const manifest = @import("../manifest.zig");
const AppState = st_mod.AppState;

fn onSearch(st: *AppState) void {
    st.dlSearch();
}

fn onCancel(st: *AppState) void {
    st.dlCancel();
}

// --- per-row / per-file contexts --------------------------------------------

const RowCtx = struct { st: *AppState, index: usize };
const FileCtx = struct { st: *AppState, path: []const u8 };

fn rowCtx(st: *AppState, index: usize) *RowCtx {
    const cx = st.frame_arena.allocator().create(RowCtx) catch unreachable;
    cx.* = .{ .st = st, .index = index };
    return cx;
}

fn fileCtx(st: *AppState, path: []const u8) *FileCtx {
    const cx = st.frame_arena.allocator().create(FileCtx) catch unreachable;
    cx.* = .{ .st = st, .path = path };
    return cx;
}

fn onOpenFiles(p: ?*anyopaque) void {
    const cx: *RowCtx = @ptrCast(@alignCast(p.?));
    cx.st.dlOpenFiles(cx.index);
}

fn onPickFile(p: ?*anyopaque) void {
    const cx: *FileCtx = @ptrCast(@alignCast(p.?));
    cx.st.dlStart(cx.path);
}

// --- quant popover ----------------------------------------------------------

/// The popover body listing this repo's selectable quants (loaded lazily).
/// Tapping one downloads it together with every support file in the repo.
fn quantPopover(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    var rows: std.ArrayList(zigui.View) = .empty;
    rows.append(fa, zigui.Text("Choose a quant")
        .font(.caption).foreground(th.colors.secondary_label).frameMaxWidth()) catch {};

    if (st.dl_files) |files| {
        var quants: usize = 0;
        var support: usize = 0;
        for (files.items) |f| {
            if (f.is_quant) quants += 1 else support += 1;
        }
        if (quants == 0) {
            rows.append(fa, zigui.Text("No GGUF model files in this repo.")
                .font(.caption).foreground(th.colors.secondary_label)) catch {};
        }
        for (files.items) |f| {
            if (!f.is_quant) continue;
            var sbuf: [32]u8 = undefined;
            rows.append(fa, zigui.HStack(.{
                zigui.Text(f.path).font(.caption),
                zigui.Spacer(),
                zigui.Text(w.fmt("{s}", .{models.humanSize(&sbuf, f.size)}))
                    .font(.caption2).foreground(th.colors.tertiary_label),
            }).spacing(12).frameMaxWidth()
                .paddingInsets(.{ .top = 6, .leading = 6, .bottom = 6, .trailing = 6 })
                .cornerRadius(6)
                .hoverFill(w.hoverTint())
                .onTap(.{ .ctx = fileCtx(st, f.path), .func = onPickFile })) catch {};
        }
        if (support > 0) {
            rows.append(fa, zigui.Divider()) catch {};
            rows.append(fa, zigui.Text(w.fmt("+ {d} support file(s) download automatically", .{support}))
                .font(.caption2).foreground(th.colors.tertiary_label).frameMaxWidth()) catch {};
        }

        // Curated cross-repo sidecars (FLUX/Wan VAE + text encoder) that this
        // model needs, pulled into the same folder so it's runnable.
        const sidecars = if (st.dl_filepick_idx >= 0 and @as(usize, @intCast(st.dl_filepick_idx)) < st.dl_results.items.len)
            manifest.sidecarsFor(st.dl_results.items[@intCast(st.dl_filepick_idx)].id)
        else
            &.{};
        if (sidecars.len > 0) {
            rows.append(fa, zigui.Divider()) catch {};
            rows.append(fa, zigui.Text("Also fetched (other repos):")
                .font(.caption2).foreground(th.colors.secondary_label).frameMaxWidth()) catch {};
            for (sidecars) |sc| {
                rows.append(fa, zigui.HStack(.{
                    zigui.Icon(.download, 11, th.colors.tertiary_label),
                    zigui.Text(sc.label).font(.caption2).foreground(th.colors.tertiary_label),
                    zigui.Spacer(),
                }).spacing(5).frameMaxWidth()) catch {};
            }
        }
    } else {
        rows.append(fa, zigui.Text("Loading files…")
            .font(.caption).foreground(th.colors.secondary_label)) catch {};
    }

    const content = zigui.ScrollViewState(&st.model_picker_scroll, zigui.VStack(rows.items).spacing(2).frameMaxWidth())
        .frameWidth(320).frameMaxHeight();
    return w.card(content).frameWidth(340);
}

// --- result rows ------------------------------------------------------------

fn resultRow(st: *AppState, index: usize, repo: st_mod.HFRepo) zigui.View {
    const th = w.t();

    const dl_btn = zigui.HStack(.{
        zigui.Icon(.download, 14, th.colors.on_accent),
        zigui.Text("Get").font(.callout).foreground(th.colors.on_accent),
        zigui.Icon(.chevron_down, 13, th.colors.on_accent),
    }).spacing(5)
        .paddingInsets(.{ .top = 6, .leading = 11, .bottom = 6, .trailing = 9 })
        .background(th.colors.accent)
        .cornerRadius(8)
        .onTap(.{ .ctx = rowCtx(st, index), .func = onOpenFiles });
    // Attach the quant popover only to the row that opened it.
    const trailing = if (st.dl_filepick_idx == @as(i64, @intCast(index)))
        dl_btn.popover(st.dl_filepick_open.binding(), quantPopover(st))
    else
        dl_btn;

    return zigui.HStack(.{
        mb.kindBadge(repo.kind),
        zigui.VStack(.{
            zigui.Text(repo.name()).font(.subheadline),
            zigui.HStack(.{
                zigui.Text(repo.author()).font(.caption2).foreground(th.colors.tertiary_label),
                zigui.Icon(.download, 11, th.colors.tertiary_label),
                zigui.Text(w.fmt("{d}", .{repo.downloads})).font(.caption2).foreground(th.colors.tertiary_label),
                zigui.Icon(.heart, 11, th.colors.tertiary_label),
                zigui.Text(w.fmt("{d}", .{repo.likes})).font(.caption2).foreground(th.colors.tertiary_label),
            }).spacing(5),
        }).spacing(3),
        zigui.Spacer(),
        trailing,
    }).spacing(10).frameMaxWidth();
}

// --- active download --------------------------------------------------------

fn activeRow(st: *AppState) zigui.View {
    const th = w.t();
    const done = st.downloader.bytes_done.load(.acquire);
    const total = st.downloader.bytes_total.load(.acquire);
    const speed = st.downloader.speed_bps.load(.acquire);
    const frac: f32 = if (total > 0)
        std.math.clamp(@as(f32, @floatFromInt(done)) / @as(f32, @floatFromInt(total)), 0, 1)
    else
        0;
    const pct: u32 = @intFromFloat(frac * 100);

    var dbuf: [32]u8 = undefined;
    var tbuf: [32]u8 = undefined;
    var spbuf: [32]u8 = undefined;
    const fidx = st.downloader.file_index.load(.acquire);
    const fcount = st.downloader.file_count.load(.acquire);
    const model = st.dl_active orelse "Downloading…";

    // Current file line: "umt5-xxl.gguf (2 of 4)".
    const file_line = if (st.dl_active_file) |f|
        if (fcount > 1) w.fmt("{s}  ({d} of {d})", .{ f, fidx, fcount }) else w.fmt("{s}", .{f})
    else
        "Preparing…";

    return w.card(zigui.VStack(.{
        zigui.HStack(.{
            zigui.Icon(.download, 16, th.colors.accent),
            zigui.Text(model).font(.subheadline),
            zigui.Spacer(),
            w.tintedButton(.close, "Cancel", th.colors.destructive, zigui.actionCtx(AppState, st, onCancel)),
        }).spacing(8).frameMaxWidth(),
        zigui.Text(file_line).font(.caption2).foreground(th.colors.tertiary_label).frameMaxWidth(),
        zigui.ProgressView(frac).frameMaxWidth(),
        zigui.Text(w.fmt("{d}% · {s} / {s} · {s}/s", .{
            pct,
            models.humanSize(&dbuf, done),
            models.humanSize(&tbuf, total),
            models.humanSize(&spbuf, speed),
        })).font(.caption).foreground(th.colors.secondary_label),
    }).spacing(8));
}

// --- view -------------------------------------------------------------------

pub fn view(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    const search_field = zigui.HStack(.{
        zigui.Icon(.search, 15, th.colors.secondary_label),
        zigui.TextField("Search HuggingFace…", &st.dl_search)
            .onSubmit(zigui.actionCtx(AppState, st, onSearch))
            .frameMaxWidth(),
    }).spacing(8)
        .paddingInsets(.{ .top = 7, .leading = 11, .bottom = 7, .trailing = 12 })
        .background(th.colors.control_background)
        .cornerRadius(8)
        .border(th.colors.separator, th.metrics.hairline)
        .frameMaxWidth();

    const search_bar = zigui.HStack(.{
        search_field,
        zigui.Picker(st.dl_category.binding(), &[_][]const u8{ "All", "Chat", "Image", "Video", "TTS" })
            .frameWidth(280),
        w.primaryButton(.search, "Search", zigui.actionCtx(AppState, st, onSearch)),
    }).spacing(8).frameMaxWidth();

    // Results list.
    var rows: std.ArrayList(zigui.View) = .empty;
    if (st.dl_results.items.len == 0) {
        rows.append(fa, w.emptyState(
            .download,
            "Search HuggingFace for GGUF models",
            "Type a name (e.g. \"qwen\", \"llama\", \"flux\") and press Search.",
        )) catch {};
    } else {
        for (st.dl_results.items, 0..) |repo, i| {
            rows.append(fa, resultRow(st, i, repo)) catch {};
            if (i + 1 < st.dl_results.items.len) rows.append(fa, zigui.Divider()) catch {};
        }
    }
    const list = zigui.ScrollViewState(&st.dl_scroll, zigui.VStack(rows.items).spacing(8).frameMaxWidth())
        .frameMaxWidth()
        .frameMaxHeight();

    var col: std.ArrayList(zigui.View) = .empty;
    col.append(fa, search_bar) catch {};
    if (st.downloader.isBusy() or st.dl_active != null) col.append(fa, activeRow(st)) catch {};
    col.append(fa, w.card(list).frameMaxHeight()) catch {};

    return zigui.VStack(col.items).spacing(12).frameMaxWidth().frameMaxHeight();
}
