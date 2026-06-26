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
const downloader = @import("../backends/downloader.zig");
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
        // List quants smallest-first. Sort an index array so the underlying
        // file set (owned by state) keeps its order.
        var qidx: std.ArrayList(usize) = .empty;
        for (files.items, 0..) |f, i| {
            if (f.is_quant) qidx.append(fa, i) catch {};
        }
        std.mem.sort(usize, qidx.items, files.items, struct {
            fn lt(items: []const st_mod.RepoFile, a: usize, b: usize) bool {
                return items[a].size < items[b].size;
            }
        }.lt);
        for (qidx.items) |i| {
            const f = files.items[i];
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

// --- results table ----------------------------------------------------------

// Column positions in the results table. The `sortable` numeric columns map to
// the comparators in `orderLess`; keep these in sync with `columns` in `view`.
const COL_TYPE = 0;
const COL_NAME = 1;
const COL_SIZE = 2;
const COL_DOWNLOADS = 3;
const COL_LIKES = 4;
const COL_UPDATED = 5;
const COL_GET = 6;

const OrderCtx = struct { items: []const st_mod.HFRepo, col: i64, asc: bool };

/// Less-than over *display order* (indices into `dl_results`), per the active
/// sort. We sort an index array rather than `dl_results` itself so row indices
/// stay stable — `dl_filepick_idx` and incoming size events both key off them.
fn orderLess(ctx: OrderCtx, ia: usize, ib: usize) bool {
    const a = ctx.items[ia];
    const b = ctx.items[ib];
    return switch (ctx.col) {
        COL_SIZE => blk: {
            // Repos whose size hasn't loaded yet always sink to the bottom.
            if (a.size_loaded != b.size_loaded) break :blk a.size_loaded;
            if (!a.size_loaded or a.size_min == b.size_min) break :blk false;
            break :blk if (ctx.asc) a.size_min < b.size_min else a.size_min > b.size_min;
        },
        COL_DOWNLOADS => ordI64(a.downloads, b.downloads, ctx.asc),
        COL_LIKES => ordI64(a.likes, b.likes, ctx.asc),
        COL_UPDATED => ordI64(a.last_modified, b.last_modified, ctx.asc),
        else => false,
    };
}

fn ordI64(a: i64, b: i64, asc: bool) bool {
    if (a == b) return false;
    return if (asc) a < b else a > b;
}

// --- cell formatting --------------------------------------------------------

/// "6.0 GB–80 GB" once the repo's quants have been measured, a single value if
/// all quants are the same size, or "…" while the background fetch is pending.
fn sizeRangeText(repo: st_mod.HFRepo) []const u8 {
    if (!repo.size_loaded) return "…";
    var b1: [32]u8 = undefined;
    var b2: [32]u8 = undefined;
    if (repo.size_min == repo.size_max) return w.fmt("{s}", .{models.humanSize(&b1, repo.size_min)});
    return w.fmt("{s}–{s}", .{ models.humanSize(&b1, repo.size_min), models.humanSize(&b2, repo.size_max) });
}

/// Compact count: 980, 12.3k, 4.1M.
fn countText(n: i64) []const u8 {
    if (n < 1000) return w.fmt("{d}", .{n});
    const f: f64 = @floatFromInt(n);
    if (n < 1_000_000) return w.fmt("{d:.1}k", .{f / 1000.0});
    return w.fmt("{d:.1}M", .{f / 1_000_000.0});
}

/// The last-modified date as "YYYY-MM-DD" (or "—" when unknown). Absolute rather
/// than relative so it needs no wall-clock; sorting still uses the raw epoch.
fn dateText(epoch: i64) []const u8 {
    if (epoch <= 0) return "—";
    const es = std.time.epoch.EpochSeconds{ .secs = @intCast(epoch) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    return w.fmt("{d:0>4}-{d:0>2}-{d:0>2}", .{ yd.year, md.month.numeric(), @as(u16, md.day_index) + 1 });
}

fn nameCell(repo: st_mod.HFRepo) zigui.View {
    const th = w.t();
    return zigui.VStack(.{
        zigui.Text(repo.name()).font(.subheadline).truncated(),
        zigui.Text(repo.author()).font(.caption2).foreground(th.colors.tertiary_label).truncated(),
    }).spacing(2).alignment(zigui.Alignment.leading);
}

fn numCell(s: []const u8) zigui.View {
    return zigui.Text(s).font(.caption).foreground(w.t().colors.secondary_label);
}

/// True when a repo already lives in one of the scanned model folders (app dir or
/// any Settings folder). Downloads land in `…/<author>/<name>/`, so we match a
/// scanned model whose folder is `<name>` inside an `<author>` folder — robust to
/// path separators and the kind subfolder.
fn isDownloaded(st: *AppState, repo: st_mod.HFRepo) bool {
    const name = repo.name();
    const author = repo.author();
    if (name.len == 0 or author.len == 0) return false;
    for (st.model_list.items.items) |m| {
        if (!std.ascii.eqlIgnoreCase(std.fs.path.basename(m.dir), name)) continue;
        const parent = std.fs.path.dirname(m.dir) orelse continue;
        if (std.ascii.eqlIgnoreCase(std.fs.path.basename(parent), author)) return true;
    }
    return false;
}

/// The trailing action for a row: an "Installed" badge when the repo is already
/// downloaded, otherwise the "Get" button (with the quant popover on the active
/// row). `orig` is the index into `dl_results` (stable across sorting).
fn getCell(st: *AppState, orig: usize, installed: bool) zigui.View {
    const th = w.t();
    if (installed) {
        return zigui.HStack(.{
            zigui.Icon(.check, 13, w.green()),
            zigui.Text("Installed").font(.callout).foreground(th.colors.secondary_label),
        }).spacing(4)
            .paddingInsets(.{ .top = 5, .leading = 9, .bottom = 5, .trailing = 9 });
    }
    const dl_btn = zigui.HStack(.{
        zigui.Icon(.download, 13, th.colors.on_accent),
        zigui.Text("Get").font(.callout).foreground(th.colors.on_accent),
        zigui.Icon(.chevron_down, 12, th.colors.on_accent),
    }).spacing(4)
        .paddingInsets(.{ .top = 5, .leading = 9, .bottom = 5, .trailing = 7 })
        .background(th.colors.accent)
        .cornerRadius(7)
        .onTap(.{ .ctx = rowCtx(st, orig), .func = onOpenFiles });
    if (st.dl_filepick_idx == @as(i64, @intCast(orig)))
        return dl_btn.popover(st.dl_filepick_open.binding(), quantPopover(st));
    return dl_btn;
}

/// The sortable results table: one row per repo in the current sort order.
fn resultsTable(st: *AppState) zigui.View {
    const fa = st.frame_arena.allocator();
    const n = st.dl_results.items.len;

    // Display order: indices into dl_results, sorted per dl_sort (index < 0 keeps
    // the HF API's download-ranked order).
    const order = fa.alloc(usize, n) catch return zigui.Text("Out of memory");
    for (order, 0..) |*o, i| o.* = i;
    const sort = st.dl_sort.get();
    if (sort.index >= 0 and n > 1) {
        std.mem.sort(usize, order, OrderCtx{
            .items = st.dl_results.items,
            .col = sort.index,
            .asc = sort.dir == .ascending,
        }, orderLess);
    }

    const rows = fa.alloc([]const zigui.View, n) catch return zigui.Text("Out of memory");
    for (order, 0..) |orig, p| {
        const repo = st.dl_results.items[orig];
        const cells = fa.alloc(zigui.View, 7) catch return zigui.Text("Out of memory");
        cells[COL_TYPE] = mb.kindBadge(repo.kind);
        cells[COL_NAME] = nameCell(repo);
        cells[COL_SIZE] = numCell(sizeRangeText(repo));
        cells[COL_DOWNLOADS] = numCell(countText(repo.downloads));
        cells[COL_LIKES] = numCell(countText(repo.likes));
        cells[COL_UPDATED] = numCell(dateText(repo.last_modified));
        cells[COL_GET] = getCell(st, orig, isDownloaded(st, repo));
        rows[p] = cells;
    }

    const columns = [_]zigui.DataColumn{
        .{ .title = "", .width = 58 },
        .{ .title = "Name" },
        .{ .title = "Size", .width = 108, .sortable = true, .trailing = true },
        .{ .title = "Downloads", .width = 78, .sortable = true, .trailing = true },
        .{ .title = "Likes", .width = 54, .sortable = true, .trailing = true },
        .{ .title = "Updated", .width = 74, .sortable = true, .trailing = true },
        .{ .title = "", .width = 84, .trailing = true },
    };
    return zigui.DataTable(&columns, rows, st.dl_sort.binding(), &st.dl_scroll);
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
    const retry = st.downloader.retry_attempt.load(.acquire);
    const model = st.dl_active orelse "Downloading…";

    // Current file line: "umt5-xxl.gguf (2 of 4)", or a reconnect status while
    // retrying a dropped connection.
    const file_line = if (retry > 0)
        w.fmt("Reconnecting… ({d}/{d})", .{ retry, downloader.Backend.max_stalls })
    else if (st.dl_active_file) |f|
        if (fcount > 1) w.fmt("{s}  ({d} of {d})", .{ f, fidx, fcount }) else w.fmt("{s}", .{f})
    else
        "Preparing…";
    const file_color = if (retry > 0) th.colors.accent else th.colors.tertiary_label;

    return w.card(zigui.VStack(.{
        zigui.HStack(.{
            zigui.Icon(.download, 16, th.colors.accent),
            zigui.Text(model).font(.subheadline),
            zigui.Spacer(),
            w.tintedButton(.close, "Cancel", th.colors.destructive, zigui.actionCtx(AppState, st, onCancel)),
        }).spacing(8).frameMaxWidth(),
        zigui.Text(file_line).font(.caption2).foreground(file_color).frameMaxWidth(),
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
            .frameWidth(360),
        w.primaryButton(.search, "Search", zigui.actionCtx(AppState, st, onSearch)),
    }).spacing(8).frameMaxWidth();

    // Results: an empty-state prompt, or the sortable table.
    const results = if (st.dl_results.items.len == 0)
        w.emptyState(
            .download,
            "Search HuggingFace for GGUF models",
            "Type a name (e.g. \"qwen\", \"llama\", \"flux\") and press Search.",
        )
    else
        resultsTable(st);

    var col: std.ArrayList(zigui.View) = .empty;
    col.append(fa, search_bar) catch {};
    if (st.downloader.isBusy() or st.dl_active != null) col.append(fa, activeRow(st)) catch {};
    col.append(fa, w.card(results).frameMaxHeight()) catch {};

    return zigui.VStack(col.items).spacing(12).frameMaxWidth().frameMaxHeight();
}
