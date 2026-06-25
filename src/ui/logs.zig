//! Logs screen: a monospaced, scrollable view of backend log lines captured via
//! each backend's log callback (wired in later phases). Reads the shared LogRing.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onClear(st: *AppState) void {
    for (st.logs.lines.items) |l| st.gpa.free(l);
    st.logs.lines.clearRetainingCapacity();
}

pub fn view(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();
    const lines = st.logs.lines.items;

    var body: zigui.View = undefined;
    if (lines.len == 0) {
        body = w.emptyState(.scroll_text, "No log output", "Backend log lines will stream in here.");
    } else {
        var rows: std.ArrayList(zigui.View) = .empty;
        for (lines) |line| {
            // WrappedText so long lines (paths, error messages) wrap instead of
            // overflowing the card. `frameAlign(.leading)` keeps short lines pinned
            // to the left edge (a max-width frame centers its content by default).
            rows.append(fa, zigui.WrappedText(line).font(.caption2).foreground(th.colors.secondary_label).frameMaxWidth().frameAlign(.leading)) catch {};
        }
        body = zigui.ScrollViewState(&st.log_scroll, zigui.VStack(rows.items).spacing(2).alignment(zigui.Alignment.leading).frameMaxWidth())
            .frameMaxWidth().frameMaxHeight();
    }

    return zigui.VStack(.{
        w.header("Logs", zigui.components.ButtonRoled("Clear", .normal, zigui.actionCtx(AppState, st, onClear))),
        w.card(body).frameMaxHeight(),
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
