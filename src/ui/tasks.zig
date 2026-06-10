//! Tasks screen: a live view of in-flight inference jobs across all three
//! backends, aggregating each backend's `JobState` (running / step / total).

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn jobRow(name: []const u8, status: []const u8, frac: ?f32) zigui.View {
    const th = w.t();
    const head = zigui.HStack(.{
        w.statusDot(w.green()),
        zigui.Text(name).font(.subheadline),
        zigui.Spacer(),
        zigui.Text(status).font(.caption).foreground(th.colors.secondary_label),
    }).spacing(8).frameMaxWidth();

    if (frac) |f| {
        return zigui.VStack(.{ head, zigui.ProgressView(f).frameMaxWidth() }).spacing(6).frameMaxWidth();
    }
    return head;
}

pub fn view(st: *AppState) zigui.View {
    const fa = st.frame_arena.allocator();
    var rows: std.ArrayList(zigui.View) = .empty;

    if (st.llama.isBusy()) {
        rows.append(fa, jobRow("Chat (llama.cpp)", "generating…", null)) catch {};
    }
    if (st.sd.isBusy()) {
        const step = st.sd.job.step.load(.acquire);
        const total = st.sd.job.total.load(.acquire);
        rows.append(fa, jobRow(
            "Image (stable-diffusion)",
            w.fmt("step {d}/{d}", .{ step, total }),
            st.sd.job.fraction(),
        )) catch {};
    }
    if (st.video.isBusy()) {
        const step = st.video.job.step.load(.acquire);
        const total = st.video.job.total.load(.acquire);
        rows.append(fa, jobRow(
            "Video (Wan 2.2)",
            w.fmt("step {d}/{d}", .{ step, total }),
            st.video.job.fraction(),
        )) catch {};
    }
    if (st.tts.isBusy()) {
        rows.append(fa, jobRow("Audio (qwen3-tts)", "synthesizing…", null)) catch {};
    }

    var body: zigui.View = undefined;
    if (rows.items.len == 0) {
        body = w.emptyState(.list, "No active tasks", "Inference jobs in flight will appear here.");
    } else {
        body = zigui.VStack(rows.items).spacing(14).frameMaxWidth();
    }

    return zigui.VStack(.{
        w.header("Tasks", zigui.Spacer()),
        w.card(body).frameMaxHeight(),
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
