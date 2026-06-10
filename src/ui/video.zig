//! Video generation screen (Wan 2.2 via stable-diffusion.cpp). A two-column form
//! + frame preview, mirroring the Image screen. The selected video model is the
//! Wan diffusion .gguf; its VAE and umt5 text-encoder are auto-discovered next to
//! it (see state.generateVideo).
//!
//! NOTE: video runs on the CPU backend — upstream ggml's Metal backend doesn't
//! implement the IM2COL_3D op Wan needs (see backends/video.zig).

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onGenerate(st: *AppState) void {
    st.generateVideo();
}

fn leftPanel(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    var rows: std.ArrayList(zigui.View) = .empty;
    rows.append(fa, w.sectionHeader("Prompt")) catch {};
    rows.append(fa, zigui.TextEditor(&st.vid_prompt, &st.vid_scroll, false)
        .frameHeight(90)
        .padding(8)
        .background(th.colors.control_background)
        .cornerRadius(6)
        .border(th.colors.separator, th.metrics.hairline)
        .frameMaxWidth()) catch {};

    rows.append(fa, w.settingRow(w.fmt("Steps: {d:.0}", .{st.vid_steps.get()}), zigui.Slider(st.vid_steps.binding(), 1, 40).frameWidth(160))) catch {};
    rows.append(fa, w.settingRow("Frames", zigui.Stepper(w.fmt("{d}", .{st.vid_frames_n.get()}), st.vid_frames_n.binding(), 5, 81, 4))) catch {};

    rows.append(fa, zigui.Text("Metal · 256×256")
        .font(.caption2).foreground(th.colors.secondary_label).frameMaxWidth()) catch {};

    const busy = st.video.isBusy();
    if (busy) {
        const frac = st.video.job.fraction();
        const step = st.video.job.step.load(.acquire);
        const total = st.video.job.total.load(.acquire);
        rows.append(fa, zigui.ProgressView(frac).frameMaxWidth()) catch {};
        rows.append(fa, zigui.Text(w.fmt("Generating… step {d}/{d}", .{ step, total }))
            .font(.caption).foreground(th.colors.secondary_label)) catch {};
    } else {
        rows.append(fa, w.primaryButtonWide(.sparkles, "Generate", zigui.actionCtx(AppState, st, onGenerate))) catch {};
    }

    return w.card(zigui.VStack(rows.items).spacing(10)).frameWidth(340);
}

fn rightPanel(st: *AppState) zigui.View {
    if (st.vid_result) |frames| {
        if (frames.len > 0) {
            const idx = if (st.vid_play_idx < frames.len) st.vid_play_idx else 0;
            const th = w.t();
            return w.card(zigui.VStack(.{
                zigui.Image(frames[idx]).frameMaxWidth().frameMaxHeight(),
                zigui.Text(w.fmt("frame {d}/{d} · {d} fps", .{ idx + 1, frames.len, st.vid_fps }))
                    .font(.caption).foreground(th.colors.secondary_label).frameMaxWidth(),
            }).spacing(8)).frameMaxWidth().frameMaxHeight();
        }
    }
    return w.card(w.emptyState(.film, "No video yet", "Enter a prompt and press Generate."))
        .frameMaxWidth()
        .frameMaxHeight();
}

pub fn view(st: *AppState) zigui.View {
    return zigui.VStack(.{
        w.header("Video Generation", w.modelPicker(st, .video)),
        zigui.HStack(.{
            zigui.VStack(.{ leftPanel(st), zigui.Spacer() }).frameWidth(340).frameMaxHeight(),
            rightPanel(st),
        }).spacing(12).frameMaxWidth().frameMaxHeight(),
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
