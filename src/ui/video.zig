//! Video generation screen (Wan 2.2 via stable-diffusion.cpp). A two-column form
//! + frame preview, mirroring the Image screen. The selected video model is the
//! Wan diffusion .gguf; its VAE and umt5 text-encoder are auto-discovered next to
//! it (see state.generateVideo).
//!
//! NOTE: video runs on the GPU (Metal/CUDA/Vulkan) via two local ggml/sd.cpp
//! patches — a direct CONV_3D op and left/causal PAD that the Wan/LTX VAE need
//! (see backends/video.zig).

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onGenerate(st: *AppState) void {
    st.generateVideo();
}

fn onChooseImage(st: *AppState) void {
    st.chooseVideoImage();
}

fn onClearImage(st: *AppState) void {
    st.clearVideoImage();
}

fn leftPanel(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    var rows: std.ArrayList(zigui.View) = .empty;
    rows.append(fa, w.sectionHeader("Prompt")) catch {};
    rows.append(fa, zigui.TextEditor(&st.vid_prompt, &st.vid_scroll, false)
        .softWrap()
        .frameHeight(90)
        .padding(8)
        .background(th.colors.control_background)
        .cornerRadius(6)
        .border(th.colors.separator, th.metrics.hairline)
        .frameMaxWidth()) catch {};

    // Negative prompt is hidden for now — generateVideo always passes the default
    // Wan negative (st.vid_negative stays empty → default in generateVideo).

    // Output size, split into two compact pickers (Wan needs a real frame size —
    // tiny sizes give mush; a single 5-way picker is too wide for the panel).
    rows.append(fa, zigui.Picker(st.vid_orient.binding(), &[_][]const u8{ "Landscape", "Portrait", "Square" }).frameMaxWidth()) catch {};
    const sz = st.videoSize();
    rows.append(fa, w.settingRow(
        w.fmt("Quality · {d}×{d}", .{ sz.w, sz.h }),
        zigui.Picker(st.vid_quality.binding(), &[_][]const u8{ "480p", "720p" }).frameWidth(120),
    )) catch {};
    // Wan/LTX need real denoising: below ~10 steps the latent stays mostly
    // noise and the VAE decodes it to foggy mush, so the floor is 10 (default is
    // 30 — see state.zig). The slider can't go lower to avoid that footgun.
    rows.append(fa, w.settingRow(w.fmt("Steps: {d:.0}", .{st.vid_steps.get()}), zigui.Slider(st.vid_steps.binding(), 10, 50).frameWidth(160))) catch {};
    rows.append(fa, w.settingRow(w.fmt("CFG: {d:.1}", .{st.vid_cfg.get()}), zigui.Slider(st.vid_cfg.binding(), 1, 10).frameWidth(160))) catch {};
    rows.append(fa, w.settingRow("Frames", zigui.Stepper(w.fmt("{d}", .{st.vid_frames_n.get()}), st.vid_frames_n.binding(), 5, 121, 4))) catch {};

    // Optional start frame (image-to-video, Wan TI2V).
    rows.append(fa, w.sectionHeader("Start frame (optional)")) catch {};
    if (st.vid_init_image) |im| {
        const thumb: zigui.canvas.Image = .{
            .width = im.width,
            .height = im.height,
            .pixels = im.pixels[0 .. @as(usize, im.width) * @as(usize, im.height) * 4],
        };
        rows.append(fa, zigui.Image(thumb).scaledToFit().frameMaxWidth().frameHeight(120)
            .cornerRadius(6)) catch {};
        rows.append(fa, zigui.HStack(.{
            w.secondaryButton(.image, "Change", zigui.actionCtx(AppState, st, onChooseImage)),
            w.tintedButton(.close, "Remove", th.colors.destructive, zigui.actionCtx(AppState, st, onClearImage)),
        }).spacing(8)) catch {};
    } else {
        rows.append(fa, w.secondaryButton(.image, "Add start image", zigui.actionCtx(AppState, st, onChooseImage))) catch {};
    }

    const busy = st.video.isBusy();
    if (busy) {
        const frac = st.video.job.fraction();
        const step = st.video.job.step.load(.acquire);
        const total = st.video.job.total.load(.acquire);
        // The VAE decode now tiles, so its per-tile progress comes through the
        // same callback as sampling; `decoding` tells the two phases apart and
        // `frac` (step/total) is the right fraction for whichever is running.
        const decoding = st.video.decoding.load(.acquire);
        rows.append(fa, zigui.ProgressView(frac).frameMaxWidth()) catch {};
        rows.append(fa, zigui.Text(if (decoding)
            w.fmt("Decoding frames… tile {d}/{d}", .{ step, total })
        else
            w.fmt("Generating… step {d}/{d}", .{ step, total }))
            .font(.caption).foreground(th.colors.secondary_label)) catch {};
    } else {
        rows.append(fa, w.primaryButtonWide(.sparkles, "Generate", zigui.actionCtx(AppState, st, onGenerate))) catch {};
    }

    return w.card(zigui.VStack(rows.items).spacing(10)).frameWidth(340);
}

fn onOpenFolder(st: *AppState) void {
    st.openOutputsFolder();
}

fn rightPanel(st: *AppState) zigui.View {
    if (st.vid_result) |frames| {
        if (frames.len > 0) {
            const idx = if (st.vid_play_idx < frames.len) st.vid_play_idx else 0;
            const th = w.t();
            return w.card(zigui.VStack(.{
                zigui.Image(frames[idx]).scaledToFit().frameMaxWidth().frameMaxHeight(),
                zigui.HStack(.{
                    zigui.Text(w.fmt("frame {d}/{d} · {d} fps", .{ idx + 1, frames.len, st.vid_fps }))
                        .font(.caption).foreground(th.colors.secondary_label),
                    zigui.Spacer(),
                    w.secondaryButton(.folder, "Open folder", zigui.actionCtx(AppState, st, onOpenFolder)),
                }).spacing(8).frameMaxWidth(),
            }).spacing(8).frameMaxWidth().frameMaxHeight()).frameMaxWidth().frameMaxHeight();
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
