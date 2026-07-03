//! Video generation screen (Wan 2.2 / LTX-2.3 via stable-diffusion.cpp). A
//! two-column form + frame preview, mirroring the Image screen. The selected
//! video model is the diffusion .gguf; its VAE, text-encoder and (LTX) audio
//! VAE / connectors / spatial upscaler sidecars are auto-discovered next to it
//! (see state.generateVideo). LTX models also produce an audio track, saved as
//! a WAV beside the MP4.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onGenerate(st: *AppState) void {
    st.generateVideo();
}

fn onCancel(st: *AppState) void {
    st.video.cancel();
}

fn onToggleAdvanced(st: *AppState) void {
    st.vid_advanced.set(!st.vid_advanced.get());
}

fn onChooseImage(st: *AppState) void {
    st.chooseVideoImage();
}

fn onClearImage(st: *AppState) void {
    st.clearVideoImage();
}

fn onChooseEndImage(st: *AppState) void {
    st.chooseVideoEndImage();
}

fn onClearEndImage(st: *AppState) void {
    st.clearVideoEndImage();
}

fn onChooseLora(st: *AppState) void {
    st.chooseVideoLora();
}

fn onClearLora(st: *AppState) void {
    st.clearVideoLora();
}

/// Thumbnail + Change/Remove rows for a picked frame (start or end).
fn framePickRows(st: *AppState, rows: *std.ArrayList(zigui.View), im: st_mod.codecs.DecodedImage, change: zigui.Callback, remove: zigui.Callback) void {
    const th = w.t();
    const fa = st.frame_arena.allocator();
    const thumb: zigui.canvas.Image = .{
        .width = im.width,
        .height = im.height,
        .pixels = im.pixels[0 .. @as(usize, im.width) * @as(usize, im.height) * 4],
    };
    rows.append(fa, zigui.Image(thumb).scaledToFit().frameMaxWidth().frameHeight(100)
        .cornerRadius(6)) catch {};
    rows.append(fa, zigui.HStack(.{
        w.secondaryButton(.image, "Change", change),
        w.tintedButton(.close, "Remove", th.colors.destructive, remove),
    }).spacing(8)) catch {};
}

fn leftPanel(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    var rows: std.ArrayList(zigui.View) = .empty;
    rows.append(fa, w.sectionHeader("Prompt")) catch {};
    // TextEditor paints its own themed surface (fill + border); wrapping it in
    // `.background`/`.border` again double-frames it (see chat.zig input bar).
    rows.append(fa, zigui.TextEditor(&st.vid_prompt, &st.vid_scroll, false)
        .softWrap()
        .frameHeight(90)
        .padding(8)
        .frameMaxWidth()) catch {};

    // Output size, split into two compact pickers (Wan needs a real frame size —
    // tiny sizes give mush; a single 5-way picker is too wide for the panel).
    rows.append(fa, zigui.Picker(st.vid_orient.binding(), &[_][]const u8{ "Landscape", "Portrait", "Square" }).frameMaxWidth()) catch {};
    const sz = st.videoSize();
    // Match the slider column: 160pt frame, minus the 8pt the slider groove
    // insets each side (slider_knob_r) as trailing pad, so the segmented box's
    // right edge lines up with the grooves below instead of overhanging them.
    rows.append(fa, w.settingRow(
        w.fmt("Quality · {d}×{d}", .{ sz.w, sz.h }),
        zigui.Picker(st.vid_quality.binding(), &[_][]const u8{ "480p", "720p" })
            .frameWidth(152)
            .paddingInsets(.{ .trailing = 8 }),
    )) catch {};
    // Wan/LTX need real denoising: below ~10 steps the latent stays mostly
    // noise and the VAE decodes it to foggy mush, so the floor is 10 (default is
    // 30 — see state.zig). The slider can't go lower to avoid that footgun.
    rows.append(fa, w.settingRow(w.fmt("Steps: {d:.0}", .{st.vid_steps.get()}), zigui.Slider(st.vid_steps.binding(), 10, 50).frameWidth(160))) catch {};
    rows.append(fa, w.settingRow(w.fmt("CFG: {d:.1}", .{st.vid_cfg.get()}), zigui.Slider(st.vid_cfg.binding(), 1, 10).frameWidth(160))) catch {};
    // Frame count on the 8N+1 ladder, with a live duration readout. FPS has no
    // control — it's a model property (24 LTX / 16 Wan, set by
    // applyVideoFamilyDefaults on model change), used here for the seconds math.
    const frames = st.videoFrames();
    const secs = @as(f32, @floatFromInt(frames)) / @as(f32, @floatFromInt(st.vid_fps_n.get()));
    rows.append(fa, w.settingRow(
        w.fmt("{d} frames · ~{d:.1}s", .{ frames, secs }),
        zigui.Slider(st.vid_frames.binding(), 9, 193).frameWidth(160),
    )) catch {};

    // Optional start frame (image-to-video: Wan TI2V / LTX I2V).
    rows.append(fa, w.sectionHeader("Start frame (optional)")) catch {};
    if (st.vid_init_image) |im| {
        framePickRows(st, &rows, im, zigui.actionCtx(AppState, st, onChooseImage), zigui.actionCtx(AppState, st, onClearImage));
    } else {
        rows.append(fa, w.secondaryButton(.image, "Add start image", zigui.actionCtx(AppState, st, onChooseImage))) catch {};
    }

    rows.append(fa, zigui.HStack(.{
        zigui.Text("Advanced").font(.subheadline),
        zigui.Spacer(),
        zigui.Toggle("", st.vid_advanced.binding()),
    }).frameMaxWidth().onTap(zigui.actionCtx(AppState, st, onToggleAdvanced))) catch {};

    if (st.vid_advanced.get()) {
        rows.append(fa, w.settingRow("Seed", zigui.TextField("random", &st.vid_seed).frameWidth(120))) catch {};
        // STG (LTX spatio-temporal guidance; sd.cpp SLG). 0 = off.
        rows.append(fa, w.settingRow(w.fmt("STG: {d:.1}", .{st.vid_slg.get()}), zigui.Slider(st.vid_slg.binding(), 0, 4).frameWidth(160))) catch {};
        // Two-stage hires: needs the LTX spatial upscaler sidecar in the model
        // folder; generateVideo silently skips it when absent.
        rows.append(fa, w.settingRow("2× hires (LTX)", zigui.Toggle("", st.vid_hires.binding()))) catch {};
        rows.append(fa, w.sectionHeader("Negative prompt")) catch {};
        rows.append(fa, zigui.TextEditor(&st.vid_negative, &st.vid_neg_scroll, false)
            .softWrap()
            .frameHeight(56)
            .padding(8)
            .frameMaxWidth()) catch {};
        // Optional end frame (LTX first/last-frame-to-video).
        rows.append(fa, w.sectionHeader("End frame (LTX, optional)")) catch {};
        if (st.vid_end_image) |im| {
            framePickRows(st, &rows, im, zigui.actionCtx(AppState, st, onChooseEndImage), zigui.actionCtx(AppState, st, onClearEndImage));
        } else {
            rows.append(fa, w.secondaryButton(.image, "Add end image", zigui.actionCtx(AppState, st, onChooseEndImage))) catch {};
        }
        // Style LoRA (.safetensors/.gguf) applied to the diffusion model.
        if (st.vid_lora_path) |lp| {
            rows.append(fa, zigui.Text(w.fmt("LoRA: {s}", .{std.fs.path.basename(lp)}))
                .font(.caption).foreground(th.colors.secondary_label)) catch {};
            rows.append(fa, zigui.HStack(.{
                w.secondaryButton(.file, "Change", zigui.actionCtx(AppState, st, onChooseLora)),
                w.tintedButton(.close, "Remove", th.colors.destructive, zigui.actionCtx(AppState, st, onClearLora)),
            }).spacing(8)) catch {};
            rows.append(fa, w.settingRow(w.fmt("LoRA scale: {d:.2}", .{st.vid_lora_scale.get()}), zigui.Slider(st.vid_lora_scale.binding(), 0, 2).frameWidth(160))) catch {};
        } else {
            rows.append(fa, w.secondaryButton(.file, "Add LoRA", zigui.actionCtx(AppState, st, onChooseLora))) catch {};
        }
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
        rows.append(fa, zigui.HStack(.{
            zigui.Text(if (decoding)
                w.fmt("Decoding frames… tile {d}/{d}", .{ step, total })
            else
                w.fmt("Generating… step {d}/{d}", .{ step, total }))
                .font(.caption).foreground(th.colors.secondary_label),
            zigui.Spacer(),
            w.tintedButton(.close, "Cancel", th.colors.destructive, zigui.actionCtx(AppState, st, onCancel)),
        }).spacing(8).frameMaxWidth()) catch {};
    } else {
        rows.append(fa, w.primaryButtonWide(.sparkles, "Generate", zigui.actionCtx(AppState, st, onGenerate))) catch {};
    }

    return w.card(zigui.VStack(rows.items).spacing(10)).frameWidth(w.input_col);
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
            zigui.VStack(.{ leftPanel(st), zigui.Spacer() }).frameWidth(w.input_col).frameMaxHeight(),
            rightPanel(st),
        }).spacing(12).frameMaxWidth().frameMaxHeight(),
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
