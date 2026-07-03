//! Image generation screen: two-column form + preview over the in-process
//! stable-diffusion.cpp backend (SD / SDXL / FLUX / Krea2). The selected model
//! is a single checkpoint or a diffusion .gguf whose VAE/text-encoder sidecars
//! are auto-discovered next to it (see state.generateImage).

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onGenerate(st: *AppState) void {
    st.generateImage();
}

fn onCancel(st: *AppState) void {
    st.sd.cancel();
}

fn onToggleAdvanced(st: *AppState) void {
    st.img_advanced.set(!st.img_advanced.get());
}

fn onChooseInit(st: *AppState) void {
    st.chooseImageInit();
}

fn onClearInit(st: *AppState) void {
    st.clearImageInit();
}

fn onChooseLora(st: *AppState) void {
    st.chooseImageLora();
}

fn onClearLora(st: *AppState) void {
    st.clearImageLora();
}

fn leftPanel(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    var rows: std.ArrayList(zigui.View) = .empty;
    rows.append(fa, w.sectionHeader("Prompt")) catch {};
    rows.append(fa, zigui.TextEditor(&st.img_prompt, &st.img_scroll, false)
        .softWrap()
        .frameHeight(90)
        .padding(8)
        .background(th.colors.control_background)
        .cornerRadius(6)
        .border(th.colors.separator, th.metrics.hairline)
        .frameMaxWidth()) catch {};

    rows.append(fa, w.settingRow(w.fmt("Steps: {d:.0}", .{st.img_steps.get()}), zigui.Slider(st.img_steps.binding(), 1, 50).frameWidth(160))) catch {};
    rows.append(fa, w.settingRow(w.fmt("Guidance: {d:.1}", .{st.img_cfg.get()}), zigui.Slider(st.img_cfg.binding(), 1, 15).frameWidth(160))) catch {};

    // Optional image-to-image source (SDEdit variation at Strength).
    rows.append(fa, w.sectionHeader("Source image (optional)")) catch {};
    if (st.img_init_image) |im| {
        const thumb: zigui.canvas.Image = .{
            .width = im.width,
            .height = im.height,
            .pixels = im.pixels[0 .. @as(usize, im.width) * @as(usize, im.height) * 4],
        };
        rows.append(fa, zigui.Image(thumb).scaledToFit().frameMaxWidth().frameHeight(100)
            .cornerRadius(6)) catch {};
        rows.append(fa, zigui.HStack(.{
            w.secondaryButton(.image, "Change", zigui.actionCtx(AppState, st, onChooseInit)),
            w.tintedButton(.close, "Remove", th.colors.destructive, zigui.actionCtx(AppState, st, onClearInit)),
        }).spacing(8)) catch {};
        rows.append(fa, w.settingRow(w.fmt("Strength: {d:.2}", .{st.img_strength.get()}), zigui.Slider(st.img_strength.binding(), 0.05, 1).frameWidth(160))) catch {};
    } else {
        rows.append(fa, w.secondaryButton(.image, "Add source image", zigui.actionCtx(AppState, st, onChooseInit))) catch {};
    }

    rows.append(fa, zigui.HStack(.{
        zigui.Text("Advanced").font(.subheadline),
        zigui.Spacer(),
        zigui.Toggle("", st.img_advanced.binding()),
    }).frameMaxWidth().onTap(zigui.actionCtx(AppState, st, onToggleAdvanced))) catch {};

    if (st.img_advanced.get()) {
        rows.append(fa, w.settingRow("Width", zigui.Stepper(w.fmt("{d}", .{st.img_width.get()}), st.img_width.binding(), 256, 2048, 64))) catch {};
        rows.append(fa, w.settingRow("Height", zigui.Stepper(w.fmt("{d}", .{st.img_height.get()}), st.img_height.binding(), 256, 2048, 64))) catch {};
        rows.append(fa, w.settingRow("Seed", zigui.TextField("random", &st.img_seed).frameWidth(120))) catch {};
        rows.append(fa, w.sectionHeader("Negative prompt")) catch {};
        rows.append(fa, zigui.TextEditor(&st.img_negative, &st.img_neg_scroll, false)
            .softWrap()
            .frameHeight(56)
            .padding(8)
            .background(th.colors.control_background)
            .cornerRadius(6)
            .border(th.colors.separator, th.metrics.hairline)
            .frameMaxWidth()) catch {};
        // Style LoRA (.safetensors/.gguf) applied to the diffusion model.
        if (st.img_lora_path) |lp| {
            rows.append(fa, zigui.Text(w.fmt("LoRA: {s}", .{std.fs.path.basename(lp)}))
                .font(.caption).foreground(th.colors.secondary_label)) catch {};
            rows.append(fa, zigui.HStack(.{
                w.secondaryButton(.file, "Change", zigui.actionCtx(AppState, st, onChooseLora)),
                w.tintedButton(.close, "Remove", th.colors.destructive, zigui.actionCtx(AppState, st, onClearLora)),
            }).spacing(8)) catch {};
            rows.append(fa, w.settingRow(w.fmt("LoRA scale: {d:.2}", .{st.img_lora_scale.get()}), zigui.Slider(st.img_lora_scale.binding(), 0, 2).frameWidth(160))) catch {};
        } else {
            rows.append(fa, w.secondaryButton(.file, "Add LoRA", zigui.actionCtx(AppState, st, onChooseLora))) catch {};
        }
    }

    const busy = st.sd.isBusy();
    if (busy) {
        const frac = st.sd.job.fraction();
        const step = st.sd.job.step.load(.acquire);
        const total = st.sd.job.total.load(.acquire);
        rows.append(fa, zigui.ProgressView(frac).frameMaxWidth()) catch {};
        rows.append(fa, zigui.HStack(.{
            zigui.Text(w.fmt("Generating… step {d}/{d}", .{ step, total }))
                .font(.caption).foreground(th.colors.secondary_label),
            zigui.Spacer(),
            w.tintedButton(.close, "Cancel", th.colors.destructive, zigui.actionCtx(AppState, st, onCancel)),
        }).spacing(8).frameMaxWidth()) catch {};
    } else {
        rows.append(fa, w.primaryButtonWide(.sparkles, "Generate", zigui.actionCtx(AppState, st, onGenerate))) catch {};
    }

    return w.card(zigui.VStack(rows.items).spacing(10)).frameWidth(340);
}

fn onOpenFolder(st: *AppState) void {
    st.openOutputsFolder();
}

fn rightPanel(st: *AppState) zigui.View {
    if (st.img_result) |img| {
        // `scaledToFit` keeps the image inside the preview pane (aspect-fit,
        // centered) instead of rendering at native pixels and overflowing.
        return w.card(zigui.VStack(.{
            zigui.Image(img).scaledToFit().frameMaxWidth().frameMaxHeight(),
            zigui.HStack(.{
                zigui.Spacer(),
                w.secondaryButton(.folder, "Open folder", zigui.actionCtx(AppState, st, onOpenFolder)),
            }).frameMaxWidth(),
        }).spacing(10).frameMaxWidth().frameMaxHeight())
            .frameMaxWidth().frameMaxHeight();
    }
    return w.card(w.emptyState(.image, "No image yet", "Enter a prompt and press Generate."))
        .frameMaxWidth()
        .frameMaxHeight();
}

pub fn view(st: *AppState) zigui.View {
    return zigui.VStack(.{
        w.header("Image Generation", w.modelPicker(st, .image)),
        zigui.HStack(.{
            zigui.VStack(.{ leftPanel(st), zigui.Spacer() }).frameWidth(340).frameMaxHeight(),
            rightPanel(st),
        }).spacing(12).frameMaxWidth().frameMaxHeight(),
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
