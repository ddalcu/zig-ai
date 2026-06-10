//! Image generation screen. Phase 1 renders the two-column form + preview;
//! generation against stable-diffusion.cpp is wired in Phase 3.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onGenerate(st: *AppState) void {
    st.generateImage();
}

fn onToggleAdvanced(st: *AppState) void {
    st.img_advanced.set(!st.img_advanced.get());
}

fn leftPanel(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    var rows: std.ArrayList(zigui.View) = .empty;
    rows.append(fa, w.sectionHeader("Prompt")) catch {};
    rows.append(fa, zigui.TextEditor(&st.img_prompt, &st.img_scroll, false)
        .frameHeight(90)
        .padding(8)
        .background(th.colors.control_background)
        .cornerRadius(6)
        .border(th.colors.separator, th.metrics.hairline)
        .frameMaxWidth()) catch {};

    rows.append(fa, w.settingRow(w.fmt("Steps: {d:.0}", .{st.img_steps.get()}), zigui.Slider(st.img_steps.binding(), 1, 50).frameWidth(160))) catch {};
    rows.append(fa, w.settingRow(w.fmt("Guidance: {d:.1}", .{st.img_cfg.get()}), zigui.Slider(st.img_cfg.binding(), 1, 15).frameWidth(160))) catch {};

    rows.append(fa, zigui.HStack(.{
        zigui.Text("Advanced").font(.subheadline),
        zigui.Spacer(),
        zigui.Toggle("", st.img_advanced.binding()),
    }).frameMaxWidth().onTap(zigui.actionCtx(AppState, st, onToggleAdvanced))) catch {};

    if (st.img_advanced.get()) {
        rows.append(fa, w.settingRow("Width", zigui.Stepper(w.fmt("{d}", .{st.img_width.get()}), st.img_width.binding(), 256, 1024, 64))) catch {};
        rows.append(fa, w.settingRow("Height", zigui.Stepper(w.fmt("{d}", .{st.img_height.get()}), st.img_height.binding(), 256, 1024, 64))) catch {};
    }

    const busy = st.sd.isBusy();
    if (busy) {
        const frac = st.sd.job.fraction();
        const step = st.sd.job.step.load(.acquire);
        const total = st.sd.job.total.load(.acquire);
        rows.append(fa, zigui.ProgressView(frac).frameMaxWidth()) catch {};
        rows.append(fa, zigui.Text(w.fmt("Generating… step {d}/{d}", .{ step, total }))
            .font(.caption).foreground(th.colors.secondary_label)) catch {};
    } else {
        rows.append(fa, w.primaryButtonWide(.sparkles, "Generate", zigui.actionCtx(AppState, st, onGenerate))) catch {};
    }

    return w.card(zigui.VStack(rows.items).spacing(10)).frameWidth(340);
}

fn rightPanel(st: *AppState) zigui.View {
    if (st.img_result) |img| {
        return w.card(zigui.Image(img).frameMaxWidth().frameMaxHeight()).frameMaxWidth().frameMaxHeight();
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
