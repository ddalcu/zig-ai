//! Audio / TTS screen. Phase 1 renders the form; synthesis against
//! qwen3-tts.cpp and SDL audio playback are wired in Phase 4.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onSynthesize(st: *AppState) void {
    st.synthesize();
}

fn onChooseWav(st: *AppState) void {
    st.chooseRefWav();
}

fn onToggleRecord(st: *AppState) void {
    st.toggleRecord();
}

fn onClearRef(st: *AppState) void {
    st.clearRef();
}

fn onPreviewRec(st: *AppState) void {
    st.previewRec();
}

/// The voice-clone block: pick a reference WAV or record one, then show which
/// reference is active with preview/clear controls.
fn cloneSection(st: *AppState) zigui.View {
    const th = w.t();

    const record_btn = if (st.tts_recording)
        w.tintedButton(.square, "Stop", th.colors.destructive, zigui.actionCtx(AppState, st, onToggleRecord))
    else
        w.secondaryButton(.mic, "Record", zigui.actionCtx(AppState, st, onToggleRecord));

    const status: zigui.View = if (st.tts_recording)
        zigui.Text(w.fmt("Recording… {d:.1} s", .{
            @as(f32, @floatFromInt(st.tts_rec.items.len)) / 24000.0,
        })).font(.callout).foreground(th.colors.destructive).frameMaxWidth()
    else if (st.tts_ref_path) |p|
        zigui.HStack(.{
            zigui.Icon(.file, 14, th.colors.secondary_label),
            zigui.WrappedText(std.fs.path.basename(p)).font(.callout)
                .foreground(th.colors.secondary_label).frameMaxWidth(),
            zigui.IconButton(.close, 14, zigui.actionCtx(AppState, st, onClearRef)),
        }).spacing(6).frameMaxWidth()
    else if (st.tts_rec.items.len > 0)
        zigui.HStack(.{
            zigui.Icon(.audio_lines, 14, th.colors.secondary_label),
            zigui.Text(w.fmt("Recorded {d:.1} s", .{
                @as(f32, @floatFromInt(st.tts_rec.items.len)) / 24000.0,
            })).font(.callout).foreground(th.colors.secondary_label).frameMaxWidth(),
            zigui.IconButton(.play, 14, zigui.actionCtx(AppState, st, onPreviewRec)),
            zigui.IconButton(.close, 14, zigui.actionCtx(AppState, st, onClearRef)),
        }).spacing(6).frameMaxWidth()
    else
        zigui.Text("No reference — default voice.").font(.callout)
            .foreground(th.colors.tertiary_label).frameMaxWidth();

    return zigui.VStack(.{
        w.sectionHeader("Voice clone (optional)"),
        zigui.HStack(.{
            w.secondaryButton(.folder, "WAV file…", zigui.actionCtx(AppState, st, onChooseWav)),
            record_btn,
        }).spacing(8).frameMaxWidth(),
        status,
    }).spacing(8).frameMaxWidth();
}

fn leftPanel(st: *AppState) zigui.View {
    const th = w.t();

    return w.card(zigui.VStack(.{
        w.sectionHeader("Text"),
        zigui.TextEditor(&st.tts_text, &st.tts_scroll, false)
            .softWrap()
            .frameHeight(140)
            .padding(8)
            .background(th.colors.control_background)
            .cornerRadius(6)
            .border(th.colors.separator, th.metrics.hairline)
            .frameMaxWidth(),
        w.settingRow(w.fmt("Temperature: {d:.2}", .{st.tts_temperature.get()}), zigui.Slider(st.tts_temperature.binding(), 0, 1.5).frameWidth(160)),
        cloneSection(st),
        if (st.tts.isBusy())
            zigui.Text("Synthesizing…").font(.callout).foreground(th.colors.secondary_label).frameMaxWidth()
        else
            w.primaryButtonWide(.play, "Synthesize & Play", zigui.actionCtx(AppState, st, onSynthesize)),
    }).spacing(10)).frameWidth(360);
}

fn rightPanel(st: *AppState) zigui.View {
    if (st.tts_last_samples > 0) {
        return w.card(zigui.VStack(.{
            zigui.Spacer(),
            zigui.Text(w.fmt("{d} samples @ 24 kHz", .{st.tts_last_samples}))
                .font(.title3).foreground(w.t().colors.secondary_label).frameMaxWidth(),
            zigui.Spacer(),
        }).frameMaxWidth().frameMaxHeight()).frameMaxWidth().frameMaxHeight();
    }
    return w.card(w.emptyState(.audio_lines, "No audio yet", "Enter text and press Synthesize."))
        .frameMaxWidth()
        .frameMaxHeight();
}

pub fn view(st: *AppState) zigui.View {
    return zigui.VStack(.{
        w.header("Audio / TTS", w.modelPicker(st, .tts)),
        zigui.HStack(.{
            // Pin the form card to the top (trailing Spacer fills the rest),
            // matching the Image/Video screens — otherwise the HStack centers it.
            zigui.VStack(.{ leftPanel(st), zigui.Spacer() }).frameWidth(360).frameMaxHeight(),
            rightPanel(st),
        }).spacing(12).frameMaxWidth().frameMaxHeight(),
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
