//! A simple in-app text editor for the two config files (`system-prompt.md` and
//! `mcp.json`), built on zigui's `TextEditor`. Keeps editing in-app so the user
//! never has to drop out to an OS editor. Reached from Settings and the MCP
//! screen via `AppState.openEditor`.

const std = @import("std");
const zigui = @import("zigui");
const app = @import("zigui_app");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;

fn onBack(st: *AppState) void {
    // Return to the screen that makes sense for the file being edited.
    const dest: st_mod.Screen = switch (st.editor_target) {
        .system_prompt => .settings,
        .mcp_json => .mcp,
    };
    st.screen.set(@intFromEnum(dest));
}

fn onSave(st: *AppState) void {
    st.saveEditor();
    st.editor_saved_until_ms = app.c.SDL_GetTicks() + 1500;
}

fn onReload(st: *AppState) void {
    st.openEditor(st.editor_target); // re-reads the file from disk
}

pub fn view(st: *AppState) zigui.View {
    const th = w.t();
    const saved = app.c.SDL_GetTicks() < st.editor_saved_until_ms;

    const save_label = if (saved) "Saved" else "Save";
    const save_icon: zigui.IconName = if (saved) .check else .hard_drive;

    const header = zigui.HStack(.{
        w.secondaryButton(.chevron_left, "Back", zigui.actionCtx(AppState, st, onBack)),
        zigui.Text(st.editor_target.title()).font(.title3),
        zigui.Spacer(),
        w.secondaryButton(.refresh, "Revert", zigui.actionCtx(AppState, st, onReload)),
        w.primaryButton(save_icon, save_label, zigui.actionCtx(AppState, st, onSave)),
    }).spacing(10).frameMaxWidth();

    const hint = switch (st.editor_target) {
        .system_prompt => "Edit the system prompt sent before every chat. Markdown is fine.",
        .mcp_json => "Edit the MCP server registry. Each server needs a \"command\" and \"args\"; fill any <PLACEHOLDER> or empty env value.",
    };

    const line_numbers = st.editor_target == .mcp_json;
    const editor = zigui.TextEditor(&st.editor_buf, &st.editor_scroll, line_numbers)
        .padding(10)
        .background(th.colors.control_background)
        .cornerRadius(th.metrics.corner_radius)
        .border(th.colors.separator, th.metrics.hairline)
        .frameMaxWidth()
        .frameMaxHeight();

    return zigui.VStack(.{
        header,
        zigui.Text(hint).font(.caption).foreground(th.colors.tertiary_label).frameMaxWidth(),
        editor,
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}
