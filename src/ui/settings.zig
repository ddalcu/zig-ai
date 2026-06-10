//! Settings: model scan folders, inference threads, GPU toggle, and appearance.
//! The theme picker (System / Light / Dark) applies live — the run loop re-reads
//! the theme each frame via the app's theme provider (see main's
//! setThemeProvider). "System" follows the OS light/dark setting.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const models = @import("../models.zig");
const AppState = st_mod.AppState;

fn onAddDir(st: *AppState) void {
    const text = st.new_dir.text();
    if (text.len == 0) return;
    const dup = st.gpa.dupe(u8, text) catch return;
    st.model_dirs.append(st.gpa, dup) catch {
        st.gpa.free(dup);
        return;
    };
    st.new_dir.setText("") catch {};
    st.rescanModels();
}

const DirCtx = struct { st: *AppState, index: usize };

fn onRemoveDir(p: ?*anyopaque) void {
    const cx: *DirCtx = @ptrCast(@alignCast(p.?));
    if (cx.index >= cx.st.model_dirs.items.len) return;
    const removed = cx.st.model_dirs.orderedRemove(cx.index);
    cx.st.gpa.free(removed);
    cx.st.rescanModels();
}

fn onEditSystemPrompt(st: *AppState) void {
    st.openEditor(.system_prompt);
}

fn onOpenMcp(st: *AppState) void {
    st.screen.set(@intFromEnum(st_mod.Screen.mcp));
}

fn dirRow(st: *AppState, index: usize, dir: []const u8) zigui.View {
    const cx = st.frame_arena.allocator().create(DirCtx) catch unreachable;
    cx.* = .{ .st = st, .index = index };
    return zigui.HStack(.{
        zigui.Text(dir).font(.caption).foreground(w.t().colors.secondary_label),
        zigui.Spacer(),
        zigui.Text("×").foreground(w.t().colors.tertiary_label)
            .onTap(.{ .ctx = cx, .func = onRemoveDir }),
    }).frameMaxWidth();
}

pub fn view(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    // Default (built-in) directories shown for context.
    var dir_rows: std.ArrayList(zigui.View) = .empty;
    dir_rows.append(fa, w.sectionHeader("Default folders")) catch {};
    for (models.default_dirs) |dd| {
        dir_rows.append(fa, zigui.Text(w.fmt("~/{s}  ·  {s}", .{ dd.sub, dd.source.label() }))
            .font(.caption).foreground(th.colors.tertiary_label).frameMaxWidth()) catch {};
    }
    if (st.model_dirs.items.len > 0) {
        dir_rows.append(fa, zigui.Divider()) catch {};
        dir_rows.append(fa, w.sectionHeader("Added folders")) catch {};
        for (st.model_dirs.items, 0..) |d, i| dir_rows.append(fa, dirRow(st, i, d)) catch {};
    }
    dir_rows.append(fa, zigui.Divider()) catch {};
    dir_rows.append(fa, zigui.HStack(.{
        zigui.TextField("Add a model folder…", &st.new_dir)
            .onSubmit(zigui.actionCtx(AppState, st, onAddDir))
            .frameMaxWidth(),
        zigui.components.ButtonRoled("Add", .normal, zigui.actionCtx(AppState, st, onAddDir)),
    }).spacing(8).frameMaxWidth()) catch {};

    const folders = w.card(zigui.VStack(dir_rows.items).spacing(8));

    const perf = w.card(zigui.VStack(.{
        w.sectionHeader("Performance"),
        w.settingRow("Threads", zigui.Stepper(
            w.fmt("{d}", .{st.threads.get()}),
            st.threads.binding(),
            1,
            32,
            1,
        )),
        zigui.Divider(),
        w.settingRow("Use GPU (Metal)", zigui.Toggle("", st.use_gpu.binding())),
    }).spacing(10));

    const appearance = w.card(zigui.VStack(.{
        w.sectionHeader("Appearance"),
        w.settingRow("Theme", zigui.Picker(st.theme_pref.binding(), &[_][]const u8{ "System", "Light", "Dark" })),
    }).spacing(10));

    const agent_card = w.card(zigui.VStack(.{
        w.sectionHeader("Agent"),
        w.settingRow("Agent mode (let chat use MCP tools)", zigui.Toggle("", st.agent_mode.binding())),
        zigui.Text("In agent mode the model can call tools from your MCP servers and act over several steps.")
            .font(.caption).foreground(th.colors.tertiary_label).frameMaxWidth(),
        zigui.Divider(),
        zigui.HStack(.{
            w.secondaryButton(.edit, "Edit System Prompt", zigui.actionCtx(AppState, st, onEditSystemPrompt)),
            w.secondaryButton(.cpu, "Manage MCP Servers", zigui.actionCtx(AppState, st, onOpenMcp)),
            zigui.Spacer(),
        }).spacing(8).frameMaxWidth(),
    }).spacing(10));

    return zigui.VStack(.{
        w.header("Settings", zigui.Spacer()),
        folders,
        perf,
        appearance,
        agent_card,
        zigui.Spacer(),
    }).spacing(14).frameMaxWidth().frameMaxHeight();
}
