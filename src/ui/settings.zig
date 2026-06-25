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

/// An f32 slider view over the i64 `chat_n_ctx` state, snapping to 1024-token
/// steps. (Slider needs an f32 binding; the setting itself stays an integer.)
fn ctxBinding(st: *AppState) zigui.Binding(f32) {
    const G = struct {
        fn get(p: *anyopaque) f32 {
            const s: *AppState = @ptrCast(@alignCast(p));
            return @floatFromInt(s.chat_n_ctx.get());
        }
        fn set(p: *anyopaque, v: f32) void {
            const s: *AppState = @ptrCast(@alignCast(p));
            const snapped: i64 = @as(i64, @intFromFloat(@round(v / 1024.0))) * 1024;
            s.chat_n_ctx.set(@max(snapped, 1024));
        }
    };
    return .{ .ctx = st, .getFn = G.get, .setFn = G.set };
}

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

    const ctx_cap = st.chatCtxCap();
    const ctx_cur = @min(@as(u32, @intCast(@max(st.chat_n_ctx.get(), 0))), ctx_cap);
    const generation = w.card(zigui.VStack(.{
        w.sectionHeader("Chat generation"),
        w.settingRow(w.fmt("Temperature: {d:.2}", .{st.chat_temp.get()}), zigui.Slider(st.chat_temp.binding(), 0, 2).frameWidth(180)),
        w.settingRow(w.fmt("Top-P: {d:.2}", .{st.chat_top_p.get()}), zigui.Slider(st.chat_top_p.binding(), 0, 1).frameWidth(180)),
        w.settingRow("Top-K", zigui.Stepper(w.fmt("{d}", .{st.chat_top_k.get()}), st.chat_top_k.binding(), 0, 200, 5)),
        w.settingRow(
            w.fmt("Context: {d} / {d} tokens", .{ ctx_cur, ctx_cap }),
            zigui.Slider(ctxBinding(st), 2048, @floatFromInt(@max(ctx_cap, 4096))).frameWidth(180),
        ),
        zigui.Text("Applies to new messages, capped at the selected model's trained context. Larger uses more memory.")
            .font(.caption).foreground(th.colors.tertiary_label).frameMaxWidth(),
    }).spacing(10));

    // Built-in zigui theme families (macOS, Windows 10, …), in registry order.
    const family_names = comptime blk: {
        var names: [zigui.theme_registry.all.len][]const u8 = undefined;
        for (zigui.theme_registry.all, 0..) |fam, i| names[i] = fam.displayName();
        break :blk names;
    };
    const appearance = w.card(zigui.VStack(.{
        w.sectionHeader("Appearance"),
        w.settingRow("Mode", zigui.Picker(st.theme_pref.binding(), &[_][]const u8{ "System", "Light", "Dark" })),
        w.settingRow("Theme", zigui.Picker(st.theme_family.binding(), &family_names)),
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

    // Header stays fixed; the cards scroll (the page outgrew the viewport).
    return zigui.VStack(.{
        w.header("Settings", zigui.Spacer()),
        zigui.ScrollViewState(&st.settings_scroll, zigui.VStack(.{
            folders,
            perf,
            generation,
            appearance,
            agent_card,
        }).spacing(14).frameMaxWidth()).frameMaxWidth().frameMaxHeight(),
    }).spacing(14).frameMaxWidth().frameMaxHeight();
}
