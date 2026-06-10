//! The MCP servers screen: a marketplace of preset servers (add with one tap)
//! plus the list of configured servers with live status, an enable toggle and a
//! remove button. The escape hatch for power users is the "Edit mcp.json" button,
//! which opens the in-app text editor on the raw registry.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const mcp = @import("../mcp.zig");
const AppState = st_mod.AppState;

fn onEditJson(st: *AppState) void {
    st.openEditor(.mcp_json);
}

const AddCtx = struct { st: *AppState, idx: usize };
fn onAddPreset(p: ?*anyopaque) void {
    const cx: *AddCtx = @ptrCast(@alignCast(p.?));
    cx.st.addMcpPresetByIndex(cx.idx);
}
fn onEditServer(p: ?*anyopaque) void {
    const cx: *AddCtx = @ptrCast(@alignCast(p.?));
    cx.st.openMcpEdit(cx.idx);
}

fn onCancelCfg(st: *AppState) void {
    st.cancelMcpConfig();
}
fn onConfirmCfg(st: *AppState) void {
    st.confirmMcpConfig();
}

/// The inline form shown under a preset row while it is being configured: one
/// labelled text field per input, then Cancel / Add server.
fn configForm(st: *AppState, fa: std.mem.Allocator, preset: mcp.Preset) zigui.View {
    const th = w.t();
    var rows: std.ArrayList(zigui.View) = .empty;
    const n = @min(preset.inputs.len, st_mod.max_mcp_inputs);
    for (0..n) |j| {
        const inp = preset.inputs[j];
        const label = if (inp.secret) w.fmt("{s} (secret)", .{inp.label}) else inp.label;
        const placeholder = if (inp.hint.len > 0) inp.hint else inp.label;
        rows.append(fa, zigui.VStack(.{
            zigui.Text(label).font(.caption).foreground(th.colors.secondary_label).frameMaxWidth(),
            zigui.TextField(placeholder, &st.mcp_cfg_fields[j])
                .cornerRadius(8)
                .frameHeight(32)
                .frameMaxWidth(),
        }).spacing(3).frameMaxWidth()) catch {};
    }
    const confirm = if (st.mcp_cfg_editing)
        w.primaryButton(.check, "Save", zigui.actionCtx(AppState, st, onConfirmCfg))
    else
        w.primaryButton(.plus, "Add server", zigui.actionCtx(AppState, st, onConfirmCfg));
    rows.append(fa, zigui.HStack(.{
        zigui.Spacer(),
        w.tintedButton(.close, "Cancel", th.colors.secondary_label, zigui.actionCtx(AppState, st, onCancelCfg)),
        confirm,
    }).spacing(8).frameMaxWidth()) catch {};

    return zigui.VStack(rows.items).spacing(8)
        .padding(10)
        .background(th.colors.secondary_background)
        .cornerRadius(8)
        .frameMaxWidth();
}

const NameCtx = struct { st: *AppState, name: []const u8 };
fn onRemove(p: ?*anyopaque) void {
    const cx: *NameCtx = @ptrCast(@alignCast(p.?));
    cx.st.removeMcpServer(cx.name);
}

const ToggleCtx = struct { st: *AppState, name: []const u8, disabled: bool };
fn onToggle(p: ?*anyopaque) void {
    const cx: *ToggleCtx = @ptrCast(@alignCast(p.?));
    cx.st.toggleMcpServer(cx.name, cx.disabled);
}

/// Live status (color + label) for a configured server, matched by name from the
/// runtime snapshot. Disabled servers aren't spawned, so they have no row.
fn statusFor(rows: []const mcp.Manager.StatusInfo, name: []const u8) ?mcp.Manager.StatusInfo {
    for (rows) |r| if (std.mem.eql(u8, r.name, name)) return r;
    return null;
}

fn serverRow(st: *AppState, fa: std.mem.Allocator, name: []const u8, cmd_summary: []const u8, disabled: bool, status: ?mcp.Manager.StatusInfo) zigui.View {
    const th = w.t();

    // Preset-derived entries with inputs get an Edit button that reopens the
    // config form pre-filled (custom entries are edited via mcp.json).
    const edit_idx: ?usize = blk: {
        const i = mcp.presetIndexById(name) orelse break :blk null;
        break :blk if (mcp.catalog[i].needsConfig()) i else null;
    };

    // Status dot + label.
    var dot = th.colors.tertiary_label;
    var detail: []const u8 = if (disabled) "disabled" else "enabled";
    if (!disabled) {
        if (status) |s| {
            switch (s.state) {
                .running => {
                    dot = w.green();
                    detail = w.fmt("{d} tool(s)", .{s.tools});
                },
                .starting => {
                    dot = w.orange();
                    detail = "starting…";
                },
                .failed => {
                    dot = th.colors.destructive;
                    detail = if (s.msg) |m| m else "failed";
                },
            }
        } else {
            dot = w.orange();
            detail = "starting…";
        }
    }

    const tcx = fa.create(ToggleCtx) catch unreachable;
    tcx.* = .{ .st = st, .name = name, .disabled = !disabled };
    const rcx = fa.create(NameCtx) catch unreachable;
    rcx.* = .{ .st = st, .name = name };

    const toggle_label = if (disabled) "Enable" else "Disable";
    const toggle_icon: zigui.IconName = if (disabled) .play else .pause;

    const edit_btn = if (edit_idx) |i| blk: {
        const ecx = fa.create(AddCtx) catch unreachable;
        ecx.* = .{ .st = st, .idx = i };
        break :blk w.tintedButton(.edit, "Edit", th.colors.accent, .{ .ctx = ecx, .func = onEditServer });
    } else zigui.HStack(.{});

    const row = zigui.HStack(.{
        w.statusDot(dot),
        zigui.VStack(.{
            zigui.Text(name).font(.subheadline),
            zigui.Text(cmd_summary).font(.caption2).foreground(th.colors.tertiary_label),
        }).spacing(1).alignment(zigui.Alignment.leading),
        zigui.Spacer(),
        zigui.Text(detail).font(.caption).foreground(th.colors.secondary_label),
        edit_btn,
        w.tintedButton(toggle_icon, toggle_label, th.colors.accent, .{ .ctx = tcx, .func = onToggle }),
        w.tintedButton(.trash, "Remove", th.colors.destructive, .{ .ctx = rcx, .func = onRemove }),
    }).spacing(10).frameMaxWidth();

    // While this entry is being edited, its pre-filled form renders below.
    const editing_this = st.mcp_cfg_editing and edit_idx != null and
        st.mcp_cfg_idx == @as(i64, @intCast(edit_idx.?));
    if (editing_this) {
        return zigui.VStack(.{ row, configForm(st, fa, mcp.catalog[edit_idx.?]) })
            .spacing(10).frameMaxWidth();
    }
    return row;
}

fn presetRow(st: *AppState, fa: std.mem.Allocator, idx: usize, preset: mcp.Preset, already: bool) zigui.View {
    const th = w.t();
    const acx = fa.create(AddCtx) catch unreachable;
    acx.* = .{ .st = st, .idx = idx };

    var info: std.ArrayList(zigui.View) = .empty;
    info.append(fa, zigui.Text(preset.name).font(.subheadline)) catch {};
    info.append(fa, zigui.Text(preset.description).font(.caption).foreground(th.colors.secondary_label)) catch {};
    if (preset.note.len > 0)
        info.append(fa, zigui.Text(preset.note).font(.caption2).foreground(th.colors.tertiary_label)) catch {};

    // Editing an existing server renders its form under the SERVER row, not here.
    const configuring = !st.mcp_cfg_editing and st.mcp_cfg_idx == @as(i64, @intCast(idx));
    const action = if (already)
        zigui.HStack(.{ zigui.Icon(.check, 14, w.green()), zigui.Text("Added").font(.caption).foreground(th.colors.secondary_label) }).spacing(5)
    else if (configuring)
        zigui.Text("Configure ↓").font(.caption).foreground(th.colors.secondary_label)
    else
        w.tintedButton(.plus, "Add", th.colors.accent, .{ .ctx = acx, .func = onAddPreset });

    const row = zigui.HStack(.{
        zigui.VStack(info.items).spacing(2).alignment(zigui.Alignment.leading),
        zigui.Spacer(),
        action,
    }).spacing(10).frameMaxWidth();

    if (configuring) {
        return zigui.VStack(.{ row, configForm(st, fa, preset) }).spacing(10).frameMaxWidth();
    }
    return row;
}

pub fn view(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    const header = zigui.HStack(.{
        zigui.Text("MCP Servers").font(.title),
        zigui.Spacer(),
        w.secondaryButton(.edit, "Edit mcp.json", zigui.actionCtx(AppState, st, onEditJson)),
    }).frameMaxWidth();

    const intro = zigui.Text("Connect tools the agent can use. Add a preset below, or edit mcp.json to add your own. Secrets and placeholders are filled in the editor.")
        .font(.caption).foreground(th.colors.secondary_label).frameMaxWidth();

    // Configured servers (from mcp.json) overlaid with live runtime status.
    const status_rows = st.mcp_mgr.statusAlloc(fa);
    var cfg_rows: std.ArrayList(zigui.View) = .empty;
    cfg_rows.append(fa, w.sectionHeader("Your servers")) catch {};

    var present = std.StringHashMap(void).init(fa);
    var any_cfg = false;
    if (st.home) |home| {
        var reg = mcp.loadRegistry(st.gpa, home);
        defer reg.deinit();
        for (reg.servers.items, 0..) |s, i| {
            if (i > 0) cfg_rows.append(fa, zigui.Divider()) catch {};
            // Dupe strings into the frame arena so they outlive `reg.deinit()`.
            const name = fa.dupe(u8, s.name) catch continue;
            present.put(name, {}) catch {};
            const summary = summarize(fa, s);
            cfg_rows.append(fa, serverRow(st, fa, name, summary, s.disabled, statusFor(status_rows, name))) catch {};
            any_cfg = true;
        }
    }
    if (!any_cfg) {
        cfg_rows.append(fa, zigui.Text("No servers configured yet. Add one from the catalogue below.")
            .font(.caption).foreground(th.colors.tertiary_label).frameMaxWidth()) catch {};
    }
    const configured = w.card(zigui.VStack(cfg_rows.items).spacing(8));

    // Preset catalogue.
    var cat_rows: std.ArrayList(zigui.View) = .empty;
    cat_rows.append(fa, w.sectionHeader("Add a preset")) catch {};
    for (mcp.catalog, 0..) |preset, i| {
        if (i > 0) cat_rows.append(fa, zigui.Divider()) catch {};
        const already = present.contains(preset.id);
        cat_rows.append(fa, presetRow(st, fa, i, preset, already)) catch {};
    }
    const catalogue = w.card(zigui.VStack(cat_rows.items).spacing(8));

    const body = zigui.VStack(.{ configured, catalogue, zigui.Spacer() }).spacing(14).frameMaxWidth();

    return zigui.VStack(.{
        header,
        intro,
        zigui.ScrollViewState(&st.models_scroll, body).frameMaxWidth().frameMaxHeight(),
    }).spacing(12).frameMaxWidth().frameMaxHeight();
}

/// A short "command arg arg…" summary of a server entry for display.
fn summarize(fa: std.mem.Allocator, s: mcp.ServerConfig) []const u8 {
    if (s.url) |u| return fa.dupe(u8, u) catch "";
    var out: std.ArrayList(u8) = .empty;
    if (s.command) |c| out.appendSlice(fa, c) catch {};
    for (s.args) |a| {
        out.append(fa, ' ') catch {};
        out.appendSlice(fa, a) catch {};
    }
    if (out.items.len > 80) return out.items[0..80];
    return out.items;
}
