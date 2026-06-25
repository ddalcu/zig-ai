//! Chat screen: a streaming transcript of bubbles plus an iMessage-style input
//! pill. Chat runs through the local HTTP server (the single chat engine), which
//! delivers already-separated content / reasoning / tool calls; `AppState.pumpChat`
//! drains them. Reasoning shows as a collapsible block, tool results as a
//! collapsed disclosure — no markup parsing happens here.

const std = @import("std");
const zigui = @import("zigui");
const app = @import("zigui_app");
const w = @import("widgets.zig");
const markdown = @import("markdown.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;
const ChatMessage = st_mod.ChatMessage;

const bubble_max: f32 = 560;

const CopyCtx = struct { st: *AppState, text: []const u8 };

fn copyCtx(st: *AppState, text: []const u8) *CopyCtx {
    const cx = st.frame_arena.allocator().create(CopyCtx) catch unreachable;
    cx.* = .{ .st = st, .text = text };
    return cx;
}

fn onCopy(p: ?*anyopaque) void {
    const cx: *CopyCtx = @ptrCast(@alignCast(p.?));
    app.setClipboardText(cx.st.gpa, cx.text);
    // Flag this button (by its text's address) for a transient "Copied"
    // confirmation; `busyCheck` keeps the loop awake so it reverts on its own.
    cx.st.copied_key = @intFromPtr(cx.text.ptr);
    cx.st.copied_until_ms = app.c.SDL_GetTicks() + 1500;
}

/// A subtle "Copy" caption button that puts `text` on the clipboard (the
/// transcript is static text, so this is how you copy from it). Shows a
/// momentary "Copied" check after a tap.
fn copyButton(st: *AppState, text: []const u8) zigui.View {
    const copied = st.copied_key == @intFromPtr(text.ptr) and
        app.c.SDL_GetTicks() < st.copied_until_ms;
    const icon: zigui.IconName = if (copied) .check else .copy;
    const caption = if (copied) "Copied" else "Copy";
    return zigui.HStack(.{
        zigui.Icon(icon, 12, w.t().colors.tertiary_label),
        zigui.Text(caption).font(.caption2).foreground(w.t().colors.tertiary_label),
    }).spacing(4).onTap(.{ .ctx = copyCtx(st, text), .func = onCopy });
}

const ThinkCtx = struct { msg: *ChatMessage };

fn onToggleThink(p: ?*anyopaque) void {
    const cx: *ThinkCtx = @ptrCast(@alignCast(p.?));
    const m = cx.msg;
    const thinking = m.streaming and m.content.items.len == 0;
    m.think_expanded = !(m.think_expanded orelse thinking);
}

/// A collapsible reasoning disclosure: a muted "Thinking…/Reasoning" header that
/// toggles a faint, indented block of the model's chain-of-thought. `reasoning`
/// is supplied by the server (`reasoning_content`), already separated from the
/// answer.
fn reasoningBlock(st: *AppState, msg: *ChatMessage, reasoning: []const u8, thinking: bool) zigui.View {
    const th = w.t();
    const expanded = msg.think_expanded orelse thinking; // live while thinking
    const label = if (thinking) "Thinking…" else "Reasoning";
    const chevron: zigui.IconName = if (expanded) .chevron_down else .chevron_right;

    const tc = st.frame_arena.allocator().create(ThinkCtx) catch unreachable;
    tc.* = .{ .msg = msg };
    const head = zigui.HStack(.{
        zigui.Icon(.sparkles, 12, th.colors.tertiary_label),
        zigui.Text(label).font(.caption).foreground(th.colors.secondary_label),
        zigui.Icon(chevron, 11, th.colors.tertiary_label),
        zigui.Spacer(),
    }).spacing(5).onTap(.{ .ctx = tc, .func = onToggleThink });

    if (!expanded) return head;

    const body = maxWidth(zigui.WrappedText(reasoning)
        .font(.caption)
        .foreground(th.colors.secondary_label)
        .paddingInsets(.{ .top = 8, .leading = 12, .bottom = 8, .trailing = 12 })
        .background(th.colors.secondary_label.withAlpha(0.07))
        .cornerRadius(10), bubble_max);

    // Pin the body to the leading edge (like the answer bubble); without this the
    // fixed-width box is centered by the VStack's default cross-axis alignment.
    return zigui.VStack(.{ head, leading(body) }).spacing(5).frameMaxWidth();
}

fn onToggleTool(p: ?*anyopaque) void {
    const cx: *ThinkCtx = @ptrCast(@alignCast(p.?));
    cx.msg.tool_expanded = !cx.msg.tool_expanded;
}

/// A `tool` message rendered as a collapsed, reduced-opacity disclosure (a muted
/// "Tool result" header with a chevron) that expands to the raw output. Mirrors
/// mlx-serve's ChatView ToolResultBlockView.
fn toolResultBlock(st: *AppState, msg: *ChatMessage) zigui.View {
    const th = w.t();
    const chevron: zigui.IconName = if (msg.tool_expanded) .chevron_down else .chevron_right;
    const tc = st.frame_arena.allocator().create(ThinkCtx) catch unreachable;
    tc.* = .{ .msg = msg };
    const head = zigui.HStack(.{
        zigui.Icon(.cpu, 12, th.colors.tertiary_label),
        zigui.Text("Tool result").font(.caption).foreground(th.colors.tertiary_label),
        zigui.Icon(chevron, 11, th.colors.tertiary_label),
        zigui.Spacer(),
    }).spacing(5).onTap(.{ .ctx = tc, .func = onToggleTool });
    if (!msg.tool_expanded) return leading(head);
    const body = maxWidth(zigui.WrappedText(msg.content.items)
        .font(.caption2)
        .foreground(th.colors.secondary_label)
        .paddingInsets(.{ .top = 8, .leading = 12, .bottom = 8, .trailing = 12 })
        .background(th.colors.secondary_label.withAlpha(0.06))
        .cornerRadius(8), bubble_max);
    return zigui.VStack(.{ head, leading(body) }).spacing(5).frameMaxWidth();
}

/// A muted "Called <tool>" line under an assistant turn that invoked a tool.
fn toolCallIndicator(name: []const u8) zigui.View {
    const th = w.t();
    return zigui.HStack(.{
        zigui.Icon(.cpu, 11, th.colors.tertiary_label),
        zigui.Text(w.fmt("Called {s}", .{name})).font(.caption2).foreground(th.colors.tertiary_label),
        zigui.Spacer(),
    }).spacing(5).frameMaxWidth();
}

fn onSend(st: *AppState) void {
    st.sendChat();
}

fn onToggleAgent(st: *AppState) void {
    st.agent_mode.set(!st.agent_mode.get());
}

fn onToggleAgentTools(st: *AppState) void {
    st.agent_tools_open.set(!st.agent_tools_open.get());
}

/// A split toggle pill: the left half flips agent/MCP mode; the chevron opens a
/// popover of the agent's capabilities (built-in tools + each running MCP
/// server's tools), accent-filled when on.
fn agentPill(st: *AppState) zigui.View {
    const th = w.t();
    const on = st.agent_mode.get();
    const tools = st.mcp_mgr.toolCount();
    const fg = if (on) th.colors.on_accent else th.colors.secondary_label;
    const label = if (on) w.fmt("Agent + MCP · {d} tools", .{tools}) else "Agent + MCP";

    const main = zigui.HStack(.{
        zigui.Icon(.zap, 13, fg),
        zigui.Text(label).font(.caption).foreground(fg),
    }).spacing(5)
        .paddingInsets(.{ .top = 6, .leading = 10, .bottom = 6, .trailing = 8 })
        .onTap(zigui.actionCtx(AppState, st, onToggleAgent));

    // A hairline split, then the dropdown chevron — opens the capabilities popover.
    const split = zigui.Rectangle(fg.withAlpha(0.3)).frame(1, 14);
    const chev = zigui.Icon(.chevron_down, 13, fg)
        .paddingInsets(.{ .top = 6, .leading = 7, .bottom = 6, .trailing = 10 })
        .onTap(zigui.actionCtx(AppState, st, onToggleAgentTools))
        .popover(st.agent_tools_open.binding(), agentToolsPopover(st));

    var pill = zigui.HStack(.{ main, split, chev }).spacing(0).cornerRadius(8);
    if (on)
        pill = pill.background(th.colors.accent)
    else
        pill = pill.background(th.colors.control_background).border(th.colors.separator, th.metrics.hairline);
    return pill;
}

/// A section label inside the capabilities popover.
fn toolGroupHeader(title: []const u8) zigui.View {
    return zigui.Text(title).font(.caption).foreground(w.t().colors.secondary_label)
        .frameMaxWidth().frameAlign(.leading);
}

/// One tool row: name over a truncated one-line description.
fn toolRow(name: []const u8, desc: []const u8) zigui.View {
    const th = w.t();
    return zigui.VStack(.{
        zigui.Text(name).font(.subheadline).truncated(),
        zigui.Text(desc).font(.caption2).foreground(th.colors.tertiary_label).truncated(),
    }).spacing(0).alignment(zigui.Alignment.leading).frameMaxWidth().frameAlign(.leading)
        .paddingInsets(.{ .top = 1, .leading = 4, .bottom = 1, .trailing = 4 });
}

/// The capabilities popover: built-in "agent tools" plus each running MCP
/// server's tools (qualified as `server__tool`), grouped by server.
fn agentToolsPopover(st: *AppState) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();
    const tools = st.mcp_mgr.toolListAlloc(fa);

    var rows: std.ArrayList(zigui.View) = .empty;
    rows.append(fa, toolGroupHeader("Agent tools")) catch {};
    for (tools) |t| {
        if (std.mem.indexOf(u8, t.qualified, "__") != null) continue; // MCP, handled below
        rows.append(fa, toolRow(t.qualified, t.description)) catch {};
    }

    var cur: []const u8 = "";
    var any_mcp = false;
    for (tools) |t| {
        const sep = std.mem.indexOf(u8, t.qualified, "__") orelse continue;
        const server = t.qualified[0..sep];
        const name = t.qualified[sep + 2 ..];
        if (!std.mem.eql(u8, server, cur)) {
            rows.append(fa, zigui.Divider()) catch {};
            rows.append(fa, toolGroupHeader(server)) catch {};
            cur = server;
            any_mcp = true;
        }
        rows.append(fa, toolRow(name, t.description)) catch {};
    }
    if (!any_mcp) {
        rows.append(fa, zigui.Divider()) catch {};
        rows.append(fa, zigui.Text("No MCP servers running — add one in the MCP tab.")
            .font(.caption2).foreground(th.colors.tertiary_label).frameMaxWidth()) catch {};
    }

    // Hug the content; scroll once the list would get tall.
    const list = zigui.VStack(rows.items).spacing(3).frameMaxWidth();
    const body = if (rows.items.len > 16)
        zigui.ScrollViewState(&st.agent_tools_scroll, list).frameMaxWidth().frameHeight(360)
    else
        list;
    return body.padding(8).frameWidth(320);
}

fn onStop(st: *AppState) void {
    st.chat.cancel();
}

/// Cap a view's width so bubbles don't span the whole pane.
fn maxWidth(v: zigui.View, width: f32) zigui.View {
    var out = v;
    var f = out.mods.frame orelse zigui.view.FrameSpec{};
    f.max_width = width;
    out.mods.frame = f;
    return out;
}

fn leading(v: zigui.View) zigui.View {
    return zigui.HStack(.{ v, zigui.Spacer() }).frameMaxWidth();
}

fn bubbleView(st: *AppState, msg: *ChatMessage) zigui.View {
    const th = w.t();
    const fa = st.frame_arena.allocator();

    // Tool-result messages render as a collapsed disclosure, not a chat bubble.
    if (msg.role == .tool) return toolResultBlock(st, msg);

    const is_user = msg.role == .user;

    // User messages have no reasoning; render the plain bubble path.
    if (is_user) {
        const shown = if (msg.streaming and msg.content.items.len == 0)
            "…"
        else if (msg.streaming)
            w.fmt("{s}…", .{msg.content.items})
        else
            msg.content.items;
        const bubble = maxWidth(zigui.WrappedText(shown)
            .foreground(th.colors.on_accent)
            .paddingInsets(.{ .top = 8, .leading = 12, .bottom = 8, .trailing = 12 })
            .background(th.colors.accent)
            .cornerRadius(14), bubble_max);
        const row = zigui.HStack(.{ zigui.Spacer(), bubble }).frameMaxWidth();
        if (msg.streaming or msg.content.items.len == 0) return row;
        const footer = zigui.HStack(.{ zigui.Spacer(), copyButton(st, msg.content.items) }).frameMaxWidth();
        return zigui.VStack(.{ row, footer }).spacing(3).frameMaxWidth();
    }

    // Assistant. The server already separated reasoning / answer / tool call.
    const reasoning = std.mem.trim(u8, msg.reasoning.items, " \t\r\n");
    const answer = msg.content.items;
    const thinking = msg.streaming and answer.len == 0;

    var parts: std.ArrayList(zigui.View) = .empty;
    if (reasoning.len > 0) parts.append(fa, reasoningBlock(st, msg, reasoning, thinking)) catch {};

    // The answer bubble — markdown-rendered, shown once there's text (no trailing
    // "…": the bouncing dots below indicate ongoing generation).
    if (answer.len > 0) {
        const bubble = maxWidth(markdown.view(fa, answer)
            .paddingInsets(.{ .top = 8, .leading = 12, .bottom = 8, .trailing = 12 })
            .background(th.colors.control_background)
            .cornerRadius(14), bubble_max);
        parts.append(fa, zigui.HStack(.{ bubble, zigui.Spacer() }).frameMaxWidth()) catch {};
    }

    // Animated "thinking…" dots anchored below the streaming reply (and shown on
    // their own while we wait for the first token).
    if (msg.streaming) {
        const dots = zigui.LoadingDots(app.c.SDL_GetTicks(), th.colors.secondary_label)
            .paddingInsets(.{ .top = 4, .leading = 14, .bottom = 2 });
        parts.append(fa, leading(dots)) catch {};
    }

    if (msg.tool_call) |tc| parts.append(fa, toolCallIndicator(tc.name)) catch {};

    // Footer (Copy + token stats) once the message has settled with answer text.
    if (!msg.streaming and answer.len > 0) {
        const copy = copyButton(st, answer);
        const footer = if (msg.tokens > 0)
            zigui.HStack(.{
                copy,
                zigui.Text(w.fmt("~{d} tokens · {d:.0} tok/s", .{ msg.tokens, msg.tps }))
                    .font(.caption2)
                    .foreground(th.colors.tertiary_label),
                zigui.Spacer(),
            }).spacing(10).frameMaxWidth()
        else
            leading(copy);
        parts.append(fa, footer) catch {};
    }

    return zigui.VStack(parts.items).spacing(5).frameMaxWidth();
}

fn sendButton(st: *AppState) zigui.View {
    const th = w.t();
    const streaming = st.pending != null;
    const cb = if (streaming)
        zigui.actionCtx(AppState, st, onStop)
    else
        zigui.actionCtx(AppState, st, onSend);
    const fill = if (streaming) th.colors.destructive else th.colors.accent;
    // A proper stop square while generating; an up-arrow to send otherwise.
    const glyph = if (streaming)
        zigui.RoundedRectangle(th.colors.on_accent, 2).frame(12, 12)
    else
        zigui.Icon(.arrow_up, 19, th.colors.on_accent);
    return zigui.ZStack(.{
        zigui.Circle(fill).frame(34, 34),
        glyph,
    }).frame(34, 34).onTap(cb);
}

pub fn view(st: *AppState) zigui.View {
    const th = w.t();
    const model = st.selectedModel(st.sel_llm.get());

    const header = zigui.HStack(.{
        zigui.Text("Chat").font(.title),
        zigui.Spacer(),
        agentPill(st),
        w.modelPicker(st, .text),
        w.statusDot(if (st.llm_loaded) w.green() else th.colors.tertiary_label),
        w.secondaryButton(.edit, "New", zigui.actionCtx(AppState, st, AppState.newChat)),
    }).spacing(10).frameMaxWidth();

    // A thin status strip while a tool call is in flight.
    const tool_strip: ?zigui.View = if (st.agent_busy) zigui.HStack(.{
        zigui.Icon(.cpu, 13, th.colors.accent),
        zigui.Text(w.fmt("Running {s}…", .{st.agentToolName()}))
            .font(.caption).foreground(th.colors.secondary_label),
        zigui.Spacer(),
    }).spacing(6).frameMaxWidth() else null;

    var transcript: zigui.View = undefined;
    if (st.messages.items.len == 0) {
        transcript = w.emptyState(
            .message_circle,
            "Start a conversation",
            if (model == null) "Pick a chat model in the Models tab first." else "Type a message below to begin.",
        );
    } else {
        const fa = st.frame_arena.allocator();
        var rows: std.ArrayList(zigui.View) = .empty;
        for (st.messages.items) |msg| rows.append(fa, bubbleView(st, msg)) catch {};
        const bubbles = zigui.VStack(rows.items)
            .spacing(12)
            .padding(8)
            .frameMaxWidth();
        transcript = zigui.ScrollViewState(&st.chat_scroll, bubbles).frameMaxWidth().frameMaxHeight();
    }

    const input_bar = zigui.HStack(.{
        // The TextField paints its own surface (background + border) via the
        // theme painter, so we just give it the pill radius + a fixed height —
        // adding `.background`/`.border` (or outer padding) double the chrome.
        zigui.TextField("Message…", &st.chat_input)
            .onSubmit(zigui.actionCtx(AppState, st, onSend))
            .cornerRadius(18)
            .frameHeight(38)
            .frameMaxWidth(),
        sendButton(st),
    }).spacing(8).frameMaxWidth();

    const fa = st.frame_arena.allocator();
    var col: std.ArrayList(zigui.View) = .empty;
    col.append(fa, header) catch {};
    if (tool_strip) |ts| col.append(fa, ts) catch {};
    col.append(fa, transcript) catch {};
    col.append(fa, input_bar) catch {};
    return zigui.VStack(col.items).spacing(12).frameMaxWidth().frameMaxHeight();
}
