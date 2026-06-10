//! Chat screen: a streaming transcript of bubbles plus an iMessage-style input
//! pill. Generation runs in-process on the llama worker; tokens stream in via
//! `AppState.pumpChat`.

const std = @import("std");
const zigui = @import("zigui");
const app = @import("zigui_app");
const w = @import("widgets.zig");
const st_mod = @import("../state.zig");
const AppState = st_mod.AppState;
const ChatMessage = st_mod.ChatMessage;

const bubble_max: f32 = 560;

// --- Reasoning parsing -------------------------------------------------------
// The core llama.cpp C API we link streams raw text, so reasoning models emit
// their chain-of-thought inline. Different chat templates wrap it differently:
// most use `<think>…</think>`, while Gemma's harmony-style template (chat.cpp
// `COMMON_CHAT_FORMAT_PEG_GEMMA4`) uses `<|channel>thought … <channel|>`. We
// split whichever pair is present and render the reasoning as a collapsible,
// muted section above the answer. (Empty reasoning — e.g. Qwen with thinking
// off emits `<think>\n\n</think>` — renders nothing.)

const TagPair = struct { open: []const u8, close: []const u8 };

/// Reasoning delimiters we recognize, tried in order.
const think_tags = [_]TagPair{
    .{ .open = "<think>", .close = "</think>" },
    .{ .open = "<|channel>thought", .close = "<channel|>" },
};

/// Stray control tokens that may trail the answer (turn/channel/tool markers);
/// we cut the answer at the first one so they never leak into the bubble.
const stray_tokens = [_][]const u8{ "<turn|>", "<|turn>", "<|channel>", "<channel|>", "<|tool_call>" };

const Split = struct {
    /// Trimmed reasoning text; empty when there is no (non-empty) think block.
    reasoning: []const u8,
    /// The user-facing answer (everything after the closing tag).
    answer: []const u8,
    /// True while inside an unterminated think block (still streaming reasoning).
    thinking: bool,
};

/// Trim whitespace and drop anything from the first stray control token on.
fn cleanAnswer(s: []const u8) []const u8 {
    var a = std.mem.trim(u8, s, " \t\r\n");
    for (stray_tokens) |m| {
        if (std.mem.indexOf(u8, a, m)) |i| a = a[0..i];
    }
    return std.mem.trim(u8, a, " \t\r\n");
}

fn splitThink(content: []const u8) Split {
    // A completed think block: reasoning is everything before the close tag
    // (minus a leading open tag), answer is everything after it.
    for (think_tags) |t| {
        if (std.mem.indexOf(u8, content, t.close)) |ci| {
            var r = content[0..ci];
            const rt = std.mem.trimStart(u8, r, " \t\r\n");
            if (std.mem.startsWith(u8, rt, t.open)) r = rt[t.open.len..];
            return .{
                .reasoning = std.mem.trim(u8, r, " \t\r\n"),
                .answer = cleanAnswer(content[ci + t.close.len ..]),
                .thinking = false,
            };
        }
    }
    // No closing tag yet: anything after an opening tag is in-progress
    // reasoning with no answer text so far.
    for (think_tags) |t| {
        if (std.mem.indexOf(u8, content, t.open)) |oi| {
            return .{
                .reasoning = std.mem.trim(u8, content[oi + t.open.len ..], " \t\r\n"),
                .answer = "",
                .thinking = true,
            };
        }
    }
    return .{ .reasoning = "", .answer = cleanAnswer(content), .thinking = false };
}

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
    const sp = splitThink(cx.msg.content.items);
    cx.msg.think_expanded = !(cx.msg.think_expanded orelse sp.thinking);
}

/// A collapsible reasoning disclosure: a muted "Thinking…/Reasoning" header that
/// toggles a faint, indented block of the model's chain-of-thought.
fn reasoningBlock(st: *AppState, msg: *ChatMessage, sp: Split) zigui.View {
    const th = w.t();
    const expanded = msg.think_expanded orelse sp.thinking; // live while thinking
    const label = if (sp.thinking) "Thinking…" else "Reasoning";
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

    const body = maxWidth(zigui.WrappedText(sp.reasoning)
        .font(.caption)
        .foreground(th.colors.secondary_label)
        .paddingInsets(.{ .top = 8, .leading = 12, .bottom = 8, .trailing = 12 })
        .background(th.colors.secondary_label.withAlpha(0.07))
        .cornerRadius(10), bubble_max);

    // Pin the body to the leading edge (like the answer bubble); without this the
    // fixed-width box is centered by the VStack's default cross-axis alignment.
    return zigui.VStack(.{ head, leading(body) }).spacing(5).frameMaxWidth();
}

fn onSend(st: *AppState) void {
    st.sendChat();
}

fn onToggleAgent(st: *AppState) void {
    st.agent_mode.set(!st.agent_mode.get());
}

/// A compact toggle pill for agent mode, accent-filled when on.
fn agentPill(st: *AppState) zigui.View {
    const th = w.t();
    const on = st.agent_mode.get();
    const tools = st.mcp_mgr.toolCount();
    const fg = if (on) th.colors.on_accent else th.colors.secondary_label;
    const label = if (on) w.fmt("Agent · {d} tools", .{tools}) else "Agent";
    var pill = zigui.HStack(.{
        zigui.Icon(.zap, 13, fg),
        zigui.Text(label).font(.caption).foreground(fg),
    }).spacing(5)
        .paddingInsets(.{ .top = 6, .leading = 10, .bottom = 6, .trailing = 11 })
        .cornerRadius(8)
        .onTap(zigui.actionCtx(AppState, st, onToggleAgent));
    if (on)
        pill = pill.background(th.colors.accent)
    else
        pill = pill.background(th.colors.control_background).border(th.colors.separator, th.metrics.hairline);
    return pill;
}

fn onStop(st: *AppState) void {
    st.llama.cancel();
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
    const is_user = msg.role == .user;
    const fa = st.frame_arena.allocator();

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

    // Assistant: split out the `<think>` reasoning from the answer.
    const sp = splitThink(msg.content.items);

    var parts: std.ArrayList(zigui.View) = .empty;
    if (sp.reasoning.len > 0) parts.append(fa, reasoningBlock(st, msg, sp)) catch {};

    // The answer bubble is hidden while the model is still thinking (no answer
    // text yet); otherwise it shows the answer, with a streaming ellipsis.
    const show_answer = !(msg.streaming and sp.thinking and sp.answer.len == 0);
    if (show_answer) {
        const answer = if (msg.streaming and sp.answer.len == 0)
            "…"
        else if (msg.streaming)
            w.fmt("{s}…", .{sp.answer})
        else
            sp.answer;
        const bubble = maxWidth(zigui.WrappedText(answer)
            .foreground(th.colors.label)
            .paddingInsets(.{ .top = 8, .leading = 12, .bottom = 8, .trailing = 12 })
            .background(th.colors.control_background)
            .cornerRadius(14), bubble_max);
        parts.append(fa, zigui.HStack(.{ bubble, zigui.Spacer() }).frameMaxWidth()) catch {};
    }

    // Footer (Copy + token stats) once the message has settled with answer text.
    if (!msg.streaming and sp.answer.len > 0) {
        const copy = copyButton(st, sp.answer);
        const footer = if (msg.tokens > 0)
            zigui.HStack(.{
                copy,
                zigui.Text(w.fmt("{d} tokens · {d:.0} tok/s", .{ msg.tokens, msg.tps }))
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
