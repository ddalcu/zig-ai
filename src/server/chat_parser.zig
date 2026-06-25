//! Streaming chat-output parser — the heart of the single chat engine.
//!
//! Models stream raw text that interleaves three things: chain-of-thought
//! reasoning (`<think>…</think>` or Gemma's `<|channel>thought…<channel|>`),
//! the user-facing answer, and — in agent mode — a tool call
//! (`<tool_call>{…}</tool_call>`). This parser classifies the stream into
//! reasoning / content / tool-call as bytes arrive, stripping the markup, so the
//! HTTP server can emit clean `content`, separate `reasoning_content`, and
//! structured `tool_calls`. Both the GUI (over HTTP) and external clients go
//! through this one path, so there's nothing to keep in sync.
//!
//! Tag-boundary safe: a marker split across token chunks (e.g. `<thi` then `nk>`)
//! is held back until it completes, so partial markup never leaks.

const std = @import("std");
const agent = @import("../agent.zig");

pub const ToolCall = agent.ToolCall;

const ReasonPair = struct { open: []const u8, close: []const u8 };
const reason_pairs = [_]ReasonPair{
    .{ .open = "<think>", .close = "</think>" },
    .{ .open = "<|channel>thought", .close = "<channel|>" },
};
const tool_open = agent.open_tag; // "<tool_call>"
const tool_close = agent.close_tag; // "</tool_call>"

// Markup that should never reach the user; seeing one ends visible output (it
// usually means end-of-turn or a template mismatch leaking control tokens).
const terminators = [_][]const u8{
    "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<turn|>", "<|turn>", "<|tool_call>",
};

const Kind = enum { reason_open, tool_open, terminator };
const Marker = struct { str: []const u8, kind: Kind, pair: usize };

// Markers we look for while emitting plain content, longest-first so a longer
// marker wins a tie at the same index (e.g. `<|channel>thought` over a bare
// terminator). Built at comptime from the tables above.
const content_markers = blk: {
    var list: [reason_pairs.len + 1 + terminators.len]Marker = undefined;
    var n: usize = 0;
    for (reason_pairs, 0..) |p, i| {
        list[n] = .{ .str = p.open, .kind = .reason_open, .pair = i };
        n += 1;
    }
    list[n] = .{ .str = tool_open, .kind = .tool_open, .pair = 0 };
    n += 1;
    for (terminators) |t| {
        list[n] = .{ .str = t, .kind = .terminator, .pair = 0 };
        n += 1;
    }
    break :blk list;
};

pub const Parser = struct {
    const State = enum { content, reasoning, tool, done };

    gpa: std.mem.Allocator,
    state: State = .content,
    close_marker: []const u8 = "", // active reasoning close tag
    pending: std.ArrayList(u8) = .empty, // bytes not yet classified (maybe a partial tag)
    tool_body: std.ArrayList(u8) = .empty, // accumulated `<tool_call>` body
    have_tool: bool = false,

    pub fn init(gpa: std.mem.Allocator) Parser {
        return .{ .gpa = gpa };
    }
    pub fn deinit(self: *Parser) void {
        self.pending.deinit(self.gpa);
        self.tool_body.deinit(self.gpa);
    }

    /// Feed a raw text chunk; classified bytes are appended to `reasoning` and
    /// `content` (the caller's accumulators — diff their lengths for stream deltas).
    pub fn feed(self: *Parser, chunk: []const u8, reasoning: *std.ArrayList(u8), content: *std.ArrayList(u8)) void {
        self.pending.appendSlice(self.gpa, chunk) catch return;
        self.process(reasoning, content, false);
    }

    /// Flush at end of stream. Returns the parsed tool call, if the model emitted
    /// one (`args_json` etc. live in `arena`).
    pub fn finish(self: *Parser, arena: std.mem.Allocator, reasoning: *std.ArrayList(u8), content: *std.ArrayList(u8)) ?ToolCall {
        self.process(reasoning, content, true);
        if (self.have_tool or self.tool_body.items.len > 0) {
            const text = std.fmt.allocPrint(arena, "{s}{s}", .{ tool_open, self.tool_body.items }) catch return null;
            return agent.parseToolCall(arena, text);
        }
        return null;
    }

    fn consume(self: *Parser, n: usize) void {
        const rest = self.pending.items[n..];
        std.mem.copyForwards(u8, self.pending.items[0..rest.len], rest);
        self.pending.items.len = rest.len;
    }

    fn process(self: *Parser, reasoning: *std.ArrayList(u8), content: *std.ArrayList(u8), final: bool) void {
        while (true) switch (self.state) {
            .done => {
                self.pending.clearRetainingCapacity();
                return;
            },
            .content => {
                if (earliest(self.pending.items, &content_markers)) |hit| {
                    content.appendSlice(self.gpa, self.pending.items[0..hit.idx]) catch {};
                    self.consume(hit.idx + hit.len);
                    switch (hit.kind) {
                        .reason_open => {
                            self.state = .reasoning;
                            self.close_marker = reason_pairs[hit.pair].close;
                        },
                        .tool_open => self.state = .tool,
                        .terminator => self.state = .done,
                    }
                } else {
                    const keep = if (final) 0 else partialLen(self.pending.items, &content_markers);
                    const cut = self.pending.items.len - keep;
                    content.appendSlice(self.gpa, self.pending.items[0..cut]) catch {};
                    self.consume(cut);
                    return;
                }
            },
            .reasoning => {
                // Earliest of the active close tag or any terminator.
                const close_at = std.mem.indexOf(u8, self.pending.items, self.close_marker);
                const term = earliest(self.pending.items, &term_markers);
                const term_at = if (term) |t| t.idx else null;
                if (close_at != null and (term_at == null or close_at.? <= term_at.?)) {
                    reasoning.appendSlice(self.gpa, self.pending.items[0..close_at.?]) catch {};
                    self.consume(close_at.? + self.close_marker.len);
                    self.state = .content;
                } else if (term_at) |ti| {
                    reasoning.appendSlice(self.gpa, self.pending.items[0..ti]) catch {};
                    self.state = .done;
                    self.pending.clearRetainingCapacity();
                } else {
                    const keep = if (final) 0 else @max(
                        partialOne(self.pending.items, self.close_marker),
                        partialLen(self.pending.items, &term_markers),
                    );
                    const cut = self.pending.items.len - keep;
                    reasoning.appendSlice(self.gpa, self.pending.items[0..cut]) catch {};
                    self.consume(cut);
                    return;
                }
            },
            .tool => {
                if (std.mem.indexOf(u8, self.pending.items, tool_close)) |ci| {
                    self.tool_body.appendSlice(self.gpa, self.pending.items[0..ci]) catch {};
                    self.have_tool = true;
                    self.consume(ci + tool_close.len);
                    self.state = .done; // one tool call per turn; ignore any trailer
                } else {
                    const keep = if (final) 0 else partialOne(self.pending.items, tool_close);
                    const cut = self.pending.items.len - keep;
                    self.tool_body.appendSlice(self.gpa, self.pending.items[0..cut]) catch {};
                    self.consume(cut);
                    return;
                }
            },
        };
    }
};

// Terminator-only marker table (used inside reasoning/tool states).
const term_markers = blk: {
    var list: [terminators.len]Marker = undefined;
    for (terminators, 0..) |t, i| list[i] = .{ .str = t, .kind = .terminator, .pair = 0 };
    break :blk list;
};

const Hit = struct { idx: usize, len: usize, kind: Kind, pair: usize };

/// Earliest complete marker in `s` (smallest index; longest marker on a tie).
fn earliest(s: []const u8, markers: []const Marker) ?Hit {
    var best: ?Hit = null;
    for (markers) |m| {
        if (std.mem.indexOf(u8, s, m.str)) |i| {
            if (best == null or i < best.?.idx or (i == best.?.idx and m.str.len > best.?.len)) {
                best = .{ .idx = i, .len = m.str.len, .kind = m.kind, .pair = m.pair };
            }
        }
    }
    return best;
}

/// Longest suffix of `s` that is a strict prefix of `marker` (boundary holdback).
fn partialOne(s: []const u8, marker: []const u8) usize {
    var l = @min(s.len, marker.len - 1);
    while (l > 0) : (l -= 1) {
        if (std.mem.eql(u8, s[s.len - l ..], marker[0..l])) return l;
    }
    return 0;
}

fn partialLen(s: []const u8, markers: []const Marker) usize {
    var max: usize = 0;
    for (markers) |m| max = @max(max, partialOne(s, m.str));
    return max;
}

// ---------------------------------------------------------------------------
const testing = std.testing;

fn run(input: []const []const u8) struct { r: []u8, c: []u8, tool: ?ToolCall, arena: *std.heap.ArenaAllocator } {
    const arena = testing.allocator.create(std.heap.ArenaAllocator) catch unreachable;
    arena.* = std.heap.ArenaAllocator.init(testing.allocator);
    const a = arena.allocator();
    var p = Parser.init(testing.allocator);
    defer p.deinit();
    var r: std.ArrayList(u8) = .empty;
    var c: std.ArrayList(u8) = .empty;
    for (input) |chunk| p.feed(chunk, &r, &c);
    const tc = p.finish(a, &r, &c);
    return .{ .r = r.toOwnedSlice(testing.allocator) catch unreachable, .c = c.toOwnedSlice(testing.allocator) catch unreachable, .tool = tc, .arena = arena };
}

fn free(res: anytype) void {
    testing.allocator.free(res.r);
    testing.allocator.free(res.c);
    res.arena.deinit();
    testing.allocator.destroy(res.arena);
}

test "plain content passes through" {
    const res = run(&.{ "Hello ", "world" });
    defer free(res);
    try testing.expectEqualStrings("Hello world", res.c);
    try testing.expectEqualStrings("", res.r);
    try testing.expect(res.tool == null);
}

test "think block goes to reasoning, answer to content" {
    const res = run(&.{"<think>plan it</think>The answer."});
    defer free(res);
    try testing.expectEqualStrings("plan it", res.r);
    try testing.expectEqualStrings("The answer.", res.c);
}

test "tags split across chunks are held back" {
    const res = run(&.{ "<th", "ink>hmm</th", "ink>done" });
    defer free(res);
    try testing.expectEqualStrings("hmm", res.r);
    try testing.expectEqualStrings("done", res.c);
}

test "tool call is extracted and stripped from content" {
    const res = run(&.{"Let me check.\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>"});
    defer free(res);
    try testing.expectEqualStrings("Let me check.", std.mem.trim(u8, res.c, " \t\r\n"));
    try testing.expect(res.tool != null);
    try testing.expectEqualStrings("get_weather", res.tool.?.name);
    try testing.expect(std.mem.indexOf(u8, res.tool.?.args_json, "Paris") != null);
}

test "reasoning then tool call" {
    const res = run(&.{"<think>need a tool</think><tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>"});
    defer free(res);
    try testing.expectEqualStrings("need a tool", res.r);
    try testing.expectEqualStrings("", std.mem.trim(u8, res.c, " \t\r\n"));
    try testing.expect(res.tool != null and std.mem.eql(u8, res.tool.?.name, "x"));
}

test "unclosed think keeps reasoning, no content leak" {
    const res = run(&.{"<think>still thinking and never closed"});
    defer free(res);
    try testing.expectEqualStrings("still thinking and never closed", res.r);
    try testing.expectEqualStrings("", res.c);
}

test "terminator cuts content" {
    const res = run(&.{"Answer here<|im_end|>garbage"});
    defer free(res);
    try testing.expectEqualStrings("Answer here", res.c);
}

test "tool call with dropped close tag still parses" {
    const res = run(&.{"<tool_call>{\"name\":\"x\",\"arguments\":{\"a\":1}}"});
    defer free(res);
    try testing.expect(res.tool != null and std.mem.eql(u8, res.tool.?.name, "x"));
}
