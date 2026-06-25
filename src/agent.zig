//! Agent-mode helpers: assemble the system prompt (user persona + live MCP tool
//! catalogue) and parse the `<tool_call>…</tool_call>` blocks the model emits.
//!
//! The design targets small local models driven through llama.cpp's plain chat
//! template (no native function-calling): tools are advertised in the system
//! prompt and the model replies with a single fenced JSON object, which we parse
//! out of the streamed text. The result of running the tool is fed back as the
//! next user message, ReAct-style.

const std = @import("std");
const mcp = @import("mcp.zig");

pub const open_tag = "<tool_call>";
pub const close_tag = "</tool_call>";

/// Max agentic tool round-trips per user turn before we stop, to bound runaway
/// loops on a model that keeps calling tools.
pub const max_iterations: u32 = 12;

/// Build the full system prompt: the user-editable base text, plus — in agent
/// mode with at least one tool available — a generated catalogue of the tools.
/// Returned slice is allocated in `arena`.
pub fn buildSystemPrompt(
    arena: std.mem.Allocator,
    base: []const u8,
    manager: ?*mcp.Manager,
    agent_mode: bool,
) []const u8 {
    if (!agent_mode) return base;
    const mgr = manager orelse return base;
    const tools = mgr.toolListAlloc(arena);
    if (tools.len == 0) return base;

    var out: std.ArrayList(u8) = .empty;
    out.appendSlice(arena, base) catch return base;
    out.appendSlice(arena,
        \\
        \\
        \\# Agent mode
        \\
        \\You can call tools to act on the user's behalf. Work autonomously: take
        \\multiple steps without pausing for confirmation, and only stop to ask the
        \\user when you hit genuine ambiguity or you are finished.
        \\
        \\## Available tools
        \\
        \\Address each tool by its exact name.
        \\
        \\
    ) catch {};
    for (tools) |t| {
        out.appendSlice(arena, "- `") catch {};
        out.appendSlice(arena, t.qualified) catch {};
        out.appendSlice(arena, "`") catch {};
        if (t.description.len > 0) {
            out.appendSlice(arena, " — ") catch {};
            appendOneLine(arena, &out, t.description);
        }
        if (t.schema.len > 0 and !std.mem.eql(u8, t.schema, "{}")) {
            out.appendSlice(arena, "\n    input schema: ") catch {};
            appendOneLine(arena, &out, t.schema);
        }
        out.append(arena, '\n') catch {};
    }
    out.appendSlice(arena,
        \\
        \\To use a tool, end your reply with exactly:
        \\
    ) catch {};
    out.appendSlice(arena, open_tag) catch {};
    out.appendSlice(arena, "{\"name\": \"<tool>\", \"arguments\": { ... }}") catch {};
    out.appendSlice(arena, close_tag) catch {};
    out.appendSlice(arena, "\n\nDo not invent tools that are not listed.\n") catch {};
    return out.items;
}

/// Return the first balanced `{…}` JSON object in `s` (string- and escape-aware),
/// or null. Used to recover a tool call when the model omits the close tag.
fn extractJsonObject(s: []const u8) ?[]const u8 {
    const start = std.mem.indexOfScalar(u8, s, '{') orelse return null;
    var depth: usize = 0;
    var in_str = false;
    var esc = false;
    var i = start;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (in_str) {
            if (esc) {
                esc = false;
            } else if (c == '\\') {
                esc = true;
            } else if (c == '"') {
                in_str = false;
            }
        } else if (c == '"') {
            in_str = true;
        } else if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
            if (depth == 0) return s[start .. i + 1];
        }
    }
    return null;
}

/// Recover a tool-call object whose trailing `}`(s) the model dropped: take from
/// the first `{` to the end of `s`, and if braces are net-open, append enough
/// closing braces to balance. Returns null if there's no `{` or it's not the
/// missing-close case (so we don't fabricate structure from genuine garbage).
fn balanceObject(arena: std.mem.Allocator, s: []const u8) ?[]const u8 {
    const start = std.mem.indexOfScalar(u8, s, '{') orelse return null;
    var depth: isize = 0;
    var in_str = false;
    var esc = false;
    for (s[start..]) |c| {
        if (in_str) {
            if (esc) esc = false else if (c == '\\') esc = true else if (c == '"') in_str = false;
        } else if (c == '"') {
            in_str = true;
        } else if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
        }
    }
    if (depth <= 0) return null; // balanced (handled elsewhere) or over-closed
    var out: std.ArrayList(u8) = .empty;
    out.appendSlice(arena, s[start..]) catch return null;
    var i: isize = 0;
    while (i < depth) : (i += 1) out.append(arena, '}') catch return null;
    return out.items;
}

fn appendOneLine(arena: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) void {
    for (s) |ch| {
        out.append(arena, if (ch == '\n' or ch == '\r') ' ' else ch) catch return;
    }
}

pub const ToolCall = struct {
    name: []const u8, // slice into the source text
    args_json: []const u8, // slice into the source text
};

/// Find the first `<tool_call>{...}</tool_call>` block in `text` and extract the
/// tool name and the raw `arguments` JSON. Slices point into `text`. Returns
/// null if there's no well-formed call.
pub fn parseToolCall(arena: std.mem.Allocator, text: []const u8) ?ToolCall {
    const start = std.mem.indexOf(u8, text, open_tag) orelse return null;
    const after = start + open_tag.len;
    // The call object is between the tags when both are present, else everything
    // after the open tag. Small local models are sloppy: they drop the close tag
    // and/or trailing `}`, so try a clean balanced object first, then a
    // brace-repaired one.
    const region = if (std.mem.indexOfPos(u8, text, after, close_tag)) |e| text[after..e] else text[after..];
    const body = extractJsonObject(region) orelse balanceObject(arena, region) orelse return null;

    const value = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch return null;
    if (value != .object) return null;
    const name_v = value.object.get("name") orelse return null;
    if (name_v != .string) return null;

    var args_json: []const u8 = "{}";
    if (value.object.get("arguments")) |a| {
        // Re-serialize the arguments object so it's a clean, owned JSON string.
        args_json = std.json.Stringify.valueAlloc(arena, a, .{}) catch "{}";
    }
    const name = arena.dupe(u8, name_v.string) catch return null;
    return .{ .name = name, .args_json = args_json };
}

test "parseToolCall extracts name and arguments" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const text =
        \\Sure, let me read it.
        \\<tool_call>{"name": "fs__read_file", "arguments": {"path": "/tmp/x"}}</tool_call>
    ;
    const tc = parseToolCall(arena, text) orelse return error.NoCall;
    try std.testing.expectEqualStrings("fs__read_file", tc.name);
    try std.testing.expect(std.mem.indexOf(u8, tc.args_json, "/tmp/x") != null);
}

test "parseToolCall tolerates a missing close tag" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const text = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
    const tc = parseToolCall(arena, text) orelse return error.NoCall;
    try std.testing.expectEqualStrings("get_weather", tc.name);
    try std.testing.expect(std.mem.indexOf(u8, tc.args_json, "Paris") != null);
}

test "parseToolCall repairs a dropped trailing brace" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // outer `}` missing before the close tag (a real gemma-E2B failure)
    const text = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}</tool_call>";
    const tc = parseToolCall(arena, text) orelse return error.NoCall;
    try std.testing.expectEqualStrings("get_weather", tc.name);
    try std.testing.expect(std.mem.indexOf(u8, tc.args_json, "Tokyo") != null);
}

test "parseToolCall returns null without a block" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try std.testing.expect(parseToolCall(arena, "just a normal answer") == null);
}
