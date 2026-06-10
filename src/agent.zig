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
    const end_rel = std.mem.indexOfPos(u8, text, after, close_tag) orelse return null;
    const body = std.mem.trim(u8, text[after..end_rel], " \t\r\n");
    if (body.len == 0) return null;

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

test "parseToolCall returns null without a block" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try std.testing.expect(parseToolCall(arena, "just a normal answer") == null);
}
