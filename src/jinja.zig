//! Thin Zig binding over the vendored Jinja engine (deps/jinja, Apache-2.0).
//! Renders a model's actual `chat_template` (a Jinja string from the GGUF) with a
//! messages array — so any model gets its correct prompt format, not llama.cpp's
//! hardcoded-template guess.

const std = @import("std");

const c = @cImport({
    @cInclude("jinja_wrapper.h");
});

/// Render `template_str` (Jinja) with `messages_json` (a JSON array of
/// `{role, content}`), `tools_json` (a JSON array or null), and `extra_json` (a
/// JSON object of extra vars, e.g. bos_token/eos_token). Returns the rendered
/// prompt owned by `gpa`, or null if the template failed to render.
pub fn renderChat(
    gpa: std.mem.Allocator,
    template_str: []const u8,
    messages_json: []const u8,
    tools_json: ?[]const u8,
    extra_json: []const u8,
    add_generation_prompt: bool,
) ?[]u8 {
    const tz = gpa.dupeZ(u8, template_str) catch return null;
    defer gpa.free(tz);
    const mz = gpa.dupeZ(u8, messages_json) catch return null;
    defer gpa.free(mz);
    const ez = gpa.dupeZ(u8, extra_json) catch return null;
    defer gpa.free(ez);
    var toz: ?[:0]u8 = null;
    if (tools_json) |t| toz = gpa.dupeZ(u8, t) catch return null;
    defer if (toz) |p| gpa.free(p);

    const out = c.jinja_render_chat(
        tz.ptr,
        mz.ptr,
        if (toz) |p| p.ptr else null,
        ez.ptr,
        if (add_generation_prompt) 1 else 0,
    );
    if (out == null) return null;
    defer c.jinja_str_free(out);
    return gpa.dupe(u8, std.mem.span(out)) catch null;
}
