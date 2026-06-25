//! Minimal markdown → zigui view rendering for chat bubbles.
//!
//! Block level: ATX headings (`#`/`##`/`###`), fenced code blocks (```` ``` ````),
//! unordered/ordered lists, and paragraphs (soft line breaks join with a space).
//! Inline: `**bold**`/`*italic*` markers are stripped, `` `code` `` is unwrapped
//! verbatim, and `[text](url)` collapses to `text`. zigui has no monospace face
//! or rich inline runs, so code blocks render in the UI font on a contrasting
//! card — enough to read structured replies without a zigui change.

const std = @import("std");
const zigui = @import("zigui");
const w = @import("widgets.zig");

/// Build a view tree for `src`. `fa` is the per-frame arena. The result fills its
/// width (the caller caps it with `maxWidth(..., bubble_max)`); blocks wrap to it.
pub fn view(fa: std.mem.Allocator, src: []const u8) zigui.View {
    const th = w.t();

    // Lines, materialized so the block parser can look ahead (fences, lists,
    // multi-line paragraphs).
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |ln| lines.append(fa, ln) catch {};

    var blocks: std.ArrayList(zigui.View) = .empty;
    var i: usize = 0;
    while (i < lines.items.len) {
        const line = std.mem.trimStart(u8, lines.items[i], " \t");

        // Fenced code block: collect verbatim until the closing fence.
        if (std.mem.startsWith(u8, line, "```")) {
            i += 1;
            var code: std.ArrayList(u8) = .empty;
            while (i < lines.items.len) : (i += 1) {
                if (std.mem.startsWith(u8, std.mem.trimStart(u8, lines.items[i], " \t"), "```")) {
                    i += 1;
                    break;
                }
                if (code.items.len > 0) code.append(fa, '\n') catch {};
                code.appendSlice(fa, lines.items[i]) catch {};
            }
            blocks.append(fa, codeBlock(th, code.items)) catch {};
            continue;
        }

        // ATX heading.
        if (headingLevel(line)) |lvl| {
            const txt = std.mem.trim(u8, line[lvl..], " #\t");
            blocks.append(fa, zigui.WrappedText(cleanInline(fa, txt))
                .font(headingFont(lvl))
                .foreground(th.colors.label)
                .frameMaxWidth()
                .frameAlign(.leading)) catch {};
            i += 1;
            continue;
        }

        // List: consecutive item lines.
        if (listMarker(line) != null) {
            var items: std.ArrayList(zigui.View) = .empty;
            while (i < lines.items.len) {
                const t2 = std.mem.trimStart(u8, lines.items[i], " \t");
                const m = listMarker(t2) orelse break;
                const body = cleanInline(fa, std.mem.trimStart(u8, t2[m.skip..], " "));
                items.append(fa, listItem(th, m.bullet, body)) catch {};
                i += 1;
            }
            blocks.append(fa, zigui.VStack(items.items)
                .spacing(3)
                .alignment(.leading)
                .frameMaxWidth()
                .paddingInsets(.{ .leading = 4 })) catch {};
            continue;
        }

        // Blank line.
        if (line.len == 0) {
            i += 1;
            continue;
        }

        // Paragraph: gather consecutive plain lines (soft breaks → spaces).
        var para: std.ArrayList(u8) = .empty;
        while (i < lines.items.len) {
            const t2 = std.mem.trimStart(u8, lines.items[i], " \t");
            if (t2.len == 0 or std.mem.startsWith(u8, t2, "```") or
                headingLevel(t2) != null or listMarker(t2) != null) break;
            if (para.items.len > 0) para.append(fa, ' ') catch {};
            para.appendSlice(fa, std.mem.trim(u8, lines.items[i], " \t")) catch {};
            i += 1;
        }
        blocks.append(fa, zigui.WrappedText(cleanInline(fa, para.items))
            .foreground(th.colors.label)
            .frameMaxWidth()
            .frameAlign(.leading)) catch {};
    }

    return zigui.VStack(blocks.items).spacing(8).alignment(.leading).frameMaxWidth();
}

fn codeBlock(th: zigui.Theme, code: []const u8) zigui.View {
    return zigui.WrappedText(code)
        .foreground(th.colors.secondary_label)
        .frameMaxWidth()
        .frameAlign(.leading)
        .paddingInsets(.{ .top = 8, .leading = 10, .bottom = 8, .trailing = 10 })
        .background(th.colors.window_background)
        .cornerRadius(8);
}

fn listItem(th: zigui.Theme, bullet: []const u8, body: []const u8) zigui.View {
    return zigui.HStack(.{
        zigui.Text(bullet).foreground(th.colors.secondary_label),
        zigui.WrappedText(body).foreground(th.colors.label).frameMaxWidth().frameAlign(.leading),
    }).spacing(8).alignment(.top).frameMaxWidth();
}

fn headingFont(lvl: usize) zigui.view.FontToken {
    return switch (lvl) {
        1 => .title2,
        2 => .title3,
        else => .headline,
    };
}

/// Number of leading `#` (1–6) if `s` is an ATX heading (`#` run followed by a
/// space), else null.
fn headingLevel(s: []const u8) ?usize {
    var n: usize = 0;
    while (n < s.len and s[n] == '#') : (n += 1) {}
    if (n >= 1 and n <= 6 and n < s.len and s[n] == ' ') return n;
    return null;
}

const ListM = struct { skip: usize, bullet: []const u8 };

/// Recognize a list item at the start of `s`: `- `/`* `/`+ ` → a bullet, or
/// `<digits>. ` → the kept number. `skip` is how many bytes precede the content.
fn listMarker(s: []const u8) ?ListM {
    if (s.len >= 2 and (s[0] == '-' or s[0] == '*' or s[0] == '+') and s[1] == ' ')
        return .{ .skip = 2, .bullet = "•" };
    var j: usize = 0;
    while (j < s.len and std.ascii.isDigit(s[j])) : (j += 1) {}
    if (j > 0 and j + 1 < s.len and s[j] == '.' and s[j + 1] == ' ')
        return .{ .skip = j + 2, .bullet = s[0 .. j + 1] };
    return null;
}

/// Strip inline markdown markers: `*`/`**` emphasis is removed, `` `code` `` is
/// unwrapped (content kept verbatim, so identifiers inside it survive), and
/// `[text](url)` collapses to `text`. `_` is intentionally left alone so
/// snake_case identifiers aren't mangled. Returns `s` unchanged when it has no
/// markers (no allocation in the common case).
fn cleanInline(fa: std.mem.Allocator, s: []const u8) []const u8 {
    if (std.mem.indexOfAny(u8, s, "*`[") == null) return s;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const ch = s[i];
        switch (ch) {
            '`' => {
                if (std.mem.indexOfScalarPos(u8, s, i + 1, '`')) |close| {
                    out.appendSlice(fa, s[i + 1 .. close]) catch {};
                    i = close + 1;
                } else i += 1; // unmatched: drop the backtick
            },
            '*' => i += if (i + 1 < s.len and s[i + 1] == '*') 2 else 1,
            '[' => {
                const rb = std.mem.indexOfScalarPos(u8, s, i, ']');
                if (rb != null and rb.? + 1 < s.len and s[rb.? + 1] == '(') {
                    if (std.mem.indexOfScalarPos(u8, s, rb.? + 1, ')')) |rp| {
                        out.appendSlice(fa, s[i + 1 .. rb.?]) catch {};
                        i = rp + 1;
                        continue;
                    }
                }
                out.append(fa, ch) catch {};
                i += 1;
            },
            else => {
                out.append(fa, ch) catch {};
                i += 1;
            },
        }
    }
    return out.items;
}
