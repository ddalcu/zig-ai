//! Zig bindings for the vendored single-header media encoders in src/codecs/*.c
//! (PNG via stb_image_write, H.264/MP4 via minih264 + minimp4 — no ffmpeg).
//! The C objects are compiled into the exe by build.zig.

const std = @import("std");

extern fn zigai_write_png(path: [*:0]const u8, rgba: [*]const u8, w: c_int, h: c_int) c_int;
extern fn zigai_encode_mp4(path: [*:0]const u8, frames: [*]const [*]const u8, n: c_int, w: c_int, h: c_int, fps: c_int) c_int;
extern fn zigai_load_image(path: [*:0]const u8, w: *c_int, h: *c_int) ?[*]u8;
extern fn zigai_load_image_mem(data: [*]const u8, len: c_int, w: *c_int, h: *c_int) ?[*]u8;
extern fn zigai_free_image(pixels: [*]u8) void;

/// A decoded RGBA8 image: `pixels` is `width*height*4` bytes, owned by stb_image
/// and freed with `freeImage` (NOT the Zig allocator).
pub const DecodedImage = struct { width: u32, height: u32, pixels: [*]u8 };

/// Decode an image file (PNG/JPG/WEBP/…) to RGBA8, or null on failure. Free the
/// result's `pixels` with `freeImage`.
pub fn loadImage(path: [:0]const u8) ?DecodedImage {
    var w: c_int = 0;
    var h: c_int = 0;
    const px = zigai_load_image(path.ptr, &w, &h) orelse return null;
    if (w <= 0 or h <= 0) {
        zigai_free_image(px);
        return null;
    }
    return .{ .width = @intCast(w), .height = @intCast(h), .pixels = px };
}

/// Decode an image from an in-memory buffer (e.g. an `@embedFile`'d PNG) to
/// RGBA8, or null on failure. Free with `freeImage`.
pub fn loadImageMem(bytes: []const u8) ?DecodedImage {
    var w: c_int = 0;
    var h: c_int = 0;
    const px = zigai_load_image_mem(bytes.ptr, @intCast(bytes.len), &w, &h) orelse return null;
    if (w <= 0 or h <= 0) {
        zigai_free_image(px);
        return null;
    }
    return .{ .width = @intCast(w), .height = @intCast(h), .pixels = px };
}

/// Free a buffer returned by `loadImage` (uses stb_image's allocator).
pub fn freeImage(img: DecodedImage) void {
    zigai_free_image(img.pixels);
}

/// Write an RGBA8 buffer (`w*h*4` bytes) to `path` as PNG. Returns true on success.
pub fn writePng(path: [:0]const u8, rgba: []const u8, w: u32, h: u32) bool {
    return zigai_write_png(path.ptr, rgba.ptr, @intCast(w), @intCast(h)) != 0;
}

/// Encode RGBA8 frames (each a pointer to `w*h*4` bytes) to an H.264/MP4 at `path`.
/// Returns true on success.
pub fn encodeMp4(path: [:0]const u8, frames: []const [*]const u8, w: u32, h: u32, fps: u32) bool {
    if (frames.len == 0) return false;
    return zigai_encode_mp4(path.ptr, frames.ptr, @intCast(frames.len), @intCast(w), @intCast(h), @intCast(fps)) != 0;
}
