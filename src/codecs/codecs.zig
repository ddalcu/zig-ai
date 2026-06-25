//! Zig bindings for the vendored single-header media encoders in src/codecs/*.c
//! (PNG via stb_image_write, H.264/MP4 via minih264 + minimp4 — no ffmpeg).
//! The C objects are compiled into the exe by build.zig.

const std = @import("std");

extern fn zigai_write_png(path: [*:0]const u8, rgba: [*]const u8, w: c_int, h: c_int) c_int;
extern fn zigai_encode_mp4(path: [*:0]const u8, frames: [*]const [*]const u8, n: c_int, w: c_int, h: c_int, fps: c_int) c_int;

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
