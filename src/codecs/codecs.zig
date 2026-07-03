//! Zig bindings for the vendored single-header media encoders in src/codecs/*.c
//! (PNG via stb_image_write, H.264/MP4 via minih264 + minimp4 — no ffmpeg).
//! The C objects are compiled into the exe by build.zig.

const std = @import("std");

extern fn zigai_write_png(path: [*:0]const u8, rgba: [*]const u8, w: c_int, h: c_int) c_int;
extern fn zigai_encode_png_mem(rgba: [*]const u8, w: c_int, h: c_int, out_len: *c_int) ?[*]u8;
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

/// Encode an RGBA8 buffer as PNG in memory; free the result with `freePng`.
pub fn encodePngMem(rgba: []const u8, w: u32, h: u32) ?[]u8 {
    var len: c_int = 0;
    const p = zigai_encode_png_mem(rgba.ptr, @intCast(w), @intCast(h), &len) orelse return null;
    if (len <= 0) {
        zigai_free_image(p);
        return null;
    }
    return p[0..@intCast(len)];
}

/// Free a buffer returned by `encodePngMem` (stb allocator).
pub fn freePng(png: []u8) void {
    zigai_free_image(png.ptr);
}

/// Encode RGBA8 frames (each a pointer to `w*h*4` bytes) to an H.264/MP4 at `path`.
/// Returns true on success.
pub fn encodeMp4(path: [:0]const u8, frames: []const [*]const u8, w: u32, h: u32, fps: u32) bool {
    if (frames.len == 0) return false;
    return zigai_encode_mp4(path.ptr, frames.ptr, @intCast(frames.len), @intCast(w), @intCast(h), @intCast(fps)) != 0;
}

/// Write interleaved f32 samples in [-1,1] to `path` as a PCM16 WAV.
/// Returns true on success. Pure Zig via libc stdio (matches the C encoders'
/// I/O path; std.fs would need the new std.Io plumbing).
pub fn writeWav(path: [:0]const u8, samples: []const f32, sample_rate: u32, channels: u32) bool {
    if (samples.len == 0 or channels == 0 or sample_rate == 0) return false;
    const cio = @cImport(@cInclude("stdio.h"));
    const f = cio.fopen(path.ptr, "wb") orelse return false;
    defer _ = cio.fclose(f);

    const data_bytes: u32 = @intCast(samples.len * 2);
    const byte_rate: u32 = sample_rate * channels * 2;
    const block_align: u16 = @intCast(channels * 2);

    var header: [44]u8 = undefined;
    @memcpy(header[0..4], "RIFF");
    std.mem.writeInt(u32, header[4..8], 36 + data_bytes, .little);
    @memcpy(header[8..12], "WAVE");
    @memcpy(header[12..16], "fmt ");
    std.mem.writeInt(u32, header[16..20], 16, .little); // PCM fmt chunk size
    std.mem.writeInt(u16, header[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, header[22..24], @intCast(channels), .little);
    std.mem.writeInt(u32, header[24..28], sample_rate, .little);
    std.mem.writeInt(u32, header[28..32], byte_rate, .little);
    std.mem.writeInt(u16, header[32..34], block_align, .little);
    std.mem.writeInt(u16, header[34..36], 16, .little); // bits per sample
    @memcpy(header[36..40], "data");
    std.mem.writeInt(u32, header[40..44], data_bytes, .little);
    if (cio.fwrite(&header, 1, header.len, f) != header.len) return false;

    // Convert + write in chunks so we don't need an allocator.
    var buf: [4096]i16 = undefined;
    var off: usize = 0;
    while (off < samples.len) {
        const n = @min(buf.len, samples.len - off);
        for (samples[off .. off + n], 0..) |v, i| {
            const clamped = @max(@as(f32, -1.0), @min(@as(f32, 1.0), v));
            buf[i] = @intFromFloat(@round(clamped * 32767.0));
        }
        if (cio.fwrite(&buf, 2, n, f) != n) return false;
        off += n;
    }
    return true;
}
