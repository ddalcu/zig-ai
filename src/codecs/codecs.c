// PNG encoder for zig-ai (compiled by build.zig, called from Zig).
// Uses the already-vendored single-header stb_image_write — NO ffmpeg.
// The MP4/H.264 encoder lives in codecs_h264.c (minih264 + minimp4), kept in its
// own translation unit because those libs' implementations can't share a TU.

#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Save an RGBA8 image as PNG. Returns 1 on success.
int zigai_write_png(const char *path, const unsigned char *rgba, int w, int h) {
    if (!path || !rgba || w <= 0 || h <= 0) return 0;
    return stbi_write_png(path, w, h, 4, rgba, w * 4) ? 1 : 0;
}
