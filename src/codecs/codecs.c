// PNG encoder for zig-ai (compiled by build.zig, called from Zig).
// Uses the already-vendored single-header stb_image_write — NO ffmpeg.
// The MP4/H.264 encoder lives in codecs_h264.c (minih264 + minimp4), kept in its
// own translation unit because those libs' implementations can't share a TU.

#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Save an RGBA8 image as PNG. Returns 1 on success.
int zigai_write_png(const char *path, const unsigned char *rgba, int w, int h) {
    if (!path || !rgba || w <= 0 || h <= 0) return 0;
    return stbi_write_png(path, w, h, 4, rgba, w * 4) ? 1 : 0;
}

// Decode an image file (PNG/JPG/…) to RGBA8. Writes dimensions to *w/*h and
// returns a malloc'd w*h*4 buffer (free with zigai_free_image), or NULL on error.
// Used for the video init frame (image-to-video).
unsigned char *zigai_load_image(const char *path, int *w, int *h) {
    if (!path || !w || !h) return 0;
    int channels = 0;
    return stbi_load(path, w, h, &channels, 4); // force 4 channels (RGBA)
}

// Decode an image from an in-memory buffer (e.g. an @embedFile'd PNG) to RGBA8.
// Same contract as zigai_load_image. Used for the embedded app icon.
unsigned char *zigai_load_image_mem(const unsigned char *data, int len, int *w, int *h) {
    if (!data || len <= 0 || !w || !h) return 0;
    int channels = 0;
    return stbi_load_from_memory(data, len, w, h, &channels, 4);
}

void zigai_free_image(unsigned char *pixels) {
    if (pixels) stbi_image_free(pixels);
}
