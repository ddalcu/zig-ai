// H.264/MP4 encoder for zig-ai: minih264 (CC0) + minimp4 (public domain), NO ffmpeg.
//
// minih264's IMPLEMENTATION and the encode logic live in THIS one TU so that the
// code filling H264E_* structs and the code reading them agree on layout — a
// split there silently byte-swaps the bitstream. minimp4's IMPLEMENTATION must be
// in a SEPARATE TU (codecs_mp4.c): both libs define a `static nal_put_esc`, so
// their impls collide in one TU. Here we include minimp4 for DECLARATIONS only.
//
// Config macros must match across TUs (they change struct sizes):
//   * H264E_SVC_API 0 / H264E_MAX_THREADS 0 — plain single-layer AVC, no threads
//     (defaults are SVC=1/THREADS=4, which add fields; a zeroed num_layers then
//     makes minih264 emit garbage SVC-extension NALs).
//   * MINIMP4_TRANSCODE_SPS_ID 0 — minih264 already emits correct SPS/PPS ids;
//     minimp4's transcoder otherwise rejects them.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H264E_SVC_API 0
#define H264E_MAX_THREADS 0
#define MINIH264_IMPLEMENTATION
#include "minih264e.h"

#define MINIMP4_TRANSCODE_SPS_ID 0
#include "minimp4.h" // declarations only (implementation in codecs_mp4.c)

// MP4 output buffer: minimp4 writes at arbitrary offsets (it patches the moov),
// so accumulate into a growable buffer addressed by offset, then dump to disk.
typedef struct {
    unsigned char *data;
    size_t len, cap;
    int err;
} membuf_t;

static int membuf_write(int64_t offset, const void *buffer, size_t size, void *token) {
    membuf_t *m = (membuf_t *)token;
    size_t end = (size_t)offset + size;
    if (end > m->cap) {
        size_t ncap = m->cap ? m->cap : (1u << 16);
        while (ncap < end) ncap *= 2;
        unsigned char *nd = (unsigned char *)realloc(m->data, ncap);
        if (!nd) { m->err = 1; return 1; }
        m->data = nd;
        m->cap = ncap;
    }
    memcpy(m->data + offset, buffer, size);
    if (end > m->len) m->len = end;
    return 0;
}

// Fill planar I420 (limited-range BT.601) from RGBA. The Y plane has stride `ew`
// and the chroma planes stride `ew/2`; only the real w*h region is written, so a
// caller that pre-clears the planes to black gets clean padding to a 16-multiple.
static void fill_i420(const unsigned char *rgba, int w, int h, int ew,
                      unsigned char *y, unsigned char *u, unsigned char *v) {
    const int cw = ew / 2;
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            const unsigned char *p = rgba + ((size_t)j * w + i) * 4;
            int r = p[0], g = p[1], b = p[2];
            int yy = (66 * r + 129 * g + 25 * b + 128) / 256 + 16;
            y[(size_t)j * ew + i] = (unsigned char)(yy < 16 ? 16 : yy > 235 ? 235 : yy);
        }
    }
    for (int j = 0; j < h / 2; j++) {
        for (int i = 0; i < w / 2; i++) {
            int r = 0, g = 0, b = 0;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++) {
                    const unsigned char *q = rgba + (((size_t)(j * 2 + dy) * w) + (i * 2 + dx)) * 4;
                    r += q[0]; g += q[1]; b += q[2];
                }
            r >>= 2; g >>= 2; b >>= 2;
            int uu = (-38 * r - 74 * g + 112 * b + 128) / 256 + 128;
            int vv = (112 * r - 94 * g - 18 * b + 128) / 256 + 128;
            u[(size_t)j * cw + i] = (unsigned char)(uu < 0 ? 0 : uu > 255 ? 255 : uu);
            v[(size_t)j * cw + i] = (unsigned char)(vv < 0 ? 0 : vv > 255 ? 255 : vv);
        }
    }
}

// Encode `n` RGBA frames (frames[k] points to w*h*4 bytes) to an H.264/MP4 file.
// Dimensions are padded up to a multiple of 16 (the encoder's requirement), any
// padding left black. Returns 1 on success.
int zigai_encode_mp4(const char *path, const unsigned char *const *frames,
                     int n, int w, int h, int fps) {
    if (!path || !frames || n <= 0 || w <= 0 || h <= 0) return 0;
    if (fps <= 0) fps = 8;
    const int ew = (w + 15) & ~15;
    const int eh = (h + 15) & ~15;

    H264E_create_param_t cp;
    memset(&cp, 0, sizeof cp);
    cp.width = ew;
    cp.height = eh;
    cp.gop = 30;
    cp.const_input_flag = 1; // encoder won't modify our plane buffers
    cp.enableNEON = 1;
    const int desired_frame_bytes = ew * eh / 4; // ~2 bits/px → good quality
    cp.vbv_size_bytes = desired_frame_bytes * fps * 2;

    int sizeof_persist = 0, sizeof_scratch = 0;
    if (H264E_sizeof(&cp, &sizeof_persist, &sizeof_scratch)) return 0;

    int ok = 0;
    H264E_persist_t *enc = (H264E_persist_t *)malloc((size_t)sizeof_persist);
    H264E_scratch_t *scr = (H264E_scratch_t *)malloc((size_t)sizeof_scratch);
    unsigned char *y = (unsigned char *)malloc((size_t)ew * eh);
    unsigned char *u = (unsigned char *)malloc((size_t)(ew / 2) * (eh / 2));
    unsigned char *v = (unsigned char *)malloc((size_t)(ew / 2) * (eh / 2));
    membuf_t mb = {0};
    MP4E_mux_t *mux = NULL;
    mp4_h26x_writer_t mw;
    int writer_open = 0;

    if (!enc || !scr || !y || !u || !v) goto done;
    if (H264E_init(enc, &cp)) goto done;

    mux = MP4E_open(0 /*sequential*/, 0 /*fragmentation*/, &mb, membuf_write);
    if (!mux) goto done;
    if (mp4_h26x_write_init(&mw, mux, ew, eh, 0 /*is_hevc*/)) goto done;
    writer_open = 1;

    for (int f = 0; f < n; f++) {
        memset(y, 16, (size_t)ew * eh);
        memset(u, 128, (size_t)(ew / 2) * (eh / 2));
        memset(v, 128, (size_t)(ew / 2) * (eh / 2));
        fill_i420(frames[f], w, h, ew, y, u, v);

        H264E_io_yuv_t io;
        io.yuv[0] = y; io.stride[0] = ew;
        io.yuv[1] = u; io.stride[1] = ew / 2;
        io.yuv[2] = v; io.stride[2] = ew / 2;

        H264E_run_param_t rp;
        memset(&rp, 0, sizeof rp);
        rp.frame_type = 0;
        rp.encode_speed = 8; // fast; keeps the UI hitch small
        rp.desired_frame_bytes = desired_frame_bytes;
        rp.qp_min = 10;
        rp.qp_max = 50;

        unsigned char *coded = NULL;
        int coded_size = 0;
        if (H264E_encode(enc, scr, &rp, &io, &coded, &coded_size)) goto done;
        if (mp4_h26x_write_nal(&mw, coded, coded_size, 90000 / fps)) goto done;
    }
    ok = !mb.err;

done:
    if (writer_open) mp4_h26x_write_close(&mw);
    if (mux) MP4E_close(mux);
    if (ok && mb.data && mb.len) {
        FILE *fp = fopen(path, "wb");
        if (fp) {
            ok = (fwrite(mb.data, 1, mb.len, fp) == mb.len);
            fclose(fp);
        } else {
            ok = 0;
        }
    } else {
        ok = 0;
    }
    free(mb.data);
    free(y); free(u); free(v);
    free(enc); free(scr);
    return ok;
}
