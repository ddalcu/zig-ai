//! In-process stable-diffusion.cpp **video** backend (Wan 2.2 / LTX-2.3). A
//! worker thread owns the sd context and runs generate_video; diffusion progress
//! (step/total) and the final frames (+ optional audio track, LTX) stream back
//! to the UI through a `Channel` + `JobState` atomics.
//!
//! Unlike the image backend (single `model_path`), a video model is loaded from
//! separate files: the diffusion model, the VAE, and per-family sidecars — Wan
//! uses a umt5-xxl (t5xxl) text encoder; LTX uses a Gemma-3 LLM + audio VAE +
//! embeddings connectors (+ an optional spatial latent upscaler for the
//! two-stage hires pipeline). See deps/stable-diffusion.cpp/docs/{wan,ltx2}.md.

const std = @import("std");
const zigui = @import("zigui");
const channel = @import("../channel.zig");

pub const c = @cImport({
    @cInclude("stable-diffusion.h");
    @cInclude("stdlib.h"); // free()
});

const pt = @cImport({
    @cInclude("pthread.h");
});

pub const Params = struct {
    steps: i32 = 30,
    cfg: f32 = 6.0,
    flow_shift: f32 = 3.0,
    width: i32 = 480,
    height: i32 = 480,
    frames: i32 = 33,
    fps: i32 = 16,
    seed: i64 = -1,
    n_threads: i32 = 4,
    /// Spatio-temporal guidance (LTX "STG", sd.cpp "SLG") scale; 0 = off.
    slg_scale: f32 = 0,
    /// Multiplier for the optional LoRA passed to `submit`.
    lora_scale: f32 = 1.0,
    /// Two-stage hires pipeline: generate at `width`×`height`, upscale latents
    /// 2× through the model named in `ModelSpec.upscaler`, then refine for
    /// `hires_steps` at `hires_denoise` strength (LTX spatial upscaler flow).
    hires: bool = false,
    hires_steps: i32 = 4,
    hires_denoise: f32 = 0.7,
};

/// A borrowed set of model file paths (caller owns the slices). `diffusion` and
/// `vae` are always required. Wan uses `t5xxl`; LTX uses `llm` (Gemma) +
/// `audio_vae` + `connectors` (+ optional `upscaler` for two-stage hires).
pub const ModelSpec = struct {
    diffusion: []const u8,
    vae: []const u8,
    t5xxl: ?[]const u8 = null,
    llm: ?[]const u8 = null,
    audio_vae: ?[]const u8 = null,
    connectors: ?[]const u8 = null,
    upscaler: ?[]const u8 = null,
};

/// An owned copy of a ModelSpec (worker-thread lifetime), used as the context
/// cache key so we only reload when the file set actually changes.
const ModelPaths = struct {
    diffusion: []u8,
    vae: []u8,
    t5xxl: ?[]u8 = null,
    llm: ?[]u8 = null,
    audio_vae: ?[]u8 = null,
    connectors: ?[]u8 = null,
    upscaler: ?[]u8 = null,
};

/// An audio track that accompanied the generated frames (LTX models generate
/// sound). Interleaved f32 samples in [-1,1]; UI owns after receipt.
pub const Audio = struct { samples: []f32, sample_rate: u32, channels: u32 };

pub const Event = union(enum) {
    progress: struct { step: i32, total: i32 },
    /// Decoded RGBA8 frames; UI owns `images` (and each `.pixels`) after receipt.
    /// `audio` (LTX) is owned by the UI too.
    frames: struct { images: []zigui.canvas.Image, fps: i32, audio: ?Audio },
    err: []u8,
};

/// An optional frame for image-to-video (`init`, Wan TI2V / LTX I2V) or the
/// last frame (`end_image`, LTX FLF2V). `rgba` is owned (`width*height*4`
/// bytes); the worker converts it to packed RGB for sd.cpp.
pub const InitImage = struct { width: u32, height: u32, rgba: []u8 };

const Request = struct {
    paths: ModelPaths,
    prompt: []u8,
    negative: []u8,
    params: Params,
    init_image: ?InitImage = null,
    end_image: ?InitImage = null,
    lora_path: ?[]u8 = null,
};

/// The backend whose generation is currently running, so the global C progress
/// callback can reach it. Only one video job runs at a time.
var g_active: ?*Backend = null;

/// Forward sd.cpp / ggml log lines (incl. any "unsupported op" errors) to stderr
/// so failures are diagnosable. Only warnings+errors to keep it quiet.
fn logCb(level: c.sd_log_level_t, text: [*c]const u8, data: ?*anyopaque) callconv(.c) void {
    _ = data;
    if (level == c.SD_LOG_ERROR or level == c.SD_LOG_WARN) {
        std.debug.print("{s}", .{text});
    }
}

fn progressCb(step: c_int, steps: c_int, time: f32, data: ?*anyopaque) callconv(.c) void {
    _ = time;
    _ = data;
    const self = g_active orelse return;
    // sd.cpp routes BOTH diffusion sampling and the tiled VAE decode through this
    // one callback. Sampling counts 1..N monotonically; the decode restarts the
    // tile count at 0, so a step that moves backwards marks the decode phase.
    if (step < self.last_step) self.decoding.store(true, .release);
    self.last_step = step;
    self.job.setProgress(@intCast(step), @intCast(steps));
    self.events.push(.{ .progress = .{ .step = @intCast(step), .total = @intCast(steps) } });
}

pub const Backend = struct {
    gpa: std.mem.Allocator,
    events: channel.Channel(Event),
    job: channel.JobState = .{},

    thread: ?std.Thread = null,
    mutex: pt.pthread_mutex_t = undefined,
    cond: pt.pthread_cond_t = undefined,
    sync_ready: bool = false,
    shutdown: bool = false,
    has_request: bool = false,
    request: ?Request = null,

    // Owned by the worker thread.
    ctx: ?*c.sd_ctx_t = null,
    loaded: ?ModelPaths = null,

    /// Set true once a model is loaded; read by the UI (tray status, RAM proxy).
    model_ready: std.atomic.Value(bool) = .init(false),

    /// The ctx a generation is currently running on, published by the worker for
    /// `cancel()` (sd_cancel_generation only flips an atomic flag inside the
    /// ctx, so calling it from the UI thread is safe while the pointer is set).
    cancel_ctx: std.atomic.Value(?*c.sd_ctx_t) = .init(null),

    /// True while the worker is in the VAE-decode phase (vs diffusion sampling),
    /// so the UI labels the progress bar "Decoding". Written by `progressCb` on
    /// the worker thread; read by the UI thread.
    decoding: std.atomic.Value(bool) = .init(false),
    /// Last step `progressCb` saw (worker-thread only). The tiled decode restarts
    /// the count, so a backward move flags the sampling→decode transition.
    last_step: i32 = 0,

    pub fn init(gpa: std.mem.Allocator) Backend {
        return .{ .gpa = gpa, .events = channel.Channel(Event).init(gpa) };
    }

    pub fn start(self: *Backend) !void {
        if (self.thread != null) return;
        _ = pt.pthread_mutex_init(&self.mutex, null);
        _ = pt.pthread_cond_init(&self.cond, null);
        self.sync_ready = true;
        self.thread = try std.Thread.spawn(.{}, workerMain, .{self});
    }

    pub fn deinit(self: *Backend) void {
        if (self.thread) |th| {
            _ = pt.pthread_mutex_lock(&self.mutex);
            self.shutdown = true;
            _ = pt.pthread_cond_signal(&self.cond);
            _ = pt.pthread_mutex_unlock(&self.mutex);
            th.join();
        }
        if (self.sync_ready) {
            _ = pt.pthread_mutex_destroy(&self.mutex);
            _ = pt.pthread_cond_destroy(&self.cond);
        }
        if (self.ctx) |ctx| c.free_sd_ctx(ctx);
        self.freePaths(self.loaded);
        self.freeRequest(self.request);
        self.events.deinit();
    }

    fn freePaths(self: *Backend, p_opt: ?ModelPaths) void {
        const p = p_opt orelse return;
        self.gpa.free(p.diffusion);
        self.gpa.free(p.vae);
        if (p.t5xxl) |s| self.gpa.free(s);
        if (p.llm) |s| self.gpa.free(s);
        if (p.audio_vae) |s| self.gpa.free(s);
        if (p.connectors) |s| self.gpa.free(s);
        if (p.upscaler) |s| self.gpa.free(s);
    }

    fn freeRequest(self: *Backend, req_opt: ?Request) void {
        const req = req_opt orelse return;
        self.freePaths(req.paths);
        self.gpa.free(req.prompt);
        self.gpa.free(req.negative);
        if (req.init_image) |im| self.gpa.free(im.rgba);
        if (req.end_image) |im| self.gpa.free(im.rgba);
        if (req.lora_path) |s| self.gpa.free(s);
    }

    pub fn isBusy(self: *Backend) bool {
        return self.job.isRunning();
    }

    /// Ask sd.cpp to stop the running generation as soon as possible. Safe to
    /// call from the UI thread; a no-op when nothing is generating.
    pub fn cancel(self: *Backend) void {
        if (self.cancel_ctx.load(.acquire)) |ctx| {
            c.sd_cancel_generation(ctx, c.SD_CANCEL_ALL);
        }
    }

    /// Free the cached model to release memory (see `llama.Backend.unload` for the
    /// locking rationale). The next submit reloads on demand.
    pub fn unload(self: *Backend) void {
        if (self.thread == null) return;
        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        if (self.job.isRunning()) return;
        if (self.ctx) |ctx| {
            c.free_sd_ctx(ctx);
            self.ctx = null;
        }
        self.freePaths(self.loaded);
        self.loaded = null;
        self.model_ready.store(false, .release);
    }

    pub fn submit(
        self: *Backend,
        spec: ModelSpec,
        prompt: []const u8,
        negative: []const u8,
        params: Params,
        init_image: ?InitImage, // borrowed; duped here
        end_image: ?InitImage, // borrowed; duped here (LTX FLF2V)
        lora_path: ?[]const u8, // borrowed; duped here
    ) !void {
        const paths = try self.dupePaths(spec);
        errdefer self.freePaths(paths);
        const pr = try self.gpa.dupe(u8, prompt);
        errdefer self.gpa.free(pr);
        const ng = try self.gpa.dupe(u8, negative);
        errdefer self.gpa.free(ng);
        const init_img = try self.dupeImage(init_image);
        errdefer if (init_img) |im| self.gpa.free(im.rgba);
        const end_img = try self.dupeImage(end_image);
        errdefer if (end_img) |im| self.gpa.free(im.rgba);
        const lora = try self.dupeOpt(lora_path);
        errdefer if (lora) |s| self.gpa.free(s);

        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        self.freeRequest(self.request);
        self.request = .{
            .paths = paths,
            .prompt = pr,
            .negative = ng,
            .params = params,
            .init_image = init_img,
            .end_image = end_img,
            .lora_path = lora,
        };
        self.has_request = true;
        self.job.beginJob();
        _ = pt.pthread_cond_signal(&self.cond);
    }

    fn dupeImage(self: *Backend, im_opt: ?InitImage) !?InitImage {
        const im = im_opt orelse return null;
        return .{
            .width = im.width,
            .height = im.height,
            .rgba = try self.gpa.dupe(u8, im.rgba),
        };
    }

    fn dupeOpt(self: *Backend, s: ?[]const u8) !?[]u8 {
        return if (s) |v| try self.gpa.dupe(u8, v) else null;
    }

    fn dupePaths(self: *Backend, spec: ModelSpec) !ModelPaths {
        var p: ModelPaths = .{
            .diffusion = try self.gpa.dupe(u8, spec.diffusion),
            .vae = try self.gpa.dupe(u8, spec.vae),
        };
        errdefer self.freePaths(p);
        p.t5xxl = try self.dupeOpt(spec.t5xxl);
        p.llm = try self.dupeOpt(spec.llm);
        p.audio_vae = try self.dupeOpt(spec.audio_vae);
        p.connectors = try self.dupeOpt(spec.connectors);
        p.upscaler = try self.dupeOpt(spec.upscaler);
        return p;
    }

    fn workerMain(self: *Backend) void {
        while (true) {
            _ = pt.pthread_mutex_lock(&self.mutex);
            while (!self.has_request and !self.shutdown)
                _ = pt.pthread_cond_wait(&self.cond, &self.mutex);
            if (self.shutdown) {
                _ = pt.pthread_mutex_unlock(&self.mutex);
                return;
            }
            const req = self.request.?;
            self.request = null;
            self.has_request = false;
            _ = pt.pthread_mutex_unlock(&self.mutex);

            self.process(req);
            self.freeRequest(req);
            self.job.endJob();
        }
    }

    fn emitErr(self: *Backend, comptime f: []const u8, args: anytype) void {
        const msg = std.fmt.allocPrint(self.gpa, f, args) catch return;
        self.events.push(.{ .err = msg });
    }

    fn optEql(a: ?[]const u8, b: ?[]const u8) bool {
        if (a == null and b == null) return true;
        if (a == null or b == null) return false;
        return std.mem.eql(u8, a.?, b.?);
    }

    fn samePaths(a: ModelPaths, b: ModelPaths) bool {
        return std.mem.eql(u8, a.diffusion, b.diffusion) and
            std.mem.eql(u8, a.vae, b.vae) and
            optEql(a.t5xxl, b.t5xxl) and optEql(a.llm, b.llm) and
            optEql(a.audio_vae, b.audio_vae) and optEql(a.connectors, b.connectors) and
            optEql(a.upscaler, b.upscaler);
    }

    fn specOf(p: ModelPaths) ModelSpec {
        return .{
            .diffusion = p.diffusion,
            .vae = p.vae,
            .t5xxl = p.t5xxl,
            .llm = p.llm,
            .audio_vae = p.audio_vae,
            .connectors = p.connectors,
            .upscaler = p.upscaler,
        };
    }

    /// Free the loaded sd context and its (leaked) VRAM. The next `ensureCtx`
    /// reloads from disk. Safe to call when nothing is loaded.
    fn freeCtx(self: *Backend) void {
        if (self.ctx) |ctx| {
            c.free_sd_ctx(ctx);
            self.ctx = null;
        }
        self.freePaths(self.loaded);
        self.loaded = null;
        self.model_ready.store(false, .release);
    }

    fn ensureCtx(self: *Backend, paths: ModelPaths, params: Params) bool {
        if (self.loaded) |lp| {
            if (samePaths(lp, paths) and self.ctx != null) return true;
        }
        if (self.ctx) |ctx| {
            c.free_sd_ctx(ctx);
            self.ctx = null;
            self.model_ready.store(false, .release);
        }
        self.freePaths(self.loaded);
        self.loaded = null;

        // The *_path fields are only read during new_sd_ctx, so a scratch arena
        // for the NUL-terminated copies (freed right after) is enough.
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const zptr = struct {
            fn f(al: std.mem.Allocator, s: []const u8) [*c]const u8 {
                return (al.dupeZ(u8, s) catch return null).ptr;
            }
        }.f;

        var cparams: c.sd_ctx_params_t = undefined;
        c.sd_ctx_params_init(&cparams);
        cparams.diffusion_model_path = zptr(a, paths.diffusion);
        cparams.vae_path = zptr(a, paths.vae);
        if (paths.t5xxl) |s| cparams.t5xxl_path = zptr(a, s);
        if (paths.llm) |s| cparams.llm_path = zptr(a, s); // LTX Gemma-3 text encoder
        if (paths.audio_vae) |s| cparams.audio_vae_path = zptr(a, s); // LTX audio VAE
        if (paths.connectors) |s| cparams.embeddings_connectors_path = zptr(a, s); // LTX
        cparams.n_threads = params.n_threads;
        cparams.diffusion_flash_attn = false;

        const ctx = c.new_sd_ctx(&cparams);
        if (ctx == null) {
            self.emitErr("failed to load video model (diffusion={s})", .{paths.diffusion});
            return false;
        }
        self.ctx = ctx;
        self.loaded = self.dupePaths(specOf(paths)) catch null;
        self.model_ready.store(true, .release);
        return true;
    }

    fn process(self: *Backend, req: Request) void {
        if (!self.ensureCtx(req.paths, req.params)) return;
        const ctx = self.ctx.?;

        const prompt_z = self.gpa.dupeZ(u8, req.prompt) catch return;
        defer self.gpa.free(prompt_z);
        const neg_z = self.gpa.dupeZ(u8, req.negative) catch return;
        defer self.gpa.free(neg_z);

        var vp: c.sd_vid_gen_params_t = undefined;
        c.sd_vid_gen_params_init(&vp);
        // Decode the video VAE in overlapped spatial tiles. This gives the UI real
        // per-tile decode progress (sd.cpp's progress callback fires per tile) and
        // lowers peak memory vs decoding every frame in one monolithic graph. Left
        // as spatial-only (no temporal tiling) so the tile count is a single
        // monotonic 0→N pass rather than restarting per temporal chunk.
        vp.vae_tiling_params.enabled = true;
        vp.prompt = prompt_z.ptr;
        vp.negative_prompt = neg_z.ptr;
        vp.width = req.params.width;
        vp.height = req.params.height;
        vp.seed = req.params.seed;
        vp.video_frames = req.params.frames;
        vp.fps = req.params.fps;
        vp.sample_params.sample_steps = req.params.steps;
        vp.sample_params.guidance.txt_cfg = req.params.cfg;
        vp.sample_params.guidance.slg.scale = req.params.slg_scale;
        vp.sample_params.sample_method = c.EULER_SAMPLE_METHOD;
        vp.sample_params.flow_shift = req.params.flow_shift;

        // Two-stage hires (LTX spatial latent upscaler): low-res pass at
        // width×height, model-backed 2× latent upscale, short refine pass.
        var upscaler_z: ?[:0]u8 = null;
        defer if (upscaler_z) |s| self.gpa.free(s);
        if (req.params.hires) {
            if (req.paths.upscaler) |up| {
                if (self.gpa.dupeZ(u8, up)) |z| {
                    upscaler_z = z;
                    vp.hires.enabled = true;
                    vp.hires.upscaler = c.SD_HIRES_UPSCALER_MODEL;
                    vp.hires.model_path = z.ptr;
                    vp.hires.scale = 2.0;
                    vp.hires.steps = req.params.hires_steps;
                    vp.hires.denoising_strength = req.params.hires_denoise;
                } else |_| {}
            }
        }

        // Optional style LoRA.
        var lora_z: ?[:0]u8 = null;
        defer if (lora_z) |s| self.gpa.free(s);
        var lora_arr: [1]c.sd_lora_t = undefined;
        if (req.lora_path) |lp| {
            if (self.gpa.dupeZ(u8, lp)) |z| {
                lora_z = z;
                lora_arr[0] = .{ .is_high_noise = false, .multiplier = req.params.lora_scale, .path = z.ptr };
                vp.loras = &lora_arr;
                vp.lora_count = 1;
            } else |_| {}
        }

        // Image-to-video (init) and first/last-frame (end, LTX FLF2V): sd.cpp
        // wants packed RGB (3 channels), so drop the alpha from our RGBA buffers.
        var init_rgb: ?[]u8 = null;
        defer if (init_rgb) |buf| self.gpa.free(buf);
        if (req.init_image) |im| {
            if (rgbaToRgb(self.gpa, im.rgba, im.width, im.height)) |rgb| {
                vp.init_image = .{ .width = im.width, .height = im.height, .channel = 3, .data = rgb.ptr };
                init_rgb = rgb;
            }
        }
        var end_rgb: ?[]u8 = null;
        defer if (end_rgb) |buf| self.gpa.free(buf);
        if (req.end_image) |im| {
            if (rgbaToRgb(self.gpa, im.rgba, im.width, im.height)) |rgb| {
                vp.end_image = .{ .width = im.width, .height = im.height, .channel = 3, .data = rgb.ptr };
                end_rgb = rgb;
            }
        }

        g_active = self;
        c.sd_set_log_callback(logCb, self);
        c.sd_set_progress_callback(progressCb, self);
        self.job.setProgress(0, req.params.steps);
        self.last_step = 0;
        self.decoding.store(false, .release);
        self.cancel_ctx.store(ctx, .release);

        var frames_ptr: [*c]c.sd_image_t = null;
        var num_frames: c_int = 0;
        var audio_ptr: [*c]c.sd_audio_t = null;
        const ok = c.generate_video(ctx, &vp, &frames_ptr, &num_frames, &audio_ptr);
        self.cancel_ctx.store(null, .release);
        c.sd_cancel_generation(ctx, c.SD_CANCEL_RESET);
        g_active = null;
        self.decoding.store(false, .release);

        // Collect the audio track (LTX generates sound; Wan returns none) before
        // any early-out so it can't leak.
        var audio: ?Audio = null;
        if (audio_ptr != null) {
            audio = self.collectAudio(audio_ptr[0]);
            c.free_sd_audio(audio_ptr);
        }

        if (!ok or frames_ptr == null or num_frames <= 0) {
            if (frames_ptr != null) c.free_sd_images(frames_ptr, num_frames);
            if (audio) |au| self.gpa.free(au.samples);
            self.emitErr("video generation failed", .{});
            return;
        }

        const n: usize = @intCast(num_frames);
        var images = self.gpa.alloc(zigui.canvas.Image, n) catch {
            c.free_sd_images(frames_ptr, num_frames);
            if (audio) |au| self.gpa.free(au.samples);
            self.emitErr("out of memory collecting frames", .{});
            return;
        };
        var made: usize = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const fr = frames_ptr[i];
            if (fr.data == null) continue;
            const rgba = self.toRgba(fr) catch break;
            images[made] = .{ .width = fr.width, .height = fr.height, .pixels = rgba };
            made += 1;
        }
        c.free_sd_images(frames_ptr, num_frames);

        if (made == 0) {
            self.gpa.free(images);
            if (audio) |au| self.gpa.free(au.samples);
            self.emitErr("no frames decoded", .{});
            return;
        }
        if (made != n) images = self.gpa.realloc(images, made) catch images[0..made];

        self.events.push(.{ .frames = .{ .images = images, .fps = req.params.fps, .audio = audio } });
    }

    /// Copy an sd_audio_t into an owned `Audio` (null when empty or OOM).
    fn collectAudio(self: *Backend, au: c.sd_audio_t) ?Audio {
        if (au.data == null or au.sample_count == 0 or au.channels == 0) return null;
        const total: usize = @intCast(au.sample_count * au.channels);
        const samples = self.gpa.dupe(f32, au.data[0..total]) catch return null;
        return .{ .samples = samples, .sample_rate = au.sample_rate, .channels = au.channels };
    }

    /// Convert an sd_image_t (RGB or RGBA) to a zigui RGBA8 pixel buffer.
    fn toRgba(self: *Backend, img: c.sd_image_t) ![]u8 {
        const w = img.width;
        const h = img.height;
        const ch = img.channel;
        const out = try self.gpa.alloc(u8, @as(usize, w) * @as(usize, h) * 4);
        const src = img.data;
        var i: usize = 0;
        const n: usize = @as(usize, w) * @as(usize, h);
        while (i < n) : (i += 1) {
            const so = i * ch;
            const do = i * 4;
            out[do + 0] = src[so + 0];
            out[do + 1] = if (ch >= 2) src[so + 1] else src[so + 0];
            out[do + 2] = if (ch >= 3) src[so + 2] else src[so + 0];
            out[do + 3] = if (ch >= 4) src[so + 3] else 255;
        }
        return out;
    }
};

/// Convert a tightly-packed RGBA8 buffer to packed RGB (drops alpha).
fn rgbaToRgb(gpa: std.mem.Allocator, rgba: []const u8, width: u32, height: u32) ?[]u8 {
    const px: usize = @as(usize, width) * @as(usize, height);
    if (rgba.len < px * 4) return null;
    const rgb = gpa.alloc(u8, px * 3) catch return null;
    var i: usize = 0;
    while (i < px) : (i += 1) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return rgb;
}
