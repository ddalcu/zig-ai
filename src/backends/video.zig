//! In-process stable-diffusion.cpp **video** backend (Wan 2.2). A worker thread
//! owns the sd context and runs txt2vid via `generate_video`; diffusion progress
//! (step/total) and the final frames stream back to the UI through a `Channel` +
//! `JobState` atomics.
//!
//! Unlike the image backend (single `model_path`), a Wan video model is loaded
//! from three separate files: the diffusion model, the VAE, and the umt5-xxl
//! (t5xxl) text encoder. See deps/stable-diffusion.cpp/docs/wan.md.

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
};

/// A set of model file paths identifying one Wan model. The trio is what makes a
/// video "model" loadable, so we key the cached context on all three.
/// A borrowed set of model file paths (caller owns the slices). `diffusion` and
/// `vae` are always required. Wan uses `t5xxl`; LTX uses `llm` (Gemma) +
/// `audio_vae` + `connectors`.
pub const ModelSpec = struct {
    diffusion: []const u8,
    vae: []const u8,
    t5xxl: ?[]const u8 = null,
    llm: ?[]const u8 = null,
    audio_vae: ?[]const u8 = null,
    connectors: ?[]const u8 = null,
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
};

pub const Event = union(enum) {
    progress: struct { step: i32, total: i32 },
    /// Decoded RGBA8 frames; UI owns `images` (and each `.pixels`) after receipt.
    frames: struct { images: []zigui.canvas.Image, fps: i32 },
    err: []u8,
};

const Request = struct {
    paths: ModelPaths,
    prompt: []u8,
    negative: []u8,
    params: Params,
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
    }

    fn freeRequest(self: *Backend, req_opt: ?Request) void {
        const req = req_opt orelse return;
        self.freePaths(req.paths);
        self.gpa.free(req.prompt);
        self.gpa.free(req.negative);
    }

    pub fn isBusy(self: *Backend) bool {
        return self.job.isRunning();
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
    ) !void {
        const paths = try self.dupePaths(spec);
        errdefer self.freePaths(paths);
        const pr = try self.gpa.dupe(u8, prompt);
        errdefer self.gpa.free(pr);
        const ng = try self.gpa.dupe(u8, negative);
        errdefer self.gpa.free(ng);

        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        self.freeRequest(self.request);
        self.request = .{ .paths = paths, .prompt = pr, .negative = ng, .params = params };
        self.has_request = true;
        self.job.beginJob();
        _ = pt.pthread_cond_signal(&self.cond);
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
            optEql(a.audio_vae, b.audio_vae) and optEql(a.connectors, b.connectors);
    }

    fn specOf(p: ModelPaths) ModelSpec {
        return .{
            .diffusion = p.diffusion, .vae = p.vae, .t5xxl = p.t5xxl,
            .llm = p.llm, .audio_vae = p.audio_vae, .connectors = p.connectors,
        };
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
        cparams.vae_decode_only = true; // txt2vid: we only decode latents to frames
        cparams.diffusion_flash_attn = false;
        // Video runs entirely on the GPU (Metal). Two local ggml/sd.cpp patches
        // make this possible against upstream ggml-org's Metal backend:
        //   1. sd.cpp ggml_ext_conv_3d → GGML_OP_CONV_3D (has a Metal kernel)
        //      instead of the IM2COL_3D decomposition (no Metal kernel).
        //   2. ggml Metal PAD kernel extended to support left/causal padding
        //      (the Wan/LTX VAE needs it); see ggml-metal.metal kernel_pad_impl.

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
        vp.prompt = prompt_z.ptr;
        vp.negative_prompt = neg_z.ptr;
        vp.width = req.params.width;
        vp.height = req.params.height;
        vp.seed = req.params.seed;
        vp.video_frames = req.params.frames;
        vp.fps = req.params.fps;
        vp.sample_params.sample_steps = req.params.steps;
        vp.sample_params.guidance.txt_cfg = req.params.cfg;
        vp.sample_params.sample_method = c.EULER_SAMPLE_METHOD;
        vp.sample_params.flow_shift = req.params.flow_shift;

        g_active = self;
        c.sd_set_log_callback(logCb, self);
        c.sd_set_progress_callback(progressCb, self);
        self.job.setProgress(0, req.params.steps);

        var frames_ptr: [*c]c.sd_image_t = null;
        var num_frames: c_int = 0;
        var audio_ptr: [*c]c.sd_audio_t = null;
        const ok = c.generate_video(ctx, &vp, &frames_ptr, &num_frames, &audio_ptr);
        g_active = null;

        if (audio_ptr != null) c.free_sd_audio(audio_ptr); // Wan has no audio track
        if (!ok or frames_ptr == null or num_frames <= 0) {
            if (frames_ptr != null) c.free(frames_ptr);
            self.emitErr("video generation failed", .{});
            return;
        }

        const n: usize = @intCast(num_frames);
        var images = self.gpa.alloc(zigui.canvas.Image, n) catch {
            self.freeCFrames(frames_ptr, n);
            c.free(frames_ptr);
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
        self.freeCFrames(frames_ptr, n);
        c.free(frames_ptr);

        if (made == 0) {
            self.gpa.free(images);
            self.emitErr("no frames decoded", .{});
            return;
        }
        if (made != n) images = self.gpa.realloc(images, made) catch images[0..made];

        self.events.push(.{ .frames = .{ .images = images, .fps = req.params.fps } });
    }

    fn freeCFrames(_: *Backend, frames_ptr: [*c]c.sd_image_t, n: usize) void {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (frames_ptr[i].data != null) c.free(frames_ptr[i].data);
        }
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
