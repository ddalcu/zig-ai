//! In-process stable-diffusion.cpp image backend. A worker thread owns the sd
//! context and runs txt2img/img2img; diffusion progress (step/total) and the
//! final image stream back to the UI through a `Channel` + `JobState` atomics.

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
    steps: i32 = 20,
    cfg: f32 = 7.0,
    width: i32 = 512,
    height: i32 = 512,
    seed: i64 = -1,
    n_threads: i32 = 4,
    /// img2img renoise strength in (0,1]; only used when an init image is set.
    strength: f32 = 0.6,
    /// Multiplier for the optional LoRA passed to `submit`.
    lora_scale: f32 = 1.0,
};

/// How to load an image model. A classic Stable-Diffusion checkpoint is a single
/// self-contained file (`model`). Split models pair the diffusion weights with a
/// VAE and text encoder(s) — FLUX.2 and Krea2 use an LLM (Qwen3 / Qwen3-VL),
/// FLUX.1 uses CLIP-L + T5-XXL. Mirrors `video.ModelSpec`.
pub const ModelSpec = struct {
    model: ?[]const u8 = null,
    diffusion: ?[]const u8 = null,
    vae: ?[]const u8 = null,
    clip_l: ?[]const u8 = null,
    t5xxl: ?[]const u8 = null,
    llm: ?[]const u8 = null,
};

/// An owned copy of a ModelSpec (worker-thread lifetime), used as the context
/// cache key so we only reload when the file set actually changes.
const ModelPaths = struct {
    model: ?[]u8 = null,
    diffusion: ?[]u8 = null,
    vae: ?[]u8 = null,
    clip_l: ?[]u8 = null,
    t5xxl: ?[]u8 = null,
    llm: ?[]u8 = null,

    /// The primary file, for status/error messages.
    fn primary(self: ModelPaths) []const u8 {
        return self.model orelse self.diffusion orelse "(none)";
    }
};

pub const Event = union(enum) {
    progress: struct { step: i32, total: i32 },
    image: zigui.canvas.Image, // RGBA8; UI owns pixels after receiving
    err: []u8,
};

/// An optional source image for image-to-image (SDEdit "variation": renoise at
/// `Params.strength`). `rgba` is owned (`width*height*4` bytes); the worker
/// converts it to packed RGB for sd.cpp.
pub const InitImage = struct { width: u32, height: u32, rgba: []u8 };

const Request = struct {
    paths: ModelPaths,
    prompt: []u8,
    negative: []u8,
    params: Params,
    init_image: ?InitImage = null,
    lora_path: ?[]u8 = null,
};

/// The backend whose generation is currently running, so the global C progress
/// callback can reach it. Only one sd job runs at a time.
var g_active: ?*Backend = null;

/// Forward sd.cpp / ggml warnings+errors to stderr so failures are diagnosable.
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

    /// The ctx a generation is currently running on, published by the worker for
    /// `cancel()` (sd_cancel_generation only flips an atomic flag inside the
    /// ctx, so calling it from the UI thread is safe while the pointer is set).
    cancel_ctx: std.atomic.Value(?*c.sd_ctx_t) = .init(null),

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

    fn freeRequest(self: *Backend, req_opt: ?Request) void {
        const req = req_opt orelse return;
        self.freePaths(req.paths);
        self.gpa.free(req.prompt);
        self.gpa.free(req.negative);
        if (req.init_image) |im| self.gpa.free(im.rgba);
        if (req.lora_path) |s| self.gpa.free(s);
    }

    fn freePaths(self: *Backend, p_opt: ?ModelPaths) void {
        const p = p_opt orelse return;
        if (p.model) |s| self.gpa.free(s);
        if (p.diffusion) |s| self.gpa.free(s);
        if (p.vae) |s| self.gpa.free(s);
        if (p.clip_l) |s| self.gpa.free(s);
        if (p.t5xxl) |s| self.gpa.free(s);
        if (p.llm) |s| self.gpa.free(s);
    }

    fn dupeOpt(self: *Backend, s: ?[]const u8) !?[]u8 {
        return if (s) |v| try self.gpa.dupe(u8, v) else null;
    }

    fn dupePaths(self: *Backend, spec: ModelSpec) !ModelPaths {
        var p: ModelPaths = .{};
        errdefer self.freePaths(p);
        p.model = try self.dupeOpt(spec.model);
        p.diffusion = try self.dupeOpt(spec.diffusion);
        p.vae = try self.dupeOpt(spec.vae);
        p.clip_l = try self.dupeOpt(spec.clip_l);
        p.t5xxl = try self.dupeOpt(spec.t5xxl);
        p.llm = try self.dupeOpt(spec.llm);
        return p;
    }

    fn optEql(a: ?[]const u8, b: ?[]const u8) bool {
        if (a == null and b == null) return true;
        if (a == null or b == null) return false;
        return std.mem.eql(u8, a.?, b.?);
    }

    fn samePaths(a: ModelPaths, b: ModelPaths) bool {
        return optEql(a.model, b.model) and optEql(a.diffusion, b.diffusion) and
            optEql(a.vae, b.vae) and optEql(a.clip_l, b.clip_l) and
            optEql(a.t5xxl, b.t5xxl) and optEql(a.llm, b.llm);
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
        lora_path: ?[]const u8, // borrowed; duped here
    ) !void {
        const paths = try self.dupePaths(spec);
        errdefer self.freePaths(paths);
        const pr = try self.gpa.dupe(u8, prompt);
        errdefer self.gpa.free(pr);
        const ng = try self.gpa.dupe(u8, negative);
        errdefer self.gpa.free(ng);
        const init_img: ?InitImage = if (init_image) |im| .{
            .width = im.width,
            .height = im.height,
            .rgba = try self.gpa.dupe(u8, im.rgba),
        } else null;
        errdefer if (init_img) |im| self.gpa.free(im.rgba);
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
            .lora_path = lora,
        };
        self.has_request = true;
        self.job.beginJob();
        _ = pt.pthread_cond_signal(&self.cond);
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

    fn ensureCtx(self: *Backend, paths: ModelPaths, params: Params) bool {
        // Reuse the loaded ctx when the file set is unchanged: sd.cpp keeps model
        // params resident, so back-to-back generations skip the (multi-GB) reload.
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
        // for the NUL-terminated copies (freed right after) suffices.
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
        // Single-file checkpoint vs. split (FLUX/Krea2) model.
        if (paths.model) |s| cparams.model_path = zptr(a, s);
        if (paths.diffusion) |s| cparams.diffusion_model_path = zptr(a, s);
        if (paths.vae) |s| cparams.vae_path = zptr(a, s);
        if (paths.clip_l) |s| cparams.clip_l_path = zptr(a, s);
        if (paths.t5xxl) |s| cparams.t5xxl_path = zptr(a, s);
        if (paths.llm) |s| cparams.llm_path = zptr(a, s); // FLUX.2 Qwen3 / Krea2 Qwen3-VL
        cparams.n_threads = params.n_threads;

        const ctx = c.new_sd_ctx(&cparams);
        if (ctx == null) {
            self.emitErr("failed to load image model: {s}", .{paths.primary()});
            return false;
        }
        self.ctx = ctx;
        self.loaded = self.dupePaths(specOf(paths)) catch null;
        self.model_ready.store(true, .release);
        return true;
    }

    fn specOf(p: ModelPaths) ModelSpec {
        return .{ .model = p.model, .diffusion = p.diffusion, .vae = p.vae, .clip_l = p.clip_l, .t5xxl = p.t5xxl, .llm = p.llm };
    }

    fn process(self: *Backend, req: Request) void {
        if (!self.ensureCtx(req.paths, req.params)) return;
        const ctx = self.ctx.?;

        const prompt_z = self.gpa.dupeZ(u8, req.prompt) catch return;
        defer self.gpa.free(prompt_z);
        const neg_z = self.gpa.dupeZ(u8, req.negative) catch return;
        defer self.gpa.free(neg_z);

        var gp: c.sd_img_gen_params_t = undefined;
        c.sd_img_gen_params_init(&gp);
        gp.prompt = prompt_z.ptr;
        gp.negative_prompt = neg_z.ptr;
        gp.width = req.params.width;
        gp.height = req.params.height;
        gp.seed = req.params.seed;
        gp.sample_params.sample_steps = req.params.steps;
        gp.sample_params.guidance.txt_cfg = req.params.cfg;
        // sample_method/scheduler stay at the init sentinels: sd.cpp then picks the
        // model's own defaults (flow models get flow-Euler; SD gets euler_a…).

        // Image-to-image ("variation"): renoise the source at `strength`. sd.cpp
        // wants packed RGB, so drop the alpha from our RGBA buffer.
        var init_rgb: ?[]u8 = null;
        defer if (init_rgb) |buf| self.gpa.free(buf);
        if (req.init_image) |im| {
            if (rgbaToRgb(self.gpa, im.rgba, im.width, im.height)) |rgb| {
                gp.init_image = .{ .width = im.width, .height = im.height, .channel = 3, .data = rgb.ptr };
                gp.strength = req.params.strength;
                init_rgb = rgb;
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
                gp.loras = &lora_arr;
                gp.lora_count = 1;
            } else |_| {}
        }

        g_active = self;
        c.sd_set_log_callback(logCb, self);
        c.sd_set_progress_callback(progressCb, self);
        self.job.setProgress(0, req.params.steps);
        self.cancel_ctx.store(ctx, .release);

        var images: [*c]c.sd_image_t = null;
        var num_images: c_int = 0;
        const ok = c.generate_image(ctx, &gp, &images, &num_images);
        self.cancel_ctx.store(null, .release);
        // Clear any pending cancel flag so it can't bleed into the next run.
        c.sd_cancel_generation(ctx, c.SD_CANCEL_RESET);
        g_active = null;

        if (!ok or images == null or num_images <= 0 or images[0].data == null) {
            if (images != null) c.free_sd_images(images, num_images);
            self.emitErr("image generation failed", .{});
            return;
        }
        const img = images[0];
        const rgba = self.toRgba(img) catch {
            c.free_sd_images(images, num_images);
            self.emitErr("out of memory converting image", .{});
            return;
        };
        c.free_sd_images(images, num_images);

        self.events.push(.{ .image = .{
            .width = img.width,
            .height = img.height,
            .pixels = rgba,
        } });
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
