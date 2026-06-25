//! In-process stable-diffusion.cpp image backend. A worker thread owns the sd
//! context and runs txt2img; diffusion progress (step/total) and the final image
//! stream back to the UI through a `Channel` + `JobState` atomics.

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
};

/// How to load an image model. A classic Stable-Diffusion checkpoint is a single
/// self-contained file (`model`). FLUX-style models are split across files: the
/// diffusion weights plus a VAE and a text encoder — FLUX.2 uses an LLM (Qwen3),
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

const Request = struct {
    paths: ModelPaths,
    prompt: []u8,
    negative: []u8,
    params: Params,
};

/// The backend whose generation is currently running, so the global C progress
/// callback can reach it. Only one sd job runs at a time.
var g_active: ?*Backend = null;

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

    fn freeRequest(self: *Backend, req_opt: ?Request) void {
        const req = req_opt orelse return;
        self.freePaths(req.paths);
        self.gpa.free(req.prompt);
        self.gpa.free(req.negative);
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

    pub fn submit(self: *Backend, spec: ModelSpec, prompt: []const u8, negative: []const u8, params: Params) !void {
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
        // NOTE: we deliberately do NOT reuse a cached sd_ctx across generations.
        // Reusing it crashes on the 2nd generation — stable-diffusion.cpp/ggml hands
        // the new graph a tensor with a dangling Metal buffer, segfaulting in
        // ggml_metal_buffer_get_id during the text encoder. Recreating the context
        // each run (a model reload) is the reliable fix until that upstream reuse
        // bug is resolved. (`loaded` is still tracked for error messages / status.)
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
        // Single-file checkpoint vs. split (FLUX) model.
        if (paths.model) |s| cparams.model_path = zptr(a, s);
        if (paths.diffusion) |s| cparams.diffusion_model_path = zptr(a, s);
        if (paths.vae) |s| cparams.vae_path = zptr(a, s);
        if (paths.clip_l) |s| cparams.clip_l_path = zptr(a, s);
        if (paths.t5xxl) |s| cparams.t5xxl_path = zptr(a, s);
        if (paths.llm) |s| cparams.llm_path = zptr(a, s); // FLUX.2 Qwen3 text encoder
        cparams.n_threads = params.n_threads;
        cparams.vae_decode_only = true;

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
        gp.sample_params.sample_method = c.EULER_A_SAMPLE_METHOD;

        g_active = self;
        c.sd_set_progress_callback(progressCb, self);
        self.job.setProgress(0, req.params.steps);

        const images = c.generate_image(ctx, &gp);
        g_active = null;

        if (images == null or images[0].data == null) {
            self.emitErr("image generation failed", .{});
            return;
        }
        const img = images[0];
        const rgba = self.toRgba(img) catch {
            c.free(img.data);
            c.free(images);
            self.emitErr("out of memory converting image", .{});
            return;
        };
        c.free(img.data);
        c.free(images);

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
