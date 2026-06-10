//! In-process qwen3-tts.cpp text-to-speech backend. A worker thread owns the TTS
//! engine and runs synthesis; the resulting PCM (float32 mono @ 24 kHz) streams
//! back to the UI through a `Channel`, which hands it to SDL audio for playback.
//!
//! Note: qwen3-tts loads a *directory* containing `qwen3-tts-0.6b-f16.gguf` and
//! `qwen3-tts-tokenizer-f16.gguf`, not a single file — so the model selection
//! passes the containing folder.

const std = @import("std");
const channel = @import("../channel.zig");

pub const c = @cImport({
    @cInclude("qwen3tts_c_api.h");
});

const pt = @cImport({
    @cInclude("pthread.h");
});

pub const Params = struct {
    temperature: f32 = 0.9,
    n_threads: i32 = 4,
    use_gpu: bool = true,
};

pub const Event = union(enum) {
    audio: struct { samples: []f32, sample_rate: i32 }, // UI owns samples
    err: []u8,
};

/// Voice-clone reference: none (default voice), a WAV file on disk (any rate;
/// qwen3-tts resamples), or raw samples (must be 24 kHz mono f32 in [-1,1]).
pub const Ref = union(enum) {
    none,
    file: []const u8,
    samples: []const f32,
};

const Request = struct {
    model_dir: []u8,
    text: []u8,
    params: Params,
    ref_path: ?[]u8 = null,
    ref_samples: ?[]f32 = null,
};

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
    engine: ?*c.Qwen3Tts = null,
    loaded_dir: ?[]u8 = null,

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
        if (self.engine) |e| c.qwen3_tts_destroy(e);
        if (self.loaded_dir) |p| self.gpa.free(p);
        self.freeRequest(self.request);
        self.events.deinit();
    }

    fn freeRequest(self: *Backend, req_opt: ?Request) void {
        const req = req_opt orelse return;
        self.gpa.free(req.model_dir);
        self.gpa.free(req.text);
        if (req.ref_path) |p| self.gpa.free(p);
        if (req.ref_samples) |s| self.gpa.free(s);
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
        if (self.engine) |e| {
            c.qwen3_tts_destroy(e);
            self.engine = null;
        }
        if (self.loaded_dir) |p| {
            self.gpa.free(p);
            self.loaded_dir = null;
        }
        self.model_ready.store(false, .release);
    }

    pub fn submit(self: *Backend, model_dir: []const u8, text: []const u8, ref: Ref, params: Params) !void {
        const md = try self.gpa.dupe(u8, model_dir);
        errdefer self.gpa.free(md);
        const tx = try self.gpa.dupe(u8, text);
        errdefer self.gpa.free(tx);
        const rp: ?[]u8 = switch (ref) {
            .file => |p| try self.gpa.dupe(u8, p),
            else => null,
        };
        errdefer if (rp) |p| self.gpa.free(p);
        const rs: ?[]f32 = switch (ref) {
            .samples => |s| try self.gpa.dupe(f32, s),
            else => null,
        };

        _ = pt.pthread_mutex_lock(&self.mutex);
        defer _ = pt.pthread_mutex_unlock(&self.mutex);
        self.freeRequest(self.request);
        self.request = .{ .model_dir = md, .text = tx, .params = params, .ref_path = rp, .ref_samples = rs };
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

    fn ensureEngine(self: *Backend, dir: []const u8, params: Params) bool {
        if (self.loaded_dir) |ld| {
            if (std.mem.eql(u8, ld, dir) and self.engine != null) return true;
        }
        if (self.engine) |e| {
            c.qwen3_tts_destroy(e);
            self.engine = null;
            self.model_ready.store(false, .release);
        }
        if (self.loaded_dir) |ld| {
            self.gpa.free(ld);
            self.loaded_dir = null;
        }
        const dir_z = self.gpa.dupeZ(u8, dir) catch return false;
        defer self.gpa.free(dir_z);

        // Upstream qwen3-tts.cpp uses GPU automatically when ggml has a GPU
        // backend (Metal); its create() takes only (model_dir, n_threads).
        const engine = c.qwen3_tts_create(dir_z.ptr, params.n_threads);
        if (engine == null) {
            self.emitErr("failed to load TTS model from: {s}", .{dir});
            return false;
        }
        self.engine = engine;
        self.loaded_dir = self.gpa.dupe(u8, dir) catch null;
        self.model_ready.store(true, .release);
        return true;
    }

    fn process(self: *Backend, req: Request) void {
        if (!self.ensureEngine(req.model_dir, req.params)) return;
        const engine = self.engine.?;

        const text_z = self.gpa.dupeZ(u8, req.text) catch return;
        defer self.gpa.free(text_z);

        var tparams: c.Qwen3TtsParams = undefined;
        c.qwen3_tts_default_params(&tparams);
        tparams.temperature = req.params.temperature;
        tparams.n_threads = req.params.n_threads;

        const audio = if (req.ref_path) |p| blk: {
            const pz = self.gpa.dupeZ(u8, p) catch return;
            defer self.gpa.free(pz);
            break :blk c.qwen3_tts_synthesize_with_voice_file(engine, text_z.ptr, pz.ptr, &tparams);
        } else if (req.ref_samples) |s|
            c.qwen3_tts_synthesize_with_voice_samples(engine, text_z.ptr, s.ptr, @intCast(s.len), &tparams)
        else
            c.qwen3_tts_synthesize(engine, text_z.ptr, &tparams);
        if (audio == null) {
            const err = c.qwen3_tts_get_error(engine);
            self.emitErr("synthesis failed: {s}", .{err});
            return;
        }
        defer c.qwen3_tts_free_audio(audio);

        const n: usize = @intCast(audio.*.n_samples);
        const samples = self.gpa.alloc(f32, n) catch {
            self.emitErr("out of memory copying audio", .{});
            return;
        };
        @memcpy(samples, audio.*.samples[0..n]);
        self.events.push(.{ .audio = .{ .samples = samples, .sample_rate = audio.*.sample_rate } });
    }
};
