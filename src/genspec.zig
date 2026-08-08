//! Shared image/video model-spec resolution: given a selected diffusion model
//! (path/name/dir), discover its sidecar files (VAE, text encoders, LTX audio
//! VAE / connectors / spatial upscaler) and build the backend ModelSpec. Used
//! by both the GUI (state.zig) and the HTTP API (server/api.zig) so the
//! per-family wiring lives in exactly one place.

const std = @import("std");
const models = @import("models.zig");
const sd = @import("backends/sd.zig");
const video = @import("backends/video.zig");

pub const Error = error{
    MissingVae,
    MissingEncoder,
    MissingLtxSidecars,
    OutOfMemory,
};

/// Default negative prompt for video (Wan's standard quality-suppression list).
/// Used by the GUI and the HTTP API when a request leaves the field blank —
/// an empty negative hurts Wan quality badly.
pub const default_video_negative =
    "low quality, worst quality, blurry, jpeg artifacts, overexposed, static, " ++
    "still image, watermark, subtitles, text, deformed, disfigured, extra limbs, " ++
    "fused fingers, messy background, ugly";

/// Default negative prompt for LTX video, verbatim from mlx-serve's
/// LTX_NEGATIVE_PROMPT (its server applies this for every LTX generation).
/// Unlike the Wan list it also suppresses audio artifacts, since LTX-2.3
/// generates a soundtrack.
pub const default_ltx_negative =
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out " ++
    "colors, excessive noise, grainy texture, poor lighting, flickering, motion " ++
    "blur, distorted proportions, unnatural skin tones, deformed facial features, " ++
    "asymmetrical face, missing facial features, extra limbs, disfigured hands, " ++
    "wrong hand count, artifacts around text, subtitles, closed captions, " ++
    "burned-in captions, on-screen text, text overlay, lower thirds, " ++
    "karaoke-style lyrics, watermark, inconsistent perspective, camera shake, " ++
    "incorrect depth of field, background too sharp, background clutter, " ++
    "distracting reflections, harsh shadows, inconsistent lighting direction, " ++
    "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, " ++
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated " ++
    "expressions, wrong gaze direction, mismatched lip sync, silent or muted " ++
    "audio, distorted voice, robotic voice, echo, background noise, off-sync " ++
    "audio, incorrect dialogue, added dialogue, repetitive speech, jittery " ++
    "movement, awkward pauses, incorrect timing, unnatural transitions, " ++
    "inconsistent framing, tilted camera, flat lighting, inconsistent tone, " ++
    "cinematic oversaturation, stylized filters, or AI artifacts.";

/// A short human-readable reason for a resolution failure (alert / HTTP 400).
pub fn message(e: Error) []const u8 {
    return switch (e) {
        error.MissingVae => "no VAE found next to the model (e.g. *vae*.safetensors)",
        error.MissingEncoder => "no text encoder found next to the model (Qwen/CLIP-L+T5-XXL for image; umt5/gemma/qwen for video)",
        error.MissingLtxSidecars => "LTX needs an audio-VAE + embeddings connectors next to the model",
        error.OutOfMemory => "out of memory",
    };
}

fn nameContains(name: []const u8, needle: []const u8) bool {
    return std.ascii.indexOfIgnoreCase(name, needle) != null;
}

/// An image ModelSpec plus the owned sidecar paths its slices point into.
/// `spec.model`/`spec.diffusion` borrow the caller's model path (which must
/// outlive the spec's use).
pub const ImageSpec = struct {
    gpa: std.mem.Allocator,
    spec: sd.ModelSpec,
    vae: ?[]u8 = null,
    enc1: ?[]u8 = null,
    enc2: ?[]u8 = null,

    pub fn deinit(self: *ImageSpec) void {
        if (self.vae) |s| self.gpa.free(s);
        if (self.enc1) |s| self.gpa.free(s);
        if (self.enc2) |s| self.gpa.free(s);
    }
};

/// Build the sd.ModelSpec for an image model. A classic SD checkpoint is one
/// self-contained file. Split models pair the diffusion file with a VAE and
/// text encoder(s) — Krea2 = Wan 2.1 VAE + Qwen3-VL LLM, FLUX.2 = FLUX VAE +
/// Qwen3 LLM, FLUX.1 = CLIP-L + T5-XXL — discovered as sidecars in `dir`.
pub fn resolveImage(gpa: std.mem.Allocator, path: []const u8, name: []const u8, dir: []const u8) Error!ImageSpec {
    var out: ImageSpec = .{ .gpa = gpa, .spec = .{ .model = path } };
    errdefer out.deinit();

    if (nameContains(name, "krea")) {
        out.vae = models.findSupport(gpa, dir, &.{ "vae", "ae." }, &.{ ".safetensors", ".gguf" }, &.{"audio"});
        if (out.vae == null) return error.MissingVae;
        out.enc1 = models.findSupport(gpa, dir, &.{"qwen"}, &.{ ".gguf", ".safetensors" }, &.{"mmproj"});
        if (out.enc1 == null) return error.MissingEncoder;
        out.spec = .{ .diffusion = path, .vae = out.vae, .llm = out.enc1 };
        return out;
    }
    if (nameContains(name, "flux")) {
        out.vae = models.findSupport(gpa, dir, &.{ "vae", "ae." }, &.{ ".safetensors", ".gguf" }, &.{"audio"});
        if (out.vae == null) return error.MissingVae;
        // FLUX.2 ships a Qwen3 LLM text encoder; FLUX.1 uses CLIP-L + T5-XXL.
        out.enc1 = models.findSupport(gpa, dir, &.{"qwen"}, &.{ ".gguf", ".safetensors" }, &.{"mmproj"});
        if (out.enc1 != null) {
            out.spec = .{ .diffusion = path, .vae = out.vae, .llm = out.enc1 };
            return out;
        }
        out.enc1 = models.findSupport(gpa, dir, &.{ "clip_l", "clip-l" }, &.{ ".safetensors", ".gguf" }, &.{});
        out.enc2 = models.findSupport(gpa, dir, &.{ "t5xxl", "t5-xxl" }, &.{ ".safetensors", ".gguf" }, &.{});
        if (out.enc1 == null or out.enc2 == null) return error.MissingEncoder;
        out.spec = .{ .diffusion = path, .vae = out.vae, .clip_l = out.enc1, .t5xxl = out.enc2 };
        return out;
    }
    return out; // single-file checkpoint
}

/// Video model family, detected from the sidecars present next to the
/// diffusion model. Drives per-family defaults (CFG, FPS, frame ladder,
/// flow shift, negative prompt) in the GUI and the HTTP API.
pub const Family = enum { wan, ltx, minimax };

/// A video ModelSpec plus the owned sidecar paths its slices point into.
pub const VideoSpec = struct {
    gpa: std.mem.Allocator,
    spec: video.ModelSpec,
    family: Family = .wan,
    vae: []u8,
    t5: ?[]u8 = null,
    gemma: ?[]u8 = null,
    qwen: ?[]u8 = null,
    avae: ?[]u8 = null,
    conn: ?[]u8 = null,
    upscaler: ?[]u8 = null,

    pub fn deinit(self: *VideoSpec) void {
        self.gpa.free(self.vae);
        if (self.t5) |s| self.gpa.free(s);
        if (self.gemma) |s| self.gpa.free(s);
        if (self.qwen) |s| self.gpa.free(s);
        if (self.avae) |s| self.gpa.free(s);
        if (self.conn) |s| self.gpa.free(s);
        if (self.upscaler) |s| self.gpa.free(s);
    }
};

/// Build the video.ModelSpec for a video model. Wan pairs the diffusion .gguf
/// with a VAE + umt5-xxl encoder; LTX with a video VAE + Gemma-3 LLM + audio
/// VAE + embeddings connectors (+ optional spatial upscaler for two-stage
/// hires); MiniMax-H3 with a video VAE + Qwen3-VL-32B LLM + audio VAE (audio
/// VAE optional — without it the joint model still generates video, silently).
/// The family is detected by which sidecars are present.
pub fn resolveVideo(gpa: std.mem.Allocator, path: []const u8, dir: []const u8) Error!VideoSpec {
    const vae = models.findSupport(gpa, dir, &.{"vae"}, &.{ ".safetensors", ".gguf" }, &.{"audio"}) orelse
        return error.MissingVae;
    var out: VideoSpec = .{
        .gpa = gpa,
        .vae = vae,
        .spec = .{ .diffusion = path, .vae = vae },
    };
    errdefer out.deinit();

    out.gemma = models.findSupport(gpa, dir, &.{"gemma"}, &.{".gguf"}, &.{});
    if (out.gemma) |g| {
        // LTX.
        out.family = .ltx;
        out.avae = models.findSupport(gpa, dir, &.{"audio_vae"}, &.{ ".safetensors", ".gguf" }, &.{});
        out.conn = models.findSupport(gpa, dir, &.{ "connector", "connectors" }, &.{ ".safetensors", ".gguf" }, &.{});
        if (out.avae == null or out.conn == null) return error.MissingLtxSidecars;
        out.upscaler = models.findSupport(gpa, dir, &.{ "upscaler", "upsampler" }, &.{ ".safetensors", ".gguf" }, &.{});
        out.spec.llm = g;
        out.spec.audio_vae = out.avae;
        out.spec.connectors = out.conn;
        out.spec.upscaler = out.upscaler;
        return out;
    }
    out.qwen = models.findSupport(gpa, dir, &.{"qwen"}, &.{".gguf"}, &.{"mmproj"});
    if (out.qwen) |q| {
        // MiniMax-H3.
        out.family = .minimax;
        out.avae = models.findSupport(gpa, dir, &.{"audio_vae"}, &.{ ".safetensors", ".gguf" }, &.{});
        out.spec.llm = q;
        out.spec.audio_vae = out.avae;
        return out;
    }
    // Wan: umt5/t5xxl text encoder.
    out.t5 = models.findSupport(gpa, dir, &.{ "umt5", "t5xxl", "t5-xxl" }, &.{".gguf"}, &.{});
    if (out.t5 == null) return error.MissingEncoder;
    out.spec.t5xxl = out.t5;
    return out;
}
