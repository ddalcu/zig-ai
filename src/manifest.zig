//! Curated sidecar manifest for split models. Some image/video models can't run
//! from a single repo — FLUX needs a VAE + text encoder, Wan needs a VAE + umt5
//! encoder — and those live in *separate* HuggingFace repos. When the user
//! downloads such a model, the downloader also pulls these sidecars into the same
//! model folder (so the backend's `findSupport` discovers them beside the
//! diffusion weights and the model is runnable immediately).
//!
//! `dest` renames a sidecar on disk when its source name would otherwise be
//! mis-scanned — e.g. FLUX's `Qwen3-4B-*.gguf` would look like a standalone chat
//! model, so we save it as `…-text-encoder.gguf` (which `classifyName` treats as
//! a support file, while `findSupport`'s "qwen" needle still matches it).

const std = @import("std");

pub const Sidecar = struct {
    /// HuggingFace repo id the file comes from.
    repo: []const u8,
    /// Path of the file within that repo.
    file: []const u8,
    /// Optional on-disk basename override (default: the file's basename).
    dest: ?[]const u8 = null,
    /// Human description shown in the download UI.
    label: []const u8,
};

pub const Entry = struct {
    /// Case-insensitive substring matched against the model's repo id.
    match: []const u8,
    sidecars: []const Sidecar,
};

/// Verified against the node-omni download set (same backends).
pub const entries = [_]Entry{
    .{
        .match = "flux.2-klein",
        .sidecars = &.{
            .{
                .repo = "unsloth/Qwen3-4B-GGUF",
                .file = "Qwen3-4B-Q4_K_M.gguf",
                .dest = "flux2-qwen3-text-encoder.gguf",
                .label = "Qwen3-4B text encoder (~2.2 GB)",
            },
            .{
                .repo = "Comfy-Org/flux2-dev",
                .file = "split_files/vae/flux2-vae.safetensors",
                .label = "FLUX.2 VAE (~0.3 GB)",
            },
        },
    },
    .{
        .match = "wan2.2-ti2v",
        .sidecars = &.{
            .{
                .repo = "city96/umt5-xxl-encoder-gguf",
                .file = "umt5-xxl-encoder-Q5_K_M.gguf",
                .label = "umt5-xxl text encoder (~4 GB)",
            },
            .{
                .repo = "QuantStack/Wan2.2-TI2V-5B-GGUF",
                .file = "VAE/Wan2.2_VAE.safetensors",
                .label = "Wan 2.2 VAE (~0.3 GB)",
            },
        },
    },
};

/// The sidecars a given repo needs (empty if it's self-contained).
pub fn sidecarsFor(repo_id: []const u8) []const Sidecar {
    for (entries) |e| {
        if (std.ascii.indexOfIgnoreCase(repo_id, e.match) != null) return e.sidecars;
    }
    return &.{};
}

const models = @import("models.zig");

/// A curated, one-tap model bundle for the Download tab. Some models (LTX in
/// particular) aren't a single repo or even a single top-level file: LTX's
/// diffusion quant, video/audio VAEs and connectors live in *subfolders* of one
/// repo (which the normal top-level tree listing can't reach), and its text
/// encoder is a Gemma model in a *different* repo. A `Recommended` entry names
/// the exact files so "Get" downloads a known-good, runnable set into one folder.
pub const Recommended = struct {
    kind: models.Kind,
    /// Short title shown on the card (also the download's display name).
    title: []const u8,
    /// One-line description of what gets pulled.
    note: []const u8,
    /// Destination folder is named after this repo ("author/name").
    repo: []const u8,
    /// Every file in the bundle (each may come from a different repo and may be
    /// renamed on disk via `dest`). Downloaded together into one folder. Using
    /// `Sidecar` per item gives uniform cross-repo + rename support — including
    /// the renames some backends require (qwen3-tts.cpp loads fixed filenames).
    items: []const Sidecar,
};

/// Curated bundles, verified against the configs in README.md and the upstream
/// loaders. Each file path/dest was checked against the live HuggingFace trees.
pub const recommended = [_]Recommended{
    .{
        .kind = .video,
        .title = "LTX-2.3 (distilled 1.1)",
        .note = "22B video · diffusion + video/audio VAE + connectors + Gemma-3 encoder",
        .repo = "unsloth/LTX-2.3-GGUF",
        .items = &.{
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_S.gguf", .label = "LTX-2.3 diffusion (Q3_K_S)" },
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "vae/ltx-2.3-22b-distilled_video_vae.safetensors", .label = "LTX video VAE" },
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "vae/ltx-2.3-22b-distilled_audio_vae.safetensors", .label = "LTX audio VAE" },
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors", .label = "LTX connectors" },
            // Keep "gemma" (the video backend's findSupport locates it) but add
            // "encoder" so the scanner treats it as support, not a chat model.
            .{ .repo = "unsloth/gemma-3-12b-it-GGUF", .file = "gemma-3-12b-it-Q4_K_M.gguf", .dest = "gemma-3-12b-it-text-encoder.gguf", .label = "Gemma-3 12B text encoder (~7 GB)" },
        },
    },
    .{
        .kind = .video,
        .title = "Wan 2.2 TI2V (5B)",
        .note = "5B text+image-to-video · fits 16 GB · diffusion + VAE + umt5-xxl encoder",
        .repo = "QuantStack/Wan2.2-TI2V-5B-GGUF",
        .items = &.{
            .{ .repo = "QuantStack/Wan2.2-TI2V-5B-GGUF", .file = "Wan2.2-TI2V-5B-Q5_K_M.gguf", .label = "Wan 2.2 TI2V 5B diffusion (Q5_K_M)" },
            .{ .repo = "QuantStack/Wan2.2-TI2V-5B-GGUF", .file = "VAE/Wan2.2_VAE.safetensors", .label = "Wan 2.2 VAE" },
            .{ .repo = "city96/umt5-xxl-encoder-gguf", .file = "umt5-xxl-encoder-Q5_K_M.gguf", .label = "umt5-xxl text encoder (~4 GB)" },
        },
    },
    .{
        .kind = .image,
        .title = "FLUX.2 klein (4B)",
        .note = "4B diffusion + Qwen3 text encoder + VAE",
        .repo = "unsloth/FLUX.2-klein-4B-GGUF",
        .items = &.{
            .{ .repo = "unsloth/FLUX.2-klein-4B-GGUF", .file = "flux-2-klein-4b-Q4_K_M.gguf", .label = "FLUX.2 klein diffusion (Q4_K_M)" },
            // Same sidecars as the `flux.2-klein` entry above (renamed encoder so
            // the scanner doesn't list it as a chat model).
            .{ .repo = "unsloth/Qwen3-4B-GGUF", .file = "Qwen3-4B-Q4_K_M.gguf", .dest = "flux2-qwen3-text-encoder.gguf", .label = "Qwen3-4B text encoder (~2.2 GB)" },
            .{ .repo = "Comfy-Org/flux2-dev", .file = "split_files/vae/flux2-vae.safetensors", .label = "FLUX.2 VAE (~0.3 GB)" },
        },
    },
    .{
        .kind = .tts,
        .title = "Qwen3-TTS 0.6B",
        .note = "0.6B talker + vocoder · supports voice cloning",
        // This repo is converted *with qwen3-tts.cpp's own tooling* (note the
        // name), so its GGUFs use the exact tensor layout + filenames the vendored
        // loader needs — community conversions (different tensor names) fail right
        // after the text tokenizer loads. Files already match, so no rename.
        .repo = "Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF",
        .items = &.{
            .{ .repo = "Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF", .file = "qwen3-tts-0.6b-f16.gguf", .label = "Qwen3-TTS talker (0.6B F16)" },
            .{ .repo = "Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF", .file = "qwen3-tts-tokenizer-f16.gguf", .label = "Qwen3-TTS vocoder" },
        },
    },
};

test "sidecarsFor matches FLUX.2 klein by repo id" {
    const s = sidecarsFor("unsloth/FLUX.2-klein-4B-GGUF");
    try std.testing.expectEqual(@as(usize, 2), s.len);
    try std.testing.expect(sidecarsFor("some/Llama-3-GGUF").len == 0);
}
