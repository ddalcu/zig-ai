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
        // Krea2 (Raw/Turbo): Krea2 DiT + Wan 2.1 VAE + Qwen3-VL-4B LLM encoder
        // (see deps/stable-diffusion.cpp/docs/krea2.md). The encoder is renamed
        // so the scanner treats it as a support file ("encoder") while
        // `findSupport`'s "qwen" needle still matches it.
        .match = "krea-2",
        .sidecars = &.{
            .{
                .repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                .file = "split_files/vae/wan_2.1_vae.safetensors",
                .label = "Wan 2.1 VAE (~0.3 GB)",
            },
            .{
                .repo = "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                .file = "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
                .dest = "qwen3-vl-4b-text-encoder.gguf",
                .label = "Qwen3-VL-4B text encoder (~2.5 GB)",
            },
        },
    },
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
        // MiniMax-H3: the GGUF repo carries only the DiT + text encoder; the
        // video/audio VAEs live in Comfy-Org's repackaged repo. The encoder is
        // renamed so the scanner treats it as support ("encoder") while
        // `findSupport`'s "qwen" needle still matches it.
        .match = "minimax-h3",
        .sidecars = &.{
            .{
                .repo = "Comfy-Org/MiniMax-H3",
                .file = "vae/minimax_h3_video_vae_fp16.safetensors",
                .label = "MiniMax-H3 video VAE (~5 GB)",
            },
            .{
                .repo = "Comfy-Org/MiniMax-H3",
                .file = "vae/minimax_h3_audio_vae_fp32.safetensors",
                .label = "MiniMax-H3 audio VAE (~0.6 GB)",
            },
            .{
                .repo = "leejet/MiniMax-H3-GGUF",
                .file = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                .dest = "qwen3vl-32b-minimax-h3-text-encoder.gguf",
                .label = "Qwen3-VL-32B text encoder (~18 GB)",
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
        .note = "22B video+audio · diffusion + video/audio VAE + connectors + Gemma-3 encoder",
        .repo = "unsloth/LTX-2.3-GGUF",
        .items = &.{
            // Q4_K_M: video DiTs degrade visibly below 4-bit (Q3 looks soft and
            // smeary next to the ~q4 mlx-serve build of the same model).
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf", .label = "LTX-2.3 diffusion (Q4_K_M, ~14 GB)" },
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "vae/ltx-2.3-22b-distilled_video_vae.safetensors", .label = "LTX video VAE" },
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "vae/ltx-2.3-22b-distilled_audio_vae.safetensors", .label = "LTX audio VAE" },
            .{ .repo = "unsloth/LTX-2.3-GGUF", .file = "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors", .label = "LTX connectors" },
            // Keep "gemma" (the video backend's findSupport locates it) but add
            // "encoder" so the scanner treats it as support, not a chat model.
            .{ .repo = "unsloth/gemma-3-12b-it-GGUF", .file = "gemma-3-12b-it-Q4_K_M.gguf", .dest = "gemma-3-12b-it-text-encoder.gguf", .label = "Gemma-3 12B text encoder (~7 GB)" },
            // Optional: the spatial latent upscaler enables the two-stage hires
            // pipeline (low-res pass → 2× latent upscale → short refine).
            .{ .repo = "Lightricks/LTX-2.3", .file = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", .label = "LTX spatial upscaler ×2 (~1 GB)" },
        },
    },
    .{
        .kind = .video,
        .title = "MiniMax-H3 (Hailuo 3.0)",
        .note = "video + native audio · 24 fps · diffusion + video/audio VAE + Qwen3-VL-32B encoder",
        .repo = "leejet/MiniMax-H3-GGUF",
        .items = &.{
            // FL2VA build: text-to-video plus first/last-frame conditioning
            // (the app's start/end frame pickers). Q4_K_M for the same reason
            // as LTX: video DiTs degrade visibly below 4-bit.
            .{ .repo = "leejet/MiniMax-H3-GGUF", .file = "minimax_h3_fl2va-Q4_K_M.gguf", .label = "MiniMax-H3 FL2VA diffusion (Q4_K_M, ~19 GB)" },
            .{ .repo = "Comfy-Org/MiniMax-H3", .file = "vae/minimax_h3_video_vae_fp16.safetensors", .label = "MiniMax-H3 video VAE (~5 GB)" },
            .{ .repo = "Comfy-Org/MiniMax-H3", .file = "vae/minimax_h3_audio_vae_fp32.safetensors", .label = "MiniMax-H3 audio VAE (~0.6 GB)" },
            // Keep "qwen" (findSupport's needle) but add "encoder" so the
            // scanner treats it as support, not a chat model.
            .{ .repo = "leejet/MiniMax-H3-GGUF", .file = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", .dest = "qwen3vl-32b-minimax-h3-text-encoder.gguf", .label = "Qwen3-VL-32B text encoder (~18 GB)" },
        },
    },
    .{
        .kind = .image,
        .title = "Krea-2 Turbo",
        .note = "Krea2 DiT (Q4_K_M) + Wan 2.1 VAE + Qwen3-VL-4B text encoder",
        .repo = "realrebelai/KREA-2_GGUFs",
        .items = &.{
            .{ .repo = "realrebelai/KREA-2_GGUFs", .file = "TURBO/Krea-2-Turbo-Q4_K_M.gguf", .label = "Krea-2 Turbo diffusion (Q4_K_M, ~7.2 GB)" },
            .{ .repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged", .file = "split_files/vae/wan_2.1_vae.safetensors", .label = "Wan 2.1 VAE (~0.3 GB)" },
            .{ .repo = "Qwen/Qwen3-VL-4B-Instruct-GGUF", .file = "Qwen3VL-4B-Instruct-Q4_K_M.gguf", .dest = "qwen3-vl-4b-text-encoder.gguf", .label = "Qwen3-VL-4B text encoder (~2.5 GB)" },
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
