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

test "sidecarsFor matches FLUX.2 klein by repo id" {
    const s = sidecarsFor("unsloth/FLUX.2-klein-4B-GGUF");
    try std.testing.expectEqual(@as(usize, 2), s.len);
    try std.testing.expect(sidecarsFor("some/Llama-3-GGUF").len == 0);
}
