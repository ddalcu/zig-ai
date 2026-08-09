//! VRAM residency policy for the sd.cpp backends.
//!
//! Most models fit a consumer card with their weights resident, and the default
//! policy (everything on the GPU) is both simplest and fastest. MiniMax-H3 does
//! not: its Qwen3-VL-32B text encoder alone is ~17 GiB, more than a 16 GB card
//! holds, before the ~10.6 GiB pruned DiT and the ~4.9 GiB video VAE. That
//! ~33 GiB weight set fits neither the GPU nor a 32 GiB host's RAM, so H3 has
//! to time-share: components load for their phase and are released after.
//!
//! sd.cpp can plan that itself (`--auto-fit`, docs/backend.md): it derives the
//! diffusion / te / vae placements from model metadata and per-device budgets,
//! and when the set does not fit resident it gives the heavy components `disk`
//! params residency. That is strictly better than hand-assigning `te=cpu`,
//! which would park 17 GiB in system RAM and leave the rest to contend for what
//! is left.
//!
//! Kept dependency-free (std only) so the invariants below are unit-testable —
//! `backends/video.zig` can't be, it `@cImport`s the sd.cpp header.

const std = @import("std");

/// Maps onto the matching `sd_ctx_params_t` fields. Strings are sentinel-
/// terminated so they can be handed to C without copying.
pub const Policy = struct {
    /// Derive diffusion/te/vae placement from model metadata and the memory
    /// budgets, then time-share what doesn't fit. Note that sd.cpp *ignores*
    /// `backend` and `params_backend` while this is on.
    auto_fit: bool = false,
    /// Budget for graph-cut segmented offload: null = sd.cpp's default (each
    /// device's free memory minus a 512 MiB margin), "-1" = free memory minus
    /// 1 GiB, or per-device e.g. "cuda0=8".
    max_vram: ?[:0]const u8 = null,
    /// Residency + prefetch streaming layered on `max_vram`. sd.cpp ignores it
    /// without a budget, and disables it for any module that got layer-split.
    stream_layers: bool = false,
    /// File-backed weights, so the OS can evict pages under memory pressure
    /// rather than the process being killed. Matters when the total weight set
    /// approaches system RAM, which for H3 it does.
    enable_mmap: bool = false,
    /// Explicit runtime placement, e.g. "te=cpu". Ignored under `auto_fit`.
    backend: ?[:0]const u8 = null,
    /// Explicit params residency, e.g. "te=cpu" or "diffusion=disk". Ignored
    /// under `auto_fit`.
    params_backend: ?[:0]const u8 = null,
};

/// Everything resident on the GPU: Wan 2.2 and LTX-2.3 both fit.
pub const resident: Policy = .{};

/// MiniMax-H3 — see the module comment for why this can't be resident.
pub const minimax_h3: Policy = .{
    .auto_fit = true,
    // Leave a GiB for the desktop compositor rather than sd.cpp's 512 MiB.
    .max_vram = "-1",
    // NOT mmap: auto-fit gives the heavy modules `disk` params residency
    // (load per phase, free after), which owns the file reads itself. Layering
    // mmap under that is redundant, and the verified-good sd-cli run did not
    // use it.
    .enable_mmap = false,
};

/// Whether sd.cpp will actually honour `stream_layers` rather than ignore it.
pub fn streamingIsEffective(p: Policy) bool {
    return p.stream_layers and p.max_vram != null;
}

/// `backend` / `params_backend` are dead configuration under `auto_fit`; a
/// policy setting both is stating an intent sd.cpp will silently discard.
pub fn hasIgnoredPlacement(p: Policy) bool {
    return p.auto_fit and (p.backend != null or p.params_backend != null);
}

test "resident policy leaves every offload knob off" {
    try std.testing.expect(!resident.auto_fit);
    try std.testing.expect(resident.max_vram == null);
    try std.testing.expect(!resident.enable_mmap);
}

test "MiniMax-H3 cannot be GPU-resident and says so" {
    try std.testing.expect(minimax_h3.auto_fit);
    // Verified against sd-cli on a 16 GB card: auto-fit plans
    //   --backend "diffusion=CUDA0,te=cpu,vae=CUDA0"
    //   --params-backend "diffusion=disk,vae=disk"
    // placing the 18.9 GB encoder in RAM and time-sharing the rest. A budget
    // is what makes that plan fit; without one the DiT is asked to be resident.
    try std.testing.expect(minimax_h3.max_vram != null);
}

test "no policy states placement that auto-fit would discard" {
    for ([_]Policy{ resident, minimax_h3 }) |p| {
        try std.testing.expect(!hasIgnoredPlacement(p));
    }
}

test "any policy that asks for streaming actually gets it" {
    for ([_]Policy{ resident, minimax_h3 }) |p| {
        try std.testing.expectEqual(p.stream_layers, streamingIsEffective(p));
    }
}
