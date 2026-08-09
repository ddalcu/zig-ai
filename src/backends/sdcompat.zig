//! Compatibility shims across stable-diffusion.cpp lineages.
//!
//! The vendored submodule has moved between two lineages that differ in public
//! API, and `src/` is mid-migration between them:
//!
//!   * the commit the superproject records (8ef322b, not reachable from
//!     origin/master) has `sd_cancel_generation`, `free_sd_images` and the
//!     4-arg `generate_image`, but none of the backend-residency fields;
//!   * upstream master (f3fd359) added `keep_clip_on_cpu` /
//!     `offload_params_to_cpu` — which is what makes MiniMax-H3 fit a 16 GB
//!     card, see vram.zig — dropped `free_sd_images`, and dropped the cancel
//!     API, its progress callback now returning void with no way to abort a
//!     running generation.
//!
//! Rather than hard-code either, these helpers resolve at comptime, so the tree
//! builds against both and the cancel feature keeps working wherever the symbol
//! exists. NOTE: on upstream master `cancel*` are no-ops — the UI's cancel
//! button can only take effect between generations, not during one. That
//! regression is inherent to upstream and needs a real decision (carry a patch
//! for leejet#1124, or accept it); it is not something this shim can paper over.
//!
//! Parameterised by the caller's `@cImport` namespace: two textually identical
//! cImport blocks in different files still produce *distinct* struct types that
//! don't coerce, so the shim must speak the caller's own types rather than mint
//! a third set of its own.

const std = @import("std");

pub fn Compat(comptime c: type) type {
    return struct {
        /// Whether the linked sd.cpp can abort a generation already in flight.
        pub const cancel_supported = @hasDecl(c, "sd_cancel_generation");

        /// Ask the running generation to stop. No-op where unsupported.
        pub fn cancelAll(ctx: *c.sd_ctx_t) void {
            if (comptime cancel_supported) c.sd_cancel_generation(ctx, c.SD_CANCEL_ALL);
        }

        /// Clear a pending cancel flag so it cannot bleed into the next run.
        pub fn cancelReset(ctx: *c.sd_ctx_t) void {
            if (comptime cancel_supported) c.sd_cancel_generation(ctx, c.SD_CANCEL_RESET);
        }

        /// `free_sd_images` is gone upstream; its examples free the array
        /// element-wise (examples/common/resource_owners.hpp). Same ownership.
        pub fn freeImages(images: [*c]c.sd_image_t, n: c_int) void {
            if (images == null) return;
            if (comptime @hasDecl(c, "free_sd_images")) {
                c.free_sd_images(images, n);
            } else {
                var i: usize = 0;
                while (i < @as(usize, @intCast(@max(n, 0)))) : (i += 1) {
                    if (images[i].data != null) c.free(images[i].data);
                }
                c.free(@ptrCast(images));
            }
        }

        /// `generate_image` lost its out-params upstream: it now returns the
        /// image array directly and the caller derives the count from
        /// `batch_count`. Returns null on failure, matching the old `!ok`.
        pub fn generateImage(
            ctx: *c.sd_ctx_t,
            gp: *const c.sd_img_gen_params_t,
            num_images_out: *c_int,
        ) [*c]c.sd_image_t {
            if (comptime @typeInfo(@TypeOf(c.generate_image)).@"fn".params.len == 2) {
                const images = c.generate_image(ctx, gp);
                // Upstream returns exactly batch_count images on success.
                num_images_out.* = if (images == null) 0 else gp.batch_count;
                return images;
            } else {
                var images: [*c]c.sd_image_t = null;
                const ok = c.generate_image(ctx, gp, &images, num_images_out);
                if (!ok) {
                    num_images_out.* = 0;
                    return null;
                }
                return images;
            }
        }
    };
}
