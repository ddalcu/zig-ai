//! Reports which ggml compute backend the app is running on (CUDA / Vulkan /
//! Metal / CPU) and that device's memory, for the sidebar footer indicator.
//!
//! These are read-only queries against the ggml backend registry, which is
//! populated by `llama_backend_init()` (the in-process server starts the chat
//! backend at launch). We never mutate the registry, so this is safe to call
//! from the UI thread, and the memory query (e.g. cudaMemGetInfo) is itself
//! thread-safe with respect to the worker thread's compute.

const std = @import("std");

const c = @cImport({
    @cInclude("ggml-backend.h");
});

pub const Info = struct {
    /// Backend family name, a static C string: "CUDA", "Vulkan", "Metal",
    /// "CPU", ... Safe to hold without copying (lives for the process).
    label: []const u8 = "CPU",
    /// The active device is a GPU/iGPU (vs the CPU fallback).
    is_gpu: bool = false,
    /// Memory is shared with the system rather than dedicated VRAM — integrated
    /// GPUs, Apple Silicon (Metal), and Grace-class unified parts (DGX Spark).
    /// For these the reported total is host memory, so we label it accordingly.
    unified: bool = false,
    mem_used: u64 = 0,
    mem_total: u64 = 0,
    /// True once backends are loaded and a device was found. Until then the UI
    /// shows nothing rather than a misleading "CPU".
    ok: bool = false,
};

fn firstOfType(n: usize, want: c_int) c.ggml_backend_dev_t {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const dev = c.ggml_backend_dev_get(i);
        if (dev != null and c.ggml_backend_dev_type(dev) == want) return dev;
    }
    return null;
}

/// Snapshot the active accelerator + its memory. Cheap; the caller throttles.
pub fn query() Info {
    const n = c.ggml_backend_dev_count();
    if (n == 0) return .{};

    // Mirror llama.cpp's device choice: a discrete GPU wins, then an integrated
    // GPU, otherwise the CPU device.
    var dev = firstOfType(n, c.GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == null) dev = firstOfType(n, c.GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (dev == null) dev = firstOfType(n, c.GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == null) return .{};

    var props: c.struct_ggml_backend_dev_props = undefined;
    c.ggml_backend_dev_get_props(dev, &props);

    const ty = c.ggml_backend_dev_type(dev);
    const label = std.mem.span(c.ggml_backend_reg_name(c.ggml_backend_dev_backend_reg(dev)));

    return .{
        .label = label,
        .is_gpu = ty != c.GGML_BACKEND_DEVICE_TYPE_CPU,
        .unified = ty == c.GGML_BACKEND_DEVICE_TYPE_IGPU or
            std.mem.eql(u8, label, "Metal") or
            isUnifiedName(props.description),
        .mem_used = if (props.memory_total >= props.memory_free) props.memory_total - props.memory_free else 0,
        .mem_total = props.memory_total,
        .ok = true,
    };
}

/// Best-effort detection of NVIDIA Grace-class / AMD APU parts that expose
/// "dedicated" memory which is really shared system RAM. iGPU type and Metal are
/// already covered above; this catches discrete-looking unified devices by name.
fn isUnifiedName(desc: [*c]const u8) bool {
    if (desc == null) return false;
    const s = std.mem.span(desc);
    inline for (.{ "GB10", "Grace", "Spark", "Strix Halo", "Ryzen AI Max" }) |needle| {
        if (std.mem.indexOf(u8, s, needle) != null) return true;
    }
    return false;
}
