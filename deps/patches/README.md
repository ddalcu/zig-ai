# Local patches on the deps/ submodules

Applied as working-tree edits after `git submodule update` (see the root
README's Building section); CI and `scripts/build-windows.ps1` apply them
automatically.

## llama.cpp-metal-left-pad.patch (optional, performance)

Extends ggml's Metal `PAD` kernel (`kernel_pad_impl`) with left/causal
padding, which the Wan/LTX video VAEs use. All three backends share
llama.cpp's ggml (see the top-level CMakeLists.txt), so this is a
deps/llama.cpp patch.

Since leejet/stable-diffusion.cpp#1731 this is a performance optimization,
not a requirement: when the backend reports left-pad unsupported, sd.cpp
falls back to right-pad + `ggml_roll` (ROLL has Metal/Vulkan/CUDA kernels).
With the patch, Metal takes the direct left-pad path and skips the extra
roll op per conv.

## qwen3-tts.cpp-win-portability.patch

Windows portability for qwen3-tts.cpp's diagnostic memory snapshot
(`getrusage` has no Win32 equivalent; returns "unsupported" there).

## stable-diffusion.cpp — no patch (fixes live on the ddalcu fork)

`deps/stable-diffusion.cpp` tracks https://github.com/ddalcu/stable-diffusion.cpp
(our fork of leejet/stable-diffusion.cpp; `upstream` remote = leejet). The
submodule is pinned to the fork's `zig-ai` branch, which carries zig-ai's
sd.cpp fixes as real commits — currently the `force_prec_f32` conv3d
`IM2COL_3D` fallback (LTX-2.3 VAE encoder on Metal; a follow-up to
leejet#1731 worth PRing upstream). New sd.cpp fixes go on that branch, not
in this directory.

(The older `stable-diffusion.cpp-conv3d-direct.patch` was superseded by
leejet#1731 itself; `stable-diffusion.cpp-conv3d-f32-metal.patch` graduated
to the fork commit.)
