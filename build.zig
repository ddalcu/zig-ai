const std = @import("std");

// The three AI backends live as git submodules under deps/ and are compiled by
// our top-level CMakeLists.txt into static archives under build-deps/lib. See
// the comment on linkAiBackends and CMakeLists.txt for the shared-ggml design.

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Where the zigui UI library checkout lives. Locally it is the sibling
    // repo (../zigui); CI checks it out inside the workspace and passes
    // -Dzigui=<path>.
    const zigui_path = b.option([]const u8, "zigui", "Path to the zigui library checkout (default: ../zigui)") orelse
        b.pathFromRoot("../zigui");
    // Optional SDL3 install prefix (containing include/ and lib/). Defaults to
    // the Homebrew keg on macOS and the system paths elsewhere; CI passes an
    // explicit prefix on Linux (built from source) and Windows (SDL releases).
    const sdl3_prefix = b.option([]const u8, "sdl3", "SDL3 install prefix (include/ + lib/)");

    // Which in-process backends to link. llama (chat) is on by default; sd/tts
    // are added in their phases. Each maps to a prebuilt static archive set.
    const link_llama = b.option(bool, "llama", "Link the llama.cpp chat backend") orelse true;
    const link_sd = b.option(bool, "sd", "Link the stable-diffusion.cpp backend") orelse true;
    const link_tts = b.option(bool, "tts", "Link the qwen3-tts.cpp backend") orelse true;

    // GPU on Linux/Windows comes from ggml's Vulkan backend (macOS uses Metal).
    // Building it needs the Vulkan headers/loader + glslc (libvulkan-dev +
    // glslc on Linux, the LunarG SDK on Windows); -Dvulkan=false opts out.
    const use_vulkan = b.option(bool, "vulkan", "Build/link the ggml Vulkan GPU backend (default: on for Linux/Windows)") orelse
        (target.result.os.tag != .macos);
    // Where to find the Vulkan import library at link time (Windows: the SDK
    // root, providing Lib/vulkan-1.lib). Linux finds libvulkan via system paths.
    // Only used by the macOS-style static link; in the shared/DL build the
    // ggml-vulkan module links Vulkan itself and the SDK is found via the
    // VULKAN_SDK env at CMake-configure time.
    const vulkan_prefix = b.option([]const u8, "vulkan-prefix", "Vulkan SDK prefix (Windows link)");

    // ggml CUDA backend (NVIDIA). Linux/Windows only, and only in the shared/DL
    // build: it is compiled with the native toolchain (nvcc + MSVC/gcc — nvcc
    // cannot use zig cc as host) and loaded at runtime as a ggml backend module,
    // so it never links into the exe. Requires the CUDA Toolkit at build time.
    const use_cuda = b.option(bool, "cuda", "Build the ggml CUDA backend (Linux/Windows; needs CUDA Toolkit + native MSVC/gcc)") orelse false;
    // CMAKE_CUDA_ARCHITECTURES override (e.g. "native", "120", "75;80;86;89;90").
    // Default: ggml's own portable architecture list.
    const cuda_arch = b.option([]const u8, "cuda-arch", "CMAKE_CUDA_ARCHITECTURES (default: ggml's portable list)");

    // ---- zigui core module (mirror zigui/build.zig) -----------------------
    const zigui = b.addModule("zigui", .{
        .root_source_file = .{ .cwd_relative = b.fmt("{s}/src/zigui.zig", .{zigui_path}) },
        .target = target,
        .optimize = optimize,
    });
    zigui.addAnonymousImport("inter_font", .{
        .root_source_file = .{ .cwd_relative = b.fmt("{s}/assets/fonts/Inter.ttf", .{zigui_path}) },
    });
    zigui.addAnonymousImport("icon_font", .{
        .root_source_file = .{ .cwd_relative = b.fmt("{s}/assets/fonts/icons.ttf", .{zigui_path}) },
    });
    zigui.addAnonymousImport("emoji_font", .{
        .root_source_file = .{ .cwd_relative = b.fmt("{s}/assets/fonts/NotoEmoji.ttf", .{zigui_path}) },
    });

    // ---- SDL-linking runtime module (zigui's app.zig) ---------------------
    const app_mod = b.createModule(.{
        .root_source_file = .{ .cwd_relative = b.fmt("{s}/src/app.zig", .{zigui_path}) },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "zigui", .module = zigui }},
    });
    linkSdl3(app_mod, target, sdl3_prefix);

    // ---- main executable --------------------------------------------------
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "zigui", .module = zigui },
            .{ .name = "zigui_app", .module = app_mod },
        },
    });
    // The exe itself talks to SDL (audio subsystem for TTS playback).
    linkSdl3(exe_mod, target, sdl3_prefix);
    const build_opts = b.addOptions();
    build_opts.addOption(bool, "link_llama", link_llama);
    build_opts.addOption(bool, "link_sd", link_sd);
    build_opts.addOption(bool, "link_tts", link_tts);
    exe_mod.addOptions("build_options", build_opts);

    // Vendored single-header media encoders (PNG + H.264/MP4, no ffmpeg) compiled
    // straight into the exe. minih264 (CC0) + minimp4 (public domain) + a pure-C
    // stb_image_write copy all live self-contained in deps/codecs.
    exe_mod.addIncludePath(b.path("deps/codecs"));
    exe_mod.addCSourceFiles(.{
        .files = &.{
            "src/codecs/codecs.c",
            "src/codecs/codecs_h264.c",
            "src/codecs/codecs_mp4.c",
        },
        .flags = &.{"-O2"},
    });

    // Vendored Jinja engine (Apache-2.0, from mlx-serve / llama.cpp lineage) so we
    // can render a model's actual `chat_template` — llama.cpp's C API only matches
    // a hardcoded template list, which newer models (e.g. gemma-4) aren't in.
    exe_mod.addIncludePath(b.path("deps/jinja"));
    exe_mod.addCSourceFiles(.{
        .files = &.{
            "deps/jinja/lexer.cpp",
            "deps/jinja/parser.cpp",
            "deps/jinja/runtime.cpp",
            "deps/jinja/value.cpp",
            "deps/jinja/jinja_string.cpp",
            "deps/jinja/jinja_wrapper.cpp",
        },
        .flags = &.{ "-std=c++17", "-O2" },
    });
    exe_mod.link_libcpp = true; // the Jinja sources are C++

    // Build the AI backends (llama/sd/qwen3-tts) from the deps/ submodules via
    // CMake, then link the resulting archives. cmake_build is the step the exe
    // must wait on before linking.
    // macOS links the static archives straight into the exe (one shared libc++);
    // Linux/Windows build shared libraries with dynamic ggml backends and link
    // them over their C ABI (see CMakeLists.txt and linkAiBackends).
    const shared = target.result.os.tag != .macos;
    const cmake_build: ?*std.Build.Step = if (link_llama or link_sd or link_tts) blk: {
        const step = cmakeBuildStep(b, target, use_vulkan, use_cuda, cuda_arch, shared);
        linkAiBackends(b, exe_mod, target, link_llama, link_sd, link_tts, use_vulkan, vulkan_prefix, shared);
        break :blk step;
    } else null;

    const exe = b.addExecutable(.{ .name = "zig-ai", .root_module = exe_mod });
    if (cmake_build) |s| exe.step.dependOn(s);
    // On Windows, build as a GUI-subsystem app so launching it from Explorer
    // doesn't flash a console window before the UI. With no console attached,
    // stdout/stderr would be lost — src/log_capture.zig redirects them into the
    // in-app Logs view instead (and CLI/headless modes re-attach the parent
    // console so terminal output still works).
    if (target.result.os.tag == .windows) exe.subsystem = .windows;
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run zig-ai").dependOn(&run_cmd.step);
}

/// Configure + build the deps/ submodules (llama.cpp, stable-diffusion.cpp,
/// qwen3-tts.cpp) via our top-level CMakeLists.txt. Returns the build step the
/// final link must depend on. CMake/Ninja are incremental, so re-running this on
/// every `zig build` is cheap when the C++ sources are unchanged.
fn cmakeBuildStep(b: *std.Build, target: std.Build.ResolvedTarget, use_vulkan: bool, use_cuda: bool, cuda_arch: ?[]const u8, shared: bool) *std.Build.Step {
    const configure = b.addSystemCommand(&.{
        "cmake", "-S", ".", "-B", "build-deps", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
    });
    // macOS: static archives linked into the exe. Linux/Windows: shared libraries
    // with dynamically-loaded ggml backends (GGML_BACKEND_DL) so the CUDA backend
    // can be built by the native toolchain and loaded at runtime.
    if (shared) {
        configure.addArgs(&.{ "-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON" });
    } else {
        configure.addArg("-DBUILD_SHARED_LIBS=OFF");
    }
    // On Windows the deps build with MSVC (cl.exe) — it's nvcc's required host
    // compiler and the only toolchain that produces clean export DLLs here. Pin
    // it explicitly so a stray CC/CXX env var (e.g. a leftover `zig cc` from the
    // old static workflow) can't steer CMake to the wrong compiler, which would
    // drag in MinGW's windres for resource compilation and break the CUDA link.
    if (shared and target.result.os.tag == .windows) {
        configure.addArgs(&.{ "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl" });
    }
    configure.addArg(if (use_vulkan) "-DGGML_VULKAN=ON" else "-DGGML_VULKAN=OFF");
    if (use_cuda) {
        configure.addArg("-DGGML_CUDA=ON");
        if (cuda_arch) |arch| configure.addArg(b.fmt("-DCMAKE_CUDA_ARCHITECTURES={s}", .{arch}));
    }
    const compile = b.addSystemCommand(&.{ "cmake", "--build", "build-deps", "-j" });
    compile.step.dependOn(&configure.step);

    const step = b.step("deps", "Build the C/C++ AI backends (llama/sd/tts) via CMake");
    step.dependOn(&compile.step);
    return step;
}

/// Link the in-process AI backends we built from deps/. All three backends
/// share ONE ggml (built inside llama.cpp), so there are no duplicate-symbol
/// conflicts — see CMakeLists.txt.
///
/// Built with GGML_LTO=OFF, so the archives are plain native objects that
/// Zig's self-hosted linker reads directly (no `ld -r` bitcode workaround).
///
/// Per-target C++ runtime: the archives must share a C++ stdlib with the link.
/// macOS (AppleClang) and Windows (zig cc in CI) both use LLVM libc++, which
/// `link_libcpp` provides; Linux CMake builds with g++ against libstdc++, so
/// there we link the system libstdc++ instead.
fn linkAiBackends(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, llama: bool, sd: bool, tts: bool, vulkan: bool, vulkan_prefix: ?[]const u8, shared: bool) void {
    m.addIncludePath(b.path("deps/llama.cpp/include")); // llama.h
    m.addIncludePath(b.path("deps/llama.cpp/ggml/include")); // ggml*.h
    m.addIncludePath(b.path("deps/stable-diffusion.cpp/include")); // stable-diffusion.h
    m.addIncludePath(b.path("deps/qwen3-tts.cpp/src")); // qwen3tts_c_api.h

    if (shared) {
        linkAiBackendsShared(m, target, llama, sd, tts);
        return;
    }

    const L = "build-deps/lib/";
    if (llama) m.addObjectFile(.{ .cwd_relative = L ++ "libllama.a" });
    if (sd) m.addObjectFile(.{ .cwd_relative = L ++ "libstable-diffusion.a" });
    if (tts) m.addObjectFile(.{ .cwd_relative = L ++ "libqwen3_tts.a" });

    // Shared ggml set (one copy, from llama.cpp). Metal only exists on macOS.
    // ggml's CMake drops the `lib` archive prefix on Windows (ggml.a), while
    // the llama/sd/qwen3 targets above keep it — hence the split naming.
    const gp = if (target.result.os.tag == .windows) "" else "lib";
    if (target.result.os.tag == .macos)
        m.addObjectFile(.{ .cwd_relative = L ++ "libggml-metal.a" });
    m.addObjectFile(.{ .cwd_relative = b.fmt("{s}{s}ggml.a", .{ L, gp }) });
    // ggml-vulkan sits between ggml (whose backend registry references it) and
    // ggml-base (whose core symbols it needs) — ELF archive resolution is
    // strictly left-to-right.
    if (vulkan) {
        m.addObjectFile(.{ .cwd_relative = b.fmt("{s}{s}ggml-vulkan.a", .{ L, gp }) });
        if (vulkan_prefix) |p| m.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/Lib", .{p}) });
        // The Vulkan loader: vulkan-1.dll ships with every modern GPU driver
        // on Windows; libvulkan.so.1 is bundled next to the Linux binary.
        m.linkSystemLibrary(if (target.result.os.tag == .windows) "vulkan-1" else "vulkan", .{});
    }
    m.addObjectFile(.{ .cwd_relative = b.fmt("{s}{s}ggml-cpu.a", .{ L, gp }) });
    m.addObjectFile(.{ .cwd_relative = b.fmt("{s}{s}ggml-base.a", .{ L, gp }) });

    // One C++ runtime everywhere: the deps are compiled with (Apple)clang /
    // zig cc on every platform — CI sets CC/CXX to `zig cc` on Linux and
    // Windows — so zig's libc++ satisfies them at link. (Linking g++-built
    // archives would need libstdc++ instead; don't mix.)
    m.link_libcpp = true;
    if (target.result.os.tag == .macos) {
        const frameworks = [_][]const u8{
            "Metal", "MetalKit", "Foundation", "CoreFoundation",
            "Accelerate", "QuartzCore", "MetalPerformanceShaders",
        };
        for (frameworks) |f| m.linkFramework(f, .{});
    }
}

/// Link the AI backends as SHARED libraries (Linux/Windows). The exe links only
/// the C-ABI surface — llama, stable-diffusion, qwen3_tts and core ggml/ggml-base
/// — over their import libraries. The compute backends (ggml-cpu, ggml-vulkan,
/// ggml-cuda) are GGML_BACKEND_DL modules NOT linked here: ggml loads them at
/// runtime from the executable's directory and selects the best available
/// (CUDA → Vulkan → CPU). This C-ABI boundary is what lets the CUDA module be
/// compiled with a different toolchain (nvcc + MSVC/gcc) than the zig exe.
///
/// The shared libraries are placed next to the binary at package time; the rpath
/// ($ORIGIN on Linux, implicit same-dir search on Windows) finds them at runtime.
fn linkAiBackendsShared(m: *std.Build.Module, target: std.Build.ResolvedTarget, llama: bool, sd: bool, tts: bool) void {
    m.addLibraryPath(.{ .cwd_relative = "build-deps/lib" });
    if (llama) m.linkSystemLibrary("llama", .{});
    if (sd) m.linkSystemLibrary("stable-diffusion", .{});
    if (tts) m.linkSystemLibrary("qwen3_tts", .{});
    // Core ggml: `ggml` (registry/umbrella) plus `ggml-base` (which exports the
    // gguf_* / ggml_* symbols the exe itself references via gguf.h / ggml.h).
    m.linkSystemLibrary("ggml", .{});
    m.linkSystemLibrary("ggml-base", .{});

    // Find the co-located shared libs at runtime. On Windows the loader already
    // searches the executable's own directory, so no rpath is needed there.
    if (target.result.os.tag == .linux) m.addRPath(.{ .cwd_relative = "$ORIGIN" });

    // The exe still compiles its own C++ (vendored Jinja); give it a C++ runtime.
    // The deps' C++ runtime lives inside their shared libs and never crosses the
    // C-ABI boundary, so it need not match the exe's.
    m.link_libcpp = true;
}

/// Wire up SDL3 headers/libs. `-Dsdl3=<prefix>` points at an install tree with
/// include/ and lib/ (CI does this on Linux/Windows); without it we use the
/// Homebrew keg on macOS and the system paths on Linux. The extra rpaths make
/// packaged binaries pick up an SDL3 shared library placed next to the
/// executable (release tarballs bundle it).
fn linkSdl3(m: *std.Build.Module, target: std.Build.ResolvedTarget, sdl3_prefix: ?[]const u8) void {
    const b = m.owner;
    if (sdl3_prefix) |prefix| {
        m.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{prefix}) });
        m.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{prefix}) });
    } else if (target.result.os.tag == .macos) {
        const prefix: []const u8 = if (target.result.cpu.arch == .aarch64)
            "/opt/homebrew/opt/sdl3"
        else
            "/usr/local/opt/sdl3";
        m.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{prefix}) });
        m.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{prefix}) });
        m.addRPath(.{ .cwd_relative = b.fmt("{s}/lib", .{prefix}) });
    }
    switch (target.result.os.tag) {
        .macos => m.addRPath(.{ .cwd_relative = "@executable_path" }),
        .linux => m.addRPath(.{ .cwd_relative = "$ORIGIN" }),
        else => {},
    }
    m.linkSystemLibrary("SDL3", .{});
}
