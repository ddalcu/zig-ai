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

    // Build the AI backends (llama/sd/qwen3-tts) from the deps/ submodules via
    // CMake, then link the resulting archives. cmake_build is the step the exe
    // must wait on before linking.
    const cmake_build: ?*std.Build.Step = if (link_llama or link_sd or link_tts) blk: {
        const step = cmakeBuildStep(b);
        linkAiBackends(b, exe_mod, target, link_llama, link_sd, link_tts);
        break :blk step;
    } else null;

    const exe = b.addExecutable(.{ .name = "zig-ai", .root_module = exe_mod });
    if (cmake_build) |s| exe.step.dependOn(s);
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
fn cmakeBuildStep(b: *std.Build) *std.Build.Step {
    const configure = b.addSystemCommand(&.{
        "cmake", "-S", ".", "-B", "build-deps", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
    });
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
fn linkAiBackends(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, llama: bool, sd: bool, tts: bool) void {
    m.addIncludePath(b.path("deps/llama.cpp/include")); // llama.h
    m.addIncludePath(b.path("deps/llama.cpp/ggml/include")); // ggml*.h
    m.addIncludePath(b.path("deps/stable-diffusion.cpp/include")); // stable-diffusion.h
    m.addIncludePath(b.path("deps/qwen3-tts.cpp/src")); // qwen3tts_c_api.h

    const L = "build-deps/lib/";
    if (llama) m.addObjectFile(.{ .cwd_relative = L ++ "libllama.a" });
    if (sd) m.addObjectFile(.{ .cwd_relative = L ++ "libstable-diffusion.a" });
    if (tts) m.addObjectFile(.{ .cwd_relative = L ++ "libqwen3_tts.a" });

    // Shared ggml set (one copy, from llama.cpp). Metal only exists on macOS.
    if (target.result.os.tag == .macos)
        m.addObjectFile(.{ .cwd_relative = L ++ "libggml-metal.a" });
    m.addObjectFile(.{ .cwd_relative = L ++ "libggml.a" });
    m.addObjectFile(.{ .cwd_relative = L ++ "libggml-cpu.a" });
    m.addObjectFile(.{ .cwd_relative = L ++ "libggml-base.a" });

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
