cmake_minimum_required(VERSION 3.18)
project(zig_ai_engine C CXX)

# -----------------------------------------------------------------------------
# zig-ai in-process AI engine.
#
# This builds the three upstream backends — llama.cpp (chat), stable-diffusion.cpp
# (image), qwen3-tts.cpp (TTS) — so they all share ONE ggml (the copy vendored
# inside llama.cpp). This is the only "glue" we add on top of the stock submodules.
#
# How the single-ggml sharing works:
#   * llama.cpp's add_subdirectory() creates the `ggml` / `ggml-base` / `ggml-cpu`
#     / `ggml-metal` / `ggml-vulkan` / `ggml-cuda` targets.
#   * stable-diffusion.cpp guards its own ggml with `if (NOT TARGET ggml)`, so it
#     reuses ours instead of building a second copy.
#   * qwen3-tts.cpp's own CMake expects a *prebuilt* ggml, so we don't use it;
#     instead we compile its sources here into one `qwen3_tts` lib linked to ggml.
#
# Two link models, chosen by build.zig per platform:
#
#   macOS  → STATIC archives, Metal. The archives are linked straight into the
#            zig exe (one shared libc++). build.zig passes BUILD_SHARED_LIBS=OFF.
#
#   Linux/ → SHARED libraries + GGML_BACKEND_DL=ON. Each ggml backend (cpu,
#   Windows  vulkan, cuda) becomes a separately-loadable module; core ggml, llama,
#            sd and qwen3_tts are shared libs the exe links over their C ABI. This
#            is what lets the CUDA backend be compiled with the native toolchain
#            (nvcc + MSVC/gcc — nvcc cannot use zig cc as host compiler) and still
#            interoperate: the only boundary is ggml's C ABI, so the C++ runtime
#            (libc++ vs MSVC-STL vs libstdc++) no longer has to match across it.
#            At runtime ggml scans the executable's directory, loads whichever
#            backend modules are present and supported, and falls back
#            CUDA → Vulkan → CPU when a GPU/driver is absent.
#
# build.zig drives this via `cmake -S . -B build-deps && cmake --build build-deps`
# and links the resulting libraries.
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Static on Apple (archives linked into the exe); shared elsewhere (dynamic
# backends). build.zig passes BUILD_SHARED_LIBS explicitly; this default only
# matters for a direct `cmake` invocation.
if(NOT DEFINED BUILD_SHARED_LIBS)
  if(APPLE)
    set(BUILD_SHARED_LIBS OFF)
  else()
    set(BUILD_SHARED_LIBS ON)
  endif()
endif()

# Predictable output layout so build.zig can find the libraries:
#   lib/  → static archives (Apple) and import/.so libraries the exe links against
#   bin/  → runtime DLLs (Windows shared builds) incl. the loadable backend modules
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- ggml configuration (applies to the single shared ggml) -----------------
# Apple → Metal. We keep LTO OFF so the (static) archives are plain native
# objects that Zig's self-hosted Mach-O linker can read directly.
if(APPLE)
  set(GGML_METAL ON CACHE BOOL "" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE) # no external default.metallib
endif()
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE) # portable CPU baseline, not -march=native
set(GGML_LTO OFF CACHE BOOL "" FORCE)

# stable-diffusion.cpp requires GGML_MAX_NAME >= 128 (longer tensor names).
add_compile_definitions(GGML_MAX_NAME=128)

# ---- llama.cpp: provides ggml + llama ---------------------------------------
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE) # we only use the llama C API
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/llama.cpp)

# ---- stable-diffusion.cpp: reuses the ggml target above ---------------------
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# Match the global link model so sd's API export macros (SD_API) are set up
# correctly for the shared (Linux/Windows) vs static (Apple) build.
set(SD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
add_subdirectory(deps/stable-diffusion.cpp)
if(APPLE)
  target_compile_definitions(stable-diffusion PRIVATE SD_USE_METAL)
endif()

# ---- qwen3-tts.cpp: compiled here against the shared ggml --------------------
# main.cpp is the standalone CLI; we skip it. CoreML bridge is left off so we use
# the portable stub (no OBJCXX / CoreML framework needed). The library type
# follows BUILD_SHARED_LIBS; on Windows we export the whole C API automatically
# (the sources have no __declspec annotations).
set(QWEN3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/qwen3-tts.cpp)
add_library(qwen3_tts
  ${QWEN3_DIR}/src/text_tokenizer.cpp
  ${QWEN3_DIR}/src/tts_transformer.cpp
  ${QWEN3_DIR}/src/audio_tokenizer_encoder.cpp
  ${QWEN3_DIR}/src/audio_tokenizer_decoder.cpp
  ${QWEN3_DIR}/src/gguf_loader.cpp
  ${QWEN3_DIR}/src/qwen3_tts.cpp
  ${QWEN3_DIR}/src/qwen3tts_c_api.cpp
  ${QWEN3_DIR}/src/coreml_code_predictor_stub.cpp
)
target_include_directories(qwen3_tts PUBLIC ${QWEN3_DIR}/src)
target_include_directories(qwen3_tts PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/include
)
# MSVC's <cmath> only exposes M_PI etc. when _USE_MATH_DEFINES is set before the
# include (clang/gcc define them unconditionally); qwen3-tts.cpp uses M_PI.
# Harmless on non-MSVC toolchains.
target_compile_definitions(qwen3_tts PRIVATE _USE_MATH_DEFINES)
target_link_libraries(qwen3_tts PRIVATE ggml)
target_compile_features(qwen3_tts PRIVATE cxx_std_17)
set_target_properties(qwen3_tts PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
