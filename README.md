<p align="center">
  <img src="assets/icon.png" alt="zig-ai" width="180" />
</p>

<h1 align="center">zig-ai</h1>

<p align="center"><b>Local AI, in one native app — chat, image, video & voice, all on-device.</b></p>

A cross-platform desktop app that runs local AI models **in-process** — chat
(llama.cpp), image generation (stable-diffusion.cpp), video (Wan / LTX via
stable-diffusion.cpp), and text-to-speech with voice cloning (qwen3-tts.cpp) —
behind a single, clean UI built on the
[`zigui`](https://github.com/ddalcu/zigui) library. A sidebar of screens,
iMessage-style chat bubbles, a built-in Hugging Face model downloader, MCP
tool servers + agent mode, system-tray status, light/dark themes.

Everything runs in one process: each backend's C/C++ library is linked directly
and driven from a background worker thread, streaming results to the UI through a
thread-safe channel. Models, prompts and audio never leave the machine (the only
outbound traffic is the model downloader, and MCP servers you explicitly add);
the built-in OpenAI-compatible server listens on your local network only.

macOS (Apple Silicon, Metal) is the primary target; Linux and Windows builds
are produced by CI (CPU inference). Download binaries from the
[Releases](../../releases) page — each archive bundles the SDL3 library it
needs.

> **Status: proof of concept.** This app explores running the full local-AI
> stack in-process from Zig. The plan is to eventually merge it with
> [mlx-serve](https://github.com/ddalcu/mlx-serve), whose UI it mirrors.

## Screenshots

<p align="center">
  <img src="assets/screenshots/chat.png" width="49%" alt="Chat" />
  <img src="assets/screenshots/video.png" width="49%" alt="Video generation" />
</p>
<p align="center">
  <img src="assets/screenshots/image.png" width="49%" alt="Image generation" />
  <img src="assets/screenshots/models.png" width="49%" alt="Model browser / downloader" />
</p>

## Architecture

```
src/
  main.zig            entry, arg parsing, tray, headless smoke/screenshot modes
  state.zig           AppState: all reactive state + per-backend façades & pumps
  models.zig          GGUF discovery (dir scan + kind heuristics)
  config.zig          per-user config dir, system-prompt.md + mcp.json I/O
  settings_store.zig  persisted UI settings
  manifest.zig        curated cross-repo sidecar manifest (FLUX/Wan extras)
  mcp.zig             MCP: preset catalog, mcp.json registry, JSON-RPC runtime
  agent.zig           agent mode: tool-aware system prompt + tool-call parsing
  builtin.zig         built-in agent tools (read/write/list/search files, shell)
  channel.zig         Channel(T) (spinlock queue) + JobState (progress/cancel atomics)
  audio.zig           SDL3 audio playback + microphone Recorder (voice cloning)
  backends/
    llama.zig         llama.cpp chat: worker thread, streaming decode/sample loop
    sd.zig            stable-diffusion.cpp txt2img: worker + progress callback
    video.zig         Wan / LTX video generation (split-file model specs)
    tts.zig           qwen3-tts.cpp synthesis: worker -> float32 PCM (+ clone refs)
    downloader.zig    native Hugging Face search/quant-list/download (std.http)
  ui/
    shell.zig         NavigationSplitView sidebar + per-frame backend pump (root body)
    chat.zig image.zig video.zig audio.zig
    model_browser.zig downloader.zig mcp_view.zig editor.zig
    settings.zig tasks.zig logs.zig widgets.zig
```

**Threading model.** zigui's UI loop is single-threaded and rebuilds the view
every frame. Each backend runs inference on its own thread and communicates via a
`Channel` drained once per frame; a `busyCheck` hook keeps the loop awake (~60fps)
while any job runs. Worker request signaling uses pthreads; the channel uses a
pure-atomic spinlock. See `channel.zig` and `backends/llama.zig`.

## Building

Requires Zig 0.16, SDL3 (`brew install sdl3`), CMake + Ninja
(`brew install cmake ninja`), and a checkout of
[`zigui`](https://github.com/ddalcu/zigui) — by default expected as a sibling
directory (`../zigui`); point elsewhere with `-Dzigui=<path>`.

The three AI backends are vendored as git submodules under `deps/`:

| submodule | upstream |
| --- | --- |
| `deps/llama.cpp` | `ggml-org/llama.cpp` (chat + the shared ggml) |
| `deps/stable-diffusion.cpp` | `ddalcu/stable-diffusion.cpp` — our fork of `leejet/stable-diffusion.cpp` (image + video); any zig-ai patches live on the fork |
| `deps/qwen3-tts.cpp` | `predict-woo/qwen3-tts.cpp` (TTS) |

After cloning, fetch them (including each repo's nested `ggml`) and apply the
local Metal patch (see "Video runs on Metal" below):

```sh
git submodule update --init --recursive
git -C deps/llama.cpp apply ../patches/llama.cpp-metal-left-pad.patch
```

Then build:

```sh
zig build              # builds the C++ deps via CMake, links llama + sd + tts
zig build run          # build and launch the app
zig build deps         # (re)build only the C/C++ backends
```

Build options: `-Dzigui=<path>` (zigui checkout), `-Dsdl3=<prefix>` (SDL3
install tree with `include/` + `lib/`), and per-backend toggles
`-Dllama=false`, `-Dsd=false`, `-Dtts=false`.

> **Shared ggml.** `CMakeLists.txt` builds all three backends so they share the
> single ggml inside `llama.cpp` — stable-diffusion.cpp reuses it via its
> `if(NOT TARGET ggml)` guard; qwen3-tts.cpp's sources are compiled here against
> it. This avoids the duplicate-symbol conflict three independent ggml copies
> would cause. The build uses `GGML_LTO=OFF`, so the archives are native objects
> Zig's linker reads directly. `build.zig` drives the CMake build
> (`cmake -S . -B build-deps`) and links the archives from `build-deps/lib/`.

## Releases (CI)

`.github/workflows/release.yml` builds macOS (arm64), Linux (x86_64) and
Windows (x86_64, experimental) binaries on every `v*` tag and attaches them to
the GitHub release. It checks out `zigui` at the tag pinned in `ZIGUI_REF`,
applies `deps/patches/`, and bundles the SDL3 shared library next to the
binary (`@executable_path` / `$ORIGIN` rpaths). Trigger it manually with
*workflow_dispatch* to get artifacts without cutting a release.

## Models

The Models screen has a built-in Hugging Face downloader (search → pick a
quant → it fetches the quant plus every support file, and curated cross-repo
sidecars like FLUX's VAE/text-encoder). Downloads land in the app's own models
dir (`~/Library/Application Support/zig-ai/models` on macOS). The browser also
scans `~/.lmstudio/models`, `~/.mlx-serve/models`, and any folders added in
Settings for `*.gguf`, classifying each as chat / image / video / tts by
filename. (qwen3-tts loads the *folder* containing its `.gguf` + tokenizer,
not the single file.)

**Audio / voice cloning.** The Audio screen synthesizes with the model's
default voice, or clones one from a reference: pick a WAV (any sample rate) or
record a few seconds with the built-in mic recorder. The speaker encoder is
part of the TTS model — no extra files needed.

**Image.** Classic SD checkpoints are one self-contained file; split models
pair a diffusion `.gguf` with sidecars auto-discovered beside it:
- **Krea2** (Raw / Turbo) — diffusion `*.gguf` + Wan 2.1 VAE + a Qwen3-VL
  `*.gguf` text encoder. Tested with
  [realrebelai/KREA-2_GGUFs](https://huggingface.co/realrebelai/KREA-2_GGUFs) (Turbo Q4_K_M)
  + Comfy-Org `wan_2.1_vae.safetensors` +
  [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF).
  Turbo is distilled: ~8 steps at CFG 1.0.
- **FLUX.2** — diffusion `*.gguf` + FLUX VAE + a Qwen3 `*.gguf` encoder;
  **FLUX.1** — CLIP-L + T5-XXL instead.

**Video.** Drop a video model's files in one folder; the diffusion `.gguf` shows
up as a **Video** model and its sidecars are auto-discovered beside it.
- **Wan 2.2** — diffusion `*.gguf` + `*vae*.safetensors` + `umt5-xxl-*.gguf`.
  Tested with [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)
  + Comfy-Org `wan2.2_vae.safetensors` + [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf).
- **LTX-2.3** — diffusion `*.gguf` + `*video_vae*` + `*audio_vae*` + `*connectors*`
  + a Gemma-3 `*.gguf` text encoder. Tested with
  [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) (distilled-1.1 Q4_K_M)
  + [unsloth/gemma-3-12b-it-GGUF](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF).
  (LTX is built for 1280×720; very small sizes degrade badly, and frame counts are
  aligned to its temporal factor.)
- **MiniMax-H3 (Hailuo 3.0)** — video + native stereo audio in one joint model.
  Diffusion `*.gguf` + `*video_vae*` + `*audio_vae*` + a Qwen3-VL-32B `*.gguf`
  text encoder. Uses [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF)
  (FL2VA Q4_K_M) + the VAEs from
  [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).
  CFG-distilled (CFG 1.0, no negative), 24 fps, frame counts on a 17k+5 ladder.
  Supports text-to-video plus first/last-frame conditioning; the audio VAE is
  optional (omit it for silent video).

> **Video runs on Metal**, like everything else. stable-diffusion.cpp handles
> Metal's missing `IM2COL_3D` kernel since leejet#1731 (it checks
> `ggml_backend_supports_op` and falls back to the direct `GGML_OP_CONV_3D`
> op); our fork's `zig-ai` branch extends the same fallback to the
> `force_prec_f32` branch, which the LTX-2.3 VAE *encoder* needs — without it,
> image-to-video aborts on Metal. That fix is a committed part of the pinned
> submodule, so no patch is needed. One optional local patch remains: ggml's
> Metal `PAD` kernel extended with left/causal padding
> (`deps/patches/llama.cpp-metal-left-pad.patch`); without it sd.cpp falls
> back to right-pad + `ggml_roll` — still correct, just marginally slower.
> Apply it after every `git submodule update` (see Building above); CI
> applies it automatically.

## Local API

The app serves an OpenAI-compatible HTTP API on `http://0.0.0.0:8080/v1` (the
tray shows the LAN URL). Chat: `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/models`. Media generation runs against the models
selected in the GUI, with the same request fields as
[mlx-serve](https://github.com/ddalcu/mlx-serve):

- `POST /v1/images/generations` — `prompt` (required), `size` `"WxH"` (or
  `width`/`height`), `steps`, `seed`, `cfg`/`cfg_scale`, `negative_prompt`,
  `image` (base64 source) + `mode:"variation"` + `strength`, `lora_path` +
  `lora_scale`, `stream` (SSE progress). Returns
  `{"created":0,"data":[{"b64_json":"<png>"}]}`.
- `POST /v1/video/generations` — `prompt`, `num_frames`, `width`, `height`,
  `steps`, `seed`, `fps`, `cfg_scale`, `stg_scale`, `pipeline`
  (`one_stage`/`two_stage`/`two_stage_hq` — two-stage needs the LTX spatial
  upscaler sidecar), `stage2_steps`, `negative_prompt`, `first_frame_image` /
  `end_frame_image` (base64), `lora_path` + `lora_scale`, `stream`. Returns
  raw `rgb8` frames base64 plus `audio_*` PCM16 fields when the model
  generates sound (LTX).

## MCP & agent mode

The MCP screen offers one-tap presets (filesystem, GitHub, Playwright, shell,
databases, Slack, Notion, …) plus an editable `mcp.json` for custom servers.
Presets with options (folder path, tokens, DSN) collect them in an inline form
and stay editable after adding. With **agent mode** on, chat models can call
MCP tools and the built-ins (file read/write/list/search, shell) in a ReAct
loop. The system prompt is editable in-app.

## Headless verification

```sh
zig build
./zig-out/bin/zig-ai --chat-smoke "say hi" --model <chat.gguf>
./zig-out/bin/zig-ai --image-smoke "a cat" --model <sd.gguf> --out /tmp/x.ppm
./zig-out/bin/zig-ai --tts-smoke "hello" --tts-dir <tts-model-folder>
# Voice cloning: add a reference WAV.
./zig-out/bin/zig-ai --tts-smoke "hello" --tts-dir <tts-model-folder> --ref-wav voice.wav
# Wan: --t5xxl ; LTX: --llm + --audio-vae + --connectors. Optional:
# --vwidth/--vheight/--vframes/--vsteps. Writes /tmp/frame-000.ppm …
./zig-out/bin/zig-ai --video-smoke "a cat in a garden" \
    --diffusion <wan.gguf> --vae <wan2.2_vae.safetensors> --t5xxl <umt5.gguf> \
    --out /tmp/frame.ppm
./zig-out/bin/zig-ai --mcp-smoke                  # spawn configured MCP servers, list tools
./zig-out/bin/zig-ai --dl-smoke "qwen"            # HF search/tree round-trip, no download
./zig-out/bin/zig-ai --screenshot /tmp/shell.bmp --screen chat [--dark] [--mock]
```
