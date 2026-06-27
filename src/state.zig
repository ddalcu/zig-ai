//! The application state: every reactive `State(T)`, text buffer, and discovered
//! model lives here. `main` constructs one `AppState`, `body` reads it each frame
//! to rebuild the view, and event callbacks mutate it. Backend façades (llama /
//! sd / tts) are added in later phases.

const std = @import("std");
const zigui = @import("zigui");
const app = @import("zigui_app");
const models = @import("models.zig");
const config = @import("config.zig");
const codecs = @import("codecs/codecs.zig");
const channel = @import("channel.zig");
const mcp = @import("mcp.zig");
const agent = @import("agent.zig");
const manifest = @import("manifest.zig");
const chat_client = @import("backends/chat_client.zig");
const accel = @import("backends/accel.zig");
const sd = @import("backends/sd.zig");
const tts = @import("backends/tts.zig");
const video = @import("backends/video.zig");
const downloader = @import("backends/downloader.zig");
const audioplay = @import("audio.zig");

pub const HFRepo = downloader.HFRepo;
pub const RepoFile = downloader.RepoFile;

pub const Role = enum {
    system,
    user,
    assistant,
    tool,
    pub fn str(self: Role) []const u8 {
        return @tagName(self);
    }
};

/// A tool call captured from the server's structured `tool_calls` (owned).
pub const MsgToolCall = struct { name: []u8, args: []u8 };

/// User theme choice. The integer values are the `Picker` indices in Settings
/// (System / Light / Dark), so the enum doubles as the `theme_pref` State value.
pub const ThemePref = enum(i64) { system = 0, light = 1, dark = 2 };

/// Resolve a theme preference to an effective dark/light boolean, querying the
/// OS theme (cross-platform via SDL) when the choice is `.system`.
pub fn effectiveDark(pref: ThemePref) bool {
    return switch (pref) {
        .light => false,
        .dark => true,
        .system => app.systemTheme() == .dark,
    };
}

/// One chat message. Heap-allocated so a streaming assistant reply can be held
/// by a stable pointer while tokens append to its `content`.
pub const ChatMessage = struct {
    role: Role,
    content: std.ArrayList(u8) = .empty,
    /// Chain-of-thought, supplied separately by the server (`reasoning_content`).
    reasoning: std.ArrayList(u8) = .empty,
    /// The tool call this assistant turn made (from the server's `tool_calls`),
    /// or null. For a `.tool` message, `content` holds the result.
    tool_call: ?MsgToolCall = null,
    streaming: bool = false,
    tokens: u64 = 0,
    tps: f32 = 0,
    /// Reasoning disclosure state. null = default (expanded while thinking,
    /// collapsed once the answer arrives); set explicitly once the user taps it.
    think_expanded: ?bool = null,
    /// Tool-result disclosure state (collapsed by default).
    tool_expanded: bool = false,
    gpa: std.mem.Allocator,

    pub fn create(gpa: std.mem.Allocator, role: Role) !*ChatMessage {
        const m = try gpa.create(ChatMessage);
        m.* = .{ .role = role, .gpa = gpa };
        return m;
    }
    pub fn destroy(self: *ChatMessage) void {
        self.content.deinit(self.gpa);
        self.reasoning.deinit(self.gpa);
        if (self.tool_call) |tc| {
            self.gpa.free(tc.name);
            self.gpa.free(tc.args);
        }
        self.gpa.destroy(self);
    }
    pub fn setText(self: *ChatMessage, s: []const u8) !void {
        self.content.clearRetainingCapacity();
        try self.content.appendSlice(self.gpa, s);
    }
    pub fn setToolCall(self: *ChatMessage, name: []const u8, args: []const u8) void {
        const nd = self.gpa.dupe(u8, name) catch return;
        const ad = self.gpa.dupe(u8, args) catch {
            self.gpa.free(nd);
            return;
        };
        if (self.tool_call) |tc| {
            self.gpa.free(tc.name);
            self.gpa.free(tc.args);
        }
        self.tool_call = .{ .name = nd, .args = ad };
    }
};

/// Write `s` as a JSON string literal (with quotes), escaping per RFC 8259.
fn jsonStr(w: *std.Io.Writer, s: []const u8) void {
    w.writeByte('"') catch return;
    for (s) |ch| switch (ch) {
        '"' => w.writeAll("\\\"") catch return,
        '\\' => w.writeAll("\\\\") catch return,
        '\n' => w.writeAll("\\n") catch return,
        '\r' => w.writeAll("\\r") catch return,
        '\t' => w.writeAll("\\t") catch return,
        else => if (ch < 0x20) (w.print("\\u{x:0>4}", .{ch}) catch return) else (w.writeByte(ch) catch return),
    };
    w.writeByte('"') catch return;
}

/// Serialize one ChatMessage as an OpenAI message object (role/content, plus
/// `tool_calls` for an assistant call or `tool_call_id` for a tool result).
fn writeMsgObj(w: *std.Io.Writer, first: *bool, m: *ChatMessage) void {
    if (!first.*) w.writeAll(",") catch return;
    first.* = false;
    w.writeAll("{\"role\":") catch return;
    jsonStr(w, m.role.str());
    if (m.role == .tool) {
        w.writeAll(",\"tool_call_id\":\"call_0\",\"content\":") catch return;
        jsonStr(w, m.content.items);
    } else if (m.tool_call) |tc| {
        w.writeAll(",\"content\":") catch return;
        jsonStr(w, m.content.items);
        w.writeAll(",\"tool_calls\":[{\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":") catch return;
        jsonStr(w, tc.name);
        w.writeAll(",\"arguments\":") catch return;
        jsonStr(w, tc.args); // args is raw JSON; emit as a JSON string per OpenAI
        w.writeAll("}}]") catch return;
    } else {
        w.writeAll(",\"content\":") catch return;
        jsonStr(w, m.content.items);
    }
    w.writeAll("}") catch return;
}

/// Sidebar screens. The integer values back a `State(i64)` so the sidebar's
/// selection and the detail switch share one binding.
pub const Screen = enum(i64) {
    chat = 0,
    image = 1,
    video = 2,
    audio = 3,
    models = 4,
    logs = 5,
    settings = 6,
    mcp = 7,
    editor = 8,
};

/// Which config file the in-app text editor is currently editing.
pub const EditorTarget = enum {
    system_prompt,
    mcp_json,

    pub fn fileName(self: EditorTarget) []const u8 {
        return switch (self) {
            .system_prompt => config.system_prompt_file,
            .mcp_json => config.mcp_file,
        };
    }
    pub fn title(self: EditorTarget) []const u8 {
        return switch (self) {
            .system_prompt => "System Prompt",
            .mcp_json => "mcp.json",
        };
    }
};

/// Upper bound on a preset's configurable inputs (the inline config form sizes
/// its field array to this; the catalog never exceeds it).
pub const max_mcp_inputs = 4;

/// A bounded ring of log lines, shared by the Logs screen. Oldest lines are
/// dropped once `max` is reached.
pub const LogRing = struct {
    gpa: std.mem.Allocator,
    lines: std.ArrayList([]u8) = .empty,
    max: usize = 500,

    pub fn init(gpa: std.mem.Allocator) LogRing {
        return .{ .gpa = gpa };
    }
    pub fn deinit(self: *LogRing) void {
        for (self.lines.items) |l| self.gpa.free(l);
        self.lines.deinit(self.gpa);
    }
    pub fn append(self: *LogRing, line: []const u8) void {
        const dup = self.gpa.dupe(u8, line) catch return;
        self.lines.append(self.gpa, dup) catch {
            self.gpa.free(dup);
            return;
        };
        if (self.lines.items.len > self.max) {
            const old = self.lines.orderedRemove(0);
            self.gpa.free(old);
        }
    }
};

pub const AppState = struct {
    gpa: std.mem.Allocator,
    /// Reset every frame; cheap scratch space for `body`-time formatting.
    frame_arena: std.heap.ArenaAllocator,
    /// User home directory (owned), resolved from the process environment in
    /// `main`; used to locate default model folders. Null if unavailable.
    home: ?[]u8 = null,
    /// The process environment, captured in `loadConfig`; read by the CLI
    /// launcher for PATH / TMPDIR etc. (`std.process.Environ`).
    environ: std.process.Environ = undefined,

    // --- navigation -------------------------------------------------------
    screen: zigui.State(i64),

    // --- discovered models ------------------------------------------------
    model_list: models.ModelList,
    /// Guards `model_list` rebuilds vs cross-thread reads (the HTTP API server).
    model_list_lock: channel.SpinLock = .{},
    /// Selected chat model's training context cap (read from GGUF, cached by path).
    chat_ctx_cap: u32 = 32768,
    chat_ctx_cap_path: ?[]u8 = null,
    /// Model paths persisted from a previous run, resolved to the matching
    /// `sel_*` on the first model scan (see `rescanModels`). One per task so each
    /// screen's selection is sticky across runs.
    startup_chat_model: ?[]u8 = null,
    startup_sd_model: ?[]u8 = null,
    startup_video_model: ?[]u8 = null,
    startup_tts_model: ?[]u8 = null,
    /// Extra user-added scan directories (owned strings).
    model_dirs: std.ArrayList([]u8) = .empty,
    /// Selected model index into `model_list.items` (-1 = none) per task.
    sel_llm: zigui.State(i64),
    sel_sd: zigui.State(i64),
    sel_video: zigui.State(i64),
    sel_tts: zigui.State(i64),

    // --- server / status --------------------------------------------------
    /// Whether a chat model is currently loaded (drives the status dot).
    llm_loaded: bool = false,

    /// Display name (owned) of the model last submitted to each backend, captured
    /// at submit time, plus its on-disk size. Paired with the backend's
    /// `model_ready` atomic they drive the tray status rows and the RAM proxy
    /// (`loadedBytes`). Null until a generation has been requested for that kind.
    loaded_llm: ?[]u8 = null,
    loaded_sd: ?[]u8 = null,
    loaded_video: ?[]u8 = null,
    loaded_tts: ?[]u8 = null,

    /// Active compute backend + its memory, shown in the sidebar footer. Cached
    /// and refreshed every ~30 frames (and until backends finish loading) so the
    /// per-frame footer render doesn't poll the driver on every frame.
    accel_info: accel.Info = .{},
    accel_poll: u32 = 0,
    loaded_size_llm: u64 = 0,
    loaded_size_sd: u64 = 0,
    loaded_size_video: u64 = 0,
    loaded_size_tts: u64 = 0,

    // --- chat -------------------------------------------------------------
    chat_input: zigui.TextFieldState,
    chat_scroll: zigui.ScrollState = .{},
    messages: std.ArrayList(*ChatMessage) = .empty,
    /// The assistant message currently being streamed (borrowed; lives in
    /// `messages`), or null when idle.
    pending: ?*ChatMessage = null,
    /// Chat goes through the in-process HTTP server (the single chat engine).
    chat: chat_client.Backend,
    /// Transient "Copied" feedback: identity (text address) of the copy button
    /// last tapped, and the SDL-tick ms until which to show the confirmation.
    copied_key: usize = 0,
    copied_until_ms: u64 = 0,

    // --- agent / MCP ------------------------------------------------------
    /// When on, chat turns run the agentic tool loop (MCP tools advertised in
    /// the system prompt; `<tool_call>` blocks executed and fed back).
    agent_mode: zigui.State(bool),
    /// Whether the chat header's "Agent + MCP" capabilities popover is open.
    agent_tools_open: zigui.State(bool),
    /// Scroll offset for that popover's tool list.
    agent_tools_scroll: zigui.ScrollState = .{},
    /// The user-editable system prompt (owned). Loaded from `system-prompt.md`
    /// at startup; falls back to the built-in default.
    system_prompt: []u8 = &.{},
    /// MCP runtime: spawns the enabled servers and runs their tools.
    mcp_mgr: mcp.Manager,
    /// Agentic loop bookkeeping for the current user turn.
    agent_iters: u32 = 0,
    /// Monotonic id matching a dispatched tool call to its result event.
    agent_seq: u64 = 0,
    /// True while a tool call is in flight (keeps the loop awake; shows status).
    agent_busy: bool = false,
    /// Name of the tool currently running (for the chat status line).
    agent_tool_buf: [128]u8 = undefined,
    agent_tool_len: usize = 0,

    // --- in-app text editor (system-prompt.md / mcp.json) -----------------
    editor_buf: zigui.TextFieldState,
    editor_scroll: zigui.ScrollState = .{},
    editor_target: EditorTarget = .system_prompt,
    /// Transient "Saved" confirmation deadline (SDL ticks), 0 = not showing.
    editor_saved_until_ms: u64 = 0,

    // --- MCP preset config form -------------------------------------------
    /// Catalog index of the preset being configured inline (-1 = none). When a
    /// preset has inputs, "Add" opens this form to collect them.
    mcp_cfg_idx: i64 = -1,
    /// True when the open form edits an already-added server (pre-filled, Save
    /// rewrites the entry) rather than configuring a new one (Add appends).
    mcp_cfg_editing: bool = false,
    /// Input fields for the preset config form (one per `PresetInput`; presets
    /// have at most a couple, so a small fixed array is plenty).
    mcp_cfg_fields: [max_mcp_inputs]zigui.TextFieldState,

    // --- image ------------------------------------------------------------
    img_prompt: zigui.TextFieldState,
    img_steps: zigui.State(f32),
    img_cfg: zigui.State(f32),
    img_width: zigui.State(i64),
    img_height: zigui.State(i64),
    img_advanced: zigui.State(bool),
    img_scroll: zigui.ScrollState = .{},
    /// Last generated image, shown in the preview pane.
    img_result: ?zigui.canvas.Image = null,
    sd: sd.Backend,

    // --- video (Wan) ------------------------------------------------------
    vid_prompt: zigui.TextFieldState,
    vid_negative: zigui.TextFieldState,
    vid_neg_scroll: zigui.ScrollState = .{},
    vid_steps: zigui.State(f32),
    vid_cfg: zigui.State(f32),
    vid_frames_n: zigui.State(i64),
    /// Output size, split into two compact pickers (a single 5-way picker is too
    /// wide for the panel): orientation 0=landscape/1=portrait/2=square and
    /// quality 0=480p/1=720p. Combined by `videoSize`.
    vid_orient: zigui.State(i64),
    vid_quality: zigui.State(i64),
    vid_scroll: zigui.ScrollState = .{},
    /// Decoded frames of the last generation; cycled for playback in the view.
    vid_result: ?[]zigui.canvas.Image = null,
    vid_fps: i32 = 16,
    /// Frame currently shown; advanced each frame for simple playback.
    vid_play_idx: usize = 0,
    vid_play_tick: u32 = 0,
    /// Optional starting frame for image-to-video (Wan TI2V), decoded to RGBA8
    /// (owned by stb_image; freed via `codecs.freeImage`). `vid_image_pending`
    /// guards collecting the file-dialog result in `pumpVideo`.
    vid_init_image: ?codecs.DecodedImage = null,
    vid_image_pending: bool = false,
    video: video.Backend,

    // --- audio / tts ------------------------------------------------------
    tts_text: zigui.TextFieldState,
    tts_temperature: zigui.State(f32),
    tts_scroll: zigui.ScrollState = .{},
    /// Number of audio samples produced by the last synthesis (status display).
    tts_last_samples: usize = 0,
    tts: tts.Backend,
    player: audioplay.Player = .{},
    // Voice-clone reference: a WAV picked via the native file dialog, OR a clip
    // recorded from the mic (24 kHz mono f32 — settable straight into the
    // backend). Setting one clears the other; both empty = default voice.
    tts_ref_path: ?[]u8 = null,
    tts_rec: std.ArrayList(f32) = .empty,
    tts_recording: bool = false,
    recorder: audioplay.Recorder = .{},

    // --- settings ---------------------------------------------------------
    threads: zigui.State(i64),
    use_gpu: zigui.State(bool),
    // Chat sampling / context settings (applied to each new generation).
    chat_temp: zigui.State(f32),
    chat_top_p: zigui.State(f32),
    chat_top_k: zigui.State(i64),
    chat_n_ctx: zigui.State(i64),
    /// Theme preference; `.system` follows the OS light/dark setting live.
    theme_pref: zigui.State(i64),
    /// Visual theme family — an index into `zigui.theme_registry.all`
    /// (macOS, Windows 10, …). The light/dark palette is chosen from `theme_pref`.
    theme_family: zigui.State(i64),
    new_dir: zigui.TextFieldState,

    // --- models -----------------------------------------------------------
    models_scroll: zigui.ScrollState = .{},
    /// Which tab of the Models screen is showing: 0..3 = Chat/Image/Video/TTS
    /// (local models of that kind), 4 = Download (HuggingFace).
    models_tab: zigui.State(i64),

    // --- header model-switcher popover (one shared pair; only one screen
    // renders at a time) --------------------------------------------------
    model_picker_open: zigui.State(bool),
    model_picker_scroll: zigui.ScrollState = .{},

    // --- HuggingFace downloader -------------------------------------------
    downloader: downloader.Backend,
    dl_search: zigui.TextFieldState,
    dl_category: zigui.State(i64),
    dl_results: std.ArrayList(HFRepo) = .empty,
    /// Files of the repo whose quant popover was last opened (owned), or null.
    dl_files: ?downloader.Files = null,
    dl_scroll: zigui.ScrollState = .{},
    /// Scroll offset for the stacked active-download progress cards (so many
    /// parallel downloads scroll within a bounded area instead of clipping).
    dl_jobs_scroll: zigui.ScrollState = .{},
    /// True while a HuggingFace search is in flight (drives the loading
    /// indicator); cleared when results or an error arrive.
    dl_searching: bool = false,
    /// Which column the results table is sorted by (index = -1 keeps the HF API's
    /// own download-ranked order until the user clicks a header).
    dl_sort: zigui.State(zigui.SortColumn),
    /// The result row whose quant popover is active (-1 = none).
    dl_filepick_idx: i64 = -1,
    dl_filepick_open: zigui.State(bool),
    // Per-download progress lives in `downloader.jobs` (one `DlJob` each), so
    // several models can download in parallel — each with its own progress card.

    // --- logs -------------------------------------------------------------
    logs: LogRing,
    log_scroll: zigui.ScrollState = .{},
    settings_scroll: zigui.ScrollState = .{},

    // --- overlays ---------------------------------------------------------
    show_settings: zigui.State(bool),
    /// A modal alert: shown while `alert_present` is true, with `alert_buf` text.
    alert_present: zigui.State(bool),
    alert_buf: [512]u8 = undefined,
    alert_len: usize = 0,
    /// Modal title — defaults to the error wording; `alertOk` swaps in a neutral
    /// one for success notices. Points at a static string (no buffer needed).
    alert_title: []const u8 = "Something went wrong",

    /// Delete-model confirmation overlay. `delete_path` is the file or folder that
    /// will be removed; `delete_is_folder` picks deleteTree vs deleteFile.
    delete_present: zigui.State(bool),
    delete_path: ?[]u8 = null,
    delete_is_folder: bool = false,
    delete_name_buf: [160]u8 = undefined,
    delete_name_len: usize = 0,
    /// File count + total size of what will be removed (for the dialog).
    delete_files: usize = 0,
    delete_bytes: u64 = 0,

    /// Initialize all reactive fields. Call `deinit` to release them.
    pub fn init(gpa: std.mem.Allocator) AppState {
        return .{
            .gpa = gpa,
            .frame_arena = std.heap.ArenaAllocator.init(gpa),
            .screen = zigui.State(i64).init(gpa, @intFromEnum(Screen.chat)),
            .model_list = models.ModelList.init(gpa),
            .sel_llm = zigui.State(i64).init(gpa, -1),
            .sel_sd = zigui.State(i64).init(gpa, -1),
            .sel_video = zigui.State(i64).init(gpa, -1),
            .sel_tts = zigui.State(i64).init(gpa, -1),
            .chat_input = zigui.TextFieldState.init(gpa),
            .chat = chat_client.Backend.init(gpa),
            .agent_mode = zigui.State(bool).init(gpa, false),
            .agent_tools_open = zigui.State(bool).init(gpa, false),
            .mcp_mgr = mcp.Manager.init(gpa),
            .editor_buf = zigui.TextFieldState.init(gpa),
            .mcp_cfg_fields = .{
                zigui.TextFieldState.init(gpa),
                zigui.TextFieldState.init(gpa),
                zigui.TextFieldState.init(gpa),
                zigui.TextFieldState.init(gpa),
            },
            .img_prompt = zigui.TextFieldState.init(gpa),
            .img_steps = zigui.State(f32).init(gpa, 20),
            .img_cfg = zigui.State(f32).init(gpa, 7),
            .img_width = zigui.State(i64).init(gpa, 512),
            .img_height = zigui.State(i64).init(gpa, 512),
            .img_advanced = zigui.State(bool).init(gpa, false),
            .sd = sd.Backend.init(gpa),
            .vid_prompt = zigui.TextFieldState.init(gpa),
            .vid_negative = zigui.TextFieldState.init(gpa),
            .vid_steps = zigui.State(f32).init(gpa, 30),
            .vid_cfg = zigui.State(f32).init(gpa, 5.0),
            .vid_frames_n = zigui.State(i64).init(gpa, 49),
            .vid_orient = zigui.State(i64).init(gpa, 0),
            .vid_quality = zigui.State(i64).init(gpa, 0),
            .video = video.Backend.init(gpa),
            .tts = tts.Backend.init(gpa),
            .tts_text = zigui.TextFieldState.init(gpa),
            .tts_temperature = zigui.State(f32).init(gpa, 0.9),
            .threads = zigui.State(i64).init(gpa, 4),
            .use_gpu = zigui.State(bool).init(gpa, true),
            .chat_temp = zigui.State(f32).init(gpa, 0.7),
            .chat_top_p = zigui.State(f32).init(gpa, 0.95),
            .chat_top_k = zigui.State(i64).init(gpa, 40),
            .chat_n_ctx = zigui.State(i64).init(gpa, 16384),
            .theme_pref = zigui.State(i64).init(gpa, @intFromEnum(ThemePref.system)),
            .theme_family = zigui.State(i64).init(gpa, 0), // 0 = macOS (theme_registry.all[0])
            .new_dir = zigui.TextFieldState.init(gpa),
            .models_tab = zigui.State(i64).init(gpa, 0),
            .model_picker_open = zigui.State(bool).init(gpa, false),
            .downloader = downloader.Backend.init(gpa),
            .dl_search = zigui.TextFieldState.init(gpa),
            .dl_category = zigui.State(i64).init(gpa, 0),
            .dl_sort = zigui.State(zigui.SortColumn).init(gpa, .{}),
            .dl_filepick_open = zigui.State(bool).init(gpa, false),
            .logs = LogRing.init(gpa),
            .show_settings = zigui.State(bool).init(gpa, false),
            .alert_present = zigui.State(bool).init(gpa, false),
            .delete_present = zigui.State(bool).init(gpa, false),
        };
    }

    pub fn deinit(self: *AppState) void {
        if (self.home) |h| self.gpa.free(h);
        self.screen.deinit();
        self.model_list.deinit();
        if (self.chat_ctx_cap_path) |p| self.gpa.free(p);
        if (self.startup_chat_model) |p| self.gpa.free(p);
        if (self.startup_sd_model) |p| self.gpa.free(p);
        if (self.startup_video_model) |p| self.gpa.free(p);
        if (self.startup_tts_model) |p| self.gpa.free(p);
        for (self.model_dirs.items) |d| self.gpa.free(d);
        self.model_dirs.deinit(self.gpa);
        self.sel_llm.deinit();
        self.sel_sd.deinit();
        self.sel_video.deinit();
        self.sel_tts.deinit();
        self.chat.deinit();
        self.sd.deinit();
        self.tts.deinit();
        self.agent_mode.deinit();
        self.agent_tools_open.deinit();
        self.mcp_mgr.deinit();
        self.editor_buf.deinit();
        for (&self.mcp_cfg_fields) |*f| f.deinit();
        if (self.system_prompt.len > 0) self.gpa.free(self.system_prompt);
        if (self.loaded_llm) |s| self.gpa.free(s);
        if (self.loaded_sd) |s| self.gpa.free(s);
        if (self.loaded_video) |s| self.gpa.free(s);
        if (self.loaded_tts) |s| self.gpa.free(s);
        self.player.close();
        self.recorder.stop();
        if (self.tts_ref_path) |p| self.gpa.free(p);
        self.tts_rec.deinit(self.gpa);
        for (self.messages.items) |m| m.destroy();
        self.messages.deinit(self.gpa);
        self.chat_input.deinit();
        self.img_prompt.deinit();
        self.img_steps.deinit();
        self.img_cfg.deinit();
        self.img_width.deinit();
        self.img_height.deinit();
        self.img_advanced.deinit();
        if (self.img_result) |img| self.gpa.free(@constCast(img.pixels));
        self.vid_prompt.deinit();
        self.vid_negative.deinit();
        self.vid_steps.deinit();
        self.vid_cfg.deinit();
        self.vid_frames_n.deinit();
        self.vid_orient.deinit();
        self.vid_quality.deinit();
        self.freeVideoResult();
        self.clearVideoImage();
        self.video.deinit();
        self.tts_text.deinit();
        self.tts_temperature.deinit();
        self.threads.deinit();
        self.use_gpu.deinit();
        self.chat_temp.deinit();
        self.chat_top_p.deinit();
        self.chat_top_k.deinit();
        self.chat_n_ctx.deinit();
        self.theme_pref.deinit();
        self.theme_family.deinit();
        self.new_dir.deinit();
        self.models_tab.deinit();
        self.model_picker_open.deinit();
        self.downloader.deinit();
        self.dl_search.deinit();
        self.dl_category.deinit();
        self.dl_sort.deinit();
        self.dl_filepick_open.deinit();
        self.clearDlResults();
        self.dl_results.deinit(self.gpa);
        self.clearDlFiles();
        self.logs.deinit();
        self.show_settings.deinit();
        self.alert_present.deinit();
        self.delete_present.deinit();
        if (self.delete_path) |p| self.gpa.free(p);
        self.frame_arena.deinit();
    }

    /// Rescan all model directories into `model_list`. Guarded so the HTTP API
    /// server (another thread) can safely read the selected model's path.
    ///
    /// Selections are stored as indices into `model_list`, which the rescan
    /// reshuffles — so we snapshot each task's selected *path* first, rebuild the
    /// list, then re-resolve every selection by path (`reselect`). That keeps the
    /// current pick sticky across rescans (e.g. after a download) and, when a task
    /// has no valid selection, auto-picks the first model of that kind.
    pub fn rescanModels(self: *AppState) void {
        self.model_list_lock.lock();
        defer self.model_list_lock.unlock();

        // Snapshot current selections by path before the indices go stale.
        var kept: [4]?[]u8 = .{ null, null, null, null };
        const kinds = [4]models.Kind{ .text, .image, .video, .tts };
        for (kinds, 0..) |k, i| {
            if (self.selectedModelOfKind(k)) |m| kept[i] = self.gpa.dupe(u8, m.path) catch null;
        }
        defer for (kept) |p| {
            if (p) |s| self.gpa.free(s);
        };

        self.model_list.clear();
        models.scanDefaults(&self.model_list, self.home, self.model_dirs.items);
        for (self.model_list.items.items) |*m| m.complete = self.modelComplete(m.*);

        self.reselect(.text, &self.sel_llm, kept[0], &self.startup_chat_model);
        self.reselect(.image, &self.sel_sd, kept[1], &self.startup_sd_model);
        self.reselect(.video, &self.sel_video, kept[2], &self.startup_video_model);
        self.reselect(.tts, &self.sel_tts, kept[3], &self.startup_tts_model);
    }

    /// The selected model for `kind` (validated to actually be that kind), or null.
    fn selectedModelOfKind(self: *AppState, kind: models.Kind) ?models.ModelInfo {
        const sel: i64 = switch (kind) {
            .text => self.sel_llm.get(),
            .image => self.sel_sd.get(),
            .video => self.sel_video.get(),
            .tts => self.sel_tts.get(),
        };
        const m = self.selectedModel(sel) orelse return null;
        return if (m.kind == kind) m else null;
    }

    /// Re-point one task's selection after a rescan: prefer the path it had before
    /// (`kept`), then the persisted startup path, then auto-select the first model
    /// of `kind`. The startup slot is consumed (freed) on the first scan.
    fn reselect(self: *AppState, kind: models.Kind, sel: *zigui.State(i64), kept: ?[]const u8, startup: *?[]u8) void {
        sel.set(-1);
        const want: ?[]const u8 = kept orelse startup.*;
        if (want) |p| {
            for (self.model_list.items.items, 0..) |m, i| {
                if (m.kind == kind and std.mem.eql(u8, m.path, p)) {
                    sel.set(@intCast(i));
                    break;
                }
            }
        }
        if (startup.*) |s| {
            self.gpa.free(s);
            startup.* = null;
        }
        // Auto-select the first model of this kind if nothing resolved.
        if (sel.get() < 0) {
            for (self.model_list.items.items, 0..) |m, i| {
                if (m.kind == kind) {
                    sel.set(@intCast(i));
                    break;
                }
            }
        }
    }

    /// Decide whether an owned model's folder is a finished download: no leftover
    /// `.partial` file, and every curated sidecar the repo needs is present.
    /// Non-owned models (LM Studio, etc.) are always treated as complete. Runs
    /// once per rescan so the per-frame UI just reads `ModelInfo.complete`.
    fn modelComplete(self: *AppState, m: models.ModelInfo) bool {
        if (m.source != .zig_ai) return true;
        if (models.hasPartial(self.gpa, m.dir)) return false;
        // Reconstruct the repo id ("author/name") from the folder layout
        // (<models>/<kind>/<author>/<name>) to look up its expected sidecars.
        const name = std.fs.path.basename(m.dir);
        const parent = std.fs.path.dirname(m.dir) orelse return true;
        const author = std.fs.path.basename(parent);
        const repo_id = std.fmt.allocPrint(self.gpa, "{s}/{s}", .{ author, name }) catch return true;
        defer self.gpa.free(repo_id);
        for (manifest.sidecarsFor(repo_id)) |sc| {
            const dest = sc.dest orelse std.fs.path.basename(sc.file);
            if (!models.fileExistsIn(self.gpa, m.dir, dest)) return false;
        }
        return true;
    }

    /// Copy the currently-selected chat model's path into `out` (thread-safe; for
    /// the HTTP API server). Returns the slice, or null if no chat model is
    /// selected. Reads under `model_list_lock` so it can't race `rescanModels`.
    pub fn apiModelPath(self: *AppState, out: []u8) ?[]const u8 {
        self.model_list_lock.lock();
        defer self.model_list_lock.unlock();
        const m = self.selectedModel(self.sel_llm.get()) orelse return null;
        const n = @min(m.path.len, out.len);
        @memcpy(out[0..n], m.path[0..n]);
        return out[0..n];
    }

    /// Selected model of a given kind, or null. `sel` holds an index into
    /// `model_list.items`.
    pub fn selectedModel(self: *AppState, sel: i64) ?models.ModelInfo {
        if (sel < 0) return null;
        const idx: usize = @intCast(sel);
        if (idx >= self.model_list.items.items.len) return null;
        return self.model_list.items.items[idx];
    }

    /// Training context cap of the selected chat model, read from its GGUF and
    /// cached by path (re-read when the selection changes). Falls back to 32768.
    pub fn chatCtxCap(self: *AppState) u32 {
        const m = self.selectedModel(self.sel_llm.get()) orelse return 32768;
        const stale = self.chat_ctx_cap_path == null or !std.mem.eql(u8, self.chat_ctx_cap_path.?, m.path);
        if (stale) {
            if (self.chat_ctx_cap_path) |p| self.gpa.free(p);
            self.chat_ctx_cap_path = self.gpa.dupe(u8, m.path) catch null;
            self.chat_ctx_cap = models.readCtxCap(self.gpa, m.path) orelse 32768;
        }
        return self.chat_ctx_cap;
    }

    /// Start a fresh conversation (after any in-flight generation is cancelled).
    pub fn newChat(self: *AppState) void {
        self.chat.cancel();
        for (self.messages.items) |m| m.destroy();
        self.messages.clearRetainingCapacity();
        self.pending = null;
        self.agent_busy = false;
        self.agent_iters = 0;
        self.agent_tool_len = 0;
    }

    /// Send the current input as a user message and kick off generation.
    pub fn sendChat(self: *AppState) void {
        if (self.pending != null) return; // already generating
        const text = std.mem.trim(u8, self.chat_input.text(), " \t\n");
        if (text.len == 0) return;
        if (self.selectedModel(self.sel_llm.get()) == null) {
            self.appendNotice("No chat model selected. Pick one in the Models tab.");
            return;
        }

        // Append the user message; a fresh user turn resets the agent loop.
        const um = ChatMessage.create(self.gpa, .user) catch return;
        um.setText(text) catch {};
        self.messages.append(self.gpa, um) catch {
            um.destroy();
            return;
        };
        self.chat_input.setText("") catch {};
        self.agent_iters = 0;
        // Jump to the bottom so the new message + the thinking indicator are
        // visible (clamped to the real max on the next render).
        self.chat_scroll.offset = 1_000_000;
        self.startGeneration();
    }

    /// Append a streaming assistant placeholder and submit the whole conversation
    /// (prefixed with the system prompt, and — in agent mode — the live tool
    /// catalogue) to the llama worker. Shared by `sendChat` and the agent loop.
    fn startGeneration(self: *AppState) void {
        if (self.pending != null) return;
        const model = self.selectedModel(self.sel_llm.get()) orelse return;

        const am = ChatMessage.create(self.gpa, .assistant) catch return;
        am.streaming = true;
        self.messages.append(self.gpa, am) catch {
            am.destroy();
            return;
        };
        self.pending = am;

        const fa = self.frame_arena.allocator();
        const body = self.buildChatRequest(fa, model) orelse {
            am.streaming = false;
            self.pending = null;
            return;
        };
        self.rememberLoaded(&self.loaded_llm, &self.loaded_size_llm, model);
        self.chat.start() catch {};
        self.chat.submit(body) catch {
            am.streaming = false;
            self.pending = null;
        };
    }

    /// Build the OpenAI `/v1/chat/completions` request body for the current
    /// conversation. The raw system prompt is sent as-is; agent-mode tools are
    /// sent as `tools[]` so the SERVER injects the tool prompt (single engine).
    fn buildChatRequest(self: *AppState, fa: std.mem.Allocator, model: models.ModelInfo) ?[]const u8 {
        var b: std.Io.Writer.Allocating = .init(fa);
        const w = &b.writer;
        w.writeAll("{\"model\":") catch return null;
        jsonStr(w, model.name);
        w.writeAll(",\"stream\":true,\"messages\":[") catch return null;
        var first = true;
        if (self.system_prompt.len > 0) {
            w.writeAll("{\"role\":\"system\",\"content\":") catch {};
            jsonStr(w, self.system_prompt);
            w.writeAll("}") catch {};
            first = false;
        }
        const n = self.messages.items.len - 1; // exclude the empty assistant placeholder
        for (self.messages.items[0..n]) |m| writeMsgObj(w, &first, m);
        w.writeAll("]") catch {};

        if (self.agent_mode.get()) {
            const tools = self.mcp_mgr.toolListAlloc(fa);
            if (tools.len > 0) {
                w.writeAll(",\"tools\":[") catch {};
                for (tools, 0..) |t, i| {
                    if (i > 0) w.writeAll(",") catch {};
                    w.writeAll("{\"type\":\"function\",\"function\":{\"name\":") catch {};
                    jsonStr(w, t.qualified);
                    if (t.description.len > 0) {
                        w.writeAll(",\"description\":") catch {};
                        jsonStr(w, t.description);
                    }
                    w.writeAll(",\"parameters\":") catch {};
                    if (t.schema.len > 0 and !std.mem.eql(u8, t.schema, "{}")) w.writeAll(t.schema) catch {} else w.writeAll("{\"type\":\"object\"}") catch {};
                    w.writeAll("}}") catch {};
                }
                w.writeAll("]") catch {};
            }
        }
        // Cap the context at the model's trained limit (n_ctx beyond it is wasted
        // memory and can degrade quality).
        const n_ctx = @min(@as(u32, @intCast(@max(self.chat_n_ctx.get(), 0))), self.chatCtxCap());
        w.print(",\"temperature\":{d},\"top_p\":{d},\"top_k\":{d},\"n_ctx\":{d},\"n_threads\":{d}}}", .{
            self.chat_temp.get(), self.chat_top_p.get(), self.chat_top_k.get(), n_ctx, self.threads.get(),
        }) catch return null;
        return b.written();
    }

    /// Kick off image generation from the current prompt + controls.
    pub fn generateImage(self: *AppState) void {
        if (self.sd.isBusy()) return;
        const model = self.selectedModel(self.sel_sd.get()) orelse {
            self.alert("No image model selected. Pick one in the Models tab.");
            return;
        };
        const prompt = self.img_prompt.text();
        if (std.mem.trim(u8, prompt, " \t\n").len == 0) return;

        // A classic SD checkpoint is one self-contained file. FLUX is split: the
        // diffusion file plus a VAE and a text encoder (FLUX.2 = a Qwen3 LLM,
        // FLUX.1 = CLIP-L + T5-XXL), discovered as sidecars in the model folder.
        var vae: ?[]u8 = null;
        var enc1: ?[]u8 = null;
        var enc2: ?[]u8 = null;
        defer if (vae) |s| self.gpa.free(s);
        defer if (enc1) |s| self.gpa.free(s);
        defer if (enc2) |s| self.gpa.free(s);

        var spec: sd.ModelSpec = .{ .model = model.path };
        if (std.ascii.indexOfIgnoreCase(model.name, "flux") != null) {
            vae = models.findSupport(self.gpa, model.dir, &.{ "vae", "ae." }, &.{ ".safetensors", ".gguf" }, &.{"audio"});
            if (vae == null) {
                self.alertf("FLUX needs a VAE (e.g. flux2-vae.safetensors) next to {s}", .{model.dir});
                return;
            }
            // FLUX.2 ships a Qwen3 LLM text encoder; FLUX.1 uses CLIP-L + T5-XXL.
            enc1 = models.findSupport(self.gpa, model.dir, &.{"qwen"}, &.{ ".gguf", ".safetensors" }, &.{});
            if (enc1) |llm| {
                spec = .{ .diffusion = model.path, .vae = vae.?, .llm = llm };
            } else {
                enc1 = models.findSupport(self.gpa, model.dir, &.{ "clip_l", "clip-l" }, &.{ ".safetensors", ".gguf" }, &.{});
                enc2 = models.findSupport(self.gpa, model.dir, &.{ "t5xxl", "t5-xxl" }, &.{ ".safetensors", ".gguf" }, &.{});
                if (enc1 == null or enc2 == null) {
                    self.alertf("FLUX needs a text encoder next to {s}: Qwen3 (FLUX.2) or CLIP-L + T5-XXL (FLUX.1).", .{model.dir});
                    return;
                }
                spec = .{ .diffusion = model.path, .vae = vae.?, .clip_l = enc1.?, .t5xxl = enc2.? };
            }
        }

        self.logf("image: generating with {s}…", .{model.name});
        self.rememberLoaded(&self.loaded_sd, &self.loaded_size_sd, model);
        self.sd.start() catch {};
        self.sd.submit(spec, prompt, "", .{
            .steps = @intFromFloat(self.img_steps.get()),
            .cfg = self.img_cfg.get(),
            .width = @intCast(self.img_width.get()),
            .height = @intCast(self.img_height.get()),
            .seed = -1,
            .n_threads = @intCast(self.threads.get()),
        }) catch {};
    }

    /// Drain sd events: progress drives the bar (via atomics), final image
    /// replaces the preview. Call once per frame.
    pub fn pumpImage(self: *AppState) void {
        var tmp: std.ArrayList(sd.Event) = .empty;
        defer tmp.deinit(self.gpa);
        self.sd.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .progress => {}, // read live from self.sd.job atomics in the view
            .image => |img| {
                if (self.img_result) |old| self.gpa.free(@constCast(old.pixels));
                self.img_result = img;
                self.logf("image: generated {d}x{d}", .{ img.width, img.height });
                // Auto-save to the outputs folder so it survives + can be opened.
                if (self.home) |home| {
                    if (config.outputsPathAlloc(self.gpa, home, "image", "png")) |p| {
                        defer self.gpa.free(p);
                        if (self.gpa.dupeZ(u8, p)) |pz| {
                            defer self.gpa.free(pz);
                            if (codecs.writePng(pz, img.pixels, img.width, img.height))
                                self.logf("image: saved {s}", .{p})
                            else
                                self.logf("image: save failed", .{});
                        } else |_| {}
                    }
                }
            },
            .err => |e| {
                self.alert(e);
                self.gpa.free(e);
            },
        };
    }

    /// Reveal the outputs folder (where generations are auto-saved) in the OS file
    /// manager. Creates it first so opening never fails on a missing dir.
    pub fn openOutputsFolder(self: *AppState) void {
        const home = self.home orelse return;
        const dir = config.outputsDirAlloc(self.gpa, home) orelse return;
        defer self.gpa.free(dir);
        config.ensureDir(self.gpa, dir);
        const url = config.fileUrlAlloc(self.gpa, dir) orelse return;
        defer self.gpa.free(url);
        _ = app.openUrl(url.ptr);
    }

    /// Encode the just-generated frames to an MP4 in the outputs folder (H.264 via
    /// the vendored minih264/minimp4 — no ffmpeg). Runs on the UI thread; fast for
    /// the short clips we generate.
    fn saveVideoMp4(self: *AppState, frames: []const zigui.canvas.Image, fps: i32) void {
        if (frames.len == 0) return;
        const home = self.home orelse return;
        const ptrs = self.gpa.alloc([*]const u8, frames.len) catch return;
        defer self.gpa.free(ptrs);
        for (frames, 0..) |fr, i| ptrs[i] = fr.pixels.ptr;
        const path = config.outputsPathAlloc(self.gpa, home, "video", "mp4") orelse return;
        defer self.gpa.free(path);
        const pz = self.gpa.dupeZ(u8, path) catch return;
        defer self.gpa.free(pz);
        const f: u32 = if (fps > 0) @intCast(fps) else 8;
        if (codecs.encodeMp4(pz, ptrs, frames[0].width, frames[0].height, f))
            self.logf("video: saved {s}", .{path})
        else
            self.logf("video: save failed", .{});
    }

    fn freeVideoResult(self: *AppState) void {
        if (self.vid_result) |frames| {
            for (frames) |fr| self.gpa.free(@constCast(fr.pixels));
            self.gpa.free(frames);
            self.vid_result = null;
        }
        self.vid_play_idx = 0;
        self.vid_play_tick = 0;
    }

    /// Generate a Wan video from the selected video model. The diffusion .gguf is
    /// the selected model; its VAE and umt5 text-encoder sidecars are discovered
    /// by name in the same directory tree.
    pub fn generateVideo(self: *AppState) void {
        if (self.video.isBusy()) return;
        const model = self.selectedModel(self.sel_video.get()) orelse {
            self.alert("No video model selected. Pick a Wan or LTX model in the Models tab.");
            return;
        };
        const prompt = self.vid_prompt.text();
        if (std.mem.trim(u8, prompt, " \t\n").len == 0) return;

        // The video VAE: a *.safetensors/.gguf with "vae" but not "audio".
        const vae = models.findSupport(self.gpa, model.dir, &.{"vae"}, &.{ ".safetensors", ".gguf" }, &.{"audio"}) orelse {
            self.alertf("No video VAE found near {s}", .{model.dir});
            return;
        };
        defer self.gpa.free(vae);

        // Distinguish Wan (umt5 text encoder) from LTX (Gemma LLM + audio VAE +
        // embeddings connectors) by which sidecars are present.
        const gemma = models.findSupport(self.gpa, model.dir, &.{"gemma"}, &.{".gguf"}, &.{});
        defer if (gemma) |g| self.gpa.free(g);

        var spec: video.ModelSpec = .{ .diffusion = model.path, .vae = vae };
        var t5: ?[]u8 = null;
        var avae: ?[]u8 = null;
        var conn: ?[]u8 = null;
        defer if (t5) |s| self.gpa.free(s);
        defer if (avae) |s| self.gpa.free(s);
        defer if (conn) |s| self.gpa.free(s);

        if (gemma) |g| {
            // LTX.
            avae = models.findSupport(self.gpa, model.dir, &.{"audio_vae"}, &.{ ".safetensors", ".gguf" }, &.{});
            conn = models.findSupport(self.gpa, model.dir, &.{ "connector", "connectors" }, &.{ ".safetensors", ".gguf" }, &.{});
            if (avae == null or conn == null) {
                self.alertf("LTX needs an audio-VAE + connectors next to {s}", .{model.dir});
                return;
            }
            spec.llm = g;
            spec.audio_vae = avae;
            spec.connectors = conn;
        } else {
            // Wan: umt5/t5xxl text encoder.
            t5 = models.findSupport(self.gpa, model.dir, &.{ "umt5", "t5xxl", "t5-xxl" }, &.{".gguf"}, &.{});
            if (t5 == null) {
                self.alertf("No umt5/t5xxl encoder found near {s}", .{model.dir});
                return;
            }
            spec.t5xxl = t5;
        }

        const res = self.videoSize();
        // Wan needs a real frame size (it's trained at ~480–720p); tiny sizes give
        // mush. flow_shift follows Wan's guidance: 5 at 720p, 3 at 480p.
        const flow_shift: f32 = if (@max(res.w, res.h) >= 720) 5.0 else 3.0;
        // A negative prompt matters a lot for Wan quality; fall back to a sensible
        // default when the user left the field empty.
        const neg_in = std.mem.trim(u8, self.vid_negative.text(), " \t\n");
        const neg = if (neg_in.len > 0) neg_in else default_video_negative;

        self.logf("video: generating with {s} ({d}x{d})…", .{ model.name, res.w, res.h });
        self.rememberLoaded(&self.loaded_video, &self.loaded_size_video, model);
        // Optional image-to-video starting frame (Wan TI2V).
        const init_img: ?video.InitImage = if (self.vid_init_image) |im| .{
            .width = im.width,
            .height = im.height,
            .rgba = im.pixels[0 .. @as(usize, im.width) * @as(usize, im.height) * 4],
        } else null;
        self.video.start() catch {};
        self.video.submit(spec, prompt, neg, .{
            .steps = @intFromFloat(self.vid_steps.get()),
            .cfg = self.vid_cfg.get(),
            .flow_shift = flow_shift,
            .frames = @intCast(self.vid_frames_n.get()),
            .width = res.w,
            .height = res.h,
            .seed = -1,
            .n_threads = @intCast(self.threads.get()),
        }, init_img) catch {};
    }

    /// Output frame size from the orientation + quality pickers. Wan TI2V-5B is
    /// trained around 480–720p; 480p landscape is the default (a good 16 GB
    /// balance — 720p is heavier and may not fit on smaller cards).
    pub fn videoSize(self: *AppState) struct { w: i32, h: i32 } {
        const hd = self.vid_quality.get() == 1;
        return switch (self.vid_orient.get()) {
            1 => if (hd) .{ .w = 704, .h = 1280 } else .{ .w = 480, .h = 832 }, // portrait
            2 => if (hd) .{ .w = 960, .h = 960 } else .{ .w = 640, .h = 640 }, // square
            else => if (hd) .{ .w = 1280, .h = 704 } else .{ .w = 832, .h = 480 }, // landscape
        };
    }

    /// Default negative prompt for video (Wan's standard quality-suppression list).
    /// Used when the user leaves the negative field blank.
    pub const default_video_negative =
        "low quality, worst quality, blurry, jpeg artifacts, overexposed, static, " ++
        "still image, watermark, subtitles, text, deformed, disfigured, extra limbs, " ++
        "fused fingers, messy background, ugly";

    /// Filters for the video init-frame picker (kept static — SDL holds the
    /// pointer until the async dialog is dismissed).
    const image_filters = [_]app.FileFilter{
        .{ .name = "Images", .pattern = "png;jpg;jpeg;webp;bmp" },
    };

    /// Open the native dialog to pick a starting image for image-to-video. The
    /// result is collected (and decoded) in `pumpVideo`.
    pub fn chooseVideoImage(self: *AppState) void {
        self.vid_image_pending = true;
        _ = app.openFileDialog(&image_filters, null);
    }

    /// Drop the image-to-video starting frame.
    pub fn clearVideoImage(self: *AppState) void {
        if (self.vid_init_image) |im| codecs.freeImage(im);
        self.vid_init_image = null;
    }

    /// Drain video events: progress drives the bar (atomics); final frames replace
    /// the playback buffer. Call once per frame.
    pub fn pumpVideo(self: *AppState) void {
        // Collect the init-frame file pick (only when we opened that dialog, so we
        // don't steal the TTS reference-WAV pick that `pumpAudio` handles).
        if (self.vid_image_pending) {
            switch (app.takeFileDialogResult(self.gpa)) {
                .none => {}, // dialog still open; keep waiting
                .canceled => self.vid_image_pending = false,
                .picked => |p| {
                    defer self.gpa.free(p);
                    self.vid_image_pending = false;
                    const pz = self.gpa.dupeZ(u8, p) catch return;
                    defer self.gpa.free(pz);
                    if (codecs.loadImage(pz)) |img| {
                        self.clearVideoImage();
                        self.vid_init_image = img;
                        self.logf("video: start frame {d}x{d} from {s}", .{ img.width, img.height, p });
                    } else self.alert("Could not decode that image.");
                },
            }
        }
        var tmp: std.ArrayList(video.Event) = .empty;
        defer tmp.deinit(self.gpa);
        self.video.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .progress => {}, // read live from self.video.job atomics in the view
            .frames => |f| {
                self.freeVideoResult();
                self.vid_result = f.images;
                self.vid_fps = f.fps;
                self.logf("video: generated {d} frames", .{f.images.len});
                self.saveVideoMp4(f.images, f.fps);
            },
            .err => |e| {
                self.alert(e);
                self.gpa.free(e);
            },
        };

        // Advance simple frame playback: step once every ~4 UI frames so a 16 fps
        // clip plays at roughly real time on a 60 fps loop.
        if (self.vid_result) |frames| {
            if (frames.len > 1) {
                self.vid_play_tick +%= 1;
                if (self.vid_play_tick % 4 == 0)
                    self.vid_play_idx = (self.vid_play_idx + 1) % frames.len;
            }
        }
    }

    /// Synthesize the current TTS text and play it back.
    pub fn synthesize(self: *AppState) void {
        if (self.tts.isBusy()) return;
        const model = self.selectedModel(self.sel_tts.get()) orelse {
            self.alert("No TTS model selected. Pick one in the Models tab.");
            return;
        };
        const text = self.tts_text.text();
        if (std.mem.trim(u8, text, " \t\n").len == 0) return;
        self.rememberLoaded(&self.loaded_tts, &self.loaded_size_tts, model);
        self.tts.start() catch {};
        // Clone the chosen WAV / recorded clip when one is set, else the
        // default voice.
        const ref: tts.Ref = if (self.tts_ref_path) |p|
            .{ .file = p }
        else if (self.tts_rec.items.len > 0)
            .{ .samples = self.tts_rec.items }
        else
            .none;
        // qwen3-tts loads a folder, not the .gguf file itself.
        self.tts.submit(model.dir, text, ref, .{
            .temperature = self.tts_temperature.get(),
            .n_threads = @intCast(self.threads.get()),
            .use_gpu = self.use_gpu.get(),
        }) catch {};
    }

    // --- voice-clone reference ---------------------------------------------

    /// Filters for the reference-audio picker. Static: SDL holds the pointer
    /// until the (async) dialog is dismissed.
    const wav_filters = [_]app.FileFilter{
        .{ .name = "WAV audio", .pattern = "wav" },
    };

    /// Open the native file dialog to pick a reference WAV. The result arrives
    /// later through `pumpAudio`.
    pub fn chooseRefWav(self: *AppState) void {
        _ = self;
        _ = app.openFileDialog(&wav_filters, null);
    }

    /// Start mic capture, or stop it and keep the take as the clone reference.
    pub fn toggleRecord(self: *AppState) void {
        if (self.tts_recording) {
            self.recorder.poll(self.gpa, &self.tts_rec);
            self.recorder.stop();
            self.tts_recording = false;
            if (self.tts_rec.items.len > 0) self.setRefPath(null);
        } else {
            self.tts_rec.clearRetainingCapacity();
            if (!self.recorder.start()) {
                self.alert("Could not open the microphone. Check the OS permission for this app.");
                return;
            }
            self.tts_recording = true;
        }
    }

    /// Replace the reference-WAV path (null clears it). Takes ownership of `p`.
    fn setRefPath(self: *AppState, p: ?[]u8) void {
        if (self.tts_ref_path) |old| self.gpa.free(old);
        self.tts_ref_path = p;
    }

    /// Drop the clone reference (both file and recording) → default voice.
    pub fn clearRef(self: *AppState) void {
        self.setRefPath(null);
        self.tts_rec.clearRetainingCapacity();
    }

    /// Play back the recorded reference clip.
    pub fn previewRec(self: *AppState) void {
        if (self.tts_rec.items.len > 0)
            self.player.play(self.tts_rec.items, audioplay.Recorder.sample_rate);
    }

    /// Drain tts events: play synthesized audio and record its length. Also
    /// pumps the voice-clone inputs: accumulates mic samples while recording
    /// and collects the file-dialog result once the user picks a WAV.
    pub fn pumpAudio(self: *AppState) void {
        if (self.tts_recording) self.recorder.poll(self.gpa, &self.tts_rec);
        switch (app.takeFileDialogResult(self.gpa)) {
            .none, .canceled => {},
            .picked => |p| {
                self.setRefPath(p);
                self.tts_rec.clearRetainingCapacity(); // file replaces recording
                self.logf("tts: clone reference set to {s}", .{p});
            },
        }
        var tmp: std.ArrayList(tts.Event) = .empty;
        defer tmp.deinit(self.gpa);
        self.tts.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .audio => |a| {
                self.player.play(a.samples, a.sample_rate);
                self.tts_last_samples = a.samples.len;
                self.logf("tts: {d} samples @ {d} Hz", .{ a.samples.len, a.sample_rate });
                self.gpa.free(a.samples);
            },
            .err => |e| {
                self.alert(e);
                self.gpa.free(e);
            },
        };
    }

    /// Select model at `index` (into `model_list.items`) for its kind, routing
    /// to the matching `sel_*`. Shared by the Models tab and the header picker.
    pub fn setSelected(self: *AppState, index: usize) void {
        if (index >= self.model_list.items.items.len) return;
        const idx: i64 = @intCast(index);
        switch (self.model_list.items.items[index].kind) {
            .text => self.sel_llm.set(idx),
            .image => self.sel_sd.set(idx),
            .video => self.sel_video.set(idx),
            .tts => self.sel_tts.set(idx),
        }
    }

    // --- delete a local model ---------------------------------------------

    /// True if `dir` is a top-level scan root (the app models dir, a default
    /// folder, or a user-added one) — we must never delete those, only a model
    /// folder inside them.
    fn isScanRoot(self: *AppState, dir: []const u8) bool {
        if (self.home) |home| {
            if (config.modelsDirAlloc(self.gpa, home)) |app_models| {
                defer self.gpa.free(app_models);
                if (std.mem.eql(u8, app_models, dir)) return true;
            }
            for (models.default_dirs) |dd| {
                const p = std.fs.path.join(self.gpa, &.{ home, dd.sub }) catch continue;
                defer self.gpa.free(p);
                if (std.mem.eql(u8, p, dir)) return true;
            }
        }
        for (self.model_dirs.items) |d| if (std.mem.eql(u8, d, dir)) return true;
        return false;
    }

    /// Ask to delete the model at `index`. Only models we own (in the app's own
    /// folder) are deletable; others are read-only. Captures what will be removed
    /// and opens the confirmation overlay.
    pub fn requestDeleteModel(self: *AppState, index: usize) void {
        if (index >= self.model_list.items.items.len) return;
        const m = self.model_list.items.items[index];
        // Hard guard: never delete a model we don't own (LM Studio / mlx-serve /
        // custom folders are read-only to us).
        if (!m.source.owned()) return;
        // Delete the whole model folder (downloads live in their own folder),
        // unless that folder is a scan root — then delete only the file.
        const folder = !self.isScanRoot(m.dir);
        const target = if (folder) m.dir else m.path;
        if (self.delete_path) |p| self.gpa.free(p);
        self.delete_path = self.gpa.dupe(u8, target) catch null;
        self.delete_is_folder = folder;
        const n = @min(m.name.len, self.delete_name_buf.len);
        @memcpy(self.delete_name_buf[0..n], m.name[0..n]);
        self.delete_name_len = n;
        // How much the user is about to free (the whole folder, or one file).
        if (folder) {
            const stats = models.folderStats(self.gpa, m.dir);
            self.delete_files = stats.files;
            self.delete_bytes = stats.bytes;
        } else {
            self.delete_files = 1;
            self.delete_bytes = m.size;
        }
        self.delete_present.set(true);
    }

    pub fn deleteModelFiles(self: *const AppState) usize {
        return self.delete_files;
    }
    pub fn deleteModelBytes(self: *const AppState) u64 {
        return self.delete_bytes;
    }

    pub fn deleteModelName(self: *const AppState) []const u8 {
        return self.delete_name_buf[0..self.delete_name_len];
    }
    pub fn deleteModelPath(self: *const AppState) []const u8 {
        return self.delete_path orelse "";
    }

    pub fn cancelDeleteModel(self: *AppState) void {
        self.delete_present.set(false);
        if (self.delete_path) |p| self.gpa.free(p);
        self.delete_path = null;
    }

    fn indexOfPath(self: *AppState, path: []const u8) i64 {
        for (self.model_list.items.items, 0..) |m, i| {
            if (std.mem.eql(u8, m.path, path)) return @intCast(i);
        }
        return -1;
    }

    /// Delete the captured file/folder, rescan, and re-resolve every selection by
    /// path so the remaining picks stay correct despite index shifts.
    pub fn confirmDeleteModel(self: *AppState) void {
        const target = self.delete_path orelse return;

        // Remember selected models by path so we can re-point after the rescan.
        const sels = [_]*zigui.State(i64){ &self.sel_llm, &self.sel_sd, &self.sel_video, &self.sel_tts };
        var kept: [4]?[]u8 = .{ null, null, null, null };
        for (sels, 0..) |s, i| {
            if (self.selectedModel(s.get())) |m| kept[i] = self.gpa.dupe(u8, m.path) catch null;
        }
        defer for (kept) |k| if (k) |p| self.gpa.free(p);

        var threaded = std.Io.Threaded.init(self.gpa, .{});
        defer threaded.deinit();
        const io = threaded.io();
        const ok = if (self.delete_is_folder)
            std.Io.Dir.cwd().deleteTree(io, target)
        else
            std.Io.Dir.cwd().deleteFile(io, target);
        ok catch |e| {
            self.alertf("Failed to delete: {s}", .{@errorName(e)});
            self.cancelDeleteModel();
            return;
        };
        self.logf("models: deleted {s}", .{target});

        self.rescanModels();
        // Re-resolve each selection (or clear it if the model is gone).
        for (sels, 0..) |s, i| {
            s.set(if (kept[i]) |p| self.indexOfPath(p) else -1);
        }
        self.cancelDeleteModel();
    }

    // --- tray / loaded-model status ---------------------------------------

    /// Record the model just submitted to a backend so the tray can show its
    /// name and count its bytes toward the RAM proxy. `slot`/`size` point at the
    /// matching `loaded_*` / `loaded_size_*` fields.
    fn rememberLoaded(self: *AppState, slot: *?[]u8, size: *u64, m: models.ModelInfo) void {
        if (slot.*) |old| self.gpa.free(old);
        slot.* = self.gpa.dupe(u8, m.name) catch null;
        size.* = m.size;
    }

    /// Free every backend's cached model to reclaim memory, without shutting the
    /// workers down (the next generation reloads on demand). A backend that is
    /// mid-generation is left alone (its `unload` is a no-op while busy).
    pub fn unloadAll(self: *AppState) void {
        // Chat runs in the server now; it owns the chat model's lifetime, so the
        // GUI can only unload the in-process image/video/audio backends.
        self.sd.unload();
        self.tts.unload();
        self.video.unload();
        // Drop the remembered name/size for any backend that actually unloaded
        // (a busy one keeps its model, so leave its row intact).
        if (!self.sd.model_ready.load(.acquire)) self.clearLoaded(&self.loaded_sd, &self.loaded_size_sd);
        if (!self.tts.model_ready.load(.acquire)) self.clearLoaded(&self.loaded_tts, &self.loaded_size_tts);
        if (!self.video.model_ready.load(.acquire)) self.clearLoaded(&self.loaded_video, &self.loaded_size_video);
        self.logf("models: unloaded image/video/audio (freed resident memory)", .{});
    }

    fn clearLoaded(self: *AppState, slot: *?[]u8, size: *u64) void {
        if (slot.*) |old| self.gpa.free(old);
        slot.* = null;
        size.* = 0;
    }

    /// Total resident model bytes — the sum of the on-disk sizes of every backend
    /// whose model is currently loaded. A proxy for RAM use (it excludes the KV
    /// cache and compute buffers), shown in the tray.
    pub fn loadedBytes(self: *AppState) u64 {
        var total: u64 = 0;
        const cs = self.chat.status(); // server-reported chat model bytes
        if (cs.loaded) total += cs.bytes;
        if (self.sd.model_ready.load(.acquire)) total += self.loaded_size_sd;
        if (self.tts.model_ready.load(.acquire)) total += self.loaded_size_tts;
        if (self.video.model_ready.load(.acquire)) total += self.loaded_size_video;
        return total;
    }

    /// Active accelerator + memory for the footer indicator. Polls the driver
    /// ~twice a second (every 30 frames), and keeps retrying until backends have
    /// loaded, so the value appears as soon as the device registry is ready.
    pub fn acceleratorInfo(self: *AppState) accel.Info {
        if (!self.accel_info.ok or self.accel_poll % 30 == 0) self.accel_info = accel.query();
        self.accel_poll +%= 1;
        return self.accel_info;
    }

    // --- downloader -------------------------------------------------------

    fn clearDlResults(self: *AppState) void {
        for (self.dl_results.items) |r| self.gpa.free(r.id);
        self.dl_results.clearRetainingCapacity();
    }

    fn clearDlFiles(self: *AppState) void {
        if (self.dl_files) |f| downloader.freeFiles(self.gpa, f);
        self.dl_files = null;
    }

    /// Map the `dl_category` Picker index to a model Kind (null = "All").
    fn dlCategory(self: *AppState) ?models.Kind {
        return switch (self.dl_category.get()) {
            1 => .text,
            2 => .image,
            3 => .video,
            4 => .tts,
            else => null,
        };
    }

    /// Jump to the Models › Download tab pre-filtered to `kind` and kick off a
    /// search, so an empty per-type tab can offer a one-tap path to get models.
    pub fn browseDownloads(self: *AppState, kind: models.Kind) void {
        self.screen.set(@intFromEnum(Screen.models)); // jump to the Models screen…
        self.models_tab.set(4); // …Download tab
        self.dl_category.set(switch (kind) {
            .text => 1,
            .image => 2,
            .video => 3,
            .tts => 4,
        });
        self.dlSearch();
    }

    /// Run a HuggingFace search from the current query + category.
    pub fn dlSearch(self: *AppState) void {
        const q = std.mem.trim(u8, self.dl_search.text(), " \t\n");
        self.dl_filepick_open.set(false);
        self.dl_filepick_idx = -1;
        self.dl_searching = true;
        self.downloader.search(q, self.dlCategory());
    }

    /// Open the quant popover for result row `idx` and fetch its file list.
    pub fn dlOpenFiles(self: *AppState, idx: usize) void {
        if (idx >= self.dl_results.items.len) return;
        self.dl_filepick_idx = @intCast(idx);
        self.clearDlFiles();
        self.dl_filepick_open.set(true);
        self.downloader.listFiles(self.dl_results.items[idx].id);
    }

    /// Start downloading the chosen quant `file` plus every support file in the
    /// repo (VAE / text encoder / tokenizer / config), so the model is runnable
    /// the moment it lands.
    pub fn dlStart(self: *AppState, file: []const u8) void {
        if (self.dl_filepick_idx < 0) return;
        const idx: usize = @intCast(self.dl_filepick_idx);
        if (idx >= self.dl_results.items.len) return;
        const home = self.home orelse {
            self.alert("No home directory; cannot choose a download folder.");
            return;
        };
        const repo = self.dl_results.items[idx];

        // The chosen quant + all non-quant support files. `download` dupes these
        // synchronously, so a transient gpa list is fine.
        var list: std.ArrayList(RepoFile) = .empty;
        defer list.deinit(self.gpa);
        if (self.dl_files) |files| {
            for (files.items) |f| {
                if (f.is_quant) {
                    if (std.mem.eql(u8, f.path, file)) list.append(self.gpa, f) catch {};
                } else {
                    list.append(self.gpa, f) catch {};
                }
            }
        }
        if (list.items.len == 0) {
            // No tree loaded; fall back to the single chosen file.
            list.append(self.gpa, .{ .path = @constCast(file), .size = 0 }) catch return;
        }

        // Save into the app's own cross-platform models dir (not a hardcoded
        // ~/.mlx-serve), grouped by kind.
        const models_root = config.modelsDirAlloc(self.gpa, home) orelse {
            self.alert("Could not resolve the models directory.");
            return;
        };
        defer self.gpa.free(models_root);

        self.dl_filepick_open.set(false);
        // For split models (FLUX/Wan), also pull the cross-repo VAE/encoder into
        // the same folder so the model is runnable on arrival. The downloader
        // tracks each model as its own job, so this can run alongside others.
        self.downloader.download(repo.name(), models_root, repo.id, list.items, repo.kind, manifest.sidecarsFor(repo.id));
    }

    /// Download a curated bundle (`manifest.recommended[index]`): its exact files
    /// plus cross-repo sidecars, into one folder, so a multi-repo model like LTX
    /// is runnable in one tap.
    pub fn dlGetRecommended(self: *AppState, index: usize) void {
        if (index >= manifest.recommended.len) return;
        const rec = manifest.recommended[index];
        const home = self.home orelse {
            self.alert("No home directory; cannot choose a download folder.");
            return;
        };
        const models_root = config.modelsDirAlloc(self.gpa, home) orelse {
            self.alert("Could not resolve the models directory.");
            return;
        };
        defer self.gpa.free(models_root);

        // Every bundle file is a sidecar (carries its own repo + optional rename),
        // so the primary file set is empty; the folder is named after `rec.repo`.
        self.downloader.download(rec.title, models_root, rec.repo, &[_]RepoFile{}, rec.kind, rec.items);
    }

    /// Cancel one in-flight download by its job id.
    pub fn dlCancel(self: *AppState, job_id: u64) void {
        self.downloader.cancelJob(job_id);
    }

    /// Drain downloader events: results/files replace their buffers, progress is
    /// read live from atomics, errors raise an alert, done triggers a rescan.
    pub fn pumpDownloader(self: *AppState) void {
        var tmp: std.ArrayList(downloader.Event) = .empty;
        defer tmp.deinit(self.gpa);
        self.downloader.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .results => |repos| {
                self.dl_searching = false;
                self.clearDlResults();
                self.dl_results.appendSlice(self.gpa, repos) catch {};
                self.gpa.free(repos); // ids now owned by dl_results
            },
            .repo_size => |rs| {
                // Match by id (the table may have re-sorted since the search), so a
                // stale event from a superseded search simply finds nothing.
                for (self.dl_results.items) |*r| {
                    if (std.mem.eql(u8, r.id, rs.id)) {
                        r.size_min = rs.min;
                        r.size_max = rs.max;
                        r.size_loaded = true;
                        break;
                    }
                }
                self.gpa.free(rs.id);
            },
            .files => |f| {
                // Keep only if it matches the row whose popover is open.
                self.clearDlFiles();
                self.dl_files = f;
            },
            .file => |fe| {
                // Route the new current-file name to its job (UI thread owns it).
                if (self.downloader.jobById(fe.job)) |j| {
                    if (j.cur_file) |a| self.gpa.free(a);
                    j.cur_file = fe.name; // take ownership
                } else self.gpa.free(fe.name); // job already reaped
            },
            .progress => {}, // read live from each job's atomics in the view
            .done => |de| {
                self.logf("download: saved into {s}", .{de.folder});
                if (self.downloader.jobById(de.job)) |j|
                    self.alertOk("Download complete", "{s} downloaded successfully.", .{j.name})
                else
                    self.alertOk("Download complete", "Model downloaded successfully.", .{});
                self.gpa.free(de.folder);
                self.downloader.finishJob(de.job);
                self.rescanModels();
            },
            .job_err => |je| {
                self.alert(je.msg);
                self.gpa.free(je.msg);
                self.downloader.finishJob(je.job);
            },
            .canceled => |id| {
                self.downloader.finishJob(id);
            },
            .err => |e| {
                self.dl_searching = false;
                self.alert(e);
                self.gpa.free(e);
            },
        };
    }

    /// Append a formatted line to the Logs ring (UI thread only).
    pub fn logf(self: *AppState, comptime f: []const u8, args: anytype) void {
        var buf: [256]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, f, args) catch return;
        self.logs.append(line);
    }

    /// Raise a modal alert (and also log it). Use for errors the user must see —
    /// failed model load, missing model, generation failure — instead of quietly
    /// switching to the Logs screen.
    pub fn alert(self: *AppState, text: []const u8) void {
        self.alert_title = "Something went wrong"; // restore the error title
        const n = @min(text.len, self.alert_buf.len);
        @memcpy(self.alert_buf[0..n], text[0..n]);
        self.alert_len = n;
        self.alert_present.set(true);
        self.logs.append(text);
    }

    pub fn alertf(self: *AppState, comptime f: []const u8, args: anytype) void {
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, f, args) catch return;
        self.alert(line);
    }

    /// A non-error modal (e.g. "Download complete") with a custom `title`.
    pub fn alertOk(self: *AppState, title: []const u8, comptime f: []const u8, args: anytype) void {
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, f, args) catch return;
        const n = @min(line.len, self.alert_buf.len);
        @memcpy(self.alert_buf[0..n], line[0..n]);
        self.alert_len = n;
        self.alert_title = title;
        self.alert_present.set(true);
        self.logs.append(line);
    }

    pub fn alertText(self: *AppState) []const u8 {
        return self.alert_buf[0..self.alert_len];
    }

    fn appendNotice(self: *AppState, text: []const u8) void {
        const m = ChatMessage.create(self.gpa, .assistant) catch return;
        m.setText(text) catch {};
        self.messages.append(self.gpa, m) catch m.destroy();
    }

    /// Drain llama events into the conversation. Call once per frame.
    pub fn pumpChat(self: *AppState) void {
        self.llm_loaded = self.chat.status().loaded;

        var tmp: std.ArrayList(chat_client.Event) = .empty;
        defer tmp.deinit(self.gpa);
        self.chat.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .content => |s| {
                if (self.pending) |p| p.content.appendSlice(self.gpa, s) catch {};
                if (self.chat_scroll.atBottom()) self.chat_scroll.offset = 1_000_000;
                self.gpa.free(s);
            },
            .reasoning => |s| {
                if (self.pending) |p| p.reasoning.appendSlice(self.gpa, s) catch {};
                if (self.chat_scroll.atBottom()) self.chat_scroll.offset = 1_000_000;
                self.gpa.free(s);
            },
            .tool => |tc| {
                if (self.pending) |p| p.setToolCall(tc.name, tc.args);
                self.gpa.free(tc.name);
                self.gpa.free(tc.args);
            },
            .done => |d| {
                if (self.pending) |p| {
                    p.streaming = false;
                    // The server doesn't stream a token count, so approximate for
                    // the tok/s readout from the emitted text length.
                    const approx: u64 = @intCast((p.content.items.len + p.reasoning.items.len) / 4);
                    p.tokens = approx;
                    p.tps = if (d.ms > 0) @as(f32, @floatFromInt(approx)) / (@as(f32, @floatFromInt(d.ms)) / 1000.0) else 0;
                    self.pending = null;
                    self.logf("chat: ~{d} tokens in {d} ms", .{ approx, d.ms });
                    self.maybeRunTool(p);
                }
            },
            .err => |e| {
                if (self.pending) |p| {
                    p.setText(e) catch {};
                    p.streaming = false;
                } else self.appendNotice(e);
                self.pending = null;
                self.gpa.free(e);
            },
        };
    }

    // --- agent loop + MCP -------------------------------------------------

    /// After an assistant turn finishes, if agent mode is on and the server
    /// returned a tool call, dispatch it to the MCP manager. The result comes
    /// back through `pumpMcp`, which continues the loop.
    fn maybeRunTool(self: *AppState, msg: *ChatMessage) void {
        if (!self.agent_mode.get()) return;
        const tc = msg.tool_call orelse return;
        if (self.agent_iters >= agent.max_iterations) {
            self.appendNotice("Agent stopped: reached the tool-call limit for this turn.");
            return;
        }
        if (!self.mcp_mgr.hasTool(tc.name)) {
            const fa = self.frame_arena.allocator();
            const err = std.fmt.allocPrint(fa, "Error: unknown tool \"{s}\". Use only the tools provided.", .{tc.name}) catch "Error: unknown tool.";
            self.continueWithToolResult(err);
            return;
        }
        self.agent_iters += 1;
        self.agent_seq += 1;
        self.agent_busy = true;
        self.setAgentToolName(tc.name);
        self.logf("agent: calling {s} (step {d})", .{ tc.name, self.agent_iters });
        self.mcp_mgr.callAsync(self.agent_seq, tc.name, tc.args);
    }

    fn setAgentToolName(self: *AppState, name: []const u8) void {
        const n = @min(name.len, self.agent_tool_buf.len);
        @memcpy(self.agent_tool_buf[0..n], name[0..n]);
        self.agent_tool_len = n;
    }

    pub fn agentToolName(self: *const AppState) []const u8 {
        return self.agent_tool_buf[0..self.agent_tool_len];
    }

    /// Append the tool result as a `tool` message and resume generation. (The
    /// server maps role:"tool" into the model's prompt; see api.zig.)
    fn continueWithToolResult(self: *AppState, text: []const u8) void {
        const m = ChatMessage.create(self.gpa, .tool) catch return;
        m.setText(text) catch {};
        self.messages.append(self.gpa, m) catch {
            m.destroy();
            return;
        };
        self.startGeneration();
    }

    /// Drain MCP events: tool results resume the agent loop; logs go to the ring.
    /// Call once per frame.
    pub fn pumpMcp(self: *AppState) void {
        var tmp: std.ArrayList(mcp.Event) = .empty;
        defer tmp.deinit(self.gpa);
        self.mcp_mgr.events.drain(&tmp);
        for (tmp.items) |ev| switch (ev) {
            .changed => {},
            .log => |s| {
                self.logs.append(s);
                self.gpa.free(s);
            },
            .result => |r| {
                // Ignore stale results from a previous turn (e.g. after New chat).
                if (r.seq == self.agent_seq) {
                    self.agent_busy = false;
                    self.agent_tool_len = 0;
                    self.continueWithToolResult(r.text);
                }
                self.gpa.free(r.text);
            },
        };
    }

    // --- config: system prompt + MCP --------------------------------------

    /// Load the persisted config files and bring up the MCP runtime. Called once
    /// from `main` after the home directory and environment are known.
    pub fn loadConfig(self: *AppState, environ: std.process.Environ) void {
        self.environ = environ;
        const home = self.home orelse return;
        config.ensureDefaults(self.gpa, home);
        if (self.system_prompt.len > 0) self.gpa.free(self.system_prompt);
        self.system_prompt = config.loadSystemPrompt(self.gpa, home);
        self.mcp_mgr.setContext(home, environ);
        self.mcp_mgr.start() catch {};
    }

    /// Reload the system prompt from disk (after an external or in-app edit).
    fn reloadSystemPrompt(self: *AppState) void {
        const home = self.home orelse return;
        if (self.system_prompt.len > 0) self.gpa.free(self.system_prompt);
        self.system_prompt = config.loadSystemPrompt(self.gpa, home);
    }

    // --- in-app text editor -----------------------------------------------

    /// Open the editor on a config file: load its current bytes into the buffer
    /// and switch to the editor screen.
    pub fn openEditor(self: *AppState, target: EditorTarget) void {
        self.editor_target = target;
        const home = self.home;
        var text: []u8 = &.{};
        var owned = false;
        if (home) |h| {
            if (config.read(self.gpa, h, target.fileName())) |bytes| {
                text = bytes;
                owned = true;
            }
        }
        // Seed an empty mcp.json / the default prompt so the buffer isn't blank.
        if (text.len == 0) {
            switch (target) {
                .system_prompt => self.editor_buf.setText(config.default_system_prompt) catch {},
                .mcp_json => self.editor_buf.setText("{\n  \"mcpServers\": {}\n}\n") catch {},
            }
        } else {
            self.editor_buf.setText(text) catch {};
        }
        if (owned) self.gpa.free(text);
        self.editor_scroll = .{};
        self.editor_saved_until_ms = 0;
        self.screen.set(@intFromEnum(Screen.editor));
    }

    /// Save the editor buffer back to its file and apply the change.
    pub fn saveEditor(self: *AppState) void {
        const home = self.home orelse {
            self.alert("No home directory; cannot save.");
            return;
        };
        const data = self.editor_buf.text();
        if (!config.write(self.gpa, home, self.editor_target.fileName(), data)) {
            self.alert("Failed to write the file.");
            return;
        }
        switch (self.editor_target) {
            .system_prompt => self.reloadSystemPrompt(),
            .mcp_json => self.mcp_mgr.reload(),
        }
        self.logf("editor: saved {s}", .{self.editor_target.fileName()});
    }

    // --- MCP marketplace actions ------------------------------------------

    /// Handle an "Add" tap on a catalog preset: if it needs configuration, open
    /// the inline form to collect values; otherwise add it straight away.
    pub fn addMcpPresetByIndex(self: *AppState, idx: usize) void {
        if (idx >= mcp.catalog.len) return;
        const preset = mcp.catalog[idx];
        if (preset.needsConfig()) {
            self.openMcpConfig(idx);
        } else {
            self.addMcpPreset(preset);
        }
    }

    /// Add a preset server to mcp.json and respawn the runtime.
    pub fn addMcpPreset(self: *AppState, preset: mcp.Preset) void {
        const home = self.home orelse return;
        if (mcp.addPreset(self.gpa, home, preset)) {
            self.logf("mcp: added preset {s}", .{preset.id});
            self.mcp_mgr.reload();
        }
    }

    /// Open the inline config form for catalog preset `idx`, clearing its fields.
    pub fn openMcpConfig(self: *AppState, idx: usize) void {
        self.mcp_cfg_idx = @intCast(idx);
        self.mcp_cfg_editing = false;
        for (&self.mcp_cfg_fields) |*f| f.setText("") catch {};
    }

    /// Open the config form on an already-added preset server, pre-filled with
    /// the entry's current values, to edit them in place.
    pub fn openMcpEdit(self: *AppState, idx: usize) void {
        if (idx >= mcp.catalog.len) return;
        const home = self.home orelse return;
        const preset = mcp.catalog[idx];
        self.mcp_cfg_idx = @intCast(idx);
        self.mcp_cfg_editing = true;
        for (&self.mcp_cfg_fields) |*f| f.setText("") catch {};

        var reg = mcp.loadRegistry(self.gpa, home);
        defer reg.deinit();
        for (reg.servers.items) |s| {
            if (!std.mem.eql(u8, s.name, preset.id)) continue;
            var vals: [max_mcp_inputs][]const u8 = .{""} ** max_mcp_inputs;
            const n = @min(preset.inputs.len, max_mcp_inputs);
            mcp.currentValues(preset, s, vals[0..n]);
            // setText copies, so the values may die with `reg`.
            for (0..n) |i| self.mcp_cfg_fields[i].setText(vals[i]) catch {};
            break;
        }
    }

    pub fn cancelMcpConfig(self: *AppState) void {
        self.mcp_cfg_idx = -1;
        self.mcp_cfg_editing = false;
    }

    /// Write the configured preset (collecting values from the form fields) and
    /// respawn the runtime. In edit mode this rewrites the existing entry
    /// (keeping its disabled flag) instead of appending a new one.
    pub fn confirmMcpConfig(self: *AppState) void {
        if (self.mcp_cfg_idx < 0) return;
        const idx: usize = @intCast(self.mcp_cfg_idx);
        if (idx >= mcp.catalog.len) return;
        const home = self.home orelse return;
        const preset = mcp.catalog[idx];

        var values: [max_mcp_inputs][]const u8 = undefined;
        const n = @min(preset.inputs.len, max_mcp_inputs);
        for (0..n) |i| values[i] = self.mcp_cfg_fields[i].text();

        const ok = if (self.mcp_cfg_editing)
            mcp.updatePresetValues(self.gpa, home, preset, values[0..n])
        else
            mcp.addPresetWithValues(self.gpa, home, preset, values[0..n]);
        if (ok) {
            const verb: []const u8 = if (self.mcp_cfg_editing) "updated" else "added";
            self.logf("mcp: {s} preset {s}", .{ verb, preset.id });
            self.mcp_mgr.reload();
        }
        self.mcp_cfg_idx = -1;
        self.mcp_cfg_editing = false;
    }

    /// Remove a server entry by name and respawn the runtime.
    pub fn removeMcpServer(self: *AppState, name: []const u8) void {
        const home = self.home orelse return;
        if (mcp.removeServer(self.gpa, home, name)) {
            self.logf("mcp: removed {s}", .{name});
            self.mcp_mgr.reload();
        }
    }

    /// Toggle a server's disabled flag and respawn the runtime.
    pub fn toggleMcpServer(self: *AppState, name: []const u8, disabled: bool) void {
        const home = self.home orelse return;
        if (mcp.setDisabled(self.gpa, home, name, disabled)) self.mcp_mgr.reload();
    }
};
