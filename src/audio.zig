//! Minimal SDL3 audio playback for TTS output. Holds one playback stream and
//! plays float32 mono PCM clips; a new clip replaces whatever is still queued.
//! Also a `Recorder` that captures the default microphone in the same format,
//! used to take a voice-clone reference sample.

const std = @import("std");
const app = @import("zigui_app");
const c = app.c;

pub const Player = struct {
    stream: ?*c.SDL_AudioStream = null,
    opened_rate: i32 = 0,

    /// Play a float32 mono clip at `rate` Hz. SDL copies the samples, so the
    /// caller may free them immediately after this returns.
    pub fn play(self: *Player, samples: []const f32, rate: i32) void {
        if (samples.len == 0) return;
        if (self.stream == null or self.opened_rate != rate) {
            self.close();
            _ = c.SDL_InitSubSystem(c.SDL_INIT_AUDIO);
            var spec = c.SDL_AudioSpec{
                .format = c.SDL_AUDIO_F32,
                .channels = 1,
                .freq = rate,
            };
            self.stream = c.SDL_OpenAudioDeviceStream(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, null, null);
            self.opened_rate = rate;
            if (self.stream) |s| _ = c.SDL_ResumeAudioStreamDevice(s);
        }
        if (self.stream) |s| {
            _ = c.SDL_ClearAudioStream(s); // start the new clip fresh
            _ = c.SDL_PutAudioStreamData(s, samples.ptr, @intCast(samples.len * @sizeOf(f32)));
        }
    }

    pub fn close(self: *Player) void {
        if (self.stream) |s| {
            c.SDL_DestroyAudioStream(s);
            self.stream = null;
        }
    }
};

/// Captures the default microphone as float32 mono @ 24 kHz — exactly the
/// format qwen3-tts expects for a voice-clone reference, so no resampling is
/// needed. SDL converts from whatever the device delivers. The caller polls
/// each frame (the run loop must stay awake while recording — see busyCheck).
pub const Recorder = struct {
    stream: ?*c.SDL_AudioStream = null,

    pub const sample_rate: i32 = 24000;

    /// Open the default recording device and start capturing. On macOS the
    /// first call triggers the OS microphone-permission prompt.
    pub fn start(self: *Recorder) bool {
        if (self.stream != null) return true;
        _ = c.SDL_InitSubSystem(c.SDL_INIT_AUDIO);
        var spec = c.SDL_AudioSpec{
            .format = c.SDL_AUDIO_F32,
            .channels = 1,
            .freq = sample_rate,
        };
        self.stream = c.SDL_OpenAudioDeviceStream(c.SDL_AUDIO_DEVICE_DEFAULT_RECORDING, &spec, null, null);
        const s = self.stream orelse return false;
        _ = c.SDL_ResumeAudioStreamDevice(s);
        return true;
    }

    pub fn recording(self: *const Recorder) bool {
        return self.stream != null;
    }

    /// Drain whatever the device captured since the last call into `out`.
    pub fn poll(self: *Recorder, gpa: std.mem.Allocator, out: *std.ArrayList(f32)) void {
        const s = self.stream orelse return;
        var buf: [4096]f32 = undefined;
        while (true) {
            const got = c.SDL_GetAudioStreamData(s, &buf, @sizeOf(@TypeOf(buf)));
            if (got <= 0) break;
            const n: usize = @intCast(got);
            out.appendSlice(gpa, buf[0 .. n / @sizeOf(f32)]) catch return;
        }
    }

    /// Stop capturing and release the device (any undrained tail is dropped —
    /// call `poll` first).
    pub fn stop(self: *Recorder) void {
        if (self.stream) |s| {
            c.SDL_DestroyAudioStream(s);
            self.stream = null;
        }
    }
};
