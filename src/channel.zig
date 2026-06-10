//! Thread-safe primitives that connect background inference workers to the
//! single-threaded UI loop. A `Channel(T)` is a mutex-guarded queue: the worker
//! `push`es events, the UI `drain`s them once per frame. `JobState` is a bundle
//! of atomics the UI reads every frame (without locking) to show progress and to
//! request cancellation.

const std = @import("std");

/// A spinlock built on a single atomic flag. Critical sections here are tiny
/// (a list append / a slice copy) and contention is near-zero, so spinning beats
/// pulling in the new std.Io-based Mutex (which needs an `Io` handle) — and it
/// needs no initialization, so it survives being copied into its owner struct.
pub const SpinLock = struct {
    flag: std.atomic.Value(bool) = .init(false),

    pub fn lock(self: *SpinLock) void {
        while (self.flag.cmpxchgWeak(false, true, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }
    pub fn unlock(self: *SpinLock) void {
        self.flag.store(false, .release);
    }
};

/// A simple multi-producer / single-consumer queue. Event rates here are low
/// (tens of tokens/sec, a handful of progress ticks), so a spinlock around a
/// growable list is more than fast enough and trivially correct.
pub fn Channel(comptime T: type) type {
    return struct {
        const Self = @This();

        mu: SpinLock = .{},
        items: std.ArrayList(T) = .empty,
        gpa: std.mem.Allocator,

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        pub fn deinit(self: *Self) void {
            self.mu.lock();
            defer self.mu.unlock();
            self.items.deinit(self.gpa);
        }

        /// Enqueue one event (worker thread).
        pub fn push(self: *Self, v: T) void {
            self.mu.lock();
            defer self.mu.unlock();
            self.items.append(self.gpa, v) catch {};
        }

        /// Move every pending event into `out` (UI thread), clearing the queue.
        /// `out` is owned by the caller and must use the same allocator.
        pub fn drain(self: *Self, out: *std.ArrayList(T)) void {
            self.mu.lock();
            defer self.mu.unlock();
            out.appendSlice(self.gpa, self.items.items) catch {};
            self.items.clearRetainingCapacity();
        }

        /// Number of pending events (UI thread, cheap status check).
        pub fn len(self: *Self) usize {
            self.mu.lock();
            defer self.mu.unlock();
            return self.items.items.len;
        }
    };
}

/// Per-job control/observation shared between a worker and the UI. The worker
/// writes `running`/`step`/`total`; the UI writes `cancel` and reads the rest.
pub const JobState = struct {
    /// True while a worker is actively processing a job. Drives `busyCheck`.
    running: std.atomic.Value(bool) = .init(false),
    /// Set by the UI to ask the worker to stop early; cleared by the worker.
    cancel: std.atomic.Value(bool) = .init(false),
    /// Progress for multi-step jobs (image/video diffusion steps).
    step: std.atomic.Value(i32) = .init(0),
    total: std.atomic.Value(i32) = .init(0),

    pub fn beginJob(self: *JobState) void {
        self.cancel.store(false, .release);
        self.step.store(0, .release);
        self.total.store(0, .release);
        self.running.store(true, .release);
    }

    pub fn endJob(self: *JobState) void {
        self.running.store(false, .release);
        self.cancel.store(false, .release);
    }

    pub fn isRunning(self: *const JobState) bool {
        return self.running.load(.acquire);
    }

    pub fn requestCancel(self: *JobState) void {
        self.cancel.store(true, .release);
    }

    pub fn cancelRequested(self: *const JobState) bool {
        return self.cancel.load(.acquire);
    }

    pub fn setProgress(self: *JobState, step: i32, total: i32) void {
        self.step.store(step, .release);
        self.total.store(total, .release);
    }

    /// Fraction in [0, 1]; 0 when total is unknown.
    pub fn fraction(self: *const JobState) f32 {
        const tot = self.total.load(.acquire);
        if (tot <= 0) return 0;
        const st = self.step.load(.acquire);
        return std.math.clamp(@as(f32, @floatFromInt(st)) / @as(f32, @floatFromInt(tot)), 0, 1);
    }
};
