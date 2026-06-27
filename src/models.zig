//! Local model discovery. Scans a set of directories for GGUF files and
//! classifies each by filename heuristics so the Model Browser can offer the
//! right models per task (chat / image / tts). No network access — v1 reuses
//! whatever is already on disk (LM Studio, mlx-serve, user folders).

const std = @import("std");
const config = @import("config.zig");

// Directory scanning via libc: std.fs in this Zig requires the new std.Io model,
// and libc is already linked, so POSIX dirent/stat is the simplest path.
const c = @cImport({
    @cInclude("dirent.h");
    @cInclude("sys/stat.h");
});

const gguf = @cImport({
    @cInclude("gguf.h");
});

const builtin = @import("builtin");

// On Windows (mingw) the default `struct stat` / `stat()` carry a 32-bit
// `st_size`, so `stat()` fails with EOVERFLOW on files ≥2 GB — which silently
// hid multi-GB GGUF models from the scan. The `_stat64` variant has a 64-bit
// size and works for any file. Elsewhere the plain `stat` is already 64-bit.
const Stat = if (builtin.os.tag == .windows) c.struct__stat64 else c.struct_stat;

/// Stat `path` into `out`. Returns false on error. 64-bit-size-safe everywhere.
fn statPath(path: [*:0]const u8, out: *Stat) bool {
    const rc = if (builtin.os.tag == .windows) c._stat64(path, out) else c.stat(path, out);
    return rc == 0;
}

/// Read a model's training context length from GGUF metadata
/// (`<arch>.context_length`) without loading tensors. Null if unavailable.
pub fn readCtxCap(gpa: std.mem.Allocator, path: []const u8) ?u32 {
    const path_z = gpa.dupeZ(u8, path) catch return null;
    defer gpa.free(path_z);
    var params = std.mem.zeroes(gguf.struct_gguf_init_params);
    params.no_alloc = true;
    const ctx = gguf.gguf_init_from_file(path_z.ptr, params) orelse return null;
    defer gguf.gguf_free(ctx);
    const arch_id = gguf.gguf_find_key(ctx, "general.architecture");
    if (arch_id < 0) return null;
    const arch = std.mem.span(gguf.gguf_get_val_str(ctx, arch_id));
    var key_buf: [128]u8 = undefined;
    const key = std.fmt.bufPrintZ(&key_buf, "{s}.context_length", .{arch}) catch return null;
    const kid = gguf.gguf_find_key(ctx, key.ptr);
    if (kid < 0) return null;
    return switch (gguf.gguf_get_kv_type(ctx, kid)) {
        gguf.GGUF_TYPE_UINT32 => gguf.gguf_get_val_u32(ctx, kid),
        gguf.GGUF_TYPE_UINT64 => @intCast(gguf.gguf_get_val_u64(ctx, kid)),
        gguf.GGUF_TYPE_INT32 => @intCast(@max(0, gguf.gguf_get_val_i32(ctx, kid))),
        else => null,
    };
}

pub const Kind = enum {
    text,
    image,
    video,
    tts,

    pub fn label(self: Kind) []const u8 {
        return switch (self) {
            .text => "Chat",
            .image => "Image",
            .video => "Video",
            .tts => "TTS",
        };
    }

    /// Stable lowercase folder name used to group downloaded models on disk
    /// (`~/.mlx-serve/models/<folder>/<author>/<name>/`).
    pub fn folder(self: Kind) []const u8 {
        return switch (self) {
            .text => "chat",
            .image => "image",
            .video => "video",
            .tts => "audio",
        };
    }
};

/// Where a discovered model lives — drives a source badge and, crucially, whether
/// the app may delete it. We only own (and delete) models in our own folder.
pub const Source = enum {
    zig_ai,
    lmstudio,
    mlx_serve,
    custom,

    pub fn label(self: Source) []const u8 {
        return switch (self) {
            .zig_ai => "zig-ai",
            .lmstudio => "LM Studio",
            .mlx_serve => "mlx-serve",
            .custom => "Custom",
        };
    }

    /// True only for models in the app's own folder — the only ones we delete.
    pub fn owned(self: Source) bool {
        return self == .zig_ai;
    }
};

pub const ModelInfo = struct {
    /// Absolute path to the .gguf file (owned).
    path: []u8,
    /// Display name (basename without extension; borrows into `path`).
    name: []const u8,
    /// Directory containing the file (borrows into `path`); useful for TTS,
    /// which loads a *folder* of files rather than a single file.
    dir: []const u8,
    kind: Kind,
    size: u64,
    /// Which root this model was discovered under (defaults to non-owned).
    source: Source = .custom,
    /// False when the model's folder still has an interrupted download — a
    /// leftover `.partial` file or a missing curated sidecar. Computed after the
    /// scan (see `AppState.rescanModels`); only meaningful for owned models.
    /// Drives the Model Browser's "Resume" affordance instead of "Installed".
    complete: bool = true,
};

pub const ModelList = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(ModelInfo) = .empty,

    pub fn init(gpa: std.mem.Allocator) ModelList {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *ModelList) void {
        for (self.items.items) |m| self.gpa.free(m.path);
        self.items.deinit(self.gpa);
    }

    pub fn clear(self: *ModelList) void {
        for (self.items.items) |m| self.gpa.free(m.path);
        self.items.clearRetainingCapacity();
    }

    /// Count of models matching `kind`.
    pub fn countKind(self: *const ModelList, kind: Kind) usize {
        var n: usize = 0;
        for (self.items.items) |m| {
            if (m.kind == kind) n += 1;
        }
        return n;
    }
};

/// A default directory we look in (expanded against $HOME) and the source it
/// represents (for the model's origin badge).
pub const DefaultDir = struct { sub: []const u8, source: Source };

/// Legacy / third-party model locations scanned in addition to the app's own
/// folder. Kept as extra sources; we never delete from these.
pub const default_dirs = [_]DefaultDir{
    .{ .sub = ".lmstudio/models", .source = .lmstudio },
    .{ .sub = ".mlx-serve/models", .source = .mlx_serve },
};

/// Classify a model by its (base)name using the same filename heuristics for
/// both local discovery and the HuggingFace downloader. Returns null for files
/// that are not standalone models (mmproj/text-encoder/VAE/clip-vision sidecars).
pub fn classifyName(basename: []const u8) ?Kind {
    var buf: [256]u8 = undefined;
    const lower = std.ascii.lowerString(buf[0..@min(basename.len, buf.len)], basename[0..@min(basename.len, buf.len)]);
    // Skip multimodal projection sidecars and the support files that pair with a
    // video model (text encoder, VAE, clip vision) — none are standalone models.
    if (std.mem.indexOf(u8, lower, "mmproj") != null) return null;
    if (std.mem.indexOf(u8, lower, "umt5") != null or
        std.mem.indexOf(u8, lower, "t5xxl") != null or
        std.mem.indexOf(u8, lower, "t5-xxl") != null or
        std.mem.indexOf(u8, lower, "_vae") != null or
        std.mem.indexOf(u8, lower, "-vae") != null or
        std.mem.indexOf(u8, lower, "text-encoder") != null or
        std.mem.indexOf(u8, lower, "text_encoder") != null or
        std.mem.indexOf(u8, lower, "encoder") != null or
        std.mem.indexOf(u8, lower, "tokenizer") != null or // codec/audio tokenizer sidecar
        std.mem.indexOf(u8, lower, "clip_vision") != null or
        std.mem.indexOf(u8, lower, "clip-vision") != null) return null;
    if (std.mem.indexOf(u8, lower, "tts") != null) return .tts;
    // Video diffusion models (Wan, LTX); their encoder/vae siblings were skipped
    // above (umt5/t5xxl/vae/clip_vision). LTX's connectors/audio-vae are
    // .safetensors so they aren't scanned at all.
    if (std.mem.indexOf(u8, lower, "wan") != null or
        std.mem.indexOf(u8, lower, "ltx") != null) return .video;
    if (std.mem.indexOf(u8, lower, "stable-diffusion") != null or
        std.mem.indexOf(u8, lower, "stable_diffusion") != null or
        std.mem.indexOf(u8, lower, "sdxl") != null or
        std.mem.indexOf(u8, lower, "flux") != null or
        std.mem.indexOf(u8, lower, "-sd-") != null) return .image;
    return .text;
}

const max_depth = 6;

/// File count + total bytes of a directory tree. Used by the delete dialog to
/// tell the user how much (and how many files) removing a model folder frees.
pub const DirStats = struct { files: usize = 0, bytes: u64 = 0 };

pub fn folderStats(gpa: std.mem.Allocator, dir: []const u8) DirStats {
    var s: DirStats = .{};
    folderStatsDepth(gpa, dir, 0, &s);
    return s;
}

fn folderStatsDepth(gpa: std.mem.Allocator, root: []const u8, depth: usize, s: *DirStats) void {
    if (depth > max_depth) return;
    const root_z = gpa.dupeZ(u8, root) catch return;
    defer gpa.free(root_z);
    const dir = c.opendir(root_z.ptr) orelse return;
    defer _ = c.closedir(dir);
    while (c.readdir(dir)) |entry| {
        const name = std.mem.span(@as([*:0]const u8, @ptrCast(&entry.*.d_name)));
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        const full = std.fs.path.join(gpa, &.{ root, name }) catch continue;
        defer gpa.free(full);
        const full_z = gpa.dupeZ(u8, full) catch continue;
        defer gpa.free(full_z);
        var st: Stat = undefined;
        if (!statPath(full_z.ptr, &st)) continue;
        if ((st.st_mode & c.S_IFMT) == c.S_IFDIR) {
            folderStatsDepth(gpa, full, depth + 1, s);
        } else {
            s.files += 1;
            s.bytes += @intCast(@max(0, st.st_size));
        }
    }
}

/// Recursively scan `root` for *.gguf, appending discovered models to `list`.
/// Silently ignores directories that don't exist or can't be opened.
pub fn scanDir(list: *ModelList, root: []const u8, source: Source) void {
    scanDirDepth(list, root, 0, source);
}

fn scanDirDepth(list: *ModelList, root: []const u8, depth: usize, source: Source) void {
    if (depth > max_depth) return;
    const root_z = list.gpa.dupeZ(u8, root) catch return;
    defer list.gpa.free(root_z);

    const dir = c.opendir(root_z.ptr) orelse return;
    defer _ = c.closedir(dir);

    while (c.readdir(dir)) |entry| {
        const name = std.mem.span(@as([*:0]const u8, @ptrCast(&entry.*.d_name)));
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;

        const full = std.fs.path.join(list.gpa, &.{ root, name }) catch continue;
        const full_z = list.gpa.dupeZ(u8, full) catch {
            list.gpa.free(full);
            continue;
        };
        defer list.gpa.free(full_z);

        var stat: Stat = undefined;
        if (!statPath(full_z.ptr, &stat)) {
            list.gpa.free(full);
            continue;
        }

        const is_dir = (stat.st_mode & c.S_IFMT) == c.S_IFDIR;
        if (is_dir) {
            scanDirDepth(list, full, depth + 1, source);
            list.gpa.free(full);
            continue;
        }

        if (!std.mem.endsWith(u8, name, ".gguf")) {
            list.gpa.free(full);
            continue;
        }
        const kind = classifyName(name) orelse {
            list.gpa.free(full);
            continue;
        };

        const base = std.fs.path.basename(full);
        const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
        const dpath = std.fs.path.dirname(full) orelse full;

        list.items.append(list.gpa, .{
            .path = full,
            .name = stem,
            .dir = dpath,
            .kind = kind,
            .size = @intCast(stat.st_size),
            .source = source,
        }) catch {
            list.gpa.free(full);
        };
    }
}

/// Scan the default `<home>/<subdir>` locations plus any `extra` directories.
/// `home` is the user's home directory (resolved by the caller from the process
/// environment); pass null to skip the default locations.
pub fn scanDefaults(list: *ModelList, home: ?[]const u8, extra: []const []const u8) void {
    if (home) |h| {
        // The app's own (cross-platform) models dir — where downloads now land,
        // and the ONLY source we own (and allow deleting from).
        if (config.modelsDirAlloc(list.gpa, h)) |app_dir| {
            defer list.gpa.free(app_dir);
            scanDir(list, app_dir, .zig_ai);
        }
        // Legacy / third-party locations kept as extra (read-only) sources.
        for (default_dirs) |dd| {
            const path = std.fs.path.join(list.gpa, &.{ h, dd.sub }) catch continue;
            defer list.gpa.free(path);
            scanDir(list, path, dd.source);
        }
    }
    for (extra) |d| scanDir(list, d, .custom);
}

/// True if `dir` (recursively) contains any `*.partial` file — the marker an
/// in-progress/interrupted download leaves behind. Used to flag a model folder
/// as an incomplete install.
pub fn hasPartial(gpa: std.mem.Allocator, dir: []const u8) bool {
    return hasPartialDepth(gpa, dir, 0);
}

fn hasPartialDepth(gpa: std.mem.Allocator, root: []const u8, depth: usize) bool {
    if (depth > max_depth) return false;
    const root_z = gpa.dupeZ(u8, root) catch return false;
    defer gpa.free(root_z);
    const dir = c.opendir(root_z.ptr) orelse return false;
    defer _ = c.closedir(dir);
    while (c.readdir(dir)) |entry| {
        const name = std.mem.span(@as([*:0]const u8, @ptrCast(&entry.*.d_name)));
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        if (std.mem.endsWith(u8, name, ".partial")) return true;
        const full = std.fs.path.join(gpa, &.{ root, name }) catch continue;
        defer gpa.free(full);
        const full_z = gpa.dupeZ(u8, full) catch continue;
        defer gpa.free(full_z);
        var st: Stat = undefined;
        if (!statPath(full_z.ptr, &st)) continue;
        if ((st.st_mode & c.S_IFMT) == c.S_IFDIR and hasPartialDepth(gpa, full, depth + 1)) return true;
    }
    return false;
}

/// True if a file named `name` exists directly inside `dir`.
pub fn fileExistsIn(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) bool {
    const full = std.fs.path.join(gpa, &.{ dir, name }) catch return false;
    defer gpa.free(full);
    const full_z = gpa.dupeZ(u8, full) catch return false;
    defer gpa.free(full_z);
    var st: Stat = undefined;
    return statPath(full_z.ptr, &st);
}

/// Recursively search `root` for the first file whose lowercased basename
/// contains any of `needles` and ends with any of `exts`. Returns an owned path
/// (caller frees) or null. Used to locate a Wan model's VAE / text-encoder
/// sidecars, which live alongside (or under) the diffusion .gguf.
pub fn findSupport(gpa: std.mem.Allocator, root: []const u8, needles: []const []const u8, exts: []const []const u8, exclude: []const []const u8) ?[]u8 {
    return findSupportDepth(gpa, root, needles, exts, exclude, 0);
}

fn findSupportDepth(gpa: std.mem.Allocator, root: []const u8, needles: []const []const u8, exts: []const []const u8, exclude: []const []const u8, depth: usize) ?[]u8 {
    if (depth > max_depth) return null;
    const root_z = gpa.dupeZ(u8, root) catch return null;
    defer gpa.free(root_z);

    const dir = c.opendir(root_z.ptr) orelse return null;
    defer _ = c.closedir(dir);

    while (c.readdir(dir)) |entry| {
        const name = std.mem.span(@as([*:0]const u8, @ptrCast(&entry.*.d_name)));
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;

        const full = std.fs.path.join(gpa, &.{ root, name }) catch continue;
        const full_z = gpa.dupeZ(u8, full) catch {
            gpa.free(full);
            continue;
        };
        defer gpa.free(full_z);

        var stat: Stat = undefined;
        if (!statPath(full_z.ptr, &stat)) {
            gpa.free(full);
            continue;
        }
        if ((stat.st_mode & c.S_IFMT) == c.S_IFDIR) {
            if (findSupportDepth(gpa, full, needles, exts, exclude, depth + 1)) |hit| {
                gpa.free(full);
                return hit;
            }
            gpa.free(full);
            continue;
        }

        if (matchesSupport(name, needles, exts, exclude)) return full;
        gpa.free(full);
    }
    return null;
}

fn matchesSupport(name: []const u8, needles: []const []const u8, exts: []const []const u8, exclude: []const []const u8) bool {
    var buf: [256]u8 = undefined;
    const lower = std.ascii.lowerString(buf[0..@min(name.len, buf.len)], name[0..@min(name.len, buf.len)]);
    var ext_ok = exts.len == 0;
    for (exts) |e| if (std.mem.endsWith(u8, lower, e)) {
        ext_ok = true;
    };
    if (!ext_ok) return false;
    for (exclude) |x| if (std.mem.indexOf(u8, lower, x) != null) return false;
    for (needles) |n| if (std.mem.indexOf(u8, lower, n) != null) return true;
    return false;
}

/// Format a byte count compactly (e.g. "4.1 GB"). Writes into `buf`.
pub fn humanSize(buf: []u8, bytes: u64) []const u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
    var val: f64 = @floatFromInt(bytes);
    var unit: usize = 0;
    while (val >= 1024 and unit + 1 < units.len) : (unit += 1) val /= 1024;
    return std.fmt.bufPrint(buf, "{d:.1} {s}", .{ val, units[unit] }) catch "?";
}
