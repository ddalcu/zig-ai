//! Native HuggingFace model downloader. Uses Zig's std HTTP client over the
//! std.Io threaded backend — no curl, no pthread — so it builds and runs on
//! Windows, Linux and macOS. Each network request runs on its own `std.Thread`
//! (search and download are short-lived, one-shot jobs), pushing results back to
//! the UI through a `Channel(Event)` and live progress through atomics.
//!
//! Endpoints (huggingface.co):
//!   - search:  GET /api/models?filter=gguf&sort=downloads&...&search=<q>
//!   - tree:    GET /api/models/{id}/tree/main          (per-repo .gguf files)
//!   - file:    GET /{id}/resolve/main/{path}           (the actual download)

const std = @import("std");
const channel = @import("../channel.zig");
const models = @import("../models.zig");
const manifest = @import("../manifest.zig");

const Io = std.Io;

/// One HuggingFace repo in the search results. `id` is "author/name" (owned).
pub const HFRepo = struct {
    id: []u8,
    downloads: i64 = 0,
    likes: i64 = 0,
    kind: models.Kind = .text,

    /// "author" portion of the id (borrows into `id`).
    pub fn author(self: HFRepo) []const u8 {
        const slash = std.mem.indexOfScalar(u8, self.id, '/') orelse return "";
        return self.id[0..slash];
    }
    /// "name" portion of the id (borrows into `id`).
    pub fn name(self: HFRepo) []const u8 {
        const slash = std.mem.indexOfScalar(u8, self.id, '/') orelse return self.id;
        return self.id[slash + 1 ..];
    }
};

/// One top-level file inside a repo. `path` is owned. `is_quant` marks a
/// standalone, user-selectable model weight (an interchangeable GGUF quant);
/// everything else (VAE, text encoders, tokenizers, configs) is a *support*
/// file that ships automatically alongside the chosen quant.
pub const RepoFile = struct {
    path: []u8,
    size: u64 = 0,
    is_quant: bool = false,
};

/// The set of files for a particular repo, as returned by the tree API.
pub const Files = struct {
    repo_id: []u8, // owned
    items: []RepoFile, // owned (each .path owned)
};

pub const Event = union(enum) {
    results: []HFRepo, // search complete; UI takes ownership
    files: Files, // tree listing complete; UI takes ownership
    file: []u8, // a new file in the set started downloading (owned basename)
    progress, // a download tick (UI reads the byte atomics)
    done: []u8, // whole set finished; final folder (owned)
    err: []u8, // owned message
};

/// One queued file in a multi-file download (the chosen quant + its support
/// files). `path` is owned; `size` (from the tree API) drives overall progress.
pub const DlFile = struct {
    path: []u8,
    size: u64,
};

pub fn freeRepos(gpa: std.mem.Allocator, repos: []HFRepo) void {
    for (repos) |r| gpa.free(r.id);
    gpa.free(repos);
}

pub fn freeFiles(gpa: std.mem.Allocator, files: Files) void {
    gpa.free(files.repo_id);
    for (files.items) |f| gpa.free(f.path);
    gpa.free(files.items);
}

/// Heap-allocated networking context with a stable address. `Threaded.io()`
/// captures a pointer to the embedded `Threaded`, and `Client` keeps a
/// connection pool, so neither may be copied after init — hence the box.
const Net = struct {
    threaded: Io.Threaded,
    client: std.http.Client,

    fn io(self: *Net) Io {
        return self.threaded.io();
    }
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    events: channel.Channel(Event),
    job: channel.JobState = .{},

    /// Live download progress (bytes can exceed i32, so these live outside
    /// JobState which is step/total i32). Read by the UI every frame.
    bytes_done: std.atomic.Value(u64) = .init(0),
    bytes_total: std.atomic.Value(u64) = .init(0),
    speed_bps: std.atomic.Value(u64) = .init(0),
    /// "File i of N" position within the current multi-file download.
    file_index: std.atomic.Value(u32) = .init(0),
    file_count: std.atomic.Value(u32) = .init(0),

    net: ?*Net = null,

    pub fn init(gpa: std.mem.Allocator) Backend {
        return .{ .gpa = gpa, .events = channel.Channel(Event).init(gpa) };
    }

    pub fn deinit(self: *Backend) void {
        if (self.net) |n| {
            n.client.deinit();
            n.threaded.deinit();
            self.gpa.destroy(n);
        }
        self.events.deinit();
    }

    /// Lazily build the networking context at a stable heap address. Called from
    /// the UI thread before spawning a worker (mirrors the other backends'
    /// `start`). Returns the context or null on OOM.
    fn ensureNet(self: *Backend) ?*Net {
        if (self.net) |n| return n;
        const n = self.gpa.create(Net) catch return null;
        n.threaded = Io.Threaded.init(self.gpa, .{});
        n.client = .{ .allocator = self.gpa, .io = n.threaded.io() };
        self.net = n;
        return n;
    }

    pub fn isBusy(self: *Backend) bool {
        return self.job.isRunning();
    }

    fn emitErr(self: *Backend, comptime f: []const u8, args: anytype) void {
        const msg = std.fmt.allocPrint(self.gpa, f, args) catch return;
        self.events.push(.{ .err = msg });
    }

    // --- search ---------------------------------------------------------------

    /// Kick off an async HF search. `query` and `category` (a models.Kind, or
    /// null for "All") are duped; results arrive as a `.results` event.
    pub fn search(self: *Backend, query: []const u8, category: ?models.Kind) void {
        const q = self.gpa.dupe(u8, query) catch return;
        const th = std.Thread.spawn(.{}, searchWorker, .{ self, q, category }) catch {
            self.gpa.free(q);
            self.emitErr("could not start search thread", .{});
            return;
        };
        th.detach();
    }

    fn searchWorker(self: *Backend, query: []u8, category: ?models.Kind) void {
        defer self.gpa.free(query);
        const net = self.ensureNet() orelse {
            self.emitErr("downloader: out of memory", .{});
            return;
        };

        // Bias the search term toward the category's known model families; the
        // post-filter on classified Kind does the real narrowing.
        const bias: []const u8 = switch (category orelse return self.runSearch(net, query, null)) {
            .text => "",
            .image => " flux",
            .video => " wan",
            .tts => " tts",
        };
        const term = std.fmt.allocPrint(self.gpa, "{s}{s}", .{ query, bias }) catch {
            self.emitErr("downloader: out of memory", .{});
            return;
        };
        defer self.gpa.free(term);
        self.runSearch(net, term, category);
    }

    fn runSearch(self: *Backend, net: *Net, term: []const u8, category: ?models.Kind) void {
        // Percent-encode the search term into the query string.
        var url_buf: std.ArrayList(u8) = .empty;
        defer url_buf.deinit(self.gpa);
        url_buf.appendSlice(self.gpa, "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=50&full=true&config=false&search=") catch return;
        percentEncode(self.gpa, &url_buf, term) catch return;
        url_buf.appendSlice(self.gpa, "&expand[]=downloads&expand[]=likes&expand[]=pipeline_tag") catch return;

        const body = self.httpGet(net, url_buf.items, null) catch |e| {
            self.emitErr("HF search failed: {s}", .{@errorName(e)});
            return;
        };
        defer self.gpa.free(body);

        const repos = self.parseSearch(body, category) catch {
            self.emitErr("could not parse HF search response", .{});
            return;
        };
        self.events.push(.{ .results = repos });
    }

    const ApiModel = struct {
        id: ?[]const u8 = null,
        modelId: ?[]const u8 = null,
        downloads: ?i64 = null,
        likes: ?i64 = null,
    };

    fn parseSearch(self: *Backend, body: []const u8, category: ?models.Kind) ![]HFRepo {
        const parsed = std.json.parseFromSlice([]ApiModel, self.gpa, body, .{
            .ignore_unknown_fields = true,
        }) catch return error.ParseFailed;
        defer parsed.deinit();

        var out: std.ArrayList(HFRepo) = .empty;
        errdefer {
            for (out.items) |r| self.gpa.free(r.id);
            out.deinit(self.gpa);
        }
        for (parsed.value) |m| {
            const id = m.id orelse m.modelId orelse continue;
            const nm = lastSegment(id);
            const kind = models.classifyName(nm) orelse continue;
            if (category) |want| if (kind != want) continue;
            const id_dup = try self.gpa.dupe(u8, id);
            out.append(self.gpa, .{
                .id = id_dup,
                .downloads = m.downloads orelse 0,
                .likes = m.likes orelse 0,
                .kind = kind,
            }) catch {
                self.gpa.free(id_dup);
                return error.OutOfMemory;
            };
        }
        return out.toOwnedSlice(self.gpa);
    }

    // --- file tree (quants) ---------------------------------------------------

    /// Kick off an async listing of a repo's top-level files (selectable quants
    /// plus the support files that ship with them).
    pub fn listFiles(self: *Backend, repo_id: []const u8) void {
        const id = self.gpa.dupe(u8, repo_id) catch return;
        const th = std.Thread.spawn(.{}, listFilesWorker, .{ self, id }) catch {
            self.gpa.free(id);
            self.emitErr("could not start tree thread", .{});
            return;
        };
        th.detach();
    }

    fn listFilesWorker(self: *Backend, repo_id: []u8) void {
        const net = self.ensureNet() orelse {
            self.gpa.free(repo_id);
            self.emitErr("downloader: out of memory", .{});
            return;
        };
        const url = std.fmt.allocPrint(self.gpa, "https://huggingface.co/api/models/{s}/tree/main", .{repo_id}) catch {
            self.gpa.free(repo_id);
            return;
        };
        defer self.gpa.free(url);

        const body = self.httpGet(net, url, null) catch |e| {
            self.gpa.free(repo_id);
            self.emitErr("HF file list failed: {s}", .{@errorName(e)});
            return;
        };
        defer self.gpa.free(body);

        const items = self.parseTree(body) catch {
            self.gpa.free(repo_id);
            self.emitErr("could not parse HF file list", .{});
            return;
        };
        self.events.push(.{ .files = .{ .repo_id = repo_id, .items = items } });
    }

    const ApiTreeEntry = struct {
        type: ?[]const u8 = null,
        path: ?[]const u8 = null,
        size: ?u64 = null,
    };

    fn parseTree(self: *Backend, body: []const u8) ![]RepoFile {
        const parsed = std.json.parseFromSlice([]ApiTreeEntry, self.gpa, body, .{
            .ignore_unknown_fields = true,
        }) catch return error.ParseFailed;
        defer parsed.deinit();

        var out: std.ArrayList(RepoFile) = .empty;
        errdefer {
            for (out.items) |f| self.gpa.free(f.path);
            out.deinit(self.gpa);
        }
        for (parsed.value) |e| {
            const t = e.type orelse continue;
            if (!std.mem.eql(u8, t, "file")) continue;
            const path = e.path orelse continue;
            if (std.mem.indexOfScalar(u8, path, '/') != null) continue; // top-level only
            const cls = classifyTreeFile(path) orelse continue;
            const path_dup = try self.gpa.dupe(u8, path);
            out.append(self.gpa, .{ .path = path_dup, .size = e.size orelse 0, .is_quant = cls }) catch {
                self.gpa.free(path_dup);
                return error.OutOfMemory;
            };
        }
        // Quants first (so the popover lists them up top), then support files,
        // each group sorted by name.
        std.mem.sort(RepoFile, out.items, {}, struct {
            fn lt(_: void, a: RepoFile, b: RepoFile) bool {
                if (a.is_quant != b.is_quant) return a.is_quant;
                return std.mem.lessThan(u8, a.path, b.path);
            }
        }.lt);
        return out.toOwnedSlice(self.gpa);
    }

    // --- download -------------------------------------------------------------

    /// Begin downloading a whole model: the chosen quant plus every support file
    /// in `files`, into `<models_root>/<kind>/<author>/<name>/` (grouped by model
    /// kind: chat/image/video/audio — so all of a multi-file image/video model's
    /// files land together in one folder). `models_root` is the app's own
    /// cross-platform models dir (see `config.modelsDirAlloc`). Overall progress
    /// streams through the byte atomics + `.progress`; each file start pushes
    /// `.file`; the set finishing pushes `.done` with the folder.
    pub fn download(self: *Backend, models_root: []const u8, repo_id: []const u8, files: []const RepoFile, kind: models.Kind, sidecars: []const manifest.Sidecar) void {
        if (self.isBusy()) return;
        if (files.len == 0) return;
        const h = self.gpa.dupe(u8, models_root) catch return;
        const id = self.gpa.dupe(u8, repo_id) catch {
            self.gpa.free(h);
            return;
        };
        // Dupe the file set on the UI thread (callers' slices are transient).
        const set = self.gpa.alloc(DlFile, files.len) catch {
            self.gpa.free(h);
            self.gpa.free(id);
            return;
        };
        var n: usize = 0;
        while (n < files.len) : (n += 1) {
            const p = self.gpa.dupe(u8, files[n].path) catch {
                for (set[0..n]) |f| self.gpa.free(f.path);
                self.gpa.free(set);
                self.gpa.free(h);
                self.gpa.free(id);
                return;
            };
            set[n] = .{ .path = p, .size = files[n].size };
        }

        self.job.beginJob();
        self.bytes_done.store(0, .release);
        self.bytes_total.store(0, .release);
        self.speed_bps.store(0, .release);
        self.file_index.store(0, .release);
        self.file_count.store(@intCast(files.len + sidecars.len), .release);

        // `sidecars` point into the static `manifest.entries`, so the slice is
        // safe to hand to the worker thread without copying.
        const th = std.Thread.spawn(.{}, downloadWorker, .{ self, h, id, set, kind, sidecars }) catch {
            for (set) |f| self.gpa.free(f.path);
            self.gpa.free(set);
            self.gpa.free(h);
            self.gpa.free(id);
            self.job.endJob();
            self.emitErr("could not start download thread", .{});
            return;
        };
        th.detach();
    }

    fn downloadWorker(self: *Backend, models_root: []u8, repo_id: []u8, set: []DlFile, kind: models.Kind, sidecars: []const manifest.Sidecar) void {
        defer {
            self.gpa.free(models_root);
            self.gpa.free(repo_id);
            for (set) |f| self.gpa.free(f.path);
            self.gpa.free(set);
            self.job.endJob();
        }
        const net = self.ensureNet() orelse {
            self.emitErr("downloader: out of memory", .{});
            return;
        };
        self.runAll(net, models_root, repo_id, set, kind, sidecars) catch |e| {
            if (e == error.Canceled) return;
            self.emitErr("download failed: {s}", .{@errorName(e)});
        };
    }

    fn runAll(self: *Backend, net: *Net, models_root: []u8, repo_id: []u8, set: []DlFile, kind: models.Kind, sidecars: []const manifest.Sidecar) !void {
        const gpa = self.gpa;
        const io = net.io();

        const author = firstSegment(repo_id);
        const name = lastSegment(repo_id);
        const dir = try std.fs.path.join(gpa, &.{ models_root, kind.folder(), author, name });
        defer gpa.free(dir);
        Io.Dir.cwd().createDirPath(io, dir) catch |e| switch (e) {
            error.PathAlreadyExists => {},
            else => return e,
        };

        // Overall total is the sum of the (tree-reported) file sizes; the bar
        // tracks the running sum of completed bytes across all files.
        var total: u64 = 0;
        for (set) |f| total += f.size;
        self.bytes_total.store(total, .release);

        var base_done: u64 = 0;
        for (set, 0..) |f, i| {
            self.file_index.store(@intCast(i + 1), .release);
            const base = std.fs.path.basename(f.path);
            self.events.push(.{ .file = gpa.dupe(u8, base) catch return error.OutOfMemory });
            const written = try self.downloadOne(net, dir, repo_id, f.path, null, base_done);
            // Prefer the real byte count when the tree size was missing/wrong.
            base_done += if (written > f.size) written else f.size;
            self.bytes_done.store(base_done, .release);
        }

        // Curated cross-repo sidecars (FLUX VAE/encoder, Wan VAE/umt5) into the
        // SAME folder, so the model is runnable. Their sizes are unknown up front,
        // so they extend `bytes_total` as they download.
        for (sidecars, 0..) |sc, i| {
            self.file_index.store(@intCast(set.len + i + 1), .release);
            const shown = sc.dest orelse std.fs.path.basename(sc.file);
            self.events.push(.{ .file = gpa.dupe(u8, shown) catch return error.OutOfMemory });
            const written = self.downloadOne(net, dir, sc.repo, sc.file, sc.dest, base_done) catch |e| {
                if (e == error.Canceled) return e;
                // A missing/renamed sidecar shouldn't abort the whole model.
                self.emitErr("sidecar {s} failed: {s}", .{ sc.label, @errorName(e) });
                continue;
            };
            base_done += written;
            self.bytes_total.store(base_done, .release);
            self.bytes_done.store(base_done, .release);
        }

        const folder = try gpa.dupe(u8, dir);
        self.events.push(.{ .done = folder });
    }

    /// Download a single file into `dir` (saved as `dest_name`, or the file's
    /// basename), from `repo_id`, reporting overall progress as `base_done +
    /// <bytes of this file>`. Returns this file's total size.
    fn downloadOne(self: *Backend, net: *Net, dir: []const u8, repo_id: []const u8, file: []const u8, dest_name: ?[]const u8, base_done: u64) !u64 {
        const gpa = self.gpa;
        const io = net.io();

        const dest = try std.fs.path.join(gpa, &.{ dir, dest_name orelse std.fs.path.basename(file) });
        defer gpa.free(dest);
        const partial = try std.fmt.allocPrint(gpa, "{s}.partial", .{dest});
        defer gpa.free(partial);

        // Skip files already present (e.g. a re-run after a partial set).
        if (Io.Dir.openFileAbsolute(io, dest, .{})) |existing| {
            const sz = if (existing.stat(io)) |s| s.size else |_| 0;
            existing.close(io);
            return sz;
        } else |_| {}

        // Resume support: if a .partial already exists, request the remainder.
        var have: u64 = 0;
        if (Io.Dir.openFileAbsolute(io, partial, .{})) |existing| {
            if (existing.stat(io)) |s| have = s.size else |_| {}
            existing.close(io);
        } else |_| {}

        const url = try std.fmt.allocPrint(gpa, "https://huggingface.co/{s}/resolve/main/{s}", .{ repo_id, file });
        defer gpa.free(url);
        const uri = try std.Uri.parse(url);

        var range_buf: [64]u8 = undefined;
        var extra_headers: []const std.http.Header = &.{};
        if (have > 0) {
            const rng = try std.fmt.bufPrint(&range_buf, "bytes={d}-", .{have});
            extra_headers = &.{.{ .name = "range", .value = rng }};
        }

        var req = try net.client.request(.GET, uri, .{ .extra_headers = extra_headers });
        defer req.deinit();
        try req.sendBodiless();

        var redirect_buf: [8 * 1024]u8 = undefined;
        var response = try req.receiveHead(&redirect_buf);

        // 206 => server honored Range (append); anything else => start over.
        const resumed = response.head.status == .partial_content and have > 0;
        const start_at: u64 = if (resumed) have else 0;

        var out_file = if (resumed)
            try Io.Dir.openFileAbsolute(io, partial, .{ .mode = .write_only })
        else
            try Io.Dir.createFileAbsolute(io, partial, .{ .truncate = true });
        // Close on any early return; the success path closes explicitly *before*
        // the rename, which Windows requires.
        errdefer out_file.close(io);

        var file_buf: [64 * 1024]u8 = undefined;
        var fw = out_file.writer(io, &file_buf);
        if (resumed) try fw.seekTo(start_at);

        var transfer_buf: [64 * 1024]u8 = undefined;
        const reader = response.reader(&transfer_buf);

        const t_start = (Io.Clock.awake).now(io);
        var done: u64 = start_at;
        var chunk: [64 * 1024]u8 = undefined;
        while (true) {
            if (self.job.cancelRequested()) {
                fw.interface.flush() catch {};
                return error.Canceled;
            }
            const n = reader.readSliceShort(&chunk) catch return error.ReadFailed;
            if (n == 0) break; // EOF
            try fw.interface.writeAll(chunk[0..n]);
            done += n;
            self.bytes_done.store(base_done + done, .release);

            const now = (Io.Clock.awake).now(io);
            const elapsed_ns: i128 = @as(i128, now.nanoseconds) - @as(i128, t_start.nanoseconds);
            if (elapsed_ns > 0) {
                const transferred = done - start_at;
                const bps = @divTrunc(@as(i128, transferred) * std.time.ns_per_s, elapsed_ns);
                self.speed_bps.store(@intCast(@max(0, bps)), .release);
            }
            self.events.push(.progress);
        }
        try fw.interface.flush();
        out_file.close(io);

        try Io.Dir.renameAbsolute(partial, dest, io);
        return done;
    }

    // --- shared HTTP ----------------------------------------------------------

    /// One-shot GET capturing the (uncompressed) response body into an owned,
    /// growable buffer. Caller frees the returned slice.
    fn httpGet(self: *Backend, net: *Net, url: []const u8, extra: ?[]const std.http.Header) ![]u8 {
        var body: Io.Writer.Allocating = .init(self.gpa);
        defer body.deinit();
        const res = try net.client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &body.writer,
            .extra_headers = extra orelse &.{},
        });
        if (res.status.class() != .success) return error.HttpError;
        return body.toOwnedSlice();
    }
};

// --- helpers -----------------------------------------------------------------

/// Classify a top-level repo file for the downloader. Returns `true` if it is a
/// standalone, user-selectable model quant; `false` if it is a support file to
/// fetch automatically (VAE, text encoder, tokenizer, config); `null` to ignore
/// it entirely (README, images, .gitattributes, …).
fn classifyTreeFile(path: []const u8) ?bool {
    var buf: [256]u8 = undefined;
    const n = @min(path.len, buf.len);
    const lower = std.ascii.lowerString(buf[0..n], path[0..n]);
    // Multimodal projection weights are an optional add-on, not needed to run
    // the base model — skip to avoid surprising multi-GB extras.
    if (std.mem.indexOf(u8, lower, "mmproj") != null) return null;

    if (std.mem.endsWith(u8, lower, ".gguf")) {
        // Some GGUFs are required *components* shipped beside the model weight
        // (tokenizer, vocoder, text encoder, connectors). classifyName already
        // rejects umt5/t5xxl/vae/clip; these keywords catch the rest. They're
        // support files, not interchangeable quants.
        const support_kw = [_][]const u8{
            "tokenizer", "vocoder", "vocab", "encoder", "connector",
        };
        for (support_kw) |kw| {
            if (std.mem.indexOf(u8, lower, kw) != null) return false;
        }
        // A GGUF that still classifies as a standalone model is an
        // interchangeable quant (the user picks one); anything else is support.
        return models.classifyName(path) != null;
    }
    // Non-GGUF support files the backends load beside the main weight.
    const support_exts = [_][]const u8{
        ".safetensors", ".json", ".model",   ".txt",
        ".vocab",       ".spm",  ".tiktoken", ".merges",
    };
    for (support_exts) |ext| {
        if (std.mem.endsWith(u8, lower, ext)) return false;
    }
    return null;
}

fn firstSegment(id: []const u8) []const u8 {
    const slash = std.mem.indexOfScalar(u8, id, '/') orelse return id;
    return id[0..slash];
}

fn lastSegment(id: []const u8) []const u8 {
    const slash = std.mem.lastIndexOfScalar(u8, id, '/') orelse return id;
    return id[slash + 1 ..];
}

/// Append `s` to `out`, percent-encoding everything outside the RFC 3986
/// unreserved set. Portable replacement for curl's `--data-urlencode`.
fn percentEncode(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    const hex = "0123456789ABCDEF";
    for (s) |ch| {
        const unreserved = (ch >= 'A' and ch <= 'Z') or
            (ch >= 'a' and ch <= 'z') or
            (ch >= '0' and ch <= '9') or
            ch == '-' or ch == '_' or ch == '.' or ch == '~';
        if (unreserved) {
            try out.append(gpa, ch);
        } else {
            try out.append(gpa, '%');
            try out.append(gpa, hex[ch >> 4]);
            try out.append(gpa, hex[ch & 0xF]);
        }
    }
}
