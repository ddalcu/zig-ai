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
    /// Last-modified time as a Unix epoch (seconds); 0 means unknown.
    last_modified: i64 = 0,
    /// Quant size range in bytes (smallest..largest GGUF in the repo). Only valid
    /// once `size_loaded` — the search response doesn't carry per-file sizes, so
    /// these are filled in lazily by background tree fetches (see `repo_size`).
    size_min: u64 = 0,
    size_max: u64 = 0,
    size_loaded: bool = false,
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

/// A resolved quant size range for one repo, delivered by background enrichment
/// after the initial search results. `id` is owned ("author/name").
pub const RepoSize = struct { id: []u8, min: u64, max: u64 };

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
    repo_size: RepoSize, // a repo's quant size range resolved; UI takes ownership of `id`
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
    /// Consecutive retry attempt for the current file (0 = streaming normally,
    /// >0 = reconnecting after a dropped/failed connection). Drives the UI's
    /// "Reconnecting… (n/N)" status. See `downloadOne`.
    retry_attempt: std.atomic.Value(u32) = .init(0),

    /// Bumped on every search so a still-running background size-enrichment pass
    /// (from an earlier search) stops as soon as a newer search starts.
    search_gen: std.atomic.Value(u64) = .init(0),

    /// Count of worker threads currently touching `net`. `deinit` waits for this
    /// to reach 0 before tearing down the Io/Client — tearing it down mid-request
    /// panics in std's cancel path.
    active_workers: std.atomic.Value(u32) = .init(0),
    /// Set by `deinit` so long-running workers (enrichment) bail promptly.
    shutting_down: std.atomic.Value(bool) = .init(false),

    /// Short-lived response cache for cheap JSON GETs (search + file trees), so
    /// e.g. opening a repo's quant dropdown reuses the tree the size-enrichment
    /// pass already fetched instead of re-hitting HuggingFace. Keyed by URL,
    /// 30-minute TTL. Downloads bypass this (they stream with Range headers).
    http_cache: std.ArrayList(CacheEntry) = .empty,
    cache_lock: channel.SpinLock = .{},

    net: ?*Net = null,

    /// One cached HTTP response. `url` and `body` are owned; `at_ms` is a monotonic
    /// timestamp (ms) from the `Io` clock.
    const CacheEntry = struct { url: []u8, body: []u8, at_ms: i64 };
    const cache_ttl_ms: i64 = 30 * 60 * 1000;

    pub fn init(gpa: std.mem.Allocator) Backend {
        return .{ .gpa = gpa, .events = channel.Channel(Event).init(gpa) };
    }

    pub fn deinit(self: *Backend) void {
        // Stop background work and wait for in-flight requests to finish before
        // tearing down the Io/Client — destroying them mid-request panics inside
        // std's cancellation path.
        self.shutting_down.store(true, .release);
        _ = self.search_gen.fetchAdd(1, .monotonic); // halt the enrichment loop
        self.job.requestCancel(); // abort an active download
        while (self.active_workers.load(.acquire) > 0) std.Thread.yield() catch {};

        for (self.http_cache.items) |e| {
            self.gpa.free(e.url);
            self.gpa.free(e.body);
        }
        self.http_cache.deinit(self.gpa);
        if (self.net) |n| {
            n.client.deinit();
            n.threaded.deinit();
            self.gpa.destroy(n);
        }
        self.events.deinit();
    }

    /// Mark a worker thread as live (it's about to use `net`); pair with
    /// `workerExit` via `defer` so `deinit` knows when teardown is safe.
    fn workerEnter(self: *Backend) void {
        _ = self.active_workers.fetchAdd(1, .acq_rel);
    }
    fn workerExit(self: *Backend) void {
        _ = self.active_workers.fetchSub(1, .acq_rel);
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
        // A newer search supersedes any in-flight size enrichment from an older one.
        const gen = self.search_gen.fetchAdd(1, .monotonic) + 1;
        const q = self.gpa.dupe(u8, query) catch return;
        const th = std.Thread.spawn(.{}, searchWorker, .{ self, q, category, gen }) catch {
            self.gpa.free(q);
            self.emitErr("could not start search thread", .{});
            return;
        };
        th.detach();
    }

    fn searchWorker(self: *Backend, query: []u8, category: ?models.Kind, gen: u64) void {
        self.workerEnter();
        defer self.workerExit();
        defer self.gpa.free(query);
        const net = self.ensureNet() orelse {
            self.emitErr("downloader: out of memory", .{});
            return;
        };

        // Bias the search term toward the category's known model families; the
        // post-filter on classified Kind does the real narrowing.
        const bias: []const u8 = switch (category orelse return self.runSearch(net, query, null, gen)) {
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
        self.runSearch(net, term, category, gen);
    }

    fn runSearch(self: *Backend, net: *Net, term: []const u8, category: ?models.Kind, gen: u64) void {
        // Percent-encode the search term into the query string.
        var url_buf: std.ArrayList(u8) = .empty;
        defer url_buf.deinit(self.gpa);
        url_buf.appendSlice(self.gpa, "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=50&full=true&config=false&search=") catch return;
        percentEncode(self.gpa, &url_buf, term) catch return;
        url_buf.appendSlice(self.gpa, "&expand[]=downloads&expand[]=likes&expand[]=pipeline_tag&expand[]=lastModified") catch return;

        const body = self.httpGet(net, url_buf.items, null) catch |e| {
            self.emitErr("HF search failed: {s}", .{@errorName(e)});
            return;
        };
        defer self.gpa.free(body);

        const repos = self.parseSearch(body, category) catch {
            self.emitErr("could not parse HF search response", .{});
            return;
        };

        // Snapshot the repo ids before handing `repos` to the UI (which takes
        // ownership), so the background size pass has its own copy to work from.
        const ids = self.dupeIds(repos);
        self.events.push(.{ .results = repos });
        if (ids) |id_list| self.enrichSizes(net, id_list, gen);
    }

    /// Duplicate every repo id into a freshly owned slice-of-slices, or null on
    /// OOM (callers degrade gracefully — sizes simply stay unresolved).
    fn dupeIds(self: *Backend, repos: []const HFRepo) ?[][]u8 {
        const ids = self.gpa.alloc([]u8, repos.len) catch return null;
        var n: usize = 0;
        for (repos) |r| {
            ids[n] = self.gpa.dupe(u8, r.id) catch {
                for (ids[0..n]) |id| self.gpa.free(id);
                self.gpa.free(ids);
                return null;
            };
            n += 1;
        }
        return ids;
    }

    /// Fetch each repo's file tree in turn and emit its quant size range as a
    /// `repo_size` event. Bails the moment a newer search bumps `search_gen`.
    /// Takes ownership of `ids` (frees them).
    fn enrichSizes(self: *Backend, net: *Net, ids: [][]u8, gen: u64) void {
        defer {
            for (ids) |id| self.gpa.free(id);
            self.gpa.free(ids);
        }
        for (ids) |id| {
            if (self.shutting_down.load(.acquire)) return; // app is quitting
            if (self.search_gen.load(.monotonic) != gen) return; // superseded
            const range = self.fetchQuantSizeRange(net, id) orelse continue;
            const id_dup = self.gpa.dupe(u8, id) catch continue;
            self.events.push(.{ .repo_size = .{ .id = id_dup, .min = range.min, .max = range.max } });
        }
    }

    /// Fetch a repo's tree and return the min/max size of its GGUF quants, or null
    /// if the fetch fails or the repo has no sized quants.
    fn fetchQuantSizeRange(self: *Backend, net: *Net, repo_id: []const u8) ?struct { min: u64, max: u64 } {
        const url = std.fmt.allocPrint(self.gpa, "https://huggingface.co/api/models/{s}/tree/main", .{repo_id}) catch return null;
        defer self.gpa.free(url);
        const body = self.httpGet(net, url, null) catch return null;
        defer self.gpa.free(body);
        const items = self.parseTree(body) catch return null;
        defer {
            for (items) |f| self.gpa.free(f.path);
            self.gpa.free(items);
        }
        var mn: u64 = std.math.maxInt(u64);
        var mx: u64 = 0;
        var any = false;
        for (items) |f| {
            if (!f.is_quant or f.size == 0) continue;
            any = true;
            mn = @min(mn, f.size);
            mx = @max(mx, f.size);
        }
        if (!any) return null;
        return .{ .min = mn, .max = mx };
    }

    const ApiModel = struct {
        id: ?[]const u8 = null,
        modelId: ?[]const u8 = null,
        downloads: ?i64 = null,
        likes: ?i64 = null,
        lastModified: ?[]const u8 = null,
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
                .last_modified = if (m.lastModified) |s| parseIso8601(s) orelse 0 else 0,
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
        self.workerEnter();
        defer self.workerExit();
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
        self.workerEnter();
        defer self.workerExit();
        defer {
            self.gpa.free(models_root);
            self.gpa.free(repo_id);
            for (set) |f| self.gpa.free(f.path);
            self.gpa.free(set);
            self.retry_attempt.store(0, .release);
            self.job.endJob();
        }
        const net = self.ensureNet() orelse {
            self.emitErr("downloader: out of memory", .{});
            return;
        };
        self.runAll(net, models_root, repo_id, set, kind, sidecars) catch |e| {
            // Aborted = the retry loop already reported a detailed failure.
            if (e == error.Canceled or e == error.Aborted) return;
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
                if (e == error.Aborted) continue; // already reported; sidecars are optional
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
    /// Max *consecutive* reconnect attempts that download nothing before a file is
    /// declared failed. Because the counter resets whenever an attempt makes
    /// progress, a download survives an unlimited number of dropped connections
    /// (HuggingFace's specialty) as long as each reconnect transfers some bytes.
    pub const max_stalls: u32 = 20;

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

        // Reconnect-and-resume loop. On a transient failure we sleep (exponential
        // backoff) and retry, resuming from the .partial. `error.Aborted` means a
        // failure was already surfaced to the user via `emitErr`.
        self.retry_attempt.store(0, .release);
        var stalls: u32 = 0;
        while (true) {
            const have_before = partialSize(io, partial);
            const written = self.downloadAttempt(net, dest, partial, repo_id, file, base_done) catch |e| {
                if (e == error.Canceled) return e;
                if (e == error.NotAvailable) {
                    self.emitErr("download failed: {s} is not available on HuggingFace", .{std.fs.path.basename(file)});
                    return error.Aborted;
                }
                // Retryable (dropped connection, 5xx, range reset, read error…).
                if (partialSize(io, partial) > have_before) stalls = 0 else stalls += 1;
                if (stalls >= max_stalls) {
                    self.emitErr("download failed after {d} retries: {s}", .{ max_stalls, @errorName(e) });
                    return error.Aborted;
                }
                self.retry_attempt.store(stalls, .release);
                self.events.push(.progress); // refresh the "Reconnecting…" status
                if (self.sleepOrCancel(io, backoffMs(stalls))) return error.Canceled;
                continue;
            };
            self.retry_attempt.store(0, .release);
            return written;
        }
    }

    /// A single download attempt: open/resume the .partial, GET (with a Range
    /// header when resuming), stream to disk, and rename into place. Returns the
    /// file's total size, or a (mostly retryable) error. `error.NotAvailable` =
    /// permanent (4xx); everything else the caller may retry.
    fn downloadAttempt(self: *Backend, net: *Net, dest: []const u8, partial: []const u8, repo_id: []const u8, file: []const u8, base_done: u64) !u64 {
        const gpa = self.gpa;
        const io = net.io();

        // Resume: if a .partial already exists, request the remainder.
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
        const status = response.head.status;

        // Requested range is past EOF (stale/oversized .partial) → reset and retry
        // from scratch on the next attempt.
        if (status == .range_not_satisfiable) {
            if (Io.Dir.createFileAbsolute(io, partial, .{ .truncate = true })) |f| f.close(io) else |_| {}
            return error.RangeReset;
        }
        if (status != .ok and status != .partial_content) {
            const code = @intFromEnum(status);
            if (code >= 400 and code < 500) return error.NotAvailable; // 404/401/403 — won't fix
            return error.ServerError; // 5xx etc. — transient
        }
        // Connection (re)established — clear any "Reconnecting…" status.
        self.retry_attempt.store(0, .release);

        // 206 => server honored Range (append); 200 => full file (start over).
        const resumed = status == .partial_content and have > 0;
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
            const n = reader.readSliceShort(&chunk) catch {
                fw.interface.flush() catch {}; // keep what we got so the retry resumes
                return error.ReadFailed;
            };
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

    /// Current size of a .partial file (0 if absent) — used to tell whether a
    /// failed attempt made any progress.
    fn partialSize(io: Io, path: []const u8) u64 {
        if (Io.Dir.openFileAbsolute(io, path, .{})) |f| {
            defer f.close(io);
            if (f.stat(io)) |s| return s.size else |_| {}
        } else |_| {}
        return 0;
    }

    /// Exponential backoff (1s, 2s, 4s, 8s, 16s, then capped at 30s) for the
    /// `n`-th consecutive stalled attempt (n ≥ 1).
    fn backoffMs(n: u32) u64 {
        const shift: u6 = @intCast(@min(n -| 1, 5));
        return @min(@as(u64, 30_000), @as(u64, 1000) << shift);
    }

    /// Sleep `ms`, but wake early (returning true) if the download is canceled.
    fn sleepOrCancel(self: *Backend, io: Io, ms: u64) bool {
        var slept: u64 = 0;
        while (slept < ms) {
            if (self.job.cancelRequested()) return true;
            const step = @min(ms - slept, 200);
            Io.sleep(io, Io.Duration.fromMilliseconds(step), .awake) catch return true;
            slept += step;
        }
        return false;
    }

    // --- shared HTTP ----------------------------------------------------------

    /// One-shot GET capturing the (uncompressed) response body into an owned,
    /// growable buffer. Caller frees the returned slice.
    fn httpGet(self: *Backend, net: *Net, url: []const u8, extra: ?[]const std.http.Header) ![]u8 {
        // Plain GETs (no custom headers) are cacheable; serve a fresh copy from
        // the cache when one is still warm.
        const cacheable = extra == null;
        if (cacheable) {
            if (self.cacheGet(net, url)) |hit| return hit;
        }
        var body: Io.Writer.Allocating = .init(self.gpa);
        defer body.deinit();
        const res = try net.client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &body.writer,
            .extra_headers = extra orelse &.{},
        });
        if (res.status.class() != .success) return error.HttpError;
        const out = try body.toOwnedSlice();
        if (cacheable) self.cachePut(net, url, out); // stores its own copy
        return out;
    }

    fn nowMs(net: *Net) i64 {
        return Io.Timestamp.now(net.io(), .awake).toMilliseconds();
    }

    /// Return an owned copy of a still-warm cached response for `url`, or null.
    fn cacheGet(self: *Backend, net: *Net, url: []const u8) ?[]u8 {
        const now = nowMs(net);
        self.cache_lock.lock();
        defer self.cache_lock.unlock();
        for (self.http_cache.items) |e| {
            if (std.mem.eql(u8, e.url, url)) {
                if (now - e.at_ms <= cache_ttl_ms) return self.gpa.dupe(u8, e.body) catch null;
                return null; // stale — a refetch will replace it
            }
        }
        return null;
    }

    /// Store a copy of `body` for `url` (replacing any prior entry) and drop any
    /// entries that have aged out, so the cache stays small.
    fn cachePut(self: *Backend, net: *Net, url: []const u8, body: []const u8) void {
        const now = nowMs(net);
        const url_dup = self.gpa.dupe(u8, url) catch return;
        const body_dup = self.gpa.dupe(u8, body) catch {
            self.gpa.free(url_dup);
            return;
        };
        self.cache_lock.lock();
        defer self.cache_lock.unlock();
        // Evict expired entries first.
        var i: usize = 0;
        while (i < self.http_cache.items.len) {
            const e = self.http_cache.items[i];
            if (std.mem.eql(u8, e.url, url) or now - e.at_ms > cache_ttl_ms) {
                self.gpa.free(e.url);
                self.gpa.free(e.body);
                _ = self.http_cache.swapRemove(i);
            } else i += 1;
        }
        self.http_cache.append(self.gpa, .{ .url = url_dup, .body = body_dup, .at_ms = now }) catch {
            self.gpa.free(url_dup);
            self.gpa.free(body_dup);
        };
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

/// Parse an ISO-8601 timestamp ("2024-01-15T12:34:56.000Z") into a Unix epoch in
/// seconds. Returns null if it's too short or malformed. Only the date and
/// hh:mm:ss are read (sub-seconds and zone suffix are ignored — HF stamps are UTC).
fn parseIso8601(s: []const u8) ?i64 {
    if (s.len < 19 or s[4] != '-' or s[7] != '-' or s[13] != ':' or s[16] != ':') return null;
    const y = std.fmt.parseInt(i64, s[0..4], 10) catch return null;
    const mo = std.fmt.parseInt(i64, s[5..7], 10) catch return null;
    const d = std.fmt.parseInt(i64, s[8..10], 10) catch return null;
    const h = std.fmt.parseInt(i64, s[11..13], 10) catch return null;
    const mi = std.fmt.parseInt(i64, s[14..16], 10) catch return null;
    const se = std.fmt.parseInt(i64, s[17..19], 10) catch return null;
    return daysFromCivil(y, mo, d) * 86400 + h * 3600 + mi * 60 + se;
}

/// Days since the Unix epoch for a civil (proleptic Gregorian) date. Howard
/// Hinnant's algorithm; valid for the modern dates HF returns (years > 0).
fn daysFromCivil(year: i64, m: i64, d: i64) i64 {
    const y = year - @as(i64, if (m <= 2) 1 else 0);
    const era = @divFloor(y, 400);
    const yoe = y - era * 400; // [0, 399]
    const doy = @divFloor(153 * (if (m > 2) m - 3 else m + 9) + 2, 5) + d - 1; // [0, 365]
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy; // [0, 146096]
    return era * 146097 + doe - 719468;
}

test "parseIso8601: epoch round-trip" {
    // 2024-01-15T00:00:00Z = 1705276800
    try std.testing.expectEqual(@as(?i64, 1705276800), parseIso8601("2024-01-15T00:00:00.000Z"));
    // The Unix epoch itself.
    try std.testing.expectEqual(@as(?i64, 0), parseIso8601("1970-01-01T00:00:00Z"));
    try std.testing.expectEqual(@as(?i64, null), parseIso8601("not-a-date"));
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
