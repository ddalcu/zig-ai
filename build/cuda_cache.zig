//! Fingerprint for the prebuilt ggml CUDA backend module.
//!
//! `ggml-cuda.dll` / `libggml-cuda.so` is a GGML_BACKEND_DL module: nothing
//! links against it, ggml `dlopen`s it from the executable's directory at
//! runtime and rejects it unless `reg->api_version == GGML_BACKEND_API_VERSION`
//! (see deps/llama.cpp/ggml/src/ggml-backend-reg.cpp). That loose coupling is
//! what lets us compile it once and reuse the binary instead of paying ~180 .cu
//! files x N architectures on every build.
//!
//! The coupling is loose, not absent, so reuse needs a guard. A stale module
//! fails in two very different ways:
//!
//!   * Loudly — ggml's exports or api_version moved, so the load fails and ggml
//!     falls back to Vulkan/CPU. Recoverable, merely slow.
//!   * SILENTLY — GGML_MAX_NAME changed. It sizes `ggml_tensor`'s embedded name
//!     buffer, so every field after it shifts; a mismatched module reads garbage
//!     off correctly-typed pointers. Nothing fails, the math is just wrong.
//!
//! The second case is why this is a content hash rather than a git rev: the
//! submodules are routinely patched in place (deps/patches/, local edits), and
//! a rev alone would call a modified tree clean.
//!
//! Tests: `zig build test` (or `zig test build/cuda_cache.zig`).

const std = @import("std");

/// Everything that can invalidate a prebuilt CUDA module. Rendered to a text
/// file stored beside the binary and compared on the next build.
pub const Key = struct {
    /// Zig target triple. Also covers OS-dependent source differences (CRLF
    /// checkouts, path separators), so the tree hash need not normalize them.
    target: []const u8,
    /// Hex digest of the ggml source tree (see TreeHasher).
    ggml_hash: []const u8,
    /// GGML_MAX_NAME. The silent-corruption field — see the note above.
    max_name: u32,
    /// CMAKE_CUDA_ARCHITECTURES. A module built for `120` has no SASS for a
    /// Turing card, so this is part of the identity, not just the cost.
    arch: []const u8,
    /// nvcc release that produced the module, e.g. "13.2". Recorded as a
    /// COMMENT, deliberately outside the identity — see `render`.
    toolkit: []const u8 = "unknown",
};

/// Canonical `key=value` rendering. Stable field order so byte comparison is
/// meaningful; a leading comment makes a stale file self-explanatory when a
/// human finds it in a cache dir.
///
/// `toolkit` is written as a comment, so it is provenance rather than identity.
/// Two reasons. It is not an ABI key: a module built by a different nvcc is
/// still a valid ggml backend, and the cudart/cublas it imports travel with it
/// in the same bundle, so they can never disagree. And making it identity would
/// mean every consumer had to install the ~4 GB CUDA Toolkit merely to compute
/// the current key and discover it already had a valid module — which is most of
/// what reusing the module was meant to avoid.
pub fn render(gpa: std.mem.Allocator, key: Key) ![]u8 {
    return std.fmt.allocPrint(gpa,
        \\# zig-ai prebuilt ggml CUDA module fingerprint — do not edit
        \\# built with CUDA Toolkit {s}
        \\target={s}
        \\ggml={s}
        \\max_name={d}
        \\arch={s}
        \\
    , .{ key.toolkit, key.target, key.ggml_hash, key.max_name, key.arch });
}

/// True when two rendered fingerprints describe the same module.
///
/// Deliberately not `mem.eql`: these round-trip through git checkouts, zip
/// archives and PowerShell redirection, any of which can flip LF to CRLF or add
/// a trailing newline. Comments and blank lines are ignored so the header can
/// change without invalidating every cached module.
pub fn matches(stored: []const u8, current: []const u8) bool {
    var a = lines(stored);
    var b = lines(current);
    while (true) {
        const la = a.next();
        const lb = b.next();
        if (la == null and lb == null) return true;
        if (la == null or lb == null) return false;
        if (!std.mem.eql(u8, la.?, lb.?)) return false;
    }
}

/// Iterator over significant lines: CR stripped, trailing space trimmed, blank
/// and `#` comment lines skipped.
fn lines(text: []const u8) LineIter {
    return .{ .it = std.mem.splitScalar(u8, text, '\n') };
}

const LineIter = struct {
    it: std.mem.SplitIterator(u8, .scalar),

    fn next(self: *LineIter) ?[]const u8 {
        while (self.it.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0 or line[0] == '#') continue;
            return line;
        }
        return null;
    }
};

/// Content hash of a set of files.
///
/// Feed files in sorted order (`sortPaths`) so the digest is independent of
/// directory-walk order, which differs between filesystems.
pub const TreeHasher = struct {
    h: std.crypto.hash.sha2.Sha256,

    pub fn init() TreeHasher {
        return .{ .h = std.crypto.hash.sha2.Sha256.init(.{}) };
    }

    /// Length-prefixed framing on both the path and the contents. Without it,
    /// ("ab", "c") and ("a", "bc") hash identically — a real collision for a
    /// tree where a rename shifts bytes across the path/content boundary.
    pub fn addFile(self: *TreeHasher, rel_path: []const u8, contents: []const u8) void {
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, rel_path.len, .little);
        self.h.update(&len);
        self.h.update(rel_path);
        std.mem.writeInt(u64, &len, contents.len, .little);
        self.h.update(&len);
        self.h.update(contents);
    }

    /// Lowercase hex digest.
    pub fn final(self: *TreeHasher) [64]u8 {
        var digest: [32]u8 = undefined;
        self.h.final(&digest);
        return std.fmt.bytesToHex(digest, .lower);
    }
};

/// Rewrite `\` to `/` in place so a Windows walk and a POSIX walk produce the
/// same relative paths. Returns the same slice for call-site convenience.
pub fn normalizePath(path: []u8) []u8 {
    std.mem.replaceScalar(u8, path, '\\', '/');
    return path;
}

/// Sort comparator for relative paths — pass to `std.mem.sort`.
pub fn pathLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// A directory to fold into the tree hash.
pub const Root = struct {
    /// Path relative to the base directory passed to `hashSources`.
    path: []const u8,
    /// `false` hashes only the directory's own files. ggml/src is listed
    /// non-recursively on purpose: its top-level headers are the ABI the CUDA
    /// module is compiled against, while its subdirectories are the *other*
    /// backends (ggml-cpu, ggml-metal, ggml-vulkan). Those are independently
    /// loaded modules that cannot affect ggml-cuda, and folding them in would
    /// invalidate the cache on every unrelated Metal or Vulkan change.
    recursive: bool = true,
};

/// Files whose contents decide whether a prebuilt CUDA module is still valid.
/// Paths are relative to the repository root.
pub const ggml_cuda_roots = [_]Root{
    // The public ggml headers and the top-level ABI/registry sources.
    .{ .path = "deps/llama.cpp/ggml/include" },
    .{ .path = "deps/llama.cpp/ggml/src", .recursive = false },
    // The CUDA backend itself — ~180 .cu/.cuh files.
    .{ .path = "deps/llama.cpp/ggml/src/ggml-cuda" },
    // Build wiring: compile flags and definitions change the emitted code even
    // when no source does.
    .{ .path = "deps/llama.cpp/ggml/cmake" },
};

/// Hash every file under `roots`, in a walk-order- and OS-independent way.
///
/// A missing root is an error rather than a skip: if upstream renames a
/// directory, a silently-shortened hash would keep matching a stale module,
/// which is the exact failure this whole module exists to prevent.
pub fn hashSources(
    gpa: std.mem.Allocator,
    io: std.Io,
    base: std.Io.Dir,
    roots: []const Root,
) ![64]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var paths: std.ArrayList([]const u8) = .empty;
    for (roots) |root| {
        var dir = try base.openDir(io, root.path, .{ .iterate = true });
        defer dir.close(io);

        if (root.recursive) {
            var walker = try dir.walk(arena);
            while (try walker.next(io)) |entry| {
                if (entry.kind != .file) continue;
                const joined = try std.fs.path.join(arena, &.{ root.path, entry.path });
                try paths.append(arena, normalizePath(joined));
            }
        } else {
            var it = dir.iterate();
            while (try it.next(io)) |entry| {
                if (entry.kind != .file) continue;
                const joined = try std.fs.path.join(arena, &.{ root.path, entry.name });
                try paths.append(arena, normalizePath(joined));
            }
        }
    }

    std.mem.sort([]const u8, paths.items, {}, pathLessThan);

    var hasher = TreeHasher.init();
    for (paths.items) |path| {
        const contents = try base.readFileAlloc(io, path, arena, .limited(64 << 20));
        hasher.addFile(path, contents);
    }
    return hasher.final();
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

const sample: Key = .{
    .target = "x86_64-windows-gnu",
    .ggml_hash = "abc123",
    .max_name = 160,
    .arch = "120-real",
    .toolkit = "13.2",
};

test "render is deterministic and includes every field" {
    const gpa = testing.allocator;
    const a = try render(gpa, sample);
    defer gpa.free(a);
    const b = try render(gpa, sample);
    defer gpa.free(b);
    try testing.expectEqualStrings(a, b);

    try testing.expect(std.mem.indexOf(u8, a, "x86_64-windows-gnu") != null);
    try testing.expect(std.mem.indexOf(u8, a, "abc123") != null);
    try testing.expect(std.mem.indexOf(u8, a, "160") != null);
    try testing.expect(std.mem.indexOf(u8, a, "120-real") != null);
    try testing.expect(std.mem.indexOf(u8, a, "13.2") != null);
}

test "every field change invalidates the fingerprint" {
    const gpa = testing.allocator;
    const base = try render(gpa, sample);
    defer gpa.free(base);

    var variants: [4]Key = .{sample} ** 4;
    variants[0].target = "x86_64-linux-gnu";
    variants[1].ggml_hash = "def456";
    variants[2].max_name = 64; // the silent-corruption case
    variants[3].arch = "75-real;120-real";

    for (variants) |v| {
        const other = try render(gpa, v);
        defer gpa.free(other);
        try testing.expect(!matches(base, other));
    }
}

test "toolkit version is provenance, not identity" {
    const gpa = testing.allocator;
    const base = try render(gpa, sample);
    defer gpa.free(base);

    var newer = sample;
    newer.toolkit = "12.6";
    const other = try render(gpa, newer);
    defer gpa.free(other);

    // Still a hit: a module built by a different nvcc is a valid ggml backend,
    // and validating it must not require the toolkit to be installed.
    try testing.expect(matches(base, other));
    // But the value is still recorded, so a bad module can be traced.
    try testing.expect(std.mem.indexOf(u8, other, "12.6") != null);
}

test "matches survives CRLF, trailing newlines and comment churn" {
    const gpa = testing.allocator;
    const current = try render(gpa, sample);
    defer gpa.free(current);

    // A git checkout with core.autocrlf, plus an extra blank line.
    const crlf = try std.mem.replaceOwned(u8, gpa, current, "\n", "\r\n");
    defer gpa.free(crlf);
    const padded = try std.fmt.allocPrint(gpa, "{s}\r\n", .{crlf});
    defer gpa.free(padded);
    try testing.expect(matches(padded, current));

    // The header comment is documentation, not identity.
    const recommented = try std.fmt.allocPrint(gpa,
        \\# a completely different header
        \\target={s}
        \\ggml={s}
        \\max_name={d}
        \\arch={s}
    , .{ sample.target, sample.ggml_hash, sample.max_name, sample.arch });
    defer gpa.free(recommented);
    try testing.expect(matches(recommented, current));
}

test "matches rejects a truncated fingerprint" {
    const gpa = testing.allocator;
    const current = try render(gpa, sample);
    defer gpa.free(current);
    // A half-written file (interrupted build) must never count as a hit.
    try testing.expect(!matches(current[0 .. current.len / 2], current));
    try testing.expect(!matches("", current));
}

test "TreeHasher detects content, path and file-count changes" {
    var base = TreeHasher.init();
    base.addFile("ggml-cuda/mmq.cu", "kernel");
    base.addFile("ggml-cuda/vecdotq.cuh", "header");
    const want = base.final();

    var same = TreeHasher.init();
    same.addFile("ggml-cuda/mmq.cu", "kernel");
    same.addFile("ggml-cuda/vecdotq.cuh", "header");
    try testing.expectEqualStrings(&want, &same.final());

    var edited = TreeHasher.init();
    edited.addFile("ggml-cuda/mmq.cu", "kernel!");
    edited.addFile("ggml-cuda/vecdotq.cuh", "header");
    try testing.expect(!std.mem.eql(u8, &want, &edited.final()));

    var renamed = TreeHasher.init();
    renamed.addFile("ggml-cuda/mmq2.cu", "kernel");
    renamed.addFile("ggml-cuda/vecdotq.cuh", "header");
    try testing.expect(!std.mem.eql(u8, &want, &renamed.final()));

    var added = TreeHasher.init();
    added.addFile("ggml-cuda/mmq.cu", "kernel");
    added.addFile("ggml-cuda/vecdotq.cuh", "header");
    added.addFile("ggml-cuda/new.cu", "");
    try testing.expect(!std.mem.eql(u8, &want, &added.final()));
}

test "TreeHasher framing prevents path/content boundary collisions" {
    // Without length prefixes both of these hash "ab" ++ "c".
    var one = TreeHasher.init();
    one.addFile("ab", "c");
    var two = TreeHasher.init();
    two.addFile("a", "bc");
    try testing.expect(!std.mem.eql(u8, &one.final(), &two.final()));
}

test "normalizePath makes a Windows walk match a POSIX walk" {
    var win = "ggml-cuda\\template-instances\\fattn.cu".*;
    try testing.expectEqualStrings(
        "ggml-cuda/template-instances/fattn.cu",
        normalizePath(&win),
    );
}

/// Build a small stand-in for the ggml tree: a root with a top-level header, a
/// nested "other backend" directory, and a nested CUDA-ish directory.
fn writeSampleTree(tmp: *std.testing.TmpDir) !void {
    const io = std.testing.io;
    try tmp.dir.createDirPath(io, "src/ggml-metal");
    try tmp.dir.createDirPath(io, "src/ggml-cuda/template-instances");
    try tmp.dir.writeFile(io, .{ .sub_path = "src/ggml-impl.h", .data = "abi" });
    try tmp.dir.writeFile(io, .{ .sub_path = "src/ggml-metal/metal.m", .data = "apple" });
    try tmp.dir.writeFile(io, .{ .sub_path = "src/ggml-cuda/mmq.cu", .data = "kernel" });
    try tmp.dir.writeFile(io, .{
        .sub_path = "src/ggml-cuda/template-instances/fattn.cu",
        .data = "instantiation",
    });
}

test "hashSources is stable and reacts to edits under a recursive root" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeSampleTree(&tmp);

    const roots = [_]Root{
        .{ .path = "src", .recursive = false },
        .{ .path = "src/ggml-cuda" },
    };

    const first = try hashSources(gpa, io, tmp.dir, &roots);
    const second = try hashSources(gpa, io, tmp.dir, &roots);
    try testing.expectEqualStrings(&first, &second);

    // A nested .cu edit must invalidate — this is the common case (submodule bump).
    try tmp.dir.writeFile(io, .{
        .sub_path = "src/ggml-cuda/template-instances/fattn.cu",
        .data = "instantiation v2",
    });
    try testing.expect(!std.mem.eql(u8, &first, &try hashSources(gpa, io, tmp.dir, &roots)));
}

test "hashSources ignores other backends but not the shared ABI header" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeSampleTree(&tmp);

    const roots = [_]Root{
        .{ .path = "src", .recursive = false },
        .{ .path = "src/ggml-cuda" },
    };
    const base = try hashSources(gpa, io, tmp.dir, &roots);

    // A Metal-only change must NOT invalidate a CUDA module — otherwise every
    // unrelated backend commit costs a full multi-hour rebuild.
    try tmp.dir.writeFile(io, .{ .sub_path = "src/ggml-metal/metal.m", .data = "apple v2" });
    try testing.expectEqualStrings(&base, &try hashSources(gpa, io, tmp.dir, &roots));

    // The shared ABI header must invalidate: it is what ggml-cuda compiles against.
    try tmp.dir.writeFile(io, .{ .sub_path = "src/ggml-impl.h", .data = "abi v2" });
    try testing.expect(!std.mem.eql(u8, &base, &try hashSources(gpa, io, tmp.dir, &roots)));
}

test "hashSources fails loudly when a root disappears" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeSampleTree(&tmp);

    // An upstream rename must break the build, not silently keep matching.
    try testing.expectError(error.FileNotFound, hashSources(gpa, io, tmp.dir, &.{
        .{ .path = "src/ggml-hip" },
    }));
}

test "sorted paths make the digest walk-order independent" {
    var forward = [_][]const u8{ "b.cu", "a.cu", "c.cuh" };
    var reverse = [_][]const u8{ "c.cuh", "b.cu", "a.cu" };
    std.mem.sort([]const u8, &forward, {}, pathLessThan);
    std.mem.sort([]const u8, &reverse, {}, pathLessThan);

    var h1 = TreeHasher.init();
    for (forward) |p| h1.addFile(p, p);
    var h2 = TreeHasher.init();
    for (reverse) |p| h2.addFile(p, p);
    try testing.expectEqualStrings(&h1.final(), &h2.final());
}
