//! Model Context Protocol (MCP) support: the preset server catalog, the on-disk
//! `mcp.json` registry (parse/serialize/edit), and a runtime `Manager` that
//! spawns the enabled servers, speaks JSON-RPC 2.0 over their stdio pipes, and
//! exposes their tools to the agent loop.
//!
//! Cross-platform per the project rule: servers are spawned with the portable
//! `std.process.spawn` (pipes for stdin/stdout), and all blocking pipe I/O and
//! idle waits go through `std.Io` (a per-thread `Io.Threaded`). No popen, no
//! pthread. One background worker thread owns every server and serializes the
//! handshake / tools-list / tools-call round-trips — the agent only ever has one
//! tool call in flight, so a single worker is both correct and simple.

const std = @import("std");
const Io = std.Io;
const channel = @import("channel.zig");
const config = @import("config.zig");
const builtins = @import("builtin.zig");

// ===========================================================================
// Preset catalog
// ===========================================================================

/// Where a user-supplied value goes when a preset is configured.
pub const InputKind = enum {
    /// Set as an environment variable named `key` (e.g. an API token).
    env,
    /// Substitute for the placeholder token `key` in the preset's args
    /// (e.g. `<PATH>` → the folder the user chose).
    arg,
};

/// A value the user fills in when adding a preset. Collected by a small inline
/// form in the MCP screen, then written into `mcp.json`.
pub const PresetInput = struct {
    key: []const u8,
    label: []const u8,
    kind: InputKind = .env,
    secret: bool = false,
    /// Placeholder/help text shown in the form field.
    hint: []const u8 = "",
};

pub const Preset = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    command: []const u8,
    args: []const []const u8,
    inputs: []const PresetInput = &.{},
    /// A short caveat shown in the marketplace (e.g. "requires Docker running").
    note: []const u8 = "",

    /// Whether adding this preset needs the user to fill in values first.
    pub fn needsConfig(p: Preset) bool {
        return p.inputs.len > 0;
    }
};

/// The curated preset servers, mirroring the mlx-serve catalog. Adding one writes
/// its command/args/env skeleton into `mcp.json`; the user then fills any tokens
/// (placeholders like `<PATH>` or empty env values) via the text editor.
pub const catalog = [_]Preset{
    .{
        .id = "filesystem",
        .name = "Filesystem",
        .description = "Read, write and list files under a folder you choose.",
        .command = "npx",
        .args = &.{ "-y", "@modelcontextprotocol/server-filesystem", "<PATH>" },
        .inputs = &.{.{ .key = "<PATH>", .label = "Folder path", .kind = .arg, .hint = "/Users/you/project" }},
    },
    .{
        .id = "github",
        .name = "GitHub",
        .description = "Repositories, issues, pull requests and code search.",
        .command = "npx",
        .args = &.{ "-y", "@modelcontextprotocol/server-github" },
        .inputs = &.{.{ .key = "GITHUB_PERSONAL_ACCESS_TOKEN", .label = "Personal access token", .secret = true, .hint = "ghp_…" }},
    },
    .{
        .id = "playwright",
        .name = "Playwright",
        .description = "Drive a browser: navigate, click, type, screenshot.",
        .command = "npx",
        .args = &.{ "-y", "@playwright/mcp@latest" },
        .note = "First run downloads a browser (~300 MB).",
    },
    .{
        .id = "shell",
        .name = "Shell",
        .description = "Run arbitrary shell commands on this machine.",
        .command = "npx",
        .args = &.{ "-y", "@mkusaka/mcp-shell-server" },
        .note = "Powerful — the agent can run any command. Use with care.",
    },
    .{
        .id = "dbhub",
        .name = "DBHub",
        .description = "Query Postgres, MySQL, SQLite or SQL Server.",
        .command = "npx",
        .args = &.{ "-y", "@bytebase/dbhub@latest", "--transport", "stdio", "--dsn", "<DSN>" },
        .inputs = &.{.{ .key = "<DSN>", .label = "Database DSN", .kind = .arg, .secret = true, .hint = "postgres://user:pass@host/db" }},
    },
    .{
        .id = "slack",
        .name = "Slack",
        .description = "Read channels, post messages and search.",
        .command = "npx",
        .args = &.{ "-y", "@zencoderai/slack-mcp-server" },
        .inputs = &.{
            .{ .key = "SLACK_BOT_TOKEN", .label = "Bot token", .secret = true, .hint = "xoxb-…" },
            .{ .key = "SLACK_TEAM_ID", .label = "Team ID", .hint = "T01234567" },
        },
    },
    .{
        .id = "notion",
        .name = "Notion",
        .description = "Read and write Notion pages and databases.",
        .command = "npx",
        .args = &.{ "-y", "@notionhq/notion-mcp-server" },
        .inputs = &.{.{ .key = "NOTION_TOKEN", .label = "Integration token", .secret = true, .hint = "ntn_…" }},
    },
    .{
        .id = "azure-devops",
        .name = "Azure DevOps",
        .description = "Work items, pull requests, builds, wikis and test plans.",
        .command = "npx",
        .args = &.{ "-y", "@azure-devops/mcp", "<ORG>" },
        .inputs = &.{.{ .key = "<ORG>", .label = "Organization", .kind = .arg, .hint = "my-org" }},
    },
    .{
        .id = "docker",
        .name = "Docker",
        .description = "Manage containers, images and networks.",
        .command = "npx",
        .args = &.{ "-y", "docker-mcp" },
        .note = "Requires the Docker daemon to be running.",
    },
    .{
        .id = "kubernetes",
        .name = "Kubernetes",
        .description = "Inspect and manage pods, deployments and services.",
        .command = "npx",
        .args = &.{ "-y", "mcp-server-kubernetes" },
        .note = "Uses your ~/.kube/config or KUBECONFIG.",
    },
};

pub fn presetById(id: []const u8) ?Preset {
    for (catalog) |p| if (std.mem.eql(u8, p.id, id)) return p;
    return null;
}

pub fn presetIndexById(id: []const u8) ?usize {
    for (catalog, 0..) |p, i| if (std.mem.eql(u8, p.id, id)) return i;
    return null;
}

// ===========================================================================
// mcp.json registry (parse / serialize / edit)
// ===========================================================================

pub const EnvPair = struct { key: []u8, value: []u8 };

/// One server entry from `mcp.json`. All slices are owned (freed by `deinit`).
pub const ServerConfig = struct {
    name: []u8,
    command: ?[]u8 = null,
    args: [][]u8 = &.{},
    env: []EnvPair = &.{},
    url: ?[]u8 = null,
    disabled: bool = false,

    pub fn deinit(self: *ServerConfig, gpa: std.mem.Allocator) void {
        gpa.free(self.name);
        if (self.command) |c| gpa.free(c);
        for (self.args) |a| gpa.free(a);
        gpa.free(self.args);
        for (self.env) |e| {
            gpa.free(e.key);
            gpa.free(e.value);
        }
        gpa.free(self.env);
        if (self.url) |u| gpa.free(u);
    }
};

pub const Registry = struct {
    gpa: std.mem.Allocator,
    servers: std.ArrayList(ServerConfig) = .empty,

    pub fn deinit(self: *Registry) void {
        for (self.servers.items) |*s| s.deinit(self.gpa);
        self.servers.deinit(self.gpa);
    }
};

/// Parse `mcp.json` bytes into an owned `Registry`. Preserves entry order (the
/// std JSON object map is insertion-ordered). Unknown/extra fields are ignored.
/// Returns an empty registry on any parse error so the app degrades gracefully.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Registry {
    var reg: Registry = .{ .gpa = gpa };
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch return reg;
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return reg;
    const servers_v = root.object.get("mcpServers") orelse return reg;
    if (servers_v != .object) return reg;

    var it = servers_v.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const v = entry.value_ptr.*;
        if (v != .object) continue;
        const o = v.object;

        var sc: ServerConfig = .{ .name = gpa.dupe(u8, name) catch continue };
        if (o.get("command")) |c| {
            if (c == .string) sc.command = gpa.dupe(u8, c.string) catch null;
        }
        if (o.get("url")) |u| {
            if (u == .string) sc.url = gpa.dupe(u8, u.string) catch null;
        }
        if (o.get("disabled")) |d| {
            if (d == .bool) sc.disabled = d.bool;
        }
        if (o.get("args")) |a| {
            if (a == .array) {
                var list: std.ArrayList([]u8) = .empty;
                for (a.array.items) |item| {
                    if (item == .string) {
                        const dup = gpa.dupe(u8, item.string) catch continue;
                        list.append(gpa, dup) catch gpa.free(dup);
                    }
                }
                sc.args = list.toOwnedSlice(gpa) catch &.{};
            }
        }
        if (o.get("env")) |e| {
            if (e == .object) {
                var list: std.ArrayList(EnvPair) = .empty;
                var eit = e.object.iterator();
                while (eit.next()) |ee| {
                    if (ee.value_ptr.* != .string) continue;
                    const k = gpa.dupe(u8, ee.key_ptr.*) catch continue;
                    const val = gpa.dupe(u8, ee.value_ptr.*.string) catch {
                        gpa.free(k);
                        continue;
                    };
                    list.append(gpa, .{ .key = k, .value = val }) catch {
                        gpa.free(k);
                        gpa.free(val);
                    };
                }
                sc.env = list.toOwnedSlice(gpa) catch &.{};
            }
        }
        reg.servers.append(gpa, sc) catch sc.deinit(gpa);
    }
    return reg;
}

/// Serialize a registry back to pretty JSON (caller owns the bytes).
pub fn serialize(gpa: std.mem.Allocator, servers: []const ServerConfig) ![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var servers_obj = try std.json.ObjectMap.init(a, &.{}, &.{});
    for (servers) |s| {
        var o = try std.json.ObjectMap.init(a, &.{}, &.{});
        if (s.command) |c| try o.put(a, "command", .{ .string = c });
        if (s.url) |u| try o.put(a, "url", .{ .string = u });
        if (s.args.len > 0) {
            var arr = std.json.Array.init(a);
            for (s.args) |arg| try arr.append(.{ .string = arg });
            try o.put(a, "args", .{ .array = arr });
        }
        if (s.env.len > 0) {
            var env_obj = try std.json.ObjectMap.init(a, &.{}, &.{});
            for (s.env) |e| try env_obj.put(a, e.key, .{ .string = e.value });
            try o.put(a, "env", .{ .object = env_obj });
        }
        if (s.disabled) try o.put(a, "disabled", .{ .bool = true });
        try servers_obj.put(a, s.name, .{ .object = o });
    }
    var root = try std.json.ObjectMap.init(a, &.{}, &.{});
    try root.put(a, "mcpServers", .{ .object = servers_obj });

    return std.json.Stringify.valueAlloc(gpa, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });
}

/// Read the registry from disk (empty if the file is missing).
pub fn loadRegistry(gpa: std.mem.Allocator, home: []const u8) Registry {
    if (config.read(gpa, home, config.mcp_file)) |bytes| {
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
    return .{ .gpa = gpa };
}

/// Write a registry to disk.
pub fn saveRegistry(gpa: std.mem.Allocator, home: []const u8, servers: []const ServerConfig) bool {
    const bytes = serialize(gpa, servers) catch return false;
    defer gpa.free(bytes);
    return config.write(gpa, home, config.mcp_file, bytes);
}

/// Build a `ServerConfig` from a preset + the user-supplied `values` (one per
/// entry in `p.inputs`, in order; missing/empty values leave the placeholder).
/// `env`-kind inputs become env vars; `arg`-kind inputs substitute their
/// placeholder token in the args.
fn buildServerConfig(gpa: std.mem.Allocator, p: Preset, values: []const []const u8) !ServerConfig {
    var sc: ServerConfig = .{ .name = try gpa.dupe(u8, p.id) };
    sc.command = try gpa.dupe(u8, p.command);

    var args: std.ArrayList([]u8) = .empty;
    for (p.args) |arg| {
        var replacement: ?[]const u8 = null;
        for (p.inputs, 0..) |inp, i| {
            if (inp.kind == .arg and std.mem.eql(u8, inp.key, arg) and i < values.len and values[i].len > 0)
                replacement = values[i];
        }
        const d = try gpa.dupe(u8, replacement orelse arg);
        try args.append(gpa, d);
    }
    sc.args = try args.toOwnedSlice(gpa);

    var env: std.ArrayList(EnvPair) = .empty;
    for (p.inputs, 0..) |inp, i| {
        if (inp.kind != .env) continue;
        const val = if (i < values.len) values[i] else "";
        try env.append(gpa, .{ .key = try gpa.dupe(u8, inp.key), .value = try gpa.dupe(u8, val) });
    }
    sc.env = try env.toOwnedSlice(gpa);
    return sc;
}

/// Append a preset to `mcp.json`, filling its inputs from `values` (one per
/// `p.inputs` entry). No-op if a server with that id already exists.
pub fn addPresetWithValues(gpa: std.mem.Allocator, home: []const u8, p: Preset, values: []const []const u8) bool {
    var reg = loadRegistry(gpa, home);
    defer reg.deinit();
    for (reg.servers.items) |s| if (std.mem.eql(u8, s.name, p.id)) return true;

    var sc = buildServerConfig(gpa, p, values) catch return false;
    reg.servers.append(gpa, sc) catch {
        sc.deinit(gpa);
        return false;
    };
    return saveRegistry(gpa, home, reg.servers.items);
}

/// Append a preset with no configured values (placeholders/empty env kept).
pub fn addPreset(gpa: std.mem.Allocator, home: []const u8, p: Preset) bool {
    return addPresetWithValues(gpa, home, p, &.{});
}

/// The inverse of `buildServerConfig`: read the current value of each of
/// `p.inputs` out of a configured entry, for pre-filling the edit form.
/// env-kind inputs read their env var; arg-kind inputs read the arg at the
/// placeholder's position in the preset's args template ("" while it still
/// holds the unconfigured placeholder). `out` slices alias `sc` — copy them
/// before `sc` is freed.
pub fn currentValues(p: Preset, sc: ServerConfig, out: [][]const u8) void {
    for (p.inputs, 0..) |inp, i| {
        if (i >= out.len) break;
        out[i] = "";
        switch (inp.kind) {
            .env => for (sc.env) |e| {
                if (std.mem.eql(u8, e.key, inp.key)) out[i] = e.value;
            },
            .arg => for (p.args, 0..) |template_arg, j| {
                if (!std.mem.eql(u8, template_arg, inp.key)) continue;
                if (j < sc.args.len and !std.mem.eql(u8, sc.args[j], inp.key))
                    out[i] = sc.args[j];
            },
        }
    }
}

/// Rewrite an existing preset-derived entry with new input values (same shape
/// as adding fresh), preserving its disabled flag. False when no entry with
/// the preset's id exists.
pub fn updatePresetValues(gpa: std.mem.Allocator, home: []const u8, p: Preset, values: []const []const u8) bool {
    var reg = loadRegistry(gpa, home);
    defer reg.deinit();
    for (reg.servers.items) |*s| {
        if (!std.mem.eql(u8, s.name, p.id)) continue;
        var sc = buildServerConfig(gpa, p, values) catch return false;
        sc.disabled = s.disabled;
        s.deinit(gpa);
        s.* = sc;
        return saveRegistry(gpa, home, reg.servers.items);
    }
    return false;
}

test "preset values: build → extract round-trips, update preserves disabled" {
    const gpa = std.testing.allocator;
    const p = presetById("filesystem").?; // one arg-kind input: <PATH>
    var sc = try buildServerConfig(gpa, p, &.{"/tmp/proj"});
    defer sc.deinit(gpa);
    var vals: [max_test_inputs][]const u8 = undefined;
    currentValues(p, sc, vals[0..p.inputs.len]);
    try std.testing.expectEqualStrings("/tmp/proj", vals[0]);

    // Unconfigured: the placeholder arg reads back as "".
    var sc2 = try buildServerConfig(gpa, p, &.{});
    defer sc2.deinit(gpa);
    currentValues(p, sc2, vals[0..p.inputs.len]);
    try std.testing.expectEqualStrings("", vals[0]);

    // env-kind inputs round-trip too.
    const g = presetById("github").?;
    var sc3 = try buildServerConfig(gpa, g, &.{"ghp_secret"});
    defer sc3.deinit(gpa);
    currentValues(g, sc3, vals[0..g.inputs.len]);
    try std.testing.expectEqualStrings("ghp_secret", vals[0]);
}

const max_test_inputs = 4;

/// Remove a server entry by name.
pub fn removeServer(gpa: std.mem.Allocator, home: []const u8, name: []const u8) bool {
    var reg = loadRegistry(gpa, home);
    defer reg.deinit();
    var i: usize = 0;
    while (i < reg.servers.items.len) : (i += 1) {
        if (std.mem.eql(u8, reg.servers.items[i].name, name)) {
            var sc = reg.servers.orderedRemove(i);
            sc.deinit(gpa);
            return saveRegistry(gpa, home, reg.servers.items);
        }
    }
    return true;
}

/// Toggle a server's `disabled` flag.
pub fn setDisabled(gpa: std.mem.Allocator, home: []const u8, name: []const u8, disabled: bool) bool {
    var reg = loadRegistry(gpa, home);
    defer reg.deinit();
    for (reg.servers.items) |*s| {
        if (std.mem.eql(u8, s.name, name)) {
            s.disabled = disabled;
            return saveRegistry(gpa, home, reg.servers.items);
        }
    }
    return true;
}

// ===========================================================================
// Runtime manager
// ===========================================================================

/// A discovered tool, qualified by its server so the agent addresses it
/// unambiguously as `server__tool`.
pub const Tool = struct {
    server: []u8,
    name: []u8,
    qualified: []u8,
    description: []u8,
    schema: []u8, // inputSchema as JSON text

    fn deinit(self: *Tool, gpa: std.mem.Allocator) void {
        gpa.free(self.server);
        gpa.free(self.name);
        gpa.free(self.qualified);
        gpa.free(self.description);
        gpa.free(self.schema);
    }
};

pub const RunState = enum { starting, running, failed };

/// A snapshot row for the MCP screen (owned copies, rebuilt by the worker).
pub const ServerStatus = struct {
    name: []u8,
    state: RunState,
    tools: usize,
    msg: ?[]u8,
};

pub const Event = union(enum) {
    /// Server set / tool list changed; the UI should refresh its snapshot.
    changed: void,
    /// A finished tool call: `text` is the result (owned; UI/agent frees).
    result: struct { seq: u64, ok: bool, text: []u8 },
    log: []u8,
};

const Job = union(enum) {
    reload: void,
    call: struct { seq: u64, qualified: []u8, args_json: []u8 },
    stop: void,
};

/// A live server process owned exclusively by the worker thread.
const Server = struct {
    name: []u8,
    child: std.process.Child,
    reader: *Io.File.Reader,
    read_buf: []u8,
    next_id: u64 = 1,
};

pub const Manager = struct {
    gpa: std.mem.Allocator,
    events: channel.Channel(Event),

    home: ?[]u8 = null,
    environ: ?std.process.Environ = null,

    thread: ?std.Thread = null,
    jobs_mu: channel.SpinLock = .{},
    jobs: std.ArrayList(Job) = .empty,
    shutdown: std.atomic.Value(bool) = .init(false),

    // Snapshot state read by the UI / agent (guarded by `snap_mu`).
    snap_mu: channel.SpinLock = .{},
    tools_snapshot: std.ArrayList(Tool) = .empty,
    status_snapshot: std.ArrayList(ServerStatus) = .empty,
    /// True while at least one server is up; lets the chat header show a dot.
    any_running: std.atomic.Value(bool) = .init(false),

    pub fn init(gpa: std.mem.Allocator) Manager {
        return .{ .gpa = gpa, .events = channel.Channel(Event).init(gpa) };
    }

    pub fn deinit(self: *Manager) void {
        if (self.thread) |th| {
            // The worker checks `shutdown` between jobs and on the `.stop` job, so
            // an idle runtime tears down immediately. The one slow case is quitting
            // while a server is still in its (blocking) initialize/tools-list
            // handshake — e.g. a first-run `npx -y` downloading the package — where
            // join waits for that read to finish (or the OS to reap the child).
            self.pushJob(.stop);
            self.shutdown.store(true, .release);
            th.join();
        }
        self.clearTools();
        self.tools_snapshot.deinit(self.gpa);
        self.clearStatus();
        self.status_snapshot.deinit(self.gpa);
        self.jobs.deinit(self.gpa);
        if (self.home) |h| self.gpa.free(h);
        self.events.deinit();
    }

    pub fn setContext(self: *Manager, home: []const u8, environ: std.process.Environ) void {
        if (self.home) |h| self.gpa.free(h);
        self.home = self.gpa.dupe(u8, home) catch null;
        self.environ = environ;
    }

    /// Start the worker and load the registry (idempotent).
    pub fn start(self: *Manager) !void {
        if (self.thread != null) {
            self.pushJob(.reload);
            return;
        }
        self.thread = try std.Thread.spawn(.{}, workerMain, .{self});
        self.pushJob(.reload);
    }

    /// Re-read mcp.json and (re)spawn enabled servers.
    pub fn reload(self: *Manager) void {
        self.pushJob(.reload);
    }

    /// Queue a tool call; the result arrives as an `Event.result` with `seq`.
    pub fn callAsync(self: *Manager, seq: u64, qualified: []const u8, args_json: []const u8) void {
        const q = self.gpa.dupe(u8, qualified) catch return;
        const a = self.gpa.dupe(u8, args_json) catch {
            self.gpa.free(q);
            return;
        };
        self.pushJob(.{ .call = .{ .seq = seq, .qualified = q, .args_json = a } });
    }

    pub fn isRunning(self: *const Manager) bool {
        return self.any_running.load(.acquire);
    }

    fn pushJob(self: *Manager, job: Job) void {
        self.jobs_mu.lock();
        defer self.jobs_mu.unlock();
        self.jobs.append(self.gpa, job) catch {};
    }

    fn takeJob(self: *Manager) ?Job {
        self.jobs_mu.lock();
        defer self.jobs_mu.unlock();
        if (self.jobs.items.len == 0) return null;
        return self.jobs.orderedRemove(0);
    }

    // --- snapshot helpers (UI/agent side) ---------------------------------

    fn clearTools(self: *Manager) void {
        for (self.tools_snapshot.items) |*t| t.deinit(self.gpa);
        self.tools_snapshot.clearRetainingCapacity();
    }
    fn clearStatus(self: *Manager) void {
        for (self.status_snapshot.items) |s| {
            self.gpa.free(s.name);
            if (s.msg) |m| self.gpa.free(m);
        }
        self.status_snapshot.clearRetainingCapacity();
    }

    /// Append, to `out`, the qualified name + description of every tool. Used to
    /// build the agent system prompt. Strings are duped into `arena`.
    pub fn toolListAlloc(self: *Manager, arena: std.mem.Allocator) []ToolInfo {
        self.snap_mu.lock();
        defer self.snap_mu.unlock();
        const n = builtins.specs.len + self.tools_snapshot.items.len;
        var list = arena.alloc(ToolInfo, n) catch return &.{};
        var i: usize = 0;
        // Built-in tools first — always available, no server required.
        for (builtins.specs) |s| {
            list[i] = .{ .qualified = s.name, .description = s.description, .schema = s.schema };
            i += 1;
        }
        for (self.tools_snapshot.items) |t| {
            list[i] = .{
                .qualified = arena.dupe(u8, t.qualified) catch "",
                .description = arena.dupe(u8, t.description) catch "",
                .schema = arena.dupe(u8, t.schema) catch "",
            };
            i += 1;
        }
        return list;
    }

    pub fn hasTool(self: *Manager, qualified: []const u8) bool {
        if (builtins.isBuiltin(qualified)) return true;
        self.snap_mu.lock();
        defer self.snap_mu.unlock();
        for (self.tools_snapshot.items) |t| {
            if (std.mem.eql(u8, t.qualified, qualified)) return true;
        }
        return false;
    }

    pub fn toolCount(self: *Manager) usize {
        self.snap_mu.lock();
        defer self.snap_mu.unlock();
        return builtins.specs.len + self.tools_snapshot.items.len;
    }

    /// Snapshot the server status rows into `arena` for the UI.
    pub fn statusAlloc(self: *Manager, arena: std.mem.Allocator) []StatusInfo {
        self.snap_mu.lock();
        defer self.snap_mu.unlock();
        var list = arena.alloc(StatusInfo, self.status_snapshot.items.len) catch return &.{};
        for (self.status_snapshot.items, 0..) |s, i| {
            list[i] = .{
                .name = arena.dupe(u8, s.name) catch "",
                .state = s.state,
                .tools = s.tools,
                .msg = if (s.msg) |m| arena.dupe(u8, m) catch null else null,
            };
        }
        return list;
    }

    pub const ToolInfo = struct { qualified: []const u8, description: []const u8, schema: []const u8 };
    pub const StatusInfo = struct { name: []const u8, state: RunState, tools: usize, msg: ?[]const u8 };

    fn log(self: *Manager, comptime f: []const u8, args: anytype) void {
        const msg = std.fmt.allocPrint(self.gpa, f, args) catch return;
        self.events.push(.{ .log = msg });
    }
};

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

const WorkerCtx = struct {
    mgr: *Manager,
    io: Io,
    gpa: std.mem.Allocator,
    servers: std.ArrayList(Server) = .empty,
};

fn workerMain(mgr: *Manager) void {
    // The environ matters: `Threaded` defaults to an EMPTY environment, which
    // makes `std.process.spawn` search Zig's fallback PATH (/usr/local/bin:
    // /bin:/usr/bin) for bare command names AND hands children an empty env.
    // `setContext` runs before `start`, so `mgr.environ` is set by now.
    var threaded = Io.Threaded.init(mgr.gpa, .{ .environ = mgr.environ orelse .empty });
    defer threaded.deinit();
    var ctx: WorkerCtx = .{ .mgr = mgr, .io = threaded.io(), .gpa = mgr.gpa };
    defer teardownAll(&ctx);

    while (!mgr.shutdown.load(.acquire)) {
        if (mgr.takeJob()) |job| {
            switch (job) {
                .reload => doReload(&ctx),
                .call => |c| {
                    doCall(&ctx, c.seq, c.qualified, c.args_json);
                    mgr.gpa.free(c.qualified);
                    mgr.gpa.free(c.args_json);
                },
                .stop => return,
            }
        } else {
            // Idle: nothing queued. Sleep briefly (keeps the thread cheap).
            Io.sleep(ctx.io, Io.Duration.fromMilliseconds(20), .awake) catch {};
        }
    }
}

fn teardownAll(ctx: *WorkerCtx) void {
    for (ctx.servers.items) |*s| killServer(ctx, s);
    ctx.servers.deinit(ctx.gpa);
}

fn killServer(ctx: *WorkerCtx, s: *Server) void {
    // `kill` closes stdin/stdout/stderr and reaps the process; the reader holds
    // a copy of the now-closed stdout fd but is never read again.
    s.child.kill(ctx.io);
    ctx.gpa.free(s.read_buf);
    ctx.gpa.destroy(s.reader);
    ctx.gpa.free(s.name);
}

/// Stop every server, re-read mcp.json, and spawn the enabled ones.
fn doReload(ctx: *WorkerCtx) void {
    const mgr = ctx.mgr;
    for (ctx.servers.items) |*s| killServer(ctx, s);
    ctx.servers.clearRetainingCapacity();

    // Reset the published snapshot.
    mgr.snap_mu.lock();
    mgr.clearTools();
    mgr.clearStatus();
    mgr.snap_mu.unlock();
    mgr.any_running.store(false, .release);

    const home = mgr.home orelse {
        mgr.events.push(.{ .changed = {} });
        return;
    };
    var reg = loadRegistry(mgr.gpa, home);
    defer reg.deinit();

    for (reg.servers.items) |sc| {
        if (sc.disabled) continue;
        if (sc.command == null) {
            publishStatus(ctx, sc.name, .failed, 0, "no command (http transport not supported)");
            continue;
        }
        publishStatus(ctx, sc.name, .starting, 0, null);
        spawnAndInit(ctx, sc) catch |e| {
            publishStatus(ctx, sc.name, .failed, 0, @errorName(e));
            mgr.log("mcp: {s} failed to start: {s}", .{ sc.name, @errorName(e) });
        };
    }

    mgr.any_running.store(ctx.servers.items.len > 0, .release);
    mgr.events.push(.{ .changed = {} });
}

const InitError = error{ SpawnFailed, HandshakeFailed, NoTools, OutOfMemory };

// --- PATH augmentation -------------------------------------------------------
// GUI-launched apps don't get the user's login-shell PATH (macOS Finder/Dock
// apps inherit launchd's minimal `/usr/bin:/bin:/usr/sbin:/sbin`; Linux desktop
// launchers can be similarly bare), so bare `npx` presets would fail outside a
// terminal. We append well-known per-user tool dirs that exist on disk.
// Windows is exempt: GUI apps inherit the registry user PATH there, and Zig's
// spawn already does PATHEXT (.cmd) resolution.

const native_os = @import("builtin").os.tag;

/// `PATH` from `environ` plus any well-known tool dirs (Homebrew, nvm, bun,
/// volta, ~/.local) that exist on disk and aren't already listed. Returns
/// gpa-owned text, or null when there is nothing to add (then the inherited
/// environment is already right).
fn augmentedPathAlloc(gpa: std.mem.Allocator, io: Io, environ: std.process.Environ, home: []const u8) ?[]u8 {
    if (native_os == .windows) return null;
    const base = std.process.Environ.getPosix(environ, "PATH") orelse "";

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    buf.appendSlice(gpa, base) catch return null;
    var added: usize = 0;

    const fixed = [_][]const u8{ "/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin" };
    for (fixed) |dir| appendPathDir(gpa, io, &buf, dir, &added);

    const home_subs = [_][]const u8{ ".local/bin", ".bun/bin", ".volta/bin", ".deno/bin" };
    for (home_subs) |sub| {
        const dir = std.fmt.allocPrint(gpa, "{s}/{s}", .{ home, sub }) catch continue;
        defer gpa.free(dir);
        appendPathDir(gpa, io, &buf, dir, &added);
    }

    // nvm doesn't symlink into any global bin dir — node/npx live only under
    // the per-version prefix, so pick the newest installed version.
    if (newestNvmBinAlloc(gpa, io, home)) |dir| {
        defer gpa.free(dir);
        appendPathDir(gpa, io, &buf, dir, &added);
    }

    if (added == 0) return null;
    return buf.toOwnedSlice(gpa) catch null;
}

/// Append `dir` to the ':'-separated list in `buf` if it isn't already a
/// segment and exists on disk.
fn appendPathDir(gpa: std.mem.Allocator, io: Io, buf: *std.ArrayList(u8), dir: []const u8, added: *usize) void {
    var it = std.mem.tokenizeScalar(u8, buf.items, ':');
    while (it.next()) |seg| if (std.mem.eql(u8, seg, dir)) return;
    Io.Dir.accessAbsolute(io, dir, .{}) catch return;
    if (buf.items.len > 0) buf.append(gpa, ':') catch return;
    buf.appendSlice(gpa, dir) catch return;
    added.* += 1;
}

/// `<home>/.nvm/versions/node/<newest>/bin`, or null when nvm isn't installed.
fn newestNvmBinAlloc(gpa: std.mem.Allocator, io: Io, home: []const u8) ?[]u8 {
    const root = std.fmt.allocPrint(gpa, "{s}/.nvm/versions/node", .{home}) catch return null;
    defer gpa.free(root);
    var dir = Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var best: [3]u32 = .{ 0, 0, 0 };
    var best_name_buf: [64]u8 = undefined;
    var best_name_len: usize = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if (e.kind != .directory) continue;
        if (e.name.len > best_name_buf.len) continue;
        const v = parseNodeVersion(e.name) orelse continue;
        if (best_name_len != 0 and !versionLess(best, v)) continue;
        best = v;
        @memcpy(best_name_buf[0..e.name.len], e.name);
        best_name_len = e.name.len;
    }
    if (best_name_len == 0) return null;
    return std.fmt.allocPrint(gpa, "{s}/{s}/bin", .{ root, best_name_buf[0..best_name_len] }) catch null;
}

/// Parse "v25.8.2" → {25,8,2} (missing parts are 0).
fn parseNodeVersion(name: []const u8) ?[3]u32 {
    if (name.len < 2 or name[0] != 'v') return null;
    var parts: [3]u32 = .{ 0, 0, 0 };
    var idx: usize = 0;
    var it = std.mem.splitScalar(u8, name[1..], '.');
    while (it.next()) |p| {
        if (idx >= 3) return null;
        parts[idx] = std.fmt.parseInt(u32, p, 10) catch return null;
        idx += 1;
    }
    if (idx == 0) return null;
    return parts;
}

fn versionLess(a: [3]u32, b: [3]u32) bool {
    for (a, b) |x, y| {
        if (x != y) return x < y;
    }
    return false;
}

/// Resolve a bare command name against the ':'-separated `path_text`.
/// `std.process.spawn` searches the PARENT's PATH (the env map we pass only
/// reaches the child), so to launch from an augmented PATH we must hand spawn
/// an absolute argv[0] ourselves. Returns gpa-owned path, or null when the
/// command is already a path / nothing matched (spawn then falls back to the
/// parent-PATH search as before).
fn resolveCommandAlloc(gpa: std.mem.Allocator, io: Io, command: []const u8, path_text: []const u8) ?[]u8 {
    if (std.mem.indexOfScalar(u8, command, '/') != null) return null;
    var it = std.mem.tokenizeScalar(u8, path_text, ':');
    while (it.next()) |dir| {
        if (!std.fs.path.isAbsolute(dir)) continue;
        const full = std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, command }) catch return null;
        Io.Dir.accessAbsolute(io, full, .{}) catch {
            gpa.free(full);
            continue;
        };
        return full;
    }
    return null;
}

fn spawnAndInit(ctx: *WorkerCtx, sc: ServerConfig) !void {
    const gpa = ctx.gpa;

    // argv = command + args.
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, sc.command.?);
    for (sc.args) |a| try argv.append(gpa, a);

    // Build the child environment: the parent's, plus per-entry env vars, plus
    // an augmented PATH so tools the server spawns in turn (npx's
    // `#!/usr/bin/env node`, package child processes) resolve too. When the
    // augmentation found anything, also rewrite argv[0] to an absolute path —
    // see `resolveCommandAlloc` for why spawn can't do it from the env map.
    var env_map: ?std.process.Environ.Map = null;
    defer if (env_map) |*m| m.deinit();
    var aug_path: ?[]u8 = null;
    defer if (aug_path) |p| gpa.free(p);
    var cmd_resolved: ?[]u8 = null;
    defer if (cmd_resolved) |p| gpa.free(p);
    if (ctx.mgr.environ) |environ| build: {
        if (ctx.mgr.home) |home|
            aug_path = augmentedPathAlloc(gpa, ctx.io, environ, home);
        if (sc.env.len == 0 and aug_path == null) break :build; // plain inherit
        var m = std.process.Environ.createMap(environ, gpa) catch break :build;
        for (sc.env) |e| m.put(e.key, e.value) catch {};
        if (aug_path) |p| m.put("PATH", p) catch {};
        env_map = m;
    }
    if (aug_path) |p| {
        if (resolveCommandAlloc(gpa, ctx.io, sc.command.?, p)) |abs| {
            cmd_resolved = abs;
            argv.items[0] = abs;
        }
    }

    var child = std.process.spawn(ctx.io, .{
        .argv = argv.items,
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .ignore,
        .environ_map = if (env_map) |*m| m else null,
    }) catch |e| {
        ctx.mgr.log("mcp: {s}: spawn '{s}' failed: {s}", .{ sc.name, argv.items[0], @errorName(e) });
        return InitError.SpawnFailed;
    };
    // `kill` closes stdin/stdout/stderr itself, so don't close them here too.
    errdefer child.kill(ctx.io);

    const read_buf = try gpa.alloc(u8, 1 << 20); // 1 MiB: holds a full JSON line
    errdefer gpa.free(read_buf);
    const reader = try gpa.create(Io.File.Reader);
    errdefer gpa.destroy(reader);
    reader.* = child.stdout.?.readerStreaming(ctx.io, read_buf);

    var srv: Server = .{
        .name = try gpa.dupe(u8, sc.name),
        .child = child,
        .reader = reader,
        .read_buf = read_buf,
    };
    errdefer gpa.free(srv.name);

    // JSON-RPC handshake: initialize → notifications/initialized → tools/list.
    {
        const init_params =
            \\{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"zig-ai","version":"0.1.0"}}
        ;
        const resp = try request(ctx, &srv, "initialize", init_params);
        gpa.free(resp);
        try notify(ctx, &srv, "notifications/initialized");
    }
    const tools_resp = try request(ctx, &srv, "tools/list", "{}");
    defer gpa.free(tools_resp);

    const count = ingestTools(ctx, srv.name, tools_resp) catch 0;
    publishStatus(ctx, sc.name, .running, count, null);
    ctx.mgr.log("mcp: {s} up — {d} tool(s)", .{ sc.name, count });

    try ctx.servers.append(gpa, srv);
}

/// Send a JSON-RPC request and block for its matching response. Returns the
/// `result` (or `error`) value as owned JSON text.
fn request(ctx: *WorkerCtx, srv: *Server, method: []const u8, params_json: []const u8) ![]u8 {
    const gpa = ctx.gpa;
    const id = srv.next_id;
    srv.next_id += 1;

    const line = try std.fmt.allocPrint(gpa, "{{\"jsonrpc\":\"2.0\",\"id\":{d},\"method\":\"{s}\",\"params\":{s}}}\n", .{ id, method, params_json });
    defer gpa.free(line);
    try srv.child.stdin.?.writeStreamingAll(ctx.io, line);

    // Read responses until we see one with our id (skip notifications/logs).
    var attempts: usize = 0;
    while (attempts < 10_000) : (attempts += 1) {
        // Inclusive so the trailing '\n' is consumed (the exclusive variant
        // leaves the delimiter, which would spin on empty reads); then trim it.
        const raw_inc = srv.reader.interface.takeDelimiterInclusive('\n') catch return InitError.HandshakeFailed;
        const raw = std.mem.trimEnd(u8, raw_inc, "\r\n");
        if (raw.len == 0) continue;
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, raw, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;
        const obj = parsed.value.object;
        const got = obj.get("id") orelse continue; // notification: no id
        const got_id: i64 = switch (got) {
            .integer => |n| n,
            .float => |f| @intFromFloat(f),
            else => continue,
        };
        if (got_id != @as(i64, @intCast(id))) continue;
        const payload = obj.get("result") orelse obj.get("error") orelse return InitError.HandshakeFailed;
        return std.json.Stringify.valueAlloc(gpa, payload, .{});
    }
    return InitError.HandshakeFailed;
}

fn notify(ctx: *WorkerCtx, srv: *Server, method: []const u8) !void {
    const gpa = ctx.gpa;
    const line = try std.fmt.allocPrint(gpa, "{{\"jsonrpc\":\"2.0\",\"method\":\"{s}\"}}\n", .{method});
    defer gpa.free(line);
    try srv.child.stdin.?.writeStreamingAll(ctx.io, line);
}

/// Parse a `tools/list` result and publish each tool to the manager snapshot.
fn ingestTools(ctx: *WorkerCtx, server_name: []const u8, result_json: []const u8) !usize {
    const gpa = ctx.gpa;
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, result_json, .{}) catch return 0;
    defer parsed.deinit();
    if (parsed.value != .object) return 0;
    const tools_v = parsed.value.object.get("tools") orelse return 0;
    if (tools_v != .array) return 0;

    var count: usize = 0;
    for (tools_v.array.items) |tv| {
        if (tv != .object) continue;
        const o = tv.object;
        const name_v = o.get("name") orelse continue;
        if (name_v != .string) continue;
        const name = name_v.string;
        const desc = if (o.get("description")) |d| (if (d == .string) d.string else "") else "";
        const schema: []u8 = if (o.get("inputSchema")) |s|
            (std.json.Stringify.valueAlloc(gpa, s, .{}) catch (gpa.dupe(u8, "{}") catch continue))
        else
            (gpa.dupe(u8, "{}") catch continue);

        var t: Tool = .{
            .server = gpa.dupe(u8, server_name) catch continue,
            .name = gpa.dupe(u8, name) catch continue,
            .qualified = std.fmt.allocPrint(gpa, "{s}__{s}", .{ server_name, name }) catch continue,
            .description = gpa.dupe(u8, desc) catch continue,
            .schema = schema,
        };
        ctx.mgr.snap_mu.lock();
        ctx.mgr.tools_snapshot.append(gpa, t) catch {
            ctx.mgr.snap_mu.unlock();
            t.deinit(gpa);
            continue;
        };
        ctx.mgr.snap_mu.unlock();
        count += 1;
    }
    return count;
}

/// Execute a queued `server__tool` call and push the result event.
fn doCall(ctx: *WorkerCtx, seq: u64, qualified: []const u8, args_json: []const u8) void {
    const gpa = ctx.gpa;

    // Built-in tools run locally on this worker thread (no server involved).
    if (builtins.isBuiltin(qualified)) {
        const text = builtins.execute(gpa, ctx.io, qualified, args_json);
        defer gpa.free(text);
        pushResult(ctx, seq, true, text);
        return;
    }

    // Split "server__tool".
    const sep = std.mem.indexOf(u8, qualified, "__") orelse {
        pushResult(ctx, seq, false, "invalid tool name");
        return;
    };
    const server_name = qualified[0..sep];
    const tool_name = qualified[sep + 2 ..];

    var srv: ?*Server = null;
    for (ctx.servers.items) |*s| {
        if (std.mem.eql(u8, s.name, server_name)) {
            srv = s;
            break;
        }
    }
    const server = srv orelse {
        pushResult(ctx, seq, false, "tool's server is not running");
        return;
    };

    // params = {"name": "<tool>", "arguments": <args>}
    const args = if (std.mem.trim(u8, args_json, " \t\r\n").len == 0) "{}" else args_json;
    const params = std.fmt.allocPrint(gpa, "{{\"name\":\"{s}\",\"arguments\":{s}}}", .{ tool_name, args }) catch {
        pushResult(ctx, seq, false, "out of memory");
        return;
    };
    defer gpa.free(params);

    const result = request(ctx, server, "tools/call", params) catch {
        pushResult(ctx, seq, false, "tool call failed (server error or timeout)");
        return;
    };
    defer gpa.free(result);

    // Extract the text content from the MCP result envelope.
    const text = extractText(gpa, result) catch gpa.dupe(u8, result) catch {
        pushResult(ctx, seq, false, "out of memory");
        return;
    };
    defer gpa.free(text);
    pushResult(ctx, seq, true, text);
}

/// Pull the concatenated `content[].text` out of a `tools/call` result. Falls
/// back to the raw JSON when the shape is unexpected.
fn extractText(gpa: std.mem.Allocator, result_json: []const u8) ![]u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, result_json, .{}) catch return gpa.dupe(u8, result_json);
    defer parsed.deinit();
    if (parsed.value != .object) return gpa.dupe(u8, result_json);
    const content = parsed.value.object.get("content") orelse return gpa.dupe(u8, result_json);
    if (content != .array) return gpa.dupe(u8, result_json);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (content.array.items) |item| {
        if (item != .object) continue;
        const tv = item.object.get("text") orelse continue;
        if (tv != .string) continue;
        if (out.items.len > 0) try out.append(gpa, '\n');
        try out.appendSlice(gpa, tv.string);
    }
    if (out.items.len == 0) return gpa.dupe(u8, result_json);
    return out.toOwnedSlice(gpa);
}

fn pushResult(ctx: *WorkerCtx, seq: u64, ok: bool, text: []const u8) void {
    const dup = ctx.gpa.dupe(u8, text) catch return;
    ctx.mgr.events.push(.{ .result = .{ .seq = seq, .ok = ok, .text = dup } });
}

fn publishStatus(ctx: *WorkerCtx, name: []const u8, state: RunState, tools: usize, msg: ?[]const u8) void {
    const gpa = ctx.gpa;
    const mgr = ctx.mgr;
    mgr.snap_mu.lock();
    defer mgr.snap_mu.unlock();
    // Replace an existing row for this server, else append.
    for (mgr.status_snapshot.items) |*s| {
        if (std.mem.eql(u8, s.name, name)) {
            s.state = state;
            s.tools = tools;
            if (s.msg) |m| gpa.free(m);
            s.msg = if (msg) |mm| gpa.dupe(u8, mm) catch null else null;
            return;
        }
    }
    const row: ServerStatus = .{
        .name = gpa.dupe(u8, name) catch return,
        .state = state,
        .tools = tools,
        .msg = if (msg) |mm| gpa.dupe(u8, mm) catch null else null,
    };
    mgr.status_snapshot.append(gpa, row) catch {
        gpa.free(row.name);
        if (row.msg) |m| gpa.free(m);
    };
}

// ===========================================================================
// Tests
// ===========================================================================

test "parse and serialize round-trips a server entry" {
    const gpa = std.testing.allocator;
    const json =
        \\{"mcpServers":{"github":{"command":"npx","args":["-y","x"],"env":{"TOK":"abc"},"disabled":false}}}
    ;
    var reg = parse(gpa, json);
    defer reg.deinit();
    try std.testing.expectEqual(@as(usize, 1), reg.servers.items.len);
    const s = reg.servers.items[0];
    try std.testing.expectEqualStrings("github", s.name);
    try std.testing.expectEqualStrings("npx", s.command.?);
    try std.testing.expectEqual(@as(usize, 2), s.args.len);
    try std.testing.expectEqual(@as(usize, 1), s.env.len);
    try std.testing.expectEqualStrings("TOK", s.env[0].key);

    const out = try serialize(gpa, reg.servers.items);
    defer gpa.free(out);
    // Re-parse the serialized form to confirm it's stable.
    var reg2 = parse(gpa, out);
    defer reg2.deinit();
    try std.testing.expectEqualStrings("github", reg2.servers.items[0].name);
    try std.testing.expectEqualStrings("abc", reg2.servers.items[0].env[0].value);
}

test "buildServerConfig substitutes arg placeholders and env values" {
    const gpa = std.testing.allocator;
    // filesystem: one arg-kind input replacing <PATH>.
    const fs = presetById("filesystem").?;
    var sc = try buildServerConfig(gpa, fs, &.{"/tmp/work"});
    defer sc.deinit(gpa);
    try std.testing.expectEqualStrings("npx", sc.command.?);
    try std.testing.expectEqualStrings("/tmp/work", sc.args[sc.args.len - 1]); // <PATH> replaced
    try std.testing.expectEqual(@as(usize, 0), sc.env.len);

    // github: one env-kind input → env pair.
    const gh = presetById("github").?;
    var sc2 = try buildServerConfig(gpa, gh, &.{"ghp_secret"});
    defer sc2.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), sc2.env.len);
    try std.testing.expectEqualStrings("GITHUB_PERSONAL_ACCESS_TOKEN", sc2.env[0].key);
    try std.testing.expectEqualStrings("ghp_secret", sc2.env[0].value);

    // Empty value leaves the placeholder intact.
    var sc3 = try buildServerConfig(gpa, fs, &.{""});
    defer sc3.deinit(gpa);
    try std.testing.expectEqualStrings("<PATH>", sc3.args[sc3.args.len - 1]);
}

test "catalog ids are unique" {
    for (catalog, 0..) |a, i| {
        for (catalog, 0..) |b, j| {
            if (i == j) continue;
            try std.testing.expect(!std.mem.eql(u8, a.id, b.id));
        }
    }
}
