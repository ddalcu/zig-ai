//! Test aggregator. Rooted in `src/` so modules can resolve their relative
//! imports (e.g. `server/chat_parser.zig` importing `../agent.zig`). Run with
//! `zig test src/tests.zig`.

test {
    _ = @import("agent.zig");
    _ = @import("server/chat_parser.zig");
    _ = @import("launcher.zig");
}
