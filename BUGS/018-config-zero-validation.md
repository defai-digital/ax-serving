# BUG-018: `AXS_SCHED_MAX_QUEUE` and `AXS_GLOBAL_QUEUE_DEPTH` Accept Zero

**Severity:** Low
**File:** `crates/ax-serving-api/src/config.rs`
**Lines:** 521–523, 624–626
**Status:** ✅ FIXED (2026-03-28)

## Description

Two config fields accept zero without clamping, unlike their neighboring fields which all use `.max(1)`.

**`AXS_SCHED_MAX_QUEUE` (line 521–523):**
```rust
if let Some(n) = env_parse::<usize>("AXS_SCHED_MAX_QUEUE") {
    self.sched_max_queue = n;  // No .max(1) clamp
}
```

**`AXS_GLOBAL_QUEUE_DEPTH` (line 624–626):**
```rust
if let Some(n) = env_parse::<usize>("AXS_GLOBAL_QUEUE_DEPTH") {
    self.orchestrator.global_queue_depth = n;  // No minimum enforcement
}
```

Contrast with neighboring `AXS_SCHED_MAX_INFLIGHT`:
```rust
if let Some(n) = env_parse::<usize>("AXS_SCHED_MAX_INFLIGHT") {
    self.sched_max_inflight = n.max(1);  // Clamped to at least 1
}
```

## Impact

Setting either to `0` causes immediate request rejection under any load — almost certainly a misconfiguration. With `AXS_SCHED_MAX_QUEUE=0`, every request that arrives when all inflight slots are occupied is immediately rejected (503).

## Fix

```rust
self.sched_max_queue = n.max(1);
self.orchestrator.global_queue_depth = n.max(1);
```

## Fix Applied

Added `.max(1)` clamp to both env var handlers, matching the pattern used by `AXS_SCHED_MAX_INFLIGHT` and `AXS_GLOBAL_QUEUE_MAX`.
