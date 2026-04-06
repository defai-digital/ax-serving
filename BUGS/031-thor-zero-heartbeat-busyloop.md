# BUG-031: Thor Agent Zero Heartbeat Interval Causes Busy-Loop

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/agent.rs`
**Lines:** 101, 173–176
**Status:** ✅ FIXED (2026-03-29)

## Description

If the control plane's registration response contains `"heartbeat_interval_ms": 0`, the agent will call `tokio::time::sleep(Duration::from_millis(0))` in a tight loop. While tokio yields on zero-duration sleeps, this still results in a continuous stream of HTTP requests to the control plane with virtually no throttling:

```rust
let heartbeat_interval_ms = response["heartbeat_interval_ms"].as_u64().unwrap_or(5_000);
// ...
tokio::time::sleep(std::time::Duration::from_millis(session.heartbeat_interval_ms)).await;
```

## Impact

A misconfigured or buggy control plane returning `heartbeat_interval_ms: 0` causes the agent to hammer the control plane with heartbeats as fast as the HTTP round-trip allows, consuming excessive resources on both sides.

## Fix

Clamp the value to a minimum:

```rust
let heartbeat_interval_ms = response["heartbeat_interval_ms"]
    .as_u64()
    .unwrap_or(5_000)
    .max(1_000);
```

## Fix Applied
Added `.max(1_000)` clamp to `heartbeat_interval_ms` in `register()`, ensuring minimum 1-second interval.
