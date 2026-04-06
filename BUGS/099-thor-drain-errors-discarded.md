# BUG-099: Drain/drain-complete errors silently discarded

**Severity:** Medium  
**File:** `crates/ax-thor-agent/src/main.rs:58, 68`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
let _ = agent::drain(&client, &config, &runtime).await;
// ... inflight wait ...
let _ = agent::drain_complete(&client, &config, &runtime).await;
```

Both `drain()` and `drain_complete()` return `Result<()>`, but errors are discarded with `let _ =`. If `drain()` fails (e.g., control plane unreachable), `drain_complete()` is still called on the same session. The control plane's worker state machine expects `active → draining → drained`. If drain() fails, the state is still `active`. Calling drain_complete() on an `active` worker is undefined.

## Why It's A Bug

At minimum, the error should be logged for operational debugging. Currently, a failed drain is completely silent.

## Suggested Fix

```rust
if let Err(err) = agent::drain(&client, &config, &runtime).await {
    tracing::warn!(%err, "drain request failed");
}
```

## Fix Applied
Replaced `let _ = agent::drain(...)` and `let _ = agent::drain_complete(...)` with `if let Err(e)` + `tracing::warn!`.
