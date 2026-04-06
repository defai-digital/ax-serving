# BUG-010: Unbounded Busy-Wait During Thor Agent Shutdown

**Severity:** High
**File:** `crates/ax-thor-agent/src/main.rs`
**Lines:** 51–53
**Status:** ✅ FIXED (2026-03-28)

## Description

After the drain signal is sent, the shutdown code polls `inflight` in a loop with no upper bound on iterations or elapsed time.

```rust
let _ = agent::drain(&client, &config, &runtime).await;
while runtime.inflight.load(std::sync::atomic::Ordering::Relaxed) > 0 {
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
}
heartbeat_task.abort();
let _ = agent::drain_complete(&client, &config, &runtime).await;
```

If a request is stuck (e.g., SGLang runtime is unresponsive), `inflight` never decrements to zero and the process never terminates.

## Impact

A stuck request prevents the Thor agent from ever exiting. In production, this causes zombie processes, failed deployments, and hanging orchestration. Requires manual `kill -9` to resolve.

## Fix

Add a configurable idle timeout:

```rust
let shutdown_deadline = tokio::time::Instant::now()
    + Duration::from_secs(config.shutdown_timeout_secs.unwrap_or(30));
while runtime.inflight.load(Ordering::Relaxed) > 0 {
    if tokio::time::Instant::now() > shutdown_deadline {
        tracing::warn!("shutdown timeout exceeded with inflight requests; forcing exit");
        break;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
}
```

## Fix Applied

Added a configurable shutdown deadline (`AXS_THOR_SHUTDOWN_TIMEOUT_SECS`, default 30s). The busy-wait loop now checks `tokio::time::Instant::now() > shutdown_deadline` each iteration and breaks with a warning if exceeded. `ThorConfig` gains a `shutdown_timeout_secs: Option<u64>` field.
