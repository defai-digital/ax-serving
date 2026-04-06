# BUG-039: Health Poller Marks Model Dead on First Spawn Failure

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/llamacpp.rs`
**Lines:** 1235–1238
**Status:** ✅ FIXED (2026-03-29)

## Description

The health poller has a configurable `max_restarts` budget, but if `spawn_server` fails, the model is immediately marked Dead and the poller exits, ignoring any remaining restart attempts:

```rust
Err(e) => {
    warn!("failed to spawn llama-server: {e}");
    health.store(HealthState::Dead as u8, Ordering::Relaxed);
    break; // immediately gives up, ignoring remaining restart budget
}
```

The `restart_count < max_restarts` check at line 1192 only gates the decision to attempt a restart, but a spawn failure is treated as fatal.

## Impact

A transient spawn failure (e.g., temporary `fork()` failure under memory pressure) permanently kills the model when it should have retried. With `max_restarts=3`, a single transient spawn failure wastes the remaining 2 restart attempts.

## Fix

On spawn failure, increment `restart_count` and continue the loop instead of immediately marking Dead:

```rust
Err(e) => {
    warn!("failed to spawn llama-server: {e}");
    if restart_count >= max_restarts {
        health.store(HealthState::Dead as u8, Ordering::Relaxed);
        break;
    }
    // continue loop to retry after next backoff
}
```

## Fix Applied
Changed `break` to `continue` on spawn failure, allowing the restart loop to try again if budget remains. The `restart_count >= max_restarts` check at the top of the loop handles exhaustion.
