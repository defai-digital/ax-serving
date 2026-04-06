# BUG-089: Bench `JoinError` Silently Discarded Across All Subcommands

**Severity:** Low
**File:** `crates/ax-serving-bench/src/mixed.rs:215`, `compare.rs:189`, `service_perf.rs:181`, `multi_worker.rs:183`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
for h in handles {
    let _ = h.await;
}
```

If a spawned task panics (e.g., lock poisoned, assertion failure), `h.await` returns `Err(JoinError)`. The error is silently discarded, masking serious issues.

## Fix

Log the error:

```rust
for h in handles {
    if let Err(e) = h.await {
        if e.is_panic() {
            warn!("task panicked: {e}");
        }
    }
}
```

## Fix Applied
Changed `let _ = h.await` to check `if let Err(e) = h.await && e.is_panic()` and log with `tracing::warn!` in mixed.rs, compare.rs, and multi_worker.rs.
