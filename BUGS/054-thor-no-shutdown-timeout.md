# BUG-054: No Graceful Shutdown Timeout for axum Server

**Severity:** High
**File:** `crates/ax-thor-agent/src/main.rs:46-48`
**Status:** ✅ FIXED (2026-03-29)

## Description

`axum::serve(listener, app).with_graceful_shutdown(shutdown).await` has no deadline. If a client holds open a streaming connection indefinitely, the process will never exit after ctrl-C. The `shutdown_timeout_secs` config (line 53) only governs the *post-server* inflight drain loop, not the server's own graceful shutdown phase.

## Impact

The process hangs forever on shutdown if any streaming response is active. Operations teams must resort to `kill -9`, which can leave the backend (sglang) in an inconsistent state.

## Fix

Wrap the serve call in a `tokio::time::timeout`:

```rust
let shutdown_timeout = config.shutdown_timeout_secs.unwrap_or(30);
tokio::time::timeout(
    Duration::from_secs(shutdown_timeout),
    axum::serve(listener, app).with_graceful_shutdown(shutdown),
)
.await???;
```

## Fix Applied
Wrapped `axum::serve(...).with_graceful_shutdown(...)` in `tokio::time::timeout()` with `shutdown_timeout_secs + 5` seconds. Also clamped `shutdown_timeout_secs` to minimum 1 (BUG-071).
