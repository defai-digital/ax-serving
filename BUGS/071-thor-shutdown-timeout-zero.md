# BUG-071: Thor `shutdown_timeout_secs=0` Causes Immediate Deadline Expiry

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/config.rs:65-67`, `main.rs:53-54`
**Status:** ✅ FIXED (2026-03-29)

## Description

If `AXS_THOR_SHUTDOWN_TIMEOUT_SECS=0`, the shutdown deadline is set to `Instant::now() + 0s`, which is immediately in the past. The drain loop will break on the first iteration without waiting for any inflight requests to complete.

## Impact

Zero is a plausible misconfiguration (someone intending "no timeout"). It silently causes inflight requests to be abandoned.

## Fix

Either reject `0` during config parsing or apply a minimum:

```rust
.map(|v| v.parse::<u64>().ok())
.map(|v| v.max(1))
```

## Fix Applied
Clamped `shutdown_timeout_secs` to `.max(1)` in both the axum server timeout and the drain loop deadline.
