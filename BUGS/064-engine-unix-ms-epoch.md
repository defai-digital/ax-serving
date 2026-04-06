# BUG-064: `unix_ms_now()` Returns 0 on Pre-Epoch Clocks

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/llamacpp.rs:235-240`
**Status:** ❌ FALSE POSITIVE

## Description

```rust
duration_since(UNIX_EPOCH).unwrap_or_default()
```

Silently returns 0 if the system clock is before epoch. When `last_opened_ms` is 0 (default) and `unix_ms_now()` also returns 0, the circuit breaker recovery check `elapsed_ms < recovery_ms` evaluates to `0 < 10000` = true, which is correct. But if the clock is wrong and then corrects, `last_opened_ms = 0` would cause `elapsed_ms` to be enormous, immediately bypassing the breaker.

## Fix

Log a warning if `duration_since` fails, or use `Instant` for monotonic timing instead of `SystemTime` for breaker state.
