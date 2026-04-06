# BUG-038: Health Poller Exponential Backoff Overflow With Large Restart Count

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/llamacpp.rs`
**Lines:** 1204
**Status:** ✅ FIXED (2026-03-29)

## Description

`restart_count` is a `u32` that increments each restart attempt. If `restart_count >= 64`, `2u64.pow(restart_count)` overflows:

```rust
let backoff = Duration::from_secs((2u64.pow(restart_count)).min(16));
```

The `.min(16)` doesn't help because the overflow happens before `min` is evaluated. While the default `max_restarts` is 3, a user could set `AXS_HEALTH_POLLER_MAX_RESTARTS=100`.

## Impact

In debug builds: panic (crash) of the health poller thread, which then stops monitoring the model. In release builds: wraparound producing a near-zero backoff, causing rapid restart loops.

## Fix

Cap the exponent before `pow`:

```rust
let exp = restart_count.min(4); // 2^4 = 16 is the cap anyway
let backoff = Duration::from_secs(1u64 << exp);
```

## Fix Applied
Replaced `2u64.pow(restart_count).min(16)` with `1u64 << restart_count.min(4)`. The exponent is capped to 4 before shifting, preventing overflow for any restart_count value.
