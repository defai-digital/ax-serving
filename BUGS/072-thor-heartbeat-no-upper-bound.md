# BUG-072: Thor `heartbeat_interval_ms` Not Upper-Bounded

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/agent.rs:101-104`
**Status:** ✅ FIXED (2026-03-29)

## Description

The control plane can return any `u64` value for `heartbeat_interval_ms`. It's clamped to a minimum of 1,000 ms but has no maximum. A misconfigured or malicious control plane could return `u64::MAX`, causing the worker to go silent for effectively forever.

## Impact

The worker would appear dead to the control plane, yet never attempt to re-register or heartbeat. The operator would see a phantom worker.

## Fix

Add an upper bound:

```rust
.unwrap_or(5_000)
.clamp(1_000, 300_000) // 1s - 5min
```

## Fix Applied
Changed `.max(1_000)` to `.clamp(1_000, 300_000)` — heartbeat interval is now bounded between 1s and 5min.
