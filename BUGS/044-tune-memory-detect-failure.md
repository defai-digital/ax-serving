# BUG-044: Detect Memory GB Returns 0 on `sysctl` Failure

**Severity:** Low
**File:** `crates/ax-serving-cli/src/tune.rs`
**Lines:** 253–255
**Status:** ✅ FIXED (2026-03-29)

## Description

If `sysctl hw.memsize` fails (e.g., restricted sandbox), `detect_memory_gb()` returns `0`. The `HardwareProfile` is constructed with `total_memory_gb: 0` and the `run_tune()` output prints `Memory: 0 GB`.

## Impact

The `tune` command succeeds silently with a clearly wrong memory value (0 GB). A downstream operator might use the generated config without noticing, though the actual tuning parameters (based on SKU class) would still be correct.

## Fix

Return an error or emit a warning when memory detection fails instead of silently proceeding with 0.

## Fix Applied
If `detect_memory_gb()` computes 0, emit a `WARNING` via `eprintln!` and return 8 (GB) as a conservative fallback.
