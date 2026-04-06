# BUG-105: `mixed.rs` gate failure prints "FAIL" but doesn't return error

**Severity:** Medium  
**File:** `crates/ax-serving-bench/src/mixed.rs:315-321`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
let gate_pass = overall_p99 < cfg.target_p99_ms as f64;
println!(
    "Gate: P99 < {}ms: {} (actual: {:.0}ms)",
    cfg.target_p99_ms, if gate_pass { "PASS" } else { "FAIL" }, overall_p99
);
```

If the gate fails, the function still returns `Ok(())`. In CI, the benchmark command exits 0 regardless of whether the latency target was met.

## Why It's A Bug

The `--target-p99-ms` flag is documented as a "pass/fail gate" but doesn't cause a failure exit code. CI pipelines using this gate will always pass.

## Suggested Fix

Add `anyhow::ensure!(gate_pass, "P99 gate failed: {overall_p99:.0}ms > {}ms target", cfg.target_p99_ms)` or return an error when the gate fails.

## Fix Applied
Changed `print_report` return type to `Result<()>` and added `anyhow::ensure!(gate_pass, ...)` to return an error when the P99 gate fails.
