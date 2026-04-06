# BUG-090: Bench Inconsistent Percentile Calculations Across Subcommands

**Severity:** Low
**File:** `crates/ax-serving-bench/src/bench_common.rs:43` vs `service_perf.rs:357`
**Status:** ⏳ DEFERRED

## Description

Three different percentile methods are used:

| File | Formula | Result for len=100, p=95 |
|------|---------|--------------------------|
| `bench_common.rs` | `sorted[len * p / 100]` | index 95 |
| `service_perf.rs` | `sorted[(p/100 * (len-1)).round()]` | index 94 |
| `perf.rs` | `sorted[(len * p / 100).min(len-1)]` | index 95 |

For the same dataset, different subcommands report different P95/P99 values.

## Fix

Consolidate to a single `percentile` function in `bench_common.rs`.
