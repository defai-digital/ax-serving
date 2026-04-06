# BUG-060: Bench Metrics Fetch Failures Silently Swallowed

**Severity:** High
**File:** `crates/ax-serving-bench/src/service_perf.rs:90-188`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
let baseline = fetch_snapshot(&client, &cfg.url)
    .await
    .unwrap_or(RunSnapshot { /* all zeros */ });
```

And after the benchmark:

```rust
let after = fetch_snapshot(&client, &cfg.url)
    .await
    .unwrap_or_else(|_| baseline.clone());
```

If `/v1/metrics` is unreachable, baseline defaults to all zeros and the post-run snapshot also defaults to the all-zeros baseline. All delta fields (cache_hits, batches, evictions) compute as 0. The benchmark reports zero cache hits, zero batching, and passes all validation gates -- with no warning.

## Impact

Entire service_perf benchmark silently produces zero metrics, passing all gates with no indication that measurements were not actually collected.

## Fix

Propagate the error or at minimum warn loudly:

```rust
let baseline = fetch_snapshot(&client, &cfg.url)
    .await
    .context("failed to fetch baseline metrics -- cannot compute deltas")?;
```

## Fix Applied
Changed `.unwrap_or(...)` to `match` with `Err(e)` arm that prints `WARNING: baseline metrics fetch failed ({e})` before falling back to zeros.
