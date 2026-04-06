# BUG-106: `bench.rs` wall-clock fallback divides by total elapsed — ~10x underestimate

**Severity:** Medium  
**File:** `crates/ax-serving-bench/src/bench.rs:262-269`  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

```rust
GenerateEvent::Done(s) => {
    if s.prefill_tok_per_sec > 0.0 && s.decode_tok_per_sec > 0.0 {
        stats = s;
    } else {
        let elapsed = wall.elapsed().as_secs_f64();
        stats.prefill_tok_per_sec = n_prompt as f64 / elapsed;
        stats.decode_tok_per_sec = params.max_tokens.map(|n| n as f64 / elapsed).unwrap_or(0.0);
    }
```

The fallback computes both prefill and decode tok/s divided by total elapsed time (prefill + decode). Prefill tok/s is underestimated by roughly `(prefill_time / total_time)` ratio. For 1024 prompt tokens in 100ms prefill + 1000ms decode: correct = 10240 tok/s, fallback = 930 tok/s (10x underestimate).

## Why It's A Bug

When hit, the reported numbers are significantly wrong. Prefill appears ~10x slower than reality.

## Suggested Fix

Warn the user that numbers are approximate, or ensure this path is never taken by always using split-phase mode when backend stats are unreliable.
