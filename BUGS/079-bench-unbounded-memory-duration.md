# BUG-079: Bench Unbounded Memory Growth in Duration Mode

**Severity:** Medium
**File:** `crates/ax-serving-bench/src/multi_worker.rs:123-181`
**Status:** ⏳ DEFERRED

## Description

In duration mode (`--duration-secs`), `total_requests` is set to `usize::MAX`. The `handles` Vec and the `latencies` Mutex<Vec<f64>> grow without bound:

```rust
let mut handles = Vec::new();
loop {
    // ...
    handles.push(h);
    req_index += 1;
}
```

A 1-hour run at 100 RPS accumulates ~360K handles and latency entries (~20 MB of handles + data). A 24-hour run would accumulate ~8.6M entries.

## Fix

Periodically await and drain completed handles:

```rust
if handles.len() >= 1000 {
    handles.retain(|h| !h.is_finished());
}
```
