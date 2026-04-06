# BUG-059: Bench `drain_channel` Returns `Ok(zeros)` When Channel Closes Without `Done`

**Severity:** High
**File:** `crates/ax-serving-bench/src/bench.rs:256-278`
**Status:** ✅ FIXED (2026-03-29)

## Description

If the backend drops the sender without sending `Done` or `Error`, `rx.recv().await` returns `None`, the loop exits, and the function returns `Ok(stats)` with all-zero `GenerationStats`. The caller reports `0.0` prefill and decode tok/s with no error indication.

Notably, `soak.rs` has a `received_done` guard that catches this, but `bench.rs` does not.

## Impact

Benchmark silently reports zero performance metrics instead of failing, masking backend crashes or connection issues.

## Fix

Detect channel close without Done event:

```rust
let mut received_done = false;
while let Some(event) = rx.recv().await {
    match event {
        GenerateEvent::Done { .. } => { received_done = true; break; }
        // ...
    }
}
if !received_done {
    anyhow::bail!("generation channel closed without Done event");
}
```

## Fix Applied
Track `got_done` flag; after the while loop, if `!got_done`, return `anyhow::bail!("generation channel closed without Done event")`.
