# BUG-029: Silent Failure When Generation Channel Closes Without Terminal Event

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/main.rs`
**Lines:** 476–498
**Status:** ✅ FIXED (2026-03-29)

## Description

The inference event loop exits cleanly when the channel closes (`rx.recv().await` returns `None`), regardless of whether a `GenerateEvent::Done` or `GenerateEvent::Error` was received. If the backend drops the sender due to an internal crash or bug without sending a terminal event, the loop exits with `final_stats = None`, and the function returns `Ok(())`:

```rust
let mut final_stats = None;
while let Some(event) = rx.recv().await {
    match event {
        GenerateEvent::Done(s) => { final_stats = Some((s, n_tokens)); break; }
        GenerateEvent::Error(e) => { return Err(anyhow::anyhow!(e)); }
        // ...
    }
}
Ok(final_stats) // Ok(None) when channel closed without Done/Error
```

## Impact

The user sees partial or no output, the process exits with code 0 (success), and has no indication that inference failed. In scripts or CI pipelines, the successful exit code masks the failure.

## Fix

After the while loop, check for missing terminal event:

```rust
if final_stats.is_none() {
    return Err(anyhow::anyhow!(
        "generation ended unexpectedly — backend closed the event channel without sending Done or Error"
    ));
}
```

## Fix Applied
After the `while let` loop, check `if final_stats.is_none()` and return `Err(anyhow!("generation channel closed without Done or Error event"))`.
