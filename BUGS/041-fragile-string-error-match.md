# BUG-041: Error Matching by String Comparison Is Fragile

**Severity:** Low
**File:** `crates/ax-serving-engine/src/ax_engine.rs`
**Lines:** 609
**Status:** ⏳ DEFERRED

## Description

```rust
Err(err) if err.to_string() == "receiver dropped" => return Ok(()),
```

If the `anyhow::bail!("receiver dropped")` message changes (typo fix, rewording, etc.), this match arm silently stops working and the error falls through to the generic error path, producing a confusing error message.

## Impact

Fragile coupling between error producer and consumer. A message change would cause a regression where graceful receiver-dropped handling becomes an error reported to the user.

## Fix

Use a custom error type or an error downcast rather than string matching:
```rust
#[derive(Debug, thiserror::Error)]
#[error("receiver dropped")]
struct ReceiverDropped;
// Then match with err.downcast_ref::<ReceiverDropped>()
```
