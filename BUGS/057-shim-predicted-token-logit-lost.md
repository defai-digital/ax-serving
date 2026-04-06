# BUG-057: Shim Predicted Token Logit Silently Lost When `next_id >= logits.len()`

**Severity:** High
**File:** `crates/ax-serving-shim/src/context.rs:127-129`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
if (next_id as usize) < ctx_ref.logits.len() {
    ctx_ref.logits[next_id as usize] = 20.0;
}
// else: silently ignored -- all logits remain -20.0
```

If the backend returns a `next_id` >= `vocab_size` (the logits buffer length), the `if` branch is skipped. All logits remain at -20.0. When `llama_sample_token_greedy` runs, every token has identical logit (-20.0), so `.max_by` returns the **first** entry -- an arbitrary token. The model's actual prediction is silently discarded. No error is logged.

## Impact

Silent correctness failure. The shim returns a random token instead of the model's predicted token, producing garbage output with no error indication.

## Fix

Log an error when the predicted token falls outside the logit buffer:

```rust
if (next_id as usize) < ctx_ref.logits.len() {
    ctx_ref.logits[next_id as usize] = 20.0;
} else {
    tracing::error!(
        "llama_eval: predicted token {next_id} exceeds vocab_size {}",
        ctx_ref.logits.len()
    );
}
```

## Fix Applied
Added `tracing::warn!` when `next_id >= logits.len()`, logging both the token ID and vocab size so the out-of-range condition is visible.
