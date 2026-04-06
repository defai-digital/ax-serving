# BUG-087: Shim `penalty_last_n == 0` Diverges From llama.cpp

**Severity:** Low
**File:** `crates/ax-serving-shim/src/sampling.rs:187`
**Status:** ⏳ DEFERRED

## Description

```rust
if candidates.is_null() || last_tokens.is_null() || penalty_last_n == 0 {
    return;
}
```

If `penalty_last_n` is 0 but `last_tokens` is non-null with valid data, the function returns immediately without applying any penalty. In llama.cpp, `penalty_last_n == 0` means "use the full window" (often configured as the context length). The shim treats it as "skip entirely."

## Fix

When `penalty_last_n == 0`, use a default window size or compute the penalty over the full `last_tokens` buffer.
