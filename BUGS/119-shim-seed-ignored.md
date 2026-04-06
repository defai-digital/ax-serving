# BUG-119: `llama_new_context_with_model` ignores `seed` parameter entirely

**Severity:** Low  
**File:** `crates/ax-serving-shim/src/context.rs:12,30-53`  
**Status:** ❌ FALSE POSITIVE  
**Introduced:** 2026-03-29

## Description

```rust
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub seed: u32,  // stored but never used
    pub _pad: [u32; 6],
}
```

The `seed` field is accepted from the C caller but never used anywhere. In llama.cpp, the seed controls the RNG for sampling.

## Why It's A Bug

C callers expecting deterministic output by setting a specific seed get non-deterministic results.

## Suggested Fix

Store the seed in `LlamaContext` and use it to seed an RNG in sampling functions, or document that the shim only supports greedy sampling.
