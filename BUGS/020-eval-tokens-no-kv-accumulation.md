# BUG-020: `eval_tokens` Creates Fresh KV Cache Every Call, Breaking Documented Contract

**Severity:** High
**File:** `crates/ax-serving-engine/src/ax_engine.rs`
**Lines:** 787–798
**Status:** ❌ FALSE POSITIVE

## Description

The `InferenceBackend::eval_tokens` trait documentation states *"Tokens accumulate across calls (KV cache is extended)."* However, the `AxEngineBackend` implementation calls `create_model_kv()` on every invocation, creating a fresh empty KV cache each time:

```rust
fn eval_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<u32> {
    anyhow::ensure!(!tokens.is_empty(), "eval_tokens: empty input");
    let loaded = self.get_model(handle)?;
    let weights = WeightStore::new(&loaded.mapped);
    let mut kv = loaded.model.create_model_kv(); // <-- NEW empty KV each call!
    let mut logits = vec![0.0f32; loaded.metadata.vocab_size as usize];
    loaded.model.forward_batch(tokens, &mut kv, &weights, &mut logits)?;
    Ok(ax_core::sampling::argmax(&logits))
}
```

The C API shim (`ax-serving-shim`) accumulates tokens in `token_buf` across calls and passes the full buffer to `eval_tokens`, expecting the backend to build on prior context. But since `eval_tokens` re-creates the KV cache each time, the shim's accumulation pattern works *despite* the bug (by re-processing all tokens each call). Any caller that passes only incremental tokens will lose all prior context.

## Impact

Any caller relying on KV accumulation semantics gets broken results — each eval is isolated rather than building on prior context. The API contract is violated. While the current shim works around this by re-passing all tokens, this is O(n²) in total tokens and defeats the purpose of incremental KV caching.

## Fix

Store per-handle KV state (e.g., in a `HashMap<ModelHandle, KV>` inside `LoadedModel` or the backend) so that successive `eval_tokens` calls on the same handle extend the same KV cache. Only create a new KV cache if the handle has no prior state.
