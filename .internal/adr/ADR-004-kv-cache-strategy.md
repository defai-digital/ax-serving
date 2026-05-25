# ADR-004: KV Cache Strategy — mistralrs Paged Attention + ax-engine Prefix Cache

**Status**: Accepted (Phase 1: mistralrs paged attn; Phase 2: ax-engine prefix cache)
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

---

## Context

Two KV cache implementations exist:

**mistralrs-core paged attention** (`mistralrs-paged-attn`):
- vLLM-style block allocation on GPU
- Fixed block size (default 16 tokens/block)
- Handles multi-sequence concurrency
- No prefix cache (blocks are per-request)

**ax-engine prefix cache** (`kv/paged_kv.rs`, `kv/gpu_cache.rs`):
- LRU prefix cache keyed by (model_fingerprint, token_hash)
- GPU KV block export/import across requests
- Demonstrated speedups: 25× (41-tok prompt), 134× (201-tok), 505× (high-reuse)
- Requires direct Metal buffer access for block transfer

The prefix cache is one of ax-engine's unique contributions. It provides orders-of-
magnitude speedup for workloads with shared system prompts (chatbots, API servers).

---

## Decision

### Phase 1

Use mistralrs-core's paged attention as-is. No modifications to mistralrs-core KV
management. Reason: Phase 1 priority is basic correctness and performance parity.
Modifying mistralrs-core's attention infrastructure adds significant risk.

KV cache behavior in Phase 1:
- Block size: 16 tokens (mistralrs default)
- f16 KV: auto-enabled when context_length ≥ 256 (matches ax-engine policy)
- No prefix reuse: each request fills from token 0

### Phase 2

Port ax-engine's prefix cache as a **layer above** mistralrs paged attention:

```
ax-serving prefix cache (ax-serving-api)
  └── exports/imports KV blocks via InferenceBackend::export_kv_block()
       └── mistralrs-core paged attention (GPU resident)
```

`InferenceBackend` gains two new methods:

```rust
pub trait InferenceBackend {
    // ... existing methods ...

    /// Export a completed KV block to host memory for caching.
    fn export_kv_block(&self, handle: ModelHandle, block_idx: usize) -> Result<KvBlockPayload>;

    /// Import a cached KV block back to GPU and advance the sequence position.
    fn import_kv_block(&self, handle: ModelHandle, block: &KvBlockPayload) -> Result<()>;
}
```

The `PagedKvManager` from ax-engine is ported to `ax-serving-api/src/kv/` with:
- Block pool with refcounting (prevent eviction of in-use blocks)
- LRU eviction policy (tick-based)
- `PrefixCacheKey`: (model_id, group_id, token_count, token_hash)
- Max entries: `AXS_PREFIX_CACHE_MAX_ENTRIES` (default: 1024)
- Block size: `AXS_PAGED_KV_BLOCK_SIZE` (default: 16)

---

## Phase 2 Export/Import Protocol

1. **Prefill begin**: `registry.begin_paged_kv_request(request_id, prompt_tokens)`
   - Hash the prefix tokens
   - Look up cached blocks in LRU
   - For each cached block, call `backend.import_kv_block(block)` to restore GPU state
   - Return `reusable_prefix_count`

2. **Compute only uncached tokens**: advance position from `reusable_prefix_count`

3. **After prefill**: `registry.remember_paged_kv_prefix(request_id, prompt_tokens)`
   - Call `backend.export_kv_block()` for each newly computed block
   - Insert into LRU prefix cache

4. **Request end**: RAII `PrefixCacheGuard` drops → `registry.end_paged_kv_request()`
   → release block ownership

---

## Consequences

### Positive

- **Phase 1**: Simple, fast to implement, immediate performance parity
- **Phase 2**: Prefix cache preserves ax-engine's biggest unique advantage
- **Clean boundary**: prefix cache logic stays in ax-serving-api, not in mistralrs-core

### Negative

- **Phase 2 complexity**: export/import requires Metal buffer read-back to host memory.
  Benchmark required to ensure export latency does not negate cache benefit.
- **Block format coupling**: exported blocks encode mistralrs-core's internal layout.
  If mistralrs-core changes its KV layout, export/import breaks.

---

## Alternatives Considered

### A: Modify mistralrs-core to support prefix cache internally

Rejected for Phase 1. Upstream contribution possible in Phase 3 after we have a
working implementation at the serving layer.

### B: Keep ax-engine KV cache, bridge to mistralrs-core

Rejected. Two independent KV caches for the same GPU memory region creates
aliasing risks and doubles memory usage.

### C: No prefix cache at all

Rejected for Phase 2. Prefix cache is the primary differentiator for server workloads.
Repeated system prompts are the norm in production API deployments.
