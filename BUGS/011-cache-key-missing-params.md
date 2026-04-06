# BUG-011: Cache Key Omits Inference-Affecting Parameters

**Severity:** Medium
**File:** `crates/ax-serving-api/src/rest/routes.rs`
**Lines:** 2664–2676 (`CacheKeyPayload`), 2717–2731 (`TextCacheKeyPayload`)
**Status:** ✅ FIXED (2026-03-28)

## Description

The cache key structs used for response caching omit several fields from the request schema that directly affect output:

**Missing from both `CacheKeyPayload` and `TextCacheKeyPayload`:**
- `stop` (stop sequences) — completely changes output
- `frequency_penalty` — changes token probabilities
- `presence_penalty` — changes token probabilities
- `grammar` — constrains output format
- `response_format` — forces JSON output
- `mirostat` / `mirostat_tau` / `mirostat_eta` — changes sampler behavior

**Additionally missing from `CacheKeyPayload`:**
- `tools` — affects tool-call generation
- `tool_choice` — affects tool-call behavior

```rust
#[derive(Serialize)]
struct CacheKeyPayload<'a> {
    version: &'static str,
    resolved_model_path: &'a str,
    resolved_model_arch: &'a str,
    messages: Vec<NormalizedMessage>,
    temperature: String,
    top_p: String,
    min_p: Option<String>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
    repeat_penalty: String,
    // Missing: stop, frequency_penalty, presence_penalty,
    //          grammar, response_format, mirostat*,
    //          tools, tool_choice
}
```

## Impact

**Silent data correctness bug.** Request A sends `grammar: "root ::= ..."` and gets a cached response. Request B sends no `grammar` but identical messages/temperature/top_p. The cache key matches, so Request B receives Request A's grammar-constrained output. Similarly, different stop sequences or sampler settings produce incorrect cache hits.

## Fix

Add all inference-affecting fields from `ChatCompletionRequest` and `CompletionRequest` into both cache key structs.

## Fix Applied

Added all missing inference-affecting fields to both `CacheKeyPayload` (v2→v3) and `TextCacheKeyPayload` (v1→v2): `stop` (sorted Vec), `frequency_penalty`, `presence_penalty`, `grammar`, `response_format`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `logprobs`, `top_logprobs`. Chat cache key also adds `tools` and `tool_choice`. Float fields normalized at 4dp. Version bump ensures no stale cache hits from old keys.
