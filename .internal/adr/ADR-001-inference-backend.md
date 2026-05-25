# ADR-001: Use mistralrs-core as the Inference Backend

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

---

## Context

ax-engine implemented a custom inference stack: Metal GPU kernels in `.metal` files, Rust
compute primitives (`compute/` module), a Backend trait dispatching between CPU
(Accelerate/NEON) and Metal, and quantization kernels (Q4_K, Q6_K, Q8_0).

After more than 100 optimization cycles — simdgroup_matrix GEMM, f16 A/B tiles, half
dequant params, f16 KV cache, FA2/SDPA attention, BN=32/64 tile variants, pair kernels,
N-tail boundary kernels, precomputed f16 weights, Q8_0 batch matmul — the engine
reaches **~70% of mistral.rs throughput** on LLaMA 3.1 8B Q4_K_M on Apple M3.

mistral.rs (archived at `ax-engine/archived/mistral.rs/`) has proven performance:
- LLaMA 3.1 8B already **beats llama.cpp** for both decode and prefill
- Candle Metal kernels (simdgroup_matrix, fused dequant+matmul) are production-tested
- Supports 20+ model architectures; GGUF + safetensors; ISQ quantization

The gap is not closure-able within a reasonable engineering timeframe without essentially
rewriting the inference path to match what Candle already provides.

---

## Decision

ax-serving uses **mistralrs-core** as its inference backend. The custom ax-engine
Metal kernels, `ax-metal` crate, and `compute/` module are **not ported** to ax-serving.

The `ax-serving-engine` crate is a thin adapter layer:

```
ax-serving-engine/
  src/
    lib.rs          — InferenceBackend trait definition
    backend.rs      — MistralrsBackend: impl InferenceBackend
    thermal.rs      — Port of ax-engine's ThermalMonitor
    memory.rs       — MemoryBudget check (sysctl-based)
```

`MistralrsBackend` wraps mistralrs-core's `Pipeline` trait. Model loading uses
`GgufModelBuilder` for .gguf files or `TextModelBuilder` for safetensors. Forward
passes route through mistralrs-core's event-loop engine.

---

## Consequences

### Positive

- **Immediate performance parity with llama.cpp** — mistral.rs already achieves this
- **20+ architectures for free** — LLaMA, Mistral, Qwen, Gemma, DeepSeek, Phi, etc.
- **Zero custom Metal code** — Candle Metal kernels maintained upstream
- **GGUF + safetensors** — broader model compatibility
- **ISQ quantization** — quantize safetensors at load time (no pre-conversion needed)
- **Smaller codebase** — drop ~3000 lines of custom Metal shaders + 6000 lines of
  compute primitives

### Negative

- **External dependency risk** — mistralrs-core API can change; mitigated by pinning
  to a specific git rev and committing a fork if upstream breaks
- **Less control over kernel optimization** — if we need a specific kernel fix, we
  must contribute upstream to Candle (acceptable; preferred over private forks)
- **Candle tensor allocations** — Candle uses heap tensors internally, not the
  ax-engine arena approach. Long-run memory behavior must be validated by soak test.
- **Coverage gaps** — model families not yet in mistralrs-core (GPT-J, MiniMax,
  GLM-5, full DeepSeek 671B) require the llama.cpp fallback defined in ADR-010.
  The `InferenceBackend` trait isolates all callers from this detail.

### Neutral

- ax-engine's `ForwardPass` trait is replaced by `InferenceBackend` — the serving
  layer gains a cleaner boundary
- ax-engine's arch_registry (LlamaForward/Gemma3Forward/Qwen3Forward) is replaced by
  mistralrs-core's pipeline auto-detection — less code to maintain

---

## Alternatives Considered

### A: Continue optimizing ax-engine Metal kernels

Rejected. 100+ cycles with diminishing returns. The architectural gap (Candle's
thread-group occupancy tuning, simdgroup reuse patterns) would require rewriting the
kernel strategy from scratch — at which point we are reimplementing Candle.

### B: Fork Candle and pull it into ax-engine

Rejected. Candle is a large dependency (candle-core, candle-nn, candle-metal-kernels).
Forking it adds more maintenance burden than using mistralrs-core as a dependency.

### C: Hybrid: mistralrs-core for prefill, ax-engine for decode

Rejected. Integration complexity is high (two tensor systems, two KV cache formats).
The decode gap vs llama.cpp is also present in ax-engine (~2× gap), so the hybrid
gives no meaningful advantage over pure mistralrs-core.

---

## Backend Extensibility

mistralrs-core is the default and primary backend. For model families it does not
support, ax-serving implements a fallback chain via the `InferenceBackend` trait.
See **ADR-010** for the full multi-backend selection strategy.

## Version Pinning Strategy

```toml
# Cargo.toml
mistralrs-core = {
  git = "https://github.com/EricLBuehler/mistral.rs",
  rev = "PINNED_COMMIT_SHA",
  features = ["metal"]
}
```

If upstream breaks compatibility: create a fork at
`github.com/ax-team/mistral.rs` and update the git reference.
Evaluate upstreaming fixes before diverging permanently.
