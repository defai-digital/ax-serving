# ADR-005: Metal Backend Strategy — Candle Kernels, No Custom Shaders

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

---

## Context

ax-engine invested heavily in custom Metal compute kernels:
- `attention.metal` — FA2 prefill, SDPA decode, hd128/hd256 variants
- `dequant.metal` — Q4_K, Q6_K, Q8_0 simdgroup_matrix GEMM, half A/B tiles
- `elementwise.metal` — batched RMSNorm, RoPE, SiLU/GELU, KV append

Despite this investment, the custom kernels reach only ~70% of mistral.rs (which uses
Candle's Metal kernels). The gap persists because:

1. Candle's `candle-metal-kernels` crate has been continuously tuned by the HuggingFace
   team and community across many hardware profiles
2. simdgroup_matrix occupancy on Apple Silicon requires careful BM/BN/BK tile selection
   that Candle has already optimized
3. ax-engine's FA2/SDPA rewrites were attempted (ADR documented as "REJECTED" in
   ax-engine MEMORY.md) due to structural underutilization

**Rejected ax-engine kernel experiments (not to repeat)**:
- Pair gate+up kernels: register pressure → regression
- simd_sum batch (K-parallel): 3–5× regression vs tiled GEMM
- FA2 prefill attention: serial score loop, half threads idle
- SDPA decode: single-threaded merge bottleneck
- TG=128 for Gemma3 head_dim=256: 6–37% regression vs TG=256
- N_DST=4 decode matvec at TG=32: -12% regression

---

## Decision

ax-serving uses **Candle Metal kernels exclusively**. No custom `.metal` shader files
are included in the ax-serving codebase.

The `ax-metal` crate from ax-engine is **not ported** to ax-serving.

Metal acceleration is provided through the `candle-metal-kernels` crate, surfaced via
mistralrs-core's Pipeline implementations.

### Feature Flag

```toml
# ax-serving-engine/Cargo.toml
[features]
default = ["metal"]
metal = ["mistralrs-core/metal"]
cpu-only = []   # for CI / non-Metal environments
```

### Backend Selection (runtime)

```
AXS_BACKEND=auto    # default: Metal if available, else CPU
AXS_BACKEND=metal   # force Metal (error if unavailable)
AXS_BACKEND=cpu     # force CPU
```

### Benchmark Gate

CI runs `ax-serving-bench bench` after every merge. Perf regression > 5% on any
model/quant combo blocks merge. Baseline stored in `benchmarks/baseline.json`.

---

## Lessons Learned from ax-engine (Must Not Repeat)

The following were tested in ax-engine and found to be regressions or neutral. They
are documented here to prevent re-investigation in ax-serving:

| Optimization | Result | Root Cause |
|---|---|---|
| FA2 prefill attention kernel | Regression | Serial score loop, half threads idle |
| SDPA decode attention kernel | -7% | Single-threaded merge step |
| Pair gate+up kernels | Regression | Register pressure at TG=256 |
| simd_sum K-parallel batch | 3–5× regression | Arithmetic intensity 1.9 vs 51 FLOP/byte |
| BK=32 K-tiles | -4 to -18% | Less K-reuse than BK=64 |
| TG=128 for head_dim=256 | -6 to -37% | Insufficient parallelism |
| N_DST=4 decode matvec at TG=32 | -12% | 4× weight bandwidth, x already cached |
| LTO / codegen-units=1 | Neutral | GPU-bound; compile flags don't matter |
| Precomputed f16 weights | Flat/regressive | Dense f16 reads 4× bytes; loses L2 locality |
| Fused norm+f16 cast | -19 to -31% | Driver overlaps separate dispatches anyway |

These learnings apply to the Apple UMA architecture and are unlikely to change with
the Candle kernel approach since the GPU hardware is identical.

---

## Kernel Improvement Path

If specific Candle Metal kernels need optimization:

1. **Profile first**: Use `ax-serving-bench profile` to identify the specific kernel
2. **Reproduce in isolation**: Write a minimal Metal test harness
3. **Contribute upstream**: Submit PR to `candle-metal-kernels` with benchmark data
4. **Temporary fork**: If upstream response is slow and the fix is critical, fork
   `candle` and update the git dependency; plan to merge within 90 days

We **do not** maintain a private kernel library in ax-serving.

---

## Consequences

### Positive

- **Zero custom shader maintenance** — no `.metal` files to update when Metal SDK changes
- **Upstream improvements flow automatically** — Candle kernel updates benefit ax-serving
- **Faster time-to-value** — no kernel tuning phase; start with production-ready kernels
- **Community support** — Candle Metal bugs are fixed by the HuggingFace community

### Negative

- **Less control** — if Candle introduces a kernel regression, we must wait for upstream fix
  or fork (mitigated by benchmark CI gate)
- **Feature lag** — new quantization types (e.g., Q3_K, FP8) depend on Candle support

---

## Alternatives Considered

### A: Port ax-engine custom .metal shaders to ax-serving

Rejected. Custom kernels are at 70% of Candle's performance. Porting them would
continue that 30% deficit. The engineering time is better spent on serving-layer features.

### B: Hybrid: Candle for prefill, ax-engine kernels for decode

Rejected. The decode gap in ax-engine (2× behind llama.cpp) is also present. No
compelling reason to maintain custom decode kernels at worse performance.
