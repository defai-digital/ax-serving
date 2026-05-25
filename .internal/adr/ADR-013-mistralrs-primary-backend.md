# ADR-013: Re-enable mistralrs v0.7.0 as Primary Inference Backend

**Status**: Accepted
**Date**: 2026-03-10
**Deciders**: AutomatosX Team (DEFAI Private Limited)
**Supersedes**: Temporary llama.cpp-only policy in `config/backends.yaml` (established 2026-03-02)
**Extends**: ADR-001 (mistralrs as backend), ADR-010 (multi-backend fallback chain)

---

## Context

ADR-001 designated mistralrs-core as the primary inference backend. In practice,
mistralrs was **disabled** in early March 2026 due to two production-blocking issues
measured on Apple Silicon M-series hardware:

1. **Catastrophic prefill performance**: PP-256 throughput was ~50 t/s vs. llama.cpp's
   ~304 t/s — a 6× gap making interactive use non-viable.
2. **Candle Metal buffer memory leak** (`huggingface/candle#2271`): temporary Metal
   buffers were never released (missing `autoreleasepool`), causing RSS growth that
   eventually triggered swap and collapsed throughput from ~100 t/s to ~0.6 t/s during
   sustained serving. Fatal in production.

All inference was temporarily routed to `llama-server` subprocess
(`config/backends.yaml: default_backend: llama_cpp`) pending upstream fixes.

### What has changed (mistralrs v0.7.0, January 2026)

| Issue | Status in v0.7.0 |
|---|---|
| Metal buffer memory leak (Candle#2271) | **Fixed** — Candle 0.9.2 upgrade adds `autoreleasepool` wrapping |
| Prefill performance collapse | **Fixed** — PP-256 now ~606 t/s vs. llama.cpp ~737 t/s (18% gap, down from 6×) |
| Decode performance regression (PR#1580 — halved t/s) | **Fixed** — PR#1575 restores Metal-specific path |
| PagedAttention KV cache GPU fallback to swap | **Fixed** — PR#1506 uses `StorageModePrivate` |
| Qwen3 MoE near-single-threaded (<5 t/s) | **Fixed** — PA memory fix + GGUF arch support (PR#1488) |

### Benchmark comparison (LLaMA 3.1 8B Q8_0, M-series, mistralrs v0.7.0)

| Metric | mistralrs v0.7.0 | llama.cpp | Gap |
|---|---:|---:|---|
| Decode TG-256 | ~38 t/s | ~39 t/s | ~3% slower |
| Prefill PP-256 | ~606 t/s | ~737 t/s | ~18% slower |
| Prefill PP-256 (vs. disabled period) | ~606 t/s | — | was 49 t/s (12× improvement) |

The remaining gap (~3% decode, ~18% prefill) is within the acceptable range for the
primary backend, especially given that the serving layer overhead from the llama-server
subprocess HTTP round-trip itself costs more than this on every request.

---

## Decision

**Re-enable mistralrs as the primary backend** for model families it supports natively.
llama.cpp remains the backend for architectures mistralrs does not support, and as the
`auto`-mode fallback for unknown architectures.

### Updated routing table (`config/backends.yaml`)

| Family | Backend | Rationale |
|---|---|---|
| `llama` | `native` | LLaMA 3.x — mistralrs primary |
| `qwen` | `native` | Qwen 2/3 dense — mistralrs primary; MoE fixed in v0.7.0 |
| `gemma` | `native` | Gemma 2/3 — mistralrs primary |
| `mistral` | `native` | Mistral — mistralrs primary |
| `phi` | `native` | Phi-2/3/4 — mistralrs primary |
| `starcoder` | `native` | Starcoder2 — mistralrs primary |
| `deepseek` | `auto` | Dense distills → native; 671B OOMs → propagates (use `backend_hint: llama_cpp` for 671B) |
| `glm` | `auto` | GLM-4 → native; GLM-5 unknown arch → falls back to llama.cpp |
| `gptj` | `llama_cpp` | No mistralrs support |
| `gpt_neox` | `llama_cpp` | No mistralrs support |
| `falcon` | `llama_cpp` | No mistralrs support |
| `minimax` | `llama_cpp` | No mistralrs support |
| default | `auto` | Try native first; fall back to llama.cpp on unsupported arch |

### DeepSeek 671B note

`auto` mode tries native first. For 671B models, native will return OOM
(`insufficient memory`) which `is_unsupported_model_error()` correctly classifies
as a real failure (not a fallback trigger). Users loading 671B models must set
`backend_hint: llama_cpp` in the `POST /v1/models` request body. This is documented
in `config/serving.example.yaml` and the runbook.

### Version pin

Cargo.toml already pins `mistralrs = "0.7.0"` and `mistralrs-core = "0.7.0"`.
No version change required.

---

## Consequences

### Positive

- **Eliminates subprocess HTTP overhead** for all native-routed models — TTFT and
  streaming chunk latency improve proportionally to the round-trip cost (~5–20 ms
  per request depending on response size).
- **Recovers designed architecture** — ADR-001 intent fulfilled; llama.cpp returns
  to its intended fallback role.
- **Better resource utilisation** — mistralrs manages KV cache natively in UMA;
  llama-server subprocess is not spawned for native families.
- **Continuous batching** — mistralrs supports multiple concurrent requests in a
  single forward pass (unlike the subprocess model); multi-tenant throughput improves.

### Negative

- **18% prefill gap vs. llama.cpp** remains for native families. Acceptable; the
  subprocess overhead eliminated more than compensates at serving-layer level.
- **Regression risk** — mistralrs has had Metal-specific regressions in 2025
  (decoded speed halved, memory leak). Mitigation: pin to `v0.7.0`, run 24h soak
  before any version bump, monitor `axs_rss_bytes` metric in production.
- **GLM-5 and unknown arches** go through `auto` with a native attempt first — adds
  one failed load attempt latency (~100–500 ms) before llama.cpp fallback. Acceptable
  for model load time (one-time cost).

### Neutral

- llama.cpp subprocess infrastructure remains fully operational for fallback families.
  `llama-server` on PATH (via `brew install llama.cpp`) is still required.

---

## Validation Gate

Before this ADR is considered fully implemented, the following must pass on an M3 Pro:

1. `cargo test --workspace` — all unit tests green
2. `AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test orchestration` — integration green
3. 24h soak: `cargo run -p ax-serving-bench --release -- soak -m ./models/llama3.gguf --duration_min 1440`
   - RSS drift < 5% (memory leak check)
   - P95 latency drift < 5%
4. `axs_rss_bytes` metric must remain stable under the soak (no unbounded growth)

---

## Alternatives Considered

### A: Keep llama.cpp as default, re-enable native only for specific families

Rejected. The `auto` fallback mode already handles unknown families. Setting
`native` explicitly for known-good families is cleaner and avoids the failed-load
latency penalty on every model load.

### B: Wait for further mistralrs improvements (e.g. close the 18% prefill gap)

Rejected. The 18% prefill gap is smaller than the subprocess HTTP overhead we are
currently paying on every request for native-capable models. Waiting provides
negative expected value.

### C: Evaluate mlx as a third backend

Deferred. mlx achieves ~44 t/s decode vs. ~39 t/s llama.cpp on Apple Silicon —
the fastest option. However, mlx has no Rust bindings (Python-only), making
integration significantly more complex. Revisit in Phase 3 if native performance
becomes the primary bottleneck.
