# ADR-010: Multi-Backend Inference Selection with Fallback Chain

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)
**Supersedes**: Nothing (extends ADR-001)

---

## Context

ADR-001 established mistralrs-core as the inference backend. However, the target
model matrix has since expanded to include:

- **GPT-family OSS** (GPT-J, GPT-NeoX, Falcon) — older architectures; mistralrs-core
  coverage is partial
- **GLM-5** (Zhipu AI) — mistralrs-core supports GLM-4; GLM-5 support is not yet confirmed
- **MiniMax** (MiniMax AI) — sparse MoE; mistralrs-core has no current support
- **DeepSeek-V3/R1 full 671B** — does not fit in UMA; requires CPU+GPU offload that
  mistralrs-core does not implement for Apple Silicon

mistralrs-core is the right primary backend. But for a serving platform targeting
7 model families, single-backend coverage is insufficient. A fallback chain allows
ax-serving to serve models that mistralrs-core does not (yet) support without
blocking the roadmap on upstream PRs.

---

## Decision

ax-serving implements a **backend selection registry** with the following fallback chain:

```
Priority 1: mistralrs-core   (MistralrsBackend)
Priority 2: ax-engine        (AxEngineBackend, for custom model adaptors)
Priority 3: llama.cpp        (LlamaCppBackend, subprocess + C API bridge)
```

The `InferenceBackend` trait (defined in ADR-001) is the sole abstraction. All
serving-layer code (`ServingLayer`, REST handlers, gRPC service, C API shim) uses
`Arc<dyn InferenceBackend>` and is unaware of which backend is active.

Selection is performed at **model load time**, not at server startup.

---

## Backend Capability Registry

The registry maps `(model_arch, quant_type)` to the preferred backend:

```rust
// ax-serving-engine/src/capability.rs (new)
pub fn preferred_backend(arch: &str, quant: QuantType) -> BackendKind {
    match arch {
        // mistralrs-core native support
        "llama" | "mistral" | "qwen2" | "qwen3" | "gemma" | "gemma3"
        | "phi3" | "starcoder2" | "deepseek2" | "glm4" => BackendKind::Mistralrs,

        // llama.cpp fallback (no mistralrs-core support as of 2026-03-01)
        "gpt2" | "gptj" | "gpt_neox" | "falcon"
        | "minimax" | "glm5" => BackendKind::LlamaCpp,

        // Unknown: try mistralrs first, fall back to llama.cpp
        _ => BackendKind::Auto,
    }
}
```

`BackendKind::Auto` attempts `MistralrsBackend::load_model`; if it returns
`Err(UnsupportedArchitecture)`, retries with `LlamaCppBackend`.

---

## Backend Descriptions

### Priority 1: MistralrsBackend (mistralrs-core)

- **Coverage**: LLaMA 3.x, Mistral, Qwen 2/3, Gemma/Gemma3, DeepSeek V2/V3
  distills, GLM-4, Phi 3/4, Starcoder2
- **Performance**: Best — Candle Metal kernels, native UMA paged KV
- **Quantization**: GGUF Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16; ISQ
- **Activation**: Default for all models in the capability table above

### Priority 2: AxEngineBackend (ax-engine adapter)

- **Coverage**: Custom architectures where we write a bespoke `InferenceBackend`
  implementation for models not yet in mistralrs-core
- **Use case**: When a model family is strategically important and the upstream
  PR timeline is unacceptable; faster than waiting for llama.cpp GGUF support too
- **Implementation**: `ax-serving-engine/src/backends/axengine.rs` — wraps
  ax-engine's forward pass implementations (LlamaForward etc.) when porting is needed
- **Activation**: Explicit opt-in via `AXS_BACKEND_PREFERENCE=ax-engine,mistralrs,llama.cpp`
  or when a model arch is registered as `BackendKind::AxEngine`

> Note: The "Do Not Port" list in CLAUDE.md refers to custom Metal shaders and
> compute primitives. Porting a *forward pass adaptor* (not Metal kernels) is
> permitted for model families not covered by mistralrs-core.

### Priority 3: LlamaCppBackend (subprocess bridge)

- **Coverage**: Any model with a GGUF file that llama.cpp supports
- **Activation**: Models in the `llama.cpp fallback` column of MMS §4; any arch
  returning `BackendKind::LlamaCpp` from the capability registry; also activated
  for CPU-offload mode when model exceeds UMA budget
- **Implementation**: `ax-serving-engine/src/backends/llamacpp.rs` — manages a
  `llama-server` subprocess via stdin/stdout JSON-RPC, or via the `libllama.dylib`
  C API (ax-serving-shim in reverse — loading the system llama.cpp dylib)
- **Memory**: Supports `--n-gpu-layers N` to offload N layers to GPU; remaining
  layers run on CPU. Enables 671B MoE models on Ultra SKUs.
- **Performance**: ~0.5–0.8× mistralrs for supported models due to subprocess overhead

---

## Selection Logic (Runtime)

```
load_model(path, config) →
  1. Parse GGUF model type field → arch string
  2. Lookup preferred_backend(arch, quant)
  3. If Auto or Mistralrs:
       try MistralrsBackend::load_model(path, config)
       if Err(UnsupportedArch) → fall through to step 4
       else return handle
  4. If AxEngine or fallthrough:
       try AxEngineBackend::load_model(path, config)
       if Err(UnsupportedArch) → fall through to step 5
       else return handle
  5. If LlamaCpp or fallthrough:
       LlamaCppBackend::load_model(path, config)  # returns error if llama.cpp not found
```

The `ModelMetadata` returned on successful load includes `backend: BackendKind`
so the registry and health endpoint can report which backend is active per model.

---

## Configuration

```bash
# Override automatic selection for all models
AXS_BACKEND_PREFERENCE=mistralrs,llama.cpp   # (default) skip ax-engine
AXS_BACKEND_PREFERENCE=mistralrs             # never fall back; fail if unsupported

# llama.cpp backend settings
AXS_LLAMA_CPP_BIN=/usr/local/bin/llama-server   # path to llama-server binary
AXS_LLAMA_CPP_GPU_LAYERS=auto                   # auto = fit as many as UMA allows
AXS_LLAMA_CPP_GPU_LAYERS=32                     # explicit layer count

# ax-engine backend (only needed if explicitly enabling)
AXS_BACKEND_PREFERENCE=ax-engine,mistralrs,llama.cpp
```

In `serving.toml`:

```toml
[backend]
preference = ["mistralrs", "llama.cpp"]   # list, tried in order
llama_cpp_bin = "/usr/local/bin/llama-server"
llama_cpp_gpu_layers = "auto"
```

---

## Model-Backend Capability Table (Phase 1)

| Model Family | mistralrs | ax-engine | llama.cpp | Notes |
|---|---|---|---|---|
| LLaMA 3.x (1B–8B) | Primary | — | Fallback | Recommend mistralrs |
| LLaMA 3.1 70B | Primary | — | Fallback | Pro/Max/Ultra only |
| LLaMA 3.1 405B | — | — | CPU offload | Ultra only |
| GPT-2 | Partial | — | Primary | llama.cpp preferred |
| GPT-J / NeoX | — | — | Primary | mistralrs lacks arch |
| Falcon | — | — | Primary | mistralrs lacks arch |
| Gemma3 (1B–27B) | Primary | — | Fallback | |
| Qwen3 (0.6B–32B) | Primary | — | Fallback | |
| Qwen3 MoE | Primary | — | Fallback | |
| GLM-4-9B | Primary | — | Fallback | |
| GLM-5 | TBD | TBD | Fallback | Fallback until confirmed |
| DeepSeek-R1 Distill (≤14B) | Primary | — | Fallback | |
| DeepSeek-V3/R1 671B | — | — | CPU offload | Ultra only |
| MiniMax-Text-01 | — | TBD | CPU offload | mistralrs support TBD |

---

## Consequences

### Positive

- **Full target model coverage** — all 7 model families served, even when mistralrs-core
  lags upstream
- **No serving-layer changes** — all backends satisfy `InferenceBackend`; REST/gRPC/C API
  are unaffected
- **CPU-offload path** — enables very large models (671B) on Apple Silicon Ultra
- **Future-proof** — adding a new backend requires only a new `impl InferenceBackend`
  and a `preferred_backend()` entry; no protocol changes

### Negative

- **Subprocess complexity** — llama.cpp backend requires managing a child process,
  health-checking it, and bridging JSON-RPC; adds ~500 lines of backend code
- **llama.cpp version pinning** — like mistralrs-core, must pin llama.cpp to a specific
  release to avoid ABI/API surprises
- **Benchmark gap** — llama.cpp backend will underperform mistralrs on shared model
  families; must track separate baselines in BMS (see BMS §4)

### Neutral

- `ModelMetadata` gains a `backend` field — transparent to clients (not in REST/gRPC
  response schema unless explicitly requested via `/health` or `GetMetrics`)

---

## Alternatives Considered

### A: mistralrs-only; block roadmap on upstream PRs

Rejected. GLM-5 and MiniMax are strategic targets with unknown upstream timelines.
Blocking the Phase 2 roadmap on external contributors is unacceptable.

### B: Embed llama.cpp as a Rust library (via llama-sys)

Partially accepted for Phase 2 investigation. The subprocess approach is simpler
for Phase 1 and avoids linking complexity. If latency from subprocess overhead
becomes measurable (> 5 ms), evaluate `llama-sys` linking.

### C: Implement all missing architectures natively in ax-serving

Rejected for full 671B models — the memory management and operator coverage
required would replicate Candle + mistralrs-core effort. Reserve for specific
small-model adaptors via `AxEngineBackend` only.

---

## Version Pinning

```toml
# llama.cpp: pin to a specific release tag (track mistralrs parity commits)
# ax-serving-engine/Cargo.toml or scripts/install-llama.sh
LLAMA_CPP_VERSION=b5120  # update when new GGUF architectures are needed
```
