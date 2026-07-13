# AX Serving positioning

AX Serving is a runtime-neutral inference fleet control plane for private and
hybrid infrastructure.

It is positioned above inference engines:

- AX Engine owns MLX execution on Apple Silicon;
- vLLM or SGLang owns CUDA execution;
- AX Serving publishes logical models, admits requests, selects compatible
  endpoints, and operates the fleet.

This is not a claim that AX Serving is faster than an inference runtime or is a
drop-in replacement for vLLM. The gateway complements runtimes and must prove
its own value through overhead, availability, goodput, and operational safety.

## Primary user

The primary user is a platform or infrastructure team that already has more
than one runtime endpoint, hardware pool, model deployment, trust boundary, or
availability requirement. Team size and model parameter count are not reliable
qualification rules; measured workload and operational complexity are.

## Core promise

One authenticated OpenAI-compatible endpoint can operate certified AX
Engine/MLX and CUDA runtime deployments without linking their SDKs or silently
treating unlike model artifacts as equivalent.

## Differentiators to prove

- runtime-neutral protocol and portable gateway dependency boundary;
- explicit deployment identity and fail-closed cross-runtime equivalence;
- conservative pre-commit retry and cancellation-safe streaming;
- active-active fleet state and capacity fencing;
- tenant admission, drain, rollout, diagnostics, and audit;
- a simple one-gateway profile as well as an HA profile.

These are product hypotheses until conformance and production-envelope gates
are retained for a release. Marketing must distinguish implemented source,
mock-tested behavior, live runtime certification, and measured performance.

## Anti-scope

AX Serving is not:

- a consumer chat application;
- a model converter or artifact registry;
- a tensor, pipeline, prefill/decode, or KV-cache splitter;
- a token scheduler, batching engine, or hardware kernel project;
- a reason to hide quantization/tokenizer/template differences;
- a hyperscale claim without matching validation evidence.

## Message hierarchy

1. “AX Serving manages inference runtimes; it is not one.”
2. “Hybrid means one fleet of MLX and CUDA pools, one endpoint per attempt.”
3. “AX Engine versus llama.cpp is the engine comparison.”
4. “AX Serving overhead and mixed-fleet behavior are serving comparisons.”
5. “vLLM remains the CUDA execution runtime behind AX Serving.”

## Evidence policy

Public capability language requires a tagged release and passing contract
tests. Runtime compatibility language additionally requires a pinned live
runtime/image result. Performance and production language requires complete
raw artifacts for the PRD envelope and soak gates.

The current source tree must therefore be described as an implemented
runtime-neutral architecture awaiting live hybrid and production-envelope
certification, not as an already certified production service.
