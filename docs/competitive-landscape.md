# Competitive category map

This document avoids version-specific feature claims. Recheck official project
documentation at the time of any external comparison.

## Compare matching layers

| Category | Representative systems | AX relationship |
| --- | --- | --- |
| Local/runtime engine | AX Engine, llama.cpp | AX Engine is compared here under matched model and artifact contracts. |
| CUDA serving runtime | vLLM, SGLang, TGI | AX Serving manages certified endpoints; it does not replace their token schedulers. |
| Local application/runtime UX | Ollama, LM Studio | Usually a different buyer and operating model. |
| Gateway/control plane | Inference gateways, Kubernetes inference routing, internal fleet platforms | AX Serving competes here on safe mixed-runtime operations. |

Feature matrices become misleading when an engine's kernel feature is compared
with a gateway's routing feature. Every comparison should state which process
owns tokenization, batching, cache, lifecycle, admission, and fleet state.

## AX Serving's intended wedge

- private fleets that contain Apple Silicon and CUDA capacity;
- operators who want one API but cannot assume runtime artifacts are
  semantically equivalent;
- teams that need runtime-SDK isolation, conservative retries, drain, HA state,
  and explicit trust boundaries;
- environments where a small deployment should not require a large
  orchestration stack.

## Where another product is likely better

- one model on one machine with no fleet control: use the runtime directly;
- CUDA kernel and token-scheduler performance: use and tune vLLM/SGLang/TGI;
- desktop model discovery and end-user chat UX: use a desktop/local product;
- distributed graph execution across accelerators: use a runtime designed for
  that execution model;
- Kubernetes-native inference routing already standardized by the operator's
  platform: integrate with that control plane unless AX Serving's identity and
  mixed-runtime guarantees are materially needed.

## Comparison evidence

Engine comparisons require matched source revision, tokenizer, template,
sampler, token accounting, hardware, warmup, and quantization disclosure.
Serving comparisons require direct-runtime baseline, same endpoint through AX
Serving, and mixed-fleet fault/overload scenarios. Goodput, TTFT tails,
availability correctness, duplicate commitments, and overhead matter more than
an isolated best tokens-per-second number.

Do not publish “faster,” “production-grade,” scale, compatibility, or market
leadership claims from mock tests, incomplete runs, or null baselines.
