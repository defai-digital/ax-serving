# Competitive category map

Compare systems at the layer where they make decisions.

| Category | Representative systems | AX relationship |
| --- | --- | --- |
| Apple/local inference runtime | AX Engine, llama.cpp | AX Engine is the execution-layer comparison. |
| NVIDIA inference engine | vLLM, SGLang, TensorRT-LLM | Runs as a Dynamo backend; AX does not replace its token scheduler. |
| NVIDIA distributed inference system | NVIDIA Dynamo | Owns worker routing, KV, disaggregation, planner, scaling, and backend lifecycle inside an NVIDIA domain. |
| Cross-domain federation/control plane | Internal fleet platforms, multi-provider AI gateways | AX Serving competes here on private Mac + PC + Thor governance and safe semantic routing. |
| Agent application framework | LangGraph, AutoGen, CrewAI and application stacks | Outside AX scope; they call the inference API. |

## Intended wedge

- private fleets with useful Apple Silicon plus NVIDIA capacity;
- separately operated NVIDIA PC and Thor domains;
- central tenant/privacy/locality/cost/SLO policy above runtimes;
- explicit model identity and fail-closed equivalence across artifact formats/runtimes;
- conservative retry, decision audit/replay, HA state, drain, and rollout;
- environments that want upstream Dynamo without making it the global Mac/tenant policy plane.

## Where another system is better

- one NVIDIA Dynamo domain that already satisfies all traffic and policy: use Dynamo directly;
- CUDA kernels/token scheduling: use and tune the backend runtime;
- NVIDIA worker/KV routing and scaling: use Dynamo;
- one Mac/AX Engine endpoint that already satisfies all traffic and policy: use AX Engine directly;
- end-user desktop/chat UX: use a desktop/local application;
- agent planning/tools/memory: use an agent framework;
- standardized Kubernetes inference routing that already meets all identity/policy needs: integrate
  with that platform instead of adding AX.

## Competitive evidence

AX must not claim superiority from a feature checklist. It must report:

- direct runtime/domain versus through-AX overhead;
- policy correctness and duplicate-commitment behavior under faults;
- cost/load, goodput, tail latency, privacy/locality, or operator-workflow value;
- routing regret and quality-floor violations for adaptive policies;
- exact AX, Dynamo, backend, image, model, tokenizer/template/quantization, and hardware identities.

The Dynamo adapter and Thor domain are not currently certified; public comparisons must preserve
that status.
