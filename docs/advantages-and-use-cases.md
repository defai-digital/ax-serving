# AX Serving use cases and tradeoffs

AX Serving is useful when one client endpoint must operate multiple inference
runtimes without moving runtime-specific scheduling into the gateway.

## Strong fits

- A private fleet containing AX Engine/MLX and vLLM or SGLang/CUDA pools.
- Multiple immutable runtime deployments published behind logical model names.
- Central authentication, tenant admission, drain, rollout, and diagnostics.
- Active-active gateways with Redis/Valkey-backed worker leases and capacity.
- Applications that need OpenAI-compatible chat, completion, embedding, and
  incremental SSE while runtimes remain independently deployable.
- Operators who need failover to stop at model/tokenizer/template/quantization
  equivalence boundaries.

## Poor fits

- One local model with no fleet operations: use AX Engine, llama.cpp, or the
  selected runtime server directly.
- A requirement to split one model or KV cache across MLX and CUDA: AX Serving
  routes whole attempts and does not implement distributed tensor execution.
- A need for CUDA token scheduling itself: use vLLM or SGLang; AX Serving can
  manage those endpoints but does not replace them.
- Hyperscale or feature claims that have not passed this project's published
  validation envelope.

## Architectural advantages

- The portable gateway does not link AX Engine, MLX, CUDA, or llama.cpp.
- Runtime adapters use a versioned protocol and runtime-authoritative
  readiness/inventory rather than gateway-side model parsing.
- Hard eligibility and explicit equivalence prevent best-effort routing from
  becoming silent semantic failover.
- Retry is conservative: one pre-commit retry only for connect failure or
  authenticated typed non-admission.
- Public, admin, control-plane, dispatch, and runtime credentials are distinct.
- Small deployments retain an in-memory profile; HA deployments add shared
  state without changing the public API.

## Tradeoffs

- An agent adds one network hop and an adapter certification obligation.
- Explicit identity and equivalence require operational discipline.
- Redis/Valkey becomes a production dependency for active-active gateways.
- AX Serving cannot optimize inside a runtime as deeply as that runtime's own
  scheduler, so telemetry must remain conservative.
- A passing source test suite is not live runtime or performance certification.

## Positioning

| Question | Correct comparison |
| --- | --- |
| Is MLX execution competitive with a local runtime? | Benchmark AX Engine versus llama.cpp under a matched artifact contract. |
| What is the cost of the gateway? | Compare the same runtime directly and through AX Serving. |
| Does a mixed fleet improve availability or cost? | Compare homogeneous and mixed fleets under drain/failure/overload scenarios. |
| Does AX Serving replace vLLM? | No. vLLM remains the CUDA execution and token-scheduling runtime. |

See the [quick start](../QUICKSTART.md),
[operations runbook](runbooks/multi-worker.md), and
[performance evidence guide](perf/service-tuning.md).
