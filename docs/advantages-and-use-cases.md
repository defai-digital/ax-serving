# AX Serving use cases and trade-offs

AX Serving is useful when one client/API policy must govern more than one independently operated
inference domain.

## Strong fits

- A private fleet containing Mac AX Engine and NVIDIA Dynamo capacity.
- NVIDIA PC and Thor devices that must remain separate performance/failure domains.
- Tenant, privacy, residency, locality, budget, or SLO policy that no one runtime can enforce
  globally.
- Logical models backed by multiple explicitly certified deployments.
- Central admission, audit, drain, rollout, diagnostics, and HA across domains.
- A need to use installed Mac/edge capacity without exposing runtime addresses to applications.

## Poor fits

- One model on one NVIDIA deployment: use Dynamo directly.
- One model on one Mac: use AX Engine directly.
- A need for CUDA token scheduling or KV-aware NVIDIA worker routing: use Dynamo and its backend.
- A need to split a graph, KV cache, or prefill/decode phase across Mac, PC, and Thor.
- An agent application framework with tools, MCP, memory, sandboxes, and workflow state.
- A business case that cannot pass a cost/load, availability, privacy/locality, or operator-workflow
  value gate.

## Architectural advantages

- The gateway links no AX Engine, Dynamo, MLX, Metal, CUDA, or backend runtime SDK.
- AX chooses a domain while Dynamo retains NVIDIA-local optimization.
- PC and Thor have separate identities, artifacts, calibration, qualification, and rollout.
- Hard eligibility and explicit equivalence prevent semantic failover by model name alone.
- Retry is conservative and has one owner at each layer.
- Decisions are versioned, bounded, auditable, and replayable without retaining prompts.
- Small evaluation and active-active HA profiles share one public API.

## Trade-offs

- Federation adds a hop, an adapter, and another operational control plane.
- Coarse domain telemetry cannot optimize NVIDIA workers as well as Dynamo can; it should not try.
- Explicit identity/equivalence and upstream compatibility manifests require operational discipline.
- Cross-domain quality labels and value evidence are workload-specific.
- Thor requires separate live qualification despite generic ARM64/Blackwell prerequisites.
- If the fleet stays homogeneous, the added complexity has little value.

## Correct comparisons

| Question | Comparison |
| --- | --- |
| Is Mac execution competitive? | AX Engine direct versus the relevant local/runtime baseline under matched artifacts. |
| Is NVIDIA execution efficient? | Tune and benchmark the chosen Dynamo/backend graph directly. |
| What does AX cost? | The same Mac or Dynamo domain directly versus through AX Serving. |
| Does federation help? | Single-domain policy versus mixed-domain policy under normal, saturated, drain, outage, privacy, and budget scenarios. |
| Does AX replace Dynamo/vLLM? | No. Dynamo and its backend remain the NVIDIA execution system. |

See the [runtime responsibility inventory](contracts/ax-serving-runtime-responsibility-inventory.md)
for the public architecture and ownership boundaries.
