# ADR-015: Agent-Aware Inference Fabric, Not Agent Orchestrator

| Field | Value |
| --- | --- |
| Status | Accepted |
| Decision date | 2026-07-14 |
| Owners | AX Serving maintainers |
| Scope | Public inference metadata, request profile, routing, shared affinity state, runtime agent, telemetry |
| Extends | [ADR-013](ADR-013-RUNTIME-NEUTRAL-HYBRID-INFERENCE-CONTROL-PLANE.md) |
| Related PRD | [Agent-aware inference fabric](../prd/PRD-AGENT-AWARE-INFERENCE-FABRIC.md) |
| Implementation | [Agent Session Contract v1 spec](../specs/TECH-SPEC-AGENT-SESSION-FABRIC-CONTRACT.md) |
| Peer decision | `/Users/akiralam/code/ax-engine/.internal/adr/ADR-001-AGENT-AWARE-INFERENCE-RUNTIME.md` |

## Context

AX Serving already classifies bounded request metadata, enforces tenant/priority/deadline policy,
filters runtime capabilities, derives tenant-scoped cache-affinity keys, performs inference-aware
endpoint selection, and safely proxies streaming requests through runtime agents. These are the
right foundations for agent workload optimization.

Agent turns repeatedly send long related prefixes and pause between calls. Generic load balancing
can route the next turn away from warm runtime state, causing avoidable prefill cost. A session hint
lets the gateway prefer an eligible prior worker and lets a compatible runtime retain useful prefix
state.

However, “agent runtime” can also imply planners, tool execution, durable tasks, sandboxes, and
memory. Those responsibilities would turn the gateway into an application framework and create
new credential, state, and security domains. ADR-013 deliberately keeps runtime execution outside
the gateway; the same discipline must apply to agent orchestration.

## Decision drivers

- Improve agent-loop latency without tokenizing prompts or owning KV state in the gateway.
- Preserve runtime neutrality across AX Engine, vLLM, SGLang, and future adapters.
- Make cache locality work in single-gateway and HA deployments.
- Keep failover safe because complete input is replayable.
- Protect raw session identifiers and prevent cardinality leaks.
- Reuse existing capability, admission, equivalence, retry, and fleet-state mechanisms.

## Decision

AX Serving will be an **agent-aware inference fabric**. It accepts Agent Session Contract v1,
derives a tenant-scoped opaque key, keeps a bounded soft worker binding, negotiates per-model
runtime support, and forwards normalized hints through the runtime agent.

### Ownership boundary

| AX Serving owns | Runtime owns | External harness owns |
| --- | --- | --- |
| Public auth and tenant session-key derivation | Tokenization and template semantics | Agent planner and loop |
| Request admission, priority, deadline | Batching and token scheduling | Tool/MCP/A2A interaction |
| Capability/equivalence filtering | KV/prefix cache contents and eviction | Credentials, approvals, sandbox |
| Soft session affinity and worker selection | Speculation and constrained output | Durable history, memory, checkpoints |
| Typed retry and stream commitment | Runtime-local cache diagnostics | Tasks, artifacts, handoffs |
| Fleet/HA affinity records and bounded telemetry | Model output/tool-call parsing | Business audit semantics |

### Session semantics

- The public `x-ax-session-id` is an opaque correlation input, not a database key exposed to
  runtimes. The gateway derives a 128-bit tenant-scoped key using the configured
  `AXS_CACHE_AFFINITY_SECRET` and a distinct v1 domain string.
- The raw value exists only during request classification. It is not logged, retained, forwarded,
  returned, or used as a metric label.
- The binding is a strong routing preference after hard eligibility. It never overrides readiness,
  lease freshness, drain, capacity, pool pinning, deployment identity/equivalence, or policy.
- If the preferred worker is unavailable or saturated, AX Serving may select another eligible
  worker. Because the harness sends complete input, this is a cold-cache performance event, not a
  state-loss error.
- Bindings are written only after trusted runtime admission/success and expire automatically.
- Concurrent requests in one session are allowed; v1 defines no turn ordering or session lock.

### Capability semantics

The stable per-model capability is `inference.agent-session.v1`.

- When the public request requires runtime support, capability absence removes the deployment from
  eligibility.
- When support is optional, Serving still uses soft affinity but omits normalized runtime hints for
  an incapable runtime.
- Runtime type strings do not imply capability. The AX runtime agent derives capability from the
  actual AX Engine model/runtime metadata and a conformance-tested adapter.

### State semantics

Memory and Redis/Valkey stores implement bounded TTL `SessionBindingRecord` state containing only an
opaque key, logical/deployment identity, worker ID, timestamps, and generation. No prompt,
conversation, tool, token, or KV content enters fleet state.

Shared affinity improves HA placement but is not required for correctness. Store failure follows an
operator-configured fail-open-for-affinity policy: continue normal eligible routing, emit a bounded
diagnostic, and never disable authentication, admission, or equivalence checks.

### Transport semantics

The gateway creates `x-ax-agent-*` normalized headers after public validation. Clients cannot set
or override the internal session key. `ax-runtime-agent` forwards only the explicit v1 allowlist to
a capable runtime and continues to deny public credentials and hop-by-hop headers.

No request body rewriting is required beyond the existing logical-to-runtime model field. Unknown
JSON fields and SSE bytes remain preserved.

## Alternatives considered

### A. Keep only generic `x-ax-cache-affinity`

Rejected as the product contract. It is useful but lacks typed workload/resource hints, runtime
capability negotiation, explicit security semantics, shared HA state, and truthful diagnostics.

### B. Add the narrow agent-aware fabric contract

Accepted. It extends current routing and state primitives without violating runtime neutrality.

### C. Add planner, tools, memory, and workflow APIs to AX Serving

Rejected. It duplicates agent frameworks, puts tool credentials and durable user state into the
gateway, and obscures the product's inference control-plane role.

### D. Make session stickiness hard and fail when the worker is lost

Rejected for v1. This converts performance state into correctness state, weakens resilience, and
conflicts with complete replayable requests.

### E. Hash prompts in the gateway for exact prefix routing

Rejected. The gateway would need tokenizer/template/model semantics, violating ADR-013. Exact KV
events may be consumed later from a runtime-owned authenticated source.

### F. Put the raw session ID in runtime headers or Redis

Rejected. It leaks cross-request application identity and creates unnecessary privacy and
cardinality risk. A tenant-scoped keyed digest is sufficient.

## Consequences

### Positive

- AX Serving gains an agent-specific performance story without becoming a framework.
- Existing cache-affinity, capability, fleet-state, and routing work is reused.
- AX Engine can optimize local prefix retention while CUDA runtimes remain optional participants.
- Worker loss and cache eviction remain safe and observable.
- The public contract is small enough for multiple harnesses and runtimes.

### Negative

- Public headers, protocol capability, shared state, and diagnostics become compatibility surfaces.
- Redis state and HA tests expand.
- Correct capability discovery requires runtime-agent fixtures per runtime/version.
- A second affinity signal must be integrated carefully with the legacy cache-affinity policy.

### Risks

- Accidental logging or metric labeling of session identifiers.
- Sticky traffic overloading a warm worker if capacity filters are bypassed.
- Stale HA bindings lowering cache hit rate or hiding deployment changes.
- Marketing confusing soft inference sessions with durable agent memory.

## Implementation constraint

The AX Serving coder must inspect `/Users/akiralam/code/ax-engine` and its agentic status ledger
before implementation, at every shared milestone, and before completion. Shared contract changes
must be reconciled in both specifications; a Serving-only interpretation is not compatible v1.

