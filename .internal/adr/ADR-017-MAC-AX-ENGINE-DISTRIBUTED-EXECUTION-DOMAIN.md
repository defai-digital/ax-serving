# ADR-017: Mac AX Engine Distributed Execution Domain

| Field | Value |
| --- | --- |
| Status | Accepted for incremental implementation |
| Decision date | 2026-07-28 |
| Scope | Cross-Mac model-parallel inference owned by AX Engine |
| Extends | [ADR-016](ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md) |
| Product requirements | [Mac distributed inference PRD](../prd/PRD-MAC-DISTRIBUTED-INFERENCE.md) |
| Technical design | [Mac distributed execution-domain specification](../specs/TECH-SPEC-MAC-DISTRIBUTED-EXECUTION-DOMAIN.md) |
| Evidence | [Implementation and certification status](../IMPLEMENTATION-STATUS.md) |

## Context

ADR-016 makes AX Serving a federation plane. It selects one execution domain and leaves runtime
semantics to that domain's execution owner. The current Mac domain kind, `mac_ax_engine`, represents
one independently runnable AX Engine node. That contract cannot represent a model whose weights do
not fit on any one Mac.

The product needs an additive way to expose a group of Macs that jointly execute one model. The
gateway must not become a tensor router, parse model files, move activations, own KV cache, or
coordinate ranks on the request path. Those responsibilities remain inside AX Engine.

### Reference implementation evidence

The decision is based on the pinned source checkouts under `.internal/reference`:

| Project | Reviewed revision | Relevant evidence |
| --- | --- | --- |
| vLLM | `0b6aa3c47ce69a1f3c8a19cafe3b9dc2871d1f6b` | Typed TP/PP/DP group construction, model-native parallel layers, local and Ray executors |
| NVIDIA Dynamo | `195c03545bfac35fdf7b3b2f96fe8235d70a6cbf` | Request/control/state planes, discovery leases, routing reservations, disaggregated serving |
| exo | `b5375f8cee4368d09e1ce96a56b9f81fb0bc81aa` | Cross-Mac MLX TP/PP, topology-aware placement, shard metadata, gang lifecycle, partial downloads |
| SGLang | `0a49226d19d606db40643584bd6883a6adc1287f` | Multi-node 405B execution, PP micro-batching, asynchronous P2P, dynamic chunking |

The useful lessons are:

1. Parallel execution topology is an engine contract, not a gateway routing policy.
2. A distributed model instance is a gang: all required ranks share one generation and readiness
   boundary.
3. The control plane and tensor data plane need separate protocols and failure semantics.
4. Placement must validate every rank's memory budget; aggregate memory alone is unsafe.
5. Pipeline parallelism is the safest initial cross-node capacity mechanism on slower links.
6. Tensor parallelism needs model-native sharded layers and frequent collectives, so it follows PP.
7. Pipeline throughput requires micro-batching and asynchronous communication, but those are
   optimizations after a correct static pipeline.

### Current AX constraints

- `mac_ax_engine` is valid only with node scope and owner `ax_engine`.
- The portable gateway must remain free of AX Engine, MLX, Metal, and model-runtime SDKs.
- The runtime owner owns distributed execution and hardware kernels.
- One AX request attempt enters one execution domain and remains there.
- Cross-domain retry remains bounded by admission and response commitment.
- AX fleet state must not contain prompts, generated text, tensors, KV blocks, or rank-local cache
  indexes.

## Decision

Add a new execution-domain kind:

```text
kind             = mac_ax_engine_cluster
endpoint_scope   = domain
execution_owner  = ax_engine
```

`mac_ax_engine` remains a node-scoped whole-model endpoint. Its meaning does not change.

AX Serving treats one Mac cluster as one admission and failure boundary. A cluster coordinator or
adapter exposes one OpenAI-compatible endpoint, one stable domain identity, one compatibility
manifest digest, one aggregate observation, and one lease-fenced registration. Internal AX Engine
ranks are never registered as AX workers.

### Ownership

| Concern | AX Serving | Cluster coordinator/adapter | AX Engine |
| --- | --- | --- | --- |
| Public API, auth, tenant policy | Owns | Does not own | Does not own |
| Logical model and equivalence | Owns | Reports immutable identity | Supplies runtime identity |
| Cross-domain admission and reservation | Owns | Enforces admitted attempt | Does not override |
| Cluster membership and generation | Observes aggregate state | Owns | Participates |
| Placement manifest and gang lifecycle | Stores digest and bounded status | Owns/control-plane coordinator | Executes |
| Layer/tensor partitioning | Does not own | Does not implement | Owns |
| Activation, collective, KV transport | Does not own | Does not proxy | Owns |
| Tokenization, batching, generation | Does not own | Byte-preserving proxy | Owns |
| Rank failure and restart | Marks domain unavailable | Fences generation and reconciles | Fails/restarts instance |

### Initial execution strategy

The first certified execution strategy is static pipeline parallelism:

- homogeneous Apple Silicon architecture and one pinned AX Engine build;
- explicit rank-to-node and layer-range manifest;
- no live resharding;
- one complete cluster generation loaded and warmed before readiness;
- one failed required rank makes the cluster unavailable;
- restart creates a new generation and fences stale ranks;
- fixed bounded context, output, and concurrency limits derived from the certified memory plan.

Tensor parallelism, hybrid PP/TP, topology auto-placement, dynamic chunking, and transparent
in-request rank replacement are later phases.

### Protocol evolution

Protocol 1.2 introduces `mac_ax_engine_cluster` and the
`control.mac-cluster.v1` capability. A cluster registration is invalid unless:

- protocol major is 1 and minor is at least 2;
- `control.execution-domain.v1` is present;
- `control.mac-cluster.v1` is present;
- endpoint scope is `domain`;
- execution owner is `ax_engine`;
- descriptor and observation manifest digests agree;
- worker, domain, pool, trust boundary, and hardware class agree.

The rank manifest is not embedded in registration. Registration carries only its immutable digest
and bounded aggregate observations. The full manifest stays with the AX Engine cluster controller
and retained certification artifacts.

## Consequences

### Positive

- Models larger than one Mac's usable memory can become one AX execution domain.
- Existing federation, identity, equivalence, retry, audit, and adapter boundaries remain useful.
- The gateway remains runtime-SDK-free and does not duplicate an inference engine.
- Single-node Mac deployments remain simple and backward compatible.
- The design can adopt better PP/TP implementations in AX Engine without changing the public API.

### Negative

- AX Engine needs a substantial distributed-runtime feature, not merely an AX Serving routing edit.
- A model-parallel cluster has gang failure behavior and lower availability than independent
  replicas unless multiple complete clusters are deployed.
- Slow or unstable Mac interconnects can make a model fit while producing poor latency.
- Certification expands to topology, transport, rank lifecycle, memory headroom, and distributed
  numerical correctness.

### Neutral trade-offs

- Initial PP targets capacity before throughput.
- The coordinator is logically centralized per cluster generation even if membership discovery is
  peer-assisted.
- AX Serving sees coarser telemetry than the cluster coordinator; this is intentional.

## Alternatives considered

### Change `mac_ax_engine` from node scope to domain scope

Rejected. It silently changes an existing wire contract and makes older agents ambiguous.

### Register every Mac rank as an AX worker

Rejected. Independent gateway routing to one rank would violate gang execution and expose engine
internals as routable endpoints.

### Implement pipeline/tensor routing in `ax-serving-api`

Rejected. It would link the federation plane to model semantics and put high-volume tensor traffic
through the public gateway.

### Embed exo as the production Mac runtime

Rejected as the product architecture. exo is valuable reference code, but it would displace AX
Engine ownership and couple certification to exo's model-specific patches and lifecycle.

### Make Dynamo manage Mac ranks

Rejected for the initial implementation. It couples the Mac execution path to Dynamo's control
stack without removing the need for AX federation policy.

### Begin with cross-Mac tensor parallelism

Rejected. TP requires frequent collectives and model-specific sharding. Static PP has a smaller
correctness surface and is more tolerant of ordinary LAN links.

## Compliance checks

A change complies with this ADR only when:

- the gateway still selects one domain, never a rank;
- `mac_ax_engine` remains node-scoped;
- all distributed model semantics stay in AX Engine;
- the adapter remains byte-preserving and runtime-SDK-free;
- every cluster generation has an immutable manifest digest;
- missing rank, stale generation, unknown identity, or insufficient headroom fails closed;
- cluster readiness requires the complete gang;
- retry ownership remains unambiguous;
- no prompt, output, KV, activation, or rank-local cache state enters AX fleet state;
- performance and availability claims are backed by retained topology-specific evidence.
