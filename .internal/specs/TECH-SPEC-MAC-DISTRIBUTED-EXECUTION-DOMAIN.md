# Technical Specification: Mac AX Engine Distributed Execution Domain

| Field | Value |
| --- | --- |
| Status | Approved target; phases 0-1 source/mock implemented |
| Last updated | 2026-07-28 |
| Decision | [ADR-017](../adr/ADR-017-MAC-AX-ENGINE-DISTRIBUTED-EXECUTION-DOMAIN.md) |
| Product requirements | [Mac distributed inference PRD](../prd/PRD-MAC-DISTRIBUTED-INFERENCE.md) |
| Parent specification | [Federated inference control plane](TECH-SPEC-FEDERATED-INFERENCE-CONTROL-PLANE.md) |
| Evidence | [Implementation and certification status](../IMPLEMENTATION-STATUS.md) |

## 1. Purpose

This specification defines the additive contracts by which AX Serving can admit a complete
model-parallel AX Engine cluster as one execution domain. It separates the federation control
plane from the AX Engine cluster control and tensor data planes.

It does not define MLX kernels or commit AX Engine to one transport implementation. It defines the
boundary those implementations must satisfy.

## 2. Invariants

1. `mac_ax_engine` remains node-scoped.
2. `mac_ax_engine_cluster` is domain-scoped and owned by `ax_engine`.
3. One cluster exposes one AX registration endpoint; internal ranks are not AX workers.
4. The portable gateway has no AX Engine, MLX, Metal, model-loader, or tensor-transport dependency.
5. AX selects a domain and deployment. AX Engine selects and executes the parallel plan.
6. One admitted attempt never moves between ranks, plans, or domains.
7. Rank topology, tensors, activations, KV, prompts, and outputs never enter AX fleet state.
8. Every ready observation binds to one immutable compatibility/parallelism manifest digest.
9. Missing, stale, partial, or contradictory cluster state fails closed.
10. A cluster failure after admission ambiguity is not an AX-safe retry signal.

## 3. Logical architecture

```text
                         AX-owned federation plane

Client -> AX gateway -> domain selection -> domain reservation
                              |
                              v
                  Mac cluster adapter endpoint
                  registration / lease / proxy
                              |
                 AX Engine cluster control plane
                              |
                   coordinator generation G
                      /        |        \
                   rank 0    rank 1    rank N
                      \________|________/
                      AX Engine data plane
```

The adapter and coordinator may initially be one process, but their interfaces remain distinct:

- adapter side: AX protocol, auth boundary, OpenAI JSON/SSE proxy;
- coordinator side: membership, manifest, rank lifecycle, aggregate observation;
- engine side: model execution and high-volume data transfer.

## 4. Reference-derived design choices

### vLLM

- Use a typed parallel configuration and deterministic rank groups.
- Put TP/PP behavior in model and executor abstractions.
- Keep control RPC separate from tensor communication.

### Dynamo

- Separate request, control, and state propagation paths.
- Apply hard eligibility before scoring.
- Reserve selected capacity and make cleanup single-owner and idempotent.
- Use leases/discovery plus a short local inhibition window to cover propagation delay.

### exo

- Model the instance as a gang with explicit rank/world/layer metadata.
- Consider topology and already-downloaded artifacts in placement.
- Download only shard-relevant safetensor files when the index proves the mapping.
- Do not adopt aggregate-only memory admission or model-specific monkey-patching as the primary
  architecture.

### SGLang

- Begin with PP where cross-node TP communication is too expensive.
- Add non-blocking P2P and micro-batching after correctness.
- Treat dynamic chunking as a profile-driven optimization, not a default correctness mechanism.

The primary pinned source paths used for this comparison were:

| Project | Source paths |
| --- | --- |
| vLLM | `vllm/distributed/parallel_state.py`, `vllm/v1/executor/abstract.py`, `vllm/v1/executor/ray_executor.py`, `vllm/model_executor/layers/linear.py`, `vllm/distributed/kv_transfer/` |
| Dynamo | `lib/llm/src/kv_router/`, `components/src/dynamo/router/`, `components/src/dynamo/global_router/`, `components/src/dynamo/planner/`, `docs/fern/diagrams/discovery_plane_lease.d2` |
| exo | `src/exo/master/placement.py`, `src/exo/master/placement_utils.py`, `src/exo/worker/plan.py`, `src/exo/worker/engines/mlx/auto_parallel.py`, `src/exo/download/impl_shard_downloader.py` |
| SGLang | `python/sglang/srt/managers/scheduler_pp_mixin.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/model_loader/loader.py`, `docs_new/docs/advanced_features/pipeline_parallelism.mdx` |

## 5. Protocol 1.2

### 5.1 Domain kind

Add:

```rust
pub enum ExecutionDomainKind {
    MacAxEngine,
    MacAxEngineCluster,
    NvidiaDynamoPc,
    NvidiaDynamoThor,
    CompatibilityRuntimeEndpoint,
    Unknown,
}
```

Valid combinations:

| Kind | Scope | Owner | Minimum protocol |
| --- | --- | --- | --- |
| `mac_ax_engine` | `node` | `ax_engine` | 1.1 when descriptor is used |
| `mac_ax_engine_cluster` | `domain` | `ax_engine` | 1.2 |
| `nvidia_dynamo_pc` | `domain` | `dynamo` | 1.1 |
| `nvidia_dynamo_thor` | `domain` | `dynamo` | 1.1 |
| `compatibility_runtime_endpoint` | `node` | configured runtime | 1.1 |

### 5.2 Capability

Add:

```text
control.mac-cluster.v1
```

Cluster registration requires both:

```text
control.execution-domain.v1
control.mac-cluster.v1
```

The gateway advertises the capability only after it implements fail-closed descriptor validation
and catalog matching. A 1.2 peer that omits the capability cannot register a cluster kind.

### 5.3 Descriptor and observation

Reuse `ExecutionDomainDescriptor` and `DomainObservation`.

Required cluster descriptor fields:

- stable domain ID;
- `kind = mac_ax_engine_cluster`;
- `endpoint_scope = domain`;
- `execution_owner = ax_engine`;
- pool and trust domain;
- certified aggregate hardware class;
- architecture, initially `arm64`;
- compatibility manifest digest;
- bounded labels such as region, site, transport class, and coordinator implementation.

Required ready observation fields:

- monotonically increasing cluster generation;
- ready state;
- at least one ready frontend/coordinator instance;
- aggregate capacity;
- matching manifest digest;
- exact runtime model inventory and supported operations.

The observation does not enumerate ranks.

## 6. Parallelism manifest

The complete `ParallelismManifestV1` is a signed or otherwise integrity-protected certification
artifact retained outside gateway fleet state.

Conceptual schema:

```text
version
cluster_id
generation
created_at

model:
  artifact_digest
  revision
  tokenizer_digest
  template_digest
  quantization
  architecture
  total_layers

runtime:
  ax_engine_version
  build_digest
  mlx_version
  os_baseline

parallelism:
  kind: pipeline | tensor | hybrid
  pp_size
  tp_size
  micro_batch_limit
  chunking_profile_digest?

transport:
  kind
  security_profile
  topology_digest
  minimum_bandwidth
  maximum_latency

ranks:
  rank_id
  node_identity_digest
  stage
  tensor_rank
  layer_range
  non_layer_ownership
  required_weight_files_digest

memory_plan:
  per-rank weight_bytes
  non_layer_bytes
  quantization_metadata_bytes
  kv_budget_bytes
  activation_budget_bytes
  communication_budget_bytes
  allocator_reserve_bytes
  os_headroom_bytes
```

AX registration carries a digest of the complete artifact through the existing compatibility
manifest field. The coordinator provides the artifact to rank processes through its authenticated
control plane.

## 7. Coordinator state machine

```text
Absent
  -> Planned
  -> Downloading
  -> Connecting
  -> Loading
  -> Warming
  -> Ready
  -> Draining
  -> Stopped

Any non-terminal state -> Failed
Failed -> Planned only with a higher generation
```

Rules:

- all rank transitions are generation-fenced;
- a rank cannot skip manifest verification;
- loading begins only after every required rank has required artifacts;
- warmup begins only after the full communication group is connected;
- ready requires successful warmup on every rank;
- rank failure clears ready before reconciliation;
- a replacement rank joins only a new generation in the initial implementation;
- drain removes admission before stopping the gang.

## 8. Placement

### 8.1 Initial static planner

Phase 1 accepts an operator-generated manifest. Validation is deterministic:

1. All ranks and nodes are unique where required.
2. Rank IDs are contiguous.
3. PP layer ranges are half-open, non-overlapping, gap-free, and cover all layers.
4. Required non-layer parameters have an owner.
5. Per-rank memory demand does not exceed certified usable memory.
6. Transport edges required by the plan exist.
7. All nodes match the runtime, architecture, trust, and model compatibility constraints.

### 8.2 Memory admission

For rank `r`:

```text
demand(r) =
  assigned_weight_bytes
  + non_layer_weight_bytes
  + quantization_metadata_bytes
  + kv_budget_bytes
  + activation_budget_bytes
  + communication_buffer_bytes
  + allocator_reserve_bytes
  + os_headroom_bytes
```

Admission requires:

```text
demand(r) <= certified_usable_memory(r)
```

Aggregate memory is diagnostic only.

### 8.3 Automatic planner

Phase 3 may generate candidate plans. Hard filters run before scoring. A candidate score may use:

- worst-rank memory headroom;
- slowest required link bandwidth;
- measured latency;
- expected stage imbalance;
- artifact download locality;
- thermal/reliability history.

The planner first runs advisory-only and records its candidate plan beside the active static plan.

## 9. AX Engine execution contract

### 9.1 Pipeline parallel MVP

- Rank 0 accepts the request and performs tokenization/embedding as defined by AX Engine.
- Each PP rank owns a contiguous layer range.
- Activations move directly from stage to stage.
- The final stage performs final normalization/head unless the manifest assigns a different explicit
  owner.
- Sampled tokens and required control metadata return through the engine-defined pipeline.
- Each rank owns KV only for its local layers.
- Request cancellation is broadcast to all ranks and is idempotent.

### 9.2 Micro-batching

Later PP optimization introduces bounded micro-batches:

- stable request and micro-batch sequence IDs;
- non-blocking send handles;
- bounded in-flight depth;
- ordered commit;
- cancellation tombstones;
- no buffer reuse before transfer completion.

### 9.3 Tensor parallelism

TP is not implemented by slicing opaque loaded tensors in the AX adapter. AX Engine must expose
model-native column/row/QKV/expert parallel abstractions with explicit checkpoint loaders and
collective semantics.

## 10. AX Serving admission and routing

Target routing is two-stage:

1. Domain selection chooses an eligible domain/deployment and acquires a domain-keyed reservation.
2. Execution dispatch sends the request to the selected domain endpoint.

For `mac_ax_engine`, an optional node-selection stage may still choose a whole-model Mac endpoint.
For `mac_ax_engine_cluster`, the domain endpoint is already the complete execution target.

Cluster hard eligibility includes:

- desired domain enabled;
- exact kind/scope/owner;
- qualification;
- fresh observation;
- ready state;
- manifest match;
- model identity/equivalence;
- operation/capability/limit support;
- trust/locality requirements;
- aggregate admission capacity.

No rank-level score is allowed.

## 11. Reservation and retry

Phase 1 introduces domain-keyed reservation storage:

```text
key = domain_id
member = attempt_id
value = lease expiry and selected observation generation
```

Properties:

- bounded and idempotent;
- renewal by the same attempt;
- cleanup owned by one RAII guard;
- generation mismatch fences dispatch;
- Redis/Valkey mutation is atomic;
- local worker inflight counters do not substitute for domain admission.

Retry rules:

- connection failure before admission may select an equivalent domain;
- typed cluster `not_admitted` may select an equivalent domain;
- generic `5xx`, stream interruption, or timeout after ambiguous admission is not AX-retryable;
- AX never retries to an internal rank.

## 12. Artifact handling

Phase 2 may download full model artifacts for simplicity. Phase 3 adds shard-aware download:

1. Fetch and verify the safetensors index.
2. Resolve tensors in the rank's layer range.
3. Add explicitly assigned non-layer tensors.
4. Download the union of referenced files and common tokenizer/config artifacts.
5. Verify file and manifest digests before load.

If an index is missing, ambiguous, or uses unsupported naming, fail closed or download the complete
certified artifact. Never infer model identity from filenames.

## 13. Security

Trust boundaries:

- client to AX gateway;
- AX gateway to cluster adapter;
- adapter to coordinator;
- coordinator to ranks;
- rank-to-rank data plane.

Requirements:

- mDNS/DNS-SD may discover addresses but never supplies identity or authorization;
- cluster control messages include cluster ID, generation, manifest digest, sender rank, and replay
  protection;
- data-plane endpoints are derived from the authenticated manifest;
- credentials and raw topology secrets are not stored in decision records;
- untrusted custom model code is disabled in certified profiles.

## 14. Observability

AX-level bounded fields:

- domain ID, generation, manifest digest;
- ready/state/reason code;
- ready frontend count;
- aggregate active, waiting, and maximum requests;
- supported model inventory;
- selection/rejection reason and policy version.

Coordinator-only diagnostics:

- rank state and last transition;
- stage/layer assignment;
- memory plan versus observed usage;
- link throughput/latency/error;
- download/load/warmup progress;
- per-stage compute and transfer time;
- pipeline bubble estimate.

No metric label may contain prompt, output, user ID, raw session ID, or unbounded model input.

## 15. Failure semantics

| Failure | Initial behavior |
| --- | --- |
| Coordinator lease loss | Domain becomes unroutable; ranks drain or stop after bounded orphan timeout |
| Required rank loss | Clear readiness, abort affected requests, fail generation |
| Rank rejoins stale generation | Reject and retain diagnostic evidence |
| Manifest mismatch | Fail registration/readiness |
| Partial artifact | Rank remains downloading/failed; gang not ready |
| Data-plane partition | Abort affected generation; no transparent AX retry |
| Adapter-to-coordinator failure before admission | Typed not-admitted only when proven |
| Failure after admission ambiguity | Propagate failure; no AX retry |
| AX gateway restart | Shared reservation and domain lease preserve bounds |

## 16. Repository implementation map

### Phase 0

- `ax-serving-protocol`: protocol 1.2, new domain kind and capability, validation and fixtures.
- `ax-serving-api`: desired catalog validation and fail-closed endpoint matching.
- Internal docs/status: canonical additive architecture and current evidence.

### Phase 1

- Implemented in `ax-mac-cluster-adapter`: runtime-neutral manifest validation, gang readiness,
  rank bootstrap/control APIs, aggregate protocol-v1.2 registration, heartbeat, drain, and
  byte-preserving rank-0 proxy.
- Implemented in `ax-serving-api`: two-stage domain/deployment selection, domain-keyed reservation,
  observation-generation fencing, bounded rejected-candidate evidence, and shared decision
  retention in memory or Redis/Valkey.
- Implemented source/mock coverage for stale/missing/failed ranks, topology thresholds, credential
  separation, rank-specific artifact plans, reservation fencing, and decision round trips.

This phase does not contain a tensor data plane or an AX Engine distributed executor.

### AX Engine repository

- Parallelism manifest parser and validation.
- Distributed executor and rank lifecycle.
- PP model partition/load path.
- Activation transport, cancellation, and memory accounting.

The AX Engine work is outside this repository and must be pinned by immutable compatibility
manifest before AX Serving enables the domain.

## 17. Test strategy

### Unit

- enum serialization and unknown-kind tolerance;
- kind/scope/owner validation;
- protocol/capability gating;
- catalog runtime matching;
- no v1.0/v1.1 implicit migration into a cluster domain;
- manifest equality and stale observation rejection.

### Mock integration

- register/heartbeat/drain one fake cluster;
- incomplete gang never ready;
- generation fencing;
- reservation saturation and cleanup;
- safe pre-admission retry and ambiguous failure;
- two gateways plus Redis/Valkey.

### Live

- two-node minimum static PP;
- every rank-loss position;
- streaming, cancellation, deadline, drain, restart;
- memory pressure and maximum certified context/concurrency;
- direct-versus-federated performance;
- topology-specific soak.

## 18. Multi-phase implementation plan

### Phase 0: additive protocol foundation

- Land ADR, PRD, specification, and index/status updates.
- Add protocol 1.2 cluster kind/capability.
- Add validation, fixture, catalog matching, and compatibility tests.
- Do not add an agent flag or advertise runtime support.

Exit: source accepts a valid explicit cluster descriptor and rejects invalid/legacy claims.

### Phase 1: mock coordinator and HA admission

- Implement coordinator-facing adapter skeleton.
- Register aggregate observation and exact manifest digest.
- Add domain-keyed reservations with observation-generation fencing.
- Persist complete decision candidates and rejection reasons.

Exit: two gateways safely admit a mock cluster without exposing ranks.

### Phase 2: AX Engine static PP

- Implement manifest validation, gang lifecycle, partial layer load, and direct activation transport
  in AX Engine.
- Support one decoder-only model family and one quantization.
- Wire typed admission, streaming, cancellation, drain, and rank failure through the adapter.

Exit: a model that cannot fit one Mac completes certified two-or-more-Mac inference.

### Phase 3: operational and performance hardening

- Add shard-aware artifact download.
- Add asynchronous PP communication and bounded micro-batching.
- Add topology measurement and advisory placement.
- Complete security, load, fault, restart, upgrade, rollback, and soak gates.

Exit: one pinned topology is production qualified.

### Phase 4: TP, hybrid, and dynamic chunking

- Add model-native TP layers and checkpoint loading in AX Engine.
- Add PP/TP hybrid manifest support.
- Add profile-derived chunking and stage balancing in advisory mode, then canary.

Exit: each enabled model/plan demonstrates a measured win and passes separate certification.

### Phase 5: adaptive federation

- Feed conservative aggregate cost/latency/capacity into the AX domain selector.
- Complete offline replay, shadow, canary, and rollback.

Exit: cluster-aware domain policy improves a declared workload without violating hard policy.
