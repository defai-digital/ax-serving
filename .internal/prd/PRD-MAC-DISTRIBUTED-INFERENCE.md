# Product Requirements: Mac Distributed Inference Domain

| Field | Value |
| --- | --- |
| Status | Canonical additive product scope; phases 0-1 source/mock implemented |
| Owner | AX Serving and AX Engine maintainers |
| Last updated | 2026-07-28 |
| Applies to | AX Serving 3.x and a separately pinned AX Engine release |
| Architecture decision | [ADR-017](../adr/ADR-017-MAC-AX-ENGINE-DISTRIBUTED-EXECUTION-DOMAIN.md) |
| Technical specification | [Mac cluster specification](../specs/TECH-SPEC-MAC-DISTRIBUTED-EXECUTION-DOMAIN.md) |
| Parent architecture | [ADR-016](../adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md) |

## 1. Executive summary

AX Serving will support an AX Engine execution domain composed of multiple Macs that jointly run
one model. This allows a logical model whose certified runtime footprint exceeds every individual
Mac to remain available through the same AX OpenAI-compatible API.

The feature is not implemented by splitting requests between existing Mac workers. One complete
Mac cluster is one model-parallel runtime endpoint. AX Engine owns layer/tensor partitioning,
distributed communication, caches, batching, and generation. AX Serving owns cross-domain policy,
admission, reservation, identity, audit, and retry boundaries.

The first release targets static pipeline parallelism for capacity. Later releases may add
micro-batching, tensor parallelism, hybrid plans, and automatic topology placement.

## 2. Customer problem and value

### 2.1 Problem

Organizations may have aggregate Apple Silicon unified memory large enough for a model but no one
Mac with enough usable memory. Independent whole-model workers cannot use that aggregate capacity.
Operators otherwise need a separate distributed system, a different API, or NVIDIA capacity.

### 2.2 Primary users

- Private AI operators with two or more Apple Silicon Macs.
- Teams that require local/offline inference for models larger than one Mac.
- Labs evaluating large quantized models without exposing data to a cloud endpoint.
- Heterogeneous AX operators who want the same logical model policy across Mac and Dynamo domains.

### 2.3 Value hypothesis

The feature is valuable only if it proves at least one of:

- a model that cannot load on any one participating Mac completes correct inference;
- local policy requirements can be met without moving the request to a less preferred domain;
- installed Mac memory achieves useful service availability at acceptable latency;
- operator effort is lower than maintaining a separate public endpoint and identity model.

Merely forming a cluster is not a value result.

## 3. Goals

### P0: safe contract

- Add an explicit `mac_ax_engine_cluster` domain without changing `mac_ax_engine`.
- Preserve protocol 1.0/1.1 interoperability.
- Require a 1.2 capability and domain-scoped descriptor for cluster registration.
- Keep rank topology and tensor state outside gateway fleet state.
- Make desired catalog validation and endpoint matching fail closed.

### P1: static pipeline MVP

- Run one certified decoder-only model family across at least two Macs.
- Load only the assigned layer range and required non-layer weights on each rank.
- Expose one OpenAI-compatible cluster endpoint through an adapter/coordinator.
- Support chat/text completion streaming, cancellation, readiness, drain, and typed pre-admission.
- Fence stale cluster generations.
- Fail the entire instance when a required rank is lost.

### P2: usable service

- Add pipeline micro-batching and asynchronous stage transfers.
- Add topology measurement and conservative automatic layer placement.
- Add shard-aware artifact download and verification.
- Support multiple independently runnable cluster replicas behind one AX domain or pool.
- Provide retained load, fault, restart, and soak evidence.

### P3: advanced parallelism

- Add model-native tensor-parallel layers inside AX Engine.
- Add qualified PP/TP hybrid plans.
- Add optional dynamic chunking from measured per-stage execution profiles.
- Evaluate faster transports separately from the control protocol.

## 4. Non-goals

- Splitting one request across Mac and NVIDIA domains.
- Sending activations, tensors, or KV cache through AX Serving.
- Registering internal ranks as independently routable AX workers.
- Transparent continuation after arbitrary rank loss in the initial release.
- Live resharding of an admitted request or loaded generation.
- Automatic execution of unverified model code.
- Claiming that aggregate memory implies acceptable throughput.
- Reimplementing Dynamo routing, vLLM, SGLang, or exo inside AX Serving.

## 5. Product requirements

### 5.1 Domain and identity

| ID | Priority | Requirement |
| --- | --- | --- |
| MCD-DOM-001 | P0 | Add `mac_ax_engine_cluster` as a distinct execution-domain kind with domain scope and `ax_engine` owner. |
| MCD-DOM-002 | P0 | Preserve `mac_ax_engine` as a node-scoped whole-model endpoint. |
| MCD-DOM-003 | P0 | Represent one complete cluster as one AX registration, lease, admission endpoint, and failure boundary. |
| MCD-DOM-004 | P0 | Reject cluster descriptors from protocol versions older than 1.2 or without `control.mac-cluster.v1`. |
| MCD-ID-001 | P0 | Bind model, tokenizer, template, quantization, AX Engine build, parallel plan, transport, and topology to an immutable manifest digest. |
| MCD-ID-002 | P0 | Require descriptor and observation manifest digests to match. |
| MCD-ID-003 | P0 | Require explicit equivalence evidence before failover between a cluster, a single Mac, or a Dynamo deployment. |

### 5.2 Placement and memory

| ID | Priority | Requirement |
| --- | --- | --- |
| MCD-PLC-001 | P1 | A plan assigns every required rank to exactly one node and every model layer to the required PP stage(s). |
| MCD-PLC-002 | P1 | Placement checks each node independently; aggregate memory alone is insufficient. |
| MCD-PLC-003 | P1 | The budget includes weights, non-layer parameters, quantization metadata, KV, activations, communication buffers, allocator reserve, and OS headroom. |
| MCD-PLC-004 | P1 | Initial plans are immutable for one cluster generation. |
| MCD-PLC-005 | P2 | Automatic placement uses measured bandwidth/latency and retained model profiles; unknown signals receive conservative treatment. |

### 5.3 Lifecycle and failure

| ID | Priority | Requirement |
| --- | --- | --- |
| MCD-LIF-001 | P1 | Lifecycle is gang-based: planned, downloading, connecting, loading, warming, ready, draining, failed. |
| MCD-LIF-002 | P1 | Ready requires every required rank at the same manifest digest and generation. |
| MCD-LIF-003 | P1 | Rank loss or generation mismatch stops new admission and fails the instance. |
| MCD-LIF-004 | P1 | Restart creates a higher generation and fences stale coordinators/ranks. |
| MCD-LIF-005 | P1 | Drain rejects new admission before shutting ranks down and has a bounded deadline. |
| MCD-LIF-006 | P2 | Reconciliation is asynchronous and never blocks the gateway request path. |

### 5.4 Execution and API

| ID | Priority | Requirement |
| --- | --- | --- |
| MCD-EXE-001 | P1 | AX Engine owns PP/TP, activation transfer, KV, batching, tokenization, templates, sampling, and kernels. |
| MCD-EXE-002 | P1 | The adapter preserves OpenAI JSON and SSE bytes subject only to existing bounded routing-field rewrite. |
| MCD-EXE-003 | P1 | Cancellation and deadlines propagate to the complete distributed instance. |
| MCD-EXE-004 | P1 | PP rank 0 accepts the logical request; internal ranks are not public endpoints. |
| MCD-EXE-005 | P2 | Pipeline micro-batching preserves per-request ordering and cancellation. |
| MCD-EXE-006 | P3 | TP is implemented with explicit model-native sharded layers, not gateway or adapter patches. |

### 5.5 Admission, retry, and HA

| ID | Priority | Requirement |
| --- | --- | --- |
| MCD-ADM-001 | P0 | Hard identity, capability, readiness, generation, manifest, trust, and capacity checks precede scoring. |
| MCD-ADM-002 | P0 | AX reserves cluster-domain capacity, not an internal rank. |
| MCD-ADM-003 | P1 | A typed pre-admission rejection proves that no rank admitted the request. |
| MCD-ADM-004 | P1 | Ambiguous failure after cluster admission is not retried by AX. |
| MCD-ADM-005 | P2 | Two AX gateways plus Redis/Valkey preserve domain reservation bounds and generation fencing. |

### 5.6 Security and privacy

| ID | Priority | Requirement |
| --- | --- | --- |
| MCD-SEC-001 | P0 | Gateway state excludes prompts, outputs, tensors, KV blocks, credentials, and rank-local cache indexes. |
| MCD-SEC-002 | P1 | Public, worker-control, adapter-dispatch, runtime-control, and rank data-plane credentials are distinct. |
| MCD-SEC-003 | P1 | LAN discovery is bootstrap only and never establishes trust. |
| MCD-SEC-004 | P1 | Non-loopback control and data-plane connections are mutually authenticated or confined to an explicitly trusted transport profile. |

### 5.7 Observability and operations

| ID | Priority | Requirement |
| --- | --- | --- |
| MCD-OBS-001 | P0 | AX receives bounded aggregate readiness, capacity, generation, reason code, and model inventory. |
| MCD-OBS-002 | P1 | Coordinator exposes per-rank diagnostics to operators without making ranks AX candidates. |
| MCD-OBS-003 | P1 | Metrics distinguish placement, download, connect, load, warmup, request, transport, and rank-failure time. |
| MCD-OBS-004 | P2 | Benchmarks retain topology, OS, AX Engine, model, quantization, context, concurrency, and transport identity. |

## 6. User experience

An operator declares one cluster domain and one deployment. A separate cluster manifest binds its
internal ranks. Normal clients continue to send:

```text
POST /v1/chat/completions
model: <logical-model>
```

Clients do not provide rank count, node addresses, layer ranges, or transport configuration.

Operator status must clearly distinguish:

- configured but no coordinator;
- coordinator present but gang incomplete;
- downloading/loading/warming;
- ready;
- degraded but still safe to admit;
- draining;
- failed or fenced;
- experimental versus certified.

## 7. Certification gates

### Correctness gate

- The target model cannot load on any one test Mac at the certified limits.
- Distributed output matches a pinned single-runtime reference within the declared deterministic or
  statistical tolerance.
- Streaming order, stop conditions, usage accounting, cancellation, and errors conform to the
  public API contract.

### Capacity gate

- Every rank remains below its certified memory ceiling with required OS headroom.
- The configured maximum context and concurrency do not cause swap, system pressure termination, or
  unbounded allocator growth.

### Fault gate

- Loss of every rank position is tested independently.
- Stale generation, duplicate coordinator, partial load, transport partition, cancellation, and
  drain cannot produce duplicate commitment.
- The domain becomes unroutable within the declared detection bound.

### Performance/value gate

- Direct AX Engine cluster and AX-federated paths are benchmarked separately.
- Gateway overhead remains within the parent PRD NFR.
- TTFT, ITL, throughput, and energy are reported; "model fits" is not presented as "model is fast."
- At least one representative workload meets an operator-approved usefulness threshold.

### Release gate

- One exact Mac hardware topology, OS, AX Engine build, model artifact, manifest, and transport is
  pinned.
- At least 60 minutes of mixed streaming and cancellation soak passes.
- Installation, upgrade, rollback, credential rotation, restart, and evidence retention are
  documented and exercised.

## 8. Phased delivery plan

| Phase | Outcome | Exit gate |
| --- | --- | --- |
| 0. Contract | Protocol 1.2 recognizes a fail-closed Mac cluster domain | Unit/fixture/API catalog tests pass; no runtime enabled |
| 1. Coordinator skeleton | Source/mock implemented: one cluster registers, heartbeats, drains, fences generations, and uses domain reservations | Live Redis/Valkey two-gateway evidence remains a qualification gate |
| 2. Static PP MVP | One model runs over two or more Macs | Correctness, stream, cancel, rank-loss tests |
| 3. Production hardening | Conservative placement, partial download, async PP, operations | Load, fault, soak, security, value gates |
| 4. Advanced parallelism | TP/hybrid and profile-driven chunking | Per-model certification and performance win |
| 5. Adaptive federation | Cluster participates in replayed/shadow cost/SLO policy | Replay, shadow, canary, rollback gates |

Phase 1 is complete. The Phase 2 source implementation now provides generation-fenced static
pipeline contracts, selective dense-Llama-3 stage loading, stage-local KV, bounded serialized
activation transport, authenticated rank services, ordered cancellation/cleanup, rank heartbeats,
and a greedy OpenAI-compatible rank-0 gateway. Mock HTTP tests and a two-stage numerical fixture
exercise a real serialized activation boundary and match monolithic forward within the established
tolerance.

Phase 2 is not a production-support claim until a pinned real-weight model is qualified on at least
two physical Macs. The current implementation deliberately excludes tensor/hybrid parallelism,
dynamic stage balancing, automatic artifact download, non-greedy sampling, tool/structured output,
and production fault/load/soak certification.
