# ADR-016: Federated Dynamo and AX Engine Control Plane

| Field | Value |
| --- | --- |
| Status | Accepted; final architecture |
| Decision date | 2026-07-15 |
| Owners | AX Serving maintainers |
| Scope | Product boundary, execution domains, routing ownership, deployment, and integration strategy |
| Supersedes | ADR-013, ADR-014, and ADR-015 |
| Product requirements | [AX Serving product requirements](../prd/PRD-AX-SERVING.md) |
| Technical design | [Federated inference control-plane specification](../specs/TECH-SPEC-FEDERATED-INFERENCE-CONTROL-PLANE.md) |
| Evidence | [Implementation and certification status](../IMPLEMENTATION-STATUS.md) |

## Context

AX Serving must create value above proven inference systems rather than rebuild them. The fleet in
scope contains three materially different execution classes:

- Apple Silicon Macs running AX Engine through the AX runtime agent;
- NVIDIA GPU PCs running an upstream NVIDIA Dynamo deployment;
- NVIDIA Thor devices running a separately qualified upstream Dynamo deployment.

NVIDIA Dynamo already owns the difficult NVIDIA-local problems: frontend and request routing,
backend integration for vLLM, SGLang, and TensorRT-LLM, KV-aware placement, disaggregated
prefill/decode, NIXL transfers, planning, scaling, failure handling, and Kubernetes resources.
Making AX Serving independently select every NVIDIA worker would duplicate Dynamo, create two
conflicting schedulers, and give operators little reason to deploy AX Serving.

The existing AX Serving source nevertheless contains valuable foundations:

- a portable, runtime-SDK-free OpenAI-compatible gateway;
- explicit logical models, pools, deployments, identity, and equivalence policy;
- bounded request classification and hard admission filters;
- safe pre-commit retry, incremental SSE, deadlines, cancellation, and drain;
- a versioned runtime-agent protocol with authoritative readiness and inventory;
- in-memory and Redis/Valkey fleet state, fencing, capacity reservations, and HA reconciliation;
- CPU-only OCI, Compose, Kubernetes, and Helm deployment surfaces;
- an AX runtime-agent path that keeps AX Engine execution on the Mac.

The architectural question is therefore not whether AX Serving should replace Dynamo. It is where
AX Serving can govern a heterogeneous private fleet without taking ownership away from either
Dynamo or AX Engine.

## Decision drivers

- Give users a reason to retain Mac and Thor capacity instead of deploying only CUDA/vLLM.
- Reuse Dynamo's NVIDIA-local routing and execution work without maintaining a private fork.
- Preserve AX Engine as the native Apple Silicon execution runtime.
- Keep NVIDIA GPU PCs and Thor in separate performance, artifact, and failure domains.
- Provide one policy, identity, security, audit, and API boundary across all execution domains.
- Make every routing and failover decision replayable, explainable, and bounded.
- Keep the gateway free of CUDA, Metal, MLX, Dynamo, vLLM, SGLang, and TensorRT-LLM SDKs.
- Separate implemented source behavior from live compatibility and production certification.

## Decision

AX Serving will be a **federated heterogeneous inference control plane**.

It selects an **execution domain** for each request. It does not select an NVIDIA worker inside a
Dynamo domain and does not schedule tokens inside an AX Engine process.

```text
OpenAI-compatible client
          |
          v
  AX Serving federation plane
  - auth and tenant policy
  - logical model and equivalence
  - domain admission and selection
  - global SLO/cost/locality policy
  - audit, retry boundary, lifecycle intent
          |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
 Mac execution pool      NVIDIA PC domain       NVIDIA Thor domain
 AX runtime agents       Dynamo deployment      Dynamo deployment
          |                      |                      |
          v                      v                      v
     AX Engine          Dynamo selects GPU      Dynamo selects Thor
     owns tokens         worker and backend      worker and backend
```

The term **domain** means one independently operated routing and failure boundary. A domain has one
control owner, one hardware/deployment class, one trust boundary, one certified model inventory,
and one admission endpoint exposed to AX Serving.

### Domain types

| Domain type | AX Serving selects | Domain-local owner | Initial status |
| --- | --- | --- | --- |
| `mac_ax_engine` | An eligible Mac node in the selected Mac pool | AX runtime agent and AX Engine | Existing implementation; live certification pending |
| `nvidia_dynamo_pc` | A PC Dynamo domain endpoint | Dynamo frontend/router/planner and selected backend | Adapter implementation and certification pending |
| `nvidia_dynamo_thor` | A Thor Dynamo domain endpoint | A separately deployed Dynamo stack and selected backend | Experimental until Thor qualification passes |

A Mac pool may contain multiple independently registered Mac endpoints, so the existing AX endpoint
picker remains useful within that pool. A Dynamo deployment is represented to AX Serving as one
domain endpoint; its internal workers are not registered as AX workers.

### Ownership boundary

| AX Serving owns | Dynamo owns inside NVIDIA domains | AX Engine path owns on Mac |
| --- | --- | --- |
| Public API, authentication, tenant quotas | NVIDIA frontend and worker discovery | Runtime readiness and model inventory |
| Logical model and deployment catalog | KV-aware worker routing | Tokenization and chat templates |
| Model identity and equivalence certification | Prefill/decode placement and disaggregation | Batching and token scheduling |
| Trust, privacy, locality, residency, and budget policy | KV events, KVBM, and NIXL transfer | KV/prefix cache contents |
| Cross-domain admission and selection | GPU backend choice within the deployed graph | MTP/speculation and local acceleration |
| Global audit, reason codes, and policy version | NVIDIA-local planner, scaling, and load shedding | Metal/MLX kernels and memory management |
| Cross-domain retry before commitment | In-domain retry, migration, and cancellation | Typed local admission and cancellation |
| Desired lifecycle intent and global rollout gates | Dynamo CRDs/operator and GPU process lifecycle | AX runtime-agent/Engine process lifecycle |

Ownership is exclusive on the request path. AX Serving must not reproduce Dynamo's worker cost
model, KV index, planner, disaggregation, or accelerator scheduling. Dynamo must not decide AX
tenant, residency, semantic-equivalence, or cross-domain policy.

### Two-stage routing

Routing has exactly two stages:

1. **Federation routing:** AX Serving resolves the logical model, applies hard policy, reserves
   domain capacity, and chooses `mac`, `nvidia-pc`, or `nvidia-thor` plus a deployment.
2. **Execution routing:** Dynamo chooses the NVIDIA worker, or the selected AX Engine endpoint
   executes locally.

AX Serving may use declared operation, modality, context/output limits, tenant, priority, deadline,
privacy, locality, cost ceiling, SLO, health, and conservative domain telemetry. It must not
tokenize prompts or import Dynamo's per-worker state. Learned or task-aware selection is a later
versioned policy evaluated in shadow mode before it can affect production traffic.

### Dynamo integration boundary

The canonical upstream is [`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo). AX Serving
integrates through a **Dynamo Domain Adapter** and versioned service contracts; it does not vendor
or fork Dynamo as the default strategy.

The adapter:

- represents one Dynamo deployment/domain as one AX execution endpoint;
- discovers the domain's ready frontend, model inventory, capabilities, and bounded aggregate
  capacity;
- registers and renews a lease through the existing AX worker protocol;
- authenticates and proxies OpenAI-compatible JSON/SSE without selecting a Dynamo worker;
- maps only documented, version-pinned Dynamo metrics and health states;
- exposes typed `not-admitted` only when the adapter can prove Dynamo did not accept the request;
- records the Dynamo release, component images, backend, CUDA version, architecture, and config
  digest in deployment identity;
- optionally translates AX desired lifecycle state to a separately certified Dynamo/Kubernetes
  controller; lifecycle translation is not required on the inference data path.

The initial upstream integration baseline is Dynamo `v1.2.1`. Every AX release must pin a Dynamo
Git tag/commit and immutable NGC image digests in a compatibility manifest. `main`, floating tags,
and undocumented internal Python/Rust APIs are not release inputs. Upgrades proceed through
offline conformance, shadow traffic, canary, and rollback.

### NVIDIA PC and Thor are separate domains

NVIDIA's Dynamo support matrix covers ARM64 on Ubuntu 24.04 and NVIDIA Blackwell, and NVIDIA's Thor
documentation demonstrates ARM64 CUDA 13 containers. This makes a Thor integration plausible; it
does not constitute Dynamo-on-Thor certification.

Therefore:

- PC and Thor have different domain IDs, pool IDs, hardware classes, capacity models, and rollout
  tracks;
- no tensor/pipeline parallel group spans PC and Thor;
- TensorRT engines, quantized artifacts, profiling data, and planner coefficients are not shared by
  default;
- cross-domain failover requires the same semantic-equivalence evidence as Mac-to-NVIDIA failover;
- Thor remains `experimental` until a pinned Dynamo/backend/container combination passes the live
  conformance, performance, thermal, memory, restart, and soak gates;
- loss of the Thor domain must not impair PC or Mac domain control paths.

### Mac integration

Mac nodes continue to run:

```text
AX Serving -> ax-runtime-agent -> ax-engine-server
```

The agent owns AX Engine readiness normalization, inventory, model identity, credential isolation,
cancellation, and byte-preserving proxying. AX Engine owns all inference semantics. The
`ax-thor-agent` crate name is historical; its `ax-runtime-agent` binary is the target Mac adapter.
A later crate rename is packaging cleanup, not an architectural change.

### Identity and equivalence

Clients address logical model aliases. Every target deployment records:

- domain and pool ID;
- execution owner and hardware class;
- runtime/Dynamo/backend versions and immutable artifact identifiers;
- model revision or artifact digest;
- tokenizer and template digests;
- quantization and supported operations/capabilities;
- context and output limits;
- trust/residency classification;
- certification artifact and policy version.

Two deployments are not equivalent merely because their model strings match. Cross-domain retry or
failover is disabled unless both deployments belong to an operator-approved equivalence class and
all required identity and workload tests pass. Bit-identical output is not promised.

### Retry ownership and response commitment

- AX Serving may make at most one cross-domain retry after a connection failure or authenticated,
  typed pre-admission rejection and only before response headers/body are committed.
- AX Serving never retries an arbitrary `5xx`, an ambiguous post-admission disconnect, or a stream
  after its first committed byte.
- Dynamo owns any in-domain migration, cancellation, or retry. AX Serving observes one domain
  attempt and must not also retry a Dynamo worker attempt.
- The adapter must propagate one AX request ID and attempt ID while preserving Dynamo trace context.
- A request admitted by one domain never migrates to another domain.

### State and feedback

AX Serving stores only bounded federation state: domain leases, deployment identity, reservations,
policy versions, reason codes, and optional tenant-scoped soft affinity. It does not store prompts,
KV blocks, Dynamo's worker index, or durable agent memory.

Feedback follows:

```text
observe -> offline evaluation -> versioned policy -> shadow -> canary -> promote/rollback
```

Production online self-modification is not part of this decision. Agent-session hints, when
implemented, remain privacy-preserving soft locality hints and never add planner, tool, MCP,
workflow, sandbox, or memory ownership to AX Serving.

### Packaging and deployment

- AX Serving gateway and AX runtime-agent images remain CPU-only and multi-architecture.
- The AX Helm chart installs the federation gateway and its control-plane dependencies, not
  Dynamo, GPU operators, AX Engine, or model weights.
- Dynamo is installed and operated from its pinned upstream release artifacts in its own namespace
  or failure domain.
- Mac agents may be external to Kubernetes and connect through an operator-provided trusted private
  network.
- `/readyz` reports control-plane readiness by default; `/routablez` reports available execution
  capacity. A fresh gateway can become ready before any domain registers.
- Active-active AX gateways require Redis/Valkey. Dynamo retains its own required state services;
  the two systems do not share internal databases.

## Consequences

### Positive

- AX Serving has a clear value proposition that Dynamo alone does not provide: one governed fleet
  across Apple Silicon, NVIDIA PCs, and Thor.
- NVIDIA-local performance improves with upstream Dynamo work without AX reimplementing it.
- Mac investment remains useful for privacy, locality, energy, offline, and installed-capacity use
  cases.
- PC and Thor can evolve independently without unsafe shared assumptions.
- Existing AX Serving admission, identity, equivalence, safe retry, HA, and packaging work remains
  directly useful.
- A thin, pinned adapter limits blast radius when Dynamo changes.

### Negative

- Operators run two control systems with an explicit federation boundary.
- The Dynamo adapter and compatibility manifest become maintained integration surfaces.
- Cross-domain telemetry is necessarily coarser than Dynamo's internal worker telemetry.
- End-to-end certification requires Mac, PC, and Thor hardware and multiple failure environments.
- Some existing AX per-worker CUDA code becomes compatibility-only rather than the target path.

### Neutral trade-offs

- AX Serving may make a less locally optimal NVIDIA choice than Dynamo's own router; this is
  intentional because AX chooses a domain, not a worker.
- One additional federation hop is accepted only if measured overhead remains inside the PRD gate.
- Small single-runtime users should continue to call AX Engine or Dynamo directly.

## Alternatives considered

### A. Use vLLM on every CUDA node and remove AX Serving

Valid for a homogeneous NVIDIA-only fleet. Rejected as the AX product architecture because it
provides no Mac/Thor federation, cross-domain identity, tenant policy, or one private-fleet audit
boundary.

### B. Make AX Serving manage every vLLM/SGLang/TensorRT-LLM worker

Rejected. It duplicates Dynamo's router and planner, creates conflicting retry/cache decisions, and
turns AX Serving into a weaker NVIDIA control plane.

### C. Let Dynamo manage AX Engine nodes directly

Rejected for the initial architecture. A custom Dynamo backend would couple the Mac runtime to
Dynamo's release and Python/control stack and would still not supply AX's cross-domain tenant and
equivalence policy. It may be prototyped later only if it preserves AX Engine ownership and shows a
measured operational advantage.

### D. Embed or fork Dynamo inside AX Serving

Rejected. It expands the dependency, security, and upgrade surface and loses upstream compatibility.
Contribute generally useful fixes upstream; keep AX-specific federation logic outside Dynamo.

### E. Put PC and Thor workers in one Dynamo domain

Rejected by default. CPU architecture, CUDA/container baseline, memory behavior, thermal envelope,
artifact compatibility, and planner calibration differ. A future merged domain requires explicit
evidence and a separate decision.

### F. Build a universal cognitive runtime now

Rejected. Task-aware routing, context optimization, verification, and cost intelligence may become
versioned federation policies, but only after the deterministic domain boundary is certified and a
measured workload proves value.

## Migration

1. Adopt this ADR, the consolidated PRD, and the consolidated technical specification.
2. Retain the current Mac AX runtime-agent path and certify it against a pinned AX Engine release.
3. Introduce the execution-domain abstraction as an additive protocol/config change.
4. Implement a Dynamo Domain Adapter in shadow inventory mode, then proxy mode.
5. Create a distinct NVIDIA PC Dynamo domain and pass API, identity, stream, cancel, drain, fault,
   and performance certification.
6. Add Thor as a separate experimental domain; promote it only after its independent gates pass.
7. Disable direct per-worker vLLM/SGLang registration in the production profile; retain it only as
   a time-bounded compatibility path.
8. Add policy records and offline replay before enabling cost/SLO-aware domain selection.
9. Remove compatibility paths only after migration tooling and a released replacement exist.

## Compliance checks

A future change complies only when all answers are yes:

- Does AX select a domain while Dynamo selects NVIDIA workers?
- Can the gateway build without runtime/Dynamo/accelerator SDKs?
- Are PC and Thor separate unless a later certification explicitly merges them?
- Is the full Dynamo release and image set immutable and recorded?
- Does missing identity, capability, or telemetry fail closed or receive a conservative penalty?
- Can AX prove it never retries after admission ambiguity or response commitment?
- Are public, AX control, adapter dispatch, Dynamo, and runtime credentials distinct?
- Does any learning change pass offline, shadow, canary, and rollback gates?
- Can the same public API operate with no NVIDIA domain and with no Mac domain?

## References

- [NVIDIA Dynamo overall architecture](https://docs.nvidia.com/dynamo/design-docs/overall-architecture)
- [NVIDIA Dynamo support matrix](https://docs.nvidia.com/dynamo/latest/resources/support-matrix)
- [NVIDIA Dynamo release artifacts](https://docs.nvidia.com/dynamo/latest/resources/release-artifacts)
- [NVIDIA Dynamo unified backend guide](https://docs.nvidia.com/dynamo/latest/backends/custom-backend/writing-unified-backends)
- [NVIDIA Jetson AGX Thor CUDA setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html)
- [AX Serving runtime responsibility inventory](../../docs/contracts/ax-serving-runtime-responsibility-inventory.md)
