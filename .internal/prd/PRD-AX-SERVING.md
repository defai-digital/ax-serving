# PRD - AX Serving

> **Status**: Canonical
> **Date**: 2026-05-25
> **Last Reviewed**: 2026-05-25
> **Owner**: AX Serving Team
> **Type**: Product + Engineering PRD
> **Audience**: Product, engineering, runtime-adapter owners, AX Fabric
> integrators, and enterprise deployment teams
> **Supersedes**:
> - `PRD-AX-SERVING-v3.0.md`
> - `PRD-AX-SERVING-OSS-ENTERPRISE-BOUNDARY-v1.0.md`
> - `PRD-AX-SERVING-ENTERPRISE-EXECUTION-v1.0.md`
> - `PRD-AX-SERVING-INFERENCE-EFFICIENCY-v1.0.md`
> **Related**:
> - [README](../../README.md)
> - [ICP and Demand](../../docs/icp-and-demand.md)
> - [Market Positioning](../../docs/market-positioning.md)
> - [Competitive Landscape](../../docs/competitive-landscape.md)
> - [Maintainability Refactor Plan](../../docs/maintainability-refactor-plan.md)
> - [AX Serving Node Contract](../../docs/contracts/ax-serving-node-contract.md)
> - [Runtime Responsibility Inventory](../../docs/contracts/ax-serving-runtime-responsibility-inventory.md)
> - [AX Fabric Runtime Contract](../../docs/contracts/ax-fabric-runtime-contract.md)
> - [AX Serving Public Contract Inventory](../../docs/contracts/ax-serving-public-contract-inventory.md)
> - [AX Serving PRD Implementation Plan](./IMPLEMENTATION-PLAN-AX-SERVING.md)
> - [ADR-001: Target Market And Product Scope](../adr/ADR-001-target-market-and-product-scope.md)
> - [ADR-002: Software Value Boundary Vs Hardware Deployment Context](../adr/ADR-002-software-value-boundary-vs-hardware-deployment-context.md)
> - [ADR-012: Multi-Worker Orchestration](../adr/ADR-012-local-multi-worker-orchestration.md)

---

## 1. Executive Summary

AX Serving is the serving and orchestration control plane for private AI
inference fleets.

It should behave like the serving/control-plane layer above inference engines,
not like another inference engine. AX Serving owns the northbound API,
authorization, routing, model and node inventory, fleet policy, observability,
and operator workflows. It delegates token generation, batching kernels, memory
management, and model execution to runtime nodes.

Target runtime nodes:

- Mac nodes running `ax-engine`
- PC CUDA nodes running vLLM
- NVIDIA Thor nodes running vLLM

This split prevents AX Serving from duplicating work that belongs in ax-engine
or vLLM, while still giving teams one private serving surface across mixed
hardware.

---

## 2. Product Thesis

AX Serving is the private fleet serving layer above heterogeneous inference
runtimes.

The core product value is:

- OpenAI-compatible serving API
- model and runtime inventory
- request routing and placement
- worker registration, heartbeat, drain, and replacement
- queueing, admission, retries, and overload behavior
- metrics, audit, diagnostics, and operator views
- stable contracts for AX Fabric and other private AI applications

AX Serving is not:

- a low-level inference engine
- a kernel/runtime optimization project
- a duplicate of ax-engine on Mac
- a replacement for vLLM on CUDA/NVIDIA systems
- a desktop local AI app
- a generic hyperscale cluster platform

The product claim is that AX Serving makes private inference runtimes operable
as one serving system.

### 2.1 Decision Summary

| Decision | Product rule |
|---|---|
| Product category | Runtime-neutral private AI serving and orchestration control plane |
| Runtime boundary | ax-engine and vLLM own inference execution; AX Serving owns serving policy |
| Primary node model | Mac ax-engine nodes, PC CUDA vLLM nodes, and NVIDIA Thor vLLM nodes |
| Integration style | Explicit node contracts and thin adapters, not hidden runtime crates |
| Public core | Buildable and useful without proprietary dependencies |
| Enterprise extension | Integrates through public service/node contracts |

---

## 3. Target Users And Deployments

Primary users:

- platform, infra, IT, and AI platform teams
- SMEs and enterprise departments operating private AI workloads
- AX Fabric deployments that need a stable execution control plane

Typical deployment:

- one AX Serving gateway/control plane
- multiple inference nodes
- more than one model in service
- mixed hardware classes over time
- private or controlled network
- OpenAI-compatible API expected by internal applications

Primary node classes:

| Node class | Runtime owner | Role |
|---|---|---|
| Mac | ax-engine | Apple Silicon inference node |
| PC CUDA | vLLM | NVIDIA CUDA inference node |
| NVIDIA Thor | vLLM | Thor-class accelerator inference node |

---

## 4. Product Boundary

### 4.1 AX Serving Owns

- northbound REST/gRPC serving contracts
- auth, API keys, request IDs, security headers, and trust-zone policy
- model registry at the serving/fleet level
- worker/node registry and capability inventory
- routing, placement, admission, queueing, retry, and drain policy
- runtime-neutral model lifecycle orchestration where supported by nodes
- metrics, diagnostics, audit, dashboard, and operator runbooks
- AX Fabric runtime contract
- compatibility rules for node adapters

### 4.2 Runtime Nodes Own

- model file loading into the inference runtime
- token generation
- tokenizer/runtime-specific prompt formatting
- KV cache storage and block management
- continuous batching and scheduler internals
- GPU/Metal/CUDA kernel selection
- quantization execution
- runtime-specific memory planning
- runtime-specific performance tuning

### 4.3 Out Of Scope For AX Serving

- custom Metal kernels
- CUDA kernels
- paged-attention implementation
- KV block allocator implementation
- vLLM internals
- ax-engine internals
- model architecture implementation
- training or fine-tuning
- global multi-region cloud orchestration
- per-tenant SaaS platformization

---

## 5. Current Baseline And Rework Need

The repository already has useful serving-control-plane foundations:

- OpenAI-compatible REST API
- gRPC serving/control paths
- model load, unload, reload, and registry behavior
- scheduler, queueing, adaptive concurrency, and per-model concurrency controls
- response-level cache and inflight cache deduplication
- embedding batching
- worker registration, heartbeat, drain, recovery, and fleet inventory
- dispatch policies including least-inflight, weighted round-robin,
  model-affinity, token-cost, and cache-affinity
- cache and batch telemetry in worker status
- metrics, dashboard, diagnostics, audit, and licensing surfaces
- Thor CLI/agent worker path as a worker-protocol foundation
- benchmark, soak, profile, compare, and regression runners
- CLI support for serving, tuning, doctor checks, config validation, status,
  smoke tests, and worker workflows

The current implementation also contains runtime/backend responsibilities that
should be reworked or reduced:

- embedded llama.cpp subprocess backend
- MLX subprocess backend path
- optional direct libllama backend path
- native ax-engine backend integration inside the serving crate
- local single-runtime behavior that overlaps with ax-engine's Mac runtime role
- backend-specific tuning/doctor behavior that should move toward node/runtime
  adapter checks

Target direction:

- AX Serving should become runtime-neutral serving infrastructure.
- Mac inference should be provided by ax-engine nodes.
- CUDA PC and NVIDIA Thor inference should be provided by vLLM nodes.
- AX Serving should communicate with these nodes through explicit serving/node
  contracts, not by owning their inference internals.

---

## 6. Product Requirements

### PR-1: Runtime-Neutral Serving API

AX Serving must expose one stable serving surface across runtime nodes.

Requirements:

- OpenAI-compatible chat, completions, embeddings, and model-listing APIs remain
  the primary northbound contract where supported by nodes
- runtime-specific differences are normalized where possible and surfaced
  explicitly where not possible
- clients do not need to know whether a request lands on ax-engine or vLLM
  unless they ask for runtime-specific placement

### PR-2: Inference Node Contract

AX Serving must define a node contract that lets runtimes join the fleet without
becoming internal Rust dependencies.

Requirements:

- node registration advertises runtime type, node class, hardware class, model
  inventory, capacity, queue state, health, version, and supported operations
- heartbeats report inflight, queue, memory/cache pressure where available,
  runtime health, and drain state
- nodes expose a runtime-compatible inference endpoint or proxy target
- optional model lifecycle operations are capability-gated per runtime
- node contract is documented as a public integration surface

### PR-3: Runtime Adapter Support

AX Serving must support the target runtime classes through adapters or agents.

Requirements:

- Mac nodes run ax-engine and register with AX Serving
- PC CUDA nodes run vLLM and register with AX Serving
- NVIDIA Thor nodes run vLLM and register with AX Serving
- adapters should be thin protocol bridges, not inference implementations
- runtime-specific health and metrics are translated into AX Serving's common
  node model

### PR-4: Model And Placement Semantics

AX Serving must reason about model placement at the fleet level.

Requirements:

- model identity is stable across runtimes
- model inventory includes runtime, node class, quantization/artifact format
  where known, context limits, modality support, and lifecycle capability
- routing can use requested model, runtime class, node pool, hardware class,
  health, load, queue state, and capability constraints
- placement failures produce operator-readable diagnostics

### PR-5: Routing, Admission, And Resilience

AX Serving must own fleet-level request policy.

Requirements:

- dispatch policies remain operator-selectable and documented
- routing is health-aware and capacity-aware
- queueing, overload, retry, and reroute behavior are predictable
- draining workers stop receiving new traffic and can complete existing work
- runtime-specific errors are mapped into stable serving-layer responses

### PR-6: Observability And Audit

Operators must be able to run the fleet from AX Serving surfaces.

Requirements:

- fleet health summary
- per-node runtime, hardware class, model inventory, health, queue, inflight,
  drain, and capability state
- per-model and per-runtime metrics
- audit records for admin, lifecycle, config-sensitive, and drain actions
- diagnostics that identify whether failures belong to AX Serving, ax-engine,
  vLLM, or infrastructure

### PR-7: Operator Tooling

AX Serving must reduce operational guesswork without duplicating runtime
tuners.

Requirements:

- doctor checks validate AX Serving config, auth, network exposure, node
  reachability, runtime compatibility, and common fleet misconfiguration
- tuning guidance focuses on serving-layer controls: queue depth, routing,
  timeouts, retries, health TTLs, and admission
- runtime-specific tuning is delegated to ax-engine or vLLM tooling/docs
- status and smoke-test commands work across ax-engine and vLLM nodes

### PR-8: AX Fabric Runtime Fit

AX Serving must remain the execution control plane for AX Fabric.

Requirements:

- AX Fabric can treat AX Serving as one serving endpoint
- model lifecycle, metrics, routing, and failure semantics are stable enough for
  governed private AI workflows
- AX Fabric does not depend on ax-engine, vLLM, or AX Serving internal Rust
  types directly

### PR-9: Security And Trust Zones

AX Serving must define clear trust boundaries between clients, gateway, and
runtime nodes.

Requirements:

- public serving API remains authenticated in production
- internal node registration and heartbeat surfaces are protected or isolated
- gateway-to-node traffic supports shared-token or internal-token models
- dashboard and diagnostics exposure is intentional and documented
- secrets and API keys are never logged

### PR-10: Maintainability And De-Duplication

The implementation must remove unnecessary overlap with inference runtimes.

Requirements:

- AX Serving code should not implement model architecture logic
- backend-specific code should move toward adapter modules with narrow
  contracts
- duplicated serving/orchestrator helper logic is centralized where behavior is
  intentionally shared
- current embedded runtime paths are reviewed for removal, extraction, or
  adapter conversion
- CLI entrypoints primarily parse and dispatch to focused modules
- `cargo fmt --all`, `cargo clippy --workspace --tests -- -D warnings`, and
  `cargo test --workspace` remain release gates

## 7. Success Metrics

The PRD should be evaluated by observable product and engineering outcomes, not
by the amount of runtime code inside AX Serving.

| Area | Success measure |
|---|---|
| Product clarity | Public docs consistently describe AX Serving as serving/control-plane software, not an inference engine. |
| Runtime neutrality | Node inventory and routing can represent at least `ax_engine` and `vllm` runtimes without runtime-specific gateway branches. |
| Operability | Operators can inspect fleet health by worker, pool, node class, backend, runtime, model inventory, and drain state. |
| Compatibility | Existing OpenAI-compatible REST behavior and legacy worker registration continue passing regression tests during migration. |
| Security | Public and internal APIs keep documented auth/trust-zone rules, and secrets remain out of logs and committed files. |
| Maintainability | New runtime support enters through adapter/contract surfaces unless an ADR explicitly approves a temporary compatibility path. |
| Release quality | `cargo fmt --all -- --check`, `cargo clippy --workspace --tests -- -D warnings`, and `cargo test --workspace` stay green for release candidates. |

---

## 8. Requirement Traceability

| Requirement | Primary evidence |
|---|---|
| PR-1 Runtime-neutral serving API | README, OpenAI-compatible REST tests, AX Fabric runtime contract |
| PR-2 Inference node contract | [AX Serving Node Contract](../../docs/contracts/ax-serving-node-contract.md) |
| PR-3 Runtime adapter support | `ax-runtime-agent`, node contract, multi-worker runbook, and Thor/runtime-agent e2e tests |
| PR-4 Model and placement semantics | Worker registry, admin fleet summary, routing/placement tests |
| PR-5 Routing, admission, resilience | Multi-worker runbook, orchestration tests, scheduler tests |
| PR-6 Observability and audit | Admin status/diagnostics/audit surfaces and model-management tests |
| PR-7 Operator tooling | CLI doctor, config validate, status, smoke-test, tune contracts |
| PR-8 AX Fabric runtime fit | AX Fabric runtime contract |
| PR-9 Security and trust zones | Public contract inventory, internal node contract, auth/security tests |
| PR-10 Maintainability and de-duplication | Runtime responsibility inventory, maintainability refactor plan, and implementation-plan migration slices |

Traceability is intentionally artifact-based. A requirement is not complete
because it appears in this PRD; it is complete when the linked contract,
implementation, test, and runbook evidence agree.

---

## 9. Open-Source And Enterprise Boundary

The public repository should remain a useful serving/control-plane core.

Open-source scope:

- AX Serving gateway and worker/node protocol
- Mac ax-engine node integration
- runtime-neutral model and node registry
- routing, queueing, drain, metrics, diagnostics, dashboard, and admin protocols
- public SDKs, benchmark clients, scripts, runbooks, and operator docs

Commercial or enterprise scope may include:

- supported NVIDIA Thor fleet packaging
- supported vLLM node integration bundles for enterprise hardware
- enterprise auth, governance, compliance export, and private deployment
  packaging
- supported private integrations and fleet operations tooling
- commercial licensing rights for non-AGPL use of the public core

Boundary rules:

1. The public repository remains buildable without proprietary dependencies.
2. Enterprise-only value integrates through service/node contracts, not hidden
   private crates inside the public workspace.
3. AX Serving must not absorb ax-engine or vLLM inference internals.
4. Enterprise runtime support must preserve the public node contract.
5. Public Mac/ax-engine operation should remain credible and complete for its
   target use case.

---

## 10. Consolidated Execution Plan

Sequencing lives in
[AX Serving PRD Implementation Plan](./IMPLEMENTATION-PLAN-AX-SERVING.md).
This PRD keeps only the durable product boundary and current execution state.

Current state:

- the node contract is documented and treats runtime, hardware class, capacity,
  health, supported operations, and model inventory as first-class metadata
- `ax-runtime-agent` provides the generic runtime-node adapter path for Mac
  ax-engine, PC CUDA vLLM, and NVIDIA Thor vLLM deployments
- `ax-thor-agent` remains as a compatibility binary for existing Thor
  deployments
- admin status and orchestration tests cover runtime metadata and runtime fleet
  grouping
- embedded llama.cpp, MLX, libllama, and direct native paths are compatibility
  paths, not the product direction

Remaining work:

- improve ax-engine-specific health, model inventory, and metrics translation
  behind the generic runtime-node adapter
- continue extracting, deprecating, or quarantining embedded runtime paths after
  node-adapter replacements are validated
- deepen dashboard, diagnostics, and operator workflows by runtime class,
  hardware class, pool, and model placement
- keep AX Fabric validation aligned to the public serving and node contracts

---

## 11. Non-Functional Requirements

### Reliability

- existing tests remain green during rework
- multi-worker behavior remains regression-covered
- node failures degrade to clear gateway errors
- routing and admission degrade safely when optional runtime telemetry is absent

### Operability

- admin surfaces are understandable to platform and infra operators
- setup, status inspection, drain, replacement, and recovery docs are sufficient
- CLI output supports both human use and automation where appropriate

### Product Clarity

- AX Serving is described as serving/control-plane software
- ax-engine is described as Mac inference runtime
- vLLM is described as CUDA/Thor inference runtime
- hardware claims remain deployment context, not AX Serving software claims

### Backward Practicality

- current useful capabilities are preserved until replacement node-adapter paths
  exist
- public APIs and worker contracts receive migration notes before breaking
  changes
- older workers degrade gracefully where newer heartbeat fields are absent

### Security

- secrets and API keys must not be logged or committed
- production docs require API-key or equivalent protection
- internal node APIs are protected by loopback, network policy, or token
  controls
- enterprise integrations do not weaken public-core security defaults

---

## 12. Acceptance Criteria

The PRD is satisfied when:

1. AX Serving is accurately described as a runtime-neutral private AI serving
   and orchestration control plane.
2. AX Serving no longer claims ownership of low-level inference execution.
3. Mac inference is positioned as ax-engine node responsibility.
4. PC CUDA and NVIDIA Thor inference are positioned as vLLM node
   responsibility.
5. AX Serving has a documented node contract for registration, heartbeat,
   capability advertisement, health, drain, metrics, and inference forwarding.
6. Routing and placement operate on runtime/node capabilities rather than
   embedded backend assumptions.
7. Operator workflows cover join, health inspection, drain, replacement,
   recovery, doctor checks, and fleet troubleshooting across target node
   classes.
8. AX Fabric can treat AX Serving as a stable serving/control-plane layer.
9. Historical ADRs and docs do not create conflicting active product claims.
10. Formatting, lint, and workspace test gates remain expected release
    validation commands.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| AX Serving continues duplicating ax-engine runtime work | Wasted engineering effort and unclear ownership | Move Mac inference responsibility to ax-engine nodes |
| vLLM integration becomes too runtime-specific | Gateway complexity grows | Keep adapters thin and contract-driven |
| Removing embedded paths breaks current users | Migration friction | Phase deprecation and keep compatibility paths until node adapters are ready |
| Node contract is too weak | Runtime integrations become one-off bridges | Define capability, health, metrics, lifecycle, and error-shape requirements early |
| Product messaging drifts back to hardware claims | Market confusion | Keep runtime/hardware fit as deployment context |
| Enterprise integrations bypass public contracts | Public core becomes less useful | Require service/node-contract integration |
| Mixed-runtime observability is inconsistent | Operators cannot diagnose incidents | Normalize metrics and diagnostics at the gateway |

---

## 14. Governance And Review

Best-practice rules for maintaining this PRD:

- This file is the canonical PRD; implementation plans, ADRs, and runbooks must
  link back here when they change product scope.
- Product requirements should be stable and outcome-oriented. Implementation
  plans should carry sequencing, ownership, and tactical milestones.
- Major boundary changes require an ADR or an explicit update to the requirement
  traceability table.
- Historical PRD names may remain only in `Supersedes` metadata or migration
  notes, not as active planning sources.
- Review this PRD before release planning, runtime-adapter milestones, or
  enterprise packaging changes.

---

## 15. Review Notes

This file is the single canonical PRD. Historical PRDs were consolidated into
this document and removed as standalone planning artifacts.

The 2026-05-25 rework review changes the target direction: AX Serving should
not duplicate inference-runtime responsibilities already owned by ax-engine or
vLLM. The product is now explicitly framed as the serving/control-plane layer
for Mac ax-engine nodes, PC CUDA vLLM nodes, and NVIDIA Thor vLLM nodes.
