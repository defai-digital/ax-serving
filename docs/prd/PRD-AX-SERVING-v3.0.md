# PRD — AX Serving v3.0 Department-Scale Private AI Fleet Control Plane

> **Status**: Draft
> **Date**: 2026-03-28
> **Owner**: AX Serving Team
> **Type**: Product + Engineering PRD
> **Related**:
> - [README](/Users/akiralam/code/ax-serving/README.md)
> - [ICP and Demand](/Users/akiralam/code/ax-serving/docs/icp-and-demand.md)
> - [Market Positioning](/Users/akiralam/code/ax-serving/docs/market-positioning.md)
> - [Competitive Landscape](/Users/akiralam/code/ax-serving/docs/competitive-landscape.md)
> - [Maintainability Refactor Plan](/Users/akiralam/code/ax-serving/docs/maintainability-refactor-plan.md)
> - [Open-Source / Enterprise Boundary PRD](/Users/akiralam/code/ax-serving/docs/prd/PRD-AX-SERVING-OSS-ENTERPRISE-BOUNDARY-v1.0.md)
> - [Enterprise Extraction And Release Execution PRD](/Users/akiralam/code/ax-serving/docs/prd/PRD-AX-SERVING-ENTERPRISE-EXECUTION-v1.0.md)
> - [ADR-001](/Users/akiralam/code/ax-serving/docs/adr/ADR-001-target-market-and-product-scope.md)
> - [ADR-002](/Users/akiralam/code/ax-serving/docs/adr/ADR-002-software-value-boundary-vs-hardware-deployment-context.md)
> - [ADR-012](/Users/akiralam/code/ax-serving/docs/adr/ADR-012-local-multi-worker-orchestration.md)

---

## 1. Executive Summary

`v3.0` defines AX Serving as a product, not just a serving codebase.

The target is to make AX Serving the default control plane for
**department-scale private AI fleets**:

- multi-model
- multi-worker
- team-operated
- private deployment
- governed or internal AI workloads

The product must fill the gap between:

- single-user local runtimes and desktop tools
- large GPU-first serving platforms built for hyperscale environments

AX Serving `v3.0` should be the operational layer that lets SMEs and enterprise
departments run private AI fleets with one northbound API surface and one
control plane, without needing to assemble that control plane themselves on top
of a raw inference engine.

---

## 2. Problem Statement

The current market has strong products at both ends:

- local runtime convenience
- large-scale GPU serving frameworks

But there is still a product gap for teams that need:

- more than one model
- more than one worker
- private deployment
- fleet-level health and routing control
- admin, metrics, audit, and policy visibility
- mixed worker classes over time

These teams are usually too advanced for single-machine local tools and too
small or too constrained to adopt a hyperscale GPU-serving stack.

Without AX Serving, they typically face one of three bad outcomes:

1. operate multiple ad hoc local runtimes with poor visibility
2. build custom control-plane logic on top of an inference engine
3. over-adopt a heavier serving platform than their environment actually needs

`v3.0` should solve that product gap directly.

---

## 3. Product Thesis

AX Serving is the control plane for department-scale private AI fleets.

The core software value is:

- multi-model serving
- mixed-worker orchestration
- fleet routing and lifecycle control
- operator visibility and reliability surfaces

AX Serving is not positioned as:

- a desktop local AI app
- a raw inference engine
- a generic hyperscale distributed inference platform

Hardware choice matters to deployment design, but the product claim is not
"hardware efficiency software." The product claim is that AX Serving makes
heterogeneous private inference hardware operable as one serving system.

---

## 4. Target Users And Buyers

### 4.1 Primary ICP

SMEs and enterprise departments operating private AI workloads for internal use.

Typical scope:

- fewer than ~100 users or operators directly supported by the deployment

Primary internal champions:

- platform engineering
- infra / systems teams
- AI platform teams
- IT operations
- knowledge systems / internal tooling teams

### 4.2 Deployment Context

Typical environment:

- private or controlled deployment
- more than one model in service
- more than one serving worker
- pressure to keep one serving layer for multiple internal applications
- no appetite for building a full GPU-cluster platform if it is not necessary

### 4.3 Hardware Context

This PRD assumes deployments may include:

- Mac-led control planes
- Thor-class workers for standard high-parallel `<=70B` workloads
- Mac Studio-class workers for larger-memory model tiers, including `>70B`
- future heterogeneous workers

This is deployment context, not the core software claim.

---

## 5. Goals

### G1. Product Identity

Make AX Serving clearly implement the product category defined in ADR-001:

- department-scale private AI fleet control plane

### G2. Multi-Model Fleet Operation

Make multi-model serving a first-class product capability, not an incidental
API behavior.

### G3. Mixed-Worker Fleet Operation

Make multi-worker and mixed-worker orchestration a first-class capability with
clear operational workflows.

### G4. Operator-Grade Control

Expose enough lifecycle, queue, routing, health, metrics, audit, diagnostics,
and policy visibility for team-operated deployments.

### G5. Private Deployment Fit

Keep the product private-deployment-first and practical for SME / departmental
adoption.

### G6. Ecosystem Fit

Make AX Serving the natural execution layer for governed private AI stacks,
especially AX Fabric.

---

## 6. Non-Goals

`v3.0` does not aim to become:

- a desktop AI application
- a full enterprise IAM / tenant platform
- a training or fine-tuning system
- a generic hyperscale GPU cluster platform
- a replacement for low-level inference engines
- an autoscaling cloud control plane
- a global multi-region distributed serving product

`v3.0` also does not claim hardware economics as a software superpower. Those
remain deployment-context considerations under ADR-002.

---

## 7. Current Baseline

The codebase already provides important building blocks:

- OpenAI-compatible REST API
- gRPC serving/control paths
- model load / unload / reload
- worker registration and heartbeat
- dispatch policies and queueing
- worker drain and recovery behavior
- metrics, dashboard, diagnostics, and audit surfaces
- Thor-managed worker path
- fleet inventory concepts such as worker pools and node classes

`v3.0` should treat these as foundations to harden, unify, and productize,
not as reasons to expand scope indiscriminately.

The current implementation also carries maintainability debt that will slow
down `v3.0` delivery if left untreated:

- oversized multi-responsibility route and orchestration modules
- duplicated helper logic across serving and orchestrator surfaces
- duplicated test doubles and fixtures
- centralized CLI command logic that is difficult to extend safely
- large integration-test files with weak ownership boundaries

---

## 8. v3.0 Scope

### 8.1 In Scope

- multi-model fleet serving semantics
- mixed-worker orchestration and fleet identity
- fleet-level operational visibility
- deployment guidance by worker class and model tier
- productized admin and operator workflows
- stronger contract for AX Fabric and similar private AI stacks
- documentation and packaging clarity for the target market
- maintainability refactor required to make the control plane sustainable to
  extend and operate

### 8.2 Out Of Scope

- generic public-cloud orchestration features
- global job scheduling
- distributed training
- cross-region replication
- full autoscaling orchestration
- per-tenant SaaS control-plane platformization
- speculative rewrites that do not clearly improve maintainability or product fit

---

## 9. Functional Requirements

### FR-1: Multi-Model Serving As A First-Class Capability

AX Serving must treat multi-model serving as a product capability, not just a
registry detail.

Requirements:

- operators can load and operate multiple models in one fleet
- health, metrics, and admin views expose model-level fleet state
- routing logic can make decisions with explicit model identity in scope
- documentation must define recommended operating patterns for multiple model tiers

### FR-2: Mixed-Worker Fleet Registry

The fleet registry must support heterogeneous worker classes as a stable product
concept.

Requirements:

- worker metadata supports node class, worker pool, backend, hardware label, and model inventory
- fleet inventory APIs expose these distinctions clearly
- the registry remains stable across worker restarts and re-registration flows
- mixed-worker fleets are first-class in docs and admin summaries

### FR-3: Fleet Routing And Placement Control

Routing must support department-scale fleet operation, not just raw dispatch.

Requirements:

- support existing dispatch policies with clear operator-facing semantics
- preserve health-aware routing and reroute behavior
- support placement hints via model identity, pool, worker class, and health state
- support clear routing behavior across standard-operation and larger-memory worker tiers

### FR-4: Operator Lifecycle Workflows

AX Serving must make routine fleet operations productized.

Requirements:

- install / join / status / drain / drain-complete / replacement workflows exist and are documented
- rolling worker replacement is supported without taking down the whole fleet
- failure and recovery behavior is explicit in runbooks and admin surfaces

### FR-5: Fleet-Level Visibility

AX Serving must expose enough information for a real operator to run the fleet.

Requirements:

- fleet-level health summary
- per-worker health and drain state
- inflight, queue, reroute, and model inventory visibility
- per-worker and fleet-level metrics
- auditability for admin actions and license/config changes
- diagnostics surfaces usable during deployment and incident response

### FR-6: Model Lifecycle Predictability

Model lifecycle behavior must be stable and predictable.

Requirements:

- load / unload / reload semantics are consistent across deployment modes
- failure cases are visible and documented
- health and readiness semantics remain understandable with multiple models and workers

### FR-7: Private Deployment Ergonomics

The product must remain practical for the target market.

Requirements:

- default deployment path does not require heavyweight external infrastructure
- private deployment docs are credible for SME / department operators
- mixed-worker deployments do not require users to design the control plane themselves

### FR-8: AX Fabric Runtime Fit

AX Serving must remain a strong runtime/control-plane fit for AX Fabric.

Requirements:

- serving contract remains clear and stable
- lifecycle and metrics semantics are strong enough for AX Fabric integration
- governed/private AI deployment story remains coherent across both products

### FR-9: Maintainable Control-Plane Codebase

`v3.0` must improve the internal maintainability of the codebase enough that
future product work does not continue to accumulate structural risk.

Requirements:

- oversized multi-responsibility modules are split by domain responsibility
- duplicated serving/orchestrator helper logic is centralized where behavior is
  intended to be the same
- duplicated test doubles and fixtures are moved into shared test-support
  locations
- CLI command implementation is decomposed so the main entrypoint is primarily
  parsing and dispatch
- maintainability guardrails exist to prevent immediate regression after the refactor

---

## 10. Non-Functional Requirements

### NFR-1: Reliability

- existing test coverage must remain green
- multi-worker behavior must remain regression-covered
- operator-facing lifecycle flows must be deterministic enough for repeatable runbooks

### NFR-2: Operability

- admin surfaces must be understandable to platform/infra operators
- docs must be sufficient for setup, status inspection, drain, and recovery workflows

### NFR-3: Product Clarity

- software claims and hardware claims must remain separate
- docs, PRDs, and future ADRs must align with ADR-001 and ADR-002

### NFR-4: Backward Practicality

- current useful capabilities should be preserved unless explicitly deprecated
- v3.0 must feel like a hardening and productization release, not a speculative rewrite

### NFR-5: Maintainability

- behavior-preserving refactors are preferred over large rewrites
- no single production Rust file should exceed 1500 LOC, except engine adapter
  files explicitly exempted for backend/FFI reasons
- no single integration-test Rust file should exceed 1500 LOC
- shared helpers introduced by refactor must have direct unit or integration
  coverage
- `cargo fmt --all`, `cargo clippy --workspace --tests -- -D warnings`, and
  `cargo test --workspace` must pass after the refactor work lands

---

## 11. Workstreams

### WS1 — Productized Multi-Model Fleet Semantics

Deliverables:

- operator-facing multi-model guidance
- clearer model-tier semantics in docs and admin views
- model-level fleet reporting requirements

### WS2 — Mixed-Worker Orchestration Hardening

Deliverables:

- consistent worker-class concepts
- clearer worker pool / node class semantics
- fleet-level routing and placement guidance

### WS3 — Operator Experience

Deliverables:

- improved fleet admin documentation
- stronger drain / replacement workflows
- clearer incident and recovery views

### WS4 — Contract And Integration Stability

Deliverables:

- stable AX Fabric-facing runtime contract
- lifecycle and metrics semantics aligned with product positioning
- clearer public/private deployment expectations

### WS5 — Product Packaging And Messaging

Deliverables:

- README and docs aligned to target market
- licensing and packaging language aligned with the product category
- deployment fit explained without overstating hardware-driven benefits as software features

### WS6 — Engineering Maintainability Refactor

Deliverables:

- REST and orchestrator surfaces decomposed into smaller domain-owned modules
- shared helper logic extracted for request shaping, audit helpers, and startup/reporting primitives
- shared test-support fixtures for backend doubles, temp GGUF builders, and env-var serialization
- CLI command handling decomposed into smaller modules with explicit ownership
- maintainability guardrails documented and enforced

---

## 12. Milestones

### M1 — Product Baseline Aligned

Exit:

- README, ADRs, ICP, market positioning, and PRD all tell the same product story
- mixed-worker and multi-model product claims are explicit and consistent

### M2 — Fleet Operation Credible

Exit:

- worker lifecycle workflows are documented and testable
- fleet inventory and admin surfaces are coherent for operators
- mixed-worker deployment guidance is usable

### M3 — Maintainability Foundation Landed

Exit:

- giant route/orchestration entry files are decomposed into smaller modules
- duplicated helper logic has dedicated shared homes
- shared test support replaces repeated backend and fixture implementations
- the codebase is easier to change without expanding review radius

### M4 — v3.0 Product Complete

Exit:

- AX Serving can be presented as a department-scale private AI fleet control plane
- the claim is backed by product behavior, operator workflows, and supporting docs

---

## 13. Acceptance Criteria

`v3.0` is complete when all of the following are true:

1. AX Serving can be described accurately as a control plane for
   department-scale private AI fleets.
2. Multi-model serving is visible as a first-class product capability in docs,
   admin surfaces, and workflows.
3. Mixed-worker orchestration is credible as a first-class product capability.
4. README, ADRs, market docs, and PRD are aligned with no category confusion.
5. Hardware deployment fit is documented clearly without being mislabeled as a
   pure software superpower.
6. AX Fabric can still treat AX Serving as a stable serving/control-plane layer.
7. Operators have documented workflows for worker join, health inspection,
   drain, replacement, and fleet-level troubleshooting.
8. The maintainability refactor has reduced obvious structural duplication across
   serving and orchestrator surfaces.
9. Shared backend test doubles and test fixtures are reused from dedicated
   support locations instead of being redefined repeatedly.
10. Large production and integration-test modules have been reduced to clearer
    domain-focused units.
11. The refactored codebase still passes formatting, lint, and workspace test
    gates.

---

## 14. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Product scope drifts back toward generic serving claims | Market confusion and weak differentiation | Enforce ADR-001 in PRDs and docs |
| Hardware context is marketed as software value | Weak or misleading positioning | Enforce ADR-002 language boundary |
| Mixed-worker orchestration remains technically present but product-invisible | Feature exists but does not help positioning | Make mixed-worker operation explicit in docs and admin surfaces |
| v3.0 tries to become hyperscale cluster software | Wrong competitive arena | Keep department-scale target explicit |
| AX Fabric fit becomes secondary | Stack story weakens | Treat governed private AI stack fit as a core acceptance criterion |
| Maintainability work turns into broad rewrite churn | Delivery slows and regressions rise | Prefer behavior-preserving extraction and phase the refactor |
| Duplication is replaced with over-abstraction | Code becomes harder to reason about | Prefer focused modules over generic frameworks |

---

## 15. Release Statement

The intended release statement for `v3.0` is:

```text
AX Serving v3.0 is the control plane for department-scale private AI fleets:
multi-model, mixed-worker, operator-visible, and private-deployment-first.
```

That statement should be supportable from:

- product docs
- operator runbooks
- orchestration behavior
- admin and visibility surfaces
- integration behavior with AX Fabric

---

## Appendix A — Maintainability Refactor Scope

This appendix defines the expected engineering refactor shape inside `v3.0`.

### A.1 Primary Hotspots

Current files that require decomposition or structural cleanup:

- `crates/ax-serving-api/src/rest/routes.rs`
- `crates/ax-serving-api/src/orchestration/mod.rs`
- `crates/ax-serving-cli/src/main.rs`
- `crates/ax-serving-api/tests/model_management.rs`
- `crates/ax-serving-api/tests/orchestration.rs`

### A.2 Shared Logic To Consolidate

At minimum, `v3.0` should consolidate:

- prompt-token estimation shared by serving and orchestrator request parsing
- audit helper patterns shared by serving and orchestrator admin surfaces
- startup-report shaping where the structure is intentionally parallel
- test backend doubles such as `NullBackend` and `FailingUnloadBackend`
- temp GGUF fixture builders
- env-var test serialization helpers

### A.3 Target Module Direction

Expected landing zones include:

- `crates/ax-serving-api/src/rest/inference.rs`
- `crates/ax-serving-api/src/rest/models.rs`
- `crates/ax-serving-api/src/rest/admin.rs`
- `crates/ax-serving-api/src/rest/license.rs`
- `crates/ax-serving-api/src/rest/reporting.rs`
- `crates/ax-serving-api/src/orchestration/proxy.rs`
- `crates/ax-serving-api/src/orchestration/admin.rs`
- `crates/ax-serving-api/src/orchestration/fleet.rs`
- `crates/ax-serving-api/src/orchestration/reporting.rs`
- `crates/ax-serving-api/src/orchestration/request_shape.rs`
- `crates/ax-serving-api/tests/common/`

The exact filenames may differ, but the ownership split must be clear.

### A.4 Delivery Phasing

Recommended order:

1. Extract obviously shared helper logic first.
2. Split REST and orchestrator route modules second.
3. Centralize test support after production seams stabilize.
4. Split CLI command logic after API/orchestrator boundaries are clearer.
5. Add maintainability guardrails before closing the `v3.0` release.
