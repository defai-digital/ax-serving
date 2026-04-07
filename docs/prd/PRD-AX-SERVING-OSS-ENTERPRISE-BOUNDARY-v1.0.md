# PRD — AX Serving Open-Source / Enterprise Boundary v1.0

> **Status**: Ready for execution
> **Date**: 2026-03-29
> **Owner**: AX Serving Team
> **Type**: Product + Licensing Boundary PRD
> **Related**:
> - [README](../README.md)
> - [LICENSING](../LICENSING.md)
> - [Commercial License Summary](../LICENSE-COMMERCIAL.md)
> - [PRD — AX Serving v3.0](./PRD-AX-SERVING-v3.0.md)
> - [PRD — AX Serving Enterprise Execution v1.0](./PRD-AX-SERVING-ENTERPRISE-EXECUTION-v1.0.md)
> - [AX Serving Public Contract Inventory](../contracts/ax-serving-public-contract-inventory.md)
> - [Enterprise Private Repository Bootstrap](../runbooks/enterprise-private-repo-bootstrap.md)
> - [Enterprise Release Governance](../runbooks/enterprise-release-governance.md)
> - [ADR-001](../adr/ADR-001-target-market-and-product-scope.md)
> - [ADR-012](../adr/ADR-012-local-multi-worker-orchestration.md)

---

## 1. Executive Summary

This PRD defines the recommended product and repository boundary between the
open-source and enterprise offerings of AX Serving.

The intended product split is:

- **Open-source core**: single-Mac serving and Mac-grid serving
- **Enterprise layer**: heterogeneous fleets that include NVIDIA / Thor-class
  workers, enterprise deployment tooling, governance, and supported private
  integrations

The goal is to keep the public repository useful, credible, and complete as a
serving control plane while preserving a clean commercial expansion path.

This PRD explicitly rejects a strategy where the public workspace contains
private Rust crates or hidden enterprise-only code paths.

It also defines a transition plan from the current repository state to the
target product boundary so that existing public code can evolve without a
disruptive rewrite.

---

## 2. Problem Statement

AX Serving is already positioned as a dual-licensed serving and orchestration
control plane. Without an explicit boundary, the product can drift into an
unclear and fragile state:

- the open-source edition may become an intentionally crippled teaser
- enterprise code may leak into the public workspace through feature flags or
  private crates
- hardware-specific positioning may become confused with repository structure
- deployment and licensing terms may become harder to explain to operators and
  buyers

For this product, the main question is not whether commercial licensing exists.
It already does. The question is where the technical and product boundary
should sit so that:

- the public repository remains coherent
- commercial packaging is defensible
- engineering ownership remains clear
- future enterprise modules do not contaminate the public core

---

## 3. Product Decision

### 3.1 Open-Source Core

The public repository defines the open-source core of AX Serving.

The default open-source product scope is:

- single-Mac serving
- Mac-led local serving
- Mac worker grids
- model lifecycle and serving APIs
- scheduler, metrics, diagnostics, and dashboard surfaces
- worker registration, heartbeat, drain, and routing contracts
- public SDKs, benchmarks, runbooks, and operator documentation

### 3.2 Enterprise Layer

The enterprise offering builds on the open-source core and covers one or both
of the following:

- non-AGPL commercial licensing rights for the AX Serving core itself
- separate proprietary modules, deployment bundles, and supported integrations

The intended enterprise product expansion path is:

- NVIDIA / Thor-class worker support
- heterogeneous Mac + accelerator fleets
- enterprise auth, governance, and deployment packaging
- supported private integrations and fleet operations tooling

### 3.3 Product Message

The recommended market message is:

- **AX Serving Open Source**: Mac-native serving and Mac-grid orchestration
- **AX Serving Enterprise**: heterogeneous fleet operations, enterprise
  deployment, governance, and supported accelerator-worker integrations

The message should not be:

- "the same core but some hidden backends are closed source"

That framing creates avoidable licensing confusion and weakens product trust.

### 3.4 Current-State Exception Handling

The current public repository already contains worker and orchestration
concepts that refer to Thor-managed environments, including `ax-thor-agent`.
This PRD does not require an immediate removal of those components from the
public repository.

Instead, the transition rule is:

- public code may retain generic worker protocol and management foundations
- enterprise-specific accelerator-worker implementations should move toward
  separate private repositories over time
- future NVIDIA / Thor-class support should be added in private product layers,
  not as new hidden branches inside the public workspace

---

## 4. Repository Boundary

### 4.1 Public Repository

The following remain in the public repository:

- `crates/ax-serving-engine`
- `crates/ax-serving-api`
- `crates/ax-serving-cli`
- `crates/ax-thor-agent`
- `crates/ax-serving-shim`
- public SDKs
- benchmarks, scripts, and runbooks
- documented worker and admin protocols

Rationale:

- these components define the serving control plane itself
- they are required for operator trust and ecosystem adoption
- moving them private would turn the public project into an incomplete shell

### 4.2 Private Repositories

Enterprise product layers should live in separate private repositories.

Recommended shape:

1. `ax-serving-enterprise-control-plane`
   - enterprise auth
   - org / tenant / policy management
   - governance and audit export
   - private management UI

2. `ax-serving-enterprise-workers`
   - NVIDIA / Thor-class worker implementations
   - supported accelerator runtime adapters
   - enterprise-only fleet worker packaging

3. `ax-serving-enterprise-deploy`
   - air-gapped bundles
   - private images and release packaging
   - installer / operator / Terraform / Helm assets
   - enterprise entitlement activation flows

### 4.3 Repository Ownership Matrix

| Area | Public Core | Enterprise Layer |
|---|---|---|
| OpenAI-compatible serving API | Yes | Reuse |
| Mac single-node runtime | Yes | Reuse / commercially license |
| Mac-grid orchestration | Yes | Reuse / commercially license |
| Worker registration and heartbeat contract | Yes | Reuse |
| Public SDKs | Yes | Reuse |
| Scheduler, metrics, diagnostics | Yes | Reuse |
| NVIDIA / Thor runtime adapters | Contract only | Yes |
| Enterprise auth / RBAC / tenancy | No | Yes |
| Air-gapped bundles / private packaging | No | Yes |
| Private connectors / compliance export | No | Yes |

---

## 5. Technical Boundary Rules

### 5.1 Allowed Boundary

Enterprise components should integrate with the open-source core through stable,
documented external contracts such as:

- REST APIs
- gRPC APIs
- worker registration / heartbeat / drain protocols
- config contracts
- metrics surfaces
- audit event exports
- webhooks or queue-based integration boundaries

The stable contract surfaces to preserve are:

- northbound serving APIs
- admin and diagnostics APIs
- worker lifecycle APIs
- config file and environment variable contracts where documented
- metrics and audit payload formats that enterprise tooling depends on

### 5.2 Disallowed Or Discouraged Boundary

The following are not the recommended design:

- private crates inside the public Rust workspace
- hidden enterprise code behind Cargo features in the public repository
- proprietary in-process plugins linked into public binaries
- enterprise modules that depend on undocumented internal Rust types as their
  primary integration surface

### 5.3 Engineering Principle

The product boundary should be a **service boundary**, not a **module hiding
boundary**.

That keeps the public codebase buildable, explainable, and supportable while
allowing enterprise products to move at a different pace.

### 5.4 Engineering Guardrails

The following guardrails apply to future work:

1. no new enterprise-only crate may be added under `crates/` in the public
   repository
2. no public binary may require a proprietary dependency to build or run for
   the Mac-native open-source path
3. any new worker capability intended for enterprise-only fleets must first
   define its external protocol boundary before implementation begins
4. enterprise extraction must not break the documented Mac single-node or
   Mac-grid deployment paths

---

## 6. Open-Source Scope

The open-source scope must remain operationally complete for its target use
case.

Required characteristics:

- fully usable on a single Mac
- fully usable across multiple Mac workers
- one coherent API and control plane story
- no private dependency required for normal Mac-native deployment
- public docs that describe the open-source deployment path honestly

Open source is not meant to be:

- a teaser that depends on hidden enterprise crates to become usable
- the full enterprise management plane
- the supported delivery vehicle for heterogeneous NVIDIA / Thor fleets

---

## 7. Enterprise Scope

The enterprise layer should concentrate on high-value additions that are
reasonable to sell as commercial product and delivery capability.

In scope for enterprise:

- supported NVIDIA / Thor-class fleet operation
- heterogeneous fleet policy and placement controls
- enterprise auth and governance
- private deployment bundles and installation paths
- supported private integrations
- operational tooling for managed or regulated environments

Out of scope for enterprise-only gating:

- basic REST serving
- basic model lifecycle
- basic Mac-grid orchestration
- public worker protocol foundations
- public SDK basics

---

## 8. Packaging And Distribution

### 8.1 Open Source

The public distribution path remains:

- public GitHub repository
- public source release
- public documentation
- public SDK packages where applicable

### 8.2 Enterprise

Enterprise distribution should use private channels such as:

- private repositories
- private package registries
- private container registries
- customer-specific deployment bundles

Enterprise distribution should not rely on:

- `.gitignore` to hide commercial code inside the public repo
- partially published monorepo subtrees
- manual patching of the public repository per customer

### 8.3 Release Matrix

| Deliverable | Public | Commercial Core License | Enterprise |
|---|---|---|---|
| Source repository | Public | Same code under contract | Separate private repos where needed |
| macOS binaries | Public or customer-built | Licensed use allowed | Reused where applicable |
| Mac-grid deployment | Public | Supported under contract | Reused where applicable |
| NVIDIA / Thor worker packages | No | Optional by agreement | Private |
| Air-gapped bundle | No | Optional by agreement | Private |
| Enterprise control-plane modules | No | No | Private |

---

## 9. Licensing And Contribution Model

The current repository policy of not accepting unsolicited public code
contributions is compatible with this product split and should remain in place
unless the project first introduces contributor legal controls such as CLA or
other written terms.

Reason:

- commercial licensing depends on clean copyright control
- ambiguous contribution rights make future commercial packaging harder

---

## 10. Rollout Plan

### Phase 1. Documentation Alignment

- update `README.md`
- update `LICENSING.md`
- update `LICENSE-COMMERCIAL.md`
- add this PRD

### Phase 2. Contract Hardening

- identify and document the stable integration surfaces
- keep worker/admin/config contracts explicit
- avoid introducing private crates into the public workspace

### Phase 3. Enterprise Extraction

- place accelerator-worker implementations in private repositories
- place enterprise deployment assets in private repositories
- place enterprise governance and private UI in private repositories

### Phase 4. Release And Sales Alignment

- align release packaging with the public / commercial / enterprise matrix
- align README, licensing, and commercial collateral with the same product
  boundary
- ensure customer delivery does not depend on ad hoc patches to the public repo

---

## 11. Risks And Mitigations

### R1. Public / Private Boundary Drift

Risk:

- enterprise work leaks back into the public workspace through expedient
  feature flags or private local patches

Mitigation:

- keep the guardrails in this PRD and `CONTRIBUTING.md`
- require service-boundary design before enterprise implementation

### R2. Open-Source Edition Becomes Too Weak

Risk:

- the public edition loses credibility if too many core serving features move
  behind enterprise packaging

Mitigation:

- keep Mac single-node and Mac-grid paths complete and documented
- do not gate basic orchestration or serving APIs behind enterprise-only layers

### R3. Transition Ambiguity Around Existing Thor-Named Code

Risk:

- current public Thor-related code causes confusion about whether all
  Thor-class operation is open source

Mitigation:

- treat current public Thor code as transition-state protocol/foundation
- place future accelerator-worker expansion in private repositories
- keep documentation explicit that enterprise value is the supported
  heterogeneous fleet product, not merely a string match on a backend name

---

## 12. Acceptance Criteria

This PRD is considered implemented when:

1. top-level docs consistently describe the open-source focus as Mac-native
   serving and Mac-grid serving
2. top-level docs consistently describe enterprise as the path for
   heterogeneous NVIDIA / Thor-class fleets and enterprise delivery layers
3. no new private crates are introduced into the public Rust workspace as part
   of this positioning work
4. the public repository remains usable without proprietary dependencies for
   its intended open-source deployment path

---

## 13. Final Decision

AX Serving should be run as:

- a **complete open-source serving core** for Mac-native serving and Mac-grid
  operation
- a **commercially licensable core** for customers that need non-AGPL rights
- a foundation for **separate enterprise products** that add heterogeneous
  accelerator fleets, governance, and deployment tooling through private
  repositories and service boundaries
