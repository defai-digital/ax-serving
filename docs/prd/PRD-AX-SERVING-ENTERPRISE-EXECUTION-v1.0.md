# PRD — AX Serving Enterprise Extraction And Release Execution v1.0

> **Status**: Ready for execution
> **Date**: 2026-03-29
> **Owner**: AX Serving Team
> **Type**: Execution PRD
> **Related**:
> - [README](../README.md)
> - [LICENSING](../LICENSING.md)
> - [PRD — AX Serving v3.0](./PRD-AX-SERVING-v3.0.md)
> - [PRD — AX Serving Open-Source / Enterprise Boundary v1.0](./PRD-AX-SERVING-OSS-ENTERPRISE-BOUNDARY-v1.0.md)
> - [AX Serving Public Contract Inventory](../contracts/ax-serving-public-contract-inventory.md)
> - [AX Serving Contract Change Note Template](../contracts/ax-serving-contract-change-template.md)
> - [Enterprise Compatibility Metadata Example](../contracts/enterprise-compatibility-metadata.example.yaml)
> - [Enterprise Private Repository Bootstrap](../runbooks/enterprise-private-repo-bootstrap.md)
> - [Enterprise Release Governance](../runbooks/enterprise-release-governance.md)

---

## 1. Purpose

This PRD turns the product-boundary decision into an execution plan.

It specifies:

- the target private repository layout
- the ownership of public versus private deliverables
- the versioning and compatibility policy
- the release matrix across public, commercial-core, and enterprise layers
- the migration phases required to reach the target architecture

This document assumes the product decision already made in the boundary PRD:

- open-source core = single-Mac and Mac-grid serving
- enterprise layer = heterogeneous NVIDIA / Thor-class fleets, governance,
  deployment, and supported private integrations

---

## 2. Execution Goals

### G1. Preserve A Strong Public Core

The public repository must remain a coherent serving control plane for:

- single-Mac deployments
- Mac-grid deployments

### G2. Extract Enterprise Value Cleanly

Enterprise-only value must live in separate private repositories and release
channels without polluting the public workspace.

### G3. Minimize Compatibility Friction

Enterprise layers should depend on stable external contracts from the public
core and should not require customer-specific patch sets against the public
repository.

### G4. Make Delivery Repeatable

Commercial and enterprise delivery must become a productized release process,
not an ad hoc consulting workflow.

---

## 3. Target Repository Layout

### 3.1 Public Repository

Repository:

- `ax-serving`

Responsibilities:

- serving runtime
- Mac-native control plane
- Mac-grid orchestration
- worker lifecycle protocols
- public SDKs
- public benchmarks
- public docs and runbooks

### 3.2 Private Repositories

Repository 1:

- `ax-serving-enterprise-control-plane`

Responsibilities:

- enterprise auth and identity integration
- RBAC / org / tenant / project policy UI and APIs
- governance workflows
- audit forwarding and compliance packaging
- private management UI

Repository 2:

- `ax-serving-enterprise-workers`

Responsibilities:

- NVIDIA / Thor-class worker binaries
- supported accelerator runtime adapters
- accelerator-specific fleet packaging
- enterprise worker health/reporting enrichments where needed

Repository 3:

- `ax-serving-enterprise-deploy`

Responsibilities:

- private container images
- air-gapped bundles
- installer / operator / Terraform / Helm packaging
- upgrade, migration, and entitlement activation workflows

Repository 4:

- `ax-serving-enterprise-connectors`

Responsibilities:

- SIEM / logging / compliance connectors
- private secret manager / KMS / IAM integrations
- customer-specific but reusable enterprise adapters

---

## 4. Ownership Model

| Area | Primary Owner | Notes |
|---|---|---|
| Public serving APIs | AX Serving core team | Contract stability required |
| Mac runtime and grid orchestration | AX Serving core team | Open-source complete path |
| Public docs and SDKs | AX Serving core team | Must remain usable without enterprise |
| Enterprise auth / governance | Enterprise control-plane team | Private repo |
| NVIDIA / Thor workers | Enterprise workers team | Private repo |
| Deployment bundles and installers | Enterprise deploy team | Private repo |
| Private connectors | Enterprise integrations team | Private repo |

---

## 5. Contract Strategy

### 5.1 Public Contracts To Freeze

The following contracts are the intended enterprise integration surface:

- OpenAI-compatible REST endpoints
- gRPC service contracts
- worker register / heartbeat / drain / drain-complete flows
- admin and diagnostics response shapes where documented
- metrics names and JSON metrics payloads where documented
- config schema and relevant `AXS_*` environment contracts

### 5.2 Contract Change Policy

Contract changes must follow these rules:

1. public contract changes require release-note coverage
2. breaking contract changes require a versioned migration note
3. enterprise repositories must consume released public contracts, not local
   unpublished branch assumptions

### 5.3 Forbidden Shortcut

The enterprise repositories must not depend on:

- unpublished internal Rust modules as their primary contract
- direct source inclusion from the public repository
- customer-specific forks as the default deployment model

---

## 6. Versioning And Compatibility

### 6.1 Public Core Version

The public core remains the canonical semantic version anchor.

Example:

- `ax-serving` `v2.1.x`

### 6.2 Enterprise Compatibility Window

Each enterprise release should declare:

- minimum supported public-core version
- maximum validated public-core minor version
- required migration notes if contract deltas exist

Recommended policy:

- enterprise repos support one current public-core minor line and one previous
  minor line

### 6.3 Compatibility Table

| Public Core | Enterprise Control Plane | Enterprise Workers | Status |
|---|---|---|---|
| `2.1.x` | `2.1.x-e1` | `2.1.x-e1` | Primary |
| `2.0.x` | `2.0.x-eN` | `2.0.x-eN` | Maintenance only |

The exact numbering may evolve, but the policy should remain explicit.

---

## 7. Release Matrix

### 7.1 Public Release

Artifacts:

- source repository tags
- public release notes
- public documentation
- public SDK builds where applicable

Audience:

- open-source users
- evaluators
- commercial-license prospects

### 7.2 Commercial Core Release

Artifacts:

- same source baseline under separate commercial agreement
- customer-ready release notes
- supported upgrade guidance

Audience:

- customers who license the core under non-AGPL terms without necessarily
  taking enterprise add-ons

### 7.3 Enterprise Release

Artifacts:

- private container images
- private worker binaries
- enterprise control-plane services
- air-gapped bundles
- deployment manifests
- compatibility matrix

Audience:

- customers operating heterogeneous fleets or requiring governance and private
  enterprise delivery

### 7.4 Release Table

| Artifact | Public | Commercial Core | Enterprise |
|---|---|---|---|
| Source tag | Yes | Same source under contract | Referenced only |
| Mac binaries | Optional public | Yes | Reused |
| Mac-grid deployment docs | Yes | Yes | Reused |
| NVIDIA / Thor worker binaries | No | No | Yes |
| Enterprise control plane | No | No | Yes |
| Air-gapped package | No | Optional by agreement | Yes |
| Private connector packages | No | No | Yes |

---

## 8. Migration Phases

### Phase 0. Documentation

Already completed in this planning cycle:

- product boundary clarified
- licensing docs aligned
- contribution guardrails updated

### Phase 1. Public Contract Inventory

Deliverables:

- inventory of stable public contracts
- list of undocumented internal contracts currently used by worker paths
- release-note template for contract changes

Attached planning artifacts in this repository:

- [ax-serving-public-contract-inventory.md](/Users/akiralam/code/ax-serving/docs/contracts/ax-serving-public-contract-inventory.md)
- [ax-serving-contract-change-template.md](/Users/akiralam/code/ax-serving/docs/contracts/ax-serving-contract-change-template.md)

### Phase 2. Enterprise Repo Bootstrap

Deliverables:

- create private repositories
- create baseline CI/CD
- create package and image naming conventions
- define compatibility metadata format

Attached planning artifacts in this repository:

- [enterprise-private-repo-bootstrap.md](/Users/akiralam/code/ax-serving/docs/runbooks/enterprise-private-repo-bootstrap.md)
- [enterprise-compatibility-metadata.example.yaml](/Users/akiralam/code/ax-serving/docs/contracts/enterprise-compatibility-metadata.example.yaml)

### Phase 3. Worker Extraction

Deliverables:

- move accelerator-specific worker implementations to the private workers repo
- keep public worker protocol definitions in `ax-serving`
- document any additional telemetry fields or capability extensions

### Phase 4. Deploy Extraction

Deliverables:

- move private bundles, operators, and deployment manifests to the deploy repo
- define air-gapped packaging process
- define enterprise install and upgrade path

Attached planning artifact in this repository:

- [enterprise-release-governance.md](/Users/akiralam/code/ax-serving/docs/runbooks/enterprise-release-governance.md)

### Phase 5. Control Plane Extraction

Deliverables:

- move enterprise auth, governance, and private management services to the
  control-plane repo
- keep public admin/control APIs in the public core where they are part of the
  Mac-native open-source path

---

## 9. CI/CD Requirements

### 9.1 Public CI

Must continue to validate:

- `cargo check --workspace`
- `cargo test --workspace`
- `cargo clippy --workspace --tests -- -D warnings`
- SDK build sanity

### 9.2 Enterprise CI

Must additionally validate:

- compatibility against supported public-core versions
- enterprise image build
- deployment bundle assembly
- integration smoke tests against released public-core contracts

### 9.3 Release Gate

An enterprise release is not releasable unless:

1. its target public-core version is tagged or frozen
2. compatibility metadata is published internally
3. upgrade notes exist for the supported previous enterprise release

---

## 10. Naming And Packaging Rules

### 10.1 Public Names

Keep public names simple:

- `ax-serving`
- `ax-serving-api`
- `ax-thor-agent` only while it remains part of the public transition state

### 10.2 Enterprise Names

Use explicit enterprise naming:

- `ax-serving-enterprise-control-plane`
- `ax-serving-enterprise-workers`
- `ax-serving-enterprise-deploy`
- `ax-serving-enterprise-connectors`

Avoid ambiguous names that imply public availability.

### 10.3 Image And Package Prefixes

Recommended:

- public images use public naming
- private images use `enterprise` or a private registry namespace

Example:

- public: `ghcr.io/defai-digital/ax-serving`
- private: internal registry namespace per commercial operations policy

---

## 11. Risks

### R1. Transition Lag

Risk:

- the public repo continues to carry too much transition-state enterprise logic

Mitigation:

- prioritize worker extraction after contract inventory
- avoid adding new accelerator-specific logic to public crates

### R2. Customer Fork Pressure

Risk:

- customer requests drive ad hoc forks and one-off patches

Mitigation:

- keep enterprise products in private repos
- standardize compatibility and release cadence
- forbid customer-specific public-core patching as the default model

### R3. Contract Drift

Risk:

- enterprise repos silently depend on internal behavior not covered by public
  contracts

Mitigation:

- inventory and freeze contract surfaces
- require release-note coverage for public contract changes

---

## 12. Acceptance Criteria

This execution PRD is considered complete when:

1. the private repository layout is explicitly defined
2. ownership boundaries between public and private deliverables are explicit
3. release and compatibility policy is documented
4. migration phases are clear enough to execute without redefining the target
   architecture
5. the plan remains consistent with the open-source / enterprise boundary PRD

---

## 13. Final Execution Decision

AX Serving will proceed as:

- one strong public serving core repository
- one commercially licensable core product
- a set of separate private enterprise repositories for accelerator workers,
  governance, deployment, and private integrations

The engineering rule is simple:

- **public core by source**
- **enterprise by service boundary and private delivery**
