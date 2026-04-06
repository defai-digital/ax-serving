# AX Serving Public Contract Inventory

> **Status**: Ready for execution
> **Date**: 2026-03-29
> **Owner**: AX Serving Core Team
> **Related**:
> - [PRD — AX Serving Open-Source / Enterprise Boundary v1.0](/Users/akiralam/code/ax-serving/docs/prd/PRD-AX-SERVING-OSS-ENTERPRISE-BOUNDARY-v1.0.md)
> - [PRD — AX Serving Enterprise Extraction And Release Execution v1.0](/Users/akiralam/code/ax-serving/docs/prd/PRD-AX-SERVING-ENTERPRISE-EXECUTION-v1.0.md)
> - [AX Fabric Runtime Contract](/Users/akiralam/code/ax-serving/docs/contracts/ax-fabric-runtime-contract.md)

# Purpose

This document defines the public contracts that enterprise repositories may
depend on when integrating with the open-source core.

It exists to prevent enterprise products from quietly depending on internal
Rust modules or undocumented response details.

# Contract Families

## 1. Northbound Serving APIs

These are the primary public contracts exposed to operators and upstream
products.

Stable contract family:

- `GET /health`
- `GET /v1/models`
- `POST /v1/models`
- `DELETE /v1/models/{id}`
- `POST /v1/models/{id}/reload`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/metrics`

Stability rule:

- fields and semantics documented in public docs or release notes are contract
- undocumented JSON fields may change without compatibility guarantees

Enterprise usage examples:

- enterprise control plane health polling
- management UI model inventory views
- enterprise routing clients and benchmark agents

## 2. Worker Lifecycle Contracts

These are the public contracts that separate workers use to participate in the
control plane.

Stable contract family:

- worker register flow
- worker heartbeat flow
- worker drain / undrain semantics
- worker eviction semantics where documented
- capability advertisement payloads
- worker pool, backend, health, and queue metadata that are already surfaced as
  protocol data

Stability rule:

- protocol fields already used by public orchestration or documented in tests
  may be depended on
- new enterprise-only worker capabilities must be added as protocol extensions,
  not as private Rust trait hooks

Enterprise usage examples:

- NVIDIA / Thor-class worker implementations
- heterogeneous fleet placement services
- enterprise worker health enrichments

## 3. Admin And Diagnostics Surfaces

These surfaces may be used by enterprise tooling only where their shape is
documented or release-noted.

Stable contract family:

- startup / diagnostics payloads where publicly documented
- admin status and fleet summaries where documented
- audit listing response shape where documented

Stability rule:

- documented keys are contract
- undocumented nested keys are not contract by default

Enterprise usage examples:

- private management UI
- governance status aggregation
- enterprise support tooling

## 4. Config And Environment Contracts

Stable contract family:

- documented YAML config keys
- documented `AXS_*` environment variables
- documented CLI flags

Current documented examples include:

- `AXS_LOG`
- `AXS_LOG_FORMAT`
- `AXS_MODEL_ALLOWED_DIRS`
- `AXS_MODEL_WARM_POOL_SIZE`
- `AXS_API_KEY`

Stability rule:

- documented config fields require migration notes when behavior changes
- private repos must not depend on unpublished config branches

Enterprise usage examples:

- deployment bundles
- installers and air-gapped setup tooling
- entitlement-aware operational wrappers

## 5. Metrics And Audit Payloads

Stable contract family:

- documented Prometheus metric names
- documented JSON metrics keys
- audit event payload shapes where exported or documented

Stability rule:

- metric names referenced by public docs or enterprise runbooks are contract
- temporary or debug-only metrics are not contract unless promoted here or in
  release notes

Enterprise usage examples:

- SIEM export
- compliance dashboards
- alerting and SLO policies

# Non-Contract Surfaces

Enterprise repositories must not treat these as stable contracts:

- internal Rust module paths
- private helper functions
- undocumented HTML or dashboard DOM structure
- test-only scaffolding
- local branch-only response fields
- unpublished Cargo workspace membership

# Change Policy

Contract changes must follow all of these rules:

1. changes to documented public contracts require release-note coverage
2. removals or semantic changes require a migration note
3. enterprise repositories must consume tagged or frozen public-core versions
4. contract additions should prefer additive fields and explicit defaults

# Required Review Questions

Any change touching a public contract should answer:

1. Which enterprise repository classes could depend on this surface
2. Is the change additive, behavior-changing, or breaking
3. Which public docs or runbooks need updating
4. Does compatibility metadata need a new contract version marker

# Minimum Contract Set For Enterprise Extraction

The following are the minimum public contracts required to keep enterprise code
private while keeping the public core coherent:

- northbound REST serving APIs
- worker lifecycle and capability protocol
- documented admin / diagnostics payloads used by enterprise tooling
- documented config and `AXS_*` environment contracts
- documented metrics and audit export payloads

# Decision

Enterprise products may depend only on this documented public contract layer
and separately release-noted additions.

They must not depend on the public repository's source layout as their
integration boundary.
