# AX Serving internal design index

This directory contains the canonical target architecture and the evidence ledger. Public operator
and integration documentation remains under [`../docs`](../docs/).

## Canonical documents

- [Product requirements](prd/PRD-AX-SERVING.md)
- [ADR-016: Federated Dynamo and AX Engine control plane](adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md)
- [Federated inference control-plane technical specification](specs/TECH-SPEC-FEDERATED-INFERENCE-CONTROL-PLANE.md)
- [ADR-017: Mac AX Engine distributed execution domain](adr/ADR-017-MAC-AX-ENGINE-DISTRIBUTED-EXECUTION-DOMAIN.md)
- [Mac distributed inference product requirements](prd/PRD-MAC-DISTRIBUTED-INFERENCE.md)
- [Mac distributed execution-domain technical specification](specs/TECH-SPEC-MAC-DISTRIBUTED-EXECUTION-DOMAIN.md)
- [Implementation and certification status](IMPLEMENTATION-STATUS.md)

ADR-017 and its linked PRD/specification add a Mac model-parallel domain without changing
ADR-016's federation boundary. Its phase 1 coordinator/control-plane source does not imply that AX
Engine distributed execution exists. See the public
[Mac cluster coordinator guide](../docs/integrations/mac/CLUSTER.md). Earlier hybrid, CPU-only
deployment, and agent-session ADR/PRD/spec packages were consolidated on 2026-07-15. Their valid
requirements are part of the canonical set.

## Document precedence

When documents conflict:

1. ADR-016 defines federation ownership and architecture boundaries.
2. ADR-017 additively defines the Mac distributed execution-domain boundary.
3. The applicable PRD defines product outcomes, requirements, value gates, and release gates.
4. The applicable technical specification defines target implementation and migration.
5. The status ledger says what is actually implemented or certified.
6. Public documentation describes released behavior only.

An implemented mock-tested type is not a live runtime certification. Generic Dynamo platform
support is not Thor certification. Public claims require the corresponding retained release
evidence.
