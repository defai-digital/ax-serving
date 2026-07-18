# AX Serving internal design index

This directory contains the canonical target architecture and the evidence ledger. Public operator
and integration documentation remains under [`../docs`](../docs/).

## Canonical documents

- [Product requirements](prd/PRD-AX-SERVING.md)
- [ADR-016: Federated Dynamo and AX Engine control plane](adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md)
- [Federated inference control-plane technical specification](specs/TECH-SPEC-FEDERATED-INFERENCE-CONTROL-PLANE.md)
- [Implementation and certification status](IMPLEMENTATION-STATUS.md)

These four files are the only current internal product/design authorities. Earlier hybrid,
CPU-only deployment, and agent-session ADR/PRD/spec packages were consolidated on 2026-07-15.
Their valid requirements are now part of the canonical set.

## Document precedence

When documents conflict:

1. ADR-016 defines ownership and architecture boundaries.
2. The PRD defines product outcomes, requirements, value gates, and release gates.
3. The technical specification defines the target implementation and migration.
4. The status ledger says what is actually implemented or certified.
5. Public documentation describes released behavior only.

An implemented mock-tested type is not a live runtime certification. Generic Dynamo platform
support is not Thor certification. Public claims require the corresponding retained release
evidence.
