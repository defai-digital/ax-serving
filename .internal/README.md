# AX Serving Internal Design Index

This directory contains the canonical product and architecture contracts for AX Serving.
Public operator and integration documentation remains under [`../docs`](../docs/).

## Canonical documents

- [Product requirements](prd/PRD-AX-SERVING.md)
- [ADR-013: Runtime-neutral hybrid inference control plane](adr/ADR-013-RUNTIME-NEUTRAL-HYBRID-INFERENCE-CONTROL-PLANE.md)
- [Hybrid runtime control-plane technical specification](specs/TECH-SPEC-HYBRID-RUNTIME-CONTROL-PLANE.md)
- [Implementation and certification status](IMPLEMENTATION-STATUS.md)

## Document precedence

When documents conflict, use this order:

1. Accepted ADRs define architecture decisions and ownership boundaries.
2. The PRD defines product outcomes, requirements, and release gates.
3. Technical specifications define the current implementation plan.
4. Public documentation describes released behavior only.

Public documentation must not claim a capability until the corresponding PRD release gate has
passed and the behavior is available in a released build.
