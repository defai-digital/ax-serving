# ADR-012: Multi-Worker Orchestration For Department-Scale Private AI Fleets

- Status: Accepted
- Date: 2026-03-28

## Context

AX Serving already includes a multi-worker orchestration layer, but the project
needs a clearer statement of why this exists and what market problem it solves.

The orchestration layer should not be framed as generic distributed inference
for every environment. That would place AX Serving into direct competition with
GPU-first hyperscale serving systems.

Instead, the orchestration layer exists to solve the serving needs of
department-scale private AI fleets.

## Decision

AX Serving will treat multi-worker orchestration as a core product capability
for:

- multi-model private AI serving
- mixed-worker fleets
- department-scale operational control
- private and governed deployment environments

The orchestrator is a control plane, not a token-generation engine.

Primary responsibilities:

- worker registration and heartbeats
- health-aware dispatch
- queue and inflight control
- worker drain and recovery flows
- fleet inventory and administrative visibility
- support for heterogeneous worker classes over time

The orchestration layer is intended to support deployment patterns such as:

- one Mac-led control plane with multiple serving workers
- standard-operations workers and larger-memory workers in one fleet
- future heterogeneous fleet expansion without changing the serving API layer

## Consequences

The orchestrator should be documented and evolved as:

- a fleet-operation layer
- a mixed-worker serving control plane
- an operational substrate for private AI teams

The orchestrator should not be marketed as:

- a generic hyperscale distributed serving framework
- a replacement for CUDA-first cluster serving platforms in their primary use case

Runbooks, metrics, and admin APIs should continue to emphasize:

- fleet health
- worker lifecycle
- routing behavior
- failure handling
- team-operated private deployment workflows
