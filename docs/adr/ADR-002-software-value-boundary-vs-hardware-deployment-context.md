# ADR-002: Software Value Boundary Vs Hardware Deployment Context

- Status: Accepted
- Date: 2026-03-28

## Context

AX Serving is deployed across hardware environments where power, thermal,
memory, and concurrency characteristics matter. Those deployment realities are
important, but they do not automatically become software differentiators.

There is a risk of conflating:

- hardware selection benefits
- deployment economics
- software control-plane capabilities

That creates weak positioning and unclear product claims.

## Decision

AX Serving will distinguish clearly between:

### Software Value

Software claims may include:

- multi-model serving
- multi-worker orchestration
- fleet routing and health-aware dispatch
- model lifecycle management
- queueing, admission, and operational control
- metrics, audit, diagnostics, and policy surfaces

### Deployment Context

Deployment-context statements may include:

- Thor-class workers are a strong fit for standard high-parallel `<=70B` workloads
- Mac Studio-class workers are a strong fit for larger-memory model tiers, including `>70B`
- power, thermal, and space constraints shape hardware choices
- mixed-worker fleets may be preferred in real business environments

These deployment-context statements are valid system-design context, but they
must not be presented as if AX Serving software itself creates those hardware
economics.

## Consequences

Future documentation should:

- keep software claims and hardware claims separate
- frame AX Serving as the system that makes heterogeneous hardware operable
- avoid claiming token-per-watt or thermal efficiency as a standalone software superpower

Future PRDs should:

- treat hardware fit as deployment design input
- treat mixed-worker control, multi-model serving, and fleet operation as the
  primary product value
