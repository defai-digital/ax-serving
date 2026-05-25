# ADR Index

This directory contains all architecture decision records for AX Serving.

The project currently has two ADR eras:

- product-scope ADRs, written after the product boundary was clarified
- earlier implementation ADRs

Some earlier implementation ADRs may no longer match the current codebase
exactly. Treat them as historical decisions until they are reviewed, amended, or
superseded by newer ADRs.

## Product-Scope ADRs

- [ADR-001: Target Market And Product Scope](./ADR-001-target-market-and-product-scope.md)
- [ADR-002: Software Value Boundary Vs Hardware Deployment Context](./ADR-002-software-value-boundary-vs-hardware-deployment-context.md)
- [ADR-012: Multi-Worker Orchestration For Department-Scale Private AI Fleets](./ADR-012-local-multi-worker-orchestration.md)

## Implementation ADRs

- [ADR-001: Use mistralrs-core as the Inference Backend](./ADR-001-inference-backend.md)
- [ADR-002: Retain ax-engine's Layered Serving Architecture](./ADR-002-serving-architecture.md)
- [ADR-003: Add OpenAI-Compatible REST API](./ADR-003-openai-rest-api.md)
- [ADR-004: KV Cache Strategy](./ADR-004-kv-cache-strategy.md)
- [ADR-005: Metal Backend Strategy](./ADR-005-metal-strategy.md)
- [ADR-006: Workspace and Crate Structure](./ADR-006-crate-structure.md)
- [ADR-007: llama.h C API Shim Strategy](./ADR-007-c-api-shim.md)
- [ADR-008: Thermal-Aware Scheduling](./ADR-008-thermal-scheduling.md)
- [ADR-009: Configuration Strategy](./ADR-009-configuration.md)
- [ADR-010: Multi-Backend Inference Selection with Fallback Chain](./ADR-010-backend-fallback.md)
- [ADR-011: Service-Level Performance Strategy](./ADR-011-service-level-performance-strategy.md)
- [ADR-012: Local Multi-Worker Orchestration - Direct + Optional NATS Dispatch](./ADR-012-local-multi-worker-orchestration-direct-nats.md)
- [ADR-013: Re-enable mistralrs v0.7.0 as Primary Inference Backend](./ADR-013-mistralrs-primary-backend.md)
