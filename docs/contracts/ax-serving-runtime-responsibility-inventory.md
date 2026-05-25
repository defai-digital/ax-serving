# AX Serving Runtime Responsibility Inventory

> **Status**: Active
> **Date**: 2026-05-25
> **Owner**: AX Serving Team
> **Related PRD**: [PRD-AX-SERVING.md](../../.internal/prd/PRD-AX-SERVING.md)

---

## Purpose

This inventory classifies runtime and backend code paths while AX Serving moves
toward the PRD target architecture:

- AX Serving is the serving gateway and control plane.
- Runtime nodes own inference execution.
- Mac inference should be provided by ax-engine nodes.
- PC CUDA and NVIDIA Thor inference should be provided by vLLM nodes.

---

## Classification

| Area | Current role | Target role | Status |
|---|---|---|---|
| OpenAI-compatible REST/gRPC API | Serving API | Keep in AX Serving | Gateway |
| Worker registry, heartbeat, drain | Fleet control plane | Keep in AX Serving | Gateway |
| Routing, placement, admission, retry | Fleet policy | Keep in AX Serving | Gateway |
| Metrics, audit, fleet diagnostics | Operator surface | Keep in AX Serving | Gateway |
| `ax-thor-agent` proxy | Runtime adapter | Keep as thin vLLM/Thor adapter | Adapter |
| `ax-serving serve` local worker | Embedded Mac compatibility worker | Keep only as migration bridge | Compatibility |
| Native ax-engine backend inside serving crate | Direct runtime execution | Move to ax-engine node adapter | Compatibility |
| llama.cpp subprocess backend | Direct runtime execution | Deprecate after vLLM/ax-engine node paths cover needs | Compatibility |
| MLX subprocess backend | Direct runtime execution | Deprecate or adapterize after ax-engine node path | Compatibility |
| optional libllama direct backend | Direct runtime execution | Deprecate unless required for shim compatibility | Compatibility |
| Shim library | llama-style integration compatibility | Keep only if it serves a supported integration path | Compatibility |

---

## Compatibility Rules

- Compatibility paths must not become new product architecture.
- Compatibility paths may remain while replacement node adapters are incomplete.
- New routing and observability features should use runtime/node metadata rather
  than embedded backend internals.
- Runtime-specific tuning belongs in ax-engine or vLLM documentation and
  tooling, not in AX Serving control-plane requirements.
- Breaking removal requires a migration note and an equivalent node-adapter path.

---

## Operator Signals

`ax-serving doctor` includes a runtime-boundary check:

- `PASS` when worker registration is explicitly configured for `ax_engine` or
  `vllm` runtime-node mode.
- `WARN` when AX Serving appears to be used in standalone embedded inference
  mode.
- `WARN` when an unknown runtime is configured.

This warning is intentional. It does not block local compatibility use, but it
keeps the PRD target visible to operators.

---

## Removal Gate

Before removing or disabling any compatibility backend:

1. Document the replacement node adapter.
2. Confirm OpenAI-compatible chat/completions behavior through the gateway.
3. Confirm worker registration, heartbeat, drain, and fleet status.
4. Confirm routing by model and runtime class.
5. Provide migration notes for existing configuration and CLI usage.
