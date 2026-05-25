# AX Serving PRD Implementation Plan

> **Status**: Active
> **Date**: 2026-05-25
> **Owner**: AX Serving Team
> **Source PRD**: [PRD-AX-SERVING.md](./PRD-AX-SERVING.md)

---

## 1. Product Position

AX Serving should be implemented as a runtime-neutral private AI serving
control plane.

It should not compete with inference engines such as ax-engine, vLLM, SGLang,
llama.cpp, MLX, or Ollama. Those projects own model execution, batching,
token generation, KV cache internals, runtime tuning, and hardware-specific
kernels.

AX Serving should own:

- OpenAI-compatible northbound API
- authentication and policy
- worker and model inventory
- routing, placement, admission, retry, and drain behavior
- fleet-level observability and diagnostics
- runtime adapter contracts for Mac ax-engine nodes and CUDA/Thor vLLM nodes

The product position is: private mixed-runtime fleet serving for organizations
that need one operational surface above multiple inference runtimes.

---

## 2. Implementation Principles

- Keep inference runtime internals out of AX Serving.
- Treat ax-engine and vLLM as node runtimes reached through explicit contracts.
- Preserve existing public API compatibility while introducing runtime-neutral
  node metadata.
- Keep adapters thin. They translate health, inventory, metrics, and proxy
  calls; they do not implement inference kernels or scheduler internals.
- Deprecate embedded runtime paths only after equivalent node paths exist.
- Keep each slice independently reviewable, testable, and reversible.
- Update the node contract, runbooks, and traceability table whenever an
  implementation slice changes product behavior.

---

## 3. Consolidated Workstreams

This plan replaces the earlier phase list. Completed work is recorded once
below; active work should be added only when it changes product behavior.

| Workstream | State | Evidence |
|---|---|---|
| Runtime-neutral node metadata | Complete | Registry, orchestration, and admin status tests |
| Public node contract | Complete | `docs/contracts/ax-serving-node-contract.md` |
| Generic runtime-node adapter | Complete | `ax-runtime-agent`, `AXS_NODE_*` config, runtime-agent e2e test |
| Target node runbooks | Complete | `docs/runbooks/multi-worker.md` |
| Runtime responsibility classification | Complete | `docs/contracts/ax-serving-runtime-responsibility-inventory.md` |
| ax-engine-specific telemetry translation | Active | Generic OpenAI-compatible adapter exists; richer health and metrics remain |
| Embedded runtime reduction | Active | Compatibility paths classified; removal waits for validated replacements |
| Runtime-class diagnostics depth | Active | Status/runtime buckets exist; deeper dashboard and diagnostics remain |

---

## 4. Validation

Release validation remains:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --tests -- -D warnings`
- `cargo test --workspace`

Focused validation for runtime-node changes:

- `cargo check -p ax-thor-agent --bins`
- `cargo test -p ax-thor-agent`
- orchestration tests covering legacy worker registration and runtime metadata
- markdown link validation for PRD, ADR, contract, README, and runbook links

Product validation for remaining work:

- mixed Mac ax-engine and vLLM worker registration
- routing by model, runtime, hardware class, pool, and capability
- drain and failover across runtime classes
- dashboard and CLI status output for runtime class, node class, hardware class,
  and model inventory

---

## 5. Implementation Progress

Completed slices:

- worker registry accepts vLLM and runtime-neutral metadata
- worker snapshots expose runtime, runtime version, hardware class, runtime
  endpoint, and supported operations
- routing can use runtime class hints, including `ax_engine` and `vllm`
- Thor CLI and agent default to vLLM while retaining SGLang compatibility
- the runtime-node adapter can be launched as `ax-runtime-agent` with generic
  `AXS_NODE_*` configuration for Mac ax-engine, PC CUDA vLLM, and Thor vLLM
  nodes
- local Mac worker compatibility registration defaults to `runtime =
  "ax_engine"` and `hardware_class = "mac"`
- public node contract is documented in
  `docs/contracts/ax-serving-node-contract.md`

Current evidence:

- [AX Serving Node Contract](../../docs/contracts/ax-serving-node-contract.md)
- [AX Serving Public Contract Inventory](../../docs/contracts/ax-serving-public-contract-inventory.md)
- [Multi-Worker Runbook](../../docs/runbooks/multi-worker.md)
- registry and orchestration tests covering runtime metadata and admin fleet
  runtime grouping
- embedded runtime responsibilities are classified in
  `docs/contracts/ax-serving-runtime-responsibility-inventory.md`
- `ax-serving doctor` surfaces runtime-boundary warnings for standalone
  embedded compatibility mode
- operator runbook examples now cover Mac ax-engine, PC CUDA vLLM, and Thor vLLM
  runtime-node registration through the common adapter
- `/v1/admin/diagnostics` exposes runtime-class diagnostics with model
  inventory, hardware classes, supported operations, runtime endpoints, and
  operator issue codes
- `/dashboard` renders runtime summaries from worker detail for quick operator
  visibility into runtime class, hardware, operations, models, and issue hints

Remaining work:

- richer ax-engine health and metrics translation beyond the generic
  OpenAI-compatible adapter path
- extraction or deprecation of embedded llama.cpp, MLX, libllama, and native
  runtime responsibilities after adapter replacements exist
- deeper runtime-specific diagnostics for ax-engine and vLLM beyond the common
  node contract fields
