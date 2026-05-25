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

## 3. Phases

### Phase 1: Node Contract Foundation

Objective: make the existing worker registry describe runtime-neutral nodes.

Deliverables:

- accept `vllm` as a first-class backend/runtime kind
- expose runtime type in worker snapshots
- expose supported operations inferred from worker capabilities
- keep existing `native`, `llama_cpp`, `sglang`, and `auto` compatibility
- add focused registry tests for vLLM registration and runtime metadata

### Phase 2: Adapter Contract Documentation

Objective: document the public node contract expected from ax-engine and vLLM
nodes.

Deliverables:

- registration payload fields and compatibility rules
- heartbeat payload fields and optional telemetry semantics
- proxy target expectations for OpenAI-compatible runtimes
- capability-gated model lifecycle operations
- operator-facing diagnostics for mismatched runtime/model/capability

### Phase 3: Mac ax-engine Node Path

Objective: move Mac inference responsibility to ax-engine nodes.

Deliverables:

- ax-engine node registration path
- ax-engine health and model inventory translation
- gateway routing to ax-engine node endpoints
- migration guidance for existing embedded/native Mac paths

### Phase 4: vLLM Node Path For CUDA And Thor

Objective: support PC CUDA and NVIDIA Thor nodes through vLLM.

Deliverables:

- vLLM node registration path
- vLLM health and model inventory translation
- Thor agent default runtime alignment with vLLM
- gateway routing and status diagnostics for vLLM nodes

### Phase 5: Runtime Duplication Reduction

Objective: remove or quarantine responsibilities that duplicate runtime
engines.

Deliverables:

- classify embedded llama.cpp, MLX, libllama, and native paths as keep,
  adapterize, deprecate, or remove
- move runtime-specific tuning guidance out of serving-core requirements
- keep serving-layer benchmarks focused on routing, queueing, admission,
  failover, and mixed-node behavior

---

## 4. First Implementation Slice

The first code slice should be deliberately small:

- add `vllm` to the worker backend parser and serializer
- add a runtime kind derived from backend kind
- include `runtime` in worker snapshots
- include `supported_operations` in worker snapshots
- add tests proving vLLM registration and runtime metadata work

This creates the product contract surface needed by the PRD without changing
request dispatch semantics or replacing runtime adapters in the same step.

---

## 5. Validation

Focused validation for Phase 1:

- `cargo fmt --all -- --check`
- `cargo test -p ax-serving-api registry::tests`
- existing orchestration tests should continue to accept legacy worker
  registration payloads
- markdown link validation for PRD, ADR, contract, README, and runbook links

Broader validation for later phases:

- mixed Mac ax-engine and vLLM worker registration
- routing by model and runtime capability
- drain and failover across runtime classes
- dashboard and CLI status output for runtime class, node class, and model
  inventory

---

## 6. Implementation Progress

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

Remaining major slices:

- richer ax-engine health and metrics translation beyond the generic
  OpenAI-compatible adapter path
- extraction or deprecation of embedded llama.cpp, MLX, libllama, and native
  runtime responsibilities after adapter replacements exist
- dashboard rendering of runtime diagnostics beyond the JSON admin surface
