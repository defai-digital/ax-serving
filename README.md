# AX Serving

**Category:** Department-Scale Private AI Fleet Control Plane

**Product:** The serving and orchestration layer for multi-model private AI fleets operated by SMEs and enterprise departments.


[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-black)](https://github.com/defai-digital/ax-serving)
[![rust-1.88+](https://img.shields.io/badge/rust-1.88%2B-orange)](https://www.rust-lang.org)
[![Tests: 384 passing](https://img.shields.io/badge/tests-384%20passing-brightgreen)](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml)
[![license-AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

AX Serving is the serving and orchestration control plane behind
[AX Fabric](https://github.com/defai-digital/ax-fabric). It is designed for
department-scale private AI fleets that need OpenAI-compatible APIs, runtime
model lifecycle control, scheduling, metrics, audit surfaces, and multi-worker
routing across heterogeneous workers.

For inference execution, AX Serving uses:
- `llama.cpp` by default for all model loads
- `ax-engine` when explicitly requested via `native` backend override

AX Fabric is the product-facing layer for retrieval, knowledge, and grounded
agent workflows. AX Serving is the infrastructure layer that makes that stack
deployable and operable across Mac-led and mixed-worker environments.

Status: production-ready Rust workspace for Apple Silicon
(`aarch64-apple-darwin`) with OpenAI-compatible REST, gRPC, runtime model
management, and multi-worker orchestration oriented around department-scale
private AI serving.

## Market Focus

AX Serving is built to win in three adjacent niches:

- department-scale private AI fleet control planes
- Mac-native serving and orchestration for single-node and Mac-grid deployments
- enterprise mixed-worker orchestration across NVIDIA / Thor-class, Mac Studio-class, and future workers
- serving infrastructure for governed private AI stacks such as AX Fabric

Who it is for:

- SMEs and enterprise departments with fewer than ~100 users or operators
- platform and infra teams running private AI fleets
- operators who need more than a single local runtime process
- teams that care about model lifecycle, routing, metrics, health, audit, and fleet operations
- private deployments that need an OpenAI-compatible serving layer without a cloud-first dependency

What it is not:

- not an end-user desktop chat app
- not a generic CUDA hyperscale serving stack
- not the low-level token-generation engine itself

Deployment fit:

- `Single Mac`: the default open-source deployment path
- `Mac grid`: the default open-source multi-worker deployment path
- `Enterprise heterogeneous fleet`: commercial path for NVIDIA / Thor-class workers, governed mixed-node deployments, and enterprise delivery requirements

For market positioning, competitive analysis, and ICP details, see:

- [docs/market-positioning.md](docs/market-positioning.md)
- [docs/competitive-landscape.md](docs/competitive-landscape.md)
- [docs/icp-and-demand.md](docs/icp-and-demand.md)
- [docs/prd/PRD-AX-SERVING-v3.0.md](docs/prd/PRD-AX-SERVING-v3.0.md)
- [docs/prd/PRD-AX-SERVING-OSS-ENTERPRISE-BOUNDARY-v1.0.md](docs/prd/PRD-AX-SERVING-OSS-ENTERPRISE-BOUNDARY-v1.0.md)
- [docs/prd/PRD-AX-SERVING-ENTERPRISE-EXECUTION-v1.0.md](docs/prd/PRD-AX-SERVING-ENTERPRISE-EXECUTION-v1.0.md)
- [docs/contracts/ax-serving-public-contract-inventory.md](docs/contracts/ax-serving-public-contract-inventory.md)
- [docs/runbooks/enterprise-private-repo-bootstrap.md](docs/runbooks/enterprise-private-repo-bootstrap.md)
- [docs/runbooks/enterprise-release-governance.md](docs/runbooks/enterprise-release-governance.md)
- [docs/maintainability-refactor-plan.md](docs/maintainability-refactor-plan.md)

* * *

## Licensing And Commercial Use

AX Serving is dual-licensed:

- Open-source use: `AGPL-3.0-or-later`
- Commercial use: available under separate written license

Commercial licensing is intended for organizations that want to use AX Serving
as a proprietary serving backend, private inference/control plane, embedded
runtime, OEM component, managed fleet, or enterprise integration layer without
AGPL obligations.

Commercial engagements may include:

- commercial runtime licensing
- private deployment rights
- OEM / embedded redistribution rights
- enterprise fleet and mixed-node integration work
- support, service, and deployment terms

### Open-Source And Enterprise Boundary

The public repository is the open-source core of AX Serving.

The default open-source product scope is:

- single-Mac serving
- Mac-led local serving
- Mac worker grids
- core serving, orchestration, worker, metrics, and admin protocols

Commercial offerings cover one or both of the following:

- non-AGPL licensing rights for the AX Serving core itself
- separate enterprise modules, deployment bundles, and supported integrations

The intended enterprise expansion path is:

- NVIDIA / Thor-class workers
- heterogeneous Mac + accelerator fleets
- enterprise auth, governance, and deployment packaging
- supported private integrations and fleet operations tooling

The public repository contains the public source distribution, including
single-node and multi-worker serving/orchestration capabilities. Commercial
agreements govern usage outside AGPL obligations, private packaging, and
enterprise delivery terms. The recommended technical boundary is service-level
integration, not private crates mixed into the public workspace.

See [LICENSING.md](LICENSING.md) and
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

Execution artifacts for the open-source / enterprise split:

- [docs/contracts/ax-serving-public-contract-inventory.md](docs/contracts/ax-serving-public-contract-inventory.md)
- [docs/contracts/enterprise-compatibility-metadata.example.yaml](docs/contracts/enterprise-compatibility-metadata.example.yaml)
- [docs/runbooks/enterprise-private-repo-bootstrap.md](docs/runbooks/enterprise-private-repo-bootstrap.md)
- [docs/runbooks/enterprise-release-governance.md](docs/runbooks/enterprise-release-governance.md)

* * *

## Quick Start

Prerequisites:
- Apple Silicon macOS
- Rust toolchain
- `llama-server` on `PATH` for `llama.cpp` fallback and explicit `llama_cpp` loads
- a GGUF model file

Validate your environment:

```bash
cargo check --workspace
which llama-server
```

Backend model:
- `native` = explicit `ax-engine`
- `llama_cpp` = `llama-server` (default when backend is omitted)
- `auto` = try native first, then `llama.cpp` on unsupported architectures

Start the simplest local runtime:

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf \
  --model-id default \
  --host 127.0.0.1 \
  --port 18080
```

Send a request:

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Give me three short points about Rust."}],
    "stream": false,
    "max_tokens": 96
  }'
```

For fuller setup paths, see [QUICKSTART.md](QUICKSTART.md):
- single runtime
- authenticated offline deployment
- gateway + workers
- model management
- embeddings

TypeScript SDK (Zod-validated):

```bash
cd sdk/javascript
npm install
npm run build
```

---

## Why AX Serving

Most local runtimes focus on single-process inference. AX Serving focuses on the operational layer above inference:

- OpenAI-compatible REST and gRPC serving
- runtime model load/unload/reload
- admission queueing and concurrency control
- metrics, dashboard, diagnostics, and audit surfaces
- multi-worker orchestration in the public repo
- benchmark and soak tooling in the same repo

Positioning:
- AX Fabric is the product layer
- AX Serving is the serving and orchestration layer underneath it
- inference runtimes such as `ax-engine` and `llama.cpp` remain lower-level execution backends

### Backend Architecture

AX Serving is not itself the token-generation engine. It is the serving layer that routes requests into lower-level runtimes.

- `llama.cpp` is the default backend for model loading across families.
- `ax-engine` remains an explicit opt-in path for environments that can benefit from native execution.
- routing between those backends is controlled through [`config/backends.yaml`](config/backends.yaml)
- `ax-engine` is pinned to v1.2.2-compatible `0959a65` because `v1.3.1` regressed the shipped `gdn.metal` file and the `v1.3.2` commit path does not currently compile cleanly in this workspace snapshot.

In practice, this means AX Serving owns the APIs, scheduling, orchestration, health, metrics, and model lifecycle, while model execution defaults to `llama.cpp` with `ax-engine` as an explicit override.

### Best With AX Fabric

AX Serving is designed to work with AX Fabric as part of one complete system.

- AX Serving: execution control plane, model lifecycle, routing, scheduling, APIs
- AX Fabric: document ingestion, vector search, BM25/hybrid retrieval, MCP-native data access
- Together: AX Fabric is the product layer; AX Serving is the execution layer underneath it

---

## Core Capabilities

| Capability | AX Serving |
|---|---|
| OpenAI-compatible chat/completions/embeddings | ✅ |
| Streaming SSE + non-streaming responses | ✅ |
| Runtime model management (`/v1/models`) | ✅ |
| Multi-worker orchestration (`ax-serving-api`) | ✅ |
| Dispatch policies (`least_inflight`, `weighted_round_robin`, `model_affinity`, `token_cost`) | ✅ |
| Scheduler queue/inflight controls | ✅ |
| Prometheus + JSON metrics | ✅ |
| Embedded dashboard (`/dashboard`) | ✅ |
| Built-in benchmarking (`ax-serving-bench`) | ✅ |

---

## Run Modes

### 1. Single Inference CLI

```bash
cargo run -p ax-serving-cli --bin ax-serving -- \
  -m ./models/<model>.gguf \
  -p "Hello from AX Serving" \
  -n 128
```

### 2. Single Runtime (`ax-serving serve`)

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf \
  --model-id default \
  --port 18080
```

### 3. Gateway + Workers (`ax-serving-api` + workers)

Gateway:

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving-api -- \
  --port 18080 \
  --internal-port 19090 \
  --policy least_inflight
```

Worker:

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf \
  --model-id default \
  --port 18081 \
  --orchestrator http://127.0.0.1:19090
```

This gateway + worker path is part of the open-source Mac-native deployment
story. Enterprise fleet products build on the same serving contracts while
adding supported NVIDIA / Thor-class worker integrations, deployment bundles,
and governance layers under commercial terms.

---

## API Surface

### Serving runtime (`ax-serving serve`)

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `POST /v1/models`
- `DELETE /v1/models/{id}`
- `POST /v1/models/{id}/reload`
- `GET /health`
- `GET /v1/metrics`
- `GET /metrics`
- `GET /dashboard`
- `GET /v1/license`
- `POST /v1/license`
- `GET /v1/admin/status`
- `GET /v1/admin/startup-report`
- `GET /v1/admin/diagnostics`
- `GET /v1/admin/audit`
- `GET /v1/admin/policy`

### Orchestrator (`ax-serving-api`)

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /health`
- `GET /v1/metrics`
- `GET /v1/license`
- `POST /v1/license`
- `GET /v1/admin/status`
- `GET /v1/admin/startup-report`
- `GET /v1/admin/diagnostics`
- `GET /v1/admin/audit`
- `GET /v1/admin/policy`
- `GET /v1/admin/fleet`
- `GET /v1/workers`
- `GET /v1/workers/{id}`
- `POST /v1/workers/{id}/drain`
- `POST /v1/workers/{id}/drain-complete`
- `DELETE /v1/workers/{id}`

Runtime health contract:
- `GET /health` is liveness plus readiness, not just process-up status
- `status=ok` means the runtime is ready and at least one model is available
- `status=degraded` means the process is alive but either no model is loaded or the runtime is thermally constrained

AX Fabric integration contract:
- documented in [docs/contracts/ax-fabric-runtime-contract.md](docs/contracts/ax-fabric-runtime-contract.md)

Admin/control-plane notes:
- all authenticated admin responses preserve `X-Request-ID`
- `GET /v1/admin/status` gives an operational summary
- `GET /v1/admin/startup-report` and `GET /v1/admin/diagnostics` are for runtime inspection
- worker inventory and drain APIs are orchestrator-only

### v1.4 Runtime Controls

- `AXS_SPLIT_SCHEDULER=true`
  - enables prefill/decode activity tracking in scheduler metrics

Relevant scheduler metrics:
- `prefill_tokens_active`
- `decode_sequences_active`
- `split_scheduler_enabled`

---

## Authentication

- If `AXS_API_KEY` is set, protected endpoints require bearer auth.
- If `AXS_API_KEY` is unset, startup requires `AXS_ALLOW_NO_AUTH=true`.

Recommended offline enterprise startup:

```bash
AXS_CONFIG=config/serving.offline-enterprise.yaml \
AXS_API_KEY="change-me" \
AXS_MODEL_ALLOWED_DIRS="/absolute/path/to/models" \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m /absolute/path/to/models/<model>.gguf \
  --model-id default
```

```bash
AXS_API_KEY="token1,token2" cargo run -p ax-serving-cli --bin ax-serving -- serve -m ./models/<model>.gguf
```

Client header:

```bash
Authorization: Bearer token1
```

---

## Build, Lint, Test

```bash
cargo check --workspace
cargo fmt --all -- --check
cargo clippy --workspace --tests -- -D warnings
cargo test --workspace
```

Integration tests (no model required — uses in-process mock servers):

```bash
AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test orchestration
AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test model_management
AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test graceful_shutdown
```

Release build:

```bash
cargo build --workspace --release
```

### Test Coverage

All tests run automatically in CI on every push and pull request against `main`. No model file or GPU is required — tests use in-process backends (`NullBackend`, `EchoBackend`, `FailingUnloadBackend`) that exercise the full request path without hardware.

Exact test counts change over time. Use the linked CI badge and workflow runs as the source of truth.

| Suite | What It Covers |
|---|---|
| **Unit — serving API** | Scheduler (permits, AIMD, TTFT histogram, split prefill/decode), model registry (lifecycle, idle eviction, capacity), orchestration (queue, dispatch policies, worker registry, DashMap), REST helpers (cache key normalisation, cache hit ratio), config (env layering, validation), gRPC status mapping, auth, metrics |
| **Unit — engine** | Backend routing, GGUF metadata parsing, thermal state, memory budget |
| **Unit — C shim** | Null-safe llama.h ABI compatibility |
| **Integration — model\_management** | Auth (Bearer, whitespace tolerance, 401+WWW-Authenticate), model load/unload/reload (201/200/409/404/503), health semantics (ok/degraded/critical-thermal/no-models), input validation (400/422 on every field), full inference path (chat + completions via EchoBackend), embeddings, security response headers, metrics JSON keys, dashboard HTML, license GET/SET |
| **Integration — orchestration** | Worker register/heartbeat/eviction, dispatch (least-inflight, weighted round-robin, model-affinity, token-cost), queue admission and backpressure, reroute on 5xx, chaos (all workers fail → 503), overload (queue full → 429) |
| **Integration — graceful\_shutdown** | In-flight request drains to completion before server exits |

Every CI run posts a test summary to the GitHub Actions job summary page — see the [Actions tab](https://github.com/defai-digital/ax-serving/actions) for per-run results.

---

## Benchmarking

```bash
cargo run -p ax-serving-bench --release -- bench -m ./models/<model>.gguf
```

Other benchmark modes:

- `profile`
- `mixed`
- `cache-bench`
- `soak`
- `compare`
- `regression-check`
- `multi-worker`

---

## Repository Layout

- `crates/ax-serving-engine`: backend abstraction, routing, model internals
- `crates/ax-serving-api`: REST/gRPC serving, scheduler, orchestration
- `crates/ax-serving-cli`: `ax-serving` and `ax-serving-api` binaries
- `crates/ax-serving-bench`: benchmark and soak runners
- `crates/ax-serving-shim`: C-compatible shim
- `crates/ax-serving-py`: Python bindings
- `config/`: serving and routing configuration
- `docs/`: runbooks and architecture notes

---

## Documentation

- [QUICKSTART.md](QUICKSTART.md)
- [docs/market-positioning.md](docs/market-positioning.md)
- [docs/competitive-landscape.md](docs/competitive-landscape.md)
- [docs/icp-and-demand.md](docs/icp-and-demand.md)
- [docs/prd/PRD-AX-SERVING-v3.0.md](docs/prd/PRD-AX-SERVING-v3.0.md)
- [docs/maintainability-refactor-plan.md](docs/maintainability-refactor-plan.md)
- [docs/adr/README.md](docs/adr/README.md)
- `docs/contracts/ax-fabric-runtime-contract.md`
- `sdk/javascript/README.md` (TypeScript SDK with Zod validation)
- `sdk/python/` (Python SDK)
- `docs/runbooks/multi-worker.md`
- `docs/perf/service-tuning.md`

---

## Licensing

- Open-source terms: [AGPL v3 text](LICENSE) and [licensing guide](LICENSING.md)
- Commercial terms: [commercial licensing summary](LICENSE-COMMERCIAL.md)
- Issue reporting policy: [CONTRIBUTING.md](CONTRIBUTING.md)
