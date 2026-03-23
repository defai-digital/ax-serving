# AX Serving

The Offline Serving And Orchestration Plane For AX Fabric


[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-black)](https://github.com/defai-digital/ax-serving)
[![rust-1.88+](https://img.shields.io/badge/rust-1.88%2B-orange)](https://www.rust-lang.org)
[![Tests: 384 passing](https://img.shields.io/badge/tests-384%20passing-brightgreen)](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml)
[![license-AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

AX Serving is the offline serving and orchestration layer behind [AX Fabric](https://github.com/defai-digital/ax-fabric). It provides OpenAI-compatible APIs, runtime model lifecycle control, scheduling, metrics, and multi-worker routing on Apple Silicon.

For inference execution, AX Serving uses:
- `ax-engine` for the native backend on supported model families
- `llama.cpp` for compatibility, embeddings, and fallback paths

AX Fabric is the product-facing layer for retrieval, knowledge, and grounded agent workflows. AX Serving is the infrastructure layer that makes that stack deployable and operable.

Status: production-ready Rust workspace for Apple Silicon (`aarch64-apple-darwin`) with OpenAI-compatible REST, gRPC, runtime model management, and optional multi-worker orchestration.

What AX Serving is:
- the serving/control-plane subsystem behind AX Fabric
- not the end-user product surface
- not the low-level inference engine itself

What AX Serving is for:
- single-node offline serving
- multi-worker model routing and health-aware dispatch
- operator-facing runtime control, diagnostics, metrics, and audit surfaces
- OSS and Business deployments of the public repo

* * *

## Editions

Jump to: [OSS](#oss) | [Business](#business)

| Capability | OSS | Business |
| --- | --- | --- |
| OpenAI-compatible REST + gRPC serving | Yes | Yes |
| Runtime model load/unload/reload APIs | Yes | Yes |
| Scheduler controls, metrics, dashboard, and admin APIs | Yes | Yes |
| Benchmark/soak tooling (`ax-serving-bench`) | Yes | Yes |
| Single-node offline runtime | Yes | Yes |
| Multi-worker Mac Grid orchestration | No | Yes |
| Commercial licensing terms | No | Included |
| Optional support arrangements | No | By agreement |

<details>
<summary><strong id="oss">OSS</strong></summary>

- License: AGPL-3.0-only
- Best for local builders, evaluation, and teams operating under OSS terms
- Single-node serving surface in the public repo

</details>

<details>
<summary><strong id="business">Business</strong></summary>

- Includes everything in OSS
- Available under [commercial terms](LICENSE-COMMERCIAL.md) as an alternative to AGPL obligations
- Adds multi-worker Mac Grid deployment in the public repo
- License key activation via `AXS_LICENSE_KEY` or `POST /v1/license`

</details>

Private enterprise work is handled in a separate private project. This public repository covers OSS and Business.

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
- `native` = `ax-engine`
- `llama_cpp` = `llama-server`
- `auto` = prefer `ax-engine` native, fall back to `llama.cpp` when needed

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
- multi-worker orchestration for Business deployments
- benchmark and soak tooling in the same repo

Positioning:
- AX Fabric is the product layer
- AX Serving is the serving and orchestration layer underneath it
- inference runtimes such as `ax-engine` and `llama.cpp` remain lower-level execution backends

### Backend Architecture

AX Serving is not itself the token-generation engine. It is the serving layer that routes requests into lower-level runtimes.

- `ax-engine` is the native backend used by AX Serving for supported GGUF architectures on Apple Silicon
- `llama.cpp` remains the subprocess backend for compatibility, fallback, and embedding-oriented paths
- routing between those backends is controlled through [`config/backends.yaml`](config/backends.yaml)

In practice, this means AX Serving owns the APIs, scheduling, orchestration, health, metrics, and model lifecycle, while `ax-engine` owns the native inference path.

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

### 3. Gateway + Workers (`ax-serving-api` + workers, Business)

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

### Orchestrator (`ax-serving-api`, Business)

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
- `AXS_MAX_BATCH_SIZE` / `AXS_BATCH_WINDOW_MS`
  - currently advisory scheduler hints only
  - they are exposed for future scheduler work and do not drive a batching loop today

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
- `docs/contracts/ax-fabric-runtime-contract.md`
- `sdk/javascript/README.md` (TypeScript SDK with Zod validation)
- `sdk/python/` (Python SDK)
- `docs/runbooks/multi-worker.md`
- `docs/perf/service-tuning.md`

---

## Licensing

- Open-source terms: [AGPL v3 text](LICENSE) and [licensing guide](LICENSING.md)
- Commercial terms: [commercial license](LICENSE-COMMERCIAL.md)
- Issue reporting policy: [CONTRIBUTING.md](CONTRIBUTING.md)
