# AX Serving

The Execution And Serving Control Plane For AX Fabric


[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-black)](https://github.com/defai-digital/ax-serving)
[![rust-1.88+](https://img.shields.io/badge/rust-1.88%2B-orange)](https://www.rust-lang.org)
[![cargo test · 295 passing](https://img.shields.io/badge/cargo%20test-295%20passing-brightgreen)](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml)
[![license-AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

AX Serving is the execution and model-serving control plane for [AX Fabric](https://github.com/defai-digital/ax-fabric). It turns local model runtime into an operational layer for agent workloads with OpenAI-compatible APIs, model lifecycle control, multi-worker orchestration, scheduling, and benchmarking.

AX Fabric is the product-facing knowledge and retrieval layer. AX Serving exists to power that stack by providing model access, execution control, request routing, and service operations.

Status: Production Ready | Rust Workspace | Apple Silicon (`aarch64-apple-darwin`) | OpenAI-Compatible REST + gRPC

> What AX Serving is: the execution/control-plane subsystem behind AX Fabric.

> What AX Serving does: provides model serving, execution orchestration, routing, queueing, runtime load/unload, and operator visibility for AX Fabric deployments.

> Product positioning: AX Fabric is the primary product; AX Serving is the infrastructure layer that makes AX Fabric deployable, scalable, and operable.

* * *

## Editions

Jump to: [OSS](#oss) | [Business](#business) | [Enterprise](#enterprise)

| Capability | OSS | Business | Enterprise |
| --- | --- | --- | --- |
| OpenAI-compatible REST + gRPC serving | Yes | Yes | Yes |
| Runtime model load/unload/reload APIs | Yes | Yes | Yes |
| Scheduler controls, metrics, and dashboard | Yes | Yes | Yes |
| Benchmark/soak tooling (`ax-serving-bench`) | Yes | Yes | Yes |
| Local/self-hosted deployment | Yes | Yes | Yes |
| Single-node runtime | Yes | Yes | Yes |
| Multi-node Mac Grid | No | Yes | Yes |
| Multi-node NVIDIA CUDA Grid | No | No | Yes |
| Commercial licensing terms | No | Included | Included |
| Contracted support/SLA | No | By agreement | By agreement |
| Enterprise procurement/compliance terms | No | Optional | Included by agreement |

<details>
<summary><strong id="oss">OSS</strong></summary>

- License: AGPL-3.0-only, with separate commercial licensing available.
- Includes AX Serving runtime for single-node deployments.
- Best for individual builders, prototyping, and teams operating under OSS terms.
- Multi-node/grid deployment is not part of OSS edition.

</details>

<details>
<summary><strong id="business">Business</strong></summary>

- Includes everything in OSS.
- Available under commercial terms (`LICENSE-COMMERCIAL.md`) as an alternative to AGPL obligations, with license key activation via `AXS_LICENSE_KEY` or `POST /v1/license`.
- Supports multi-node deployment on Mac Grid.
- Companies with annual revenue under USD 2M can use Business features at no cost.
- Optional paid support licenses are available.

</details>

<details>
<summary><strong id="enterprise">Enterprise</strong></summary>

- Includes everything in Business.
- Supports multi-node deployment across Mac Grid and NVIDIA CUDA Grid.
- Includes NVIDIA Jetson Thor optimizations for higher performance.
- Designed for enterprise-grade security and enterprise procurement/compliance needs.

</details>

* * *

## Quick Start (60 Seconds)

### 1. Prerequisites

- macOS on Apple Silicon
- Rust toolchain
- `llama-server` on `PATH`
- GGUF model file (example: `./models/<model>.gguf`)

```bash
cargo check --workspace
which llama-server
```

### 2. Start a Worker

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf \
  --model-id default \
  --host 127.0.0.1 \
  --port 18080
```

### 3. Send a Request

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

For full setup paths (single worker + gateway/worker), see [QUICKSTART.md](QUICKSTART.md).

TypeScript SDK (Zod-validated):

```bash
cd sdk/javascript
npm install
npm run build
```

---

## Why AX Serving

Most local runtimes optimize single-process inference. AX Serving focuses on the execution and serving layer required to run AX Fabric reliably, with model inference exposed as a managed runtime behind the product.

### Positioning

AX Serving should be understood as infrastructure, not as the top-level product.

- AX Fabric: the product users adopt for local retrieval, memory, ingestion, and grounded agent workflows
- AX Serving: the runtime/control plane that powers model execution, APIs, routing, scheduling, and deployment for AX Fabric
- Together: a complete local stack for grounded AI agents, with AX Fabric as the product surface and AX Serving as the execution substrate

- OpenAI-compatible REST endpoints
- gRPC serving/control plane
- Runtime model load/unload/reload
- Admission queue + concurrency controls
- Multi-worker orchestrator with health-aware dispatch
- Built-in benchmark and soak tooling in the same repo

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
| Dispatch policies (`least_inflight`, `weighted_round_robin`, `model_affinity`) | ✅ |
| Scheduler queue/inflight controls | ✅ |
| Prometheus + JSON metrics | ✅ |
| Embedded dashboard (`/dashboard`) | ✅ |
| Built-in benchmarking (`ax-serving-bench`) | ✅ |

---

## Run Modes

### Single Inference (No Server)

```bash
cargo run -p ax-serving-cli --bin ax-serving -- \
  -m ./models/<model>.gguf \
  -p "Hello from AX Serving" \
  -n 128
```

### Single Worker (REST + gRPC)

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf \
  --model-id default \
  --port 18080
```

### Multi-Worker (Gateway + Workers, Business/Enterprise)

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

Primary REST endpoints:

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
- `GET/POST /v1/license`
- `GET /v1/admin/status`
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

### v1.5 Control-Plane Surface

`v1.5` makes the orchestrator usable as a production-facing admin surface, not
just a request proxy.

- `GET /v1/admin/status` gives an authenticated operational summary for queue,
  dispatch policy, license state, worker counts, and reroute totals
- `GET /v1/workers` and `GET /v1/workers/{id}` expose authenticated worker
  inventory and per-worker telemetry
- `POST /v1/workers/{id}/drain` and `POST /v1/workers/{id}/drain-complete`
  support the public graceful-drain lifecycle for browser dashboards and ops
  tooling
- all authenticated admin responses preserve `X-Request-ID` so operators can
  correlate API calls with logs and incident notes

### v1.4 Runtime Controls

- `AXS_SPLIT_SCHEDULER=true`
  - enables prefill/decode activity tracking in scheduler metrics
- `AXS_MISTRALRS_MAX_SEQS=<n>`
  - controls `mistralrs` continuous-batching sequence depth

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

| Suite | Count | What It Covers |
|---|---|---|
| **Unit — serving API** | 163 | Scheduler (permits, AIMD, TTFT histogram, split prefill/decode), model registry (lifecycle, idle eviction, capacity), orchestration (queue, dispatch policies, worker registry, DashMap), REST helpers (cache key normalisation, cache hit ratio), config (env layering, validation), gRPC status mapping, auth, metrics |
| **Unit — engine** | 31 | Backend routing, GGUF metadata parsing, thermal state, memory budget |
| **Unit — C shim** | 22 | Null-safe llama.h ABI compatibility (21 exported functions) |
| **Integration — model\_management** | 54 | Auth (Bearer, whitespace tolerance, 401+WWW-Authenticate), model load/unload/reload (201/200/409/404/503), health semantics (ok/degraded/critical-thermal/no-models), input validation (400/422 on every field), full inference path (chat + completions via EchoBackend), embeddings (400/404/501), security response headers (nosniff, X-Frame-Options, X-Request-ID), metrics JSON keys, dashboard HTML, license GET/SET |
| **Integration — orchestration** | 23 | Worker register/heartbeat/eviction, dispatch (least-inflight, weighted round-robin, model-affinity, token-cost), queue admission and backpressure, reroute on 5xx, chaos (all workers fail → 503), overload (queue full → 429) |
| **Integration — graceful\_shutdown** | 2 | In-flight request drains to completion before server exits |
| **Total** | **295** | |

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
- [ROADMAP.md](ROADMAP.md)
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
