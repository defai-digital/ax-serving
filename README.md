# AX Serving

The Local LLM Serving Layer for Apple Silicon

AX Serving turns local model runtime into a production-style service with OpenAI-compatible APIs, model lifecycle control, multi-worker orchestration, and built-in benchmarking.

Status: Production Ready | Rust Workspace | Apple Silicon (`aarch64-apple-darwin`) | OpenAI-Compatible REST + gRPC

> What AX Serving does: runs GGUF models as a real service, not just a single-process inference script.

> Why teams use it: same local model workflow, but with queueing, health-aware routing, runtime load/unload, and operator visibility.

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

Most local runtimes optimize single-process inference. AX Serving focuses on service behavior around inference.

- OpenAI-compatible REST endpoints
- gRPC serving/control plane
- Runtime model load/unload/reload
- Admission queue + concurrency controls
- Multi-worker orchestrator with health-aware dispatch
- Built-in benchmark and soak tooling in the same repo

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

---

## Authentication

- If `AXS_API_KEY` is set, protected endpoints require bearer auth.
- If `AXS_API_KEY` is unset, startup requires `AXS_ALLOW_NO_AUTH=true`.

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

Release build:

```bash
cargo build --workspace --release
```

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
- `sdk/javascript/README.md` (TypeScript SDK with Zod validation)
- `sdk/python/` (Python SDK)
- `docs/runbooks/multi-worker.md`
- `docs/perf/service-tuning.md`
- `automatosx/prd/BMS-AX-SERVING-v1.0.md`

---

## Licensing

- Open-source option: [LICENSE](LICENSE), [LICENSE-AGPL.md](LICENSE-AGPL.md), and [LICENSING.md](LICENSING.md) (AGPL-3.0-only)
- Commercial option: [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)
- Issue reporting policy: [CONTRIBUTING.md](CONTRIBUTING.md)
