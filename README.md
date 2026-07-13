# AX Serving

[![CI](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml/badge.svg)](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

AX Serving is a runtime-neutral inference gateway and fleet control plane. It
provides one OpenAI-compatible endpoint over AX Engine deployments on Apple
Silicon and certified vLLM or SGLang deployments on CUDA.

AX Serving does not generate tokens. The selected runtime owns tokenization,
chat templates, batching, KV cache, speculative decoding, distributed
execution, and hardware kernels. AX Serving owns authentication, admission,
fleet state, model deployment identity, endpoint selection, safe failover,
streaming transport, and operator workflows.

Project status: the runtime-neutral architecture is implemented in the source
tree and covered by mock/conformance tests. It is not yet certified as a
production-ready hybrid release. Live AX Engine plus CUDA certification,
production-envelope latency/goodput measurements, and the required 60-minute
soak artifacts must pass before that claim is made. See the
[canonical PRD](.internal/prd/PRD-AX-SERVING.md) for the release gates.

## Product boundary

The useful comparisons are at matching layers:

| Layer | AX stack | Comparable project |
| --- | --- | --- |
| Inference runtime | AX Engine on MLX | llama.cpp or another local inference runtime |
| Serving control plane | AX Serving | a vLLM-compatible gateway/control plane |
| CUDA execution behind AX Serving | vLLM or SGLang | the same runtime used directly |

AX Engine versus llama.cpp is an engine benchmark. AX Serving versus vLLM is
not an engine benchmark: AX Serving complements vLLM by managing it alongside
MLX runtimes. Serving measurements compare direct runtime traffic with the
same traffic through AX Serving, then test mixed-fleet failure and overload
behavior.

In this repository, “hybrid” means a fleet containing MLX and CUDA deployment
pools. One request attempt executes wholly on one compatible endpoint. AX
Serving does not split a model, prefill/decode phase, or KV cache across MLX
and CUDA.

## Architecture

```text
OpenAI client
     |
     v
AX Serving gateway ---- Redis/Valkey fleet state (HA profile)
     |
     +---- ax-runtime-agent ---- AX Engine / MLX
     |
     +---- ax-runtime-agent ---- vLLM or SGLang / CUDA
```

The default `ax-serving-api` binary is portable and has no dependency on AX
Engine, MLX, llama.cpp, Metal, or CUDA. The macOS-only embedded server and
gRPC v1 API are isolated behind the `embedded-compat` feature.

Key safety properties:

- versioned worker protocol with runtime-authoritative readiness and inventory;
- explicit logical models, pools, deployment identities, and equivalence classes;
- fail-closed routing for stale, incompatible, draining, or overloaded workers;
- at most one retry, only after a proven connect failure or authenticated typed
  pre-admission rejection;
- no retry after response commitment and no arbitrary-`5xx` rerouting;
- byte-incremental SSE proxying with cancellation and phased deadlines;
- separate public, admin, worker-control, dispatch, and runtime credentials;
- Redis/Valkey lease fencing, shared capacity reservations, probe ownership,
  and active-active gateway reconciliation;
- asynchronous deployment create, update/roll, drain, delete, and job APIs.

## Quick start: portable gateway and runtime agent

Build the portable binaries:

```bash
cargo build --release \
  -p ax-serving-cli --bin ax-serving-api \
  -p ax-thor-agent --bin ax-runtime-agent
```

Start a development gateway on loopback:

```bash
AXS_ALLOW_NO_AUTH=true target/release/ax-serving-api
```

Start an OpenAI-compatible runtime separately, then register it through the
agent. This example assumes the runtime listens on port 8000 and reports its
models from `/v1/models`:

```bash
AXS_CONTROL_PLANE_URL=http://127.0.0.1:19090 \
AXS_NODE_RUNTIME=vllm \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=127.0.0.1:18081 \
AXS_NODE_ADVERTISED_ADDR=127.0.0.1:18081 \
AXS_NODE_HARDWARE_CLASS=pc-cuda \
AXS_NODE_WORKER_POOL=cuda \
target/release/ax-runtime-agent
```

Use `AXS_NODE_RUNTIME=ax_engine` and an AX Engine OpenAI-compatible server URL
for Apple Silicon. The public request model is the runtime model ID in
`legacy_compat` mode:

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "replace-with-runtime-model-id",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "stream": false
  }'
```

For cross-runtime routing, start from
[`config/serving.hybrid.example.yaml`](config/serving.hybrid.example.yaml).
Replace every placeholder identity and certification path. Do not enable an
equivalence class until both deployment artifacts pass the same conformance
workload.

See [QUICKSTART.md](QUICKSTART.md) for an authenticated setup and
[the multi-worker runbook](docs/runbooks/multi-worker.md) for HA, drain,
rollout, and incident procedures.

## Public and operator APIs

Portable gateway inference endpoints:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`

Health and observability:

- `GET /livez` — process liveness;
- `GET /readyz` — `200` only when at least one worker is routable;
- `GET /health` — JSON fleet health summary;
- `GET /v1/metrics` — admin-authenticated JSON operational metrics;
- `GET /metrics` — admin-authenticated Prometheus `axs_gateway_*` metrics;
- `GET /dashboard` — compatibility convenience UI, usable only behind an
  authenticated admin reverse proxy that injects the bearer credential;
- `GET /v1/admin/status`, `/diagnostics`, `/audit`, `/fleet`.

Do not put an admin bearer token in a dashboard URL or browser storage. For
production monitoring, scrape `/metrics` from the monitoring system and use
the supplied Grafana dashboard instead of exposing `/dashboard` directly.

Asynchronous lifecycle APIs:

- `GET|POST /admin/v1/deployments`
- `GET|PATCH|DELETE /admin/v1/deployments/{id}`
- `GET /admin/v1/jobs`
- `GET /admin/v1/jobs/{id}`

The portable operator client is built with the gateway and links no runtime
SDK:

```bash
ax-servingctl status \
  --url http://127.0.0.1:18080 \
  --api-key public-key \
  --admin-key admin-key \
  --diagnostics --json
```

The portable gateway deliberately does not expose gRPC v1 or synchronous
fleet model load/unload. Those contracts depend on gateway-local paths,
backend enums, or token-ID streams and remain embedded compatibility only.
`/v1/responses` is deferred until a runtime adapter passes its protocol
conformance gate.

## Security profiles

Development may bind only to loopback with `AXS_TLS_PROFILE=loopback_dev` and
must explicitly set `AXS_ALLOW_NO_AUTH=true` when public auth is absent.

Remote deployments use `AXS_TLS_PROFILE=trusted_mesh`. This profile asserts
that a trusted ingress/service mesh supplies TLS or mTLS; AX Serving does not
create certificates itself.

| Credential | Purpose |
| --- | --- |
| `AXS_API_KEY` | Public inference clients |
| `AXS_ADMIN_API_KEY` | Admin and worker-management routes |
| `AXS_INTERNAL_API_TOKEN` | Gateway worker-control API credential |
| `AXS_WORKER_TOKEN` | Agent copy of the worker-control credential |
| `AXS_DISPATCH_TOKEN` | Gateway-to-agent inference dispatch |
| `AXS_RUNTIME_API_KEY` | Agent-to-runtime authentication |
| `AXS_REDIS_URL` | Shared HA state; treat as a secret |
| `AXS_CACHE_AFFINITY_SECRET` | Tenant-scoped opaque affinity digest, 32+ bytes |

Public `Authorization`, cookies, proxy credentials, and hop-by-hop headers are
not forwarded to runtimes. Raw affinity hints are keyed with the operator
secret and tenant ID, then discarded.

## Build and validation

Portable checks, suitable for Linux and macOS:

```bash
cargo check -p ax-serving-protocol
cargo check -p ax-serving-api --no-default-features --features gateway
cargo check -p ax-serving-cli --no-default-features --features gateway \
  --bin ax-serving-api
cargo check -p ax-thor-agent
```

Full workspace validation on supported macOS hosts:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
AXS_ALLOW_NO_AUTH=true cargo test --workspace --all-features
```

Python 3.14 may require PyO3's forward-compatibility switch while PyO3 catches
up:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check --workspace
```

The release workflow rejects tags while benchmark evidence contains null
baselines. Benchmark and soak commands must use release builds and retain raw,
schema-versioned artifacts; incomplete files are not publishable evidence.

## Deployment

- Single gateway development: in-memory fleet state and loopback agents.
- Active-active gateway: Redis/Valkey shared state, unique `AXS_GATEWAY_ID`,
  authenticated control/dispatch channels, and trusted transport.
- Kubernetes baseline: [deploy/kubernetes](deploy/kubernetes/README.md).
- Container targets: `gateway` and `agent` in
  [packaging/container/Dockerfile](packaging/container/Dockerfile).

## Repository layout

- `crates/ax-serving-protocol` — portable worker/deployment wire contract;
- `crates/ax-serving-api` — gateway, routing, HA state, lifecycle, REST/SSE;
- `crates/ax-thor-agent` — generic AX Engine/vLLM/SGLang runtime agent;
- `crates/ax-serving-cli` — portable gateway and embedded compatibility CLI;
- `crates/ax-serving-engine` — embedded compatibility backend abstraction;
- `crates/ax-serving-bench` — benchmark, regression, and soak runners;
- `crates/ax-serving-shim` and `crates/ax-serving-py` — compatibility bindings;
- `.internal` — canonical PRD, ADR, and technical specification;
- `docs` — public contracts, operations, and performance guidance.

## Canonical design documents

- [Product requirements](.internal/prd/PRD-AX-SERVING.md)
- [ADR-013: runtime-neutral hybrid inference control plane](.internal/adr/ADR-013-RUNTIME-NEUTRAL-HYBRID-INFERENCE-CONTROL-PLANE.md)
- [Technical specification](.internal/specs/TECH-SPEC-HYBRID-RUNTIME-CONTROL-PLANE.md)
- [Node protocol contract](docs/contracts/ax-serving-node-contract.md)
- [Multi-worker operations](docs/runbooks/multi-worker.md)
- [Service tuning and evidence](docs/perf/service-tuning.md)

## Licensing

AX Serving is available under
[AGPL-3.0-or-later](LICENSE). Separate commercial terms are described in
[LICENSING.md](LICENSING.md) and [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).
