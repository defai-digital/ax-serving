# AX Serving

[![CI](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml/badge.svg)](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

AX Serving is being built as a **federated heterogeneous inference control plane** for private AI
fleets. It exposes one OpenAI-compatible API and governs execution across:

- Apple Silicon Macs running AX Engine through `ax-runtime-agent`;
- NVIDIA GPU PCs managed inside an upstream NVIDIA Dynamo domain;
- NVIDIA Thor devices managed inside a separate, independently qualified Dynamo domain.

AX Serving selects a domain. Dynamo selects NVIDIA workers. AX Engine executes on Mac.

AX Serving does not generate tokens or replace these systems. It owns public authentication,
tenant and trust policy, logical models, deployment identity/equivalence, cross-domain admission,
safe failover, audit, and operator workflows. Dynamo owns NVIDIA-local routing, KV-aware placement,
disaggregation, planning, scaling, and backend execution. AX Engine owns Apple Silicon
tokenization, templates, batching, caches, speculation, and kernels.

## Project status

The portable gateway, additive worker protocol v1.1, execution-domain catalog, explicit deployment
and equivalence checks, bounded in-memory decision records, streaming/cancellation, safe pre-commit
retry, HA fleet state, AX runtime agent, containers, Compose/Kubernetes, and Helm sources are
implemented. Protocol v1.0 fixtures and endpoint migration behavior remain supported.

The runtime-SDK-free Dynamo Domain Adapter and immutable compatibility-manifest validation are
implemented with source/mock conformance tests. Durable/replayable decision storage,
domain-aware HA reservations, and live federated certification are not yet implemented. NVIDIA
Thor is a target experimental domain, not a current support claim. See the
[status ledger](.internal/IMPLEMENTATION-STATUS.md) for exact evidence and blockers.

## Why AX Serving exists

| Layer | Owner |
| --- | --- |
| Cross-domain API, identity, tenant/privacy/locality policy, admission, audit | AX Serving |
| NVIDIA PC/Thor worker routing, KV, prefill/decode, planner, scaling | NVIDIA Dynamo |
| NVIDIA token execution | A Dynamo-certified vLLM, SGLang, or TensorRT-LLM backend |
| Apple Silicon token execution | AX Engine |

This product is useful when a fleet is genuinely heterogeneous or policy-segmented. If every
request goes to one NVIDIA deployment with one policy, call Dynamo directly; AX Serving would add
an unnecessary hop. AX Serving must prove value through cross-domain utilization, policy-correct
availability, privacy/locality, cost/SLO improvement, or simpler operations.

## Target architecture

```text
OpenAI-compatible clients
            |
            v
   AX Serving federation plane ---- Redis/Valkey (HA profile)
            |
            +---- Mac pool ---- ax-runtime-agent ---- AX Engine
            |
            +---- NVIDIA PC domain ---- ax-dynamo-adapter ---- Dynamo ---- backend workers
            |
            +---- NVIDIA Thor domain -- ax-dynamo-adapter ---- Dynamo ---- Thor workers
```

PC and Thor are always separate pools/domains by default. They do not share tensor-parallel groups,
TensorRT engine artifacts, quantization artifacts, capacity calibration, or certification. One
request attempt remains in one domain.

The canonical Dynamo upstream is
[`https://github.com/ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo). The design uses pinned
upstream releases and immutable NGC image digests through a service adapter; it does not maintain a
private Dynamo fork. Release `v1.2.1` is the initial integration reference, not an unqualified
support promise.

## Safety properties

- runtime-SDK-free portable gateway;
- versioned registration, lease, readiness, inventory, and drain protocol;
- explicit logical models, domains, pools, deployment identities, and equivalence classes;
- hard policy/capability/identity filters before scoring;
- missing or stale telemetry never interpreted as idle or compatible;
- one AX retry at most, only after connect failure or authenticated typed non-admission;
- no retry after admission ambiguity, response commitment, or first stream byte;
- Dynamo owns all retry/migration inside an NVIDIA domain;
- byte-incremental SSE, cancellation, and phased deadlines;
- separate public, admin, worker-control, dispatch, Dynamo, runtime, and fleet-store credentials;
- versioned, bounded, prompt-free routing decision diagnostics, with durable replay as a release gate;
- offline evaluation, shadow, canary, and rollback before an adaptive policy becomes active.

## Quick start: current portable gateway and Mac agent

Build the implemented portable binaries:

```bash
cargo build --release \
  -p ax-serving-cli --bin ax-serving-api \
  -p ax-thor-agent --bin ax-runtime-agent \
  -p ax-dynamo-adapter --bin ax-dynamo-adapter \
  -p ax-mac-cluster-adapter --bin ax-mac-cluster-adapter
```

Start a development gateway on loopback:

```bash
AXS_ALLOW_NO_AUTH=true target/release/ax-serving-api
```

Start `ax-engine-server` separately, then attach the current agent:

```bash
AXS_CONTROL_PLANE_URL=http://127.0.0.1:19090 \
AXS_NODE_RUNTIME=ax_engine \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=127.0.0.1:18081 \
AXS_NODE_ADVERTISED_ADDR=127.0.0.1:18081 \
AXS_NODE_HARDWARE_CLASS=apple-silicon \
AXS_NODE_WORKER_POOL=mac-mlx \
AXS_NODE_DOMAIN_ID=mac-local \
AXS_NODE_DOMAIN_QUALIFICATION=experimental \
target/release/ax-runtime-agent
```

`AXS_NODE_DOMAIN_ID` is opt-in. AX Engine agents advertise `mac_ax_engine`; direct vLLM/SGLang
agents can advertise only `compatibility_runtime_endpoint` and can never claim a Dynamo domain.
Use `certified` only when the exact runtime/model deployment has retained qualification evidence.

In `legacy_compat` mode, call the runtime model ID:

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

The current agent can also proxy direct vLLM/SGLang endpoints for migration and testing. That path
is compatibility-only in the final design; it is not the target NVIDIA production architecture.
For the NVIDIA path, follow the
[Dynamo domain guide](docs/integrations/nvidia/DYNAMO.md) and validate an immutable manifest before
starting `ax-dynamo-adapter`. Source/mock conformance is not live hardware certification.

The Mac cluster coordinator has a separate
[source-level setup guide](docs/integrations/mac/CLUSTER.md). It is useful for protocol and
coordinator integration today, but it cannot make a model span Macs until AX Engine implements the
manifest-bound pipeline executor and activation data plane.

For explicit deployment identity, start from
[`config/serving.hybrid.example.yaml`](config/serving.hybrid.example.yaml). It demonstrates explicit
Mac and compatibility-domain declarations; the CUDA entry remains a migration example, not the
target Dynamo production path. Replace every placeholder identity and never enable
cross-deployment equivalence without a retained certification artifact.

## Public and operator APIs

Portable inference endpoints:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`

Health and observability:

- `GET /livez` — process liveness;
- `GET /readyz` — control-plane readiness by default; does not require a routable domain;
- `GET /routablez` — at least one eligible inference deployment;
- `GET /health` — JSON control-plane and fleet-capacity summary;
- `GET /v1/metrics` — admin-authenticated JSON metrics;
- `GET /metrics` — admin-authenticated Prometheus metrics;
- `GET /v1/admin/status`, `/diagnostics`, `/audit`, `/decisions`, `/fleet`, `/deployments`.

Asynchronous lifecycle APIs:

- `GET|POST /admin/v1/deployments`
- `GET|PATCH|DELETE /admin/v1/deployments/{id}`
- `GET /admin/v1/jobs`
- `GET /admin/v1/jobs/{id}`

These jobs currently represent AX desired/observed state. They do not prove that AX creates a
Dynamo graph, downloads a model, or allocates a GPU. A certified external lifecycle controller is
a later integration.

The portable gateway deliberately does not expose gRPC v1, synchronous fleet model load/unload, or
`/v1/responses`. The former are embedded compatibility contracts; Responses remains gated on
end-to-end adapter certification.

## Security profiles

Loopback development requires `AXS_TLS_PROFILE=loopback_dev` and explicit
`AXS_ALLOW_NO_AUTH=true` when public auth is absent. Remote deployments use
`AXS_TLS_PROFILE=trusted_mesh`, where a trusted ingress or mesh supplies TLS/mTLS.

| Credential | Purpose |
| --- | --- |
| `AXS_API_KEY` | Public inference clients |
| `AXS_ADMIN_API_KEY` | Admin and monitoring routes |
| `AXS_INTERNAL_API_TOKEN` | Gateway worker/domain-control API |
| `AXS_WORKER_TOKEN` | Adapter copy of the control credential |
| `AXS_DISPATCH_TOKEN` | Gateway-to-adapter inference dispatch |
| `AXS_RUNTIME_API_KEY` | Mac agent-to-AX Engine authentication |
| `AXS_DYNAMO_API_KEY` | Dynamo adapter-to-frontend authentication |
| `AXS_REDIS_URL` | Shared AX HA state |
| `AXS_CACHE_AFFINITY_SECRET` | Tenant-scoped opaque affinity derivation |

Public authorization, cookies, proxy credentials, hop-by-hop headers, and AX internal headers are
not forwarded to runtimes.

## Build and validation

Portable checks:

```bash
cargo check -p ax-serving-protocol
cargo check -p ax-serving-api --no-default-features --features gateway
cargo check -p ax-serving-cli --no-default-features --features gateway --bin ax-serving-api
cargo check -p ax-serving-adapter-core
cargo check -p ax-dynamo-adapter
cargo check -p ax-thor-agent
```

Repository validation:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --tests -- -D warnings
cargo test --workspace --lib
AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test orchestration
```

Hardware-dependent compatibility and release tests require their pinned Mac, NVIDIA PC, or Thor
environments. A skipped hardware test is not support evidence.

## Deployment

- Evaluation: one gateway, in-memory state, loopback or trusted private adapters.
- HA candidate: two or more gateways, Redis/Valkey, unique gateway IDs, trusted transport, and
  separate credentials.
- Container targets: `gateway`, `agent`, and `dynamo-adapter` in
  [packaging/container/Dockerfile](packaging/container/Dockerfile).
- Compose: [deploy/compose](deploy/compose/README.md).
- Kubernetes baseline: [deploy/kubernetes](deploy/kubernetes/README.md).
- First-party CPU-only chart: [deploy/helm/ax-serving](deploy/helm/ax-serving/README.md).

The AX chart does not install Dynamo, GPU operators, AX Engine, backend runtimes, or model weights.
Dynamo remains a separately pinned and operated execution domain.

## Repository layout

- `crates/ax-serving-protocol` — portable worker/deployment/domain protocol v1.2;
- `crates/ax-serving-api` — gateway, catalog, routing, HA state, lifecycle, REST/SSE;
- `crates/ax-serving-adapter-core` — byte-preserving OpenAI/SSE adapter transport;
- `crates/ax-dynamo-adapter` — one runtime-SDK-free endpoint per NVIDIA Dynamo domain;
- `crates/ax-mac-cluster-adapter` — experimental coordinator/adapter for one future Mac AX Engine cluster;
- `crates/ax-thor-agent` — current package for the generic `ax-runtime-agent` binary;
- `crates/ax-serving-cli` — portable gateway and embedded compatibility CLI;
- `crates/ax-serving-engine` — embedded compatibility backends, not federation architecture;
- `crates/ax-serving-bench` — benchmark, regression, and soak runners;
- `deploy` — Compose, Kubernetes, Helm, monitoring, and alerting;
- `integrations/nvidia` — optional backend profiles and immutable manifest schema;
- `.internal` — canonical ADR, PRD, technical specification, and status ledger;
- `docs` — public contracts, operations, and performance guidance.

## Canonical design

- [ADR-016: Federated Dynamo and AX Engine control plane](.internal/adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md)
- [Product requirements](.internal/prd/PRD-AX-SERVING.md)
- [Technical specification](.internal/specs/TECH-SPEC-FEDERATED-INFERENCE-CONTROL-PLANE.md)
- [Implementation and certification status](.internal/IMPLEMENTATION-STATUS.md)
- [Runtime responsibility inventory](docs/contracts/ax-serving-runtime-responsibility-inventory.md)
- [Node protocol contract](docs/contracts/ax-serving-node-contract.md)

## Licensing

AX Serving is available under [AGPL-3.0-or-later](LICENSE). Separate commercial terms are
described in [LICENSING.md](LICENSING.md) and [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).
