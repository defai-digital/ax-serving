# AX Serving

[![CI](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml/badge.svg)](https://github.com/defai-digital/ax-serving/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

AX Serving is an Apache-2.0 **federation gateway and multi-domain inference control plane** for
private AI fleets. It presents one authenticated OpenAI-compatible API and selects among
independently operated execution domains. It does not run models or schedule accelerator workers.

The target architecture federates:

- Apple Silicon Macs running AX Engine through `ax-runtime-agent`;
- NVIDIA GPU PCs managed inside an upstream NVIDIA Dynamo domain;
- NVIDIA Thor devices managed inside a separate Dynamo domain with independent qualification;
- future model-parallel Mac clusters represented by the experimental
  `ax-mac-cluster-adapter`.

The defining rule is:

> **AX Serving selects an execution domain. Dynamo selects NVIDIA workers. AX Engine executes on
> Mac. No layer makes the same placement decision twice.**

AX Serving does not generate tokens or replace the systems inside a domain. It owns the
cross-domain API boundary: public authentication, tenant and trust policy, logical models,
deployment identity and equivalence, admission, conservative failover, audit, and operator
workflows.

The CPU-only control plane is independent of inference hardware: it is designed to run on Apple
Silicon, Linux AMD64, or Linux ARM64—including a Thor host when co-location is appropriate—and can
govern local-office or remote Mac, NVIDIA PC, and Thor domains over a trusted network. Control-plane
portability does not imply that every host, runtime, or execution domain is production-qualified.
Running the gateway on a Thor host is a control-plane placement choice; it does not make Thor a
qualified inference domain. See
[Control-plane placement and mixed-fleet topologies](docs/deployment-topologies.md) for the
deployment matrix, adapter placement, LAN/WAN examples, and current claim boundary.

## Is AX Serving the right layer?

Hardware mix is not the product boundary. The boundary is whether several independently operated
execution domains need one policy and API.

| Fleet situation | Recommended entry point |
| --- | --- |
| One AX Engine endpoint with one policy | Call AX Engine directly |
| One NVIDIA Dynamo deployment with one policy | Call Dynamo directly |
| Several Mac endpoints needing shared admission, failover, drain, and audit | Evaluate AX Serving |
| Several CUDA/Dynamo domains separated by office, region, trust, rollout, or failure boundary | Evaluate AX Serving |
| Mac plus NVIDIA, or PC plus separately qualified Thor | Core AX Serving use case |

An all-Mac or all-CUDA fleet can use AX Serving. A mixed-hardware fleet does not automatically need
it. AX Serving earns the extra network hop and operational layer only when federation produces
measurable policy, availability, utilization, privacy/locality, cost/SLO, or workflow value.

## Project status

The v3 target architecture is accepted. Its implementation and qualification are incremental.
“Source implemented” below means that code and automated tests exist; it does not mean live
hardware, performance, fault, security, or production certification.

| Area | Current state |
| --- | --- |
| Portable REST/SSE gateway and protocol v1.2 | Source implemented and CI-tested on Linux AMD64/ARM64 and Apple Silicon; protocol v1.0/v1.1 migration fixtures remain supported |
| Domain catalog, deployment identity, equivalence, and hard eligibility | Foundation source implemented; complete policy and live mixed-domain evidence remain |
| Mac -> `ax-runtime-agent` -> AX Engine | Source implemented with mock tests; pinned live AX Engine qualification pending |
| Mac cluster -> `ax-mac-cluster-adapter` -> AX Engine ranks | Phase 1 coordinator, gang manifest, registration, proxy, and HA admission implemented with source/mock tests; distributed AX Engine PP is not implemented |
| Direct vLLM/SGLang -> `ax-runtime-agent` | Source implemented for migration/testing as `compatibility_runtime_endpoint`; not the target NVIDIA production path |
| NVIDIA PC -> `ax-dynamo-adapter` -> Dynamo | Adapter, manifest validation, registration, observation, and proxy source implemented with mock conformance; live qualification pending |
| NVIDIA Thor -> separate `ax-dynamo-adapter` -> Dynamo domain | Source path exists but is experimental; no Thor production-support claim |
| Streaming, cancellation, deadlines, and safe pre-commit retry | Implemented with mock/fault tests; live mixed-domain evidence pending |
| Fleet state and decisions | Generation-fenced domain reservations and bounded decision persistence implemented for memory and Redis/Valkey; two-gateway partition/soak and offline replay evidence pending |
| Packaging | CPU-only gateway/adapter container, Compose, Kubernetes, and Helm sources implemented; signed release artifacts and runtime qualification have separate gates |

Source and mock conformance are development evidence only. Production claims require retained
live-runtime qualification, performance, fault, and soak artifacts.

## Why AX Serving exists

| Layer | Owner |
| --- | --- |
| Cross-domain API, identity, tenant/privacy/locality policy, admission, audit | AX Serving |
| NVIDIA PC/Thor worker routing, KV, prefill/decode, planner, scaling | NVIDIA Dynamo |
| NVIDIA token execution | A Dynamo-certified vLLM, SGLang, or TensorRT-LLM backend |
| Apple Silicon token execution | AX Engine |

This product is useful when a fleet has multiple execution, trust, region, rollout, or failure
domains. Heterogeneous hardware is an important use case, not a requirement. If every request goes
to one NVIDIA deployment with one policy, call Dynamo directly; AX Serving would add an unnecessary
hop.

AX Serving v3 is the Apache-2.0 infrastructure layer described in this repository. All
functionality shipped here remains available under that license.
Separately distributed products such as AX Fabric and AX Trust may add orchestration, governance,
attestation, managed operations, or enterprise services through public contracts. They are not
required to use, modify, redistribute, or operate AX Serving, and they do not unlock or relicense
its open-source functionality.

## Architecture

```text
                              operators / policy
                                      |
OpenAI-compatible clients             v
            |              +-------------------------+
            +------------> | AX Serving gateway(s)   | <----> Redis/Valkey
                           | auth, model, policy,     |        (HA profile)
                           | domain choice, audit     |
                           +------+----------+-------+
                                  |          |
               +------------------+          +------------------+
               |                                                |
               v                                                v
      Mac AX Engine pool                              NVIDIA execution domains
      eligible Mac endpoint                           one AX endpoint per domain
               |                                                |
       ax-runtime-agent                         ax-dynamo-adapter (PC or Thor)
               |                                                |
          AX Engine                                  Dynamo frontend/router
                                                              |
                                                 Dynamo-selected backend workers
```

The gateway and adapters remain free of AX Engine, MLX, Metal, CUDA, Dynamo, vLLM, SGLang, and
TensorRT-LLM SDKs. Integration happens through versioned wire contracts and immutable deployment
identity.

The canonical Dynamo upstream is
[`https://github.com/ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo). Each AX qualification
manifest must pin an exact released tag, commit, backend version, graph configuration, and immutable
image digests; AX Serving does not maintain a private Dynamo fork. Upstream architecture support
does not by itself qualify a PC or Thor deployment for AX Serving.

### Execution domains

An execution domain is an independently operated routing, trust, qualification, and failure
boundary.

| Domain kind | AX-visible endpoint | Local execution owner | Qualification rule |
| --- | --- | --- | --- |
| `mac_ax_engine` | Each eligible Mac node in a Mac pool | `ax-runtime-agent` and AX Engine | Exact AX Engine/model identity must be certified |
| `mac_ax_engine_cluster` | One complete model-parallel Mac cluster | `ax-mac-cluster-adapter` and AX Engine | Coordinator/control-plane source exists; AX Engine distributed execution does not |
| `nvidia_dynamo_pc` | One adapter for one PC Dynamo deployment | Dynamo and its selected backend | Exact Dynamo/backend/image/config manifest must be certified |
| `nvidia_dynamo_thor` | One adapter for one separate Thor deployment | Dynamo and a Thor-qualified backend | Always separate from PC; experimental until its own gates pass |
| `compatibility_runtime_endpoint` | One direct vLLM/SGLang or other compatible runtime node | Configured runtime | Migration and testing only; never a Dynamo-domain claim |

PC and Thor never share a tensor-parallel group, artifacts, capacity calibration, or certification
by default. The legacy `ax-thor-agent` binary is only an alias for the generic
`ax-runtime-agent`; when pointed directly at vLLM/SGLang it registers a compatibility endpoint, not
a `nvidia_dynamo_thor` domain. One request attempt remains in one domain.

### Core model

| Object | Meaning |
| --- | --- |
| Logical model | Stable client-facing model name |
| Deployment | One concrete runtime model with immutable identity and capabilities |
| Equivalence class | Operator-approved deployments allowed to satisfy the same workload or failover |
| Pool | Policy and maintenance grouping |
| Execution domain | Routing, trust, qualification, and failure boundary |
| Domain observation | Bounded readiness, inventory, and aggregate capacity reported by an adapter |
| Compatibility manifest | Digest-pinned identity of an NVIDIA domain stack and its retained evidence |

A matching model string is never enough to prove equivalence. Model revision, artifact, tokenizer,
template, quantization, operations, limits, trust policy, and certification evidence are checked
explicitly.

### Request flow

1. The gateway authenticates the client and resolves the tenant plus logical model.
2. It rejects domains that fail qualification, freshness, readiness, trust, privacy, locality,
   capability, identity, equivalence, or capacity requirements.
3. It records a versioned, bounded decision and selects one eligible domain/deployment.
4. For Mac, AX selects an eligible node in the chosen pool and AX Engine executes locally. For
   NVIDIA, AX sends one attempt to the domain adapter and Dynamo selects the worker/backend.
5. JSON or SSE is streamed back byte-incrementally with cancellation and phased deadlines.
6. AX may retry once only before admission or response commitment. Dynamo owns retry and migration
   after a request enters an NVIDIA domain.

Hard constraints are filters, not score penalties. Missing or stale observations fail closed; they
never mean idle, compatible, or inexpensive.

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

## Explicit non-goals

AX Serving does not reproduce Dynamo's worker router, KV index, disaggregation, planner, scaling,
or backend lifecycle. It does not tokenize prompts, render chat templates, parse model artifacts,
or move KV state across Mac, PC, and Thor. It is not an agent planner, tool runner, MCP host,
sandbox, workflow engine, or durable agent-memory system.

## Development quick start: portable gateway and Mac agent

For a full walkthrough, including direct-runtime migration mode, explicit hybrid routing,
authentication, HA, lifecycle, and observability, see [QUICKSTART.md](QUICKSTART.md).

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

The current agent can also proxy direct vLLM/SGLang endpoints—including on Thor—for migration and
testing. That path always registers as `compatibility_runtime_endpoint`; it is not a Dynamo domain
and is not the target NVIDIA production architecture.
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
- Container targets: `gateway`, `agent`, `dynamo-adapter`, and experimental
  `mac-cluster-adapter` in
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
- `docs` — public contracts, operations, and performance guidance.

## Public contracts and guides

- [Runtime responsibility inventory](docs/contracts/ax-serving-runtime-responsibility-inventory.md)
- [Node protocol contract](docs/contracts/ax-serving-node-contract.md)
- [Product positioning and claim boundary](docs/market-positioning.md)
- [Use cases and trade-offs](docs/advantages-and-use-cases.md)
- [Deployment topologies and qualification boundary](docs/deployment-topologies.md)
- [NVIDIA Dynamo domain integration](docs/integrations/nvidia/DYNAMO.md)
- [Multi-worker operations runbook](docs/runbooks/multi-worker.md)

## Licensing

AX Serving is open-source software under the [Apache License, Version 2.0](LICENSE). See
[LICENSING.md](LICENSING.md), [NOTICE](NOTICE), and [TRADEMARKS.md](TRADEMARKS.md) for scope,
attribution, and trademark guidance. Version 2.3.0 removes the former commercial-license activation
API and configuration; see the [v2.3 migration note](docs/migrations/v2.3-apache-2.0.md).
