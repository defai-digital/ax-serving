# AX Serving implementation and certification status

| Field | Value |
| --- | --- |
| Snapshot | Working tree reviewed 2026-07-12 |
| Architecture | Implemented in source; not yet a released certification |
| Production claim | Blocked until every PRD release gate has retained evidence |
| Canonical requirements | [PRD](prd/PRD-AX-SERVING.md) |
| Architecture decision | [ADR-013](adr/ADR-013-RUNTIME-NEUTRAL-HYBRID-INFERENCE-CONTROL-PLANE.md) |
| Detailed design | [Technical specification](specs/TECH-SPEC-HYBRID-RUNTIME-CONTROL-PLANE.md) |

This ledger separates code completion from deployment certification. A passing mock or unit test is
implementation evidence, not proof that a pinned AX Engine/vLLM fleet satisfies the production
envelope.

## Requirement status

| Requirement group | Source status | Remaining certification work |
| --- | --- | --- |
| API-001 through API-005 | Implemented for REST JSON/SSE with stable AX errors and byte-preserving unknown fields. | Run the same API corpus against pinned live AX Engine and CUDA runtimes. |
| API-006 | Gated by design. `/v1/responses` is not exposed. | Certify one adapter and approve the protocol/API contract before implementation. |
| API-007 and API-008 | gRPC v1 is isolated in `ax-serving-grpc-compat` behind `embedded-compat`; no portable v2 is claimed. | Publish the compatibility support window; add v2 only after measured demand and a separate review. |
| WKR-001 through WKR-007 | Protocol v1, negotiation, authoritative readiness, leases, inventory, drain, identity, and typed admission are implemented. | Execute the conformance suite against pinned live runtime artifacts on the supported platform matrix. |
| WKR-008 | Protocol lifecycle records and desired-state observations exist. | Implement and certify runtime/orchestrator lifecycle adapters before claiming automatic model load/unload. |
| MOD-001 through MOD-005 | Logical aliases, pools, deployment identity, strict equivalence, and aggregate model listing are implemented. | Produce equivalence artifacts for every cross-runtime class used in production. |
| MOD-006 | Async admin jobs and shared desired-state records are implemented. | Certify external lifecycle execution and failure recovery against a real orchestrator/runtime. |
| REQ-001 through REQ-005 | Request profiles, hard admission, tenant/priority controls, deadlines, and tenant-scoped keyed cache affinity are implemented. | Validate limits and fairness at the production workload envelope. |
| RTE-001 through RTE-006 | Fail-closed eligibility and inference-aware scoring are implemented with bounded decision metrics. | Retain the 256-candidate p99 benchmark and mixed-fleet distribution evidence. |
| DSP-001 through DSP-006 | Request/attempt IDs, typed pre-admission, at-most-one safe retry, incremental SSE, phased deadlines, cancellation, and stream outcome tracking are implemented. | Repeat fault injection against live runtimes and a production ingress/service mesh. |
| OPS-001 through OPS-003 | Probes, fleet/worker diagnostics, authenticated operator metrics, portable `ax-servingctl`, registration, heartbeat, drain, and compatibility separation are implemented. | Run operator drills and validate monitoring ingestion in the target environment. |
| OPS-004 | Create/update/roll/drain/delete job APIs and rollback state transitions are implemented. | Connect and certify an external runtime/orchestrator lifecycle adapter. |
| OPS-005 | Memory and Redis/Valkey stores, fencing, reservations, probe ownership, and active-active reconciliation are implemented. | CI must pass with a real Redis service; run restart and partition tests on two deployed gateways. |

P1/P2 items are not allowed to delay a P0 bug fix, and their presence in a DTO is not a support
claim. The public README must continue to describe `/v1/responses`, automatic runtime lifecycle,
and portable gRPC v2 as unavailable.

## Non-functional and security gates

| Gate | Current evidence | Status |
| --- | --- | --- |
| Runtime-SDK-free gateway graph | Local `cargo tree` excludes AX Engine, MLX, llama.cpp, tonic, prost, and gRPC build tooling from default gateway features; CI enforces the same boundary. | Implemented; remote CI pending for this commit. |
| macOS arm64, Linux x86_64, Linux arm64 | Local macOS all-feature check passes; CI has native x86_64 and arm64 Linux jobs. | Remote platform runs pending. |
| NFR-001 gateway setup overhead | Histogram instrumentation and benchmark policy exist. | No retained production-envelope result. |
| NFR-002 endpoint selection p99 | Lock-free bounded histogram and 256-candidate design are present. | No retained p99 artifact. |
| NFR-003 and NFR-004 exclusion timing | Lease TTL and active probe transitions have deterministic tests. | Live timing evidence pending. |
| NFR-005 no retry after commitment | Typed retry and cancellation/fault tests pass in the local suite. | Live runtime/ingress fault injection pending. |
| NFR-006 active-active recovery | Shared-state restore, fencing, capacity reservation, and probe-ownership tests exist. | Two deployed gateways plus Redis restart/partition artifact pending. |
| NFR-007 goodput loss below 3 percent | Benchmark contract exists. | No qualifying direct-versus-gateway artifact. |
| NFR-008 60-minute soak | Soak runner and CI manual job exist. | No qualifying 60-minute mixed-fleet artifact. |
| Credential isolation | Public/admin/control/dispatch/runtime credentials are separate; public auth forwarding is denied and tested. | Live rotation drill pending. |
| Transport security | Non-loopback startup requires the `trusted_mesh` profile and independent control/dispatch credentials; deployment docs require TLS ingress and internal mTLS. | Target mesh policy and certificate/rotation evidence pending. |
| Privacy and observability | Prompt/output capture is off; W3C trace context, OTLP/HTTP JSON, bounded metrics, dashboard, and alert templates are implemented. | Collector, dashboard, and alert delivery validation pending. |

The live AX Engine 6.8.2 source contract was reviewed at this snapshot. The
runtime agent has exact fixtures for its nested `/v1/models` capability fields,
context/output limits, readiness behavior, authenticated runtime requests, and
the currently exposed `ax_engine_*` capacity metrics. This is source-contract
evidence, not a live model certification.

## Local verification snapshot

The following checks passed on Apple Silicon macOS in this working tree:

```text
cargo fmt --all -- --check
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check --workspace --all-targets --all-features --offline
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy --workspace --all-targets --all-features --offline -- -D warnings
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --workspace --all-features --offline
PYO3_BUILD_EXTENSION_MODULE=1 cargo build --release -p ax-serving-py --offline
npm test
npm run typecheck
python -m pytest -q
ruby YAML parse for deploy/**/*.yaml
jq empty deploy/monitoring/grafana-dashboard.json
```

The all-feature Rust run listed 1,054 test cases and passed 1,052, with one model-dependent shim
test and one doctest ignored. JavaScript passed its build/typecheck and one contract test; Python
passed nine tests, and its sdist/base wheel smoke test passed without installing the optional gRPC
extra. The separate macOS native compatibility extension built and imported successfully under
Python 3.14.
The Redis test in a developer run skips when `AXS_TEST_REDIS_URL` is absent; CI supplies a real
Redis service and treats that conformance test as required.

The local container build is not evidence because the configured Colima Docker daemon was stopped.
CI builds both `gateway` and `agent` image targets and verifies their non-root configuration.

The dependency lock uses the patched PyO3 0.29.0, crossbeam-epoch 0.9.20, anyhow 1.0.103, and
memmap2 0.9.11 releases. Remote RustSec remains the authoritative dependency-audit gate; the
unmaintained-only `paste` notice is inherited by compatibility dependencies through both the
current tokenizer stack and the frozen embedded AX Engine/Metal graph.

## Release blockers

The current placeholder benchmark files contain null values and are intentionally rejected by both
CI and the release workflow. Do not remove that release guard. A release candidate remains blocked
until all of the following are attached to a source digest:

1. Pinned AX Engine and one pinned CUDA runtime pass registration, readiness, inventory, inference,
   streaming, cancellation, drain, credential, and generic-`5xx` conformance.
2. Both portable gateway jobs, the embedded macOS compatibility job, and both portable container
   targets pass remotely.
3. Direct-runtime, through-gateway, and mixed-fleet artifacts pass NFR-001, NFR-002, and NFR-007.
4. The 32-worker, 256-stream, two-gateway Redis/Valkey envelope passes restart, partition, drain,
   rollout, rollback, overload, and at least 60 minutes of soak.
5. Transport, credential rotation, monitoring, alerting, incident, upgrade, and rollback drills are
   signed off with retained evidence.

Only after those artifacts pass may the PRD status and public README use “production-ready hybrid
inference control plane.”
