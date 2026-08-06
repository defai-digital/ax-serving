# Federated fleet operations runbook

| Field | Value |
| --- | --- |
| Current scope | Portable `ax-serving-api`, protocol v1.2 foundation, and `ax-runtime-agent` |
| Target scope | Mac AX Engine pools plus separate NVIDIA PC/Thor Dynamo domains |
| Last updated | 2026-08-05 |
| Status | Current Mac/compatibility procedures are runnable; Dynamo/Thor procedures are qualification targets |

This is an operator runbook, not the first installation guide. Start with the
[quick start](../../QUICKSTART.md) if you have not yet routed one successful request through a
single runtime. Use this runbook when adding workers, enabling explicit deployment identity,
operating multiple gateways, or rehearsing drain and failure behavior.

This runbook separates current source behavior from the final architecture. Execution-domain
types/config/diagnostics ship in source, but the Dynamo Domain Adapter and live federation
certification do not. Do not use target examples or source/mock conformance as production support
evidence.

## 1. Ownership boundary

AX Serving owns public auth, tenant/trust policy, logical models, identity/equivalence,
cross-domain admission/selection, one safe pre-commit retry, audit, and global lifecycle intent.

NVIDIA Dynamo owns worker routing, KV-aware placement, prefill/decode, KVBM/NIXL, planner, scaling,
backend execution, and in-domain retry/migration. AX Engine owns token execution on Mac.

One request attempt runs wholly inside one domain. PC and Thor are separate domains.

## 2. Deployment profiles

### Loopback evaluation

- one AX gateway;
- `fleet_store: memory`;
- current `legacy_compat` or explicit test catalog;
- loopback Mac agent/runtime;
- `AXS_TLS_PROFILE=loopback_dev`;
- `AXS_ALLOW_NO_AUTH=true` only by explicit operator choice.

### Federation production candidate

- two or more AX gateways with unique `AXS_GATEWAY_ID` values;
- Redis/Valkey AX fleet state;
- `deployment_mode: explicit`;
- one pinned/certified Mac AX Engine path;
- one pinned/certified NVIDIA PC Dynamo domain and adapter;
- Thor disabled or a separate experimental/certified domain;
- authenticated public, admin, control, dispatch, Dynamo, runtime, and Redis hops;
- TLS/mTLS through an operator-provided trusted private network;
- immutable identity, compatibility manifests, equivalence, and retained release evidence.

The source tree is not a production certification.

## 3. Start the current gateway

```bash
cargo build --locked --release -p ax-serving-cli --bin ax-serving-api
```

Development:

```bash
AXS_ALLOW_NO_AUTH=true target/release/ax-serving-api
```

Active-active candidate:

```bash
AXS_CONFIG=/etc/ax-serving/serving.yaml \
AXS_FLEET_STORE=redis \
AXS_REDIS_URL='rediss://user:password@redis.example:6379/0' \
AXS_GATEWAY_ID='gateway-a' \
AXS_TLS_PROFILE=trusted_mesh \
AXS_API_KEY='public-client-key' \
AXS_ADMIN_API_KEY='admin-key' \
AXS_INTERNAL_API_TOKEN='adapter-control-key' \
AXS_DISPATCH_TOKEN='gateway-adapter-key' \
target/release/ax-serving-api
```

Start a second replica as `gateway-b` with the same catalog/key prefix and a different gateway ID.
Never duplicate a gateway ID.

## 4. Attach a current Mac AX Engine node

Build the agent:

```bash
cargo build --locked --release -p ax-thor-agent --bin ax-runtime-agent
```

Start the pinned AX Engine server on the Mac, then attach the agent:

```bash
AXS_CONTROL_PLANE_URL=https://ax-serving-control.example \
AXS_NODE_RUNTIME=ax_engine \
AXS_RUNTIME_VERSION='replace-with-pinned-version' \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=0.0.0.0:18081 \
AXS_NODE_ADVERTISED_URL=http://10.20.1.10:18081 \
AXS_NODE_ID=mac-mlx-01 \
AXS_NODE_WORKER_POOL=mac-mlx \
AXS_NODE_HARDWARE_CLASS=apple-silicon \
AXS_NODE_DOMAIN_ID=mac-local \
AXS_NODE_DOMAIN_QUALIFICATION=certified \
AXS_TRUST_DOMAIN=private-local \
AXS_TLS_PROFILE=trusted_mesh \
AXS_WORKER_TOKEN='adapter-control-key' \
AXS_DISPATCH_TOKEN='gateway-adapter-key' \
AXS_RUNTIME_API_KEY='runtime-only-key' \
target/release/ax-runtime-agent
```

Set `AXS_NODE_DOMAIN_QUALIFICATION=certified` only when the exact AX Engine/runtime/model identity
has retained qualification evidence. Otherwise use `experimental` or the default `unverified`.
`AXS_NODE_DOMAIN_COMPATIBILITY_MANIFEST=sha256:<digest>` is optional for a manifest-backed node.

Set every required identity variable from retained artifacts:

```text
AXS_MODEL_REVISION
AXS_MODEL_ARTIFACT_DIGEST
AXS_MODEL_TOKENIZER_DIGEST
AXS_MODEL_TEMPLATE_DIGEST
AXS_MODEL_QUANTIZATION
AXS_MODEL_MAX_OUTPUT_TOKENS
AXS_MODEL_CAPABILITIES
```

The agent is eligible only after AX Engine readiness and inventory are authoritative. The agent
process being alive is insufficient.

## 5. NVIDIA domains

### Current compatibility path

The current `ax-runtime-agent` can point directly at vLLM, SGLang,
TensorRT-LLM, or Thor-only TensorRT Edge-LLM for migration
and testing. This is not the final production NVIDIA architecture and must not be used as Dynamo
federation evidence. The single-PC NVIDIA Compose profiles and generic
no-retry qualification runner live under
[`deploy/compose`](../../deploy/compose/README.md#nvidia-runtime-profiles-on-one-pc);
the native Thor Edge-LLM path lives under [`deploy/thor`](../../deploy/thor/README.md)
and [`scripts/qualification/runtime`](../../scripts/qualification/runtime/README.md).

### Generic OpenAI-compatible runtime (llama.cpp example)

The same agent fronts any OpenAI-compatible runtime — llama.cpp `llama-server`, `mlxcel-server`,
Ollama, and similar — as a `compatibility_runtime_endpoint` node. Start the runtime first (for
llama.cpp, `llama-server --port 8080 --metrics`), then attach the agent:

```bash
AXS_CONTROL_PLANE_URL=https://ax-serving-control.example \
AXS_NODE_RUNTIME=llamacpp \
AXS_RUNTIME_VERSION='replace-with-pinned-version' \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8080 \
AXS_NODE_LISTEN_ADDR=0.0.0.0:18081 \
AXS_NODE_ADVERTISED_URL=http://10.20.1.11:18081 \
AXS_NODE_ID=generic-llamacpp-01 \
AXS_NODE_WORKER_POOL=generic-compat \
AXS_NODE_HARDWARE_CLASS=x86-cpu \
AXS_NODE_DOMAIN_ID=generic-local \
AXS_NODE_DOMAIN_QUALIFICATION=experimental \
AXS_TRUST_DOMAIN=private-local \
AXS_TLS_PROFILE=trusted_mesh \
AXS_WORKER_TOKEN='adapter-control-key' \
AXS_DISPATCH_TOKEN='gateway-adapter-key' \
target/release/ax-runtime-agent
```

Runtime-specific notes:

- **Readiness path**: the agent probes `GET /health` by default and treats any 2xx as ready. For a
  runtime without a health endpoint (e.g. Ollama), set `AXS_NODE_RUNTIME_HEALTH_PATH=/v1/models`
  (legacy alias `AXS_THOR_RUNTIME_HEALTH_PATH`); llama.cpp's `llama-server` already serves
  `/health` and needs no override.
- **Telemetry**: if the runtime exposes Prometheus `/metrics`, the agent recognizes the llama.cpp
  gauges `llamacpp:requests_processing` (active) and `llamacpp:requests_deferred` (queued) plus the
  built-in AX/vLLM/SGLang alias tables. For other metric names, map them explicitly with
  `AXS_NODE_METRIC_QUEUE_DEPTH` and `AXS_NODE_METRIC_ACTIVE_SEQUENCES` (legacy `AXS_THOR_METRIC_*`
  aliases), e.g. `AXS_NODE_METRIC_QUEUE_DEPTH=myruntime_requests_queued`.
- **Missing telemetry stays missing**: a runtime without queue metrics reports queue depth as
  unknown, not zero. The default dispatch policy penalizes unknown signals, so saturation-aware
  dispatch keeps working instead of over-preferring a node that merely fails to report.
- Set `AXS_NODE_DOMAIN_QUALIFICATION=certified` only with retained live-conformance evidence for
  the exact runtime/model identity; otherwise keep `experimental` or the default `unverified`.

### Target PC Dynamo path

The approved target is:

```text
AX gateway -> ax-dynamo-adapter -> pinned Dynamo frontend -> Dynamo-selected backend worker
```

Before enabling dispatch, the operator must have:

- a released upstream `ai-dynamo/dynamo` tag/commit;
- immutable frontend/planner/operator/runtime image digests;
- pinned backend, CUDA, OS/architecture, graph config, and model identities;
- a validated AX Dynamo compatibility manifest;
- adapter readiness/inventory/metrics mapping conformance;
- direct Dynamo versus through-AX overhead and fault evidence.

No runnable adapter command is documented until the binary exists and these contracts pass.

### Target Thor path

Thor uses a different domain ID, pool, manifest, qualification, and rollout. Keep it disabled or
`experimental` until live ARM64/CUDA/backend, memory/thermal, restart, stream/cancel, performance,
and soak evidence passes. Do not reuse PC TensorRT engines, quantization artifacts, capacity curves,
or planner calibration.

## 6. Validate gateway and capacity

```bash
curl -fsS http://gateway:18080/livez
curl -fsS http://gateway:18080/readyz
curl -i http://gateway:18080/routablez
curl -sS http://gateway:18080/health | jq .
curl -sS http://gateway:18080/v1/models \
  -H 'Authorization: Bearer public-client-key' | jq .
```

Expected semantics:

- `/livez` 200: process is alive;
- `/readyz` 200 in default `control_plane` mode: listeners/config/fleet store can operate, even with
  no execution capacity;
- `/routablez` 200: at least one deployment is eligible; 503 before a node/domain is routable;
- `/health`: separates control-plane health from worker/domain capacity.

Legacy `readyz_mode=eligible_workers` is migration-only. Monitoring that pages on serving capacity
uses `/routablez`, not `/readyz`.

Admin/control detail:

```bash
curl -sS http://gateway:18080/v1/admin/fleet \
  -H 'Authorization: Bearer admin-key' | jq .
curl -sS http://gateway:18080/admin/v1/deployments \
  -H 'Authorization: Bearer admin-key' | jq .
curl -sS 'http://gateway:18080/v1/admin/decisions?limit=50' \
  -H 'Authorization: Bearer admin-key' | jq .
curl -sS http://gateway:19090/internal/workers \
  -H 'X-Internal-Token: adapter-control-key' | jq .
```

Domain descriptors/observations appear when a v1.1 adapter reports them. The decision endpoint is a
bounded pre-dispatch diagnostic journal. The memory store is process-local; Redis/Valkey retains
the same records for the current fixed 24-hour TTL. Neither mode is an execution receipt, signature,
or hardware attestation.

## 7. Identity and equivalence

Every explicit deployment declares logical model, runtime model, pool/domain, runtime/backend,
model revision/artifact, tokenizer/template, quantization, operations, limits, trust class, and
required matching fields.

Cross-domain retry/failover requires:

1. both deployment IDs in one operator-certified equivalence class;
2. every required identity field present and matching;
3. `certification_artifact_digest` set to the SHA-256 or BLAKE3 digest of an immutable retained
   workload qualification artifact;
4. operation/capability/quality and trust/residency compatibility.

A syntactically valid digest only pins bytes; it does not validate test quality or fetch the
artifact. A shared model string is not equivalence. Different formats/quantizers must be disclosed
and tested.

## 8. Retry and cancellation

AX creates one request ID and a unique attempt ID per selected domain. It may perform at most two
domain attempts.

A second AX attempt is allowed only when:

- no response header/body byte was committed;
- connection failed before admission or the authenticated adapter returned typed `not-admitted`;
- the next deployment is equivalent and policy-eligible;
- the absolute deadline remains.

Generic runtime/Dynamo/backend `5xx`, post-write disconnect, timeout after ambiguous admission, and
committed streams are never retried by AX. Dynamo owns internal worker retry/migration as part of
the same AX attempt.

Client disconnect and deadlines propagate to the adapter and runtime/domain owner.

## 9. Admission controls

| Variable | Default | Meaning |
| --- | ---: | --- |
| `AXS_GLOBAL_QUEUE_MAX` | `128` | Active gateway requests |
| `AXS_GLOBAL_QUEUE_DEPTH` | `256` | Waiting requests |
| `AXS_GLOBAL_QUEUE_WAIT_MS` | `10000` | Maximum queue wait |
| `AXS_GLOBAL_QUEUE_POLICY` | `queue` | `queue`, `reject`, or `shed_oldest` |
| `AXS_TENANT_MAX_CONCURRENT` | `0` | Per-tenant active cap; zero disables |

The global queue limits work entering the gateway; it does not reserve capacity inside every
eligible runtime. A request can pass gateway admission and still receive `503` when all matching
workers are at their reported `max_inflight`. Clients should use a bounded, jittered backoff for
that pre-dispatch capacity response and keep one absolute request deadline. A zero-delay retry loop
amplifies overload. `429` admission responses include `Retry-After`; do not assume every capacity
`503` does.

In a trusted evaluation environment, set `AXS_ROUTING_TRACE=true` and retain the bounded
`x-ax-routing-trace` response header to verify candidate count, selected worker, and route reason.
Do not expose internal worker identifiers through an untrusted ingress.

Priority is `low`, `normal`, or `high`. AX does not perform token-level scheduling.

Optional cache/session affinity is tenant-scoped, bounded, and soft. It never overrides readiness,
capacity, policy, identity, or equivalence. Raw hints and prompts are not stored or forwarded.

## 10. Drain and roll

### Mac endpoint drain

1. Agent marks itself draining and stops local admission.
2. It sends an authoritative draining observation.
3. New requests select another eligible endpoint/domain or receive unavailable.
4. Existing streams finish until the deadline.
5. Agent reports drain complete and terminates/fences its registration.

Emergency admin drain:

```bash
curl -sS -X POST http://gateway:18080/v1/workers/WORKER_ID/drain \
  -H 'Authorization: Bearer admin-key'
```

### Dynamo domain drain target

The Dynamo adapter stops AX admission first. Dynamo's documented drain/rollout mechanism then owns
internal workers. AX does not remove individual Dynamo workers. Domain drain completes only when no
AX-visible admitted requests remain or the hard deadline expires.

### Deployment roll

AX async jobs represent desired state:

1. enable immutable replacement identity/manifest;
2. wait for a compatible replacement domain/endpoint to be ready;
3. stop new admission to the source;
4. drain source;
5. rollback desired state if readiness/health gates fail.

External runtime/Dynamo controllers perform actual process/graph/image changes.

## 11. Gateway upgrade and rollback

1. Verify protocol fixture compatibility and retain prior image/config/policy.
2. Add one new gateway with a unique ID.
3. Verify `/readyz`, `/routablez`, shared fleet/catalog/reservations, and metrics.
4. Send canary traffic and compare commitment/error/retry/latency signals.
5. Replace remaining gateways one at a time and drain streams.
6. Roll back to the prior immutable image/policy on guardrail failure.

Never clear Redis as a routine rollback. Incompatible state/protocol requires the release-specific
migration procedure.

## 12. Credentials and transport

Rotate independently:

1. public client identity;
2. admin/monitoring identity;
3. AX adapter control and lease identity;
4. gateway-to-adapter dispatch identity;
5. Mac runtime identity;
6. Dynamo adapter-to-frontend identity;
7. Dynamo lifecycle controller/RBAC identity;
8. Redis/Valkey identity;
9. affinity derivation secret, accepting cache-locality loss.

Never store secrets in checked-in YAML, images, URLs, metrics labels, or retained CI command output.

## 13. Observability and incidents

Baseline alerts cover:

- control-plane not ready;
- no routable deployment/domain;
- stale leases/observations or manifest mismatch;
- queue rejection/shed/timeout;
- retry increase by owner/reason;
- admission ambiguity or commitment failure;
- adapter/Dynamo/Mac error and cancellation rates;
- Redis fencing/reconciliation failure;
- rollout/drain deadline and policy rollback guard.

Incident order:

1. stop unsafe admission or disable the affected domain/deployment;
2. preserve existing streams when safe;
3. confirm retry/commitment state before manually replaying work;
4. inspect bounded pre-dispatch decisions, audit events, and pinned manifests;
5. restore an immutable prior policy/domain version;
6. retain timeline, config, traces, and raw evidence without prompts/secrets.

Loss of Thor must not affect PC/Mac control paths; loss of PC must not cause unsafe Thor/Mac failover.

## 14. Release checklist

- Mac AX Engine live conformance passed.
- NVIDIA PC Dynamo compatibility manifest and live conformance passed.
- Thor disabled/experimental or independently certified.
- Every cross-domain equivalence transition has retained evidence.
- Direct-versus-AX overhead and 256-candidate selection gates passed.
- Two-gateway Redis restart/partition/fencing/reservation tests passed.
- 60-minute mixed-domain soak passed.
- Credential rotation, security, monitoring, upgrade, and rollback drills passed.
- At least one PRD value gate passed.
- Public documentation matches implemented/certified status.
