# Hybrid fleet operations runbook

| Field | Value |
| --- | --- |
| Applies to | Portable `ax-serving-api` gateway and `ax-runtime-agent` |
| Architecture | Runtime-neutral REST/SSE data path, protocol-v1 control plane |
| Last updated | 2026-07-12 |

This runbook operates AX Engine/MLX and vLLM or SGLang/CUDA endpoints as one
fleet. It does not apply to the embedded macOS compatibility server except
where explicitly noted.

## 1. Ownership boundary

AX Serving owns public authentication, admission, logical-model resolution,
endpoint selection, safe retry, fleet leases, shared capacity reservations,
drain, desired deployment state, and diagnostics.

The runtime owns tokenization, chat templates, model loading, batching, KV
cache, token scheduling, distributed execution, and kernels. The runtime agent
normalizes discovery/readiness/metrics and proxies bytes; it must remain thin.

One request attempt runs on one endpoint. “Hybrid” means mixed endpoint pools,
not a graph or KV cache split between MLX and CUDA.

## 2. Deployment profiles

### Loopback development

- one gateway;
- `fleet_store: memory`;
- `deployment_mode: legacy_compat` or explicit test declarations;
- `AXS_TLS_PROFILE=loopback_dev`;
- `AXS_ALLOW_NO_AUTH=true` only by explicit operator choice.

### Remote production candidate

- two or more gateways with unique `AXS_GATEWAY_ID` values;
- durable Redis/Valkey shared state;
- `deployment_mode: explicit`;
- authenticated public, admin, worker-control, dispatch, and runtime hops;
- TLS at ingress and mTLS or an equivalent trusted private mesh internally;
- immutable runtime images and complete deployment identities;
- retained conformance, performance, failure, and soak evidence.

The source tree is not a production certification. Complete every PRD release
gate before applying that label to a deployment.

## 3. Start the gateway

Build only the portable target:

```bash
cargo build --locked --release -p ax-serving-cli --bin ax-serving-api
```

Development:

```bash
AXS_ALLOW_NO_AUTH=true target/release/ax-serving-api
```

Remote active-active example:

```bash
AXS_CONFIG=/etc/ax-serving/serving.yaml \
AXS_FLEET_STORE=redis \
AXS_REDIS_URL='rediss://user:password@redis.example:6379/0' \
AXS_GATEWAY_ID='gateway-a' \
AXS_TLS_PROFILE=trusted_mesh \
AXS_API_KEY='public-client-key' \
AXS_ADMIN_API_KEY='admin-key' \
AXS_INTERNAL_API_TOKEN='worker-control-key' \
AXS_DISPATCH_TOKEN='gateway-agent-key' \
target/release/ax-serving-api
```

Start another replica as `gateway-b` with the same configuration and fleet
key prefix. Never duplicate a gateway ID.

The default endpoint policy is `inference_aware`. It applies hard eligibility
filters first, then conservatively scores fresh runtime capacity, queue, KV,
batch, TTFT, error, and cache signals. Unknown and stale signals are penalized.
Compatibility policies remain available for rollback:

- `least_inflight`;
- `weighted_round_robin`;
- `model_affinity`;
- `token_cost`;
- `cache_affinity`.

## 4. Start runtime agents

Build:

```bash
cargo build --locked --release -p ax-thor-agent --bin ax-runtime-agent
```

AX Engine example:

Start the current AX Engine server as the runtime owner. Keep it on loopback
when the agent is colocated:

```bash
AX_ENGINE_API_KEY='runtime-only-key' \
/Users/akiralam/code/ax-engine/target/release/ax-engine-server \
  --host 127.0.0.1 --port 8000 \
  --model-id 'replace-with-runtime-model-id' \
  --mlx \
  --mlx-model-artifacts-dir '/replace/with/model-artifacts'
```

Then attach the portable agent:

```bash
AXS_CONTROL_PLANE_URL=https://ax-serving-control.example \
AXS_NODE_RUNTIME=ax_engine \
AXS_RUNTIME_VERSION='6.8.2' \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=0.0.0.0:18081 \
AXS_NODE_ADVERTISED_ADDR=10.20.1.10:18081 \
AXS_NODE_ID=mac-mlx-01 \
AXS_NODE_WORKER_POOL=mac-mlx \
AXS_NODE_HARDWARE_CLASS=mac \
AXS_TRUST_DOMAIN=private \
AXS_TLS_PROFILE=trusted_mesh \
AXS_WORKER_TOKEN='worker-control-key' \
AXS_DISPATCH_TOKEN='gateway-agent-key' \
AXS_RUNTIME_API_KEY='runtime-only-key' \
target/release/ax-runtime-agent
```

AX Engine 6.8.2 exposes generation and multimodal capability metadata from
`/v1/models`; the agent consumes that metadata directly. The current AX Engine
model card does not distinguish an embedding-only deployment, so set
`AXS_NODE_EMBEDDING=true` for a certified embedding deployment and
`AXS_NODE_EMBEDDING=false` when an explicit fail-closed override is required.
Do not infer embedding equivalence from a model name.

The current server exposes authoritative readiness at `/health`, inventory at
`/v1/models`, inference through OpenAI-compatible REST/SSE, and aggregate
telemetry at `/metrics`. AX Serving maps only the metrics whose semantics are
defined by AX Engine 6.8.2; unavailable TTFT or throughput observations remain
unknown and therefore cannot improve an endpoint's routing score.

vLLM example:

```bash
AXS_CONTROL_PLANE_URL=https://ax-serving-control.example \
AXS_NODE_RUNTIME=vllm \
AXS_RUNTIME_VERSION='replace-with-certified-version' \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=0.0.0.0:18081 \
AXS_NODE_ADVERTISED_ADDR=10.20.2.10:18081 \
AXS_NODE_ID=cuda-vllm-01 \
AXS_NODE_WORKER_POOL=cuda-vllm \
AXS_NODE_HARDWARE_CLASS=pc-cuda \
AXS_TRUST_DOMAIN=private \
AXS_TLS_PROFILE=trusted_mesh \
AXS_WORKER_TOKEN='worker-control-key' \
AXS_DISPATCH_TOKEN='gateway-agent-key' \
AXS_RUNTIME_API_KEY='runtime-only-key' \
target/release/ax-runtime-agent
```

For every deployment, set the observed identity fields used by its policy:

```text
AXS_MODEL_REVISION
AXS_MODEL_ARTIFACT_DIGEST
AXS_MODEL_TOKENIZER_DIGEST
AXS_MODEL_TEMPLATE_DIGEST
AXS_MODEL_QUANTIZATION
AXS_MODEL_MAX_OUTPUT_TOKENS
AXS_MODEL_CAPABILITIES
```

The agent becomes ready only when the upstream runtime is ready and inventory
has been observed. Agent process health alone is not routing readiness. A
runtime failure, stale observation, protocol mismatch, drain state, or expired
lease removes the endpoint from eligibility.

## 5. Configure deployment identity

Use [`../../config/serving.hybrid.example.yaml`](../../config/serving.hybrid.example.yaml)
as a schema example only. Replace all placeholder digests and certification
paths.

An explicit deployment declares:

- logical client model alias;
- runtime model ID;
- pool, runtime kind, hardware class, and trust domain;
- runtime/version, model revision or artifact digest;
- tokenizer and template digests;
- quantization, operations, context/output limits, modalities, and capabilities;
- required matching fields;
- optional equivalence class.

Cross-pool failover is permitted only when source and target are both listed
in the same equivalence policy and match every required identity field. A
shared model name is not equivalence. Different quantizers or formats must be
recorded and tested; do not call them exact when they are not.

## 6. Validate startup

Public probes:

```bash
curl -fsS http://gateway:18080/livez
curl -fsS http://gateway:18080/readyz       # control-plane ready (default; no workers required)
curl -fsS http://gateway:18080/routablez   # capacity: 200 only with eligible workers
curl -sS http://gateway:18080/health | jq .  # status "ok" means capacity, not process readiness
curl -sS http://gateway:18080/v1/models \
  -H 'Authorization: Bearer public-client-key' | jq .
```

Admin and control detail:

```bash
curl -sS http://gateway:18080/v1/admin/fleet \
  -H 'Authorization: Bearer admin-key' | jq .
curl -sS http://gateway:18080/admin/v1/deployments \
  -H 'Authorization: Bearer admin-key' | jq .
curl -sS http://gateway:19090/internal/workers \
  -H 'X-Internal-Token: worker-control-key' | jq .
```

Expected before traffic:

- `/livez` is `200` on every process;
- `/readyz` is `200` when the control plane is ready (config, listeners, fleet
  store, not draining). Default `readyz_mode=control_plane` does **not** require
  workers so agents can register during bootstrap;
- `/routablez` is `200` only when at least one eligible healthy non-draining
  endpoint is present (serving capacity). Use this for traffic readiness, not
  `/readyz` alone;
- `/health` `"status": "ok"` means capacity (`workers.eligible > 0`), not the
  same signal as control-plane `/readyz`;
- every worker has a current protocol lease and ready runtime observation;
- explicit deployment identity matches the catalog;
- no unexpected legacy registration participates in certified routing.

Legacy: set `orchestrator.readyz_mode = eligible_workers` or
`AXS_READYZ_MODE=eligible_workers` to restore worker-gated `/readyz` during
Fabric migration. Production installs keep the control-plane default and probe
capacity via `/routablez`.

## 7. Request and retry contract

The gateway creates one request ID and a unique attempt ID per dispatch. It
may perform at most two attempts.

A second attempt is allowed only when all conditions hold:

1. no response headers or body bytes were committed to the client;
2. the first attempt failed to connect, or the authenticated agent returned a
   typed `not-admitted` result;
3. another deployment is eligible and retry-compatible, including explicit
   equivalence for cross-pool routing;
4. the absolute request deadline has not expired.

AX Serving never retries an arbitrary runtime `5xx`, an ambiguous transport
failure after admission, or a stream after its first committed byte. A rising
retry counter therefore indicates connect failures or trusted pre-admission
capacity rejection—not generic runtime errors.

## 8. Admission and tenant controls

| Variable | Default | Meaning |
| --- | ---: | --- |
| `AXS_GLOBAL_QUEUE_MAX` | `128` | Active gateway requests |
| `AXS_GLOBAL_QUEUE_DEPTH` | `256` | Waiting requests |
| `AXS_GLOBAL_QUEUE_WAIT_MS` | `10000` | Maximum queue wait |
| `AXS_GLOBAL_QUEUE_POLICY` | `queue` | `queue`, `reject`, or `shed_oldest` |
| `AXS_TENANT_MAX_CONCURRENT` | `0` | Per-tenant active cap; zero disables |

Priority is `low`, `normal`, or `high` through `x-ax-priority`. Priority aging
and cross-client handoff prevent indefinite starvation. The gateway does not
perform token-level scheduling.

If cache affinity is needed, configure a 32-byte-or-longer random
`AXS_CACHE_AFFINITY_SECRET` and send an opaque `x-ax-cache-affinity` value.
The derived key is tenant-scoped; raw hints and prompt text are neither stored
nor forwarded.

## 9. Drain a worker

Preferred agent-controlled shutdown:

1. stop renewing ready heartbeats;
2. send drain;
3. reject new dispatch with typed `not-admitted`;
4. wait for inflight streams to reach zero or the hard timeout;
5. send drain-complete and terminate.

Manual emergency drain:

```bash
curl -sS -X POST http://gateway:18080/v1/workers/WORKER_ID/drain \
  -H 'Authorization: Bearer admin-key'
```

Watch `inflight` and drain state. Do not delete a worker merely to accelerate a
normal drain; force removal can terminate active work and is an incident action.

## 10. Roll a deployment

In explicit mode, create or update the replacement identity first. Then submit
a roll action naming the replacement deployment. The lifecycle state machine:

1. enables the replacement desired state;
2. waits until at least one replacement endpoint is ready;
3. disables the source deployment;
4. rolls desired state back if readiness does not arrive before the timeout.

All mutations return a job. Poll it until `succeeded` or `failed`:

```bash
curl -sS http://gateway:18080/admin/v1/jobs/JOB_ID \
  -H 'Authorization: Bearer admin-key' | jq .
```

These jobs coordinate AX Serving desired state. The external orchestrator or
runtime adapter still owns image rollout, model download, GPU allocation, and
process lifecycle.

## 11. Gateway upgrade and rollback

Preconditions:

- CI, protocol fixtures, Redis conformance, and dependency-boundary checks pass;
- protocol major version is compatible with active agents;
- database/key-prefix migration is backward compatible;
- the prior immutable image and configuration are retained.

Procedure:

1. add one new gateway replica with a unique ID;
2. verify `/livez`, `/readyz`, shared workers, deployments, reservations, and
   metrics;
3. send canary traffic and compare error/retry/latency signals;
4. replace remaining replicas one at a time with `maxUnavailable=0`;
5. retain the previous replicas until streams drain.

Rollback by restoring the previous immutable image while keeping the same
compatible shared-state key prefix. If the new version wrote an incompatible
protocol or state schema, stop and follow the release-specific migration plan;
never clear Redis as a routine rollback.

## 12. Credential rotation

Rotate one trust boundary at a time. Use overlap only where comma-separated
key sets are supported.

1. add the new public/admin key, deploy gateways, update clients, remove old;
2. add the new worker-control key, deploy gateways and agents in a coordinated
   window, then remove old;
3. rotate dispatch credentials by updating gateways and agents together;
4. rotate runtime credentials only on agents and runtimes;
5. rotate Redis credentials through a new URL, verify replica reconciliation,
   then revoke old access;
6. rotate affinity secret only when losing old cache locality is acceptable.

Never place secrets in YAML, logs, command output retained by CI, images, or
Prometheus labels.

## 13. Observability and alerts

Scrape `GET /metrics` with the admin bearer credential; use authenticated
`GET /v1/metrics` for JSON diagnostics. Required baseline signals include:

- `axs_gateway_requests_total`;
- `axs_gateway_dispatch_attempts_total`;
- completed, failed, cancelled, and retry counters;
- admitted/rejected counters and queue state;
- healthy, unhealthy, draining, and eligible worker gauges;
- aggregate worker inflight;
- endpoint-selection and upstream-response-header latency histograms;
- bounded endpoint-selection outcome counters.

Worker IDs, request IDs, prompts, model paths, and credentials are not metric
labels. Use bounded audit/log records with request IDs for individual cases.

Recommended alerts:

- no eligible worker for a published logical model;
- readiness failures on enough gateways to violate redundancy;
- heartbeat/observation age approaching TTL;
- sustained queue depth, rejection, or tenant quota pressure;
- safe retries above the established baseline;
- failed or timed-out lifecycle jobs;
- Redis connectivity or latency failure;
- gateway RSS growth during the validated soak window.

Alert thresholds must come from retained workload evidence, not generic
defaults.

For distributed traces, configure `OTEL_EXPORTER_OTLP_ENDPOINT` or
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` on each gateway and runtime agent. The
portable exporter supports `http/json`; optional collector credentials use
`OTEL_EXPORTER_OTLP_HEADERS` and are redacted from diagnostics. W3C
`traceparent`/`tracestate` continuity remains enabled when export is disabled.
Built-in spans exclude prompt/output content, tools, images, paths, and
credentials.

## 14. Incident response

### `/readyz` is `503`

Under the default `control_plane` mode this means the **control plane** is not
ready (config/listeners starting, gateway draining, or fleet store
stale/unavailable)—not missing workers. Check, in order:

1. process still starting (listeners not marked ready) or active drain/shutdown;
2. Redis/fleet-store connectivity and last successful store operation age;
3. configuration validation failures at startup.

If you intentionally run `readyz_mode=eligible_workers`, then also walk the
capacity checklist under `/routablez` is `503` below.

### `/routablez` is `503` (or `/health` status is `degraded`)

Serving capacity is unavailable. Check, in order:

1. runtime process readiness and `/v1/models`;
2. agent heartbeat and observation timestamp;
3. protocol major compatibility and required capabilities;
4. worker drain/lease state;
5. deployment identity/equivalence mismatch;
6. capacity reservations and tenant/global admission;
7. Redis connectivity and clock-independent TTL behavior.

Do not bypass identity or readiness filters to restore traffic. Pin a known
safe deployment or roll back the runtime artifact.

### Retry counter rises

Inspect gateway route diagnostics and agent admission state. Valid causes are
connection failures and typed pre-admission rejection. Runtime `5xx` should be
returned on the original attempt and must not increment safe retries. If they
do, treat it as a correctness regression.

### Redis/Valkey is unavailable

Existing streams may continue locally. New explicit admissions fail closed
when shared reservation or fleet state cannot be proven. Restore the durable
store; do not switch live active-active replicas to unrelated in-memory state.

### Runtime returns errors after admission

Do not reroute. Drain or disable the deployment, inspect runtime logs using the
request/attempt IDs, and roll back the runtime/model artifact. A duplicate
attempt could create double cost or divergent output.

### Queue overload

Determine whether pressure is gateway admission, worker capacity, or runtime
queue/KV saturation. Add compatible endpoints or reduce admission only after
checking runtime goodput. Increasing gateway concurrency can worsen runtime
tail latency.

## 15. Shutdown

Remove a gateway from the load balancer before termination. It stops new
admission, allows accepted streams to complete, and exits at its hard deadline.
Worker agents stop ready heartbeats, drain, and report drain-complete.

After planned maintenance verify:

- no leaked local or shared reservations;
- no stale worker lease remains eligible;
- deployment desired and observed state agree;
- retry/cancellation counters match the event;
- public credentials were not observed at runtimes.

## 16. Compatibility transports

Direct HTTP/SSE is the production data-path target. NATS remains a
compatibility transport and is configured with `max_deliver=1`; ambiguous
token-stream redelivery is not safe retry. Do not use durable broker replay as
an inference failover mechanism.

The embedded gRPC v1 API is macOS compatibility-only. It carries local model
paths, backend enums, and token-ID semantics that cannot be translated
losslessly by the portable hybrid gateway.
