# Multi-Worker Orchestration Runbook

**Version:** 1.2
**Contract:** [AX Serving Node Contract](../contracts/ax-serving-node-contract.md)
**Date:** 2026-03-03

---

## 1. Starting a Runtime Node Pool (direct mode)

AX Serving is the API gateway and control plane. Runtime nodes own inference
execution and register themselves with the gateway on startup.

Target runtime nodes:

- Mac nodes running `ax-engine`
- PC CUDA nodes running `vLLM`
- NVIDIA Thor nodes running `vLLM`

The `ax-serving serve` local worker path is still available for Mac
compatibility, but it should be treated as a migration bridge while dedicated
ax-engine node adapters mature.

### Prerequisites

- Binaries built: `cargo build -p ax-serving-cli --release`
  This produces both `ax-serving` and `ax-serving-api` in `target/release/`.
- Runtime adapter built when using external runtimes:
  `cargo build -p ax-thor-agent --release`
  This produces `ax-runtime-agent` and the legacy `ax-thor-agent` alias in
  `target/release/`.
- Orchestrator running (see §2)
- Runtime node available:
  - `ax-serving serve` compatibility worker for local Mac testing
  - ax-engine node adapter for Mac runtime-node deployments
  - vLLM node/agent for PC CUDA or NVIDIA Thor runtime-node deployments

### Start compatibility Mac workers

Open multiple terminals (or use a process supervisor):

```bash
# Worker 1 — port 18081
AXS_ORCHESTRATOR_ADDR=http://127.0.0.1:19090 \
AXS_WORKER_RUNTIME=ax_engine \
AXS_WORKER_HARDWARE_CLASS=mac \
ax-serving serve \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --model-id llama3-8b \
  --port 18081

# Worker 2 — port 8082
AXS_ORCHESTRATOR_ADDR=http://127.0.0.1:19090 \
AXS_WORKER_RUNTIME=ax_engine \
AXS_WORKER_HARDWARE_CLASS=mac \
ax-serving serve \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --model-id llama3-8b \
  --port 8082

# Workers 3 and 4: repeat on ports 8083, 8084
```

Each worker auto-registers with the orchestrator through
`AXS_ORCHESTRATOR_ADDR` or the equivalent `--orchestrator` CLI flag.

Production gateway-only deployments can block embedded compatibility workers:

```bash
AXS_EMBEDDED_RUNTIME_POLICY=deny ax-serving serve \
  -m ./models/llama3.gguf \
  --model-id llama3-8b \
  --port 18081
# Expected: startup fails and asks for ax-serving-api plus ax-engine/vLLM runtime nodes.
```

Use `AXS_EMBEDDED_RUNTIME_POLICY=warn` for migration and local testing, or
`allow` only when the compatibility path is intentionally accepted.

### Start a Mac ax-engine runtime node

Run ax-engine with an OpenAI-compatible endpoint, then place `ax-runtime-agent`
in front of it so AX Serving only owns serving policy and routing:

```bash
AXS_CONTROL_PLANE_URL=http://127.0.0.1:19090 \
AXS_NODE_RUNTIME=ax_engine \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=0.0.0.0:18081 \
AXS_NODE_ADVERTISED_ADDR=127.0.0.1:18081 \
AXS_NODE_HARDWARE_CLASS=mac \
AXS_NODE_CLASS=mac-studio \
AXS_NODE_WORKER_POOL=mac \
AXS_NODE_MAX_INFLIGHT=8 \
target/release/ax-runtime-agent
```

The agent reads `/v1/models` from the runtime endpoint, registers the adapter
address with AX Serving, sends heartbeats, and proxies OpenAI-compatible
inference requests to ax-engine. Adapter registrations include
`runtime_mode=adapter`; the legacy `ax-serving serve` worker path registers as
`runtime_mode=embedded` so diagnostics can identify compatibility workers
without guessing from backend names.

When `/v1/models` exposes runtime metadata, the agent also reports structured
`model_inventory` entries to the gateway. Supported metadata includes
`max_context`, `quantization`, `artifact_format`, `modalities`, and
`supported_operations`. The gateway keeps id-only inventory for runtimes that
only expose OpenAI-compatible model ids.

If the runtime exposes `/metrics`, the agent translates common Prometheus
gauges such as `ax_runtime_active_sequences`, `ax_runtime_queue_depth`,
`ax_runtime_decode_tok_per_sec`, `ax_runtime_ttft_p95_ms`, and
`ax_runtime_kv_pages_used` into AX Serving heartbeat telemetry. For ax-engine
or AX Serving-compatible runtimes, it also accepts `axs_*` gauges such as
`axs_scheduler_queue_depth`, `axs_scheduler_inflight_count`, and
`axs_ttft_p95_us`. If Prometheus metrics are unavailable, the agent falls back
to `/v1/metrics` JSON and reads fields such as `scheduler.queue_depth`,
`scheduler.inflight_count`, `scheduler.ttft_p95_us`, `kv_cache.utilization`,
and `batch.utilization`. Missing metrics are safe; the node remains routable
with fallback heartbeat values.

### Start a PC CUDA vLLM runtime node

Run vLLM's OpenAI-compatible server on the CUDA node, then start the adapter:

```bash
AXS_CONTROL_PLANE_URL=http://<gateway-host>:19090 \
AXS_NODE_RUNTIME=vllm \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=0.0.0.0:18081 \
AXS_NODE_ADVERTISED_ADDR=<cuda-node-ip>:18081 \
AXS_NODE_HARDWARE_CLASS=pc-cuda \
AXS_NODE_CLASS=pc-cuda \
AXS_NODE_WORKER_POOL=cuda \
AXS_NODE_MAX_INFLIGHT=16 \
target/release/ax-runtime-agent
```

AX Serving sees this as the same node contract as a Mac node, with a different
runtime and hardware class.

### Start a NVIDIA Thor vLLM runtime node

Thor deployments should also use vLLM as the runtime owner. The legacy
`ax-thor-agent` binary and `AXS_THOR_*` variables remain available, but the
generic runtime-node form is preferred for new deployments:

```bash
AXS_CONTROL_PLANE_URL=http://<gateway-host>:19090 \
AXS_NODE_RUNTIME=vllm \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=0.0.0.0:18081 \
AXS_NODE_ADVERTISED_ADDR=<thor-node-ip>:18081 \
AXS_NODE_HARDWARE_CLASS=thor \
AXS_NODE_CLASS=thor \
AXS_NODE_WORKER_POOL=thor \
AXS_NODE_MAX_INFLIGHT=16 \
target/release/ax-runtime-agent
```

### Verify workers are registered

```bash
curl -s http://127.0.0.1:19090/internal/workers | jq '.workers | length'
# Expected: 4

curl -s http://127.0.0.1:19090/internal/workers | jq '.workers[] | {id, runtime, hardware_class, health}'

curl -s http://127.0.0.1:18080/health | jq '.workers'
# Expected: { "total": 4, "healthy": 4, "unhealthy": 0, "dead": 0 }
```

---

## 2. Starting the API Gateway (`ax-serving-api`)

The API gateway is a pure dispatch process — it holds no model weights and
starts no Metal context.

```bash
# Defaults: direct mode, least_inflight policy, port 18080, internal 19090
ax-serving-api
```

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AXS_ORCHESTRATOR_HOST` | `127.0.0.1` | Public proxy bind host. Set to `0.0.0.0` to expose externally. |
| `AXS_WORKER_MODE` | `direct` | **Default.** Loopback HTTP proxy to workers. |
| `AXS_ORCHESTRATOR_PORT` | `18080` | Public proxy port (clients send requests here). |
| `AXS_INTERNAL_PORT` | `19090` | Internal API port, loopback only. |
| `AXS_INTERNAL_API_TOKEN` | unset | Optional shared token required for `/internal/*` calls and worker registration. |
| `AXS_DISPATCH_POLICY` | `least_inflight` | Worker selection algorithm. |
| `AXS_WORKER_HEARTBEAT_MS` | `5000` | Heartbeat interval hint sent to workers (ms). |
| `AXS_WORKER_TTL_MS` | `15000` | Age after which a silent worker is evicted (ms). |
| `AXS_GLOBAL_QUEUE_MAX` | `128` | Max concurrent requests before overload policy triggers. |
| `AXS_GLOBAL_QUEUE_WAIT_MS` | `10000` | Max queue wait before returning 503 (ms). |

### Dispatch policies

```bash
# Least-inflight (default) — routes to the least-loaded worker
AXS_DISPATCH_POLICY=least_inflight ax-serving-api

# Weighted round-robin — proportional to available capacity
AXS_DISPATCH_POLICY=weighted_round_robin ax-serving-api

# Model affinity — prefers workers that have previously served the model
AXS_DISPATCH_POLICY=model_affinity ax-serving-api
```

### Override ports via flags

```bash
ax-serving-api --port 9000 --internal-port 9001 --policy weighted_round_robin
```

---

## 3. Adding a Worker to a Running Pool

Runtime nodes can join a running gateway at any time.

```bash
# Start new worker on port 8085
AXS_ORCHESTRATOR_ADDR=http://127.0.0.1:19090 \
AXS_WORKER_RUNTIME=ax_engine \
AXS_WORKER_HARDWARE_CLASS=mac \
ax-serving serve -m ./models/llama3.gguf --model-id llama3-8b --port 8085

# Verify it appeared within one heartbeat interval (~5 s):
curl -s http://127.0.0.1:19090/internal/workers | jq '[.workers[] | select(.health == "healthy")] | length'
```

If a worker with the same address re-registers (e.g. after a crash), the
registry treats it as idempotent and resets its heartbeat timer.

**Orchestrator restart recovery:** if `ax-serving-api` is restarted and loses
its worker registry, each running worker detects the 404 on its next heartbeat
and automatically re-registers — no worker restart required. Workers are back
in the eligible pool within one heartbeat interval (~5 s by default).

---

## 4. Draining a Worker for Restart

Use the drain flow to gracefully remove a worker without dropping in-flight
requests.

```bash
WORKER_ID="<uuid from /internal/workers>"

# 1. Signal drain — gateway stops routing new requests to this worker.
curl -s -X POST http://127.0.0.1:19090/internal/workers/${WORKER_ID}/drain

# 2. Monitor: wait until inflight reaches 0.
watch -n 1 "curl -s http://127.0.0.1:19090/internal/workers/${WORKER_ID} | jq '.inflight'"

# 3. Once inflight == 0, call drain-complete to evict.
curl -s -X POST http://127.0.0.1:19090/internal/workers/${WORKER_ID}/drain-complete
# Returns 204 on success.

# 4. Restart the worker (it will re-register automatically).
ax-serving serve -m ./models/llama3.gguf --model-id llama3-8b --port 18081
```

> **Note:** If the worker crashes before calling drain-complete, the
> gateway's health ticker evicts it after `AXS_WORKER_TTL_MS` (default 15 s).

---

## 5. Switching to NATS Mode

NATS mode requires the `nats-dispatch` Cargo feature and a running NATS server
with JetStream enabled.

### Prerequisites

```bash
# Install nats-server (brew or GitHub releases)
brew install nats-server

# Start with JetStream enabled
nats-server --jetstream
```

### Build with NATS support

```bash
cargo build -p ax-serving-api --features nats-dispatch --release
cargo build -p ax-serving-cli --release
```

### Configuration

```bash
# Worker side — run the NatsWorker sidecar alongside ax-serving serve
AXS_NATS_URL=nats://127.0.0.1:4222 \
AXS_NATS_STREAM=ax-serving \
AXS_NATS_MAX_DELIVER=3 \
<your worker process>

# Gateway side — set mode to nats in the requesting code path
# (NatsDispatcher is used programmatically; ax-serving-api remains in direct mode)
AXS_NATS_URL=nats://127.0.0.1:4222
AXS_GLOBAL_QUEUE_WAIT_MS=10000
```

### Verify NATS connectivity

```bash
nats stream ls   # should show ax-serving stream after first request
nats consumer ls ax-serving
```

---

## 6. Monitoring

### Health and metrics endpoints

```bash
# Gateway health (workers + queue summary)
curl -s http://127.0.0.1:18080/health | jq .

# Authenticated admin status (queue, dispatch, license, worker summary)
curl -s http://127.0.0.1:18080/v1/admin/status \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq .

curl -s http://127.0.0.1:18080/v1/admin/status \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq '.workers.runtimes'

# Runtime diagnostics with model inventory, endpoint, operations, and issues.
curl -s http://127.0.0.1:18080/v1/admin/diagnostics \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq '.runtime_diagnostics.runtimes'

# Actionable recovery hints derived from runtime diagnostics.
curl -s http://127.0.0.1:18080/v1/admin/diagnostics \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq '.runtime_diagnostics.recommended_actions'

# CLI status with gateway diagnostics and recovery actions.
ax-serving status --url http://127.0.0.1:18080 --diagnostics --api-key "${AXS_API_KEY}"

# AX Fabric read-side contract validation.
ax-serving fabric validate --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"

# Check whether production can switch to AXS_EMBEDDED_RUNTIME_POLICY=deny.
ax-serving migration embedded-readiness --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"

# Redacted support bundle for escalation.
ax-serving support-bundle --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}" --output support-bundle.json

# CLI worker lifecycle commands for recovery workflows.
ax-serving workers list --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"
ax-serving workers drain "${WORKER_ID}" --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"
ax-serving workers drain "${WORKER_ID}" --complete-when-idle --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"

# Detailed metrics including per-worker inflight and reroute count
curl -s http://127.0.0.1:18080/v1/metrics | jq .

# Public worker inventory for dashboards and operator tooling
curl -s http://127.0.0.1:18080/v1/workers \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq .

# Internal worker list (loopback only)
curl -s http://127.0.0.1:19090/internal/workers | jq '.workers[] | {id, runtime, hardware_class, health, inflight, addr}'

# Runtime-level fleet summary
curl -s http://127.0.0.1:18080/v1/admin/fleet \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq '.runtimes'

# Routable model list — only healthy, non-draining workers contribute
# If a model is missing here, the serving worker is unhealthy or draining.
curl -s http://127.0.0.1:18080/v1/models | jq '.data[].id'
```

### Key metrics to watch

| Metric | Field | Alert threshold |
|--------|-------|-----------------|
| Healthy workers | `workers.healthy` | < 1 → degraded |
| Queue depth | `queue.queued` | > 80% of `AXS_GLOBAL_QUEUE_MAX` |
| Rejected requests | `queue.rejected_total` | Rising rate → increase queue or add workers |
| Reroute total | `reroute_total` | Rising rate → workers returning 5xx |
| Worker inflight | per-worker `inflight` | Near `max_inflight` → add capacity |

### Runtime diagnostic actions

`/v1/admin/diagnostics` emits machine-readable recommended actions:

| Action | Operator response |
|---|---|
| `restore_runtime_capacity` | Start or recover at least one healthy non-draining runtime node. |
| `replace_unhealthy_workers` | Drain or remove unhealthy workers, restart the runtime node, then verify heartbeat. |
| `complete_drain_when_idle` | Run `ax-serving workers drain <id> --complete-when-idle`, or wait for inflight to reach zero and call drain-complete. |
| `fix_runtime_endpoint_registration` | Restart the adapter with `AXS_NODE_RUNTIME_URL` or `AXS_WORKER_RUNTIME_ENDPOINT`. |
| `refresh_model_inventory` | Check runtime `/v1/models`, load the model, then restart or re-register the adapter. |
| `fix_runtime_class` | Register with `runtime=ax_engine` or `runtime=vllm`. |
| `fix_hardware_class` | Register ax-engine nodes as `hardware_class=mac`; register vLLM nodes as `pc-cuda` or `thor`. |
| `investigate_runtime_errors` | Check runtime logs and recent failed requests before returning affected workers to normal routing. |
| `relieve_runtime_pressure` | Reduce admission pressure, add runtime capacity, or drain and replace overloaded nodes. |
| `migrate_embedded_compatibility_path` | Move inference to `ax-runtime-agent` plus ax-engine/vLLM and use `AXS_EMBEDDED_RUNTIME_POLICY=deny` in production. |

Actions that can be advanced through AX Serving CLI include
`suggested_commands`. `ax-serving status --diagnostics` prints those commands
under each action so operators can move from diagnosis to drain, inspection, or
replacement verification without manually translating worker IDs.

Runtime-specific expectations are included under
`.runtime_diagnostics.runtimes.<runtime>.runtime_guidance`. Use that block to
verify that ax-engine nodes are registered as Mac nodes and vLLM nodes are
registered as PC CUDA or Thor nodes.

`ax-serving fabric validate` checks that AX Fabric can rely on `/health`,
`/v1/models`, and `/v1/metrics`. The command accepts both single-runtime and
gateway metrics profiles, and exits non-zero when a documented contract field is
missing.

`ax-serving migration embedded-readiness` reads gateway diagnostics and reports
whether the fleet is ready for `AXS_EMBEDDED_RUNTIME_POLICY=deny`. It requires
registered adapter workers, no `runtime_mode=embedded` workers, no unknown
runtime-mode workers, and at least one eligible healthy non-draining worker.

Telemetry-based issues such as `high_runtime_error_rate`,
`runtime_queue_backlog`, `high_runtime_kv_pressure`, and
`high_runtime_batch_pressure` come from worker heartbeat fields. Adapters can
report either counter fields such as `kv_pages_used` and `active_batch_size`, or
ratio fields such as `kv_utilization` and `batch_utilization`. Missing telemetry
is treated as unknown rather than failed so older adapters can remain compatible.
The runtime agent can derive these fields from common `ax_runtime_*`, `axs_*`,
vLLM, and `/v1/metrics` JSON aliases.

### Support escalation bundle

Use `ax-serving support-bundle` when a runtime issue needs handoff to the
platform or support team. The bundle collects gateway health, model inventory,
metrics, admin status, diagnostics, fleet, worker inventory, and recent audit
events. Sensitive keys such as API keys, tokens, secrets, authorization
headers, passwords, and license keys are recursively redacted before output.

```bash
ax-serving support-bundle \
  --url http://127.0.0.1:18080 \
  --api-key "${AXS_API_KEY}" \
  --output support-bundle.json

ax-serving support-bundle \
  --url http://127.0.0.1:18080 \
  --api-key "${AXS_API_KEY}" \
  --json | jq '.endpoints[] | {name, ok, status_code}'
```

### Log fields

```
# Enable debug logging
AXS_LOG=debug ax-serving-api

# Key structured log fields:
#   worker_id, model_id, inflight, policy, reroute, request_id
```

### Public admin worker lifecycle

Use the authenticated public API when operations tooling or browser dashboards
cannot reach the loopback-only internal router.

Equivalent CLI commands are available for operator workflows:

```bash
ax-serving workers list --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"
ax-serving workers get "${WORKER_ID}" --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"
ax-serving workers drain "${WORKER_ID}" --complete-when-idle --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"
ax-serving workers drain-complete "${WORKER_ID}" --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"
ax-serving workers remove "${WORKER_ID}" --url http://127.0.0.1:18080 --api-key "${AXS_API_KEY}"
```

```bash
WORKER_ID="<uuid from /v1/workers>"

# Start graceful drain
curl -s -X POST http://127.0.0.1:18080/v1/workers/${WORKER_ID}/drain \
  -H "Authorization: Bearer ${AXS_API_KEY}"

# Inspect a single worker snapshot
curl -s http://127.0.0.1:18080/v1/workers/${WORKER_ID} \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq .

# Complete drain and remove worker from registry
curl -i -X POST http://127.0.0.1:18080/v1/workers/${WORKER_ID}/drain-complete \
  -H "Authorization: Bearer ${AXS_API_KEY}"
```

---

## 7. Troubleshooting

### Worker not appearing in eligible set or model list

`GET /v1/models` on the gateway and the dispatch-eligible set use the same
filter: **healthy + not draining**. If a model disappears from the model list,
the serving worker has gone unhealthy.

1. Check the worker registered: `GET /internal/workers` — is the worker present?
2. Check health state: `health` field must be `"healthy"`. Unhealthy and dead workers are excluded from both dispatch and the model list.
3. Check capabilities: the worker's `capabilities` must include the requested `model_id`.
4. Check heartbeat: if `last_heartbeat_age_ms > AXS_WORKER_TTL_MS`, the worker will be evicted.
5. Check drain flag: a draining worker (`drain: true`) is excluded from the eligible set.

```bash
curl -s http://127.0.0.1:19090/internal/workers | jq '
  .workers[] | {id, health, drain, capabilities, last_heartbeat_age_ms: .heartbeat_age_ms}
'
```

### Orchestrator restarted — workers not routing

After an orchestrator restart, workers auto re-register when their next
heartbeat returns 404. This takes at most one heartbeat interval
(`AXS_WORKER_HEARTBEAT_MS`, default 5 s). No worker restart is needed.

To verify recovery:

```bash
# Wait ~5 s, then check:
curl -s http://127.0.0.1:19090/internal/workers | jq '.workers | length'
curl -s http://127.0.0.1:18080/v1/models | jq '.data[].id'
```

If workers are not recovering, check that `AXS_ORCHESTRATOR_ADDR` or
`--orchestrator` on each worker points to the correct internal port of the
restarted gateway.

### Queue overflow (429 or 503 responses)

- `429 Too Many Requests` — `max_concurrent` limit reached with `Reject` policy.
- `503 Service Unavailable` + "request shed" — `ShedOldest` policy dropped an older request.
- `503 Service Unavailable` + "timed out" — request waited > `AXS_GLOBAL_QUEUE_WAIT_MS`.

**Remedies:**
- Add more workers to absorb load.
- Increase `AXS_GLOBAL_QUEUE_MAX` (watch RSS — queue holds in-memory byte buffers).
- Reduce client concurrency upstream.

### NATS consumer lag

```bash
# Check consumer lag (pending messages)
nats consumer info ax-serving worker-<uuid>-llama3-8b

# If NumPending > 0 and workers are healthy, check:
#   1. Worker pull batch size (default 8 messages per fetch)
#   2. Worker inflight vs max_inflight
#   3. Inference backend throughput
```

### High reroute rate

```bash
curl -s http://127.0.0.1:18080/v1/metrics | jq .reroute_total
```

A rising `reroute_total` means workers are returning 5xx. Check worker logs for
inference errors, memory pressure, or Metal context failures.

```bash
# Check per-worker health in detail
curl -s http://127.0.0.1:19090/internal/workers | jq '.workers[] | select(.health != "healthy")'
```
