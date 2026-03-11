# Multi-Worker Orchestration Runbook

**Version:** 1.2
**ADR:** [ADR-012](../adr/ADR-012-local-multi-worker-orchestration.md)
**Date:** 2026-03-03

---

## 1. Starting a Worker Pool (direct mode, 4 workers)

Each worker is an `ax-serving serve` process.  Workers register themselves with
the orchestrator on startup and send periodic heartbeats.

### Prerequisites

- Binaries built: `cargo build -p ax-serving-cli --release`
  This produces both `ax-serving` and `ax-serving-api` in `target/release/`.
- Orchestrator running (see §2)
- GGUF model file available

### Start four workers

Open four terminals (or use a process supervisor):

```bash
# Worker 1 — port 18081
AXS_INTERNAL_PORT=19090 \
ax-serving serve \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --model-id llama3-8b \
  --port 18081

# Worker 2 — port 8082
ax-serving serve \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --model-id llama3-8b \
  --port 8082

# Workers 3 and 4: repeat on ports 8083, 8084
```

Each worker auto-registers with the orchestrator at `http://127.0.0.1:19090`.

### Verify workers are registered

```bash
curl -s http://127.0.0.1:19090/internal/workers | jq '.workers | length'
# Expected: 4

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

Workers can join a running gateway at any time.

```bash
# Start new worker on port 8085
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

# Detailed metrics including per-worker inflight and reroute count
curl -s http://127.0.0.1:18080/v1/metrics | jq .

# Public worker inventory for dashboards and operator tooling
curl -s http://127.0.0.1:18080/v1/workers \
  -H "Authorization: Bearer ${AXS_API_KEY}" | jq .

# Internal worker list (loopback only)
curl -s http://127.0.0.1:19090/internal/workers | jq '.workers[] | {id, health, inflight, addr}'

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
