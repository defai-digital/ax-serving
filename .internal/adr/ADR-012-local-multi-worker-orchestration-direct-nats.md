# ADR-012: Local Multi-Worker Orchestration — Direct + Optional NATS Dispatch

**Status**: Proposed
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)
**Relates to**: ADR-010 (backend fallback chain), ADR-008 (thermal scheduling), ADR-009 (configuration)

---

## Context

### What We Have

`ax-serving` Phase 1 delivers:
- `InferenceBackend` trait abstraction over mistralrs / llama.cpp (ADR-010)
- A bounded admission scheduler (`ServingLayer::scheduler`) with configurable `max_inflight` and `max_queue`
- Thermal-adaptive concurrency (ADR-008)
- Model registry with up to 16 loaded models

All of this lives in a single process. The scheduler is effective for single-user concurrency control, but it was not designed for coordinating multiple independent worker processes.

### Why Multiple Workers Are Needed

**Metal serialisation**: On Apple Silicon, all Metal compute on a given device goes through the system Metal command queue. Multiple inference pipelines within a single process time-share that queue without true parallelism. Running separate processes each with their own Metal context allows genuine pipeline overlap — relevant when mixing small models (e.g., 4× 4B concurrently on an M3 Max with 48 GB UMA).

**Crash isolation**: A panic in the mistralrs pipeline, a stalled llama-server subprocess, or a Metal memory fault can block or terminate the process. Workers in separate processes fail independently; the orchestrator can reroute around a dead worker while others continue serving.

**Rolling restarts**: Serving deployments need zero-downtime model updates. With a single process, there is no way to drain one worker while others serve. A multi-worker model with drain semantics enables graceful replacement.

### What Is Missing

There is currently no:
- Worker identity or registration protocol
- Heartbeat and health state tracking at the worker level
- Cross-worker dispatch with a policy interface
- Shared bounded queue that multiple workers drain
- Unified observability across workers

Any of these added ad-hoc without a coherent design leads to fragile operational tooling. The decision here defines the architecture for all of them.

---

## Decision

Adopt a **two-mode local orchestration design** with a single orchestrator process and N worker processes:

### Mode 1: Direct (default)

- The orchestrator selects a worker using a dispatch policy and proxies the request directly over loopback HTTP (Unix socket or TCP `127.0.0.1:port`).
- No external dependencies. Works out of the box.
- Lowest overhead: one extra loopback round-trip to the selected worker.
- Workers register themselves with the orchestrator at startup via `POST /internal/workers/register` and send periodic heartbeats.

### Mode 2: NATS (optional)

- The orchestrator publishes accepted requests to a NATS JetStream subject (`axs.requests.<model_id>`).
- Workers subscribe and pull from their JetStream consumer.
- Workers ack on success, nack on failure (triggers JetStream redelivery, capped by `max_deliver`).
- Enabled by setting `AXS_WORKER_MODE=nats`. Requires `nats-server` with JetStream enabled.

### Shared Across Both Modes

- `WorkerRegistry`: worker identity, health state machine (`healthy` → `unhealthy` → `dead`), TTL tracking, drain mode.
- `DispatchPolicy` trait: pluggable selection algorithm (`least_inflight` | `weighted_round_robin` | `model_affinity`). Applied before dispatch in both modes.
- `GlobalQueue`: bounded MPSC queue in the orchestrator with `reject` / `shed_oldest` overload policies.
- Observability: identical `/v1/metrics` schema and `/health` response regardless of mode.
- Internal REST API: `/internal/workers/register`, `/internal/workers/{id}/heartbeat`, `/internal/workers/{id}/drain`.

---

## Rationale

### Why Two Modes Instead of One

**Direct mode must be zero-dependency.** The overwhelming majority of ax-serving deployments are single-Mac developer or lab environments. Requiring every user to operate a NATS broker to serve local inference adds real friction — installation, process management, TLS configuration, monitoring — for no benefit when all workers are on the same host and the workload fits in memory.

**NATS mode is the right upgrade path, not the baseline.** Teams with structured deployment tooling (systemd, Kubernetes on Mac, Nomad) benefit from the operational properties of a durable queue: broker-side backpressure, per-message ack/nack, independent worker restart without dropping in-flight work. These teams already operate brokers. The cost of adding `async-nats` as an optional dependency is low; the cost of blocking all users on that dependency is high.

**Shared interfaces prevent drift.** Both modes implement `Dispatcher: DispatchRequest → DispatchResult` and use the same `WorkerRegistry` and `DispatchPolicy`. Feature parity is enforced by the shared bench suite running against both modes in CI.

### Why `direct` Is the Default

- Zero external dependencies: works in any environment where ax-serving can be installed.
- Latency: on loopback, one extra HTTP round-trip adds < 1 ms median. NATS mode adds a broker hop (typically 0.5–2 ms) for every request.
- Operational symmetry: the orchestrator and workers already communicate via the internal REST API (heartbeat, register, drain). Dispatch over the same channel is a natural extension.

---

## Alternatives Considered

### A: Keep the Current Single-Process Single-Dispatcher Model

**Decision: Rejected.**

The current scheduler handles one process's concurrency. It provides no mechanism to:
- Route around a crashed worker
- Coordinate dispatch across workers with different loaded models
- Implement drain for rolling restarts

Extending the current design to multi-process requires essentially the same orchestration layer described in this ADR. There is no lighter path to multi-worker that is also operationally sound.

---

### B: NATS-Only Architecture (Remove Direct Mode)

**Decision: Rejected.**

A NATS-only design would require every deployment — including single-Mac developer setups and CI environments — to run a NATS server. The concrete problems:

1. **Installation friction**: `brew install nats-server` is a one-liner, but it is still a new dependency that must be explained, documented, and tested in every environment. Many users don't know what NATS is.

2. **Failure domain expansion**: A NATS broker failure now takes down the entire dispatch path. With direct dispatch, the orchestrator–worker channel is a TCP connection on loopback with no intermediate process.

3. **Latency floor**: NATS adds at minimum one broker hop per request. For short-generation requests (< 50 tokens), this is material relative to time-to-first-token. Measured on Apple Silicon M3 Pro with local NATS, the median additional latency is 1–3 ms. The PRD target for dispatch overhead is < 5 ms median; NATS consumes a third to two-thirds of that budget before any policy logic.

4. **Operational gap**: JetStream durability semantics (ack/nack, redelivery) are valuable for long-running, dequeue-able work items — not for interactive token streaming. A streaming SSE response must be delivered within a single HTTP connection. NATS JetStream works as the dispatch/scheduling mechanism, but the actual inference response is still proxied back through the orchestrator or returned directly by the worker. This two-leg design is more complex than direct proxying without a corresponding benefit at local scale.

The correct model is: direct mode as the default, NATS as the opt-in for teams that have it and want its operational properties.

---

### C: Temporal or Prefect as the Hot-Path Dispatcher

**Decision: Rejected.**

Temporal and Prefect are excellent workflow orchestrators. Temporal in particular offers strong durability guarantees, first-class retry/compensation logic, and a solid Python/Go/Rust SDK. The team has evaluated them for batch inference scheduling.

They are wrong for serving hot-path dispatch for three reasons:

1. **Wrong abstraction level**: Temporal workflows are designed for durable, multi-step, long-running processes with explicit state machines, activity retries, and side-effect journals. A token-serving request is a sub-second-to-few-second stateless HTTP call. The overhead of workflow state persistence, signal handling, and activity scheduling (typically 5–50 ms for a minimal Temporal workflow start) is incompatible with the < 5 ms dispatch overhead target.

2. **External service dependency**: Both Temporal and Prefect require a server-side deployment (Temporal Server backed by Cassandra/PostgreSQL, or a Prefect API server). This moves the ax-serving dependency surface from "optional NATS JetStream" to "Temporal Server with persistent database" — a much heavier operational requirement for all users.

3. **Streaming incompatibility**: Temporal activities return values, not streams. SSE token streaming does not map to a Temporal activity result. Bridging this would require the activity to open a side-channel (WebSocket, gRPC stream, etc.) to the original client — reintroducing most of the complexity of direct proxying with none of the benefits of workflow management.

Temporal and Prefect remain strong candidates for **batch inference scheduling** (e.g., scheduled fine-tune evaluation jobs, overnight soak tests, model benchmark pipelines). They are not appropriate for the serving hot path.

---

### D: HTTP Long-Polling Workers (Reverse Registration)

**Decision: Rejected for Phase 1; revisit in Phase 2.**

In this model, workers long-poll the orchestrator for work (rather than the orchestrator pushing to workers). Workers call `GET /internal/dispatch/next?model_id=llama3-8b&timeout=30s` and block until a request arrives.

Benefits:
- Workers initiate all connections (simpler firewall rules in some environments).
- Orchestrator does not need to hold per-worker HTTP clients.

Why rejected now:
- Adds 1–2 RTTs of latency for the first available slot (worker must be polling at exactly the right time).
- Request assignment becomes probabilistic (multiple workers competing on the same long-poll endpoint); requires careful implementation to avoid thundering herd.
- Streaming responses become harder to route (the worker must stream the response back through the same long-poll connection or open a new reverse connection to the client).

NATS JetStream pull consumers address these problems correctly. If long-poll workers are needed for specific network topologies, they should be evaluated against NATS mode first.

---

### E: gRPC Worker-to-Worker with Service Mesh

**Decision: Deferred to Phase 3 (distributed multi-Mac).**

For a future where workers span multiple Macs in a LAN or cluster, a service-mesh approach (gRPC + health probes + load-balancer sidecar, e.g., Envoy or Linkerd) becomes appropriate. This is intentionally out of scope for the local single-Mac orchestration problem. The internal REST API defined in this ADR is designed to be straightforward to bridge to gRPC in a later phase if needed.

---

## Implementation Boundaries

The dispatch layer respects the existing `InferenceBackend` encapsulation:

1. **No backend-specific logic in the dispatcher.** Workers expose the standard OpenAI REST API (`/v1/chat/completions`). The orchestrator dispatches an HTTP request; it does not know whether the worker uses mistralrs or llama.cpp. Backend routing (ADR-010) is resolved at worker startup.

2. **Dispatch happens after model resolution.** The orchestrator resolves `model_id` to an eligible worker set. The worker itself resolves the backend at load time. No backend selection happens in the dispatch path.

3. **Queue limits are explicit and bounded.** `AXS_GLOBAL_QUEUE_MAX` is enforced in the orchestrator. Workers also enforce their own `max_inflight` via the existing scheduler. The orchestrator must not silently over-enqueue.

4. **Internal API is not public.** `/internal/workers/*` endpoints are bound to loopback and are not served on the public port. In future, mutual TLS or a Unix socket can restrict access further.

5. **NATS mode is a compile-feature or config-only path.** The `async-nats` dependency is not pulled in when `AXS_WORKER_MODE=direct`. This keeps the binary footprint and startup time unchanged for the default case.

---

## Consequences

### Positive

- **Zero-dependency default**: direct mode works everywhere ax-serving is installed.
- **Progressive adoption**: teams start with direct mode, switch to NATS when operational needs warrant it.
- **Crash isolation**: worker death is bounded; in-progress requests on a crashed worker fail, but the orchestrator reroutes to surviving workers. The orchestrator itself is a lightweight process (no model weights, no Metal context).
- **Drain semantics**: rolling restarts without dropped requests become possible.
- **Unified observability**: same metrics and health schema regardless of mode; tooling does not bifurcate.
- **Policy extensibility**: `DispatchPolicy` is a trait; adding `least_latency` or custom policies requires no changes to the dispatch framework.

### Negative

- **Two code paths to maintain**: direct and NATS modes share interfaces but have separate transport implementations. Divergence risk requires strict parity testing.
- **NATS mode adds operational surface**: broker deployment, JetStream stream configuration, redelivery tuning, dead-letter handling — all must be documented and monitored.
- **Orchestrator is a new SPOF**: in direct mode, the orchestrator process sits in front of all workers. Its crash or overload affects all traffic. Mitigation: the orchestrator holds no model weights and no Metal context; it should be fast to restart and lightweight to run.
- **Internal API security**: `/internal/workers/*` must remain loopback-only. Any future multi-host deployment must add authentication before exposing these endpoints beyond localhost.

---

## Configuration

See `.internal/prd/PRD-AX-SERVING.md`, `docs/runbooks/multi-worker.md`, and
`config/serving.example.yaml` for the current configuration surface.

Key variables:

```bash
# Mode selection
AXS_WORKER_MODE=direct          # default; no external dependencies
AXS_WORKER_MODE=nats            # requires AXS_NATS_URL

# Worker lifecycle
AXS_WORKER_HEARTBEAT_MS=5000    # how often workers heartbeat
AXS_WORKER_TTL_MS=15000         # missed-heartbeat window before dead

# Dispatch
AXS_DISPATCH_POLICY=least_inflight   # least_inflight | weighted_round_robin | model_affinity

# Queue
AXS_GLOBAL_QUEUE_MAX=128
AXS_GLOBAL_QUEUE_WAIT_MS=10000
AXS_OVERLOAD_POLICY=reject       # reject | shed_oldest

# NATS (only when WORKER_MODE=nats)
AXS_NATS_URL=nats://127.0.0.1:4222
AXS_NATS_STREAM=ax-serving
```

---

## Rollout Plan

1. **M1**: Implement `WorkerRegistry`, heartbeat TTL, health state machine. `DirectDispatcher` with `least_inflight`. Internal REST API. Feature-flagged behind `AXS_WORKER_MODE=direct` (which is the future default; initially opt-in for testing).

2. **M2**: `GlobalQueue` with both overload policies. Extended `/health` and `/v1/metrics`. Integration tests for queue saturation and overload.

3. **M3**: Complete dispatch policies (`weighted_round_robin`, `model_affinity`). Reroute on dispatch error with counter tracking.

4. **M4**: `NatsDispatcher`. Feature-flagged behind `AXS_WORKER_MODE=nats`. Same bench suite against both modes.

5. **M5**: Load test all configurations. Update BMS. Publish runbooks. Make `direct` mode the documented default.

---

## Validation Gates

All gates must pass before declaring M5 complete:

| Gate | Pass Criterion |
|---|---|
| Queue never exceeds configured max | `dispatch_queue_depth` ≤ `AXS_GLOBAL_QUEUE_MAX` at all times |
| Worker death detected within TTL | State transitions `healthy → unhealthy → dead` at configured intervals |
| Dead worker removed from rotation | No requests dispatched to `dead` worker |
| Reroute on dispatch error | `request_rerouted_total` increments; client receives correct response |
| Drain completes cleanly | No new requests sent after drain request; drain-complete only after inflight = 0 |
| Throughput target | ≥ 1.2× req/s at 16 concurrent clients vs single-worker baseline |
| p95 latency target | ≤ 0.85× p95 vs single-worker baseline |
| Dispatch overhead (direct) | < 5 ms median under 16-client load |
| Mode parity | Both `direct` and `nats` modes pass same integration test suite |
| No single-worker regression | Single-worker path PRD-v1.0 targets unaffected |
