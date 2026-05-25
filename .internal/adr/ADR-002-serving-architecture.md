# ADR-002: Retain ax-engine's Layered Serving Architecture

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

---

## Context

ax-engine's inference stack is being replaced by mistralrs-core (see ADR-001). However,
ax-engine's **serving layer** is a separate concern that is well-engineered and
independent of the inference backend:

- `server/service.rs` — tonic gRPC service with server-streaming inference
- `server/registry.rs` — thread-safe model registry (Arc<RwLock<HashMap>>)
- `metrics/` — InferenceMetrics, OpTimer, LatencyHistogram, macOS RSS via mach API
- `scheduler/` — Rayon thread pool, ThermalMonitor, P-core pinning
- `kv/paged_kv.rs` — vLLM-style paged KV manager with LRU prefix cache
- `kv/gpu_cache.rs` — GPU KV block export/import for prefix reuse

mistral.rs's serving layer (`mistralrs-server-core`) focuses on OpenAI REST and lacks:
- gRPC control plane
- Multi-model registry with load/unload lifecycle
- Prefix cache with GPU block reuse (25–505× speedup)
- macOS-native thermal integration
- Paged KV stats in metrics

---

## Decision

ax-serving retains the **serving architecture** from ax-engine, porting it to the new
`ax-serving-api` crate. The inference backend is replaced; the serving layer is
preserved.

The following components are **ported with minimal changes**:

| Component | Source | Destination | Changes |
|---|---|---|---|
| `service.rs` | ax-core/server/ | ax-serving-api/src/grpc/ | Wire to `InferenceBackend` trait |
| `registry.rs` | ax-core/server/ | ax-serving-api/src/registry.rs | Keep as-is |
| `metrics/` | ax-core/metrics/ | ax-serving-api/src/metrics/ | Keep as-is |
| `scheduler/` | ax-core/scheduler/ | ax-serving-engine/src/scheduler/ | Remove Rayon (mistralrs uses Tokio) |
| `kv/paged_kv.rs` | ax-core/kv/ | ax-serving-api/src/kv/ | Phase 2 |
| `thermal.rs` | ax-core/ | ax-serving-engine/src/thermal.rs | Keep as-is |
| `proto/ax_engine.proto` | proto/ | proto/ax_serving.proto | Rename service |

The serving layer communicates with the inference backend through the `InferenceBackend`
trait defined in `ax-serving-engine`. This keeps the gRPC service agnostic of whether
the backend is mistralrs-core, a future CUDA backend, or a test mock.

---

## Serving Layer Architecture

```rust
// ax-serving-api/src/lib.rs
pub struct ServingLayer {
    registry: ModelRegistry,
    backend: Arc<dyn InferenceBackend>,
    metrics: MetricsStore,
    paged_kv: Option<PagedKvManager>,   // Phase 2
    thermal: ThermalMonitor,
}

impl ServingLayer {
    pub async fn start_grpc(&self, addr: &str) -> Result<()> { ... }
    pub async fn start_rest(&self, addr: &str) -> Result<()> { ... }
}
```

The gRPC service and REST API both hold an `Arc<ServingLayer>`. State is not duplicated.

---

## Consequences

### Positive

- **gRPC control plane preserved** — existing gRPC clients (Go control plane, etc.)
  continue working
- **Prefix cache preserved** — 25–505× speedup for repeated prompts (Phase 2)
- **Observability preserved** — RSS, thermal, KV stats, latency histograms
- **Model lifecycle management** — load/unload/list with race-safe registry
- **RAII cleanup** — PagedKvRequestGuard prevents resource leaks

### Negative

- **More porting work** — ~1500 lines of ax-engine serving code must be ported
- **Scheduler mismatch** — ax-engine used Rayon; mistralrs-core uses Tokio. The
  scheduler module is simplified: drop Rayon pool, use Tokio spawn_blocking instead.

### Neutral

- The proto definition is renamed from `EngineService` to `ServingService` but
  message types and RPC signatures remain identical — clients need only update
  the service name.

---

## Alternatives Considered

### A: Use mistralrs-server-core as the serving layer

Rejected. mistralrs-server-core lacks: gRPC, multi-model registry with load/unload
lifecycle, prefix cache, macOS thermal integration, paged KV stats. These are key
differentiators of ax-engine's design that are worth preserving.

### B: Build a new serving layer from scratch

Rejected. ax-engine's serving layer is already correct, well-tested, and production-
ready. Rewriting it provides no benefit.
