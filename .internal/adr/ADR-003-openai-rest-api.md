# ADR-003: Add OpenAI-Compatible REST API

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

---

## Context

ax-engine exposes only a gRPC interface (`EngineService` over UDS). This is effective
for machine-to-machine control but creates friction for:

- Developers using the official OpenAI Python/Node.js SDKs (`openai.ChatCompletion.create`)
- Tools that integrate via standard OpenAI REST (LangChain, Continue.dev, Cursor, etc.)
- Simple `curl`-based testing

mistral.rs ships a full OpenAI-compatible REST server (`mistralrs-server-core`) built on
Axum. It handles `/v1/chat/completions` with SSE streaming, `/v1/models`, `/v1/completions`,
and `/v1/embeddings`.

ax-serving should expose the same REST surface to enable zero-config integration with
the OpenAI ecosystem.

---

## Decision

`ax-serving-api` includes an Axum-based HTTP server alongside the existing tonic gRPC
server. Both share the same `ServingLayer` (model registry, metrics, backend).

### Endpoints (Phase 1)

```
POST /v1/chat/completions    — streaming (SSE) + blocking
GET  /v1/models              — list loaded models
GET  /health                 — liveness + readiness (thermal state, model list)
```

### Endpoints (Phase 2)

```
POST /v1/completions         — raw token completion
POST /v1/embeddings          — embedding vectors
GET  /metrics                — Prometheus-format scrape endpoint
```

### Request/Response Schema

Strict subset of OpenAI API v1 schema. Key fields:

```json
// POST /v1/chat/completions
{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 0.9,
  "seed": 42
}
```

Streaming response via SSE (`text/event-stream`), each chunk:
```json
data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
```

Final chunk:
```json
data: {"id":"...","choices":[{"finish_reason":"stop"}]}
data: [DONE]
```

### Port Configuration

| Service | Default | Env Override |
|---|---|---|
| REST HTTP | 8080 | `AXS_REST_PORT` |
| gRPC (UDS) | `/tmp/ax-serving.sock` | `AXS_GRPC_SOCKET` |
| gRPC (TCP) | 50051 | `AXS_GRPC_PORT` |

### Server Startup

Both servers start in parallel from `ax-serving-cli serve`:

```rust
// ax-serving-api/src/lib.rs
pub async fn serve(layer: Arc<ServingLayer>, config: ServeConfig) -> Result<()> {
    tokio::try_join!(
        start_rest(layer.clone(), config.rest_addr),
        start_grpc(layer.clone(), config.grpc_socket),
    )?;
    Ok(())
}
```

---

## Consequences

### Positive

- **OpenAI SDK compatibility** — `openai.base_url = "http://localhost:8080/v1"` just works
- **Minimal additional code** — borrow schema types from mistralrs-server-core
- **SSE streaming** — same progressive output experience as OpenAI API
- **Tokio-native** — no extra threads; Axum runs on the same Tokio runtime as tonic

### Negative

- **Two servers to maintain** — REST + gRPC both need to handle model errors, sampling
  params, streaming. Duplication mitigated by shared `ServingLayer`.
- **Schema completeness** — full OpenAI schema has ~40 fields; Phase 1 supports the
  ~10 most common. Unknown fields must be gracefully ignored (not errored).

### Neutral

- The REST API does not replace gRPC — gRPC remains the preferred control plane for
  the Go management layer and programmatic model lifecycle management.

---

## Alternatives Considered

### A: gRPC only (no REST)

Rejected. Blocks all OpenAI-ecosystem tooling. Developer experience is significantly
worse for quick iteration.

### B: HTTP/JSON proxy in front of gRPC

Rejected. An extra proxy process adds latency and operational complexity. Axum in-process
is simpler and has zero IPC overhead.

### C: Use mistralrs-server-core directly

Rejected. mistralrs-server-core does not integrate with ax-serving's model registry,
prefix cache, or gRPC control plane. We need the REST handler to share `ServingLayer`.
