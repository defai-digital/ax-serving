# AX Fabric Runtime Contract

This document defines the supported runtime contract between AX Fabric and AX Serving for the `v1.3` line.

## Purpose

AX Fabric should be able to treat AX Serving as its standard offline model runtime and serving control plane.

This contract defines:

- which endpoints AX Fabric may rely on
- what health and lifecycle semantics mean
- which failure classes are expected and stable

## Core Assumptions

- AX Serving is an offline-first runtime.
- AX Serving may run with bearer auth (`AXS_API_KEY`) or explicit no-auth development mode (`AXS_ALLOW_NO_AUTH=true`).
- `GET /health` is the runtime health contract.
- `GET /v1/models` is the authoritative model-availability view.
- AX Fabric should treat only the endpoints and fields documented here as stable for `v1.3`.

## Stable Endpoints For AX Fabric

- `GET /health`
- `GET /v1/models`
- `POST /v1/models`
- `DELETE /v1/models/{id}`
- `POST /v1/models/{id}/reload`
- `POST /v1/embeddings`
- `GET /v1/metrics`

Anything outside this list should be treated as implementation detail unless separately documented.

## Health Contract

`GET /health` always returns HTTP `200` when the process is alive enough to answer.

Important fields:

- `status`: `ok` or `degraded`
- `ready`: whether the runtime is able to serve
- `model_available`: whether at least one model is loaded
- `reason`: present when degraded
- `loaded_model_count`: number of currently loaded models

Interpretation:

- `status=ok`
  - runtime is ready
  - at least one model is available
- `status=degraded`
  - process is alive, but runtime is not in the ideal serving state
  - common reasons:
    - `no_models_loaded`
    - `thermal_critical`
    - `thermal_critical_no_models`

AX Fabric integration rule:

- treat `ready=true` as "the runtime can answer requests"
- treat `model_available=true` as "a model-backed workload can be dispatched now"
- treat `status=degraded` with `reason=no_models_loaded` as a recoverable startup state, not a process failure

Expected startup sequence for a healthy local deployment:

1. process starts
2. `GET /health` returns `200` with `status=degraded`, `ready=true`, `model_available=false`
3. a model is loaded through `POST /v1/models`
4. `GET /health` returns `200` with `status=ok`, `ready=true`, `model_available=true`

## Model Lifecycle Contract

### Load

`POST /v1/models`

Success:
- HTTP `201`
- returns:
  - `model_id`
  - `state=loaded`
  - `ready`
  - `model_available`
  - `loaded_model_count`
  - `architecture`
  - `context_length`
  - `load_time_ms`

Common failures:
- HTTP `409` if model already loaded
- HTTP `422` for invalid model id / invalid file / invalid format
- HTTP `403` for disallowed path
- HTTP `503` for capacity exhaustion

### Unload

`DELETE /v1/models/{id}`

Success:
- HTTP `200`
- returns:
  - `model_id`
  - `state=unloaded`
  - `ready`
  - `model_available`
  - `loaded_model_count`

Common failure:
- HTTP `404` if model is not loaded

### Reload

`POST /v1/models/{id}/reload`

Success:
- HTTP `200`
- returns:
  - `model_id`
  - `state=loaded`
  - `ready`
  - `model_available`
  - `loaded_model_count`
  - `architecture`
  - `load_time_ms`

Common failure:
- HTTP `404` if model is not loaded

AX Fabric should not infer lifecycle success from status code alone. It should read:

- `state`
- `ready`
- `model_available`
- `loaded_model_count`

## Embeddings Contract

`POST /v1/embeddings`

AX Fabric may rely on:

- HTTP `200` on successful embedding generation
- HTTP `404` when the requested model is not loaded
- HTTP `422` for invalid request shape/validation failure
- HTTP `501` when the loaded backend does not support embeddings
- HTTP `500` when the embedding backend fails after request validation succeeds

Stable response fields on HTTP `200`:

- `object=list`
- `model`
- `data[]`
  - `object=embedding`
  - `index`
  - `embedding`
- `usage.prompt_tokens`
- `usage.total_tokens`

AX Fabric may use either:

- `encoding_format=float`
- `encoding_format=base64`

Any other `encoding_format` should be treated as client error.

## Metrics Contract

`GET /v1/metrics`

AX Fabric may rely on the following scheduler/runtime keys existing:

- `scheduler.queue_depth`
- `scheduler.inflight_count`
- `scheduler.cache_follower_waiting`
- `scheduler.ttft_p50_us`
- `scheduler.ttft_p95_us`
- `scheduler.ttft_p99_us`
- `scheduler.prefill_tokens_active`
- `scheduler.decode_sequences_active`
- `scheduler.split_scheduler_enabled`
- `loaded_models`
- `thermal`

These metrics are intended for readiness decisions, integration diagnostics, and local operator visibility. They are not a replacement for request-level success criteria.

## Failure Semantics AX Fabric Should Handle

- `404` from `POST /v1/embeddings`, `POST /v1/chat/completions`, or `POST /v1/completions`
  - requested model is not loaded
- `422` from lifecycle or inference endpoints
  - validation failure or invalid model path/configuration
- `429`
  - admission queue full
- `500`
  - backend execution failure after admission
- `501`
  - requested backend capability is not implemented
- `503`
  - throttling, capacity exhaustion, timeout, or service-side overload

## Operational Guidance

- For offline enterprise deployments, prefer `config/serving.offline-enterprise.yaml`.
- Keep bind addresses on loopback unless a controlled network boundary is intentional.
- Use `AXS_MODEL_ALLOWED_DIRS` to restrict runtime model loading to approved local directories.

## Non-Contract Items

The following are not treated as a stable AX Fabric integration contract in `v1.3`:

- internal orchestrator endpoints
- dashboard HTML structure
- benchmark report file formats
- any scheduler metric not explicitly listed in the Metrics Contract section
