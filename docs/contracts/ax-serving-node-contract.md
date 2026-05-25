# AX Serving Node Contract

> **Status**: Active
> **Date**: 2026-05-25
> **Owner**: AX Serving Team
> **Related**: [Runtime Responsibility Inventory](ax-serving-runtime-responsibility-inventory.md)

---

## Purpose

The node contract defines how inference runtimes join an AX Serving fleet.

AX Serving is the serving control plane. Runtime nodes own model loading,
token generation, batching internals, KV cache internals, kernels, and
runtime-specific tuning.

Target runtime nodes:

- Mac nodes running `ax-engine`
- PC CUDA nodes running `vllm`
- NVIDIA Thor nodes running `vllm`

---

## Registration

Nodes register with:

```text
POST /internal/workers/register
```

Required fields:

| Field | Type | Meaning |
|---|---:|---|
| `addr` | string | Address AX Serving uses to reach the worker proxy. |
| `capabilities` | object or string array | Model and modality inventory. |
| `max_inflight` | number | Maximum requests AX Serving should route concurrently. |

Runtime-neutral fields:

| Field | Type | Meaning |
|---|---:|---|
| `backend` | string | Legacy routing hint: `native`, `llama_cpp`, `sglang`, `vllm`, or `auto`. |
| `runtime` | string | Runtime owner: `ax_engine`, `vllm`, `sglang`, `llama_cpp`, or `unknown`. |
| `runtime_mode` | string | Integration mode: `adapter` for external runtime adapters, `embedded` for AX Serving compatibility workers. |
| `runtime_version` | string | Runtime version reported by the adapter, when known. |
| `hardware_class` | string | Placement class such as `mac`, `pc-cuda`, or `thor`. |
| `runtime_endpoint` | string | Runtime-compatible endpoint or proxy target, when different from `addr`. |
| `supported_operations` | string array | Operations such as `llm`, `embedding`, and `vision`. |
| `node_class` | string | Operator-defined node class, e.g. `m3-max-128g` or `thor`. |
| `worker_pool` | string | Operator-defined placement or maintenance pool. |

Compatibility rules:

- `runtime` is optional for legacy workers. If absent, AX Serving derives it
  from `backend` where possible.
- `runtime_mode` is optional for legacy workers. New runtime-node adapters should
  send `adapter`; `ax-serving serve` sends `embedded` so diagnostics can
  quarantine compatibility workers explicitly instead of relying only on backend
  heuristics.
- `supported_operations` is optional. If absent, AX Serving derives it from the
  structured capability descriptor.
- Unknown fields should be ignored by the control plane.
- Runtime-specific fields must not be required for basic routing.

Example vLLM node registration:

```json
{
  "addr": "10.0.10.21:18081",
  "backend": "vllm",
  "runtime": "vllm",
  "runtime_mode": "adapter",
  "runtime_version": "0.13.0",
  "hardware_class": "pc-cuda",
  "runtime_endpoint": "http://10.0.10.21:8000",
  "capabilities": {
    "llm": true,
    "embedding": false,
    "vision": true,
    "models": ["qwen3-32b"],
    "max_context": 32768
  },
  "model_inventory": [
    {
      "id": "qwen3-32b",
      "max_context": 32768,
      "quantization": "awq",
      "artifact_format": "safetensors",
      "modalities": ["text", "vision"],
      "supported_operations": ["llm", "vision"]
    }
  ],
  "supported_operations": ["llm", "vision"],
  "max_inflight": 16,
  "worker_pool": "cuda",
  "node_class": "pc-cuda"
}
```

---

## Heartbeat

Nodes heartbeat with:

```text
POST /internal/workers/{worker_id}/heartbeat
```

Required fields:

| Field | Type | Meaning |
|---|---:|---|
| `inflight` | number | Current routed request count. |

Optional fields:

| Field | Type | Meaning |
|---|---:|---|
| `model_ids` | string array | Current loaded model inventory. |
| `model_inventory` | object array | Optional structured per-model metadata. |
| `thermal_state` | string | Runtime or host thermal state. |
| `rss_bytes` | number | Worker process RSS. |
| `active_sequences` | number | Active decode sequences. |
| `decode_tok_per_sec` | number | Recent decode throughput. |
| `ttft_p95_ms` | number | Runtime-reported P95 TTFT. |
| `queue_depth` | number | Pending queue depth at the worker. |
| `error_rate` | number | Recent worker-side error fraction. |
| `kv_pages_used` | number | KV cache pages used, if reported. |
| `kv_pages_total` | number | KV cache page budget, if reported. |
| `kv_utilization` | number | KV/cache utilization ratio from 0.0 to 1.0, used when page counters are unavailable. |
| `prefix_reusable_tokens` | number | Prefix-cache reusable token count. |
| `active_batch_size` | number | Runtime internal batch occupancy. |
| `max_batch_size` | number | Runtime batch capacity. |
| `batch_utilization` | number | Batch utilization ratio from 0.0 to 1.0, used when batch counters are unavailable. |

Heartbeat telemetry is best effort. AX Serving must continue routing safely when
optional telemetry is absent.

`model_inventory` entries are additive. Known fields are:

| Field | Type | Meaning |
|---|---:|---|
| `id` | string | Stable model id used for routing. |
| `max_context` | number | Model context limit, when reported by the runtime. |
| `quantization` | string | Quantization or compression format, when known. |
| `artifact_format` | string | Artifact/container format such as `gguf` or `safetensors`, when known. |
| `modalities` | string array | Runtime-reported modalities such as `text` or `vision`. |
| `supported_operations` | string array | Model-level operations such as `llm`, `embedding`, or `vision`. |

If `model_inventory` is absent, AX Serving derives id-only entries from
`capabilities.models` during registration and preserves known metadata across
heartbeats while `model_ids` still contain the same model id.

Runtime-node adapters may translate runtime `/metrics` output into these
heartbeat fields. The generic `ax-runtime-agent` recognizes these common
Prometheus gauge names when present:

| Metric | Heartbeat field |
|---|---|
| `ax_runtime_active_sequences` | `active_sequences` |
| `ax_runtime_decode_tok_per_sec` | `decode_tok_per_sec` |
| `ax_runtime_ttft_p95_ms` | `ttft_p95_ms` |
| `ax_runtime_queue_depth` | `queue_depth` |
| `ax_runtime_error_rate` | `error_rate` |
| `ax_runtime_kv_pages_used` | `kv_pages_used` |
| `ax_runtime_kv_pages_total` | `kv_pages_total` |
| `ax_runtime_kv_utilization` | `kv_utilization` |
| `ax_runtime_prefix_reusable_tokens` | `prefix_reusable_tokens` |
| `ax_runtime_active_batch_size` | `active_batch_size` |
| `ax_runtime_max_batch_size` | `max_batch_size` |
| `ax_runtime_batch_utilization` | `batch_utilization` |

The adapter also accepts runtime-specific aliases where stable enough to treat
as best-effort hints:

- AX Serving / ax-engine style `axs_*` Prometheus metrics such as
  `axs_scheduler_queue_depth`, `axs_scheduler_inflight_count`,
  `axs_scheduler_decode_sequences_active`, and `axs_ttft_p95_us`
- AX Serving / ax-engine style `/v1/metrics` JSON fields such as
  `scheduler.queue_depth`, `scheduler.inflight_count`,
  `scheduler.ttft_p95_us`, `kv_cache.utilization`, and `batch.utilization`
- vLLM gauges such as `vllm:num_requests_running`,
  `vllm:num_requests_waiting`, `vllm:avg_generation_throughput_toks_per_s`,
  and `vllm:gpu_cache_usage_perc`
- vLLM `time_to_first_token_seconds_bucket` histogram buckets, translated to
  `ttft_p95_ms`

Missing metrics are not registration failures; the adapter sends safe defaults
and AX Serving keeps routing by health, capacity, and model inventory.

### Per-runtime telemetry support

The table below documents which optional heartbeat fields each known runtime
actually exposes. Blank cells mean the field is not available from that runtime
without additional instrumentation; adapters emit the safe default (`0` or
`0.0`) for those entries.

| Heartbeat field | ax-engine (AXS) | vLLM | SGLang |
|---|---|---|---|
| `active_sequences` | `axs_scheduler_inflight_count` gauge | `vllm:num_requests_running` gauge | `num_running_reqs` gauge |
| `queue_depth` | `axs_scheduler_queue_depth` gauge | `vllm:num_requests_waiting` gauge | `num_waiting_reqs` gauge |
| `decode_tok_per_sec` | `axs_decode_tok_per_sec` gauge | `vllm:avg_generation_throughput_toks_per_s` gauge | `avg_generation_throughput_toks_per_s` gauge |
| `ttft_p95_ms` | `axs_ttft_p95_us` gauge (÷ 1000) | `time_to_first_token_seconds_bucket` histogram P95 | `time_to_first_token_seconds_bucket` histogram P95 |
| `kv_pages_used` | `axs_kv_pages_used` gauge | `vllm:gpu_cache_usage_perc × total_pages` (estimated) | _(not available)_ |
| `kv_pages_total` | `axs_kv_pages_total` gauge | `vllm:num_gpu_blocks` gauge | _(not available)_ |
| `kv_utilization` | `kv_cache.utilization` JSON field | `vllm:gpu_cache_usage_perc` gauge | _(not available)_ |
| `batch_utilization` | `batch.utilization` JSON field | _(not available)_ | _(not available)_ |
| `error_rate` | `axs_error_rate` gauge | _(not available — use counters)_ | _(not available)_ |
| `active_batch_size` | `axs_scheduler_inflight_count` gauge | _(not available)_ | _(not available)_ |

**Note on `error_rate`**: vLLM exposes error counts as counters
(`vllm:num_preemptions_total`), not a fraction gauge. Computing a rate requires
delta tracking across heartbeats; adapters currently send `0.0` as the safe
default. The orchestrator's error-rate diagnostic threshold is therefore not
effective for vLLM or SGLang workers.

---

## Runtime Responsibilities

Runtime nodes own:

- model loading and unloading inside the runtime
- tokenizer and prompt formatting behavior
- inference execution
- runtime scheduler and batching internals
- KV cache internals
- hardware kernels and tuning
- runtime-native metrics collection

AX Serving owns:

- northbound API compatibility
- worker and model inventory
- placement and routing policy
- admission, retries, overload, and drain behavior
- normalized metrics and diagnostics
- operator-facing control workflows

The fleet admin surface groups workers by pool, node class, backend, and
runtime. Runtime groups are the preferred operational view for mixed ax-engine
and vLLM fleets.

---

## Adapter Guidance

Adapters should be thin protocol bridges.

The public workspace provides `ax-runtime-agent` as the generic runtime-node
adapter binary. It is currently implemented from the same code path as the
legacy `ax-thor-agent` binary, so existing Thor deployments remain compatible.
New deployments should prefer the generic `AXS_NODE_*` environment variables:

| Variable | Meaning |
|---|---|
| `AXS_CONTROL_PLANE_URL` | AX Serving internal control-plane URL. |
| `AXS_WORKER_TOKEN` | Optional internal worker registration token. |
| `AXS_NODE_RUNTIME_URL` | OpenAI-compatible runtime endpoint to proxy to. |
| `AXS_NODE_RUNTIME` | Runtime owner, e.g. `ax_engine` or `vllm`. |
| `AXS_NODE_LISTEN_ADDR` | Local adapter listen address. |
| `AXS_NODE_ADVERTISED_ADDR` | Routable adapter address registered with AX Serving. |
| `AXS_NODE_HARDWARE_CLASS` | Placement class, e.g. `mac`, `pc-cuda`, or `thor`. |
| `AXS_NODE_CLASS` | Operator-defined node class. |
| `AXS_NODE_WORKER_POOL` | Operator-defined routing or maintenance pool. |
| `AXS_NODE_MAX_INFLIGHT` | Advertised concurrent request capacity. |

An ax-engine runtime node should report `runtime = "ax_engine"`. When ax-engine
exposes an OpenAI-compatible endpoint, the generic `ax-runtime-agent` can
register and proxy it through this contract. Runtime-specific inventory,
health, and metrics mappings can be added behind the same adapter boundary
without moving ax-engine internals into AX Serving.

A vLLM adapter should report `runtime = "vllm"` and use vLLM's
OpenAI-compatible serving endpoint as the runtime endpoint. PC CUDA and NVIDIA
Thor nodes should differ by `hardware_class`, `node_class`, and operator pool,
not by a separate AX Serving inference implementation.

---

## Compatibility Paths

The existing `ax-serving serve` command can still register as a worker for
local Mac compatibility. When runtime metadata is not provided, it registers as:

- `runtime = "ax_engine"`
- `runtime_mode = "embedded"`
- `hardware_class = "mac"`
- `runtime_endpoint = "http://<worker-addr>"`

This path is a compatibility bridge for local development and migration. New
Mac inference integration should prefer `ax-runtime-agent` in front of an
OpenAI-compatible ax-engine endpoint, or another thin adapter that implements
this contract without moving ax-engine runtime internals into AX Serving.

---

## Versioning And Compatibility

This contract is additive by default.

- New optional fields may be added without breaking older workers.
- Required fields require a public contract update and migration note.
- Workers should ignore unknown fields returned by AX Serving.
- AX Serving should derive safe defaults for missing runtime metadata when
  legacy workers register with only `backend`, `addr`, `capabilities`, and
  `max_inflight`.
- Runtime-specific lifecycle operations must be capability-gated. A worker that
  cannot load, unload, or reload models through the control plane should still
  be routable when it advertises a compatible model inventory.

Breaking changes to this contract must update the PRD traceability table,
contract inventory, and relevant runbooks in the same change set.
