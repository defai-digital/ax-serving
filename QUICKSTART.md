# AX Serving Quickstart

This guide gets you from checkout to a working local LLM server.

## Prerequisites

- Apple Silicon macOS 14+
- Rust toolchain (`cargo`)
- `llama-server` available on `PATH` (from [llama.cpp](https://github.com/ggerganov/llama.cpp))
- A GGUF model file at `./models/<model>.gguf`

Validate your environment:

```bash
cargo check --workspace
which llama-server
```

---

## Path A: Single Worker (simplest)

One process handles everything — model loading, serving, and admission control.

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf \
  --model-id default \
  --host 127.0.0.1 \
  --port 18080
```

Verify it is up:

```bash
curl -sS http://127.0.0.1:18080/health
curl -sS http://127.0.0.1:18080/v1/models
```

---

## Path B: Gateway + Multiple Workers

The gateway is a stateless API proxy; workers do the inference. Workers register themselves automatically.

Start the gateway:

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving-api -- \
  --port 18080 \
  --internal-port 19090 \
  --policy least_inflight
```

Start one or more workers:

```bash
AXS_ALLOW_NO_AUTH=true \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf \
  --model-id default \
  --port 18081 \
  --orchestrator http://127.0.0.1:19090
```

Start additional workers on ports `18082`, `18083`, etc. using the same `--orchestrator` flag.

Verify worker registration:

```bash
curl -sS http://127.0.0.1:19090/internal/workers
curl -sS http://127.0.0.1:18080/v1/models
```

Dispatch policies:
- `least_inflight` — routes to the worker with the fewest in-flight requests (default).
- `weighted_round_robin` — rotates across workers proportionally to their weights.

---

## Chat completions

Non-streaming:

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Give me three short points about Rust."}
    ],
    "stream": false,
    "max_tokens": 96
  }'
```

Streaming SSE:

```bash
curl -N -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Give me three short points about Rust."}
    ],
    "stream": true,
    "max_tokens": 96
  }'
```

### TypeScript SDK (with Zod validation)

Build SDK locally:

```bash
cd sdk/javascript
npm install
npm run build
```

Example:

```ts
import { AxServingClient } from "@defai-digital/ax-serving";

const client = new AxServingClient({
  baseURL: "http://127.0.0.1:18080",
  apiKey: process.env.AXS_API_KEY,
});

const result = await client.chatCompletionsCreate({
  model: "default",
  messages: [{ role: "user", content: "Give me three short points about Rust." }],
  max_tokens: 96,
  repeat_penalty: 1.1,
});

console.log(result.choices[0]?.message?.content ?? "");
```

### Logprobs

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say hello."}],
    "logprobs": true,
    "top_logprobs": 3,
    "max_tokens": 16
  }'
```

Each response choice contains `logprobs.content[]` — one entry per token with `token`, `logprob`, `bytes`, and `top_logprobs`. Streaming works identically.

### Sampling parameters

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "seed": 42,
  "repeat_penalty": 1.1,
  "frequency_penalty": 0.2,
  "presence_penalty": 0.1,
  "mirostat": 2,
  "mirostat_tau": 5.0,
  "mirostat_eta": 0.1,
  "stop": ["<|end|>", "\n###"]
}
```

### Tool calling

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

`choices[0].message.tool_calls` contains `id`, `type`, and `function.name` + `function.arguments` (JSON string). Streaming works identically.

### Constrained generation (JSON mode and grammar)

Force JSON output:

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Return a JSON object with name and age."}],
    "response_format": {"type": "json_object"}
  }'
```

BNF grammar (llama.cpp extended syntax):

```json
{ "grammar": "root ::= (\"yes\" | \"no\")" }
```

### Vision / multimodal

Pass image URLs alongside text (requires a vision-capable model and mmproj file):

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

---

## Embeddings

Load a dedicated embedding model, or start a chat model with `--embeddings` passed to llama-server.

Basic embedding:

```bash
curl -sS http://127.0.0.1:18080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "input": "The quick brown fox"
  }'
```

Batch with options:

```bash
curl -sS http://127.0.0.1:18080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "input": ["sentence one", "sentence two"],
    "normalize": true,
    "truncate": true,
    "encoding_format": "float"
  }'
```

Base64-encoded output (efficient for large batches):

```bash
curl -sS http://127.0.0.1:18080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "input": "Hello world",
    "encoding_format": "base64"
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string \| string[] \| int[] \| int[][] | required | Text, token IDs, or batches of either |
| `normalize` | bool | `true` | L2-normalize each vector |
| `truncate` | bool | `true` | Truncate inputs exceeding context length |
| `encoding_format` | string | `"float"` | `"float"` or `"base64"` (little-endian f32 bytes) |

### Critical: batch size for embedding workloads

llama-server's default `n_ubatch=512` causes HTTP 500 errors for any input exceeding 512 tokens. Set both values to 4096 or higher:

```yaml
# config/serving.yaml
llamacpp:
  n_batch: 4096
  n_ubatch: 4096
```

Or with env vars at startup:

```bash
AXS_LLAMACPP_N_BATCH=4096 AXS_LLAMACPP_N_UBATCH=4096 \
cargo run -p ax-serving-cli --bin ax-serving -- serve -m ./models/embed-model.gguf ...
```

Without this, long document chunks will produce HTTP 500 errors from the llama-server subprocess.

---

## Runtime model management

Load a model:

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/models \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"m2","path":"/absolute/path/to/model.gguf"}'
```

Load with overrides:

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "llava",
    "path": "/models/llava-v1.6-mistral-7b.gguf",
    "mmproj_path": "/models/llava-v1.6-mistral-7b-mmproj.gguf",
    "backend": "llama_cpp",
    "n_gpu_layers": 32,
    "context_length": 4096
  }'
```

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Identifier (1–128 chars, alphanumeric + `-_./:`) |
| `path` | string | Absolute path to the `.gguf` file |
| `backend` | string? | `"llama_cpp"`, `"lib_llama"`, `"native"`, or `"auto"` |
| `mmproj_path` | string? | Multimodal projector `.gguf` for vision models |
| `n_gpu_layers` | int? | GPU layer count override (`-1` = all) |
| `context_length` | int? | Context window override (0 = model default) |

Unload a model:

```bash
curl -sS -X DELETE http://127.0.0.1:18080/v1/models/m2
```

Reload (unload then load again from the same path and config):

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/models/m2/reload
```

---

## Operator dashboard

Open in a browser while the server is running:

```
http://127.0.0.1:18080/dashboard
```

The dashboard requires no login and auto-polls `/v1/metrics` every 2 seconds. It shows:

- License status
- Worker table: health, inflight count, thermal state, chip model
- Queue depth and admission stats
- System RSS and request metrics

The **Remove** button on each worker row calls `DELETE /v1/workers/{id}` through the public proxy — no direct internal port access needed.

---

## Authentication

Recommended production setup:

```bash
AXS_API_KEY="token1,token2" \
cargo run -p ax-serving-cli --bin ax-serving -- serve \
  -m ./models/<model>.gguf --port 18080
```

Protected endpoints require:

```
Authorization: Bearer token1
```

If `AXS_API_KEY` is not set, you must explicitly opt in:

```bash
AXS_ALLOW_NO_AUTH=true
```

---

## License key

AX Serving is available under AGPL-3.0-only by default, with separate commercial licensing available. Register a commercial license key to suppress the multi-machine reminder and unlock Business edition features.

```bash
# One-time env var (both gateway and workers read this)
export AXS_LICENSE_KEY="your-key-here"

# Or persist via the API (survives restarts without the env var)
curl -sS -X POST http://127.0.0.1:18080/v1/license \
  -H 'Content-Type: application/json' \
  -d '{"key":"your-key-here"}'
```

Check current license state:

```bash
curl -sS http://127.0.0.1:18080/v1/license
```

---

## Single inference CLI (no server)

Run one local inference without starting a server:

```bash
cargo run -p ax-serving-cli --bin ax-serving -- \
  -m ./models/<model>.gguf \
  -p "Hello from AX Serving" \
  -n 64
```

Useful flags: `-v` (timing), `--chat` (chat templating), `--temp 0` (greedy), `--n-gpu-layers N`.

---

## Benchmarks

Throughput (tokens/sec, prefill + decode):

```bash
cargo run -p ax-serving-bench --release -- bench -m ./models/<model>.gguf
```

Per-token latency profile (TTFT + inter-token latency percentiles):

```bash
cargo run -p ax-serving-bench --release -- profile -m ./models/<model>.gguf
```

Mixed workload against a running server (short/medium/long prompts, P50/P95/P99):

```bash
cargo run -p ax-serving-bench --release -- mixed \
  --url http://127.0.0.1:18080 --model default
```

Cache effectiveness (cold vs. warm latency):

```bash
cargo run -p ax-serving-bench --release -- cache-bench \
  --url http://127.0.0.1:18080 --model default
```

24-hour soak test:

```bash
cargo run -p ax-serving-bench --release -- soak \
  -m ./models/<model>.gguf --duration_min 1440
```

Additional modes: `compare`, `regression-check`, `multi-worker`.

---

## Key environment variables

| Variable | Description |
|----------|-------------|
| `AXS_CONFIG` | Path to config YAML |
| `AXS_ROUTING_CONFIG` | Path to backends routing config |
| `AXS_REST_HOST` / `AXS_REST_PORT` | Bind address |
| `AXS_GRPC_HOST` / `AXS_GRPC_PORT` | TCP gRPC bind address (optional; defaults loopback) |
| `AXS_API_KEY` | Bearer token(s), comma-separated |
| `AXS_ALLOW_NO_AUTH` | Required when no `AXS_API_KEY` is set |
| `AXS_ORCHESTRATOR_HOST` / `AXS_ORCHESTRATOR_PORT` | Gateway bind address (defaults `127.0.0.1`) |
| `AXS_ORCHESTRATOR_ADDR` | Gateway address for worker registration |
| `AXS_INTERNAL_API_TOKEN` | Optional shared token for orchestrator internal APIs (`X-Internal-Token`) |
| `AXS_WORKER_HEARTBEAT_MS` | Worker heartbeat interval (default `5000`) |
| `AXS_WORKER_TTL_MS` | Worker TTL without heartbeat (default `15000`) |
| `AXS_DISPATCH_POLICY` | `least_inflight` or `weighted_round_robin` |
| `AXS_MODEL_ALLOWED_DIRS` | Optional comma-separated allowlist of model root directories for `POST /v1/models` |
| `AXS_LICENSE_KEY` | Commercial license key |
| `AXS_DASHBOARD_POLL_MS` | Dashboard refresh interval (default `2000`) |
| `AXS_LLAMACPP_N_BATCH` | llama-server logical batch size |
| `AXS_LLAMACPP_N_UBATCH` | llama-server physical micro-batch size |
| `AXS_LOG` | Log level: `debug`, `info`, `warn`, `error` |
| `AXS_LOG_FORMAT` | `pretty` (default) or `json` |

See `config/serving.example.yaml` for the full configuration reference.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `model file not found` | Verify the model path is correct |
| `failed to spawn llama-server` | Ensure `llama-server` is installed and on `PATH` |
| `401` / `403` | Check `AXS_API_KEY` matches the `Authorization: Bearer` token |
| `503` from gateway | Check `GET http://127.0.0.1:19090/internal/workers` — no healthy workers |
| Model missing from `GET /v1/models` | Worker is unhealthy or draining; check worker logs |
| Dashboard shows no workers | Start a worker with `--orchestrator http://127.0.0.1:19090` |
| Embedding fails with HTTP 500 | Set `n_ubatch` ≥ `n_batch` (both to 4096); see embeddings section above |
| License reminder in logs | Set `AXS_LICENSE_KEY` or POST to `/v1/license` |

---

## Next references

- [README.md](README.md)
- `config/serving.example.yaml` — full configuration reference
- `config/backends.yaml` — backend routing rules
- `docs/runbooks/multi-worker.md` — multi-worker deployment guide
- `docs/perf/service-tuning.md` — throughput and latency tuning
- [LICENSE](LICENSE) — full AGPL-3.0-only text
- [LICENSE-AGPL.md](LICENSE-AGPL.md) — AGPL reference
- [LICENSING.md](LICENSING.md) — dual-license policy
- [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) — commercial terms
- [CONTRIBUTING.md](CONTRIBUTING.md) — issue reporting and patch policy
