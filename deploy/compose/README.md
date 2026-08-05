# Docker Compose evaluation stack

CPU-only local evaluation for the AX Serving gateway. **Not** an HA or production certification surface.

## Quick start

```bash
# From repository root
docker compose -f deploy/compose/compose.yaml up --build gateway redis
```

Probe endpoints:

```bash
curl -i http://127.0.0.1:18080/livez
curl -i http://127.0.0.1:18080/readyz      # control-plane ready without workers
curl -i http://127.0.0.1:18080/routablez  # 503 until a runtime agent registers
```

Optional runtime agent (requires a reachable OpenAI-compatible runtime):

```bash
cp deploy/compose/.env.example deploy/compose/.env
# edit AXS_NODE_RUNTIME_URL / tokens
docker compose -f deploy/compose/compose.yaml --profile agent up --build
```

This optional profile exercises the current node-agent compatibility path. It is not the future
Dynamo Domain Adapter and does not certify NVIDIA PC or Thor federation.

The gateway and agent share evaluation-only control/dispatch token defaults so the profile is
runnable. An operator-owned `.env` overrides both ends together. `AXS_NODE_HARDWARE_CLASS`,
`AXS_NODE_WORKER_POOL`, runtime version, health path, capacity, and shutdown timeout are also
overridable; the Compose file no longer labels every external runtime as `cpu`.

## TensorRT-LLM on one NVIDIA PC

The TensorRT-LLM overlay pins the exact linux/amd64 image and TinyLlama revision used for the
compatibility test. It keeps the Hugging Face cache in a named volume and waits for `/v1/models`
before starting `ax-runtime-agent`:

```bash
cp deploy/compose/tensorrt-llm.env.example .env.trtllm
# Replace credentials and review every image/model/runtime pin.
docker compose \
  --env-file .env.trtllm \
  -f deploy/compose/compose.yaml \
  -f deploy/compose/tensorrt-llm.compose.example.yaml \
  --profile agent --profile tensorrt-llm \
  up --build
```

Run bounded stream, non-stream, inventory-stability, and concurrency checks with
an isolated Python environment:

```bash
uv run scripts/qualification/runtime/smoke_openai_runtime.py \
  --base-url http://127.0.0.1:18080 \
  --model tinyllama-trtllm \
  --runtime tensorrt_llm \
  --stability-seconds 45 \
  --requests 8 \
  --concurrency 4 \
  --allow-insecure-http \
  --output target/trtllm-compat-smoke.json
```

The runner detects inventory, transport, concurrency, response-shape, and SSE termination failures
without labeling the result as Dynamo evidence. A successful direct compatibility run is **not**
Dynamo-domain certification. Production NVIDIA deployments remain the separately pinned
`ax-dynamo-adapter` topology.

On macOS, AX Engine remains a native host process. Use
`AXS_NODE_RUNTIME_URL=http://host.docker.internal:<port>` from the agent container,
or run the agent natively with `AXS_NODE_ADVERTISED_URL`.

## Notes

- Compose uses `AXS_ALLOW_NO_AUTH=true` for evaluation only.
- The default `legacy_compat` mode makes dynamic agent inventory routable. An
  explicit production catalog is intentionally not fabricated by this stack.
- Gateway readiness does not depend on runtime capacity.
- Images are built from portable features only (no AX Engine, Dynamo, CUDA, or MLX SDK).
