# Docker Compose evaluation stack

Use this stack to evaluate the AX Serving control plane or attach one existing runtime. It is
CPU-only unless a pinned NVIDIA runtime overlay is selected. It is **not** an HA or production
certification surface.

| Goal | Command or guide |
| --- | --- |
| Verify gateway startup and health semantics | Base `gateway redis` stack below |
| Attach an already-running OpenAI-compatible runtime | Optional `agent` profile below |
| Start vLLM, SGLang, or TensorRT-LLM on one NVIDIA PC | [NVIDIA runtime profiles](#nvidia-runtime-profiles-on-one-pc) |
| Attach TensorRT Edge-LLM on Jetson Thor | [Thor guide](../thor/README.md) |
| Operate a production NVIDIA domain | [Dynamo guide](../../docs/integrations/nvidia/DYNAMO.md) |

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

This proves that the gateway, listeners, configuration, and Redis connection are healthy. It does
not prove inference: `/routablez` remains `503` and `/v1/models` remains empty until a runtime
registers.

Optional runtime agent (requires a reachable OpenAI-compatible runtime):

```bash
cp deploy/compose/.env.example deploy/compose/.env
# edit AXS_NODE_RUNTIME_URL / tokens
docker compose -f deploy/compose/compose.yaml --profile agent up --build
```

After registration, `/routablez` should return `200`; use a model from `/v1/models` in the
[request examples](../../QUICKSTART.md#4-send-requests).

This optional profile exercises the current node-agent compatibility path. It is not the future
Dynamo Domain Adapter and does not certify NVIDIA PC or Thor federation.

The gateway and agent share evaluation-only control/dispatch token defaults so the profile is
runnable. An operator-owned `.env` overrides both ends together. `AXS_NODE_HARDWARE_CLASS`,
`AXS_NODE_WORKER_POOL`, runtime version, health path, capacity, and shutdown timeout are also
overridable; the Compose file no longer labels every external runtime as `cpu`.

## NVIDIA runtime profiles on one PC

The vLLM, SGLang, and TensorRT-LLM overlays pin the exact linux/amd64 images
and TinyLlama revision used for compatibility testing. They use retained
Hugging Face caches (shared by the vLLM and SGLang examples), wait for
`/v1/models`, and register distinct runtime identities through
`ax-runtime-agent`.

```bash
cp deploy/compose/vllm.env.example .env.vllm
# Replace the two evaluation credentials, then validate without starting:
scripts/qualification/runtime/run_compose_profile.sh validate \
  --runtime vllm \
  --env-file .env.vllm
```

Use one profile at a time on a single GPU:

| Runtime | Overlay | Example environment | Host port |
|---|---|---|---:|
| vLLM | `vllm.compose.example.yaml` | `vllm.env.example` | 8001 |
| SGLang | `sglang.compose.example.yaml` | `sglang.env.example` | 8002 |
| TensorRT-LLM | `tensorrt-llm.compose.example.yaml` | `tensorrt-llm.env.example` | 8000 |

`test` starts the selected profile, waits for its exact model identity, runs
bounded stream, non-stream, inventory-stability, and concurrency checks
directly and through AX Serving, writes JSON evidence, and cleans up:

```bash
scripts/qualification/runtime/run_compose_profile.sh test \
  --runtime vllm \
  --env-file .env.vllm \
  --stability-seconds 45 \
  --output-dir target/runtime-qualification
```

The wrapper allow-lists runtime names, rejects mutable image tags and model
revisions, never sources the environment file, and does not print
credentials. It requires `uv` 0.12.2 and Python 3.12 for the qualification
interpreter boundary. It accepts Compose v2 or newer through either the
`docker compose` plugin or the standalone `docker-compose` binary.
Pass `--keep` only when you intentionally want to inspect a test stack after
qualification. A successful direct compatibility run is **not** Dynamo-domain
certification. Production NVIDIA deployments remain the separately pinned
`ax-dynamo-adapter` topology.

The qualification runner intentionally treats an inference `429`, capacity `503`, timeout, or
invalid completion as evidence rather than hiding it behind unbounded retries. For application
load behavior and overload tuning, see the
[service tuning guide](../../docs/perf/service-tuning.md).

Jetson Thor uses the distinct native TensorRT Edge-LLM path documented in
[`deploy/thor`](../thor/README.md); do not use the PC TensorRT-LLM profile as a
Thor substitute.

On macOS, AX Engine remains a native host process. Use
`AXS_NODE_RUNTIME_URL=http://host.docker.internal:PORT` from the agent container,
or run the agent natively with `AXS_NODE_ADVERTISED_URL`.

## Notes

- Compose uses `AXS_ALLOW_NO_AUTH=true` for evaluation only.
- The default `legacy_compat` mode makes dynamic agent inventory routable. An
  explicit production catalog is intentionally not fabricated by this stack.
- Gateway readiness does not depend on runtime capacity.
- Images are built from portable features only (no AX Engine, Dynamo, CUDA, or MLX SDK).

Stop the base evaluation stack with:

```bash
docker compose -f deploy/compose/compose.yaml down
```
