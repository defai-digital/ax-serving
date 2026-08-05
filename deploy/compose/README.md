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
interpreter boundary.
Pass `--keep` only when you intentionally want to inspect a test stack after
qualification. A successful direct compatibility run is **not** Dynamo-domain
certification. Production NVIDIA deployments remain the separately pinned
`ax-dynamo-adapter` topology.

Jetson Thor uses the distinct native TensorRT Edge-LLM path documented in
[`deploy/thor`](../thor/README.md); do not use the PC TensorRT-LLM profile as a
Thor substitute.

On macOS, AX Engine remains a native host process. Use
`AXS_NODE_RUNTIME_URL=http://host.docker.internal:<port>` from the agent container,
or run the agent natively with `AXS_NODE_ADVERTISED_URL`.

## Notes

- Compose uses `AXS_ALLOW_NO_AUTH=true` for evaluation only.
- The default `legacy_compat` mode makes dynamic agent inventory routable. An
  explicit production catalog is intentionally not fabricated by this stack.
- Gateway readiness does not depend on runtime capacity.
- Images are built from portable features only (no AX Engine, Dynamo, CUDA, or MLX SDK).
