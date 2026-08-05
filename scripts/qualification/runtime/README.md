# OpenAI-compatible runtime qualification

`smoke_openai_runtime.py` verifies a direct runtime or the same runtime through
AX Serving without treating the result as Dynamo-domain certification. It
checks:

- exact `/v1/models` identity;
- one non-streaming completion;
- one SSE completion ending in `[DONE]`;
- optional continuous inventory stability;
- a bounded concurrent no-retry request burst.

The runner has no third-party Python dependencies. The managed profiles pin
`uv` 0.12.2 and Python 3.12 to keep a predictable interpreter boundary:

```bash
uv run scripts/qualification/runtime/smoke_openai_runtime.py \
  --base-url http://127.0.0.1:18080 \
  --model tinyllama-trtllm \
  --runtime tensorrt_llm \
  --stability-seconds 45 \
  --requests 8 \
  --concurrency 4 \
  --allow-insecure-http \
  --output target/trtllm-runtime-smoke.json
```

Credentials are read only from the environment selected by `--api-key-env`;
never put them in the URL or command line.

For one NVIDIA PC, `run_compose_profile.sh` manages the pinned `vllm`,
`sglang`, and `tensorrt_llm` profiles. It rejects unknown runtimes and mutable
image/model pins, renders Compose before startup, prevents accidental
cross-profile GPU contention, qualifies both direct and gateway paths, and
cleans up by default:

```bash
scripts/qualification/runtime/run_compose_profile.sh test \
  --runtime sglang \
  --env-file .env.sglang \
  --output-dir target/runtime-qualification
```

For Jetson Thor, `install_tensorrt_edge_llm_thor.sh` pins NVIDIA TensorRT
Edge-LLM, `uv`, and the model revision. See
[`deploy/thor/README.md`](../../../deploy/thor/README.md). A successful direct
or runtime-agent compatibility result does not certify a separately operated
NVIDIA Dynamo topology.
