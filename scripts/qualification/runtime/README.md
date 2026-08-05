# OpenAI-compatible runtime qualification

`smoke_openai_runtime.py` verifies a direct runtime or the same runtime through
AX Serving without treating the result as Dynamo-domain certification. It
checks:

- exact `/v1/models` identity;
- one non-streaming completion;
- one SSE completion ending in `[DONE]`;
- optional continuous inventory stability;
- a bounded concurrent no-retry request burst.

The runner has no third-party Python dependencies. `uv` still gives each
operator a predictable interpreter boundary:

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

`python3` can run the same script when `uv` is unavailable. Credentials are
read only from the environment selected by `--api-key-env`; never put them in
the URL or command line.

Run the direct TensorRT-LLM endpoint once without `--runtime`, then run the AX
Serving endpoint with `--runtime tensorrt_llm`. A successful direct or
runtime-agent compatibility result does not certify the separately operated
NVIDIA Dynamo topology.
