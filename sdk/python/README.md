# AX Serving Python SDK

The `ax-serving` package is the official Python client for
[AX Serving](https://github.com/defai-digital/ax-serving), a runtime-neutral
gateway for heterogeneous LLM infrastructure.

Use the SDK to call an AX Serving gateway through its OpenAI-compatible
REST/SSE API. An optional gRPC transport is available for the embedded macOS
compatibility service.

## Install

Add the portable REST client with [uv](https://docs.astral.sh/uv/):

```bash
uv add ax-serving
```

For the embedded gRPC compatibility client:

```bash
uv add "ax-serving[grpc]"
```

The base package intentionally does not install gRPC. It also does not bundle
the AX Serving server or an inference runtime.

## Chat completions

```python
from ax_serving import Client

client = Client(
    base_url="http://127.0.0.1:18080",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Explain heterogeneous LLM serving."},
    ],
    max_tokens=128,
)

print(response.choices[0].message.content)
```

If `api_key` is omitted, the client reads `AXS_API_KEY`.

## Streaming

```python
for chunk in client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Count from one to five."}],
    stream=True,
):
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)
```

## List models

```python
for model in client.models_list():
    print(model.id)
```

## Embedded gRPC compatibility

The gRPC transport is for the embedded macOS compatibility service. AX Serving
adapters and agents expose dedicated vLLM, SGLang, TensorRT-LLM, TensorRT
Edge-LLM, and other runtime domains to the portable gateway; applications
continue to use the REST client shown above.

```python
from ax_serving import GrpcClient

with GrpcClient() as client:
    print(client.health())
    for text in client.infer(
        "default",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
    ):
        print(text, end="", flush=True)
```

## Documentation

- [Quick start](https://github.com/defai-digital/ax-serving/blob/main/QUICKSTART.md)
- [Deployment and operations guides](https://github.com/defai-digital/ax-serving/tree/main/docs)
- [Issue tracker](https://github.com/defai-digital/ax-serving/issues)

AX Serving is licensed under the Apache License 2.0.
