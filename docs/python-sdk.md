# AX Serving Python client

The `sdk/python` package provides a small synchronous REST client for the
portable gateway and a separate gRPC client for the embedded compatibility
service.

It is not a complete replacement for the upstream `openai` package. It exposes
the subset implemented and tested in this repository; unknown REST generation
options are forwarded to the gateway.

## Install for development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e 'sdk/python[dev]'
pytest -q sdk/python/tests
```

Do not use a PyPI command as release evidence until the package is present and
verified in that channel.

Portable REST users install only the base package:

```bash
pip install ax-serving
```

## Portable REST/SSE

```python
from ax_serving import Client

client = Client(
    base_url="http://127.0.0.1:18080",
    api_key="public-client-key",
)

response = client.chat.completions.create(
    model="logical/model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

Streaming:

```python
for chunk in client.chat.completions.create(
    model="logical/model",
    messages=[{"role": "user", "content": "Count to five"}],
    max_tokens=32,
    stream=True,
):
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)
```

When `api_key` is omitted, the client reads `AXS_API_KEY`. The public key is
for inference only; admin APIs require a separate operator client and
`AXS_ADMIN_API_KEY`.

## Embedded gRPC compatibility

`GrpcClient`, `grpc_socket`, and `grpc_port` target `ax.serving.v1`. That
service is available only when the macOS embedded server is built with
`embedded-compat`.

```bash
pip install 'ax-serving[grpc]'
```

```python
from ax_serving import GrpcClient

with GrpcClient(socket="/tmp/ax-serving.sock") as client:
    result = client.infer_full(
        model_id="local-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
    )
```

Do not use gRPC v1 as a hybrid gateway protocol. It carries local model paths,
backend-specific controls, and token-ID semantics that cannot be mapped
losslessly across AX Engine and CUDA runtimes.

## Errors and timeouts

REST calls raise `httpx.HTTPStatusError`; gRPC calls raise gRPC exceptions.
The current convenience client uses a 120-second HTTP timeout. Applications
with longer generation requirements should use a directly configured HTTP
client until timeout configuration is exposed by this SDK.

Gateway-generated errors include an AX machine code, request ID, retryability,
and phase. Applications must not retry solely because a status is `5xx`; follow
the returned contract and remember that the gateway itself performs at most
one safe pre-commit retry.
