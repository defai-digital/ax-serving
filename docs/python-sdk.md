# AX Serving Python SDK

## Overview

The official Python SDK (`ax_serving`) provides both a high-level OpenAI-compatible client and a low-level gRPC client for the AX Serving backend.

**Advantages:**
- **OpenAI compatibility**: Drop-in replacement for `openai` library in most cases
- **gRPC support**: Ultra-low latency via Unix domain sockets or TCP
- **Type safety**: Full type hints and dataclasses for responses
- **Streaming support**: Both sync and async-friendly iterators
- **Model management**: Load/unload/list models programmatically
- **Metrics & health**: Easy access to scheduler, cache, and thermal metrics

## Installation

```bash
# From the project root
cd sdk/python
pip install -e ".[dev]"

# Or install from PyPI once published:
# pip install ax-serving
```

## Quick Examples

### 1. Basic Chat Completion (REST)

```python
from ax_serving import Client

client = Client(base_url="http://127.0.0.1:18080")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the advantages of Rust over Python for serving LLMs."}
    ],
    max_tokens=200,
    temperature=0.7,
)

print(response.choices[0].message.content)
print("Usage:", response.usage)
```

### 2. Streaming Response

```python
from ax_serving import Client

client = Client(base_url="http://127.0.0.1:18080")

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

### 3. Using gRPC for Maximum Performance

```python
from ax_serving import GrpcClient, Client

# Option A: High-level OpenAI interface with gRPC backend
client = Client(grpc_socket="/tmp/ax-serving.sock")

# Option B: Low-level gRPC client
with GrpcClient(socket="/tmp/ax-serving.sock") as grpc:
    result = grpc.infer_full(
        model_id="default",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        temperature=0.0,
    )
    print("Response:", result.text)
    if result.metrics:
        print("Prefill t/s:", result.metrics.prefill_tok_per_sec)
```

### 4. Model Management

```python
from ax_serving import Client

client = Client(base_url="http://127.0.0.1:18080")

# List loaded models
models = client.models_list()
print("Loaded models:", [m.id for m in models])

# Load a new model (via REST or gRPC)
# client.load_model(...) is available on GrpcClient
```

### 5. Health and Metrics

```python
from ax_serving import Client

client = Client(base_url="http://127.0.0.1:18080")

# Health and metrics are currently only on GrpcClient
from ax_serving import GrpcClient

with GrpcClient() as g:
    health = g.health()
    print("Status:", health.status)
    print("Models:", health.model_ids)
    print("Thermal:", health.thermal_state)

    metrics = g.get_metrics()
    print("Cache errors:", metrics.system.errors if hasattr(metrics.system, 'errors') else "N/A")
```

## Advanced Usage

### Cache Control

```python
response = client.chat.completions.create(
    model="default",
    messages=[...],
    cache="enable",      # or "disable"
    cache_ttl="30m",     # 30 minutes
)
```

### Error Handling

The SDK raises standard exceptions (`httpx.HTTPStatusError`, gRPC errors). Use `try/except` around calls.

### Context Manager

```python
with Client(grpc_socket="/tmp/ax-serving.sock") as client:
    # safe automatic cleanup of gRPC channel
    ...
```

## Advantages vs Raw HTTP/gRPC

- **Developer experience**: Matches OpenAI Python library API
- **Performance**: gRPC backend avoids HTTP overhead
- **Type safety**: Full dataclasses (`GenerationResult`, `ModelInfo`, etc.)
- **Maintenance**: Official support, stays in sync with backend changes
- **Multi-protocol**: Same client works with REST or gRPC transparently

## Configuration

Environment variables used by the SDK:
- `AXS_API_KEY` — automatically added as Bearer token
- Server-side cache, scheduler, and thermal settings affect behavior

See `QUICKSTART.md` for server setup and `sdk/python/pyproject.toml` for dependencies.

## Links

- [QUICKSTART.md](../QUICKSTART.md) — Full server setup
- [Python source](../sdk/python/ax_serving/)
- [Test scripts](../scripts/)
