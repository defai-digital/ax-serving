"""ax-serving Python SDK.

Provides two interfaces:
- :class:`Client` — OpenAI-SDK-compatible, backed by gRPC or REST.
- :class:`GrpcClient` — Low-level gRPC client.

Quick start::

    from ax_serving import Client, GrpcClient

    # OpenAI-compatible (REST by default, gRPC when socket/port given)
    c = Client(grpc_socket="/tmp/ax-serving.sock")
    resp = c.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.choices[0].message.content)

    # Raw gRPC
    g = GrpcClient()
    print(g.health())
"""

from .types import GenerationMetrics, GenerationResult, HealthInfo, MetricsInfo, ModelInfo
from ._grpc import GrpcClient
from ._openai import Client

__all__ = [
    "Client",
    "GrpcClient",
    "ModelInfo",
    "GenerationResult",
    "GenerationMetrics",
    "HealthInfo",
    "MetricsInfo",
]
