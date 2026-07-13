"""ax-serving Python SDK.

Provides two interfaces:
- :class:`Client` — a tested OpenAI-style REST subset for the portable gateway,
  or embedded gRPC v1 when explicitly configured.
- :class:`GrpcClient` — low-level embedded-compatibility gRPC client.

Quick start::

    from ax_serving import Client

    # Portable gateway REST/SSE.
    c = Client(base_url="http://127.0.0.1:18080")
    resp = c.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.choices[0].message.content)

    # Embedded macOS compatibility service only (requires ax-serving[grpc]).
    from ax_serving import GrpcClient
    g = GrpcClient()
    print(g.health())
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from ._openai import Client
from .types import GenerationMetrics, GenerationResult, HealthInfo, MetricsInfo, ModelInfo

if TYPE_CHECKING:
    from ._grpc import GrpcClient

try:
    __version__ = version("ax-serving")
except PackageNotFoundError:
    __version__ = "0+unknown"


def __getattr__(name: str) -> Any:
    if name != "GrpcClient":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from ._grpc import GrpcClient
    except ModuleNotFoundError as error:
        raise ImportError(
            "GrpcClient is embedded compatibility only; install 'ax-serving[grpc]'"
        ) from error
    return GrpcClient

__all__ = [
    "Client",
    "GrpcClient",
    "ModelInfo",
    "GenerationResult",
    "GenerationMetrics",
    "HealthInfo",
    "MetricsInfo",
    "__version__",
]
