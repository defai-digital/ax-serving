"""Low-level gRPC client wrapping AxServingService."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import grpc

from ._proto import ax_serving_pb2 as pb
from ._proto import ax_serving_pb2_grpc as pb_grpc
from .types import GenerationMetrics, GenerationResult, HealthInfo, MetricsInfo, ModelInfo, SystemMetrics

_THERMAL_MAP = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
_FINISH_MAP = {0: "unspecified", 1: "eos", 2: "max_tokens", 3: "stop_sequence"}


def _make_channel(socket: str | None, host: str | None, port: int) -> grpc.Channel:
    """Create a gRPC channel.

    Priority:
    1. TCP ``host:port`` when *host* is explicitly set (caller prefers TCP).
    2. UDS socket when the socket file exists and no host is specified.
    3. Fallback to ``localhost:port`` TCP.
    """
    if host is not None:
        target = f"{host}:{port}"
    elif socket and os.path.exists(socket):
        target = f"unix://{socket}"
    else:
        target = f"localhost:{port}"
    return grpc.insecure_channel(target)


class GrpcClient:
    """Synchronous gRPC client for ax-serving.

    Connection priority:
    1. UDS socket (if ``socket`` is set and the path exists)
    2. TCP ``host:port``

    Examples::

        # UDS (default)
        c = GrpcClient()
        print(c.health())

        # TCP
        c = GrpcClient(host="192.168.1.10", port=50051)

        # Context manager
        with GrpcClient() as c:
            for tok in c.infer("llama3", prompt="Hello"):
                print(tok, end="", flush=True)
    """

    def __init__(
        self,
        socket: str = "/tmp/ax-serving.sock",
        host: str | None = None,
        port: int = 50051,
        timeout: float = 30.0,
    ) -> None:
        self._socket = socket
        self._host = host
        self._port = port
        self._timeout = timeout
        self._channel: grpc.Channel | None = None
        self._stub: pb_grpc.AxServingServiceStub | None = None

    def _ensure_connected(self) -> pb_grpc.AxServingServiceStub:
        if self._stub is None:
            self._channel = _make_channel(self._socket, self._host, self._port)
            self._stub = pb_grpc.AxServingServiceStub(self._channel)
        return self._stub

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> "GrpcClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Model management ─────────────────────────────────────────────────────

    def load_model(
        self,
        path: str,
        model_id: str = "",
        context_length: int = 0,
        backend: str = "auto",
    ) -> ModelInfo:
        """Load a GGUF model and return its ModelInfo."""
        _backend_map = {"cpu": 1, "metal": 2, "auto": 3}
        stub = self._ensure_connected()
        resp = stub.LoadModel(
            pb.LoadModelRequest(
                model_path=path,
                model_id=model_id or os.path.splitext(os.path.basename(path))[0],
                backend=_backend_map.get(backend.lower(), 3),
                context_length=context_length,
            ),
            timeout=self._timeout,
        )
        return ModelInfo._from_proto(resp.info)

    def unload_model(self, model_id: str) -> None:
        """Unload a loaded model."""
        stub = self._ensure_connected()
        stub.UnloadModel(pb.UnloadModelRequest(model_id=model_id), timeout=self._timeout)

    def list_models(self) -> list[ModelInfo]:
        """List all currently loaded models."""
        stub = self._ensure_connected()
        resp = stub.ListModels(pb.ListModelsRequest(), timeout=self._timeout)
        return [ModelInfo._from_proto(m) for m in resp.models]

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer(
        self,
        model_id: str,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        *,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 512,
        seed: int = 0,
    ) -> Iterator[str]:
        """Stream generated text tokens.

        Yields decoded text pieces.  The final piece is an empty string when
        ``finished=True`` (which carries the metrics).

        Args:
            model_id: ID of a loaded model.
            prompt: Raw text prompt (exclusive with *messages*).
            messages: Chat messages as ``[{"role": ..., "content": ...}, ...]``.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling (0 = disabled).
            top_p: Top-p nucleus sampling.
            repeat_penalty: Repetition penalty.
            max_tokens: Maximum tokens to generate.
            seed: Random seed.
        """
        stub = self._ensure_connected()

        sampling = pb.SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        pb_messages = []
        if messages:
            pb_messages = [
                pb.ChatMessage(role=m["role"], content=m["content"]) for m in messages
            ]

        req = pb.InferRequest(
            model_id=model_id,
            prompt=prompt or "",
            messages=pb_messages,
            sampling=sampling,
            max_tokens=max_tokens,
        )

        for resp in stub.Infer(req, timeout=self._timeout):
            if not resp.finished:
                yield resp.text
            # On finish the caller can inspect metrics via infer_full()

    def infer_full(
        self,
        model_id: str,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Collect the full generation and return a :class:`GenerationResult`."""
        stub = self._ensure_connected()

        sampling = pb.SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 0),
            top_p=kwargs.get("top_p", 0.9),
            repeat_penalty=kwargs.get("repeat_penalty", 1.1),
            seed=kwargs.get("seed", 0),
        )

        pb_messages = [
            pb.ChatMessage(role=m["role"], content=m["content"])
            for m in (messages or [])
        ]

        req = pb.InferRequest(
            model_id=model_id,
            prompt=prompt or "",
            messages=pb_messages,
            sampling=sampling,
            max_tokens=kwargs.get("max_tokens", 512),
        )

        text_parts: list[str] = []
        finish_reason = "eos"
        gen_metrics: GenerationMetrics | None = None

        for resp in stub.Infer(req, timeout=self._timeout):
            if not resp.finished:
                text_parts.append(resp.text)
            else:
                finish_reason = _FINISH_MAP.get(resp.finish_reason, "eos")
                if resp.HasField("metrics"):
                    m = resp.metrics
                    gen_metrics = GenerationMetrics(
                        prefill_tokens=m.prefill_tokens,
                        decode_tokens=m.decode_tokens,
                        prefill_tok_per_sec=m.prefill_tok_per_sec,
                        decode_tok_per_sec=m.decode_tok_per_sec,
                        total_time_ms=m.total_time_ms,
                    )

        return GenerationResult(
            text="".join(text_parts),
            finish_reason=finish_reason,
            metrics=gen_metrics,
        )

    # ── Health & Metrics ──────────────────────────────────────────────────────

    def health(self) -> HealthInfo:
        """Return current server health."""
        stub = self._ensure_connected()
        resp = stub.Health(pb.HealthRequest(), timeout=self._timeout)
        status_map = {0: "unspecified", 1: "serving", 2: "not_serving"}
        return HealthInfo(
            status=status_map.get(resp.status, "unknown"),
            model_ids=list(resp.model_ids),
            uptime_secs=resp.uptime_secs,
            thermal_state=_THERMAL_MAP.get(resp.thermal_state, "nominal"),
        )

    def get_metrics(self) -> MetricsInfo:
        """Return system and per-model metrics."""
        stub = self._ensure_connected()
        resp = stub.GetMetrics(pb.GetMetricsRequest(), timeout=self._timeout)
        sys = resp.system
        system = SystemMetrics(
            rss_bytes=sys.rss_bytes,
            peak_rss_bytes=sys.peak_rss_bytes,
            thermal_state=_THERMAL_MAP.get(sys.thermal_state, "nominal"),
            inflight_count=sys.inflight_count,
            total_requests=sys.total_requests,
            rejected_requests=sys.rejected_requests,
            avg_queue_wait_us=sys.avg_queue_wait_us,
        )
        models = [
            {
                "model_id": m.model_id,
                "total_infer_requests": m.total_infer_requests,
                "avg_prefill_tok_per_sec": m.avg_prefill_tok_per_sec,
                "avg_decode_tok_per_sec": m.avg_decode_tok_per_sec,
            }
            for m in resp.models
        ]
        return MetricsInfo(system=system, models=models)
