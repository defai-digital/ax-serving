"""Shared dataclasses for ax-serving SDK responses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    id: str
    architecture: str = ""
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    embedding_dim: int = 0
    vocab_size: int = 0
    context_length: int = 0
    file_size_bytes: int = 0
    backend: str = "auto"

    @classmethod
    def _from_proto(cls, pb) -> "ModelInfo":
        backend_map = {0: "unspecified", 1: "cpu", 2: "metal", 3: "auto"}
        return cls(
            id=pb.id,
            architecture=pb.architecture,
            n_layers=pb.n_layers,
            n_heads=pb.n_heads,
            n_kv_heads=pb.n_kv_heads,
            embedding_dim=pb.embedding_dim,
            vocab_size=pb.vocab_size,
            context_length=pb.context_length,
            file_size_bytes=pb.file_size_bytes,
            backend=backend_map.get(pb.backend, "auto"),
        )


@dataclass
class GenerationMetrics:
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_tok_per_sec: float = 0.0
    decode_tok_per_sec: float = 0.0
    total_time_ms: int = 0


@dataclass
class GenerationResult:
    text: str
    finish_reason: str = "eos"
    metrics: GenerationMetrics | None = None


@dataclass
class HealthInfo:
    status: str
    model_ids: list[str] = field(default_factory=list)
    uptime_secs: int = 0
    thermal_state: str = "nominal"


@dataclass
class SystemMetrics:
    rss_bytes: int = 0
    peak_rss_bytes: int = 0
    thermal_state: str = "nominal"
    inflight_count: int = 0
    total_requests: int = 0
    rejected_requests: int = 0
    avg_queue_wait_us: int = 0


@dataclass
class MetricsInfo:
    system: SystemMetrics = field(default_factory=SystemMetrics)
    models: list[dict] = field(default_factory=list)
