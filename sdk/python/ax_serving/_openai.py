"""High-level OpenAI-SDK-compatible interface backed by gRPC or REST."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any

from .types import ModelInfo


class _CompletionChunk:
    """Minimal OpenAI ChatCompletionChunk-compatible object."""

    def __init__(self, text: str, model: str, finish_reason: str | None = None) -> None:
        self.model = model
        self.choices = [_ChunkChoice(text, finish_reason)]


class _ChunkChoice:
    def __init__(self, text: str, finish_reason: str | None) -> None:
        self.delta = _Delta(text)
        self.finish_reason = finish_reason


class _Delta:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class _ChatCompletionResponse:
    """Minimal OpenAI ChatCompletion-compatible object."""

    def __init__(self, text: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.model = model
        self.choices = [_Choice(text)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Completions:
    """Implements chat.completions.create()."""

    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 0,
        repeat_penalty: float = 1.1,
        seed: int = 0,
        **_: Any,
    ) -> _ChatCompletionResponse | Iterator[_CompletionChunk]:
        """Create a chat completion.

        Args:
            model: ID of a loaded model.
            messages: Chat messages ``[{"role": ..., "content": ...}]``.
            stream: If True, returns an iterator of chunks.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling (0 = disabled).
            repeat_penalty: Repetition penalty.
            seed: Random seed.
        """
        kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            seed=seed,
        )

        if self._client._grpc is not None:
            return self._create_grpc(model, messages, stream, **kwargs)
        return self._create_rest(model, messages, stream, **kwargs)

    def _create_grpc(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> _ChatCompletionResponse | Iterator[_CompletionChunk]:
        grpc_client = self._client._grpc

        if stream:
            def _stream() -> Generator[_CompletionChunk, None, None]:
                for tok in grpc_client.infer(model, messages=messages, **kwargs):
                    yield _CompletionChunk(tok, model)
                yield _CompletionChunk("", model, finish_reason="stop")
            return _stream()

        result = grpc_client.infer_full(model, messages=messages, **kwargs)
        pt = result.metrics.prefill_tokens if result.metrics else 0
        ct = result.metrics.decode_tokens if result.metrics else 0
        return _ChatCompletionResponse(result.text, model, pt, ct)

    def _create_rest(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> _ChatCompletionResponse | Iterator[_CompletionChunk]:
        import httpx

        url = f"{self._client._base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 0.9),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
        }
        if kwargs.get("top_k"):
            payload["top_k"] = kwargs["top_k"]
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs["seed"]

        if stream:
            return self._rest_stream(url, payload, model)

        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return _ChatCompletionResponse(
            choice["message"]["content"],
            model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    def _rest_stream(
        self, url: str, payload: dict, model: str
    ) -> Iterator[_CompletionChunk]:
        import httpx

        with httpx.stream("POST", url, json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].lstrip()
                if not raw:
                    continue
                if raw.strip() == "[DONE]":
                    break
                import json
                data = json.loads(raw)
                choice = data["choices"][0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                finish = choice.get("finish_reason")
                yield _CompletionChunk(content, model, finish_reason=finish)


class _Chat:
    def __init__(self, client: "Client") -> None:
        self.completions = _Completions(client)


class Client:
    """OpenAI-SDK-compatible client for ax-serving.

    Backed by gRPC when *grpc_socket* or *grpc_port* is supplied,
    otherwise falls back to REST via *base_url*.

    Examples::

        # REST (default, matches OpenAI SDK interface)
        c = Client(base_url="http://localhost:18080")
        resp = c.chat.completions.create(
            model="llama3",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(resp.choices[0].message.content)

        # gRPC via UDS
        c = Client(grpc_socket="/tmp/ax-serving.sock")
        for chunk in c.chat.completions.create(
            model="llama3",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ):
            print(chunk.choices[0].delta.content, end="", flush=True)

        # gRPC via TCP
        c = Client(grpc_port=50051)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:18080",
        grpc_socket: str | None = None,
        grpc_port: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._grpc = None

        if grpc_socket is not None or grpc_port is not None:
            from ._grpc import GrpcClient
            self._grpc = GrpcClient(
                socket=grpc_socket or "/tmp/ax-serving.sock",
                host=None if grpc_socket else "localhost",
                port=grpc_port or 50051,
            )

        self.chat = _Chat(self)

    def models_list(self) -> list[ModelInfo]:
        """List loaded models (gRPC or REST)."""
        if self._grpc is not None:
            return self._grpc.list_models()

        import httpx
        resp = httpx.get(f"{self._base_url}/v1/models", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        return [ModelInfo(id=m["id"]) for m in data.get("data", [])]

    def close(self) -> None:
        if self._grpc is not None:
            self._grpc.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
