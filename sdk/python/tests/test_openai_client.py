from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from ax_serving import Client


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self._lines


def test_rest_chat_uses_explicit_api_key_and_forwards_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post))

    client = Client(base_url="http://127.0.0.1:18080/", api_key="secret")
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hello"}],
        cache="enable",
        cache_ttl="30m",
        top_k=4,
    )

    assert response.choices[0].message.content == "ok"
    assert captured["url"] == "http://127.0.0.1:18080/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["json"]["cache"] == "enable"
    assert captured["json"]["cache_ttl"] == "30m"
    assert captured["json"]["top_k"] == 4


def test_rest_chat_does_not_send_default_top_k_zero(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post))

    client = Client(base_url="http://127.0.0.1:18080")
    client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert "top_k" not in captured["json"]


def test_rest_stream_uses_api_key(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        captured["method"] = method
        captured["url"] = url
        captured.update(kwargs)
        return FakeStreamResponse(
            [
                'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
                "data: [DONE]",
            ]
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(stream=stream))

    client = Client(base_url="http://127.0.0.1:18080", api_key="secret")
    chunks = list(
        client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert [chunk.choices[0].delta.content for chunk in chunks] == ["hi"]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:18080/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer secret"}


def test_models_list_uses_axs_api_key_env(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def get(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": "default",
                        "object": "model",
                        "created": 0,
                        "owned_by": "ax-serving",
                    }
                ],
            }
        )

    monkeypatch.setenv("AXS_API_KEY", "from-env")
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=get))

    client = Client(base_url="http://127.0.0.1:18080")
    models = client.models_list()

    assert [model.id for model in models] == ["default"]
    assert captured["headers"] == {"Authorization": "Bearer from-env"}


def test_grpc_chat_ignores_rest_only_options() -> None:
    captured: dict[str, Any] = {}

    class FakeGrpc:
        def infer_full(
            self,
            model_id: str,
            messages: list[dict[str, str]],
            **kwargs: Any,
        ) -> Any:
            captured["model_id"] = model_id
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                text="ok",
                metrics=SimpleNamespace(prefill_tokens=1, decode_tokens=2),
            )

    client = Client(grpc_port=50051)
    client._grpc = FakeGrpc()

    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hello"}],
        cache="enable",
        cache_ttl="30m",
    )

    assert response.choices[0].message.content == "ok"
    assert captured["model_id"] == "default"
    assert "cache" not in captured["kwargs"]
    assert "cache_ttl" not in captured["kwargs"]
