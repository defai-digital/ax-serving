#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Qualify one OpenAI-compatible runtime path without retrying generation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import ipaddress
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA_VERSION = "com.automatosx.ax-serving.openai-runtime-smoke.v1"
MAX_ERROR_CHARS = 512
UrlOpen = Callable[..., BinaryIO]


def bounded_error(error: BaseException | str) -> str:
    return str(error).replace("\r", " ").replace("\n", " ")[:MAX_ERROR_CHARS]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative and finite")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def normalize_base_url(raw: str, *, allow_insecure_http: bool) -> str:
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("base URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a query or fragment")
    if parsed.scheme == "http" and not allow_insecure_http:
        host = parsed.hostname
        loopback = host == "localhost"
        with suppress(ValueError):
            loopback = loopback or ipaddress.ip_address(host).is_loopback
        if not loopback:
            raise ValueError(
                "plaintext HTTP is restricted to loopback; use HTTPS or --allow-insecure-http"
            )
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path = f"{path}/v1"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def request_headers(api_key: str | None, *, streaming: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream" if streaming else "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def open_request(
    url: str,
    *,
    payload: Mapping[str, Any] | None,
    api_key: str | None,
    timeout: float,
    streaming: bool = False,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> BinaryIO:
    request = urllib.request.Request(
        url,
        data=(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        ),
        headers=request_headers(api_key, streaming=streaming),
        method="POST" if payload is not None else "GET",
    )
    # Deliberately one transport call. Ambiguous generation failures are not
    # replayed because a retry could duplicate token execution.
    return urlopen(request, timeout=timeout)


def verify_model_identity(
    base_url: str,
    *,
    model: str,
    api_key: str | None,
    timeout: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> int:
    with open_request(
        f"{base_url}/models",
        payload=None,
        api_key=api_key,
        timeout=timeout,
        urlopen=urlopen,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("/v1/models response is missing its data array")
    identities = {
        item.get("id")
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model not in identities:
        raise ValueError(f"endpoint does not advertise the exact model identity {model!r}")
    return len(identities)


def build_payload(
    *,
    model: str,
    runtime: str | None,
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    if runtime:
        payload["runtime"] = runtime
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def validate_non_stream_response(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("completion response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("completion choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("completion choice is missing its message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("completion response must contain non-empty text")
    return content


def iter_sse_data(lines: Iterable[bytes]) -> Iterable[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if separator and value.startswith(" "):
            value = value[1:]
        data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)


def validate_stream_response(response: BinaryIO) -> tuple[str, int]:
    parts: list[str] = []
    chunks = 0
    done = False
    for data in iter_sse_data(response):
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("SSE data must contain a JSON object")
        chunks += 1
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("stream choices must be an array")
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("stream choice must be an object")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("stream delta must be an object")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("stream content delta must be a string")
                parts.append(content)
    text = "".join(parts)
    if not done:
        raise ValueError("stream ended before [DONE]")
    if not text:
        raise ValueError("stream response must contain non-empty text")
    return text, chunks


def execute_generation(
    base_url: str,
    *,
    payload: Mapping[str, Any],
    api_key: str | None,
    timeout: float,
    stream: bool,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    started = time.perf_counter()
    with open_request(
        f"{base_url}/chat/completions",
        payload=payload,
        api_key=api_key,
        timeout=timeout,
        streaming=stream,
        urlopen=urlopen,
    ) as response:
        if stream:
            text, chunks = validate_stream_response(response)
        else:
            text = validate_non_stream_response(
                json.loads(response.read().decode("utf-8"))
            )
            chunks = 0
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "stream_chunks": chunks,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime")
    parser.add_argument("--prompt", default="Reply with one short health-check token.")
    parser.add_argument("--max-tokens", type=positive_int, default=16)
    parser.add_argument("--requests", type=positive_int, default=8)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--timeout-seconds", type=positive_float, default=120.0)
    parser.add_argument(
        "--stability-seconds",
        type=non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--stability-interval-seconds",
        type=positive_float,
        default=5.0,
    )
    parser.add_argument("--api-key-env", default="AXS_API_KEY")
    parser.add_argument("--allow-insecure-http", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = normalize_base_url(
        args.base_url,
        allow_insecure_http=args.allow_insecure_http,
    )
    api_key = os.environ.get(args.api_key_env) or None
    advertised_models = verify_model_identity(
        base_url,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout_seconds,
    )

    common = {
        "model": args.model,
        "runtime": args.runtime,
        "max_tokens": args.max_tokens,
    }
    non_stream = execute_generation(
        base_url,
        payload=build_payload(**common, prompt=args.prompt, stream=False),
        api_key=api_key,
        timeout=args.timeout_seconds,
        stream=False,
    )
    stream = execute_generation(
        base_url,
        payload=build_payload(**common, prompt=args.prompt, stream=True),
        api_key=api_key,
        timeout=args.timeout_seconds,
        stream=True,
    )

    stability_checks = 1
    stability_started = time.monotonic()
    stability_deadline = stability_started + args.stability_seconds
    while time.monotonic() < stability_deadline:
        time.sleep(
            min(
                args.stability_interval_seconds,
                max(0.0, stability_deadline - time.monotonic()),
            )
        )
        verify_model_identity(
            base_url,
            model=args.model,
            api_key=api_key,
            timeout=args.timeout_seconds,
        )
        stability_checks += 1

    def burst_request(index: int) -> dict[str, Any]:
        payload = build_payload(
            **common,
            prompt=f"{args.prompt} Probe {index}.",
            stream=False,
        )
        return execute_generation(
            base_url,
            payload=payload,
            api_key=api_key,
            timeout=args.timeout_seconds,
            stream=False,
        )

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.concurrency, args.requests)
    ) as executor:
        future_indices = {
            executor.submit(burst_request, index): index
            for index in range(args.requests)
        }
        for future in concurrent.futures.as_completed(future_indices):
            index = future_indices[future]
            try:
                successes.append(future.result())
            except (
                OSError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as error:
                failures.append(
                    {
                        "request": index,
                        "error_type": type(error).__name__,
                        "error": bounded_error(error),
                    }
                )

    latencies = [result["elapsed_seconds"] for result in successes]
    summary = {
        "schema": SCHEMA_VERSION,
        "status": "pass" if not failures else "fail",
        "endpoint": base_url,
        "model": args.model,
        "runtime_hint": args.runtime,
        "api_key_configured": api_key is not None,
        "generation_retry_policy": "never",
        "advertised_model_count": advertised_models,
        "non_stream": non_stream,
        "stream": stream,
        "stability": {
            "requested_seconds": args.stability_seconds,
            "elapsed_seconds": time.monotonic() - stability_started,
            "inventory_checks": stability_checks,
        },
        "burst": {
            "requests": args.requests,
            "concurrency": min(args.concurrency, args.requests),
            "successes": len(successes),
            "failures": failures,
            "latency_seconds": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies) if latencies else None,
            },
        },
    }
    if args.output is not None:
        atomic_write_json(args.output, summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except (
        OSError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ) as error:
        print(f"runtime smoke setup failed: {bounded_error(error)}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
