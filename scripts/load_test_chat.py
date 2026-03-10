#!/usr/bin/env python3
"""
Service-level load test for ax-serving /v1/chat/completions.

Usage example:
  python3 scripts/load_test_chat.py \
    --url http://127.0.0.1:18080/v1/chat/completions \
    --model default \
    --requests 200 \
    --concurrency 8 \
    --max-tokens 64 \
    --prompt-tokens 39
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List


@dataclass
class RequestResult:
    ok: bool
    status: int
    latency_ms: float
    error: str | None = None


def build_prompt(n_tokens: int) -> str:
    # Approximate token count with short repeated words for consistent load shape.
    words = ["hello"] * max(1, n_tokens)
    return " ".join(words)


def do_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> RequestResult:
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            _ = resp.read()
            ms = (time.perf_counter() - t0) * 1000.0
            return RequestResult(ok=200 <= resp.status < 300, status=resp.status, latency_ms=ms)
    except urllib.error.HTTPError as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(ok=False, status=e.code, latency_ms=ms, error=str(e))
    except Exception as e:  # pylint: disable=broad-except
        ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(ok=False, status=0, latency_ms=ms, error=str(e))


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="ax-serving chat load test")
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions")
    parser.add_argument("--model", default="default")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=39)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_tokens)
    results: List[RequestResult] = []

    print(
        f"Running load test: requests={args.requests}, concurrency={args.concurrency}, "
        f"prompt_tokens~{args.prompt_tokens}, max_tokens={args.max_tokens}"
    )

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(
                do_request,
                args.url,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout_s,
            )
            for _ in range(args.requests)
        ]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    total_s = time.perf_counter() - t0

    latencies = sorted([r.latency_ms for r in results if r.ok])
    ok = sum(1 for r in results if r.ok)
    errors = len(results) - ok
    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    rps = len(results) / total_s if total_s > 0 else 0.0
    err_rate = (errors / len(results)) * 100.0 if results else 0.0

    print("\n=== Summary ===")
    print(f"Total time: {total_s:.2f}s")
    print(f"RPS: {rps:.2f}")
    print(f"Success: {ok}/{len(results)}")
    print(f"Error rate: {err_rate:.2f}%")
    print(f"Status counts: {status_counts}")

    if latencies:
        print("\n=== Latency (successful requests) ===")
        print(f"mean: {statistics.mean(latencies):.1f} ms")
        print(f"p50:  {percentile(latencies, 0.50):.1f} ms")
        print(f"p95:  {percentile(latencies, 0.95):.1f} ms")
        print(f"p99:  {percentile(latencies, 0.99):.1f} ms")
        print(f"max:  {latencies[-1]:.1f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

