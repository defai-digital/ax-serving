#!/usr/bin/env python3
"""
Comprehensive cache integration tests for ax-serving + Valkey.

Scenarios:
1) miss -> hit -> disable bypass
2) streaming bypass
3) invalid cache_ttl (400)
4) TTL max cap verification
5) key sensitivity to generation params
6) restart persistence (cache survives server restart)
7) cache outage fallback (requests still succeed)
8) cross-model cache isolation (Qwen vs Llama)

Usage:
  ./scripts/test_cache_comprehensive.py
  ./scripts/test_cache_comprehensive.py --qwen <path> --llama <path>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_http(url: str, timeout_s: int = 240) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"timeout waiting for {url}")


def http_json(url: str, payload: dict, timeout_s: int = 180) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class Valkey:
    def __init__(self, server_bin: Path, cli_bin: Path):
        self.server_bin = server_bin
        self.cli_bin = cli_bin
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [
                str(self.server_bin),
                "--bind",
                "127.0.0.1",
                "--port",
                "6379",
                "--save",
                "",
                "--appendonly",
                "no",
            ],
            stdout=open("/tmp/valkey-cache-tests.log", "w"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(120):
            try:
                if "PONG" in self.cli("PING"):
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("valkey failed to start")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None

    def cli(self, *args: str) -> str:
        return subprocess.check_output(
            [str(self.cli_bin), "-h", "127.0.0.1", "-p", "6379", *args],
            text=True,
        ).strip()

    def flushall(self) -> None:
        self.cli("FLUSHALL")

    def keys(self, pattern: str) -> set[str]:
        raw = self.cli("--raw", "KEYS", pattern)
        return {k for k in raw.splitlines() if k}


class AxServer:
    def __init__(
        self,
        ax_bin: Path,
        config: Path,
        model: Path,
        port: int,
        grpc_socket: str,
        extra_env: dict[str, str] | None = None,
        log_tag: str = "ax-cache-test",
    ):
        self.ax_bin = ax_bin
        self.config = config
        self.model = model
        self.port = port
        self.grpc_socket = grpc_socket
        self.extra_env = extra_env or {}
        self.log_tag = log_tag
        self.proc: subprocess.Popen[str] | None = None

    def start(self, cwd: Path) -> None:
        env = os.environ.copy()
        env["AXS_GRPC_SOCKET"] = self.grpc_socket
        env.update(self.extra_env)
        self.proc = subprocess.Popen(
            [
                str(self.ax_bin),
                "serve",
                "-m",
                str(self.model),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--config",
                str(self.config),
            ],
            cwd=str(cwd),
            env=env,
            stdout=open(f"/tmp/{self.log_tag}.log", "w"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_http(f"http://127.0.0.1:{self.port}/health", timeout_s=360)
        if self.proc.poll() is not None:
            raise RuntimeError(f"ax server exited early: {self.log_tag}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None

    def post(self, body: dict, timeout_s: int = 180) -> tuple[int, str]:
        return http_json(f"http://127.0.0.1:{self.port}/v1/chat/completions", body, timeout_s)

    def cache_metrics(self) -> dict:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/v1/metrics", timeout=5
        ) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return payload["cache"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ax-bin", default=str(root / "target/release/ax-llama"), help="ax-llama binary"
    )
    parser.add_argument(
        "--config",
        default=str(root / "config/serving.example.yaml"),
        help="serve config file",
    )
    parser.add_argument(
        "--qwen",
        default=str(root / "models/Qwen3-8B-Q4_K_M.gguf"),
        help="Qwen model path",
    )
    parser.add_argument(
        "--llama",
        default=str(root / "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
        help="Llama model path",
    )
    parser.add_argument(
        "--valkey-server",
        default="/opt/homebrew/opt/valkey/bin/valkey-server",
        help="valkey-server binary",
    )
    parser.add_argument(
        "--valkey-cli",
        default="/opt/homebrew/opt/valkey/bin/valkey-cli",
        help="valkey-cli binary",
    )
    return parser.parse_args()


def run_qwen_matrix(root: Path, ax_bin: Path, cfg: Path, qwen_model: Path, valkey: Valkey) -> None:
    print("[QWEN] full matrix")
    valkey.flushall()
    srv = AxServer(
        ax_bin,
        cfg,
        qwen_model,
        port=19100,
        grpc_socket="/tmp/ax-serving-cache-qwen.sock",
        log_tag="ax-cache-qwen",
    )
    srv.start(root)
    try:
        m0 = srv.cache_metrics()
        assert_true(m0 == {"enabled": True, "hits": 0, "misses": 0, "writes": 0, "errors": 0}, "baseline cache metrics mismatch")

        base = {
            "model": "default",
            "messages": [{"role": "user", "content": "cache matrix prompt"}],
            "stream": False,
            "max_tokens": 16,
            "cache": "enable",
            "cache_ttl": "1h",
        }

        # miss + hit
        s1, b1 = srv.post(base)
        s2, b2 = srv.post(base)
        j1, j2 = json.loads(b1), json.loads(b2)
        m1 = srv.cache_metrics()
        assert_true(s1 == 200 and s2 == 200, "enable requests failed")
        assert_true(j1["id"] == j2["id"], "second enable request should be cache hit")
        assert_true(m1["misses"] == 1 and m1["hits"] == 1 and m1["writes"] == 1, "miss/hit/write metrics mismatch")

        # disable bypass
        bypass = dict(base)
        bypass["cache"] = "disable"
        s3, b3 = srv.post(bypass)
        j3 = json.loads(b3)
        m2 = srv.cache_metrics()
        assert_true(s3 == 200, "disable request failed")
        assert_true(j3["id"] != j1["id"], "disable should bypass cache")
        assert_true(m2 == m1, "disable should not alter cache metrics")

        # streaming bypass
        stream = dict(base)
        stream["stream"] = True
        s4, b4 = srv.post(stream, timeout_s=120)
        m3 = srv.cache_metrics()
        assert_true(s4 == 200 and "chat.completion.chunk" in b4, "stream response mismatch")
        assert_true(m3 == m2, "stream should bypass cache")

        # invalid ttl
        bad = dict(base)
        bad["messages"] = [{"role": "user", "content": "bad ttl"}]
        bad["cache_ttl"] = "invalid-ttl"
        s5, b5 = srv.post(bad)
        assert_true(s5 == 400 and "invalid cache_ttl" in b5, "invalid cache_ttl should return 400")

        # ttl cap
        keys_before = valkey.keys("axs:chat:v1:*")
        cap = dict(base)
        cap["messages"] = [{"role": "user", "content": "ttl cap"}]
        cap["cache_ttl"] = "365d"
        s6, _ = srv.post(cap)
        keys_after = valkey.keys("axs:chat:v1:*")
        new_keys = keys_after - keys_before
        assert_true(s6 == 200 and len(new_keys) == 1, "ttl cap request did not write one key")
        ttl = int(valkey.cli("TTL", next(iter(new_keys))))
        assert_true(2_500_000 <= ttl <= 2_592_000, f"ttl cap mismatch: {ttl}")

        # key sensitivity
        sens1 = dict(base)
        sens1["messages"] = [{"role": "user", "content": "param sensitivity"}]
        sens1["max_tokens"] = 8
        sens2 = dict(sens1)
        sens2["max_tokens"] = 12
        pre = srv.cache_metrics()
        assert_true(srv.post(sens1)[0] == 200 and srv.post(sens2)[0] == 200, "param sensitivity requests failed")
        post = srv.cache_metrics()
        assert_true(post["misses"] >= pre["misses"] + 2, "params should create separate misses")

        # restart persistence
        persist = dict(base)
        persist["messages"] = [{"role": "user", "content": "restart persistence"}]
        pid = json.loads(srv.post(persist)[1])["id"]
        srv.stop()
        srv = AxServer(
            ax_bin,
            cfg,
            qwen_model,
            port=19100,
            grpc_socket="/tmp/ax-serving-cache-qwen-r.sock",
            log_tag="ax-cache-qwen-r",
        )
        srv.start(root)
        pid2 = json.loads(srv.post(persist)[1])["id"]
        assert_true(pid == pid2, "cache should persist across server restart")

    finally:
        srv.stop()


def run_outage_case(root: Path, ax_bin: Path, cfg: Path, qwen_model: Path) -> None:
    print("[QWEN] outage fallback")
    srv = AxServer(
        ax_bin,
        cfg,
        qwen_model,
        port=19101,
        grpc_socket="/tmp/ax-serving-cache-outage.sock",
        extra_env={"AXS_CACHE_URL": "redis://127.0.0.1:6399"},
        log_tag="ax-cache-outage",
    )
    srv.start(root)
    try:
        pre = srv.cache_metrics()
        body = {
            "model": "default",
            "messages": [{"role": "user", "content": "cache outage fallback"}],
            "stream": False,
            "max_tokens": 16,
            "cache": "enable",
            "cache_ttl": "1h",
        }
        status, _ = srv.post(body)
        post = srv.cache_metrics()
        assert_true(status == 200, "request should succeed even if cache backend is unavailable")
        assert_true(post["errors"] >= pre["errors"] + 1, "cache errors should increment in outage")
    finally:
        srv.stop()


def run_cross_model_isolation(
    root: Path, ax_bin: Path, cfg: Path, qwen_model: Path, llama_model: Path, valkey: Valkey
) -> None:
    print("[QWEN/LLAMA] cross-model isolation")
    valkey.flushall()
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": "cross model cache isolation"}],
        "stream": False,
        "max_tokens": 20,
        "cache": "enable",
        "cache_ttl": "1h",
    }

    q = AxServer(
        ax_bin,
        cfg,
        qwen_model,
        port=19102,
        grpc_socket="/tmp/ax-serving-cache-qwen-iso.sock",
        log_tag="ax-cache-qwen-iso",
    )
    q.start(root)
    try:
        qid1 = json.loads(q.post(body)[1])["id"]
        qid2 = json.loads(q.post(body)[1])["id"]
        qm = q.cache_metrics()
        assert_true(qid1 == qid2, "qwen second request should hit cache")
        assert_true(qm["misses"] == 1 and qm["hits"] == 1 and qm["writes"] == 1, "qwen isolation metrics mismatch")
    finally:
        q.stop()

    l = AxServer(
        ax_bin,
        cfg,
        llama_model,
        port=19103,
        grpc_socket="/tmp/ax-serving-cache-llama-iso.sock",
        log_tag="ax-cache-llama-iso",
    )
    l.start(root)
    try:
        lid1 = json.loads(l.post(body)[1])["id"]
        lid2 = json.loads(l.post(body)[1])["id"]
        lm = l.cache_metrics()
        assert_true(lid1 == lid2, "llama second request should hit cache")
        assert_true(lid1 != qid1, "llama should not reuse qwen cached response")
        assert_true(lm["misses"] == 1 and lm["hits"] == 1 and lm["writes"] == 1, "llama isolation metrics mismatch")
    finally:
        l.stop()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    ax_bin = Path(args.ax_bin)
    cfg = Path(args.config)
    qwen = Path(args.qwen)
    llama = Path(args.llama)
    vks = Path(args.valkey_server)
    vkc = Path(args.valkey_cli)

    for p in [ax_bin, cfg, qwen, llama, vks, vkc]:
        if not p.exists():
            print(f"missing required path: {p}", file=sys.stderr)
            return 2

    subprocess.run(["killall", "ax-llama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["killall", "valkey-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    valkey = Valkey(vks, vkc)
    try:
        valkey.start()
        run_qwen_matrix(root, ax_bin, cfg, qwen, valkey)
        run_outage_case(root, ax_bin, cfg, qwen)
        valkey.stop()
        valkey.start()
        run_cross_model_isolation(root, ax_bin, cfg, qwen, llama, valkey)
    finally:
        valkey.stop()
        subprocess.run(["killall", "ax-llama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("ALL CACHE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
