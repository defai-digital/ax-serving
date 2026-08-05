#!/usr/bin/env python3
"""Unit tests for the OpenAI-compatible runtime smoke runner."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).with_name("smoke_openai_runtime.py")
MODULE_SPEC = importlib.util.spec_from_file_location("smoke_openai_runtime", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


class OpenAiRuntimeSmokeTests(unittest.TestCase):
    def test_plaintext_remote_endpoint_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "restricted to loopback"):
            runner.normalize_base_url(
                "http://gpu.example.test:8000",
                allow_insecure_http=False,
            )
        self.assertEqual(
            runner.normalize_base_url(
                "http://127.0.0.1:8000",
                allow_insecure_http=False,
            ),
            "http://127.0.0.1:8000/v1",
        )

    def test_runtime_hint_is_optional_and_stream_requests_usage(self):
        direct = runner.build_payload(
            model="tiny",
            runtime=None,
            prompt="ok",
            max_tokens=4,
            stream=False,
        )
        self.assertNotIn("runtime", direct)
        gateway = runner.build_payload(
            model="tiny",
            runtime="tensorrt_llm",
            prompt="ok",
            max_tokens=4,
            stream=True,
        )
        self.assertEqual(gateway["runtime"], "tensorrt_llm")
        self.assertEqual(gateway["stream_options"], {"include_usage": True})

    def test_exact_model_identity_is_required(self):
        payload = json.dumps({"data": [{"id": "other"}]}).encode()
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            runner.verify_model_identity(
                "http://127.0.0.1:8000/v1",
                model="tiny",
                api_key=None,
                timeout=1.0,
                urlopen=lambda *_args, **_kwargs: FakeResponse(payload),
            )

    def test_stream_requires_content_and_done_marker(self):
        response = FakeResponse(
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        text, chunks = runner.validate_stream_response(response)
        self.assertEqual(text, "OK")
        self.assertEqual(chunks, 2)

        with self.assertRaisesRegex(ValueError, r"\[DONE\]"):
            runner.validate_stream_response(
                FakeResponse(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n')
            )

    def test_generation_transport_is_called_once(self):
        calls = []

        def fail_once(_request, *, timeout):
            calls.append(timeout)
            raise OSError("ambiguous transport failure")

        with self.assertRaisesRegex(OSError, "ambiguous"):
            runner.execute_generation(
                "http://127.0.0.1:8000/v1",
                payload={"model": "tiny"},
                api_key=None,
                timeout=3.0,
                stream=False,
                urlopen=fail_once,
            )
        self.assertEqual(calls, [3.0])

    def test_run_executes_stream_non_stream_and_concurrent_burst(self):
        args = Namespace(
            base_url="http://127.0.0.1:8000",
            model="tiny",
            runtime="tensorrt_llm",
            prompt="ok",
            max_tokens=4,
            requests=3,
            concurrency=2,
            timeout_seconds=1.0,
            stability_seconds=0.0,
            stability_interval_seconds=1.0,
            api_key_env="MISSING_TEST_API_KEY",
            allow_insecure_http=False,
            output=None,
        )
        result = {
            "elapsed_seconds": 0.01,
            "response_sha256": "0" * 64,
            "stream_chunks": 1,
        }
        with (
            mock.patch.object(runner, "verify_model_identity", return_value=1),
            mock.patch.object(
                runner,
                "execute_generation",
                return_value=result,
            ) as execute,
        ):
            summary = runner.run(args)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["burst"]["successes"], 3)
        self.assertEqual(execute.call_count, 5)


if __name__ == "__main__":
    unittest.main()
