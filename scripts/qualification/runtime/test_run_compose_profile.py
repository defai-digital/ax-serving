#!/usr/bin/env python3
"""Unit tests for the allow-listed NVIDIA Compose profile runner."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_compose_profile.sh")
REPO_ROOT = SCRIPT.parents[3]
EXAMPLE = REPO_ROOT / "deploy" / "compose" / "vllm.env.example"


class ComposeProfileRunnerTests(unittest.TestCase):
    def run_profile(
        self, *arguments: str, env_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "runtime.env"
            env_file.write_text(
                env_text if env_text is not None else EXAMPLE.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    *arguments,
                    "--env-file",
                    str(env_file),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_plan_is_static_and_does_not_expose_credentials(self):
        secret = "do-not-print-this-dispatch-secret"
        env_text = EXAMPLE.read_text(encoding="utf-8").replace(
            "replace-this-evaluation-dispatch-token", secret
        )
        result = self.run_profile(
            "plan",
            "--runtime",
            "vllm",
            env_text=env_text,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("image_pin=immutable-sha256", result.stdout)
        self.assertIn("qualification=direct-and-gateway", result.stdout)
        self.assertNotIn(secret, result.stdout + result.stderr)

    def test_static_validation_accepts_the_checked_in_pin(self):
        result = self.run_profile(
            "validate",
            "--runtime",
            "vllm",
            "--static-only",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_runtime_is_rejected(self):
        result = self.run_profile("plan", "--runtime", "arbitrary-command")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported runtime", result.stderr)

    def test_mutable_image_tag_is_rejected(self):
        env_text = EXAMPLE.read_text(encoding="utf-8")
        image_line = next(
            line for line in env_text.splitlines() if line.startswith("VLLM_IMAGE=")
        )
        env_text = env_text.replace(image_line, "VLLM_IMAGE=vllm/vllm-openai:latest")
        result = self.run_profile(
            "plan",
            "--runtime",
            "vllm",
            env_text=env_text,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("immutable sha256 digest", result.stderr)

    def test_operator_env_files_are_excluded_from_git_and_docker(self):
        for ignore_file in (REPO_ROOT / ".gitignore", REPO_ROOT / ".dockerignore"):
            patterns = ignore_file.read_text(encoding="utf-8").splitlines()
            self.assertIn("/.env", patterns)
            self.assertIn("/.env.*", patterns)
        docker_patterns = (REPO_ROOT / ".dockerignore").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertIn(".venv", docker_patterns)
        self.assertIn("**/__pycache__", docker_patterns)

    def test_uv_and_python_versions_are_pinned(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('UV_VERSION="0.12.2"', source)
        self.assertIn('UV_PYTHON="3.12"', source)
        self.assertIn('--python "$UV_PYTHON"', source)


if __name__ == "__main__":
    unittest.main()
