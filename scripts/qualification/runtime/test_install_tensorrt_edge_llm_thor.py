#!/usr/bin/env python3
"""Static safety tests for the pinned Jetson Thor Edge-LLM installer."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("install_tensorrt_edge_llm_thor.sh")
MANAGER = Path(__file__).with_name("manage_thor_edge_llm.sh")


class TensorRtEdgeLlmThorInstallerTests(unittest.TestCase):
    def run_installer(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_script_is_valid_bash_and_documents_actions(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = self.run_installer("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("system-deps", help_result.stdout)
        self.assertIn("install", help_result.stdout)
        self.assertIn("start", help_result.stdout)

    def test_release_uv_and_model_are_immutable(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('EDGE_TAG="v0.9.1"', source)
        self.assertIn(
            'EDGE_COMMIT="7f061f21f0a581ba234a1e233c9315b89d8e47d6"', source
        )
        self.assertIn('UV_VERSION="0.12.2"', source)
        self.assertIn('UV_PYTHON="3.12"', source)
        self.assertIn('"$uv_bin" venv --python "$UV_PYTHON"', source)
        self.assertIn('JETPACK_PACKAGE_VERSION="7.2-b187"', source)
        self.assertIn(
            'L4T_CORE_VERSION="39.2.0-20260601141651"',
            source,
        )
        self.assertIn(
            'TENSORRT_PACKAGE_VERSION="10.16.2.10-1+cuda13.2"',
            source,
        )
        self.assertIn(
            'DEFAULT_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"',
            source,
        )

    def test_plugin_path_is_discovered_from_the_pinned_build(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("runtime_plugin_path()", source)
        self.assertIn("-name 'libNvInfer_edgellm_plugin.so' -print0", source)
        self.assertNotIn(
            '$build_dir/core/libNvInfer_edgellm_plugin.so',
            source,
        )

    def test_server_defaults_to_loopback_and_remote_probe_uses_uv(self):
        installer_source = SCRIPT.read_text(encoding="utf-8")
        manager_source = MANAGER.read_text(encoding="utf-8")
        self.assertIn('bind_host="127.0.0.1"', installer_source)
        self.assertIn("~/.local/bin/uv run --no-project", manager_source)
        self.assertIn("~/.local/bin/uv --version", manager_source)
        self.assertIn("--python '$UV_PYTHON'", manager_source)
        self.assertIn("--base-url http://127.0.0.1:8000", manager_source)
        self.assertIn("--max-batch-size 1", installer_source)
        self.assertIn("--concurrency 1", manager_source)
        self.assertIn('kill -KILL "$pid"', installer_source)

    def test_malformed_model_revision_fails_before_host_mutation(self):
        result = self.run_installer(
            "system-deps",
            "--model-revision",
            "main",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact hexadecimal revision", result.stderr)

    def test_expected_hostname_guard_runs_before_preflight(self):
        result = self.run_installer(
            "preflight",
            "--expected-hostname",
            "definitely-not-this-host",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hostname mismatch", result.stderr)

    def test_host_manager_allow_lists_ssh_targets(self):
        syntax = subprocess.run(
            ["bash", "-n", str(MANAGER)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        result = subprocess.run(
            [
                "bash",
                str(MANAGER),
                "preflight",
                "--target",
                "untrusted-host",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported target", result.stderr)


if __name__ == "__main__":
    unittest.main()
