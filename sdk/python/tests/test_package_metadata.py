from __future__ import annotations

import re
import tomllib
from pathlib import Path


SDK_ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _dependency_floor(pyproject: str, package: str) -> str:
    match = re.search(rf'"{re.escape(package)}>=(\d+(?:\.\d+)*)"', pyproject)
    assert match is not None, f"missing {package} dependency floor"
    return match.group(1)


def test_runtime_dependencies_match_generated_proto_requirements() -> None:
    pyproject = (SDK_ROOT / "pyproject.toml").read_text()
    grpc_stub = (SDK_ROOT / "ax_serving/_proto/ax_serving_pb2_grpc.py").read_text()
    protobuf_stub = (SDK_ROOT / "ax_serving/_proto/ax_serving_pb2.py").read_text()

    grpc_generated = re.search(r"GRPC_GENERATED_VERSION = '([^']+)'", grpc_stub)
    assert grpc_generated is not None

    protobuf_generated = re.search(
        r"ValidateProtobufRuntimeVersion\(\s*"
        r"_runtime_version\.Domain\.PUBLIC,\s*"
        r"(\d+),\s*(\d+),\s*(\d+),",
        protobuf_stub,
        re.MULTILINE,
    )
    assert protobuf_generated is not None

    protobuf_version = ".".join(protobuf_generated.groups())

    assert _version_tuple(_dependency_floor(pyproject, "grpcio")) >= _version_tuple(
        grpc_generated.group(1)
    )
    assert _version_tuple(_dependency_floor(pyproject, "protobuf")) >= _version_tuple(
        protobuf_version
    )


def test_portable_rest_install_does_not_require_grpc_runtime() -> None:
    project = tomllib.loads((SDK_ROOT / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == ["httpx>=0.27"]

    grpc_extra = project["optional-dependencies"]["grpc"]
    assert any(dependency.startswith("grpcio>=") for dependency in grpc_extra)
    assert any(dependency.startswith("protobuf>=") for dependency in grpc_extra)
    assert not any(dependency.startswith("grpcio-tools") for dependency in grpc_extra)


def test_pypi_long_description_is_declared_and_substantive() -> None:
    project = tomllib.loads((SDK_ROOT / "pyproject.toml").read_text())["project"]
    assert project["readme"] == "README.md"

    readme = (SDK_ROOT / project["readme"]).read_text()
    assert readme.startswith("# AX Serving Python SDK")
    assert len(readme.strip()) >= 1_000
    assert "uv add ax-serving" in readme
    assert "AXS_API_KEY" in readme
