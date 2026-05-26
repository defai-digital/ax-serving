from __future__ import annotations

import re
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
