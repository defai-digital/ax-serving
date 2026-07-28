"""AX Serving-owned, profile-driven vLLM backend runtime.

The import package retains its original name during the migration window so
existing vLLM plugin entry points remain stable.
"""

from .launcher import ServeConfig, build_vllm_command, build_vllm_environment
from .preflight import PreflightReport, collect_runtime_facts, run_preflight
from .profiles import PROFILE_SCHEMA_VERSION, RuntimeProfile, get_profile, profiles

__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "PreflightReport",
    "RuntimeProfile",
    "ServeConfig",
    "build_vllm_command",
    "build_vllm_environment",
    "collect_runtime_facts",
    "get_profile",
    "profiles",
    "run_preflight",
]

__version__ = "0.1.0"
