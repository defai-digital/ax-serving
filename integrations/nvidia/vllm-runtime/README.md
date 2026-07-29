# AX Serving vLLM backend runtime

This Apache-2.0 integration contains pinned, fail-closed vLLM backend profiles
used when qualifying an NVIDIA Dynamo execution domain for AX Serving. It is
not linked into the AX Serving gateway or Dynamo adapter, and it is not an AX
Engine backend.

AX Serving selects a Dynamo domain. Dynamo selects and operates NVIDIA
workers. This package supplies a reviewed backend runtime closure for a
specific domain compatibility manifest; it does not make AX Serving a
CUDA scheduler.

The Python import package and `ax-engine-vllm-runtime` command alias remain
temporarily stable for deployments migrating from AX Engine. New automation
must use `ax-serving-vllm-runtime`.

## Profiles

Profiles validate the OS, architecture, GPU identity/compute capability,
CPython, PyTorch, CUDA, exact vLLM release, and packaged dependency-lock
digest before launching a worker:

```bash
ax-serving-vllm-runtime --list-profiles
ax-serving-vllm-runtime \
  --profile cuda-linux-x86_64-a6000-sm86 \
  --model baidu/Unlimited-OCR \
  --served-model-name baidu/Unlimited-OCR \
  --check-only
```

Architecture-specific behavior remains inside narrowly guarded vLLM plugins.
It never changes the AX Serving gateway-to-Dynamo OpenAI contract.

## Reproducible installation

Release environments are installed from the architecture-specific,
SHA-256-complete lock before installing this wheel without dependencies:

```bash
UV_TORCH_BACKEND=cu130 uv pip sync \
  --python /opt/ax-serving-vllm-venv/bin/python \
  --require-hashes locks/requirements-runtime-amd64.lock
uv pip install \
  --python /opt/ax-serving-vllm-venv/bin/python \
  --no-deps dist/ax_serving_vllm_runtime-0.1.0-py3-none-any.whl
```

Use `requirements-runtime-arm64.lock` on Thor. Both locks are embedded in the
wheel so preflight can attest the release closure after installation.

The profile remains a candidate until its native correctness, performance,
security, fault, and soak evidence is referenced by an immutable AX Serving
Dynamo compatibility manifest. PC and Thor evidence are never interchangeable.

The two BF16 Unlimited-OCR profiles pin source revision
`ee63731b6461c8afcdcc7b15352e7d2ffecc2ead`. An explicit `--revision` remains
available only for a separately reviewed artifact.

Bearer credentials are accepted only through an environment variable or
secret file. The default worker bind is loopback. A public bind requires both
`--allow-public-bind` and a configured API key.

## Licensing

This migrated integration and the AX Serving workspace are Apache-2.0; see
[LICENSE](LICENSE) and the repository-level
[licensing policy](../../../LICENSING.md).
