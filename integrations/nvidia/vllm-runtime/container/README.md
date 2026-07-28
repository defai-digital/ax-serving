# AX Serving vLLM backend container

Build from `integrations/nvidia/vllm-runtime` as the context. The image is a
qualified backend candidate for a separately operated Dynamo domain; it is not
the AX Serving gateway or `ax-dynamo-adapter` image.

```bash
cd integrations/nvidia/vllm-runtime

python -m build --wheel
wheel=dist/ax_serving_vllm_runtime-0.1.0-py3-none-any.whl
wheel_sha256="$(sha256sum "$wheel" | cut -d ' ' -f 1)"

docker build \
  --file container/Dockerfile \
  --build-arg RUNTIME_WHEEL="$wheel" \
  --build-arg RUNTIME_WHEEL_SHA256="$wheel_sha256" \
  --tag ax-serving-vllm-runtime:0.1.0-amd64 \
  .
```

Use the ARM64 lock for Thor. Record the resulting immutable image digest in
the domain compatibility manifest. A local tag is never qualification
evidence.
