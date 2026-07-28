# NVIDIA Dynamo domains

AX Serving integrates NVIDIA capacity at the Dynamo domain boundary. One
`ax-dynamo-adapter` process represents one independently deployed Dynamo
frontend and its backend graph. AX Serving never registers or selects Dynamo
GPU workers.

## Ownership

| Concern | Owner |
| --- | --- |
| Public API, tenant policy, domain selection, audit | AX Serving gateway |
| Domain registration, manifest identity, proxy boundary | `ax-dynamo-adapter` |
| NVIDIA worker routing, KV placement, retry, scaling | NVIDIA Dynamo |
| Token execution | Dynamo-selected vLLM, SGLang, or TensorRT-LLM backend |
| Apple Silicon execution | AX Engine |

The gateway and adapter are runtime-SDK-free Rust binaries. CUDA, PyTorch,
vLLM, TensorRT-LLM, NIXL, and Dynamo Python packages are not linked into them.

## Start an adapter

Build:

```bash
cargo build --release -p ax-dynamo-adapter
```

Validate an immutable compatibility manifest before deployment:

```bash
target/release/ax-dynamo-adapter check-manifest \
  --manifest /etc/ax-serving/dynamo-compatibility-manifest.json \
  --domain-kind nvidia-dynamo-pc
```

[`integrations/nvidia/compatibility-manifest.example.json`](../../../integrations/nvidia/compatibility-manifest.example.json)
shows the complete shape. Every example release, digest, timestamp, and
evidence value is a placeholder and must be replaced with qualification
receipts before deployment.

Configure the process from
[`config/dynamo-adapter.example.env`](../../../config/dynamo-adapter.example.env).
Credentials must come from a secret manager or process environment. Then run:

```bash
target/release/ax-dynamo-adapter
```

The adapter registers with protocol v1.1 as:

- kind `nvidia_dynamo_pc` or `nvidia_dynamo_thor`;
- endpoint scope `domain`;
- execution owner `dynamo`;
- runtime kind `dynamo`;
- one manifest digest identifying the complete pinned domain stack.

It reports only aggregate local admission and frontend inventory. It does not
copy Dynamo worker IDs, KV indexes, planner state, or accelerator scheduling
into AX Serving.

## Request and retry boundary

The gateway rewrites only the top-level logical model to the configured
runtime model. The adapter preserves the resulting body and unknown fields,
adds the configured Dynamo credential, and sends exactly one request to the
Dynamo frontend.

Only local drain/capacity/not-ready rejection before upstream dispatch carries
`x-ax-admission-state: not-admitted`. A Dynamo timeout, connection reset,
generic `5xx`, malformed response, or other ambiguous failure is never marked
safe for cross-domain retry. Dynamo owns all retry or migration after the
request enters the domain.

## Backend runtime profiles

The migrated Apache-2.0 vLLM profile package lives under
[`integrations/nvidia/vllm-runtime`](../../../integrations/nvidia/vllm-runtime/README.md).
It is an optional backend candidate referenced by a Dynamo compatibility
manifest, not an AX Serving gateway dependency.

PC and Thor use separate domains, manifests, profiles, evidence, calibration,
and promotion decisions. Neither inherits the other’s certification.

## Qualification

Run the source conformance tests:

```bash
cargo test -p ax-dynamo-adapter
python3 scripts/qualification/nvidia/test_soak_dynamo_domain.py
```

Run the live contract smoke with the gateway, adapter, and Dynamo domain:

```bash
AXS_DYNAMO_MANIFEST_PATH=/path/to/manifest.json \
AXS_DYNAMO_DOMAIN_KIND=nvidia_dynamo_pc \
MODEL_ID=<logical-model> \
bash scripts/qualification/nvidia/smoke_dynamo_domain.sh
```

Passing source or mock tests does not certify hardware. Promotion still
requires immutable live evidence for identity, correctness, streaming,
cancellation, faults, security, performance, and soak.
