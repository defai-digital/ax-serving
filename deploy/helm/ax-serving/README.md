# ax-serving Helm chart

First-party **CPU-only** gateway chart. Does not install AX Engine, vLLM, SGLang,
GPU operators, device plugins, or model weights.

## Install

```bash
helm upgrade --install ax-serving ./deploy/helm/ax-serving \
  -f deploy/helm/ax-serving/ci/values-production.yaml
```

## Profiles

| Profile | Path | Purpose |
| --- | --- | --- |
| minimal | `ci/values-minimal.yaml` | One replica, memory fleet store |
| production | `ci/values-production.yaml` | Two replicas, Redis, digest, existing Secret |

## CPU-only policy

Rendered manifests request only ordinary CPU/memory resources. Chart validation
rejects `nvidia.com/gpu` in gateway resource maps.
