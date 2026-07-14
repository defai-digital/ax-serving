# AX Serving Runtime Responsibility Inventory

> **Status**: Active
> **Date**: 2026-07-12
> **Owner**: AX Serving Team

---

## Purpose

This inventory classifies runtime and backend code paths while AX Serving moves
toward the PRD target architecture:

- AX Serving is the serving gateway and control plane.
- Runtime nodes own inference execution.
- Mac inference should be provided by ax-engine nodes.
- PC CUDA and NVIDIA Thor inference should be provided by vLLM nodes.

---

## Classification

| Area | Current role | Target role | Status |
|---|---|---|---|
| OpenAI-compatible REST/SSE API | Serving API | Keep in portable AX Serving | Gateway |
| `ax.serving.v1` gRPC | Local path/backend/token-ID API | Keep only behind `embedded-compat` pending deprecation policy | Compatibility |
| `ax-serving-protocol` | Runtime-neutral wire DTOs | Keep as independently versioned cross-platform contract | Protocol |
| Worker registry, heartbeat, drain | Fleet control plane | Keep in AX Serving | Gateway |
| Routing, placement, admission, retry | Fleet policy | Keep in AX Serving | Gateway |
| Metrics, audit, fleet diagnostics | Operator surface | Keep in AX Serving | Gateway |
| `ax-runtime-agent` / `ax-thor-agent` proxy | Runtime adapter | Keep as thin OpenAI-compatible runtime adapter | Adapter |
| `ax-serving serve` local worker | Embedded Mac compatibility worker | Keep only as migration bridge | Compatibility |
| Native ax-engine backend inside serving crate | Direct runtime execution | Isolated behind `embedded-compat`; prefer the AX Engine agent + HTTP `ax-engine-server` (see LAN discovery design 2026-07-14) | Compatibility |
| LAN mDNS discovery (`_ax-engine._tcp`) | Lab/home peer find | Opt-in browse/advertise only; never replaces register/heartbeat tokens | Bootstrap |
| llama.cpp subprocess backend | Direct runtime execution | Deprecate after runtime-node deployments cover migration needs | Compatibility |
| MLX subprocess backend | Direct runtime execution | Deprecate or adapterize after runtime-node replacement is validated | Compatibility |
| optional libllama direct backend | Direct runtime execution | Deprecate unless required for shim compatibility | Compatibility |
| Shim library | llama-style integration compatibility | Keep only if it serves a supported integration path | Compatibility |

---

## Compatibility Rules

- Compatibility paths must not become new product architecture.
- Compatibility paths may remain for migration, local testing, and explicitly
  approved integrations.
- New routing and observability features should use runtime/node metadata rather
  than embedded backend internals.
- The portable gateway dependency tree must not contain AX Engine, MLX,
  llama.cpp, Metal, CUDA, `prost`, or `tonic`.
- Runtime-specific tuning belongs in ax-engine or vLLM documentation and
  tooling, not in AX Serving control-plane requirements.
- Breaking removal requires a migration note and an equivalent node-adapter path.

---

## Operator Signals

The embedded `ax-serving doctor` command includes a runtime-boundary check:

- `PASS` when worker registration is explicitly configured for `ax_engine` or
  `vllm` runtime-node mode.
- `PASS` when `AXS_EMBEDDED_RUNTIME_POLICY=deny` disables embedded
  compatibility paths for gateway-only deployments.
- `WARN` when AX Serving appears to be used in standalone embedded inference
  mode.
- `WARN` when embedded compatibility paths are explicitly allowed.
- `WARN` when an unknown runtime is configured.

This warning is intentional. It does not block local compatibility use, but it
keeps the PRD target visible to operators.

## Compatibility Quarantine Policy

Embedded runtime paths are controlled by:

```text
AXS_EMBEDDED_RUNTIME_POLICY=warn
```

Supported values:

| Value | Behavior |
|---|---|
| `warn` | Default. Allow embedded compatibility paths and emit operator warnings. |
| `allow` | Explicitly allow embedded compatibility paths for migration or local testing. |
| `deny` | Block embedded local inference and require gateway plus runtime-node deployment. |

The `deny` mode applies to `ax-serving serve -m ...` and single-shot
`ax-serving -m ...` inference. It does not block `ax-serving-api` gateway mode
or runtime-node adapters such as `ax-runtime-agent`.

The portable CLI's default feature is `gateway`; building `ax-serving` itself
requires `--no-default-features --features embedded-compat` on macOS.

---

## Removal Gate

Before removing or disabling any compatibility backend:

1. Document the replacement node adapter.
2. Confirm OpenAI-compatible chat/completions behavior through the gateway.
3. Confirm worker registration, heartbeat, drain, and fleet status.
4. Confirm routing by model and runtime class.
5. Run `ax-servingctl migration embedded-readiness` against the gateway and resolve
   all blockers before switching production to
   `AXS_EMBEDDED_RUNTIME_POLICY=deny`.
6. Provide migration notes for existing configuration and CLI usage.
