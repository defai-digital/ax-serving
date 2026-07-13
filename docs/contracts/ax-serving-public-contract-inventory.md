# AX Serving public contract inventory

| Field | Value |
| --- | --- |
| Status | Source contract; release stability follows tagged release notes |
| Last updated | 2026-07-12 |
| Related | [Runtime-agent protocol](ax-serving-node-contract.md) |

This inventory identifies supported integration boundaries. Internal Rust
module paths, helper functions, dashboard DOM, and embedded backend traits are
not public contracts.

## Portable gateway

Released inference family:

- `POST /v1/chat/completions`;
- `POST /v1/completions`;
- `POST /v1/embeddings`;
- `GET /v1/models`.

The gateway preserves unknown JSON fields while classifying only bounded
routing metadata. Runtime semantics stay with the selected endpoint.
`/v1/responses` is not released until a runtime adapter passes its conformance
gate.

OpenAI compatibility means documented request/response behavior for supported
fields; it does not imply every extension of every upstream runtime.

## Errors and tracing

Gateway-generated inference errors use the AX envelope with:

- stable machine code;
- safe message/detail;
- request ID;
- retryable flag;
- admission/dispatch phase.

Responses preserve an opaque request ID. Attempt IDs are internal dispatch
evidence and must not be used as authentication or user identifiers.

## Health and observability

- `GET /livez` — process liveness;
- `GET /readyz` — routable readiness;
- `GET /health` — JSON fleet summary;
- `GET /v1/metrics` — admin-authenticated JSON operational metrics;
- `GET /metrics` — admin-authenticated Prometheus `axs_gateway_*` metrics;
- `GET /dashboard` — admin-authenticated compatibility convenience UI; its DOM is not a
  contract. Browser access requires an authenticated reverse proxy that injects the admin bearer
  credential; the token must not be placed in a URL or browser storage.

Prometheus names documented in the operations runbook are contract within a
protocol major. Metric label sets are intentionally bounded and may add
backward-compatible labels. Request IDs, worker IDs, prompts, and secrets are
not metric labels.

## Admin and lifecycle

Read/diagnostic family:

- `/v1/admin/status`;
- `/v1/admin/startup-report`;
- `/v1/admin/diagnostics`;
- `/v1/admin/audit`;
- `/v1/admin/policy`;
- `/v1/admin/fleet`;
- `/v1/admin/deployments`;
- `/v1/workers` and worker detail.

Asynchronous lifecycle family:

- `GET|POST /admin/v1/deployments`;
- `GET|PATCH|DELETE /admin/v1/deployments/{id}`;
- `GET /admin/v1/jobs`;
- `GET /admin/v1/jobs/{id}`.

Lifecycle mutations require `deployment_mode=explicit` and the admin
credential. Job records expose desired/observed state, progress, generation,
and bounded failure detail. They do not promise that AX Serving itself creates
runtime processes or downloads models.

Worker drain/remove routes are admin-only. Public inference keys are not admin
keys.

## Runtime-agent protocol

The cross-platform `ax-serving-protocol` crate and JSON fixtures define:

- version/capability negotiation;
- registration and lease fencing;
- heartbeat, readiness, inventory, and capacity observations;
- drain and deployment-job control;
- deployment/model/equivalence identity;
- request/attempt/admission/error fields.

New runtime integrations depend on this wire contract, not `InferenceBackend`
or other embedded Rust traits. Legacy registration is a migration contract and
cannot certify cross-runtime failover.

## Configuration

Documented `AXS_*` variables and YAML keys are operator contracts. The most
important trust-boundary variables are:

- `AXS_API_KEY`;
- `AXS_ADMIN_API_KEY`;
- `AXS_WORKER_TOKEN` / `AXS_INTERNAL_API_TOKEN`;
- `AXS_DISPATCH_TOKEN`;
- `AXS_RUNTIME_API_KEY`;
- `AXS_TLS_PROFILE`;
- `AXS_FLEET_STORE`, `AXS_REDIS_URL`, `AXS_FLEET_KEY_PREFIX`,
  `AXS_GATEWAY_ID`;
- `AXS_DEPLOYMENT_MODE` and explicit pool/deployment/equivalence YAML;
- queue, deadline, tenant, priority, and affinity controls documented in the
  quick start.

Secrets should be supplied through the deployment secret store, not checked-in
YAML. Configuration changes that remove or alter a documented field require a
migration note.

## Embedded compatibility

The `embedded-compat` feature includes macOS-only local inference, synchronous
model mutation, and `ax.serving.v1` gRPC. Those APIs can depend on local paths,
backend enums, and token-ID streams. They are not portable hybrid-gateway
contracts and must not be required by a Linux gateway integration.

## SDKs

The JavaScript and Python packages are convenience clients. Their released
surface follows package versioning and tests. Python gRPC access targets the
embedded compatibility service; portable hybrid clients should use REST/SSE.

## Non-contract surfaces

- internal Rust modules and workspace membership;
- embedded `InferenceBackend` implementations;
- source-only diagnostics not listed here;
- HTML/dashboard structure;
- benchmark summaries without schema-versioned raw evidence;
- placeholder deployment identities or certification files;
- private/commercial repository implementation details.

## Change policy

1. Prefer additive fields with explicit defaults.
2. Negotiate optional worker behavior through capabilities.
3. Change protocol major for incompatible wire semantics.
4. Document migrations for removed API/config behavior.
5. Test old registration/heartbeat fixtures within the supported window.
6. Keep public documentation limited to released and evidenced behavior.
