# AX Fabric integration contract

| Field | Value |
| --- | --- |
| Status | Proposed portable-gateway contract |
| Last updated | 2026-07-15 |

AX Fabric should integrate with the portable AX Serving gateway over REST/SSE.
It should not depend on embedded backend traits, local model paths, or gRPC v1
for a federated fleet.

## Stable portable endpoints

- `GET /livez`;
- `GET /readyz`;
- `GET /routablez`;
- `GET /health`;
- `GET /v1/models`;
- `POST /v1/chat/completions`;
- `POST /v1/completions`;
- `POST /v1/embeddings`;
- `GET /v1/metrics`.

`/v1/responses` is not yet part of this contract. Admin and deployment routes
use a separate operator credential and should not be called by an ordinary
inference client.

## Authentication

AX Fabric sends its public inference token as:

```text
Authorization: Bearer <AXS_API_KEY value>
```

It must not receive or reuse `AXS_ADMIN_API_KEY`, worker-control, dispatch,
runtime, Redis, or affinity secrets.

## Health and readiness

- `/livez` `200` means the gateway process can answer HTTP.
- `/readyz` `200` means the control plane is ready (config, listeners, fleet
  store). It does **not** require an eligible worker. Production installs use
  this so agents can register during bootstrap. Legacy
  `AXS_READYZ_MODE=eligible_workers` restores the old worker-gated behavior.
- `/routablez` `200` means at least one endpoint/domain is currently eligible for
  inference; `503` means capacity is unavailable (structured inference 503 with
  `Retry-After` applies). AX Fabric should use `/routablez` (or
  `workers.eligible > 0` from `/health`) as serving capacity readiness, not
  merely process readiness.
- `/health` always returns a JSON fleet summary while the process is live.

Relevant `/health` fields:

```text
status
workers.total
workers.healthy
workers.unhealthy
workers.draining
workers.eligible
queue.active
queue.queued
queue.rejected_total
queue.shed_total
queue.timeout_total
```

AX Fabric must use `/routablez` or `workers.eligible > 0`, not merely
`/livez`/`/readyz` process readiness, as serving capacity readiness.

## Model inventory

`GET /v1/models` returns logical model aliases in explicit deployment mode and
eligible runtime model IDs in legacy compatibility mode. AX Fabric should
configure logical aliases and avoid selecting runtime pools directly.

The read API does not imply that every deployment behind an alias is
equivalent. AX Serving's explicit catalog and equivalence policy enforce that
internally.

The final execution-domain and Dynamo adapter contract is approved design but not a released
public contract. AX Fabric continues to use logical models and must not address Dynamo workers,
PC/Thor pools, or Mac endpoints directly.

## Inference

AX Fabric may use chat completions, text completions, embeddings, and SSE for
the fields documented by the release. Unknown request extensions are forwarded
to the runtime unless a bounded gateway validation rule rejects them.

The gateway does not render templates or tokenize prompts. Runtime token usage
is authoritative.

Gateway-generated errors include:

- AX machine code;
- request ID;
- retryable flag;
- phase;
- safe message/detail.

AX Fabric must not retry based on status code alone. AX Serving already performs
at most one safe pre-commit retry after connect failure or typed non-admission.
Generic runtime `5xx`, post-admission ambiguity, and committed streams are not
rerouted.

## Metrics

The gateway JSON profile includes:

```text
mode
policy
workers.healthy
workers.unhealthy
workers.draining
total_inflight
reroute_total
requests.total
requests.attempts_total
requests.completed_total
requests.failed_total
requests.cancelled_total
requests.retried_total
queue.active
queue.queued
queue.permit_total
queue.rejected_total
queue.shed_total
queue.timeout_total
```

AX Fabric may use these for diagnostics but should use request responses as the
source of truth for individual work. Prometheus monitoring should scrape
`/metrics` directly.

## Optional request metadata

- `x-ax-project` identifies a configured project/tenant policy;
- `x-ax-priority` is `low`, `normal`, or `high`;
- `x-ax-request-timeout-ms` may shorten but not extend the gateway maximum;
- `x-ax-minimum-context-tokens` declares a hard routing requirement;
- `x-ax-cache-affinity` is an opaque hint accepted only when the operator has
  configured a tenant-scoped affinity secret.

AX Fabric must not put prompt text, user PII, credentials, or globally stable
cross-tenant identifiers in affinity hints.

## Embedded compatibility appendix

The macOS `ax-serving` binary may expose synchronous `POST /v1/models`, model
delete/reload, thermal/scheduler metrics, and gRPC v1. Those are local embedded
contracts, not portable fleet contracts. AX Fabric integrations that still use
them need an explicit migration plan to runtime-managed model lifecycle and
the portable gateway.
