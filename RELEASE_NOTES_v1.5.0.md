# AX Serving v1.5.0

AX Serving `v1.5.0` completes the first production-grade enterprise
control-plane slice for AX Fabric deployments.

## Highlights

- Added authenticated admin status endpoint:
  - `GET /v1/admin/status`
- Added authenticated public worker-control endpoints:
  - `GET /v1/workers`
  - `GET /v1/workers/{id}`
  - `POST /v1/workers/{id}/drain`
  - `POST /v1/workers/{id}/drain-complete`
  - `DELETE /v1/workers/{id}`
- Added request-ID propagation into handler context so admin responses can
  return a correlation ID while preserving `X-Request-ID`
- Documented the `v1.5` control-plane surface in:
  - `README.md`
  - `docs/runbooks/multi-worker.md`
  - `ROADMAP.md`

## Bug Fixes

- Fixed `GET /v1/admin/status` so `auth_required` reflects actual runtime auth
  state instead of inferring from `AXS_ALLOW_NO_AUTH`

## Validation

- `cargo test -p ax-serving-api --test orchestration`
- `cargo clippy --workspace --tests -- -D warnings`
