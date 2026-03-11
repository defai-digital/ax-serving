# AX Serving v1.7.0

`v1.7.0` adds the first fleet-control slice for offline and business deployments.

## Highlights

- Added worker fleet metadata:
  - `worker_pool`
  - `node_class`
- Added pool-aware dispatch preference:
  - request header `X-AX-Worker-Pool`
- Added authenticated fleet summary endpoint:
  - `GET /v1/admin/fleet`
- Extended worker auto-registration so workers can advertise:
  - `AXS_WORKER_POOL`
  - `AXS_WORKER_NODE_CLASS`

## Why It Matters

This release moves `ax-serving` beyond single-node control into practical multi-worker fleet operations:

- operators can group workers by pool for maintenance and placement
- request routing can prefer a target pool without breaking default fallback behavior
- admin tooling can inspect fleet capacity by pool, node class, and backend

## Verification

- `cargo test --workspace`
- `cargo clippy --workspace --tests -- -D warnings`
