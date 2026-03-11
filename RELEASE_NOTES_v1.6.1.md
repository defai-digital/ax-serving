# AX Serving v1.6.1

`v1.6.1` is a patch release that fixes audit coverage for failed license updates.

## Fixes

- Failed `POST /v1/license` requests are now audited on both:
  - the serving API
  - the orchestrator admin surface
- Validation failures and persistence failures now emit audit events with:
  - `action = "license_set"`
  - `outcome = "error"`

## Why It Matters

`v1.6.0` introduced authenticated admin diagnostics and audit surfaces. This patch closes an important gap so invalid admin mutations are no longer missing from the audit trail.

## Verification

- `cargo test -p ax-serving-api --test model_management`
- `cargo test -p ax-serving-api --test orchestration`
- `cargo clippy --workspace --tests -- -D warnings`
