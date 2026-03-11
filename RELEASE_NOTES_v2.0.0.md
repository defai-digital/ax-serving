# AX Serving v2.0.0

This release adds project-scoped runtime governance for the public OSS + Business serving stack.

## Highlights

- Added project-scoped runtime admission policy via `X-AX-Project`
- Added config-backed per-project rules for:
  - allowed models
  - max token limits
  - optional enforced worker pool
- Enforced project policy on:
  - `/v1/chat/completions`
  - `/v1/completions`
  - `/v1/embeddings`
  - orchestrator proxy admission
- Added authenticated policy visibility:
  - `GET /v1/admin/policy`
- Included project policy in startup/admin diagnostics surfaces

## Scope

- This is a real governance step for the public `ax-serving` repo
- It does not attempt to implement full private-enterprise tenant/compliance machinery

## Verification

- `cargo test -p ax-serving-api config::tests`
- `cargo test -p ax-serving-api --test model_management`
- `cargo test -p ax-serving-api --test orchestration`
- `cargo clippy -p ax-serving-api --tests -- -D warnings`
