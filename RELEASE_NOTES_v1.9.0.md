# AX Serving v1.9.0

This release adds workload-efficiency observability for the current offline serving implementation.

## Highlights

- Added request-class counters for real runtime paths:
  - `cold_requests_total`
  - `exact_cache_hits_total`
  - `cache_follower_hits_total`
  - `cache_fills_total`
- Exposed the new counters in:
  - `GET /v1/metrics`
  - `GET /metrics`
- Recorded request classes across chat and text completion flows for:
  - exact response-cache hits
  - follower-served cache hits
  - successful cache fills
  - cold inference execution

## Notes

- This release improves observability for the current exact-match response cache.
- It does not claim KV-prefix cache support.

## Verification

- `cargo test -p ax-serving-api --test model_management`
- `cargo test -p ax-serving-api metrics::tests`
- `cargo clippy -p ax-serving-api --tests -- -D warnings`
