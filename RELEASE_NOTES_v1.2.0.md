## ax-serving v1.2.0

This release republishes the project from a clean root history for public launch.

### Highlights

- Public license set to `AGPL-3.0-only`
- Commercial licensing path documented for non-AGPL use
- Public collaboration policy clarified:
  - issue reports are welcome
  - unsolicited public code contributions are not accepted

### Included In This Snapshot

- API request validation hardening for:
  - `/v1/embeddings` `encoding_format`
  - `/v1/chat/completions` and `/v1/completions` `top_logprobs`
  - `/v1/chat/completions` and `/v1/completions` `response_format.type`
- Regression tests covering the new validation paths
- Config test hardening to serialize environment-variable mutation in process-global tests

### Notes

- This history rewrite changes the published git history going forward.
- It does not revoke rights already granted on previously published code snapshots.
