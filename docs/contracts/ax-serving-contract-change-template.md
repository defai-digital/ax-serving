# AX Serving Contract Change Note Template

Use this template whenever a public contract change affects:

- REST or gRPC APIs
- worker lifecycle protocol
- documented config or `AXS_*` environment variables
- documented metrics or audit payloads

# Title

Short one-line contract change summary.

# Release

- Public core version:
- Release date:
- Owner:

# Contract Surface

- Surface family:
- Endpoint, protocol, config key, or metric name:
- Contract status:
  - additive
  - behavior change
  - breaking

# What Changed

Describe the exact before/after behavior.

# Why

State the operational or product reason for the change.

# Compatibility Impact

- Open-source impact:
- Commercial-core impact:
- Enterprise repo impact:

# Required Enterprise Actions

- control plane:
- workers:
- deploy:
- connectors:

# Migration Guidance

Provide any required rollout sequence, fallback behavior, or migration steps.

# Validation

- tests added or updated:
- docs updated:
- compatibility metadata updated:

# Example Payloads

Include short before/after examples only if they clarify the change.
