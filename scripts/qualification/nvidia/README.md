# NVIDIA domain qualification

These checks belong to AX Serving because they validate a complete NVIDIA
execution domain, not AX Engine token execution.

- `smoke_dynamo_domain.sh` validates the immutable manifest and exercises
  inventory, non-streaming chat, and SSE through the AX Serving public API.
- `soak_dynamo_domain.py` produces bounded no-retry soak evidence.
- `test_soak_dynamo_domain.py` covers the evidence runner without hardware.

PC and Thor runs must use separate manifests, output directories, and release
evidence. A direct backend success does not certify the Dynamo adapter or the
AX Serving route.
