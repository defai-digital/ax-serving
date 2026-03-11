## ax-serving v1.3.0

This release makes AX Serving the standard offline runtime/control plane line for AX Fabric.

### Highlights

- adds a stable runtime health contract through `GET /health`
- adds explicit lifecycle state in load, unload, and reload responses
- adds an offline enterprise starter profile at `config/serving.offline-enterprise.yaml`
- documents the supported AX Fabric runtime contract
- expands integration coverage for health, model availability, and lifecycle transitions

### Runtime contract changes

- `/health` now reports:
  - `status`
  - `ready`
  - `model_available`
  - `reason`
  - `loaded_model_count`
- `POST /v1/models` now returns resulting runtime state
- `DELETE /v1/models/{id}` now returns resulting runtime state
- `POST /v1/models/{id}/reload` now returns resulting runtime state

### Notes

- this release keeps AX Serving positioned as the execution and serving control plane for AX Fabric
- unrelated local scheduler/backend experiments were excluded from this release
