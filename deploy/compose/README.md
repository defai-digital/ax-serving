# Docker Compose evaluation stack

CPU-only local evaluation for the AX Serving gateway. **Not** an HA or production certification surface.

## Quick start

```bash
# From repository root
docker compose -f deploy/compose/compose.yaml up --build gateway redis
```

Probe endpoints:

```bash
curl -i http://127.0.0.1:18080/livez
curl -i http://127.0.0.1:18080/readyz      # control-plane ready without workers
curl -i http://127.0.0.1:18080/routablez  # 503 until a runtime agent registers
```

Optional runtime agent (requires a reachable OpenAI-compatible runtime):

```bash
cp deploy/compose/.env.example deploy/compose/.env
# edit AXS_NODE_RUNTIME_URL / tokens
docker compose -f deploy/compose/compose.yaml --profile agent up --build
```

On macOS, AX Engine remains a native host process. Use
`AXS_NODE_RUNTIME_URL=http://host.docker.internal:<port>` from the agent container,
or run the agent natively with `AXS_NODE_ADVERTISED_URL`.

## Notes

- Compose uses `AXS_ALLOW_NO_AUTH=true` for evaluation only.
- Gateway readiness does not depend on runtime capacity.
- Images are built from portable features only (no AX Engine / CUDA / MLX).
