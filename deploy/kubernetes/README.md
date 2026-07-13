# Kubernetes deployment contract

These manifests are an integration baseline, not a certified production
release. They deploy two portable gateway replicas with Redis/Valkey-backed
fleet state and show a runtime plus `ax-runtime-agent` sidecar pattern.

Before applying the base:

1. Build the `gateway` and `agent` targets from
   `packaging/container/Dockerfile`, scan them, and replace image references
   with immutable digests.
2. Put a reachable, durable Redis or Valkey endpoint behind the `redis-url`
   secret. Do not use an unauthenticated in-cluster cache for production.
3. Create independent public, admin, control-plane, dispatch, and affinity
   credentials:

   ```bash
   kubectl create secret generic ax-serving-secrets \
     --from-literal=redis-url='rediss://...' \
     --from-literal=api-key='...' \
     --from-literal=admin-api-key='...' \
     --from-literal=internal-api-token='...' \
     --from-literal=dispatch-token='...' \
     --from-literal=cache-affinity-secret='at-least-32-random-bytes'
   ```

4. Terminate TLS at a trusted ingress and require mTLS inside the mesh. The
   `trusted_mesh` profile is an assertion that transport security is supplied
   externally; it does not create certificates.
5. Replace `legacy_compat` in the ConfigMap with explicit pools,
   deployments, immutable identities, and certified equivalence classes
   before enabling cross-runtime failover. Start from
   `config/serving.hybrid.example.yaml`.
6. Tailor the NetworkPolicy to the actual ingress, Redis namespace, DNS
   policy, and runtime-agent ports. The base intentionally has no cloud load
   balancer or ingress resource.

Render without mutating the cluster:

```bash
kubectl kustomize deploy/kubernetes/base
```

Apply only after replacing the placeholders:

```bash
kubectl apply -k deploy/kubernetes/base
```

The gateway exposes `/livez` for process liveness, `/readyz` for routable
worker readiness, and `/health` for the JSON fleet summary. Prometheus scrapes
`/metrics` with the admin bearer credential; do not put that credential in a
PodMonitor or ServiceMonitor checked into source. A new gateway remains
unready until at least one compatible worker is eligible; this prevents an
empty replica from receiving inference traffic.

The runtime-agent example is deliberately excluded from the base
Kustomization. Runtime images, GPU resources, model revisions, adapter
metadata, and equivalence certification are deployment-specific.
