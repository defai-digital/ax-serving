# Migrate AX Engine CUDA routes to AX Serving

AX Engine’s former delegated `vllm`, `tensor_rt_llm`, and
`tensor_rt_edge_llm` routes are replaced by an AX Serving execution domain.
AX Engine remains the Apple Silicon MLX/Metal runtime.

## Mapping

| Former AX Engine setting | AX Serving replacement |
| --- | --- |
| `--support-tier vllm` | Dynamo backend kind in the compatibility manifest |
| `--vllm-server-url` | `AXS_DYNAMO_FRONTEND_URL` |
| `--vllm-upstream-model-id` | deployment `runtime_model_id` |
| `--vllm-runtime-profile` | compatibility-manifest digest and backend profile |
| `--vllm-max-in-flight` | `AXS_DYNAMO_MAX_INFLIGHT` plus Dynamo admission |
| `--support-tier tensor-rt-llm` | Dynamo manifest backend `tensorrt_llm` |
| `--support-tier tensor-rt-edge-llm` | separate Thor Dynamo domain manifest |
| delegated AX Engine Linux server | `ax-dynamo-adapter` |

## Sequence

1. Deploy and pin the upstream Dynamo domain.
2. Produce a complete compatibility manifest and validate it with
   `ax-dynamo-adapter check-manifest`.
3. Declare the domain, pool, deployment, runtime model, and equivalence policy
   in AX Serving.
4. Start `ax-dynamo-adapter` and verify it registers as a domain endpoint.
5. Run inventory, blocking, streaming, cancellation, fault, and no-duplicate
   retry conformance.
6. Shadow and canary the AX Serving route.
7. Drain the old AX Engine delegated endpoint.
8. Remove the old route only after rollback evidence is retained.

Direct vLLM/SGLang runtime-agent mode remains a migration-only compatibility
path. It cannot claim a production Dynamo domain.
