# AX Serving Advantages and Use Cases

## Core Advantages

### 1. Mac-Native Performance
- Optimized for Apple Silicon (Metal backend)
- Excellent thermal and power management awareness
- Zero-copy inference where possible via `ax-engine`

### 2. Production-Grade Orchestration
- **Multi-worker gateway** with intelligent routing (`least_inflight`, `model_affinity`, `token_cost`)
- **Adaptive scheduling** with queue management and admission control
- **Response cache** (Valkey/Redis) for exact prompt caching — dramatically improves repeated queries
- **Thermal protection** — automatically degrades gracefully under heat

### 3. Developer Experience
- **OpenAI compatible API** — works with existing clients and LangChain/LlamaIndex
- **Official Python and JavaScript SDKs**
- **gRPC + REST** dual support
- **Rich observability** — detailed metrics, dashboard, audit logs

### 4. Operational Excellence
- Runtime model loading/unloading without restart
- License management and commercial features
- Comprehensive health checks and diagnostics
- Enterprise-ready authentication and policy controls

### 5. Hybrid Backend Strategy
- Default to battle-tested `llama.cpp`
- Optional high-performance `ax-engine` (native Rust) backend
- Seamless fallback between them

## Target Use Cases

### Department-Scale Private AI
- Internal company chat, RAG, and agent workflows
- 10–200 concurrent users on Mac infrastructure
- Full data sovereignty and compliance

### Mac Grid / Private Cloud
- Multiple Mac Studios or Mac Pros working together
- Centralized gateway with automatic worker registration
- Load balancing across heterogeneous hardware

### Embedded / Offline Enterprise
- Air-gapped deployments
- Secure offline inference with commercial license
- Integration with AX Fabric for grounded agents

### Developer & Research Workstations
- Local serving for prototyping
- Benchmarking and performance tuning
- Education and internal tool development

## Comparison to Alternatives

| Feature                    | AX Serving          | llama.cpp server | vLLM / TGI     |
|---------------------------|---------------------|------------------|----------------|
| Mac Silicon optimization  | Excellent           | Good             | Poor           |
| Multi-worker orchestration| Built-in            | None             | Limited        |
| Response cache            | Yes (exact match)   | No               | Limited        |
| Dynamic model management  | Yes                 | Limited          | Yes            |
| Python SDK                | Official + rich     | Basic            | Official       |
| Thermal & power awareness | Yes                 | Basic            | No             |
| Commercial licensing path | Yes                 | No               | Yes            |

## When to Use AX Serving

**Use AX Serving when you need:**
- Production-grade serving on Apple Silicon
- Multiple models or workers
- Caching for cost/latency optimization
- Strong operational controls and observability
- OpenAI compatibility + advanced features (logprobs, tools, grammar, vision)

**Consider alternatives when:**
- You only need single-process local inference → use `llama-server` directly
- Running on NVIDIA GPUs at hyperscale → consider vLLM/TGI

## Getting Started Paths

1. **Local testing**: `cargo run -p ax-serving-cli --bin ax-serving -- serve -m model.gguf`
2. **With Python**: See `docs/python-sdk.md`
3. **Production multi-worker**: See `docs/runbooks/multi-worker.md`
4. **Enterprise**: Use commercial license + `serving.offline-enterprise.yaml`

## Best Practices

### Deployment
- Use dedicated orchestrator (`ax-serving-api`) + multiple `ax-serving serve` workers
- Prefer `least_inflight` or `model_affinity` dispatch policy
- Enable response caching for repetitive prompts

### Performance & Reliability
- Tune `AXS_SCHED_MAX_INFLIGHT`, `AXS_SCHED_MAX_QUEUE`, `AXS_SCHED_MAX_WAIT_MS` per hardware (see `docs/perf/service-tuning.md`)
- Set `AXS_MODEL_ALLOWED_DIRS` to restrict model paths
- Monitor thermal state and adjust concurrency during high load
- Use `--release` builds in production

### Operations
- Always use API keys (`AXS_API_KEY`) in production
- Implement graceful drain before restarting workers
- Set up monitoring on `/v1/admin/status` and `/v1/metrics`
- Keep warm models in memory using warm pool configuration

### Security
- Never expose internal port (`19090`) publicly
- Use allowlist directories for models
- Rotate API keys regularly

---

**See also:**
- [QUICKSTART.md](../QUICKSTART.md)
- [python-sdk.md](python-sdk.md)
- [market-positioning.md](market-positioning.md)
- [PRD](../prd/PRD-AX-SERVING-v3.0.md)
