//! ax-serving-api: multi-worker API gateway for ax-serving.
//!
//! Starts the orchestrator that proxies inference requests across registered
//! runtime-agent nodes. Holds no model weights and no accelerator context.
//!
//! # Usage
//!
//!   ax-serving-api                                  # defaults: port 18080, internal 19090
//!   ax-serving-api --host 0.0.0.0 --port 9000 --internal-port 9001 --policy weighted_round_robin
//!
//! # Key environment variables
//!
//!   AXS_ORCHESTRATOR_HOST   — public proxy host (default: 127.0.0.1)
//!   AXS_ORCHESTRATOR_PORT   — public proxy port (default: 18080)
//!   AXS_INTERNAL_PORT       — loopback-only internal API port (default: 19090)
//!   AXS_DISPATCH_POLICY     — worker selection policy (default: inference_aware)
//!                             choices: inference_aware | least_inflight | weighted_round_robin |
//!                                      model_affinity | token_cost | cache_affinity
//!   AXS_WORKER_HEARTBEAT_MS — heartbeat interval hint sent to workers (default: 5000)
//!   AXS_WORKER_TTL_MS       — eviction TTL for silent workers (default: 15000)
//!   AXS_GLOBAL_QUEUE_MAX    — max concurrent requests before overload policy triggers (default: 128)
//!   AXS_GLOBAL_QUEUE_WAIT_MS — max queue wait before 503 (default: 10000)
//!   AXS_LOG                 — tracing filter, e.g. "debug" or "ax_serving_api=trace"

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "ax-serving-api",
    version,
    about = "AX Serving API gateway — routes requests across runtime nodes",
    long_about = "\
Start the multi-worker API gateway. The gateway is a pure dispatch process:\n\
it holds no model weights and starts no accelerator context. Runtime nodes are\n\
registered via POST /internal/workers/register (done automatically by\n\
`ax-runtime-agent`).\n\
\n\
Mode: direct (default) — proxies over loopback HTTP, zero external deps.\n\
See docs/runbooks/multi-worker.md for the full deployment guide."
)]
struct Cli {
    /// Public proxy host.
    /// Overrides AXS_ORCHESTRATOR_HOST (default: 127.0.0.1).
    #[arg(long)]
    host: Option<String>,

    /// Public proxy port. Clients send OpenAI API requests here.
    /// Overrides AXS_ORCHESTRATOR_PORT (default: 18080).
    #[arg(long)]
    port: Option<u16>,

    /// Internal API port, bound to loopback only.
    /// Workers register and send heartbeats here.
    /// Overrides AXS_INTERNAL_PORT (default: 19090).
    #[arg(long)]
    internal_port: Option<u16>,

    /// Worker selection policy.
    /// Choices: inference_aware (default), least_inflight, weighted_round_robin,
    /// model_affinity, token_cost, cache_affinity.
    /// Overrides AXS_DISPATCH_POLICY.
    #[arg(long)]
    policy: Option<String>,

    /// Opt-in mDNS advertise of the *internal* control plane
    /// (`_ax-serving-gateway._tcp`) so agents can resolve AXS_CONTROL_PLANE_URL.
    #[arg(long, default_value_t = false)]
    advertise_lan: bool,

    /// Optional cluster / namespace TXT label for LAN isolation.
    #[arg(long)]
    lan_cluster: Option<String>,

    /// DNS-SD instance name (default: hostname or ax-serving-gateway).
    #[arg(long)]
    lan_instance_name: Option<String>,

    /// IPv4 published in mDNS (default: detected private address).
    #[arg(long)]
    lan_advertise_host: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let _telemetry = ax_serving_observability::init("ax-serving-api", tracing::Level::INFO)?;

    use ax_serving_api::config::ServeConfig;
    use ax_serving_api::orchestration::start_orchestrator;

    // Load config from YAML (with env-var overrides).
    let mut serve_config = ServeConfig::load_default()?;
    if let Some(ref h) = cli.host {
        serve_config.orchestrator.host = h.clone();
    }
    if let Some(p) = cli.port {
        serve_config.orchestrator.port = p;
    }
    if let Some(p) = cli.internal_port {
        serve_config.orchestrator.internal_port = p;
    }
    if let Some(ref pol) = cli.policy {
        serve_config.orchestrator.dispatch_policy = pol.clone();
    }
    serve_config.validate()?;

    let config = serve_config.orchestrator;
    let project_policy = serve_config.project_policy;

    // Keep advertiser alive for process lifetime (agents browse internal port).
    let _lan = maybe_start_gateway_advertise(&cli, &config)?;

    eprintln!(
        "[ax-serving-api] starting: mode=direct public={}:{} internal={}:{} policy={}",
        config.host,
        config.port,
        config.internal_bind_addr,
        config.internal_port,
        config.dispatch_policy,
    );

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(start_orchestrator(config, project_policy))
}

fn maybe_start_gateway_advertise(
    cli: &Cli,
    config: &ax_serving_api::config::OrchestratorConfig,
) -> Result<Option<ax_serving_discovery::LanAdvertiser>> {
    let advertise = cli.advertise_lan
        || ax_serving_discovery::env_truthy("AXS_ADVERTISE_LAN")
        || ax_serving_discovery::env_truthy("AXS_GATEWAY_ADVERTISE_LAN");
    if !advertise {
        return Ok(None);
    }

    let bind_host = config.internal_bind_addr.as_str();
    if matches!(bind_host, "127.0.0.1" | "localhost" | "::1") {
        anyhow::bail!(
            "--advertise-lan requires a non-loopback orchestrator.internal_bind_addr \
             (agents must reach the control plane from the LAN)"
        );
    }

    let explicit_host = cli
        .lan_advertise_host
        .clone()
        .or_else(|| std::env::var("AXS_LAN_ADVERTISE_HOST").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let advertise_ip =
        ax_serving_discovery::pick_advertise_ipv4(explicit_host.as_deref(), bind_host)?;
    let cluster = cli
        .lan_cluster
        .clone()
        .or_else(|| std::env::var("AXS_LAN_CLUSTER").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let instance_name = cli
        .lan_instance_name
        .clone()
        .or_else(|| std::env::var("AXS_LAN_INSTANCE_NAME").ok())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "ax-serving-gateway".into());
    let instance_id = format!(
        "axsgw-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let auth_required = std::env::var("AXS_WORKER_TOKEN")
        .ok()
        .or_else(|| std::env::var("AXS_INTERNAL_API_TOKEN").ok())
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);

    let advertiser = ax_serving_discovery::LanAdvertiser::start_gateway(
        &instance_name,
        config.internal_port,
        advertise_ip,
        env!("CARGO_PKG_VERSION"),
        auth_required,
        cluster,
        instance_id,
    )?;
    eprintln!(
        "[ax-serving-api] LAN mDNS advertise on {advertise_ip}:{} (_ax-serving-gateway._tcp)",
        config.internal_port
    );
    Ok(Some(advertiser))
}
