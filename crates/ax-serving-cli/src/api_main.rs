//! ax-serving-api: multi-worker API gateway for ax-serving.
//!
//! Starts the orchestrator that proxies inference requests across registered
//! `ax-serving serve` worker nodes. Holds no model weights and no Metal context.
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
//!   AXS_DISPATCH_POLICY     — worker selection policy (default: least_inflight)
//!                             choices: least_inflight | weighted_round_robin | model_affinity
//!   AXS_WORKER_HEARTBEAT_MS — heartbeat interval hint sent to workers (default: 5000)
//!   AXS_WORKER_TTL_MS       — eviction TTL for silent workers (default: 15000)
//!   AXS_GLOBAL_QUEUE_MAX    — max concurrent requests before overload policy triggers (default: 128)
//!   AXS_GLOBAL_QUEUE_WAIT_MS — max queue wait before 503 (default: 10000)
//!   AXS_LOG                 — tracing filter, e.g. "debug" or "ax_serving_api=trace"

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-serving-api only supports aarch64-apple-darwin (Apple Silicon M3+)");

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "ax-serving-api",
    about = "AX Serving API gateway — routes requests across ax-serving worker nodes",
    long_about = "\
Start the multi-worker API gateway. The gateway is a pure dispatch process:\n\
it holds no model weights and starts no Metal context. Worker nodes are\n\
registered via POST /internal/workers/register (done automatically by\n\
`ax-serving serve --orchestrator <addr>`).\n\
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
    /// Choices: least_inflight (default), weighted_round_robin, model_affinity.
    /// Overrides AXS_DISPATCH_POLICY.
    #[arg(long)]
    policy: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_env("AXS_LOG")
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    use ax_serving_api::config::ServeConfig;
    use ax_serving_api::orchestration::start_orchestrator;

    // Load config from YAML (with env-var overrides).
    let serve_config = ServeConfig::load_default();
    let mut config = serve_config.orchestrator;
    let license_config = serve_config.license;
    if let Some(h) = cli.host {
        config.host = h;
    }
    if let Some(p) = cli.port {
        config.port = p;
    }
    if let Some(p) = cli.internal_port {
        config.internal_port = p;
    }
    if let Some(pol) = cli.policy {
        config.dispatch_policy = pol;
    }

    eprintln!(
        "[ax-serving-api] starting: mode=direct public={}:{} internal=127.0.0.1:{} policy={}",
        config.host, config.port, config.internal_port, config.dispatch_policy,
    );

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(start_orchestrator(config, license_config))
}
