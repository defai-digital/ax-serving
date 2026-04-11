use std::path::PathBuf;
use std::sync::Arc;

use std::sync::atomic::Ordering;

use anyhow::{Context as _, Result};
use ax_serving_api::ServingLayer;
use ax_serving_engine::{LoadConfig, RouterBackend, RoutingConfig};

// ── System identity helpers ───────────────────────────────────────────────────

/// Returns the macOS computer name (e.g. "Aki's MacBook Pro") via `scutil`.
///
/// Falls back to the `HOSTNAME` env var, then `"unknown"`.
fn get_friendly_name() -> String {
    std::process::Command::new("scutil")
        .args(["--get", "ComputerName"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()))
}

/// Returns the Apple Silicon chip model (e.g. "Apple M3 Pro") via `system_profiler`.
///
/// Returns `None` if the command fails or the field is absent.
fn get_chip_model() -> Option<String> {
    let output = std::process::Command::new("system_profiler")
        .arg("SPHardwareDataType")
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.lines()
        .find(|l| l.contains("Chip:"))
        .and_then(|l| l.split_once(':').map(|(_, r)| r))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// ── Orchestrator auto-registration helpers ────────────────────────────────────

/// State returned from a successful orchestrator registration.
struct WorkerReg {
    orchestrator_addr: String,
    worker_id: String,
    heartbeat_interval_ms: u64,
}

struct HeartbeatConfig {
    orchestrator_addr: String,
    worker_id: String,
    internal_api_token: Option<String>,
    interval_ms: u64,
    self_addr: String,
    capabilities: Vec<String>,
    max_inflight: usize,
    friendly_name: String,
    chip_model: Option<String>,
    worker_pool: Option<String>,
    node_class: Option<String>,
}

struct RegisterConfig<'a> {
    capabilities: &'a [String],
    max_inflight: usize,
    friendly_name: &'a str,
    chip_model: Option<&'a str>,
    worker_pool: Option<&'a str>,
    node_class: Option<&'a str>,
    internal_api_token: Option<&'a str>,
}

/// Register this worker with the orchestrator.
///
/// `self_addr` is the `host:port` address the orchestrator will use to reach
/// this worker (e.g. `"127.0.0.1:18081"`).
async fn register_with_orchestrator(
    client: &reqwest::Client,
    orchestrator_addr: &str,
    self_addr: &str,
    cfg: RegisterConfig<'_>,
) -> Result<WorkerReg> {
    let url = format!("{orchestrator_addr}/internal/workers/register");
    let body = serde_json::json!({
        "addr": self_addr,
        "capabilities": cfg.capabilities,
        "backend": "auto",
        "max_inflight": cfg.max_inflight,
        "friendly_name": cfg.friendly_name,
        "chip_model": cfg.chip_model,
        "worker_pool": cfg.worker_pool,
        "node_class": cfg.node_class,
    });
    let mut req = client.post(&url).json(&body);
    if let Some(token) = cfg.internal_api_token {
        req = req.header("X-Internal-Token", token);
    }
    let resp = req
        .send()
        .await
        .context("failed to POST /internal/workers/register")?;

    let reg: serde_json::Value = resp
        .error_for_status()
        .context("orchestrator returned error status for registration")?
        .json()
        .await
        .context("failed to parse registration response")?;

    let worker_id = reg["worker_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing worker_id in registration response"))?
        .to_string();
    let heartbeat_interval_ms = reg["heartbeat_interval_ms"].as_u64().unwrap_or(5_000);

    Ok(WorkerReg {
        orchestrator_addr: orchestrator_addr.to_string(),
        worker_id,
        heartbeat_interval_ms,
    })
}

/// Background task: send heartbeats to the orchestrator at the specified interval.
///
/// Each heartbeat carries live metrics read from the `ServingLayer`:
/// - `inflight`      — actual in-flight request count from the scheduler
/// - `thermal_state` — current thermal state from the inference backend
/// - `model_ids`     — currently loaded model IDs (allows orchestrator to track
///   models loaded/unloaded after initial registration)
/// - `rss_bytes`     — process RSS so the orchestrator can show memory per worker
///
/// On 404 or 410 (orchestrator restarted and evicted this worker), the loop
/// automatically re-registers so the worker becomes routable again without an
/// operator restart.
///
/// Runs until aborted (`JoinHandle::abort()`).
async fn heartbeat_loop(
    client: reqwest::Client,
    layer: Arc<ServingLayer>,
    mut cfg: HeartbeatConfig,
) {
    let mut rereg_backoff: u32 = 0;
    loop {
        let sleep_ms = if rereg_backoff > 0 {
            // Exponential backoff for re-registration retries: 2^n seconds, capped at 30s.
            let backoff_ms = (1u64 << rereg_backoff.min(5)) * 1000;
            backoff_ms.min(30_000)
        } else {
            cfg.interval_ms
        };
        tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;

        let inflight = layer
            .scheduler
            .metrics
            .inflight_count
            .load(Ordering::Relaxed)
            .max(0) as usize;
        let thermal_state = format!("{:?}", layer.backend.thermal_state()).to_lowercase();
        let model_ids = layer.registry.list_ids();
        let rss_bytes = ax_serving_api::metrics::current_rss_bytes();
        // WS4: richer telemetry for TokenCostPolicy scoring.
        // When split scheduler is enabled, use decode_sequences_active (requests past
        // prefill) rather than the total inflight count, which includes prefill-phase
        // requests that haven't yet consumed KV capacity.
        let active_sequences = if layer.scheduler.split_enabled {
            layer
                .scheduler
                .metrics
                .decode_sequences_active
                .load(Ordering::Relaxed)
                .max(0) as usize
        } else {
            inflight
        };
        let ttft_p95_ms = (layer.scheduler.metrics.ttft_p95_us() + 500) / 1000;
        let decode_tok_per_sec = layer.metrics.recent_decode_tok_per_sec();

        let url = format!(
            "{}/internal/workers/{}/heartbeat",
            cfg.orchestrator_addr, cfg.worker_id
        );
        let body = serde_json::json!({
            "inflight": inflight,
            "thermal_state": thermal_state,
            "model_ids": model_ids,
            "rss_bytes": rss_bytes,
            "active_sequences": active_sequences,
            "decode_tok_per_sec": decode_tok_per_sec,
            "ttft_p95_ms": ttft_p95_ms,
        });

        let mut req = client.post(&url).json(&body);
        if let Some(token) = cfg.internal_api_token.as_deref() {
            req = req.header("X-Internal-Token", token);
        }

        match req.send().await {
            Err(e) => tracing::warn!(%e, "heartbeat to orchestrator failed"),
            Ok(resp) if matches!(resp.status().as_u16(), 404 | 410) => {
                // Orchestrator restarted and evicted this worker.  Re-register
                // so the worker becomes routable again without an operator restart.
                tracing::warn!(
                    status = %resp.status(),
                    "heartbeat rejected — orchestrator evicted this worker, re-registering"
                );
                match register_with_orchestrator(
                    &client,
                    &cfg.orchestrator_addr,
                    &cfg.self_addr,
                    RegisterConfig {
                        capabilities: &cfg.capabilities,
                        max_inflight: cfg.max_inflight,
                        friendly_name: &cfg.friendly_name,
                        chip_model: cfg.chip_model.as_deref(),
                        worker_pool: cfg.worker_pool.as_deref(),
                        node_class: cfg.node_class.as_deref(),
                        internal_api_token: cfg.internal_api_token.as_deref(),
                    },
                )
                .await
                {
                    Ok(r) => {
                        tracing::info!(
                            new_worker_id = %r.worker_id,
                            "re-registered with orchestrator after eviction"
                        );
                        cfg.worker_id = r.worker_id;
                        cfg.interval_ms = r.heartbeat_interval_ms;
                        rereg_backoff = 0;
                    }
                    Err(e) => {
                        rereg_backoff = rereg_backoff.saturating_add(1);
                        tracing::warn!(
                            %e,
                            backoff_secs = (1u64 << rereg_backoff.min(5)),
                            "re-registration failed, will retry with backoff"
                        );
                    }
                }
            }
            Ok(resp) if !resp.status().is_success() => {
                tracing::warn!(
                    status = %resp.status(),
                    "heartbeat rejected by orchestrator"
                );
            }
            Ok(_) => {}
        }
    }
}

/// Signal drain to the orchestrator: stop routing new requests to this worker.
///
/// Must be called BEFORE `drain_complete` so the orchestrator does not
/// dispatch new requests to the shutting-down worker (which would produce
/// connection-refused errors and trigger unnecessary reroutes).
async fn drain_worker(
    client: &reqwest::Client,
    orchestrator_addr: &str,
    worker_id: &str,
    internal_api_token: Option<&str>,
) -> Result<()> {
    let url = format!("{orchestrator_addr}/internal/workers/{worker_id}/drain");
    let mut req = client.post(&url);
    if let Some(token) = internal_api_token {
        req = req.header("X-Internal-Token", token);
    }
    req.send()
        .await
        .context("drain request failed")?
        .error_for_status()
        .context("drain returned error status")?;
    Ok(())
}

/// Evict this worker from the orchestrator registry (best-effort on shutdown).
///
/// Always call `drain_worker` first to stop new request routing before calling
/// this function.
async fn drain_complete(
    client: &reqwest::Client,
    orchestrator_addr: &str,
    worker_id: &str,
    internal_api_token: Option<&str>,
) -> Result<()> {
    let url = format!("{orchestrator_addr}/internal/workers/{worker_id}/drain-complete");
    let mut req = client.post(&url);
    if let Some(token) = internal_api_token {
        req = req.header("X-Internal-Token", token);
    }
    req.send()
        .await
        .context("drain-complete request failed")?
        .error_for_status()
        .context("drain-complete returned error status")?;
    Ok(())
}

// ── run_serve ─────────────────────────────────────────────────────────────────

pub(crate) fn run_serve(
    model: Option<PathBuf>,
    model_id: String,
    host: Option<String>,
    port: Option<u16>,
    serve_config_path: Option<PathBuf>,
    routing_config_path: Option<PathBuf>,
    orchestrator: Option<String>,
) -> Result<()> {
    use ax_serving_api::{ServingLayer, config::ServeConfig, start_servers};

    // Load routing config: explicit --routing-config path > AXS_ROUTING_CONFIG > ./backends.yaml.
    let routing_cfg = if let Some(path) = routing_config_path {
        RoutingConfig::from_file(&path)
            .with_context(|| format!("loading routing config from {}", path.display()))?
    } else {
        RoutingConfig::load_default()
    };

    let config_base = if let Some(path) = serve_config_path {
        ServeConfig::from_file(&path)?
    } else {
        ServeConfig::load_default()
    };
    let (cfg_host, cfg_port) = parse_rest_addr(&config_base.rest_addr)?;
    let host = host.unwrap_or(cfg_host);
    let port = port.unwrap_or(cfg_port);
    let config = ServeConfig {
        rest_addr: format!("{host}:{port}"),
        ..config_base
    };
    config.validate()?;
    let backend: Arc<dyn ax_serving_engine::InferenceBackend> = Arc::new(RouterBackend::new(
        routing_cfg,
        config.llamacpp.clone(),
        config.mlx.clone(),
    ));

    let layer = Arc::new(ServingLayer::new(backend.clone(), config.clone()));

    // Print identity info to stderr so operators can identify this worker node.
    tracing::info!(%host, %port, "worker starting");

    // Preload model (sync, before starting the async runtime).
    let capabilities: Vec<String> = if let Some(path) = model {
        let load_config = LoadConfig::default();
        layer
            .registry
            .load(&model_id, &path, load_config, backend.as_ref())?;
        tracing::info!(%model_id, "preloaded model");
        vec![model_id.clone()]
    } else {
        vec![]
    };

    // Orchestrator address from --orchestrator flag or AXS_ORCHESTRATOR_ADDR env var.
    let orchestrator_addr = orchestrator
        .or_else(|| std::env::var("AXS_ORCHESTRATOR_ADDR").ok())
        .map(|addr| crate::normalize_http_base_url(&addr, "orchestrator"))
        .transpose()?;
    let internal_api_token = std::env::var("AXS_INTERNAL_API_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // Gather system identity once at startup (blocking I/O, safe before tokio).
    let friendly_name = get_friendly_name();
    let chip_model = get_chip_model();
    let worker_pool = std::env::var("AXS_WORKER_POOL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let node_class = std::env::var("AXS_WORKER_NODE_CLASS")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // Max inflight reported to the orchestrator.  This must not overstate the
    // serving layer's own admission limits or the orchestrator will route
    // requests that the worker immediately rejects with 503.  Clamp the
    // advertised value to the effective scheduler capacity.
    let requested_max_inflight: usize = std::env::var("AXS_WORKER_MAX_INFLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| default_advertised_max_inflight(&config));
    let max_inflight = clamp_advertised_max_inflight(
        requested_max_inflight,
        config.sched_max_inflight,
        config.sched_per_model_max_inflight,
    );
    if max_inflight != requested_max_inflight {
        tracing::warn!(
            requested_max_inflight,
            max_inflight,
            "clamping advertised max_inflight to match scheduler capacity"
        );
    }

    // The address the orchestrator uses to reach this worker over loopback.
    let self_addr = format!("{host}:{port}");

    // Start servers on a multi-thread runtime.  The async block handles
    // orchestrator registration, runs the servers, then does cleanup.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .context("failed to build HTTP client for orchestrator")?;

            // Register with orchestrator (PRD §FR-1.2).
            let reg = if let Some(ref addr) = orchestrator_addr {
                match register_with_orchestrator(
                    &client,
                    addr,
                    &self_addr,
                    RegisterConfig {
                        capabilities: &capabilities,
                        max_inflight,
                        friendly_name: &friendly_name,
                        chip_model: chip_model.as_deref(),
                        worker_pool: worker_pool.as_deref(),
                        node_class: node_class.as_deref(),
                        internal_api_token: internal_api_token.as_deref(),
                    },
                )
                .await
                {
                    Ok(r) => {
                        tracing::info!(
                            "[ax-serving] registered with orchestrator {addr} as worker {}",
                            r.worker_id
                        );
                        Some(r)
                    }
                    Err(e) => {
                        tracing::warn!(%e, "orchestrator registration failed");
                        None
                    }
                }
            } else {
                None
            };

            // Spawn heartbeat background task with live metrics from ServingLayer.
            let hb_handle = reg.as_ref().map(|r| {
                let cfg = HeartbeatConfig {
                    orchestrator_addr: r.orchestrator_addr.clone(),
                    worker_id: r.worker_id.clone(),
                    internal_api_token: internal_api_token.clone(),
                    interval_ms: r.heartbeat_interval_ms,
                    self_addr: self_addr.clone(),
                    capabilities: capabilities.clone(),
                    max_inflight,
                    friendly_name: friendly_name.clone(),
                    chip_model: chip_model.clone(),
                    worker_pool: worker_pool.clone(),
                    node_class: node_class.clone(),
                };
                tokio::spawn(heartbeat_loop(client.clone(), Arc::clone(&layer), cfg))
            });

            // Run servers until SIGINT / SIGTERM.
            let server_result = start_servers(layer, &config).await;

            // Graceful shutdown (always runs, even on start_servers error — BUG-091):
            // 1. Abort heartbeat so the orchestrator's TTL timer can expire naturally.
            // 2. POST drain — tell orchestrator to stop routing new requests here.
            //    (Server is already down; this prevents further reroute churn.)
            // 3. POST drain-complete — evict from registry immediately.
            if let Some(h) = hb_handle {
                h.abort();
            }
            if let Some(ref r) = reg {
                if let Err(e) = drain_worker(
                    &client,
                    &r.orchestrator_addr,
                    &r.worker_id,
                    internal_api_token.as_deref(),
                )
                .await
                {
                    tracing::warn!(%e, "drain request failed");
                }
                if let Err(e) = drain_complete(
                    &client,
                    &r.orchestrator_addr,
                    &r.worker_id,
                    internal_api_token.as_deref(),
                )
                .await
                {
                    tracing::warn!(%e, "drain-complete request failed");
                } else {
                    tracing::info!("drain-complete sent to orchestrator");
                }
            }

            server_result
        })
}

fn default_advertised_max_inflight(config: &ax_serving_api::config::ServeConfig) -> usize {
    config
        .sched_max_inflight
        .max(1)
        .min(config.sched_per_model_max_inflight.max(1))
}

fn clamp_advertised_max_inflight(
    requested: usize,
    sched_max_inflight: usize,
    sched_per_model_max_inflight: usize,
) -> usize {
    requested
        .max(1)
        .min(sched_max_inflight.max(1))
        .min(sched_per_model_max_inflight.max(1))
}

fn parse_rest_addr(addr: &str) -> Result<(String, u16)> {
    let addr = addr.trim().trim_end_matches('/');
    let lowered = addr.to_ascii_lowercase();
    if lowered.starts_with("http://") || lowered.starts_with("https://") {
        anyhow::bail!("invalid rest_addr (URL scheme is not supported): {addr}");
    }
    if addr.contains('/') {
        anyhow::bail!("invalid rest_addr (path is not supported): {addr}");
    }
    let (host, port) = addr
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("invalid rest_addr (missing ':'): {addr}"))?;
    let host = host.trim();
    if host.is_empty() {
        anyhow::bail!("invalid rest_addr (empty host): {addr}");
    }
    let port = port
        .trim()
        .parse::<u16>()
        .with_context(|| format!("invalid rest_addr port in '{addr}'"))?;
    Ok((host.to_string(), port))
}

#[cfg(test)]
mod tests {
    use super::{clamp_advertised_max_inflight, default_advertised_max_inflight, parse_rest_addr};
    use ax_serving_api::config::ServeConfig;

    #[test]
    fn parse_rest_addr_accepts_ipv4() {
        let (host, port) = parse_rest_addr("127.0.0.1:18080").expect("should parse ipv4");
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 18080);
    }

    #[test]
    fn parse_rest_addr_accepts_ipv6_bracketed() {
        let (host, port) = parse_rest_addr("[::1]:18080").expect("should parse bracketed ipv6");
        assert_eq!(host, "[::1]");
        assert_eq!(port, 18080);
    }

    #[test]
    fn parse_rest_addr_rejects_empty_host() {
        let err = parse_rest_addr(":18080").expect_err("empty host should fail");
        assert!(err.to_string().contains("empty host"));
    }

    #[test]
    fn parse_rest_addr_rejects_url_scheme() {
        let err = parse_rest_addr("http://127.0.0.1:18080")
            .expect_err("rest_addr with URL scheme should fail");
        assert!(err.to_string().contains("URL scheme"));
    }

    #[test]
    fn parse_rest_addr_rejects_uppercase_url_scheme() {
        let err = parse_rest_addr("HTTPS://127.0.0.1:18080")
            .expect_err("rest_addr with uppercase URL scheme should fail");
        assert!(err.to_string().contains("URL scheme"));
    }

    #[test]
    fn parse_rest_addr_rejects_path() {
        let err =
            parse_rest_addr("127.0.0.1:18080/path").expect_err("rest_addr with path should fail");
        assert!(err.to_string().contains("path"));
    }

    #[test]
    fn default_advertised_max_inflight_matches_scheduler_limits() {
        let cfg = ServeConfig {
            sched_max_inflight: 16,
            sched_per_model_max_inflight: 4,
            ..ServeConfig::default()
        };
        assert_eq!(default_advertised_max_inflight(&cfg), 4);
    }

    #[test]
    fn clamp_advertised_max_inflight_caps_to_per_model_limit() {
        assert_eq!(clamp_advertised_max_inflight(8, 16, 4), 4);
    }

    #[test]
    fn clamp_advertised_max_inflight_preserves_safe_values() {
        assert_eq!(clamp_advertised_max_inflight(2, 16, 4), 2);
    }

    #[test]
    fn clamp_advertised_max_inflight_enforces_minimum_one() {
        assert_eq!(clamp_advertised_max_inflight(0, 0, 0), 1);
    }
}
