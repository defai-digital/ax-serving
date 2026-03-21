//! ax-serving: inference CLI and worker entrypoint for ax-serving.
//!
//! Supports:
//!   ax-serving -m model.gguf -p "prompt" -n 100     # single inference
//!   ax-serving serve -m model.gguf --port 18080       # start HTTP + gRPC worker
//!
//! To start the multi-worker API gateway, use `ax-serving-api` instead.

mod thor;

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-serving-cli only supports aarch64-apple-darwin (Apple Silicon M3+)");

use std::path::PathBuf;
use std::sync::Arc;

use std::sync::atomic::Ordering;

use anyhow::{Context as _, Result};
use ax_serving_api::ServingLayer;
use ax_serving_engine::{BackendType, LoadConfig, RouterBackend, RoutingConfig};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "ax-serving",
    about = "AX Serving — LLM inference and worker node for Apple Silicon",
    long_about = "\
Run a single inference or start an OpenAI-compatible worker node.\n\
\n\
To run the multi-worker API gateway, use `ax-serving-api`."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    // ── llama.cpp-compatible inference flags (default mode) ──────────────────
    /// Path to GGUF model file.
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// Input prompt.
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Max tokens to generate.
    #[arg(short = 'n', long, default_value = "512")]
    n_predict: u32,

    /// Context window size (0 = model default).
    #[arg(short = 'c', long, default_value = "0")]
    ctx_size: u32,

    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value = "0.7")]
    temp: f32,

    /// Top-k sampling (0 = disabled).
    #[arg(long, default_value = "40")]
    top_k: u32,

    /// Top-p nucleus sampling.
    #[arg(long, default_value = "0.9")]
    top_p: f32,

    /// Repetition penalty.
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,

    /// Random seed (0 = random).
    #[arg(long, default_value = "0")]
    seed: u64,

    /// Number of GPU layers (-1 = all).
    #[arg(long, default_value = "-1")]
    n_gpu_layers: i32,

    /// Print per-token timing and throughput summary.
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Wrap prompt in model-specific chat template.
    #[arg(long)]
    chat: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start an OpenAI REST + gRPC worker node.
    ///
    /// Optionally preloads a model and registers with a running `ax-serving-api`
    /// orchestrator for multi-worker deployments.
    ///
    /// Key env vars:
    ///   AXS_ORCHESTRATOR_ADDR   — orchestrator internal URL for auto-registration
    ///   AXS_WORKER_MAX_INFLIGHT — max concurrent requests this worker accepts (default: 8)
    Serve {
        /// Path to GGUF model file to preload.
        #[arg(short = 'm', long)]
        model: Option<PathBuf>,

        /// Model ID for the registry.
        #[arg(long, default_value = "default")]
        model_id: String,

        /// REST HTTP port.
        ///
        /// If omitted, uses `rest_addr` from serving config/env (default: 18080).
        #[arg(long)]
        port: Option<u16>,

        /// Bind address.
        ///
        /// If omitted, uses `rest_addr` host from serving config/env (default: 127.0.0.1).
        #[arg(long)]
        host: Option<String>,

        /// Path to serving config file (.yaml/.yml/.toml).
        #[arg(long)]
        config: Option<PathBuf>,

        /// Path to backends.yaml routing config (overrides AXS_ROUTING_CONFIG env var).
        #[arg(long)]
        routing_config: Option<PathBuf>,

        /// Orchestrator internal URL for auto-registration (e.g. http://127.0.0.1:19090).
        /// Can also be set via AXS_ORCHESTRATOR_ADDR.
        /// When set, this worker registers on startup, sends periodic heartbeats,
        /// and signals drain-complete on shutdown.
        #[arg(long)]
        orchestrator: Option<String>,
    },
    /// Prepare and inspect a managed Thor worker node.
    Thor {
        #[command(subcommand)]
        command: ThorCommand,
    },
}

#[derive(Subcommand, Debug)]
enum ThorCommand {
    /// Write the local ax-thor-agent environment file and run basic preflight checks.
    Install {
        /// Optional control-plane internal URL (e.g. http://127.0.0.1:19090).
        #[arg(long)]
        control_plane: Option<String>,
        /// Local thor-agent listen address.
        #[arg(long, default_value = "0.0.0.0:18081")]
        listen_addr: String,
        /// Worker address the control plane should route to.
        #[arg(long)]
        advertised_addr: Option<String>,
        /// Local SGLang base URL.
        #[arg(long, default_value = "http://127.0.0.1:30000")]
        sglang_url: String,
        /// Internal worker token for control-plane auth.
        #[arg(long)]
        worker_token: Option<String>,
        /// Max concurrent requests this Thor worker should advertise.
        #[arg(long, default_value_t = 8)]
        max_inflight: usize,
        /// Optional worker pool label.
        #[arg(long)]
        worker_pool: Option<String>,
        /// Optional node class label.
        #[arg(long, default_value = "thor")]
        node_class: String,
        /// Optional friendly node name.
        #[arg(long)]
        friendly_name: Option<String>,
        /// Optional chip model override.
        #[arg(long)]
        chip_model: Option<String>,
        /// Output env-file path. Defaults to ~/.config/ax-serving/thor.env
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Update Thor control-plane settings and validate registration readiness.
    Join {
        /// Control-plane internal URL (e.g. http://127.0.0.1:19090).
        #[arg(long)]
        control_plane: String,
        /// Worker address the control plane should route to.
        #[arg(long)]
        advertised_addr: Option<String>,
        /// Local thor-agent listen address.
        #[arg(long)]
        listen_addr: Option<String>,
        /// Local SGLang base URL.
        #[arg(long)]
        sglang_url: Option<String>,
        /// Internal worker token for control-plane auth.
        #[arg(long)]
        worker_token: Option<String>,
        /// Max concurrent requests this Thor worker should advertise.
        #[arg(long)]
        max_inflight: Option<usize>,
        /// Optional worker pool label.
        #[arg(long)]
        worker_pool: Option<String>,
        /// Optional node class label.
        #[arg(long)]
        node_class: Option<String>,
        /// Optional friendly node name.
        #[arg(long)]
        friendly_name: Option<String>,
        /// Optional chip model override.
        #[arg(long)]
        chip_model: Option<String>,
        /// Env-file path. Defaults to ~/.config/ax-serving/thor.env
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Show Thor local health and control-plane-visible registration state.
    Status {
        /// Env-file path. Defaults to ~/.config/ax-serving/thor.env
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_filter =
        tracing_subscriber::EnvFilter::from_env("AXS_LOG").add_directive(if cli.verbose {
            tracing::Level::DEBUG.into()
        } else {
            tracing::Level::WARN.into()
        });
    let log_format = std::env::var("AXS_LOG_FORMAT").unwrap_or_else(|_| "text".into());
    if log_format == "json" {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(log_filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(log_filter).init();
    }

    match cli.command {
        Some(Command::Serve {
            model,
            model_id,
            port,
            host,
            config,
            routing_config,
            orchestrator,
        }) => run_serve(
            model,
            model_id,
            host,
            port,
            config,
            routing_config,
            orchestrator,
        ),
        Some(Command::Thor { command }) => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async move {
                match command {
                    ThorCommand::Install {
                        control_plane,
                        listen_addr,
                        advertised_addr,
                        sglang_url,
                        worker_token,
                        max_inflight,
                        worker_pool,
                        node_class,
                        friendly_name,
                        chip_model,
                        output,
                    } => thor::install(thor::InstallArgs {
                        control_plane,
                        listen_addr,
                        advertised_addr,
                        sglang_url,
                        worker_token,
                        max_inflight,
                        worker_pool,
                        node_class,
                        friendly_name,
                        chip_model,
                        output,
                    }),
                    ThorCommand::Join {
                        control_plane,
                        advertised_addr,
                        listen_addr,
                        sglang_url,
                        worker_token,
                        max_inflight,
                        worker_pool,
                        node_class,
                        friendly_name,
                        chip_model,
                        output,
                    } => {
                        thor::join(thor::JoinArgs {
                            control_plane,
                            advertised_addr,
                            listen_addr,
                            sglang_url,
                            worker_token,
                            max_inflight,
                            worker_pool,
                            node_class,
                            friendly_name,
                            chip_model,
                            output,
                        })
                        .await
                    }
                    ThorCommand::Status { config } => {
                        thor::status(thor::StatusArgs { config }).await
                    }
                }
            })
        }
        None => {
            // Inference mode: MistralrsBackend owns its own runtime internally.
            // Run everything synchronously to avoid nested-runtime panics.
            let model = cli
                .model
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--model (-m) required"))?
                .to_path_buf();
            let prompt = cli.prompt.clone().unwrap_or_default();
            run_inference(model, prompt, &cli)
        }
    }
}

fn run_inference(model_path: PathBuf, prompt: String, cli: &Cli) -> Result<()> {
    use ax_serving_engine::{
        GenerateEvent, GenerateInput, GenerationParams, InferenceBackend as _,
    };

    // Inference mode uses RouterBackend (same as serve mode).
    let backend = RouterBackend::from_env();
    let config = LoadConfig {
        context_length: cli.ctx_size,
        backend_type: if cli.n_gpu_layers == 0 {
            BackendType::Cpu
        } else {
            BackendType::Auto
        },
        llama_cpp_n_gpu_layers: Some(cli.n_gpu_layers),
        mmproj_path: None,
        backend_hint: None,
        enable_embeddings: None,
        pooling_type: None,
    };

    // load_model uses backend's internal runtime — safe in sync context.
    let (handle, meta) = backend.load_model(&model_path, config)?;

    if cli.verbose {
        eprintln!(
            "[ax-serving] loaded {} in {}ms (ctx={})",
            meta.architecture, meta.load_time_ms, meta.context_length
        );
    }

    let params = GenerationParams {
        stream: true,
        temperature: if cli.temp == 0.0 {
            None
        } else {
            Some(cli.temp as f64)
        },
        top_p: Some(cli.top_p as f64),
        top_k: if cli.top_k == 0 {
            None
        } else {
            Some(cli.top_k as usize)
        },
        max_tokens: Some(cli.n_predict as usize),
        stop_seqs: Vec::new(),
        seed: None,
        repeat_penalty: None,
        ..Default::default()
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<GenerateEvent>(512);

    // generate() spawns on the backend's internal runtime and returns immediately.
    backend.generate(handle, GenerateInput::Text(prompt), params, tx)?;

    // Drain the event channel with a small single-thread runtime — entirely
    // separate from the backend's runtime, so no nesting.
    let gen_start = std::time::Instant::now();
    let mut n_tokens = 0usize;
    let verbose = cli.verbose;

    let stats = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let mut rx = rx;
            let mut final_stats = None;
            while let Some(event) = rx.recv().await {
                match event {
                    GenerateEvent::Token(text) => {
                        print!("{text}");
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                        n_tokens += 1;
                    }
                    GenerateEvent::Done(s) => {
                        println!();
                        final_stats = Some((s, n_tokens));
                        break;
                    }
                    GenerateEvent::Error(e) => {
                        eprintln!("\n[ax-serving] error: {e}");
                        return Err(anyhow::anyhow!(e));
                    }
                    GenerateEvent::TokenLogprob { .. } | GenerateEvent::ToolCall { .. } => {}
                }
            }
            Ok(final_stats)
        })?;

    if verbose && let Some((s, n)) = stats {
        let elapsed = gen_start.elapsed().as_secs_f64();
        eprintln!(
            "\n[ax-serving] {} tokens | prefill {:.1} tok/s | decode {:.1} tok/s | wall {:.2}s ({:.1} tok/s)",
            s.completion_tokens,
            s.prefill_tok_per_sec,
            s.decode_tok_per_sec,
            elapsed,
            n as f64 / elapsed,
        );
    }

    // backend dropped here — safe, we're in sync context.
    Ok(())
}

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
    let interval = std::time::Duration::from_millis(cfg.interval_ms);
    loop {
        tokio::time::sleep(interval).await;

        let inflight = layer
            .scheduler
            .metrics
            .inflight_count
            .load(Ordering::Relaxed)
            .max(0) as usize;
        let thermal_state = format!("{:?}", layer.backend.thermal_state()).to_lowercase();
        let model_ids = layer.registry.list_ids();
        let rss_bytes = ax_serving_api::metrics::current_rss_bytes();
        // WS4: richer telemetry for TokenCostPolicy scoring
        let active_sequences = inflight; // active_sequences == inflight for single-stream model
        let ttft_p95_ms = layer.scheduler.metrics.ttft_p95_us() / 1000;
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
                    }
                    Err(e) => {
                        tracing::warn!(%e, "re-registration failed, will retry next heartbeat");
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

fn run_serve(
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
    let backend: Arc<dyn ax_serving_engine::InferenceBackend> =
        Arc::new(RouterBackend::new(routing_cfg, config.llamacpp.clone()));

    let layer = Arc::new(ServingLayer::new(backend.clone(), config.clone()));

    // Print identity info to stderr so operators can identify this worker node.
    eprintln!("[ax-serving] worker starting on {host}:{port}");

    // Preload model (sync, before starting the async runtime).
    let capabilities: Vec<String> = if let Some(path) = model {
        let load_config = LoadConfig::default();
        layer
            .registry
            .load(&model_id, &path, load_config, backend.as_ref())?;
        eprintln!("[ax-serving] preloaded '{model_id}'");
        vec![model_id.clone()]
    } else {
        vec![]
    };

    // Orchestrator address from --orchestrator flag or AXS_ORCHESTRATOR_ADDR env var.
    let orchestrator_addr = orchestrator.or_else(|| std::env::var("AXS_ORCHESTRATOR_ADDR").ok());
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

    // Max inflight reported to the orchestrator.  Must be at least 1: zero
    // would make this worker permanently unselectable by WeightedRoundRobin
    // (weight = max_inflight - inflight = 0) while LeastInflight would still
    // route to it, causing inconsistent policy behaviour.
    let max_inflight: usize = std::env::var("AXS_WORKER_MAX_INFLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1);

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
                        eprintln!(
                            "[ax-serving] registered with orchestrator {addr} as worker {}",
                            r.worker_id
                        );
                        Some(r)
                    }
                    Err(e) => {
                        eprintln!("[ax-serving] WARNING: orchestrator registration failed: {e}");
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
            start_servers(layer, &config).await?;

            // Graceful shutdown:
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
                    eprintln!("[ax-serving] drain warning: {e}");
                }
                if let Err(e) = drain_complete(
                    &client,
                    &r.orchestrator_addr,
                    &r.worker_id,
                    internal_api_token.as_deref(),
                )
                .await
                {
                    eprintln!("[ax-serving] drain-complete warning: {e}");
                } else {
                    eprintln!("[ax-serving] drain-complete sent to orchestrator");
                }
            }

            Ok::<(), anyhow::Error>(())
        })
}

fn parse_rest_addr(addr: &str) -> Result<(String, u16)> {
    let addr = addr.trim();
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
    use super::parse_rest_addr;

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
}
