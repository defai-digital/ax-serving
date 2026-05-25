//! ax-serving: inference CLI and worker entrypoint for ax-serving.
//!
//! Supports:
//!   ax-serving -m model.gguf -p "prompt" -n 100     # single inference
//!   ax-serving serve -m model.gguf --port 18080       # start HTTP + gRPC worker
//!
//! To start the multi-worker API gateway, use `ax-serving-api` instead.

mod doctor;
mod logging;
mod output;
mod serve;
mod support;
mod thor;
mod tune;

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-serving-cli only supports aarch64-apple-darwin (Apple Silicon M3+)");

use std::path::PathBuf;

use anyhow::Result;
use ax_serving_engine::{BackendType, GenerateInput, LoadConfig, RouterBackend};
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
    ///   AXS_WORKER_MAX_INFLIGHT — max concurrent requests this worker advertises
    ///                             to the orchestrator (default: clamped to scheduler limits)
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
    /// Detect hardware and emit recommended serving configuration.
    Tune {
        /// Output file path (default: serving.toml).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Print recommendation without writing a file.
        #[arg(long)]
        dry_run: bool,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
    /// Validate serving configuration and environment.
    Doctor {
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
    /// Validate and inspect local AX Serving configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Query a running worker or API gateway.
    Status {
        /// Base URL of the worker or gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Include authenticated gateway diagnostics and recovery actions.
        #[arg(long)]
        diagnostics: bool,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
    /// Collect a redacted diagnostics bundle for support escalation.
    SupportBundle {
        /// Base URL of the API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Optional output path for the redacted JSON bundle.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Emit the redacted JSON bundle to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Validate AX Fabric-facing serving contracts.
    Fabric {
        #[command(subcommand)]
        command: FabricCommand,
    },
    /// Inspect migration readiness for compatibility paths.
    Migration {
        #[command(subcommand)]
        command: MigrationCommand,
    },
    /// Operate registered gateway workers.
    Workers {
        #[command(subcommand)]
        command: WorkersCommand,
    },
    /// Run a minimal OpenAI-compatible chat completion request.
    SmokeTest {
        /// Base URL of the worker or gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Model ID to use for the request.
        #[arg(long, default_value = "default")]
        model: String,
        /// Prompt sent to the model.
        #[arg(long, default_value = "Reply with ok.")]
        prompt: String,
        /// Max tokens requested from the model.
        #[arg(long, default_value_t = 8)]
        max_tokens: u32,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCommand {
    /// Validate the resolved serving configuration.
    Validate {
        /// Config file to validate. Defaults to the normal AXS_CONFIG/search path.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum FabricCommand {
    /// Validate the stable AX Fabric runtime endpoints.
    Validate {
        /// Base URL of the worker or API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum MigrationCommand {
    /// Check whether embedded compatibility workers can be denied.
    EmbeddedReadiness {
        /// Base URL of the API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum WorkersCommand {
    /// List registered workers.
    List {
        /// Base URL of the API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
    /// Inspect one registered worker.
    Get {
        /// Worker ID.
        id: String,
        /// Base URL of the API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
    /// Mark a worker draining.
    Drain {
        /// Worker ID.
        id: String,
        /// Base URL of the API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Wait until inflight reaches zero, then call drain-complete.
        #[arg(long)]
        complete_when_idle: bool,
        /// Maximum time to wait for idle when --complete-when-idle is set.
        #[arg(long, default_value_t = 30)]
        idle_timeout_secs: u64,
        /// Poll interval for --complete-when-idle.
        #[arg(long, default_value_t = 1000)]
        poll_interval_ms: u64,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
    /// Complete drain and remove a worker from the gateway registry.
    DrainComplete {
        /// Worker ID.
        id: String,
        /// Base URL of the API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
    },
    /// Force-remove a worker from the gateway registry.
    Remove {
        /// Worker ID.
        id: String,
        /// Base URL of the API gateway.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Bearer token. Defaults to the first AXS_API_KEY token when set.
        #[arg(long)]
        api_key: Option<String>,
        /// Emit machine-readable JSON for automation.
        #[arg(long)]
        json: bool,
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
        /// Thor inference runtime (`vllm` by default; `sglang` kept for compatibility).
        #[arg(long, default_value = "vllm")]
        runtime: String,
        /// Local runtime base URL.
        #[arg(
            long,
            visible_alias = "sglang-url",
            default_value = "http://127.0.0.1:8000"
        )]
        runtime_url: String,
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
        /// Thor inference runtime (`vllm` by default; `sglang` kept for compatibility).
        #[arg(long)]
        runtime: Option<String>,
        /// Local runtime base URL.
        #[arg(long, visible_alias = "sglang-url")]
        runtime_url: Option<String>,
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
        /// Exit with a Thor-specific readiness code when the node is not ready.
        #[arg(long)]
        require_ready: bool,
    },
    /// Poll Thor status until the node is ready or timeout expires.
    WaitReady {
        /// Env-file path. Defaults to ~/.config/ax-serving/thor.env
        #[arg(long)]
        config: Option<PathBuf>,
        /// Overall timeout before returning a readiness-mismatch exit code.
        #[arg(long, default_value_t = 60)]
        timeout_secs: u64,
        /// Poll interval between status checks.
        #[arg(long, default_value_t = 1000)]
        poll_interval_ms: u64,
    },
    /// Mark a registered Thor worker draining and optionally complete drain when idle.
    Drain {
        /// Env-file path. Defaults to ~/.config/ax-serving/thor.env
        #[arg(long)]
        config: Option<PathBuf>,
        /// After requesting drain, wait for the local agent to become idle and
        /// then send drain-complete.
        #[arg(long)]
        complete_when_idle: bool,
        /// Maximum time to wait for idle before failing drain completion.
        #[arg(long, default_value_t = 30)]
        idle_timeout_secs: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Serve {
            model,
            model_id,
            port,
            host,
            config,
            routing_config,
            orchestrator,
        }) => {
            logging::init_logging(cli.verbose);
            serve::run_serve(
                model,
                model_id,
                host,
                port,
                config,
                routing_config,
                orchestrator,
            )
        }
        Some(Command::Tune {
            output,
            dry_run,
            json,
        }) => tune::run_tune(output, dry_run, json),
        Some(Command::Doctor { json }) => doctor::run_doctor(json),
        Some(Command::Config { command }) => match command {
            ConfigCommand::Validate { config, json } => support::run_config_validate(config, json),
        },
        Some(Command::Status {
            url,
            api_key,
            diagnostics,
            json,
        }) => support::run_status(url, api_key, diagnostics, json),
        Some(Command::SupportBundle {
            url,
            api_key,
            output,
            json,
        }) => support::run_support_bundle(url, api_key, output, json),
        Some(Command::Fabric { command }) => match command {
            FabricCommand::Validate { url, api_key, json } => {
                support::run_fabric_validate(url, api_key, json)
            }
        },
        Some(Command::Migration { command }) => match command {
            MigrationCommand::EmbeddedReadiness { url, api_key, json } => {
                support::run_migration_embedded_readiness(url, api_key, json)
            }
        },
        Some(Command::Workers { command }) => match command {
            WorkersCommand::List { url, api_key, json } => {
                support::run_workers_list(url, api_key, json)
            }
            WorkersCommand::Get {
                id,
                url,
                api_key,
                json,
            } => support::run_worker_get(url, id, api_key, json),
            WorkersCommand::Drain {
                id,
                url,
                api_key,
                complete_when_idle,
                idle_timeout_secs,
                poll_interval_ms,
                json,
            } => support::run_worker_drain(
                url,
                id,
                api_key,
                complete_when_idle,
                idle_timeout_secs,
                poll_interval_ms,
                json,
            ),
            WorkersCommand::DrainComplete {
                id,
                url,
                api_key,
                json,
            } => support::run_worker_drain_complete(url, id, api_key, json),
            WorkersCommand::Remove {
                id,
                url,
                api_key,
                json,
            } => support::run_worker_remove(url, id, api_key, json),
        },
        Some(Command::SmokeTest {
            url,
            model,
            prompt,
            max_tokens,
            api_key,
            json,
        }) => support::run_smoke_test(url, model, prompt, max_tokens, api_key, json),
        Some(Command::Thor { command }) => {
            if let ThorCommand::WaitReady {
                config,
                timeout_secs,
                ..
            } = &command
            {
                thor::wait_ready_fast_path(config.clone(), *timeout_secs)?;
            }
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async move {
                match command {
                    ThorCommand::Install {
                        control_plane,
                        listen_addr,
                        advertised_addr,
                        runtime,
                        runtime_url,
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
                        runtime,
                        runtime_url,
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
                        runtime,
                        runtime_url,
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
                            runtime,
                            runtime_url,
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
                    ThorCommand::Status {
                        config,
                        require_ready,
                    } => {
                        thor::status(thor::StatusArgs {
                            config,
                            require_ready,
                        })
                        .await
                    }
                    ThorCommand::WaitReady {
                        config,
                        timeout_secs,
                        poll_interval_ms,
                    } => {
                        thor::wait_ready(thor::WaitReadyArgs {
                            config,
                            timeout_secs,
                            poll_interval_ms,
                        })
                        .await
                    }
                    ThorCommand::Drain {
                        config,
                        complete_when_idle,
                        idle_timeout_secs,
                    } => {
                        thor::drain(thor::DrainArgs {
                            config,
                            complete_when_idle,
                            idle_timeout_secs,
                        })
                        .await
                    }
                }
            })
        }
        None => {
            logging::init_logging(cli.verbose);
            // Inference mode: the selected backend owns its own execution path.
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
    use ax_serving_engine::{GenerateEvent, GenerationParams, InferenceBackend as _};

    serve::ensure_embedded_runtime_allowed("single-shot inference mode")?;

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
        // Favor llama.cpp by default for single-shot inference.
        backend_hint: Some("llama_cpp".to_string()),
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
        seed: if cli.seed == 0 { None } else { Some(cli.seed) },
        repeat_penalty: Some(cli.repeat_penalty as f64),
        ..Default::default()
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<GenerateEvent>(512);

    // generate() spawns on the backend's internal runtime and returns immediately.
    backend.generate(handle, build_inference_input(prompt, cli.chat), params, tx)?;

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
            if final_stats.is_none() {
                return Err(anyhow::anyhow!(
                    "generation channel closed without Done or Error event"
                ));
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

pub(crate) fn normalize_http_base_url(raw: &str, field: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("{field} URL is empty");
    }
    let trimmed = trimmed.trim_end_matches('/');
    if trimmed.is_empty() {
        anyhow::bail!("{field} URL is empty after trimming trailing slashes");
    }

    let mut rest = trimmed;
    let has_scheme = if let Some(scheme_end) = trimmed.find("://") {
        let scheme = &trimmed[..scheme_end];
        if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
            rest = &trimmed[scheme_end + 3..];
            true
        } else {
            anyhow::bail!("{field} has unsupported URL scheme: {trimmed}");
        }
    } else {
        false
    };

    if rest.is_empty() {
        anyhow::bail!("{field} URL is incomplete: {trimmed}");
    }
    if rest.contains('/') {
        anyhow::bail!("{field} URL must not include a path: {trimmed}");
    }
    if rest.contains('?') || rest.contains('#') {
        anyhow::bail!("{field} URL must not include query params or fragments: {trimmed}");
    }

    let normalized = if has_scheme {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };

    Ok(normalized.trim_end_matches('/').to_string())
}

fn build_inference_input(prompt: String, use_chat_template: bool) -> GenerateInput {
    if use_chat_template {
        GenerateInput::Chat(vec![ax_serving_engine::ChatMessage {
            role: "user".into(),
            content: serde_json::Value::String(prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
    } else {
        GenerateInput::Text(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::{build_inference_input, normalize_http_base_url};
    use anyhow::Result;
    use ax_serving_engine::{ChatMessage, GenerateInput};
    use serde_json::json;

    #[test]
    fn inference_input_without_chat_uses_text_prompt() {
        let input = build_inference_input("hello world".to_string(), false);
        assert!(matches!(input, GenerateInput::Text(_)));
    }

    #[test]
    fn inference_input_with_chat_uses_chat_payload() {
        let input = build_inference_input("hello world".to_string(), true);
        match input {
            GenerateInput::Chat(messages) => {
                let expected = ChatMessage {
                    role: "user".into(),
                    content: json!("hello world"),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                };
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].role, expected.role);
                assert_eq!(messages[0].content, expected.content);
            }
            _ => panic!("expected chat input"),
        }
    }

    #[test]
    fn normalize_http_base_url_adds_http_scheme_if_missing() -> Result<()> {
        let normalized = normalize_http_base_url("127.0.0.1:19090", "orchestrator")?;
        assert_eq!(normalized, "http://127.0.0.1:19090");
        Ok(())
    }

    #[test]
    fn normalize_http_base_url_keeps_http_and_trims_slash() -> Result<()> {
        let normalized = normalize_http_base_url("http://127.0.0.1:19090//", "orchestrator")?;
        assert_eq!(normalized, "http://127.0.0.1:19090");
        Ok(())
    }

    #[test]
    fn normalize_http_base_url_rejects_unsupported_scheme() {
        let err = normalize_http_base_url("ftp://127.0.0.1:19090", "orchestrator")
            .expect_err("unsupported scheme should be rejected");
        assert!(err.to_string().contains("unsupported URL scheme"));
    }

    #[test]
    fn normalize_http_base_url_accepts_uppercase_scheme() -> Result<()> {
        let normalized = normalize_http_base_url("HTTP://127.0.0.1:19090//", "orchestrator")?;
        assert_eq!(normalized, "HTTP://127.0.0.1:19090");
        Ok(())
    }

    #[test]
    fn normalize_http_base_url_rejects_path_suffix() {
        let err = normalize_http_base_url("http://127.0.0.1:19090/api", "orchestrator")
            .expect_err("path suffix should be rejected");
        assert!(err.to_string().contains("URL must not include a path"));
    }

    #[test]
    fn normalize_http_base_url_rejects_trailing_space_only() {
        let err = normalize_http_base_url("   ", "orchestrator")
            .expect_err("blank url should be rejected");
        assert!(err.to_string().contains("URL is empty"));
    }
}
