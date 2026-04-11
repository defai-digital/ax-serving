//! MlxBackend: `InferenceBackend` implementation wrapping an `mlx_lm.server`
//! subprocess.
//!
//! # Model format
//!
//! MLX models are **directories** containing `config.json` and one or more
//! `*.safetensors` weight files.  They are distinct from GGUF files — use
//! `is_mlx_model(path)` to detect them before routing here.
//!
//! # Lifecycle
//!
//! Each `load_model` call spawns a dedicated `mlx_lm.server` process on a
//! dynamically allocated port.  The process is killed when `unload_model` is
//! called or when the model entry is dropped.
//!
//! # Generation
//!
//! `generate` calls `mlx_lm.server`'s OpenAI-compatible `/v1/chat/completions`
//! (Chat input) or `/v1/completions` (Text/Tokens input) endpoint via SSE.
//!
//! # Limitations
//!
//! - No `/v1/embeddings` support (mlx-lm server does not expose this endpoint).
//! - No `/tokenize` or `/detokenize` endpoints (use llama.cpp for those needs).
//!
//! # Requirements
//!
//! `mlx_lm.server` must be on `$PATH` or configured via `AXS_MLX_BIN`.
//! Install with: `pip install mlx-lm`

use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU8, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{info, warn};

use crate::{
    CacheTelemetry, EmbedConfig, EmbedInput, EmbedResult, GenerateEvent, GenerateInput,
    GenerationParams, GenerationStats, InferenceBackend, LoadConfig, ModelHandle, ModelMetadata,
    ThermalMonitor, ThermalState,
};

const MLX_LOCAL_HOST: &str = "127.0.0.1";

// ── MlxConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the mlx-lm subprocess backend.
///
/// Loaded from `config/serving.yaml` (under the `mlx:` key) and further
/// overridden by `AXS_MLX_*` environment variables.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MlxConfig {
    /// Path to the `mlx_lm.server` binary (env: `AXS_MLX_BIN`).
    /// Default: `mlx_lm.server` (looked up on `$PATH`).
    pub bin: String,
    /// SSE token batch size before each `blocking_send` (env: `AXS_MLX_TOKEN_BATCH`).
    pub token_batch_size: usize,
    /// Maximum allowed token batch size (clamp ceiling).
    pub token_batch_max: usize,
    /// HTTP request timeout for calls to mlx_lm.server (secs).
    pub http_request_timeout_secs: u64,
    /// Timeout waiting for mlx_lm.server to become ready after spawn (secs).
    pub server_startup_timeout_secs: u64,
    /// Sleep between health poll attempts in `wait_ready()` (ms).
    pub wait_ready_poll_interval_ms: u64,
    /// Per-request timeout for each poll attempt in `wait_ready()` (secs).
    pub wait_ready_check_timeout_secs: u64,
    /// How often the background health poller runs (secs).
    pub health_poller_interval_secs: u64,
    /// Per-request timeout for health poll checks in the background poller (secs).
    pub health_poller_check_timeout_secs: u64,
    /// Consecutive health poll failures before a restart is attempted.
    pub health_poller_restart_threshold: u32,
    /// Maximum restart attempts before marking the model Dead.
    pub health_poller_max_restarts: u32,
    /// Max concurrent decoding streams (`--decode-concurrency`).
    /// `None` = let mlx_lm.server decide (default: 32).
    pub decode_concurrency: Option<u32>,
    /// Consecutive generate failures before circuit breaker trips (env: `AXS_CB_TRIP_THRESHOLD`).
    pub circuit_breaker_trip_threshold: u32,
    /// Recovery window (ms) — how long the breaker stays Open before HalfOpen
    /// (env: `AXS_CB_RECOVERY_MS`).
    pub circuit_breaker_recovery_ms: u64,
}

const DEFAULT_MLX_BIN: &str = "mlx_lm.server";
const DEFAULT_MLX_TOKEN_BATCH: usize = 4;
const DEFAULT_MLX_TOKEN_BATCH_MAX: usize = 32;
const DEFAULT_MLX_HTTP_TIMEOUT_SECS: u64 = 300;
const DEFAULT_MLX_STARTUP_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MLX_WAIT_POLL_INTERVAL_MS: u64 = 500;
const DEFAULT_MLX_WAIT_CHECK_TIMEOUT_SECS: u64 = 1;
const DEFAULT_MLX_HEALTH_INTERVAL_SECS: u64 = 5;
const DEFAULT_MLX_HEALTH_CHECK_TIMEOUT_SECS: u64 = 2;
const DEFAULT_MLX_HEALTH_RESTART_THRESHOLD: u32 = 3;
const DEFAULT_MLX_HEALTH_MAX_RESTARTS: u32 = 3;
const DEFAULT_CB_TRIP_THRESHOLD: u32 = 3;
const DEFAULT_CB_RECOVERY_MS: u64 = 10_000;

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            bin: DEFAULT_MLX_BIN.to_string(),
            token_batch_size: DEFAULT_MLX_TOKEN_BATCH,
            token_batch_max: DEFAULT_MLX_TOKEN_BATCH_MAX,
            http_request_timeout_secs: DEFAULT_MLX_HTTP_TIMEOUT_SECS,
            server_startup_timeout_secs: DEFAULT_MLX_STARTUP_TIMEOUT_SECS,
            wait_ready_poll_interval_ms: DEFAULT_MLX_WAIT_POLL_INTERVAL_MS,
            wait_ready_check_timeout_secs: DEFAULT_MLX_WAIT_CHECK_TIMEOUT_SECS,
            health_poller_interval_secs: DEFAULT_MLX_HEALTH_INTERVAL_SECS,
            health_poller_check_timeout_secs: DEFAULT_MLX_HEALTH_CHECK_TIMEOUT_SECS,
            health_poller_restart_threshold: DEFAULT_MLX_HEALTH_RESTART_THRESHOLD,
            health_poller_max_restarts: DEFAULT_MLX_HEALTH_MAX_RESTARTS,
            decode_concurrency: None,
            circuit_breaker_trip_threshold: DEFAULT_CB_TRIP_THRESHOLD,
            circuit_breaker_recovery_ms: DEFAULT_CB_RECOVERY_MS,
        }
    }
}

impl MlxConfig {
    /// Apply `AXS_MLX_*` env var overrides.
    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("AXS_MLX_BIN") {
            self.bin = v;
        }
        if let Ok(v) = std::env::var("AXS_MLX_TOKEN_BATCH")
            && let Ok(n) = v.parse::<usize>()
        {
            self.token_batch_size = n.clamp(1, self.token_batch_max);
        }
        if let Ok(v) = std::env::var("AXS_MLX_DECODE_CONCURRENCY")
            && let Ok(n) = v.parse::<u32>()
        {
            self.decode_concurrency = Some(n.max(1));
        }
        // Circuit breaker — shared env vars with LlamaCppBackend.
        if let Ok(v) = std::env::var("AXS_CB_TRIP_THRESHOLD")
            && let Ok(n) = v.parse::<u32>()
        {
            self.circuit_breaker_trip_threshold = n;
        }
        if let Ok(v) = std::env::var("AXS_CB_RECOVERY_MS")
            && let Ok(ms) = v.parse::<u64>()
        {
            self.circuit_breaker_recovery_ms = ms;
        }
    }

    /// Create from env vars only (no YAML), using struct defaults as base.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.apply_env_overrides();
        cfg
    }

    pub fn effective_batch_size(&self) -> usize {
        self.token_batch_size.clamp(1, self.token_batch_max)
    }
}

// ── Model detection ────────────────────────────────────────────────────────────

/// Returns `true` if `path` looks like an MLX model directory.
///
/// An MLX model is a **directory** containing:
/// - `config.json` (HuggingFace model config)
/// - at least one `*.safetensors` weight file
pub fn is_mlx_model(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    if !path.join("config.json").is_file() {
        return false;
    }
    std::fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
        })
        .unwrap_or(false)
}

/// Metadata extracted from an MLX model directory's `config.json`.
struct MlxModelConfig {
    model_type: String,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    hidden_size: u32,
    vocab_size: u32,
    max_position_embeddings: u32,
}

impl Default for MlxModelConfig {
    fn default() -> Self {
        Self {
            model_type: "mlx".to_string(),
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            hidden_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
        }
    }
}

/// Read model metadata from `config.json` inside an MLX model directory.
/// Returns defaults for any missing fields.
fn read_mlx_model_config(path: &Path) -> MlxModelConfig {
    let cfg_path = path.join("config.json");
    let text = match std::fs::read_to_string(&cfg_path) {
        Ok(t) => t,
        Err(_) => return MlxModelConfig::default(),
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return MlxModelConfig::default(),
    };
    MlxModelConfig {
        model_type: v["model_type"].as_str().unwrap_or("mlx").to_string(),
        num_hidden_layers: v["num_hidden_layers"].as_u64().unwrap_or(0) as u32,
        num_attention_heads: v["num_attention_heads"].as_u64().unwrap_or(0) as u32,
        num_key_value_heads: v["num_key_value_heads"]
            .as_u64()
            .or_else(|| v["num_attention_heads"].as_u64())
            .unwrap_or(0) as u32,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(0) as u32,
        vocab_size: v["vocab_size"].as_u64().unwrap_or(0) as u32,
        max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap_or(0) as u32,
    }
}

// ── Health state ──────────────────────────────────────────────────────────────

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HealthState {
    Healthy = 0,
    Unhealthy = 1,
    Dead = 2,
}

// ── Circuit breaker (mirrors llamacpp.rs) ────────────────────────────────────

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

fn unix_ms_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Per-model circuit breaker for the mlx_lm.server subprocess.
///
/// Trip threshold (`AXS_CB_TRIP_THRESHOLD`, default 3): consecutive generate
/// failures before the breaker opens.
///
/// Recovery window (`AXS_CB_RECOVERY_MS`, default 10 000 ms): how long the
/// breaker stays Open before transitioning to HalfOpen.
struct CircuitBreaker {
    state: AtomicU8,
    consecutive_generate_failures: std::sync::atomic::AtomicU32,
    last_opened_ms: AtomicU64,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            consecutive_generate_failures: std::sync::atomic::AtomicU32::new(0),
            last_opened_ms: AtomicU64::new(0),
        }
    }

    fn trip(&self) {
        self.last_opened_ms.store(unix_ms_now(), Ordering::Relaxed);
        self.state
            .store(CircuitState::Open as u8, Ordering::Release);
        self.consecutive_generate_failures
            .store(0, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::Release);
        self.consecutive_generate_failures
            .store(0, Ordering::Relaxed);
    }
}

// ── Handle allocation ─────────────────────────────────────────────────────────

static NEXT_MLX_HANDLE: AtomicU64 = AtomicU64::new(3_000_000);

fn next_mlx_handle() -> ModelHandle {
    ModelHandle(NEXT_MLX_HANDLE.fetch_add(1, Ordering::Relaxed))
}

fn find_free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(format!("{MLX_LOCAL_HOST}:0"))?;
    Ok(listener.local_addr()?.port())
}

// ── Blocking executor ─────────────────────────────────────────────────────────

type BlockingJob = Box<dyn FnOnce() + Send + 'static>;

struct BlockingExecutor {
    tx: std::sync::mpsc::Sender<BlockingJob>,
    _workers: Vec<std::thread::JoinHandle<()>>,
}

impl BlockingExecutor {
    fn new(worker_count: usize) -> Self {
        let n = worker_count.max(1);
        let (tx, rx) = std::sync::mpsc::channel::<BlockingJob>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(n);
        for i in 0..n {
            let rx = Arc::clone(&rx);
            match std::thread::Builder::new()
                .name(format!("ax-mlx-gen-{i}"))
                .spawn(move || {
                    loop {
                        let recv_result = {
                            let guard = match rx.lock() {
                                Ok(g) => g,
                                Err(e) => {
                                    warn!("mlx executor receiver lock poisoned; recovering");
                                    e.into_inner()
                                }
                            };
                            guard.recv()
                        };
                        match recv_result {
                            Ok(job) => job(),
                            Err(_) => break,
                        }
                    }
                }) {
                Ok(h) => workers.push(h),
                Err(e) => warn!(%e, thread_idx = i, "failed to spawn mlx executor worker"),
            }
        }
        Self {
            tx,
            _workers: workers,
        }
    }

    fn execute<F: FnOnce() + Send + 'static>(&self, f: F) -> Result<()> {
        self.tx
            .send(Box::new(f))
            .map_err(|_| anyhow::anyhow!("mlx blocking executor stopped"))
    }
}

// ── Per-model subprocess entry ────────────────────────────────────────────────

struct MlxProcess {
    port: u16,
    child: Arc<Mutex<Option<std::process::Child>>>,
    health: Arc<AtomicU8>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    poller: Option<std::thread::JoinHandle<()>>,
    breaker: Arc<CircuitBreaker>,
    /// Model type string from config.json; kept for diagnostics and future use.
    _model_type: String,
}

impl Drop for MlxProcess {
    fn drop(&mut self) {
        // Signal the poller to stop before killing the child so it doesn't
        // attempt a restart.
        self.stop.store(true, Ordering::SeqCst);
        let mut guard = match self.child.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("mlx child lock poisoned during drop; recovering");
                e.into_inner()
            }
        };
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // Join the poller thread to avoid detached threads outliving process
        // state (mirrors LlamaCppProcess::Drop).
        if let Some(handle) = self.poller.take() {
            let _ = handle.join();
        }
    }
}

// ── Backend ───────────────────────────────────────────────────────────────────

/// `InferenceBackend` implementation backed by `mlx_lm.server`.
pub struct MlxBackend {
    models: Arc<Mutex<HashMap<ModelHandle, MlxProcess>>>,
    thermal: ThermalMonitor,
    http: reqwest::blocking::Client,
    config: Arc<MlxConfig>,
    executor: Arc<BlockingExecutor>,
}

impl MlxBackend {
    pub fn new(config: MlxConfig) -> Self {
        let http = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.http_request_timeout_secs))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                warn!(%e, "failed to build mlx http client; using default");
                reqwest::blocking::Client::new()
            }
        };
        let executor_threads = std::env::var("AXS_MLX_EXECUTOR_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(4)
            });
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
            thermal: ThermalMonitor::new(),
            http,
            config: Arc::new(config),
            executor: Arc::new(BlockingExecutor::new(executor_threads)),
        }
    }

    fn models_lock(&self) -> MutexGuard<'_, HashMap<ModelHandle, MlxProcess>> {
        match self.models.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("mlx model registry lock poisoned; recovering");
                e.into_inner()
            }
        }
    }

    fn spawn_server(
        bin: &str,
        path: &Path,
        port: u16,
        config: &MlxConfig,
        load_config: &LoadConfig,
    ) -> Result<std::process::Child> {
        let mut cmd = std::process::Command::new(bin);
        cmd.arg("--model")
            .arg(path)
            .arg("--host")
            .arg(MLX_LOCAL_HOST)
            .arg("--port")
            .arg(port.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        if load_config.context_length > 0 {
            // mlx-lm does not have a --ctx-size flag; context_length is ignored here.
            // Kept in the signature for forward compatibility if mlx-lm adds it.
        }
        if let Some(dc) = config.decode_concurrency {
            cmd.arg("--decode-concurrency").arg(dc.to_string());
        }
        cmd.spawn()
            .with_context(|| format!("failed to spawn '{bin}'; is mlx-lm installed and on PATH?"))
    }

    fn wait_ready(
        http: &reqwest::blocking::Client,
        port: u16,
        startup_timeout: Duration,
        poll_interval: Duration,
        check_timeout: Duration,
    ) -> Result<()> {
        let url = format!("http://{MLX_LOCAL_HOST}:{port}/health");
        let deadline = Instant::now() + startup_timeout;
        loop {
            match http.get(&url).timeout(check_timeout).send() {
                Ok(r) if r.status().is_success() => return Ok(()),
                _ => {}
            }
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "mlx_lm.server on port {port} did not become ready within {startup_timeout:?}"
                );
            }
            std::thread::sleep(poll_interval);
        }
    }
}

impl Default for MlxBackend {
    fn default() -> Self {
        Self::new(MlxConfig::from_env())
    }
}

// ── Health poller ─────────────────────────────────────────────────────────────

struct MlxPollerArgs {
    port: u16,
    path: PathBuf,
    bin: String,
    load_config: LoadConfig,
    mlx_config: Arc<MlxConfig>,
    child: Arc<Mutex<Option<std::process::Child>>>,
    http: reqwest::blocking::Client,
    health: Arc<AtomicU8>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    breaker: Arc<CircuitBreaker>,
    poller_interval: Duration,
    check_timeout: Duration,
    restart_threshold: u32,
    max_restarts: u32,
    restart_wait_timeout: Duration,
    wait_ready_poll_interval: Duration,
    wait_ready_check_timeout: Duration,
}

fn run_mlx_health_poller(args: MlxPollerArgs) {
    let MlxPollerArgs {
        port,
        path,
        bin,
        load_config,
        mlx_config,
        child,
        http,
        health,
        stop,
        breaker,
        poller_interval,
        check_timeout,
        restart_threshold,
        max_restarts,
        restart_wait_timeout,
        wait_ready_poll_interval,
        wait_ready_check_timeout,
    } = args;

    let url = format!("http://{MLX_LOCAL_HOST}:{port}/health");
    let mut consecutive_failures: u32 = 0;
    let mut restart_attempts: u32 = 0;

    loop {
        std::thread::sleep(poller_interval);
        if stop.load(Ordering::Acquire) {
            return;
        }

        let ok = http
            .get(&url)
            .timeout(check_timeout)
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        if ok {
            consecutive_failures = 0;
            health.store(HealthState::Healthy as u8, Ordering::Relaxed);
            continue;
        }

        consecutive_failures += 1;
        health.store(HealthState::Unhealthy as u8, Ordering::Relaxed);

        if consecutive_failures < restart_threshold {
            continue;
        }

        if restart_attempts >= max_restarts {
            warn!(
                port,
                "mlx_lm.server health poller: max restarts ({max_restarts}) exhausted; marking model Dead"
            );
            health.store(HealthState::Dead as u8, Ordering::Relaxed);
            return;
        }

        restart_attempts += 1;
        consecutive_failures = 0;
        warn!(
            port,
            attempt = restart_attempts,
            "mlx_lm.server unhealthy; attempting restart"
        );

        // Kill old process.
        {
            let mut guard = match child.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            if let Some(mut c) = guard.take() {
                let _ = c.kill();
                let _ = c.wait();
            }
        }

        if stop.load(Ordering::Acquire) {
            return;
        }

        // Respawn.
        match MlxBackend::spawn_server(&bin, &path, port, &mlx_config, &load_config) {
            Ok(new_child) => {
                *match child.lock() {
                    Ok(g) => g,
                    Err(e) => e.into_inner(),
                } = Some(new_child);
            }
            Err(e) => {
                warn!(%e, port, "mlx_lm.server restart: spawn failed");
                health.store(HealthState::Dead as u8, Ordering::Relaxed);
                return;
            }
        }

        // Wait for new process to be ready.
        match MlxBackend::wait_ready(
            &http,
            port,
            restart_wait_timeout,
            wait_ready_poll_interval,
            wait_ready_check_timeout,
        ) {
            Ok(()) => {
                info!(
                    port,
                    attempt = restart_attempts,
                    "mlx_lm.server restarted and ready"
                );
                health.store(HealthState::Healthy as u8, Ordering::Relaxed);
                breaker.reset();
            }
            Err(e) => {
                warn!(%e, port, "mlx_lm.server restart: not ready within timeout");
                health.store(HealthState::Unhealthy as u8, Ordering::Relaxed);
            }
        }
    }
}

// ── InferenceBackend impl ─────────────────────────────────────────────────────

impl InferenceBackend for MlxBackend {
    fn load_model(&self, path: &Path, config: LoadConfig) -> Result<(ModelHandle, ModelMetadata)> {
        anyhow::ensure!(path.exists(), "model path not found: {}", path.display());
        anyhow::ensure!(
            is_mlx_model(path),
            "path is not an MLX model directory (must contain config.json and *.safetensors): {}",
            path.display()
        );

        let model_cfg = read_mlx_model_config(path);

        // Mitigate TOCTOU port race by retrying up to 3 times with a freshly
        // allocated port each attempt (mirrors LlamaCppBackend BUG-049 fix).
        const PORT_RETRY_LIMIT: usize = 3;
        let start = Instant::now();
        let mut last_err = None;
        let (port, child) = 'retry: {
            for attempt in 0..PORT_RETRY_LIMIT {
                let port =
                    find_free_port().context("failed to find free TCP port for mlx_lm.server")?;
                info!(
                    "spawning mlx_lm.server for {} on port {} (model_type={}, attempt {})",
                    path.display(),
                    port,
                    model_cfg.model_type,
                    attempt + 1
                );
                match Self::spawn_server(&self.config.bin, path, port, &self.config, &config) {
                    Ok(child) => {
                        match Self::wait_ready(
                            &self.http,
                            port,
                            Duration::from_secs(self.config.server_startup_timeout_secs),
                            Duration::from_millis(self.config.wait_ready_poll_interval_ms),
                            Duration::from_secs(self.config.wait_ready_check_timeout_secs),
                        ) {
                            Ok(()) => break 'retry (port, child),
                            Err(err) => {
                                warn!(%err, port, "mlx_lm.server not ready; retrying with new port");
                                last_err = Some(err);
                            }
                        }
                    }
                    Err(err) => {
                        warn!(%err, "mlx_lm.server spawn failed on port {port}; retrying with new port");
                        last_err = Some(err);
                    }
                }
            }
            return Err(last_err
                .unwrap()
                .context(format!("spawning mlx_lm.server for {}", path.display())));
        };

        let load_ms = start.elapsed().as_millis() as u64;
        info!(
            "mlx_lm.server ready on port {port} in {load_ms}ms (model_type={})",
            model_cfg.model_type
        );

        let meta = ModelMetadata {
            architecture: model_cfg.model_type.clone(),
            n_layers: model_cfg.num_hidden_layers,
            n_heads: model_cfg.num_attention_heads,
            n_kv_heads: model_cfg.num_key_value_heads,
            embedding_dim: model_cfg.hidden_size,
            vocab_size: model_cfg.vocab_size,
            context_length: if config.context_length > 0 {
                config.context_length
            } else if model_cfg.max_position_embeddings > 0 {
                model_cfg.max_position_embeddings
            } else {
                4096
            },
            load_time_ms: load_ms,
            peak_rss_bytes: 0,
            resolved_backend: crate::BackendType::Metal,
        };

        let handle = next_mlx_handle();
        let child_arc = Arc::new(Mutex::new(Some(child)));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let health = Arc::new(AtomicU8::new(HealthState::Healthy as u8));
        let breaker = Arc::new(CircuitBreaker::new());

        let poller = {
            let args = MlxPollerArgs {
                port,
                path: path.to_path_buf(),
                bin: self.config.bin.clone(),
                load_config: config,
                mlx_config: Arc::clone(&self.config),
                child: Arc::clone(&child_arc),
                http: self.http.clone(),
                health: Arc::clone(&health),
                stop: Arc::clone(&stop),
                breaker: Arc::clone(&breaker),
                poller_interval: Duration::from_secs(self.config.health_poller_interval_secs),
                check_timeout: Duration::from_secs(self.config.health_poller_check_timeout_secs),
                restart_threshold: self.config.health_poller_restart_threshold,
                max_restarts: self.config.health_poller_max_restarts,
                restart_wait_timeout: Duration::from_secs(self.config.server_startup_timeout_secs),
                wait_ready_poll_interval: Duration::from_millis(
                    self.config.wait_ready_poll_interval_ms,
                ),
                wait_ready_check_timeout: Duration::from_secs(
                    self.config.wait_ready_check_timeout_secs,
                ),
            };
            match std::thread::Builder::new()
                .name(format!("ax-mlx-health-{port}"))
                .spawn(move || run_mlx_health_poller(args))
            {
                Ok(h) => h,
                Err(e) => {
                    warn!(%e, port, "failed to spawn mlx health poller; cleaning up");
                    let mut guard = match child_arc.lock() {
                        Ok(g) => g,
                        Err(e2) => e2.into_inner(),
                    };
                    if let Some(mut c) = guard.take() {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                    return Err(anyhow::anyhow!(
                        "failed to spawn mlx health poller thread: {e}"
                    ));
                }
            }
        };

        self.models_lock().insert(
            handle,
            MlxProcess {
                port,
                child: child_arc,
                health,
                stop,
                poller: Some(poller),
                breaker,
                _model_type: model_cfg.model_type,
            },
        );
        Ok((handle, meta))
    }

    fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let entry = self.models_lock().remove(&handle);
        anyhow::ensure!(
            entry.is_some(),
            "no MLX model loaded with handle {:?}",
            handle
        );
        info!("unloaded MLX model {:?}", handle);
        Ok(())
    }

    fn generate(
        &self,
        handle: ModelHandle,
        input: GenerateInput,
        params: GenerationParams,
        tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> Result<()> {
        let (port, breaker) = {
            let guard = self.models_lock();
            let proc = guard
                .get(&handle)
                .ok_or_else(|| anyhow::anyhow!("invalid MLX model handle {:?}", handle))?;

            // Fail fast if the server is permanently dead (bypasses circuit breaker).
            if proc.health.load(Ordering::Relaxed) == HealthState::Dead as u8 {
                anyhow::bail!(
                    "mlx_lm.server for handle {:?} has permanently failed; unload and reload",
                    handle
                );
            }

            // Circuit breaker check.
            let cb_state = proc.breaker.state.load(Ordering::Acquire);
            if cb_state == CircuitState::Open as u8 {
                let opened_ms = proc.breaker.last_opened_ms.load(Ordering::Relaxed);
                let elapsed_ms = unix_ms_now().saturating_sub(opened_ms);
                if elapsed_ms < self.config.circuit_breaker_recovery_ms {
                    anyhow::bail!(
                        "circuit open: mlx_lm.server for handle {:?} is recovering; retry later",
                        handle
                    );
                }
                // Recovery window elapsed → transition Open → HalfOpen for one probe.
                let _ = proc.breaker.state.compare_exchange(
                    CircuitState::Open as u8,
                    CircuitState::HalfOpen as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }

            (proc.port, Arc::clone(&proc.breaker))
        };

        let http = self.http.clone();
        let batch_size = self.config.effective_batch_size();
        let trip_threshold = self.config.circuit_breaker_trip_threshold;

        self.executor.execute(move || {
            let emit_logprobs = params.logprobs.unwrap_or(false);
            let result = match (&input, params.stream) {
                (GenerateInput::Chat(msgs), true) => {
                    let body = build_mlx_chat_body(msgs, &params);
                    mlx_stream_chat(&http, port, &body, &tx, batch_size, emit_logprobs)
                }
                (GenerateInput::Chat(msgs), false) => {
                    let body = build_mlx_chat_body(msgs, &params);
                    mlx_complete_chat(&http, port, &body, &tx, emit_logprobs)
                }
                (_, true) => {
                    let body = build_mlx_completions_body(&input, &params);
                    mlx_stream_completions(&http, port, &body, &tx, batch_size, emit_logprobs)
                }
                (_, false) => {
                    let body = build_mlx_completions_body(&input, &params);
                    mlx_complete_completions(&http, port, &body, &tx, emit_logprobs)
                }
            };

            match result {
                Ok(()) => {
                    breaker
                        .consecutive_generate_failures
                        .store(0, Ordering::Relaxed);
                    // If HalfOpen, close the circuit (probe succeeded).
                    breaker
                        .state
                        .compare_exchange(
                            CircuitState::HalfOpen as u8,
                            CircuitState::Closed as u8,
                            Ordering::SeqCst,
                            Ordering::Relaxed,
                        )
                        .ok();
                }
                Err(e) => {
                    let was_half_open =
                        breaker.state.load(Ordering::Relaxed) == CircuitState::HalfOpen as u8;
                    let failures = breaker
                        .consecutive_generate_failures
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    if was_half_open || failures >= trip_threshold {
                        breaker.trip();
                    }
                    warn!("mlx stream error: {e}");
                    let _ = tx.blocking_send(GenerateEvent::Error(e.to_string()));
                }
            }
        })?;
        Ok(())
    }

    fn tokenize(&self, _handle: ModelHandle, _text: &str, _add_bos: bool) -> Result<Vec<u32>> {
        anyhow::bail!(
            "mlx-lm backend does not expose a tokenize endpoint; \
             load the model with llama.cpp for tokenization support"
        )
    }

    fn decode_tokens(&self, _handle: ModelHandle, _tokens: &[u32]) -> Result<String> {
        anyhow::bail!(
            "mlx-lm backend does not expose a detokenize endpoint; \
             load the model with llama.cpp for detokenization support"
        )
    }

    fn eos_tokens(&self, _handle: ModelHandle) -> Result<Vec<u32>> {
        // mlx-lm server does not expose token IDs; return common defaults.
        Ok(vec![2])
    }

    fn bos_token(&self, _handle: ModelHandle) -> Result<u32> {
        // mlx-lm server does not expose token IDs; return common default.
        // Cannot fall back to the trait default (tokenize) because MlxBackend
        // does not support tokenize either.
        Ok(1)
    }

    fn eval_tokens(&self, _handle: ModelHandle, _tokens: &[u32]) -> Result<u32> {
        anyhow::bail!("eval_tokens not supported by MlxBackend")
    }

    fn embed(
        &self,
        _handle: ModelHandle,
        _inputs: &EmbedInput<'_>,
        _config: &EmbedConfig,
    ) -> Result<EmbedResult> {
        anyhow::bail!(
            "mlx-lm backend does not support embeddings; \
             load the model with llama.cpp --embedding for embedding support"
        )
    }

    fn thermal_state(&self) -> ThermalState {
        self.thermal.current()
    }

    fn recommended_concurrency(&self) -> usize {
        self.config
            .decode_concurrency
            .map(|n| n as usize)
            .unwrap_or(8)
    }

    fn cache_telemetry(&self) -> CacheTelemetry {
        CacheTelemetry::default()
    }

    fn backend_name_for_handle(&self, handle: ModelHandle) -> Option<&'static str> {
        self.models_lock().get(&handle).map(|_| "mlx")
    }
}

// ── Request body builders ─────────────────────────────────────────────────────

fn apply_mlx_generation_params(body: &mut serde_json::Value, params: &GenerationParams) {
    if let Some(t) = params.temperature {
        body["temperature"] = t.into();
    }
    if let Some(p) = params.top_p {
        body["top_p"] = p.into();
    }
    if let Some(k) = params.top_k {
        body["top_k"] = (k as i64).into();
    }
    if let Some(n) = params.max_tokens {
        body["max_tokens"] = (n as i64).into();
    }
    if !params.stop_seqs.is_empty() {
        let arr: Vec<serde_json::Value> = params
            .stop_seqs
            .iter()
            .map(|s| serde_json::Value::String(s.clone()))
            .collect();
        body["stop"] = serde_json::Value::Array(arr);
    }
    if let Some(seed) = params.seed {
        body["seed"] = seed.into();
    }
    // Sampling params.
    if let Some(p) = params.min_p {
        body["min_p"] = p.into();
    }
    if let Some(r) = params.repeat_penalty {
        // mlx_lm uses "repetition_penalty" (not "repeat_penalty" like llama.cpp).
        body["repetition_penalty"] = r.into();
    }
    // Penalty params — supported by mlx_lm.server.
    if let Some(f) = params.frequency_penalty {
        body["frequency_penalty"] = f.into();
    }
    if let Some(p) = params.presence_penalty {
        body["presence_penalty"] = p.into();
    }
    // Note: mirostat / mirostat_tau / mirostat_eta are NOT supported by
    // mlx_lm.server — intentionally not forwarded.

    // Logprobs — supported by mlx_lm.server ≥ 0.19.
    if params.logprobs.unwrap_or(false) {
        body["logprobs"] = true.into();
        if let Some(n) = params.top_logprobs
            && n > 0
        {
            body["top_logprobs"] = (n as u64).into();
        }
    }
    // Grammar / response format — "__json__" is ax-serving's sentinel for
    // json_object mode; translate it to response_format (same as llamacpp.rs).
    if let Some(ref g) = params.grammar {
        if g == "__json__" {
            body["response_format"] = serde_json::json!({"type": "json_object"});
        }
        // mlx_lm.server does not support BNF grammars; non-__json__ values are
        // silently ignored (constrained generation requires mlx-engine/Outlines).
    } else if params.response_format.as_deref() == Some("json_object") {
        body["response_format"] = serde_json::json!({"type": "json_object"});
    }
    // Tool calling — pass definitions and choice verbatim.
    if let Some(ref tools) = params.tools {
        body["tools"] = tools.clone();
    }
    if let Some(ref tool_choice) = params.tool_choice {
        body["tool_choice"] = tool_choice.clone();
    }
    body["stream"] = params.stream.into();
    if params.stream {
        body["stream_options"] = serde_json::json!({"include_usage": true});
    }
}

fn build_mlx_chat_body(
    msgs: &[crate::ChatMessage],
    params: &GenerationParams,
) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = msgs
        .iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();
    let mut body = serde_json::json!({ "messages": messages });
    apply_mlx_generation_params(&mut body, params);
    body
}

fn build_mlx_completions_body(
    input: &GenerateInput,
    params: &GenerationParams,
) -> serde_json::Value {
    // Per OpenAI spec, `prompt` may be a string OR an array of token integers.
    // mlx-lm honours the array form; passing a JSON-serialised string "[1,2,3]"
    // would be treated as literal text, not tokens.
    let prompt: serde_json::Value = match input {
        GenerateInput::Text(t) => serde_json::Value::String(t.clone()),
        GenerateInput::Tokens(toks) => serde_json::Value::Array(
            toks.iter()
                .map(|&t| serde_json::Value::Number(t.into()))
                .collect(),
        ),
        GenerateInput::Chat(_) => serde_json::Value::String(String::new()),
    };
    let mut body = serde_json::json!({ "prompt": prompt });
    apply_mlx_generation_params(&mut body, params);
    body
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

fn post_mlx(
    http: &reqwest::blocking::Client,
    port: u16,
    path: &str,
    body: &serde_json::Value,
) -> Result<reqwest::blocking::Response> {
    http.post(format!("http://{MLX_LOCAL_HOST}:{port}{path}"))
        .json(body)
        .send()
        .with_context(|| format!("POST mlx_lm.server {path}"))
        .and_then(|r| {
            if r.status().is_client_error() || r.status().is_server_error() {
                let status = r.status();
                let text = r.text().unwrap_or_default();
                anyhow::bail!("mlx_lm.server {path} returned {status}: {text}")
            } else {
                Ok(r)
            }
        })
}

// ── Streaming (SSE) ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MlxSseDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct MlxLogprobEntry {
    logprob: f64,
    #[serde(default)]
    top_logprobs: Vec<MlxTopLogprob>,
}

#[derive(Deserialize)]
struct MlxTopLogprob {
    token: String,
    logprob: f64,
}

#[derive(Deserialize)]
struct MlxLogprobs {
    #[serde(default)]
    content: Vec<MlxLogprobEntry>,
}

#[derive(Deserialize)]
struct MlxSseChoice {
    delta: Option<MlxSseDelta>,
    finish_reason: Option<String>,
    logprobs: Option<MlxLogprobs>,
}

#[derive(Deserialize)]
struct MlxSseUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct MlxSseChunk {
    choices: Vec<MlxSseChoice>,
    usage: Option<MlxSseUsage>,
}

fn mlx_stream_chat(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    batch_size: usize,
    emit_logprobs: bool,
) -> Result<()> {
    let resp = post_mlx(http, port, "/v1/chat/completions", body)?;
    parse_mlx_sse(resp, tx, batch_size, emit_logprobs)
}

fn mlx_stream_completions(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    batch_size: usize,
    emit_logprobs: bool,
) -> Result<()> {
    let resp = post_mlx(http, port, "/v1/completions", body)?;
    parse_mlx_sse(resp, tx, batch_size, emit_logprobs)
}

fn parse_mlx_sse(
    resp: reqwest::blocking::Response,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    batch_size: usize,
    emit_logprobs: bool,
) -> Result<()> {
    let mut reader = std::io::BufReader::new(resp);
    let mut line = String::new();
    let mut buf = String::new();
    let mut buffered = 0usize;
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;
    let mut stop_reason = String::new();
    // When emitting logprobs, force batch_size=1 to preserve the 1:1
    // Token → TokenLogprob pairing (same strategy as LlamaCppBackend).
    let effective_batch = if emit_logprobs { 1 } else { batch_size };

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .context("reading MLX SSE stream")?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed == "data: [DONE]" {
            break;
        }
        let Some(json_str) = trimmed.strip_prefix("data: ") else {
            continue;
        };
        let Ok(chunk) = serde_json::from_str::<MlxSseChunk>(json_str) else {
            continue;
        };

        // Accumulate usage if present in this chunk.
        if let Some(u) = &chunk.usage {
            if let Some(p) = u.prompt_tokens {
                prompt_tokens = p;
            }
            if let Some(c) = u.completion_tokens {
                completion_tokens = c;
            }
        }

        let Some(choice) = chunk.choices.first() else {
            continue;
        };
        if let Some(r) = &choice.finish_reason
            && !r.is_empty()
        {
            stop_reason = r.clone();
        }

        let token_text = choice
            .delta
            .as_ref()
            .and_then(|d| d.content.as_deref())
            .unwrap_or("");
        if token_text.is_empty() {
            continue;
        }

        // Extract logprob data for this token if present.
        let lp_data = if emit_logprobs {
            choice
                .logprobs
                .as_ref()
                .and_then(|lp| lp.content.first())
                .map(|entry| {
                    let top: Vec<(String, f32)> = entry
                        .top_logprobs
                        .iter()
                        .map(|t| (t.token.clone(), t.logprob as f32))
                        .collect();
                    (entry.logprob as f32, top)
                })
        } else {
            None
        };

        if emit_logprobs {
            // Send each token individually with its logprob.
            if tx
                .blocking_send(GenerateEvent::Token(token_text.to_string()))
                .is_err()
            {
                return Ok(());
            }
            if let Some((logprob, top)) = lp_data {
                let _ = tx.blocking_send(GenerateEvent::TokenLogprob { logprob, top });
            }
        } else {
            buf.push_str(token_text);
            buffered += 1;
            if buffered >= effective_batch {
                if tx
                    .blocking_send(GenerateEvent::Token(std::mem::take(&mut buf)))
                    .is_err()
                {
                    return Ok(());
                }
                buffered = 0;
            }
        }
    }

    // Flush remaining buffered tokens (non-logprobs path only).
    if !buf.is_empty() {
        let _ = tx.blocking_send(GenerateEvent::Token(buf));
    }

    let _ = tx.blocking_send(GenerateEvent::Done(GenerationStats {
        prompt_tokens: prompt_tokens as usize,
        completion_tokens: completion_tokens as usize,
        stop_reason,
        ..Default::default()
    }));
    Ok(())
}

// ── Non-streaming ─────────────────────────────────────────────────────────────

fn mlx_complete_chat(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    emit_logprobs: bool,
) -> Result<()> {
    let val: serde_json::Value = post_mlx(http, port, "/v1/chat/completions", body)?
        .json()
        .context("decoding mlx chat completion response")?;
    emit_non_streaming_response(&val, tx, emit_logprobs)
}

fn mlx_complete_completions(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    emit_logprobs: bool,
) -> Result<()> {
    let val: serde_json::Value = post_mlx(http, port, "/v1/completions", body)?
        .json()
        .context("decoding mlx completion response")?;
    emit_non_streaming_response(&val, tx, emit_logprobs)
}

fn emit_non_streaming_response(
    val: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    emit_logprobs: bool,
) -> Result<()> {
    // Extract text from chat or text completion response.
    let text = val["choices"][0]["message"]["content"]
        .as_str()
        .or_else(|| val["choices"][0]["text"].as_str())
        .unwrap_or("")
        .to_string();

    let prompt_tokens = val["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
    let completion_tokens = val["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
    let stop_reason = val["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if emit_logprobs {
        // Parse per-token logprobs from the response (OpenAI format).
        if let Some(entries) = val["choices"][0]["logprobs"]["content"].as_array() {
            for entry in entries {
                let tok = entry["token"].as_str().unwrap_or("").to_string();
                let logprob = entry["logprob"].as_f64().unwrap_or(0.0) as f32;
                let top: Vec<(String, f32)> = entry["top_logprobs"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|t| {
                                Some((
                                    t["token"].as_str()?.to_string(),
                                    t["logprob"].as_f64()? as f32,
                                ))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if tx.blocking_send(GenerateEvent::Token(tok)).is_err() {
                    return Ok(());
                }
                let _ = tx.blocking_send(GenerateEvent::TokenLogprob { logprob, top });
            }
        } else if !text.is_empty() {
            let _ = tx.blocking_send(GenerateEvent::Token(text));
        }
    } else if !text.is_empty() {
        let _ = tx.blocking_send(GenerateEvent::Token(text));
    }
    let _ = tx.blocking_send(GenerateEvent::Done(GenerationStats {
        prompt_tokens,
        completion_tokens,
        stop_reason,
        ..Default::default()
    }));
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_mlx_model_rejects_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        assert!(!is_mlx_model(&gguf));
    }

    #[test]
    fn is_mlx_model_rejects_dir_without_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert!(!is_mlx_model(dir.path()));
    }

    #[test]
    fn is_mlx_model_rejects_dir_without_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"data").unwrap();
        assert!(!is_mlx_model(dir.path()));
    }

    #[test]
    fn is_mlx_model_accepts_valid_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"data").unwrap();
        assert!(is_mlx_model(dir.path()));
    }

    #[test]
    fn is_mlx_model_accepts_sharded_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"data").unwrap();
        assert!(is_mlx_model(dir.path()));
    }

    #[test]
    fn read_mlx_model_config_parses_fields() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"model_type": "llama", "num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096, "vocab_size": 32000, "max_position_embeddings": 8192}"#,
        )
        .unwrap();
        let cfg = read_mlx_model_config(dir.path());
        assert_eq!(cfg.model_type, "llama");
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.max_position_embeddings, 8192);
    }

    #[test]
    fn read_mlx_model_config_defaults_on_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let cfg = read_mlx_model_config(dir.path());
        assert_eq!(cfg.model_type, "mlx");
        assert_eq!(cfg.num_hidden_layers, 0);
    }

    #[test]
    fn read_mlx_model_config_defaults_on_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = read_mlx_model_config(dir.path());
        assert_eq!(cfg.model_type, "mlx");
    }
}
