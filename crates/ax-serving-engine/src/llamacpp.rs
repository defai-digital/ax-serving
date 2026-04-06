//! LlamaCppBackend: InferenceBackend implementation wrapping a `llama-server`
//! subprocess.
//!
//! # Lifecycle
//!
//! Each `load_model` call spawns a dedicated `llama-server` process on a
//! dynamically allocated port.  The process is killed when `unload_model` is
//! called or when `LlamaCppBackend` is dropped.
//!
//! # Health monitoring
//!
//! A dedicated `std::thread` per model polls `GET /health` every 5 s.
//! On 3 consecutive failures the server is restarted with exponential backoff
//! (2 s, 4 s, 8 s … capped at 16 s).  After 3 failed restart attempts the
//! model is marked `Dead`; subsequent `generate()` calls fail immediately
//! with a descriptive error.
//!
//! # Generation
//!
//! `generate` spawns a plain `std::thread` that does a blocking HTTP POST to
//! `llama-server`'s `/v1/completions` (streaming SSE).  Tokens are forwarded
//! via `tokio::sync::mpsc::Sender::blocking_send`, which is safe to call from
//! outside a Tokio context.
//!
//! # Requirements
//!
//! `llama-server` must be on `$PATH`.  The binary ships with llama.cpp builds
//! (e.g. `brew install llama.cpp` on macOS).

use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{info, warn};

/// Loopback host used for all llama-server subprocess communication.
/// llama-server is always spawned locally, so this never needs to be changed.
const LLAMACPP_LOCAL_HOST: &str = "127.0.0.1";

// ── LlamaCppConfig ────────────────────────────────────────────────────────────

/// Configuration for the llama.cpp subprocess backend.
///
/// Loaded from `config/serving.yaml` (under the `llamacpp:` key) and further
/// overridden by `AXS_*` environment variables via `apply_env_overrides()`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlamaCppConfig {
    /// SSE token batch size before each `blocking_send` (env: `AXS_LLAMACPP_TOKEN_BATCH`).
    pub token_batch_size: usize,
    /// Maximum allowed token batch size (clamp ceiling).
    pub token_batch_max: usize,
    /// Consecutive generate failures before circuit breaker trips (env: `AXS_CB_TRIP_THRESHOLD`).
    pub circuit_breaker_trip_threshold: u32,
    /// Recovery window (ms) — how long the breaker stays Open before HalfOpen
    /// (env: `AXS_CB_RECOVERY_MS`).
    pub circuit_breaker_recovery_ms: u64,
    /// HTTP request timeout for calls to llama-server (secs).
    pub http_request_timeout_secs: u64,
    /// Timeout waiting for llama-server to become ready after initial spawn (secs).
    pub server_startup_timeout_secs: u64,
    /// Timeout waiting for llama-server to become ready after a restart (secs).
    pub server_restart_timeout_secs: u64,
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
    /// Send `cache_prompt: true` in inference requests to enable llama-server's
    /// slot-based KV prefix reuse (env: `AXS_LLAMACPP_CACHE_PROMPT`).
    pub cache_prompt: bool,
    // ── Performance tuning ───────────────────────────────────────────────────
    /// Number of CPU threads for generation (env: `AXS_LLAMACPP_THREADS`).
    /// `None` = let llama-server decide (default: number of physical cores).
    pub n_threads: Option<u32>,
    /// Enable Flash Attention (env: `AXS_LLAMACPP_FLASH_ATTN`, default true).
    /// Reduces memory usage and speeds up prefill on Apple Silicon.
    pub flash_attn: bool,
    /// KV cache quantization type (env: `AXS_LLAMACPP_KV_TYPE`).
    /// Values: `"f16"` (default), `"q8_0"`, `"q4_0"`. Applied to both K and V.
    pub kv_cache_type: Option<String>,
    /// Batch size for prompt processing (env: `AXS_LLAMACPP_N_BATCH`, default 512).
    pub n_batch: Option<u32>,
    /// Physical (micro) batch size for prompt processing (env: `AXS_LLAMACPP_N_UBATCH`).
    /// Should match `n_batch` for embedding models. Defaults to llama-server's built-in
    /// default of 512 — inputs exceeding this limit fail with HTTP 500.
    pub n_ubatch: Option<u32>,
    /// Number of parallel request slots (env: `AXS_LLAMACPP_PARALLEL`, default 1).
    /// Increase to allow KV prefix reuse across concurrent requests.
    pub n_parallel: u32,
    /// Multimodal projector path for vision models (env: `AXS_LLAMACPP_MMPROJ`).
    /// Passed as `--mmproj` to llama-server.
    pub mmproj_path: Option<String>,
}

// ── LlamaCppConfig defaults ────────────────────────────────────────────────────
// All of these are exposed in serving.example.yaml and overridable via AXS_* env vars.

/// SSE tokens are batched before flushing to reduce syscalls; 4 is a low-latency default.
const DEFAULT_TOKEN_BATCH_SIZE: usize = 4;
/// Hard ceiling on token_batch_size — large batches increase TTFT noticeably.
const DEFAULT_TOKEN_BATCH_MAX: usize = 32;
/// Consecutive inference failures before the circuit breaker opens.
const DEFAULT_CB_TRIP_THRESHOLD: u32 = 3;
/// How long (ms) the circuit breaker stays open before allowing a retry.
const DEFAULT_CB_RECOVERY_MS: u64 = 10_000;
/// Timeout (secs) for a single llama-server inference HTTP request.
const DEFAULT_HTTP_REQUEST_TIMEOUT_SECS: u64 = 300;
/// Time (secs) to wait for llama-server to become ready on first spawn.
const DEFAULT_SERVER_STARTUP_TIMEOUT_SECS: u64 = 120;
/// Time (secs) to wait for llama-server to become ready after a restart.
const DEFAULT_SERVER_RESTART_TIMEOUT_SECS: u64 = 60;
/// Sleep (ms) between readiness poll attempts in `wait_ready()`.
const DEFAULT_WAIT_READY_POLL_INTERVAL_MS: u64 = 500;
/// Per-attempt timeout (secs) for each readiness probe in `wait_ready()`.
const DEFAULT_WAIT_READY_CHECK_TIMEOUT_SECS: u64 = 1;
/// How often (secs) the background health poller ticks.
const DEFAULT_HEALTH_POLLER_INTERVAL_SECS: u64 = 5;
/// Per-request timeout (secs) for background health poll requests.
const DEFAULT_HEALTH_POLLER_CHECK_TIMEOUT_SECS: u64 = 2;
/// Consecutive health failures before attempting a restart.
const DEFAULT_HEALTH_POLLER_RESTART_THRESHOLD: u32 = 3;
/// Maximum restart attempts before marking the model Dead.
const DEFAULT_HEALTH_POLLER_MAX_RESTARTS: u32 = 3;
/// Default parallel slots — 1 means no concurrent requests share a KV cache.
const DEFAULT_N_PARALLEL: u32 = 1;

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            token_batch_size: DEFAULT_TOKEN_BATCH_SIZE,
            token_batch_max: DEFAULT_TOKEN_BATCH_MAX,
            circuit_breaker_trip_threshold: DEFAULT_CB_TRIP_THRESHOLD,
            circuit_breaker_recovery_ms: DEFAULT_CB_RECOVERY_MS,
            http_request_timeout_secs: DEFAULT_HTTP_REQUEST_TIMEOUT_SECS,
            server_startup_timeout_secs: DEFAULT_SERVER_STARTUP_TIMEOUT_SECS,
            server_restart_timeout_secs: DEFAULT_SERVER_RESTART_TIMEOUT_SECS,
            wait_ready_poll_interval_ms: DEFAULT_WAIT_READY_POLL_INTERVAL_MS,
            wait_ready_check_timeout_secs: DEFAULT_WAIT_READY_CHECK_TIMEOUT_SECS,
            health_poller_interval_secs: DEFAULT_HEALTH_POLLER_INTERVAL_SECS,
            health_poller_check_timeout_secs: DEFAULT_HEALTH_POLLER_CHECK_TIMEOUT_SECS,
            health_poller_restart_threshold: DEFAULT_HEALTH_POLLER_RESTART_THRESHOLD,
            health_poller_max_restarts: DEFAULT_HEALTH_POLLER_MAX_RESTARTS,
            cache_prompt: true,
            n_threads: None,
            flash_attn: true,
            kv_cache_type: None,
            n_batch: None,
            n_ubatch: None,
            n_parallel: DEFAULT_N_PARALLEL,
            mmproj_path: None,
        }
    }
}

impl LlamaCppConfig {
    /// Apply `AXS_*` env var overrides on top of YAML-loaded values.
    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("AXS_LLAMACPP_TOKEN_BATCH")
            && let Ok(n) = v.parse::<usize>()
        {
            self.token_batch_size = n.clamp(1, self.token_batch_max);
        }
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
        if let Ok(v) = std::env::var("AXS_LLAMACPP_CACHE_PROMPT") {
            self.cache_prompt = v != "0" && v.to_lowercase() != "false";
        }
        if let Ok(v) = std::env::var("AXS_LLAMACPP_THREADS")
            && let Ok(n) = v.parse::<u32>()
        {
            self.n_threads = Some(n);
        }
        if let Ok(v) = std::env::var("AXS_LLAMACPP_FLASH_ATTN") {
            self.flash_attn = v != "0" && v.to_lowercase() != "false";
        }
        if let Ok(v) = std::env::var("AXS_LLAMACPP_KV_TYPE") {
            self.kv_cache_type = Some(v);
        }
        if let Ok(v) = std::env::var("AXS_LLAMACPP_N_BATCH")
            && let Ok(n) = v.parse::<u32>()
        {
            self.n_batch = Some(n);
        }
        if let Ok(v) = std::env::var("AXS_LLAMACPP_N_UBATCH")
            && let Ok(n) = v.parse::<u32>()
        {
            self.n_ubatch = Some(n);
        }
        if let Ok(v) = std::env::var("AXS_LLAMACPP_PARALLEL")
            && let Ok(n) = v.parse::<u32>()
        {
            self.n_parallel = n.max(1);
        }
        if let Ok(v) = std::env::var("AXS_LLAMACPP_MMPROJ") {
            self.mmproj_path = Some(v);
        }
    }

    /// Create from env vars only (no YAML), using struct defaults as the base.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.apply_env_overrides();
        cfg
    }

    /// Effective batch size, clamped to `[1, token_batch_max]`.
    pub fn effective_batch_size(&self) -> usize {
        self.token_batch_size.clamp(1, self.token_batch_max)
    }
}

fn unix_ms_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

use crate::{
    CacheTelemetry, EmbedConfig, EmbedInput, EmbedResult, GenerateEvent, GenerateInput,
    GenerationParams, GenerationStats, InferenceBackend, LoadConfig, ModelHandle, ModelMetadata,
    ThermalMonitor, ThermalState, gguf_meta::GgufMeta,
};

// ── Health state ──────────────────────────────────────────────────────────────

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HealthState {
    Healthy = 0,
    Unhealthy = 1,
    Dead = 2,
}

// ── Circuit breaker ───────────────────────────────────────────────────────────

/// Circuit breaker states (stored as `u8` in `AtomicU8`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    /// Requests flow normally.
    Closed = 0,
    /// Requests are rejected until the recovery window elapses.
    Open = 1,
    /// Recovery window elapsed; one probe request is allowed through.
    HalfOpen = 2,
}

/// Per-model circuit breaker for the llama-server subprocess.
///
/// Trip threshold (`AXS_CB_TRIP_THRESHOLD`, default 3): consecutive generate
/// failures before the breaker opens.
///
/// Recovery window (`AXS_CB_RECOVERY_MS`, default 10 000 ms): how long the
/// breaker stays Open before transitioning to HalfOpen.
struct CircuitBreaker {
    state: AtomicU8,
    consecutive_generate_failures: AtomicU32,
    last_opened_ms: AtomicU64,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            consecutive_generate_failures: AtomicU32::new(0),
            last_opened_ms: AtomicU64::new(0),
        }
    }

    fn trip(&self) {
        self.state.store(CircuitState::Open as u8, Ordering::SeqCst);
        self.last_opened_ms.store(unix_ms_now(), Ordering::Relaxed);
        self.consecutive_generate_failures
            .store(0, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::SeqCst);
        self.consecutive_generate_failures
            .store(0, Ordering::Relaxed);
    }
}

static NEXT_LLAMACPP_HANDLE: AtomicU64 = AtomicU64::new(1_000_000);

fn next_llamacpp_handle() -> ModelHandle {
    ModelHandle(NEXT_LLAMACPP_HANDLE.fetch_add(1, Ordering::Relaxed))
}

type BlockingJob = Box<dyn FnOnce() + Send + 'static>;

/// Small blocking executor to avoid per-request OS thread creation in generate().
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
                .name(format!("ax-llamacpp-gen-{i}"))
                .spawn(move || {
                    loop {
                        let recv_result = {
                            let guard = match rx.lock() {
                                Ok(guard) => guard,
                                Err(err) => {
                                    warn!("blocking executor receiver lock poisoned; continuing with poisoned state");
                                    err.into_inner()
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
                Ok(handle) => workers.push(handle),
                Err(err) => {
                    warn!(%err, thread_idx = i, "failed to spawn blocking executor worker");
                }
            }
        }
        Self {
            tx,
            _workers: workers,
        }
    }

    fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.tx
            .send(Box::new(f))
            .map_err(|_| anyhow::anyhow!("blocking executor stopped"))
    }
}

// ── Per-model subprocess entry ────────────────────────────────────────────────

struct LlamaCppProcess {
    port: u16,
    /// Child process shared with the health-poller thread.
    child: Arc<Mutex<Option<std::process::Child>>>,
    /// Atomic health state written by the poller, read by `generate`.
    health: Arc<AtomicU8>,
    /// Set to `true` by `Drop` to signal the poller to exit.
    stop: Arc<std::sync::atomic::AtomicBool>,
    /// Poller thread handle — dropping detaches the thread (does not join).
    _poller: std::thread::JoinHandle<()>,
    /// Circuit breaker for generate() failures.
    breaker: Arc<CircuitBreaker>,
    /// EOS token ID queried from the model at load time.
    eos_token: u32,
}

impl Drop for LlamaCppProcess {
    fn drop(&mut self) {
        // Signal poller to stop before killing the child so it doesn't
        // attempt a restart.
        self.stop.store(true, Ordering::SeqCst);
        let mut guard = match self.child.lock() {
            Ok(guard) => guard,
            Err(err) => {
                warn!(
                    %err,
                    "llama.cpp child lock poisoned during drop; continuing with poisoned state"
                );
                err.into_inner()
            }
        };
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // `_poller` JoinHandle drops here — detaches the thread.
    }
}

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct LlamaCppBackend {
    models: Arc<Mutex<HashMap<ModelHandle, LlamaCppProcess>>>,
    thermal: ThermalMonitor,
    http: reqwest::blocking::Client,
    config: Arc<LlamaCppConfig>,
    executor: Arc<BlockingExecutor>,
}

impl LlamaCppBackend {
    fn models_read(&self) -> MutexGuard<'_, HashMap<ModelHandle, LlamaCppProcess>> {
        match self.models.lock() {
            Ok(guard) => guard,
            Err(err) => {
                warn!(
                    %err,
                    "llama.cpp model registry read lock poisoned; continuing with poisoned state"
                );
                err.into_inner()
            }
        }
    }

    fn models_write(&self) -> MutexGuard<'_, HashMap<ModelHandle, LlamaCppProcess>> {
        match self.models.lock() {
            Ok(guard) => guard,
            Err(err) => {
                warn!(
                    %err,
                    "llama.cpp model registry write lock poisoned; continuing with poisoned state"
                );
                err.into_inner()
            }
        }
    }

    fn effective_n_gpu_layers(config: &LoadConfig) -> i32 {
        if config.backend_type == crate::BackendType::Cpu {
            0
        } else {
            config.llama_cpp_n_gpu_layers.unwrap_or(99)
        }
    }

    pub fn new(config: LlamaCppConfig) -> Self {
        let http = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.http_request_timeout_secs))
            .build()
        {
            Ok(http) => http,
            Err(err) => {
                warn!(
                    %err,
                    "failed to build blocking reqwest client with configured timeout; falling back to default"
                );
                reqwest::blocking::Client::new()
            }
        };
        let executor_threads = std::env::var("AXS_LLAMACPP_EXECUTOR_THREADS")
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

    /// Spawn `llama-server` on `port` with the given model file.
    pub(crate) fn spawn_server(
        path: &Path,
        port: u16,
        load_config: &LoadConfig,
        llama_config: &LlamaCppConfig,
    ) -> Result<std::process::Child> {
        let ngl = Self::effective_n_gpu_layers(load_config);

        let mut cmd = std::process::Command::new("llama-server");
        cmd.arg("--model")
            .arg(path)
            .arg("--host")
            .arg(LLAMACPP_LOCAL_HOST)
            .arg("--port")
            .arg(port.to_string())
            .arg("--n-gpu-layers")
            .arg(ngl.to_string())
            .arg("--log-disable")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        if load_config.context_length > 0 {
            cmd.arg("--ctx-size")
                .arg(load_config.context_length.to_string());
        }

        // ── Performance tuning ────────────────────────────────────────────────
        if let Some(t) = llama_config.n_threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        cmd.arg("--flash-attn")
            .arg(if llama_config.flash_attn { "on" } else { "off" });
        if let Some(ref kv) = llama_config.kv_cache_type {
            cmd.arg("--cache-type-k")
                .arg(kv)
                .arg("--cache-type-v")
                .arg(kv);
        }
        if let Some(b) = llama_config.n_batch {
            cmd.arg("--batch-size").arg(b.to_string());
        }
        if let Some(ub) = llama_config.n_ubatch {
            cmd.arg("--ubatch-size").arg(ub.to_string());
        }
        if llama_config.n_parallel > 1 {
            cmd.arg("--parallel")
                .arg(llama_config.n_parallel.to_string());
        }

        // ── Vision (multimodal projector) ─────────────────────────────────────
        // LoadConfig.mmproj_path overrides the global LlamaCppConfig setting so
        // individual model loads can specify their own projector file.
        let mmproj = load_config
            .mmproj_path
            .as_deref()
            .or(llama_config.mmproj_path.as_deref());
        if let Some(mp) = mmproj {
            cmd.arg("--mmproj").arg(mp);
        }

        // ── Embedding mode ────────────────────────────────────────────────────
        // Without --embedding, llama-server does not register /v1/embeddings.
        // Set via LoadConfig.enable_embeddings (auto-detected from pooling_type).
        if load_config.enable_embeddings == Some(true) {
            cmd.arg("--embedding");
        }
        if let Some(pooling) = load_config
            .pooling_type
            .as_deref()
            .filter(|v| !v.is_empty())
        {
            cmd.arg("--pooling").arg(pooling);
        }

        cmd.spawn()
            .context("failed to spawn llama-server; is it on PATH?")
    }

    /// Poll `GET /health` until the server is ready or `startup_timeout` elapses.
    ///
    /// `poll_interval` — sleep between attempts; `check_timeout` — per-request timeout.
    pub(crate) fn wait_ready(
        http: &reqwest::blocking::Client,
        port: u16,
        startup_timeout: Duration,
        poll_interval: Duration,
        check_timeout: Duration,
    ) -> Result<()> {
        let url = format!("http://{LLAMACPP_LOCAL_HOST}:{port}/health");
        let deadline = Instant::now() + startup_timeout;
        loop {
            match http.get(&url).timeout(check_timeout).send() {
                Ok(r) if r.status().is_success() => return Ok(()),
                _ => {}
            }
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "llama-server on port {port} did not become ready within {startup_timeout:?}"
                );
            }
            std::thread::sleep(poll_interval);
        }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new(LlamaCppConfig::from_env())
    }
}

impl InferenceBackend for LlamaCppBackend {
    // ── Model lifecycle ───────────────────────────────────────────────────────

    fn load_model(
        &self,
        path: &Path,
        mut config: LoadConfig,
    ) -> Result<(ModelHandle, ModelMetadata)> {
        anyhow::ensure!(path.exists(), "model file not found: {}", path.display());
        if let Some(raw_pooling) = config.pooling_type.as_deref() {
            let canonical = normalize_pooling_type(raw_pooling).ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid pooling_type '{}' ; expected one of: none, mean, cls, last, rank",
                    raw_pooling
                )
            })?;
            config.pooling_type = Some(canonical);
        }

        // Read GGUF metadata before spawning the server — fast (≤ 4096 bytes).
        let gguf_meta = crate::gguf_meta::read_gguf_meta(path).ok();

        // Auto-detect embedding models: pooling_type > 0 means the model is an
        // embedding model and llama-server must be started with --embedding.
        if config.enable_embeddings.is_none()
            && let Some(ref m) = gguf_meta
            && m.pooling_type > 0
        {
            config.enable_embeddings = Some(true);
        }
        if config.enable_embeddings.is_none() && config.pooling_type.is_some() {
            config.enable_embeddings = Some(true);
        }

        let port = find_free_port().context("failed to find free TCP port for llama-server")?;
        info!(
            "spawning llama-server for {} on port {}",
            path.display(),
            port
        );

        let start = Instant::now();
        let child = Self::spawn_server(path, port, &config, &self.config)
            .with_context(|| format!("spawning llama-server for {}", path.display()))?;

        Self::wait_ready(
            &self.http,
            port,
            Duration::from_secs(self.config.server_startup_timeout_secs),
            Duration::from_millis(self.config.wait_ready_poll_interval_ms),
            Duration::from_secs(self.config.wait_ready_check_timeout_secs),
        )
        .with_context(|| format!("waiting for llama-server on port {port}"))?;

        let load_ms = start.elapsed().as_millis() as u64;
        info!("llama-server ready on port {port} in {load_ms}ms");

        // Metal is used unless the caller explicitly requested 0 GPU layers
        // (CPU-only mode).  The default (None) maps to -ngl 99 (all layers on GPU).
        let resolved_backend = if config.llama_cpp_n_gpu_layers == Some(0) {
            crate::BackendType::Cpu
        } else {
            crate::BackendType::Metal
        };
        let meta = fetch_model_meta(
            &self.http,
            port,
            config.context_length,
            load_ms,
            gguf_meta,
            resolved_backend,
        );
        let handle = next_llamacpp_handle();

        // Query the actual EOS token from the llama-server /props endpoint.
        let eos_token = self
            .http
            .get(format!("http://{LLAMACPP_LOCAL_HOST}:{port}/props"))
            .timeout(Duration::from_secs(5))
            .send()
            .ok()
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|v| {
                v["default_generation_settings"]["eos_token_id"]
                    .as_u64()
                    .or_else(|| v["eos_token_id"].as_u64())
            })
            .unwrap_or(2) as u32;

        // Wrap child in Arc<Mutex> for shared access with the poller thread.
        let child_arc = Arc::new(Mutex::new(Some(child)));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let health = Arc::new(AtomicU8::new(HealthState::Healthy as u8));
        let breaker = Arc::new(CircuitBreaker::new());

        // Spawn the background health poller.
        let poller = {
            let args = PollerArgs {
                port,
                path: path.to_path_buf(),
                load_config: config,
                llama_config: Arc::clone(&self.config),
                child: Arc::clone(&child_arc),
                http: self.http.clone(),
                health: Arc::clone(&health),
                stop: Arc::clone(&stop),
                breaker: Arc::clone(&breaker),
                poller_interval: Duration::from_secs(self.config.health_poller_interval_secs),
                check_timeout: Duration::from_secs(self.config.health_poller_check_timeout_secs),
                restart_threshold: self.config.health_poller_restart_threshold,
                max_restarts: self.config.health_poller_max_restarts,
                restart_wait_timeout: Duration::from_secs(self.config.server_restart_timeout_secs),
                wait_ready_poll_interval: Duration::from_millis(
                    self.config.wait_ready_poll_interval_ms,
                ),
                wait_ready_check_timeout: Duration::from_secs(
                    self.config.wait_ready_check_timeout_secs,
                ),
            };
            match std::thread::Builder::new()
                .name(format!("ax-llamacpp-health-{port}"))
                .spawn(move || run_health_poller(args))
            {
                Ok(poller) => poller,
                Err(err) => {
                    warn!(
                        %err,
                        port,
                        "failed to spawn llama-server health poller; cleaning up model process"
                    );
                    {
                        let mut guard = match child_arc.lock() {
                            Ok(guard) => guard,
                            Err(lock_err) => {
                                warn!(
                                    %lock_err,
                                    "llama.cpp child lock poisoned during load cleanup; continuing with poisoned state"
                                );
                                lock_err.into_inner()
                            }
                        };
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                    return Err(anyhow::anyhow!(
                        "failed to spawn llama-server health poller: {err}"
                    ));
                }
            }
        };

        self.models_write().insert(
            handle,
            LlamaCppProcess {
                port,
                child: child_arc,
                health,
                stop,
                _poller: poller,
                breaker,
                eos_token,
            },
        );

        Ok((handle, meta))
    }

    fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let entry = self.models_write().remove(&handle);
        anyhow::ensure!(
            entry.is_some(),
            "no llama.cpp model loaded with handle {:?}",
            handle
        );
        // Drop triggers LlamaCppProcess::drop → stop signal + kill.
        info!("unloaded llama.cpp model {:?}", handle);
        Ok(())
    }

    // ── Generation ────────────────────────────────────────────────────────────

    fn generate(
        &self,
        handle: ModelHandle,
        input: GenerateInput,
        params: GenerationParams,
        tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> Result<()> {
        let (port, health, breaker) = {
            let guard = self.models_read();
            let proc = guard
                .get(&handle)
                .ok_or_else(|| anyhow::anyhow!("invalid llama.cpp model handle {:?}", handle))?;

            // Fail fast if the server is permanently dead (bypasses circuit breaker).
            if proc.health.load(Ordering::Relaxed) == HealthState::Dead as u8 {
                anyhow::bail!(
                    "llama-server for handle {:?} has permanently failed; unload and reload the model",
                    handle
                );
            }

            // Circuit breaker check.
            let cb_state = proc.breaker.state.load(Ordering::Relaxed);
            if cb_state == CircuitState::Open as u8 {
                let opened_ms = proc.breaker.last_opened_ms.load(Ordering::Relaxed);
                let elapsed_ms = unix_ms_now().saturating_sub(opened_ms);
                if elapsed_ms < self.config.circuit_breaker_recovery_ms {
                    anyhow::bail!(
                        "circuit open: llama-server for handle {:?} is recovering; retry later",
                        handle
                    );
                }
                // Recovery window elapsed → allow one probe through.
                // If the CAS fails, inspect why: if the state is still Open
                // (health poller re-tripped the breaker between our elapsed-time
                // check and now), bail out; otherwise (Closed or HalfOpen) proceed.
                if let Err(current) = proc.breaker.state.compare_exchange(
                    CircuitState::Open as u8,
                    CircuitState::HalfOpen as u8,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) && current == CircuitState::Open as u8
                {
                    anyhow::bail!(
                        "circuit open: recovery window was reset for handle {:?}; retry later",
                        handle
                    );
                } else {
                    // HalfOpen or Closed — another thread already transitioned;
                    // proceed with the probe.
                }
            }

            (
                proc.port,
                Arc::clone(&proc.health),
                Arc::clone(&proc.breaker),
            )
        };
        let _health = health; // held until end of scope; freed after generate() returns

        let http = self.http.clone();
        let batch_size = self.config.effective_batch_size();
        let trip_threshold = self.config.circuit_breaker_trip_threshold;
        let cache_prompt = self.config.cache_prompt;
        let emit_logprobs = params.logprobs.unwrap_or(false);
        let stream = params.stream;

        self.executor.execute(move || {
            let result = match (&input, stream) {
                (GenerateInput::Chat(msgs), true) => {
                    let body = build_chat_completions_body(msgs, &params, cache_prompt);
                    stream_chat_completions(&http, port, &body, &tx, batch_size, emit_logprobs)
                }
                (GenerateInput::Chat(msgs), false) => {
                    let body = build_chat_completions_body(msgs, &params, cache_prompt);
                    complete_chat_completions(&http, port, &body, &tx, emit_logprobs)
                }
                (_, true) => {
                    build_completions_body(&input, &params, cache_prompt).and_then(|body| {
                        stream_completions(&http, port, &body, &tx, batch_size, emit_logprobs)
                    })
                }
                (_, false) => build_completions_body(&input, &params, cache_prompt)
                    .and_then(|body| complete_completions(&http, port, &body, &tx, emit_logprobs)),
            };

            match result {
                Ok(()) => {
                    // On success, always reset the consecutive failure counter so
                    // sub-threshold failures don't accumulate across requests.
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
                    // On failure, update circuit breaker.
                    let was_half_open =
                        breaker.state.load(Ordering::Relaxed) == CircuitState::HalfOpen as u8;
                    let failures = breaker
                        .consecutive_generate_failures
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    if was_half_open || failures >= trip_threshold {
                        breaker.trip();
                    }
                    warn!("llama.cpp stream error: {e}");
                    let _ = tx.blocking_send(GenerateEvent::Error(e.to_string()));
                }
            }
        })?;
        Ok(())
    }

    // ── Tokenization ──────────────────────────────────────────────────────────

    fn tokenize(&self, handle: ModelHandle, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let port = {
            let guard = self.models_read();
            guard
                .get(&handle)
                .map(|p| p.port)
                .ok_or_else(|| anyhow::anyhow!("invalid llama.cpp model handle {:?}", handle))?
        };
        let body = serde_json::json!({
            "content": text,
            "add_special": add_bos,
        });
        let resp: serde_json::Value = self
            .http
            .post(format!("http://{LLAMACPP_LOCAL_HOST}:{port}/tokenize"))
            .json(&body)
            .send()
            .context("POST /tokenize")?
            .error_for_status()
            .context("llama-server /tokenize error")?
            .json()
            .context("decoding /tokenize response")?;
        let tokens = resp["tokens"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("/tokenize response missing 'tokens' array"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| anyhow::anyhow!("non-integer token id"))
                    .map(|n| n as u32)
            })
            .collect::<Result<Vec<u32>>>()?;
        Ok(tokens)
    }

    fn decode_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<String> {
        let port = {
            let guard = self.models_read();
            guard
                .get(&handle)
                .map(|p| p.port)
                .ok_or_else(|| anyhow::anyhow!("invalid llama.cpp model handle {:?}", handle))?
        };
        let token_values: Vec<serde_json::Value> = tokens
            .iter()
            .map(|&t| serde_json::Value::Number(t.into()))
            .collect();
        let body = serde_json::json!({ "tokens": token_values });
        let resp: serde_json::Value = self
            .http
            .post(format!("http://{LLAMACPP_LOCAL_HOST}:{port}/detokenize"))
            .json(&body)
            .send()
            .context("POST /detokenize")?
            .error_for_status()
            .context("llama-server /detokenize error")?
            .json()
            .context("decoding /detokenize response")?;
        let content = resp["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("/detokenize response missing 'content' field"))?
            .to_string();
        Ok(content)
    }

    fn eos_tokens(&self, handle: ModelHandle) -> Result<Vec<u32>> {
        let guard = self.models_read();
        let proc = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown model handle {:?}", handle))?;
        Ok(vec![proc.eos_token])
    }

    fn embed(
        &self,
        handle: ModelHandle,
        inputs: &EmbedInput<'_>,
        config: &EmbedConfig,
    ) -> Result<EmbedResult> {
        let port = {
            let guard = self.models_read();
            guard
                .get(&handle)
                .map(|p| p.port)
                .ok_or_else(|| anyhow::anyhow!("invalid llama.cpp model handle {:?}", handle))?
        };

        // Build the input value: string array or token array.
        let input_json = match inputs {
            EmbedInput::Strings(texts) => serde_json::to_value(texts)?,
            EmbedInput::Tokens(seqs) => serde_json::to_value(seqs)?,
        };

        let body = serde_json::json!({
            "input": input_json,
            "normalize": config.normalize,
            "truncate": config.truncate,
        });

        let resp: serde_json::Value = self
            .http
            .post(format!("http://{LLAMACPP_LOCAL_HOST}:{port}/v1/embeddings"))
            .json(&body)
            .send()
            .context("POST /v1/embeddings")?
            .error_for_status()
            .context("llama-server /v1/embeddings error")?
            .json()
            .context("decoding /v1/embeddings response")?;

        let prompt_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;

        let data = resp["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("/v1/embeddings response missing 'data' array"))?;

        // Collect in index order — llama-server may reorder batches.
        let mut indexed: Vec<(usize, Vec<f32>)> = data
            .iter()
            .map(|item| {
                let index = item["index"].as_u64().unwrap_or(0) as usize;
                let embedding = item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing 'embedding' in data[{index}]"))?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .ok_or_else(|| anyhow::anyhow!("non-float in embedding array"))
                    })
                    .collect::<Result<Vec<f32>>>()?;
                Ok((index, embedding))
            })
            .collect::<Result<Vec<_>>>()?;

        indexed.sort_unstable_by_key(|(i, _)| *i);
        let embeddings = indexed.into_iter().map(|(_, v)| v).collect();

        Ok(EmbedResult {
            embeddings,
            prompt_tokens,
        })
    }

    fn eval_tokens(&self, _handle: ModelHandle, _tokens: &[u32]) -> Result<u32> {
        Err(anyhow::anyhow!(
            "eval_tokens not supported by LlamaCppBackend"
        ))
    }

    // ── Thermal ───────────────────────────────────────────────────────────────

    fn thermal_state(&self) -> ThermalState {
        self.thermal.current()
    }

    fn recommended_concurrency(&self) -> usize {
        self.thermal.recommended_concurrency()
    }

    fn cache_telemetry(&self) -> CacheTelemetry {
        // Snapshot port list under the lock, then release before making HTTP calls
        // to avoid blocking all model operations during telemetry collection.
        let ports: Vec<u16> = {
            let guard = self.models_read();
            guard.values().map(|p| p.port).collect()
        };
        let mut total = CacheTelemetry::default();
        for port in ports {
            let url = format!("http://{LLAMACPP_LOCAL_HOST}:{port}/health");
            let resp = match self.http.get(&url).timeout(Duration::from_secs(1)).send() {
                Ok(r) if r.status().is_success() => r,
                _ => continue,
            };
            let json: serde_json::Value = match resp.json() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let idle = json.get("slots_idle").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let processing = json
                .get("slots_processing")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            total.active_batch_size += processing;
            total.max_batch_size += idle + processing;
        }
        total
    }
}

// ── Health poller ─────────────────────────────────────────────────────────────

/// Arguments for `run_health_poller` (grouped to stay under the clippy arg limit).
struct PollerArgs {
    port: u16,
    path: PathBuf,
    load_config: LoadConfig,
    llama_config: Arc<LlamaCppConfig>,
    child: Arc<Mutex<Option<std::process::Child>>>,
    http: reqwest::blocking::Client,
    health: Arc<AtomicU8>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    breaker: Arc<CircuitBreaker>,
    // LlamaCppConfig-derived values so the poller honours the YAML config.
    poller_interval: Duration,
    check_timeout: Duration,
    restart_threshold: u32,
    max_restarts: u32,
    restart_wait_timeout: Duration,
    wait_ready_poll_interval: Duration,
    wait_ready_check_timeout: Duration,
}

/// Background thread: polls `/health`, restarts llama-server on failure.
///
/// On successful restart, resets the circuit breaker to `Closed` so that
/// queued generate() calls can flow through without waiting for the recovery
/// window.
fn run_health_poller(args: PollerArgs) {
    let PollerArgs {
        port,
        path,
        load_config,
        llama_config,
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

    let url = format!("http://{LLAMACPP_LOCAL_HOST}:{port}/health");
    let mut consecutive_failures = 0u32;
    let mut restart_count = 0u32;

    loop {
        std::thread::sleep(poller_interval);

        if stop.load(Ordering::Relaxed) {
            break;
        }

        let ok = http
            .get(&url)
            .timeout(check_timeout)
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        if ok {
            if health.load(Ordering::Relaxed) != HealthState::Healthy as u8 {
                info!("llama-server on port {port} recovered");
            }
            health.store(HealthState::Healthy as u8, Ordering::Relaxed);
            consecutive_failures = 0;
            continue;
        }

        consecutive_failures += 1;
        health.store(HealthState::Unhealthy as u8, Ordering::Relaxed);
        warn!("llama-server port {port} health check failed ({consecutive_failures} consecutive)");

        if consecutive_failures < restart_threshold {
            continue;
        }

        // Consecutive failures hit threshold → attempt restart with exponential backoff.
        if restart_count >= max_restarts {
            health.store(HealthState::Dead as u8, Ordering::Relaxed);
            warn!(
                "llama-server port {port} permanently dead after {max_restarts} restart attempts"
            );
            break;
        }

        // Count every attempt (not just successful ones) so max_restarts is
        // applied consistently regardless of whether the failure is in spawn or
        // wait_ready.
        restart_count += 1;
        let backoff = Duration::from_secs(1u64 << restart_count.min(4));
        warn!(
            "restarting llama-server port {port} (attempt {restart_count}/{max_restarts}) in {backoff:?}"
        );
        std::thread::sleep(backoff);

        if stop.load(Ordering::Relaxed) {
            break;
        }

        // Kill old child and spawn a fresh server on the same port.
        {
            let mut guard = match child.lock() {
                Ok(guard) => guard,
                Err(err) => {
                    warn!(
                        %err,
                        port,
                        "llama.cpp child lock poisoned during health poller restart; continuing with poisoned state"
                    );
                    err.into_inner()
                }
            };
            if let Some(mut old) = guard.take() {
                let _ = old.kill();
                let _ = old.wait();
            }
            match LlamaCppBackend::spawn_server(&path, port, &load_config, &llama_config) {
                Ok(new_child) => {
                    *guard = Some(new_child);
                }
                Err(e) => {
                    warn!("failed to spawn llama-server: {e}; will retry if budget remains");
                    continue;
                }
            }
        }

        match LlamaCppBackend::wait_ready(
            &http,
            port,
            restart_wait_timeout,
            wait_ready_poll_interval,
            wait_ready_check_timeout,
        ) {
            Ok(()) => {
                health.store(HealthState::Healthy as u8, Ordering::Relaxed);
                // Successful restart: reset circuit breaker so generate() can proceed.
                breaker.reset();
                consecutive_failures = 0;
                info!("llama-server port {port} restarted successfully (attempt {restart_count})");
            }
            Err(e) => {
                warn!("llama-server port {port} failed to start after restart: {e}");
                // Kill the orphaned child to prevent resource leak (BUG-050).
                if let Ok(mut guard) = child.lock()
                    && let Some(c) = guard.as_mut()
                {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                health.store(HealthState::Dead as u8, Ordering::Relaxed);
                break;
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find a free TCP port by binding to :0.
fn find_free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(format!("{LLAMACPP_LOCAL_HOST}:0"))?;
    Ok(listener.local_addr()?.port())
    // listener drops here, releasing the port.
    // TOCTOU: negligible for local use.
}

/// Build a `/v1/completions` JSON body for Text or Tokens inputs.
///
/// Chat inputs must use `build_chat_completions_body` + `/v1/chat/completions`
/// so that llama-server applies the model's GGUF chat template.
fn build_completions_body(
    input: &GenerateInput,
    params: &GenerationParams,
    cache_prompt: bool,
) -> anyhow::Result<serde_json::Value> {
    let mut body = serde_json::json!({
        "stream": params.stream,
        "cache_prompt": cache_prompt,
    });

    match input {
        GenerateInput::Text(t) => {
            body["prompt"] = t.clone().into();
        }
        // Pass pre-tokenized input via the llama-server `tokens` field so the
        // server treats them as token IDs, not as literal text characters.
        GenerateInput::Tokens(toks) => {
            let token_array = serde_json::Value::Array(
                toks.iter()
                    .map(|&t| serde_json::Value::Number(t.into()))
                    .collect(),
            );
            body["tokens"] = token_array.clone();
            // Compatibility: some llama-server OpenAI paths require `prompt`
            // even for tokenized input. Provide the same token-id array there.
            body["prompt"] = token_array;
        }
        GenerateInput::Chat(_) => {
            // If this path is reached, request dispatching already violated the expected
            // input protocol for this endpoint.
            anyhow::bail!("chat inputs must use build_chat_completions_body");
        }
    };

    apply_generation_params(&mut body, params);
    Ok(body)
}

/// Build a `/v1/chat/completions` JSON body for Chat inputs.
///
/// llama-server applies the model's built-in GGUF chat template on this
/// endpoint, unlike `/v1/completions` which treats the prompt as raw text.
fn build_chat_completions_body(
    msgs: &[crate::ChatMessage],
    params: &GenerationParams,
    cache_prompt: bool,
) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = msgs
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();

    let mut body = serde_json::json!({
        "messages": messages,
        "stream": params.stream,
        // Enable llama-server slot-based KV prefix reuse. Requests sharing
        // the same prompt prefix reuse the already-computed K/V tensors,
        // reducing prefill time to near-zero for warm slots.
        "cache_prompt": cache_prompt,
    });
    if params.stream {
        // Request usage stats in the final SSE chunk. Without this,
        // llama-server does not emit a usage object in streaming mode.
        body["stream_options"] = serde_json::json!({ "include_usage": true });
    }

    apply_generation_params(&mut body, params);
    body
}

/// Apply shared sampling parameters to a request body (mutates in-place).
///
/// Sets both `max_tokens` (OpenAI field, honored by `/v1/chat/completions`) and
/// `n_predict` (llama-server native field, honored by `/v1/completions`).
/// Setting the unused field on each endpoint is harmless.
fn apply_generation_params(body: &mut serde_json::Value, params: &GenerationParams) {
    if let Some(n) = params.max_tokens {
        body["max_tokens"] = n.into();
        body["n_predict"] = n.into();
    }
    if let Some(t) = params.temperature {
        body["temperature"] = t.into();
    } else {
        body["temperature"] = 0.0.into(); // greedy
    }
    if let Some(p) = params.top_p {
        body["top_p"] = p.into();
    }
    if let Some(k) = params.top_k {
        body["top_k"] = k.into();
    }
    if !params.stop_seqs.is_empty() {
        body["stop"] = params.stop_seqs.clone().into();
    }
    if let Some(s) = params.seed {
        body["seed"] = s.into();
    }
    if let Some(r) = params.repeat_penalty {
        body["repeat_penalty"] = r.into();
    }
    if let Some(f) = params.frequency_penalty {
        body["frequency_penalty"] = f.into();
    }
    if let Some(p) = params.presence_penalty {
        body["presence_penalty"] = p.into();
    }
    // Grammar: "__json__" is ax-serving's sentinel for json_object mode;
    // translate it to llama-server's response_format field.
    if let Some(ref g) = params.grammar {
        if g == "__json__" {
            body["response_format"] = serde_json::json!({"type": "json_object"});
        } else {
            body["grammar"] = g.clone().into();
        }
    }
    // Mirostat sampling (llama-server native fields).
    if let Some(m) = params.mirostat {
        body["mirostat"] = (m as u64).into();
    }
    if let Some(tau) = params.mirostat_tau {
        body["mirostat_tau"] = tau.into();
    }
    if let Some(eta) = params.mirostat_eta {
        body["mirostat_eta"] = eta.into();
    }
    // Log probabilities (OpenAI spec — llama-server ≥ b3800 supports these).
    if params.logprobs.unwrap_or(false) {
        body["logprobs"] = true.into();
        if let Some(n) = params.top_logprobs
            && n > 0
        {
            body["top_logprobs"] = (n as u64).into();
        }
    }
    // Tool calling — pass tool definitions and choice verbatim to llama-server.
    if let Some(ref tools) = params.tools {
        body["tools"] = tools.clone();
    }
    if let Some(ref tool_choice) = params.tool_choice {
        body["tool_choice"] = tool_choice.clone();
    }
}

/// POST to `/v1/completions` (streaming SSE) and forward tokens via `tx`.
///
/// Parse a logprob entry from an OpenAI `logprobs.content[0]` JSON object.
///
/// Returns `Some((logprob, top_logprobs))` when the entry is present and valid;
/// `None` if the field is absent or not a JSON object.
fn parse_logprob_entry(entry: &serde_json::Value) -> Option<(f32, Vec<(String, f32)>)> {
    let logprob = entry["logprob"].as_f64()? as f32;
    let top = entry["top_logprobs"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let tok = t["token"].as_str()?.to_string();
                    let lp = t["logprob"].as_f64()? as f32;
                    Some((tok, lp))
                })
                .collect()
        })
        .unwrap_or_default();
    Some((logprob, top))
}

fn normalize_pooling_type(s: &str) -> Option<String> {
    let canonical = s.trim().to_ascii_lowercase();
    if matches!(
        canonical.as_str(),
        "none" | "mean" | "cls" | "last" | "rank"
    ) {
        Some(canonical)
    } else {
        None
    }
}

type TokenLogprob = (f32, Vec<(String, f32)>);
type StreamToken = (String, Option<TokenLogprob>);

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChunkLogprobTop {
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    logprob: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChunkLogprobEntry {
    #[serde(default)]
    logprob: Option<f64>,
    #[serde(default)]
    top_logprobs: Vec<ChunkLogprobTop>,
}

#[derive(Debug, Deserialize)]
struct ChunkLogprobs {
    #[serde(default)]
    content: Vec<ChunkLogprobEntry>,
}

#[derive(Debug, Deserialize)]
struct CompletionChoiceChunk {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<ChunkLogprobs>,
}

#[derive(Debug, Deserialize)]
struct CompletionSseChunk {
    #[serde(default)]
    content: String,
    #[serde(default)]
    stop: Option<bool>,
    #[serde(default)]
    choices: Vec<CompletionChoiceChunk>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ToolCallFnDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ToolCallFnDelta>,
}

#[derive(Debug, Deserialize)]
struct ChatDeltaChunk {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceChunk {
    #[serde(default)]
    delta: Option<ChatDeltaChunk>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<ChunkLogprobs>,
}

#[derive(Debug, Deserialize)]
struct ChatSseChunk {
    #[serde(default)]
    choices: Vec<ChatChoiceChunk>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

fn parse_logprob_entry_typed(entry: &ChunkLogprobEntry) -> Option<TokenLogprob> {
    let logprob = entry.logprob? as f32;
    let top = entry
        .top_logprobs
        .iter()
        .filter_map(|t| Some((t.token.clone()?, t.logprob? as f32)))
        .collect::<Vec<_>>();
    Some((logprob, top))
}

fn parse_nonstream_logprobs(val: &serde_json::Value) -> Vec<(String, TokenLogprob)> {
    val["choices"][0]["logprobs"]["content"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|e| {
                    let tok = e["token"].as_str()?.to_string();
                    let (lp, top) = parse_logprob_entry(e)?;
                    Some((tok, (lp, top)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// POST to a llama-server endpoint and return the response, failing on non-success status.
fn post_llama(
    http: &reqwest::blocking::Client,
    port: u16,
    endpoint: &str,
    body: &serde_json::Value,
) -> Result<reqwest::blocking::Response> {
    let url = format!("http://{LLAMACPP_LOCAL_HOST}:{port}{endpoint}");
    let resp = http
        .post(&url)
        .json(body)
        .send()
        .with_context(|| format!("POST {endpoint}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        anyhow::bail!("llama-server {endpoint} error {status}: {text}");
    }
    Ok(resp)
}

fn complete_completions(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    emit_logprobs: bool,
) -> Result<()> {
    let resp = post_llama(http, port, "/v1/completions", body)?;

    let val: serde_json::Value = resp.json().context("parse /v1/completions JSON")?;
    let text = val["content"]
        .as_str()
        .or_else(|| val["choices"][0]["text"].as_str())
        .unwrap_or("");

    if emit_logprobs {
        let logprobs = parse_nonstream_logprobs(&val);
        if logprobs.is_empty() && !text.is_empty() {
            let _ = tx.blocking_send(GenerateEvent::Token(text.to_string()));
        } else {
            for (tok, (lp, top)) in logprobs {
                if tx.blocking_send(GenerateEvent::Token(tok)).is_err() {
                    return Ok(());
                }
                if tx
                    .blocking_send(GenerateEvent::TokenLogprob { logprob: lp, top })
                    .is_err()
                {
                    return Ok(());
                }
            }
        }
    } else if !text.is_empty() {
        let _ = tx.blocking_send(GenerateEvent::Token(text.to_string()));
    }

    let prompt_tokens = val["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
    let completion_tokens = val["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
    let stop_reason = val["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or("stop")
        .to_string();
    let _ = tx.blocking_send(GenerateEvent::Done(GenerationStats {
        prompt_tokens,
        completion_tokens,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec: 0.0,
        stop_reason,
    }));
    Ok(())
}

fn complete_chat_completions(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    emit_logprobs: bool,
) -> Result<()> {
    let resp = post_llama(http, port, "/v1/chat/completions", body)?;

    let val: serde_json::Value = resp.json().context("parse /v1/chat/completions JSON")?;
    let text = val["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");

    if emit_logprobs {
        let logprobs = parse_nonstream_logprobs(&val);
        if logprobs.is_empty() && !text.is_empty() {
            let _ = tx.blocking_send(GenerateEvent::Token(text.to_string()));
        } else {
            for (tok, (lp, top)) in logprobs {
                if tx.blocking_send(GenerateEvent::Token(tok)).is_err() {
                    return Ok(());
                }
                if tx
                    .blocking_send(GenerateEvent::TokenLogprob { logprob: lp, top })
                    .is_err()
                {
                    return Ok(());
                }
            }
        }
    } else if !text.is_empty() {
        let _ = tx.blocking_send(GenerateEvent::Token(text.to_string()));
    }

    if let Some(tool_calls) = val["choices"][0]["message"]["tool_calls"].as_array() {
        for tc in tool_calls {
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            if name.is_empty() {
                continue;
            }
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let call_id = if id.is_empty() {
                format!("call_{}", uuid_simple())
            } else {
                id
            };
            let arguments = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let _ = tx.blocking_send(GenerateEvent::ToolCall {
                id: call_id,
                name,
                arguments,
            });
        }
    }

    let prompt_tokens = val["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
    let completion_tokens = val["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
    let stop_reason = val["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or("stop")
        .to_string();
    let _ = tx.blocking_send(GenerateEvent::Done(GenerationStats {
        prompt_tokens,
        completion_tokens,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec: 0.0,
        stop_reason,
    }));
    Ok(())
}

/// Tokens are buffered into batches of `batch_size` before each `blocking_send`
/// to reduce cross-thread wake-up overhead.
fn stream_completions(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    batch_size: usize,
    emit_logprobs: bool,
) -> Result<()> {
    let resp = post_llama(http, port, "/v1/completions", body)?;

    let mut reader = std::io::BufReader::new(resp);
    let mut line = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    // Each entry: (token_text, optional_logprob_data).
    // When emit_logprobs=true, effective_batch=1 so entries are always single-element.
    let mut token_buf: Vec<StreamToken> = Vec::new();
    let effective_batch = if emit_logprobs { 1 } else { batch_size };
    let mut stop_reason = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).context("reading SSE stream")?;
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

        let Ok(val) = serde_json::from_str::<CompletionSseChunk>(json_str) else {
            continue;
        };

        let first_choice = val.choices.first();
        let token_text = if !val.content.is_empty() {
            val.content.as_str()
        } else {
            first_choice
                .map(|c| c.text.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("")
        };

        let chunk_finish_reason = first_choice
            .and_then(|c| c.finish_reason.as_deref())
            .unwrap_or("");
        let stopped = val.stop.unwrap_or(false) || matches!(chunk_finish_reason, "stop" | "length");
        if stopped && !chunk_finish_reason.is_empty() {
            stop_reason = chunk_finish_reason.to_string();
        }

        if !token_text.is_empty() {
            // Parse per-token logprob data from `choices[0].logprobs.content[0]`.
            let lp_data = if emit_logprobs {
                first_choice
                    .and_then(|c| c.logprobs.as_ref())
                    .and_then(|lp| lp.content.first())
                    .and_then(parse_logprob_entry_typed)
            } else {
                None
            };
            token_buf.push((token_text.to_string(), lp_data));
        }

        if stopped {
            // /v1/completions returns usage in OpenAI format under the `usage`
            // object, not in the native `tokens_evaluated`/`tokens_predicted`
            // fields (those only appear on the native /completion endpoint).
            if let Some(n) = val.usage.as_ref().and_then(|u| u.prompt_tokens) {
                prompt_tokens = n as usize;
            }
            if let Some(n) = val.usage.as_ref().and_then(|u| u.completion_tokens) {
                completion_tokens = n as usize;
            }
        }

        // Flush token buffer when batch is full or stream stopped.
        if stopped || token_buf.len() >= effective_batch {
            if !token_buf.is_empty() {
                if emit_logprobs {
                    // Send each token individually with its logprob (batch_size=1).
                    for (tok, lp_opt) in token_buf.drain(..) {
                        if tx.blocking_send(GenerateEvent::Token(tok)).is_err() {
                            return Ok(());
                        }
                        if let Some((lp, top)) = lp_opt
                            && tx
                                .blocking_send(GenerateEvent::TokenLogprob { logprob: lp, top })
                                .is_err()
                        {
                            return Ok(());
                        }
                    }
                } else {
                    let joined: String = token_buf.drain(..).map(|(s, _)| s).collect();
                    if tx.blocking_send(GenerateEvent::Token(joined)).is_err() {
                        return Ok(()); // receiver dropped
                    }
                }
            }
            if stopped {
                break;
            }
        }
    }

    // Flush any remaining buffered tokens before Done.
    if !token_buf.is_empty() {
        if emit_logprobs {
            for (tok, lp_opt) in token_buf.drain(..) {
                let _ = tx.blocking_send(GenerateEvent::Token(tok));
                if let Some((lp, top)) = lp_opt {
                    let _ = tx.blocking_send(GenerateEvent::TokenLogprob { logprob: lp, top });
                }
            }
        } else {
            let joined: String = token_buf.drain(..).map(|(s, _)| s).collect();
            let _ = tx.blocking_send(GenerateEvent::Token(joined));
        }
    }

    if stop_reason.is_empty() {
        stop_reason = "stop".to_string();
    }
    let _ = tx.blocking_send(GenerateEvent::Done(GenerationStats {
        prompt_tokens,
        completion_tokens,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec: 0.0,
        stop_reason,
    }));

    Ok(())
}

/// POST to `/v1/chat/completions` (streaming SSE) and forward tokens via `tx`.
///
/// Uses the `/v1/chat/completions` endpoint so llama-server applies the
/// model's GGUF chat template. Parses `choices[0].delta.content` for tokens.
///
/// Tokens are buffered into batches of `batch_size` before each `blocking_send`.
/// When `emit_logprobs=true`, batch size is forced to 1 to preserve the 1:1
/// Token → TokenLogprob pairing.
fn stream_chat_completions(
    http: &reqwest::blocking::Client,
    port: u16,
    body: &serde_json::Value,
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    batch_size: usize,
    emit_logprobs: bool,
) -> Result<()> {
    let resp = post_llama(http, port, "/v1/chat/completions", body)?;

    let mut reader = std::io::BufReader::new(resp);
    let mut line = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut token_buf: Vec<StreamToken> = Vec::new();
    let effective_batch = if emit_logprobs { 1 } else { batch_size };
    let mut stop_reason = String::new();
    // Accumulate tool call argument deltas per call index.
    let mut tool_call_acc: std::collections::HashMap<u64, (String, String, String)> =
        std::collections::HashMap::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).context("reading SSE stream")?;
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

        let Ok(val) = serde_json::from_str::<ChatSseChunk>(json_str) else {
            continue;
        };

        let Some(choice) = val.choices.first() else {
            continue;
        };
        let delta = choice.delta.as_ref();

        // Regular token content.
        let token_text = delta.and_then(|d| d.content.as_deref()).unwrap_or("");
        if !token_text.is_empty() {
            // Parse per-token logprob data from `choices[0].logprobs.content[0]`.
            let lp_data = if emit_logprobs {
                choice
                    .logprobs
                    .as_ref()
                    .and_then(|lp| lp.content.first())
                    .and_then(parse_logprob_entry_typed)
            } else {
                None
            };
            token_buf.push((token_text.to_string(), lp_data));
        }

        // Tool call deltas — accumulate arguments across chunks.
        if let Some(tool_calls) = delta.map(|d| &d.tool_calls) {
            for tc in tool_calls {
                let idx = tc.index.unwrap_or(0);
                let entry = tool_call_acc
                    .entry(idx)
                    .or_insert_with(|| (String::new(), String::new(), String::new()));
                if let Some(id) = tc.id.as_deref() {
                    entry.0 = id.to_owned();
                }
                if let Some(name) = tc.function.as_ref().and_then(|f| f.name.as_deref()) {
                    entry.1 = name.to_owned();
                }
                if let Some(args) = tc.function.as_ref().and_then(|f| f.arguments.as_deref()) {
                    entry.2.push_str(args);
                }
            }
        }

        let finish_reason = choice.finish_reason.as_deref().unwrap_or("");
        let stopped =
            finish_reason == "stop" || finish_reason == "length" || finish_reason == "tool_calls";
        if stopped && !finish_reason.is_empty() {
            stop_reason = finish_reason.to_string();
        }

        // Read usage from any chunk that carries it.
        if let Some(n) = val.usage.as_ref().and_then(|u| u.prompt_tokens) {
            prompt_tokens = n as usize;
        }
        if let Some(n) = val.usage.as_ref().and_then(|u| u.completion_tokens) {
            completion_tokens = n as usize;
        }

        if (stopped || token_buf.len() >= effective_batch) && !token_buf.is_empty() {
            if emit_logprobs {
                for (tok, lp_opt) in token_buf.drain(..) {
                    if tx.blocking_send(GenerateEvent::Token(tok)).is_err() {
                        return Ok(());
                    }
                    if let Some((lp, top)) = lp_opt
                        && tx
                            .blocking_send(GenerateEvent::TokenLogprob { logprob: lp, top })
                            .is_err()
                    {
                        return Ok(());
                    }
                }
            } else {
                let joined: String = token_buf.drain(..).map(|(s, _)| s).collect();
                if tx.blocking_send(GenerateEvent::Token(joined)).is_err() {
                    return Ok(());
                }
            }
        }
    }

    // Flush remaining token buffer.
    if !token_buf.is_empty() {
        if emit_logprobs {
            for (tok, lp_opt) in token_buf.drain(..) {
                let _ = tx.blocking_send(GenerateEvent::Token(tok));
                if let Some((lp, top)) = lp_opt {
                    let _ = tx.blocking_send(GenerateEvent::TokenLogprob { logprob: lp, top });
                }
            }
        } else {
            let joined: String = token_buf.drain(..).map(|(s, _)| s).collect();
            let _ = tx.blocking_send(GenerateEvent::Token(joined));
        }
    }

    // Emit accumulated tool calls.
    let mut sorted_calls: Vec<(u64, (String, String, String))> =
        tool_call_acc.into_iter().collect();
    sorted_calls.sort_unstable_by_key(|(idx, _)| *idx);
    for (_, (id, name, arguments)) in sorted_calls {
        if !name.is_empty() {
            let call_id = if id.is_empty() {
                format!("call_{}", uuid_simple())
            } else {
                id
            };
            let _ = tx.blocking_send(GenerateEvent::ToolCall {
                id: call_id,
                name,
                arguments,
            });
        }
    }

    if stop_reason.is_empty() {
        stop_reason = "stop".to_string();
    }
    let _ = tx.blocking_send(GenerateEvent::Done(GenerationStats {
        prompt_tokens,
        completion_tokens,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec: 0.0,
        stop_reason,
    }));

    Ok(())
}

/// Generate a simple unique ID without pulling in uuid (use timestamp + counter).
fn uuid_simple() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:016x}{:08x}", unix_ms_now(), n)
}

/// Build `ModelMetadata` from GGUF header data + the running server's `/props`.
fn fetch_model_meta(
    http: &reqwest::blocking::Client,
    port: u16,
    ctx_override: u32,
    load_ms: u64,
    gguf_meta: Option<GgufMeta>,
    resolved_backend: crate::BackendType,
) -> ModelMetadata {
    // Prefer the server's reported context length (honours ctx_override) over
    // the GGUF header value since llama-server may apply its own capping.
    let context_length = if ctx_override > 0 {
        ctx_override
    } else {
        // Try GET /props (llama-server ≥ b3xxx)
        let from_server = http
            .get(format!("http://{LLAMACPP_LOCAL_HOST}:{port}/props"))
            .timeout(Duration::from_secs(5))
            .send()
            .ok()
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|v| v["n_ctx"].as_u64())
            .map(|n| n as u32);

        from_server
            .or_else(|| {
                gguf_meta
                    .as_ref()
                    .map(|m| m.context_length)
                    .filter(|&n| n > 0)
            })
            .unwrap_or(4096)
    };

    match gguf_meta {
        Some(m) => ModelMetadata {
            architecture: m.architecture,
            n_layers: m.block_count,
            n_heads: m.head_count,
            n_kv_heads: m.head_count_kv,
            embedding_dim: m.embedding_length,
            vocab_size: m.vocab_size,
            context_length,
            load_time_ms: load_ms,
            peak_rss_bytes: 0,
            resolved_backend,
        },
        None => ModelMetadata {
            architecture: "gguf-llamacpp".into(),
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            embedding_dim: 0,
            vocab_size: 0,
            context_length,
            load_time_ms: load_ms,
            peak_rss_bytes: 0,
            resolved_backend,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackendType;

    #[test]
    fn effective_n_gpu_layers_honors_explicit_value() {
        let cfg = LoadConfig {
            context_length: 0,
            backend_type: BackendType::Auto,
            llama_cpp_n_gpu_layers: Some(32),
            mmproj_path: None,
            backend_hint: None,
            enable_embeddings: None,
            pooling_type: None,
        };
        assert_eq!(LlamaCppBackend::effective_n_gpu_layers(&cfg), 32);
    }

    #[test]
    fn effective_n_gpu_layers_cpu_forces_zero() {
        let cfg = LoadConfig {
            context_length: 0,
            backend_type: BackendType::Cpu,
            llama_cpp_n_gpu_layers: Some(64),
            mmproj_path: None,
            backend_hint: None,
            enable_embeddings: None,
            pooling_type: None,
        };
        assert_eq!(LlamaCppBackend::effective_n_gpu_layers(&cfg), 0);
    }

    #[test]
    fn effective_n_gpu_layers_defaults_when_unset() {
        let cfg = LoadConfig {
            context_length: 0,
            backend_type: BackendType::Auto,
            llama_cpp_n_gpu_layers: None,
            mmproj_path: None,
            backend_hint: None,
            enable_embeddings: None,
            pooling_type: None,
        };
        assert_eq!(LlamaCppBackend::effective_n_gpu_layers(&cfg), 99);
    }

    #[test]
    fn completions_body_respects_non_stream_mode() {
        let params = GenerationParams {
            stream: false,
            ..Default::default()
        };
        let body = build_completions_body(&GenerateInput::Text("hello".into()), &params, true)
            .expect("build_completions_body should handle text input");
        assert_eq!(body["stream"].as_bool(), Some(false));
    }

    #[test]
    fn chat_body_only_includes_stream_options_when_streaming() {
        let msgs = vec![crate::ChatMessage {
            role: "user".into(),
            content: serde_json::Value::String("hi".into()),
        }];
        let params = GenerationParams {
            stream: false,
            ..Default::default()
        };
        let body = build_chat_completions_body(&msgs, &params, true);
        assert_eq!(body["stream"].as_bool(), Some(false));
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn normalize_pooling_type_accepts_and_canonicalizes() {
        assert_eq!(normalize_pooling_type("MEAN"), Some("mean".to_string()));
        assert_eq!(normalize_pooling_type("  cls  "), Some("cls".to_string()));
    }

    #[test]
    fn normalize_pooling_type_rejects_invalid() {
        assert_eq!(normalize_pooling_type("average"), None);
        assert_eq!(normalize_pooling_type(""), None);
    }
}
