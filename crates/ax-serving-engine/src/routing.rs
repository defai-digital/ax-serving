//! Backend routing: dispatches inference to ax-engine (native) or llama.cpp
//! based on a YAML config file.
//!
//! **Current policy**: all models route through `llama.cpp` by default.
//! `native` remains available as an explicit override for environments that
//! require ax-engine.
//! See `config/backends.yaml` for the full routing table.
//!
//! # Config file (`backends.yaml`)
//!
//! ```yaml
//! # Default backend for families not listed below.
//! # Options: native | llama_cpp | auto
//! #   native    — use ax-engine only (fail if not supported)
//! #   llama_cpp — use llama.cpp only (requires llama-server on PATH)
//! #   auto      — try native first, fall back to llama.cpp on unsupported arch
//! default_backend: llama_cpp
//!
//! # Per-family overrides.  Keys match `general.architecture` from GGUF metadata.
//! # Prefix matching: "qwen" matches "qwen2", "qwen3", etc.
//! # Exact match takes priority over prefix match.
//! families:
//!   llama:   llama_cpp   # llama, llama2, llama3, …
//!   qwen:    llama_cpp   # qwen2, qwen3, qwen2_moe, …
//!   gemma:   llama_cpp   # gemma2, gemma3, …
//!   mistral: llama_cpp   # mistral, mistral3, …
//!   phi:     llama_cpp   # phi2, phi3, …
//! ```
//!
//! Config is loaded from `$AXS_ROUTING_CONFIG` env var path, or
//! `./backends.yaml` if the file exists, or the built-in defaults.
//!
//! # Model architecture detection (primary)
//!
//! The `general.architecture` key from the GGUF file header is read before
//! any backend is invoked.  This is authoritative and does not depend on the
//! filename.
//!
//! # Filename fallback (secondary)
//!
//! If the GGUF header cannot be read (e.g. non-GGUF file, I/O error), the
//! lowercase filename is matched against known family substrings as a best-
//! effort fallback.
//!
//! # `auto` fallback scope
//!
//! In `auto` mode, native load errors are classified before falling back.
//! Only errors that indicate an unsupported architecture or quantization
//! will trigger the llama.cpp fallback.  Infrastructure failures (OOM,
//! file corruption, permission denied) are propagated immediately.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[cfg(feature = "libllama")]
use crate::libllama::LibLlamaBackend;
use crate::{
    EmbedConfig, EmbedInput, EmbedResult, GenerateEvent, GenerateInput, GenerationParams,
    InferenceBackend, LoadConfig, ModelHandle, ModelMetadata, ThermalState,
    ax_engine::AxEngineBackend,
    llamacpp::{LlamaCppBackend, LlamaCppConfig},
};

// ── Config ────────────────────────────────────────────────────────────────────

/// Which backend to use for a given model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendChoice {
    /// Use the native ax-engine backend.
    Native,
    /// Use llama.cpp via a spawned `llama-server` subprocess.
    LlamaCpp,
    /// Use libllama C API directly (no subprocess, requires `libllama` feature).
    LibLlama,
    /// Try LibLlama first; fall back to LlamaCpp; fall back to Native.
    #[default]
    Auto,
}

/// Top-level routing configuration, loaded from `backends.yaml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// Backend used when model family is not listed in `families`.
    pub default_backend: BackendChoice,
    /// Per-family overrides.  Keys are lowercase family names.
    pub families: HashMap<String, BackendChoice>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            // Default: use llama.cpp (widest model support).
            // Override per-family in the config file.
            default_backend: BackendChoice::LlamaCpp,
            families: HashMap::new(),
        }
    }
}

impl RoutingConfig {
    /// Load from a YAML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = serde_yaml::from_str(&text)?;
        Ok(cfg)
    }

    /// Load from `$AXS_ROUTING_CONFIG` or `./backends.yaml`, silently
    /// returning defaults if neither exists or cannot be parsed.
    pub fn load_default() -> Self {
        let path = std::env::var("AXS_ROUTING_CONFIG")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("backends.yaml"));

        if path.exists() {
            match Self::from_file(&path) {
                Ok(cfg) => {
                    info!("routing config loaded from {}", path.display());
                    return cfg;
                }
                Err(e) => {
                    warn!(
                        "failed to load routing config {}: {e} — using defaults",
                        path.display()
                    );
                }
            }
        }

        Self::default()
    }

    /// Resolve the `BackendChoice` for the model at `path`.
    ///
    /// Detection order:
    /// 1. Read `general.architecture` from GGUF header (authoritative).
    /// 2. Fall back to lowercase filename substring match on I/O failure.
    ///
    /// Family lookup: exact match first, then prefix match.
    pub fn resolve(&self, path: &Path) -> BackendChoice {
        let arch = match crate::gguf_meta::read_gguf_meta(path) {
            Ok(meta) if !meta.architecture.is_empty() => {
                info!("GGUF arch='{}' for {}", meta.architecture, path.display());
                meta.architecture
            }
            Ok(_) => {
                let family = detect_family_from_filename(path);
                warn!(
                    "empty GGUF architecture for {}; using filename family '{family}'",
                    path.display()
                );
                family
            }
            Err(e) => {
                let family = detect_family_from_filename(path);
                warn!(
                    "GGUF metadata unreadable for {} ({e}); using filename family '{family}'",
                    path.display()
                );
                family
            }
        };

        // 1. Exact match.
        if let Some(choice) = self.families.get(&arch) {
            return *choice;
        }
        // 2. Normalized exact match (e.g. key "gpt_j" matches arch "gptj").
        // If multiple aliases normalize to the same key, choose deterministically
        // by lexical order of the raw config key.
        let arch_norm = normalize_family_key(&arch);
        let mut exact_matches: Vec<(&str, BackendChoice)> = self
            .families
            .iter()
            .filter_map(|(family, choice)| {
                (normalize_family_key(family) == arch_norm).then_some((family.as_str(), *choice))
            })
            .collect();
        if !exact_matches.is_empty() {
            exact_matches.sort_unstable_by(|a, b| a.0.cmp(b.0));
            return exact_matches[0].1;
        }
        // 3. Prefix match on normalized keys, choosing the longest prefix for
        // deterministic behavior when multiple keys overlap.
        // Ties on prefix length are broken by lexical order of the raw key.
        let mut best_prefix: Option<(usize, &str, BackendChoice)> = None;
        for (family, choice) in &self.families {
            let family_norm = normalize_family_key(family);
            if family_norm.is_empty() {
                continue;
            }
            if arch_norm.starts_with(&family_norm) {
                let len = family_norm.len();
                if best_prefix.is_none_or(|(best_len, best_key, _)| {
                    len > best_len || (len == best_len && family.as_str() < best_key)
                }) {
                    best_prefix = Some((len, family.as_str(), *choice));
                }
            }
        }
        if let Some((_, _, choice)) = best_prefix {
            return choice;
        }
        self.default_backend
    }
}

/// Fallback: detect model family from the lowercase filename.
fn detect_family_from_filename(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    for (needle, family) in &[
        // Keep explicit family names that may embed base-model names first.
        ("deepseek", "deepseek"),
        ("llama", "llama"),
        ("qwen", "qwen"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
        // Avoid broad "phi" substring matching (e.g. "graphite" false-positive).
        ("phi-", "phi"),
        ("phi_", "phi"),
        ("phi2", "phi"),
        ("phi3", "phi"),
        ("phi4", "phi"),
        ("glm", "glm"),
        ("starcoder", "starcoder"),
        ("minimax", "minimax"),
        ("falcon", "falcon"),
        ("mamba", "mamba"),
        ("gpt-j", "gptj"),
        ("gpt_j", "gptj"),
        ("gptj", "gptj"),
        ("gpt-neox", "gpt_neox"),
        ("gpt_neox", "gpt_neox"),
        ("gptneox", "gpt_neox"),
        ("gpt2", "gpt2"),
        ("bloom", "bloom"),
    ] {
        if name.contains(needle) {
            return (*family).to_string();
        }
    }
    "unknown".to_string()
}

fn normalize_family_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Returns `true` if the native backend error indicates an **unsupported**
/// architecture or quantization — i.e. the error is expected and the
/// `auto` fallback to llama.cpp is appropriate.
///
/// Returns `false` for infrastructure failures (OOM, file corruption,
/// permissions) where falling back would mask a real problem.
fn is_unsupported_model_error(e: &anyhow::Error) -> bool {
    // Collect the full error chain into one lowercase string for matching.
    let full: String = e
        .chain()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();

    // Positive signals: native backend cannot handle this architecture/capability.
    let unsupported = full.contains("unsupported architecture")
        || full.contains("unknown gguf architecture")
        || full.contains("unsupported model architecture")
        || full.contains("architecture not supported")
        || full.contains("unsupported quantization")
        || full.contains("quantization not supported")
        || full.contains("unsupported gguf type")
        || full.contains("unsupported ggml type")
        || (full.contains("tokenizer") && full.contains("not supported"))
        || (full.contains("embeddings") && full.contains("not supported"));

    // Negative signals: real infrastructure failures — do NOT fall back.
    let real_failure = full.contains("insufficient memory")
        || full.contains("model file not found")
        || full.contains("cannot stat model file")
        || full.contains("not valid utf-8")
        || full.contains("only .gguf models are supported");

    unsupported && !real_failure
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn try_native_load_model(
    native: &AxEngineBackend,
    path: &Path,
    config: LoadConfig,
) -> Result<(ModelHandle, ModelMetadata)> {
    match catch_unwind(AssertUnwindSafe(|| native.load_model(path, config))) {
        Ok(res) => res,
        Err(payload) => {
            let msg = panic_payload_to_string(payload);
            Err(anyhow::anyhow!(
                "native backend panicked while loading {}: {}",
                path.display(),
                msg
            ))
        }
    }
}

// ── RouterBackend ─────────────────────────────────────────────────────────────

static NEXT_ROUTER_HANDLE: AtomicU64 = AtomicU64::new(2_000_000);

fn next_router_handle() -> ModelHandle {
    ModelHandle(NEXT_ROUTER_HANDLE.fetch_add(1, Ordering::Relaxed))
}

/// Which backend owns a given outer handle.
#[derive(Clone, Copy)]
enum BackendTag {
    Native,
    LlamaCpp,
    /// Direct C API backend — only usable when `libllama` feature is compiled.
    #[cfg_attr(not(feature = "libllama"), allow(dead_code))]
    LibLlama,
}

struct RouterEntry {
    tag: BackendTag,
    inner: ModelHandle,
}

/// `InferenceBackend` that routes to ax-engine native, llama.cpp, or libllama
/// based on `RoutingConfig`.
pub struct RouterBackend {
    config: RoutingConfig,
    native: Arc<AxEngineBackend>,
    llamacpp: Arc<LlamaCppBackend>,
    #[cfg(feature = "libllama")]
    libllama: Arc<LibLlamaBackend>,
    entries: RwLock<HashMap<ModelHandle, RouterEntry>>,
}

impl RouterBackend {
    pub fn new(routing: RoutingConfig, llamacpp: LlamaCppConfig) -> Self {
        Self {
            config: routing,
            native: Arc::new(AxEngineBackend::new()),
            llamacpp: Arc::new(LlamaCppBackend::new(llamacpp)),
            #[cfg(feature = "libllama")]
            libllama: Arc::new(LibLlamaBackend::new()),
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Convenience: create with defaults loaded from environment.
    pub fn from_env() -> Self {
        Self::new(RoutingConfig::load_default(), LlamaCppConfig::from_env())
    }
}

impl InferenceBackend for RouterBackend {
    fn load_model(&self, path: &Path, config: LoadConfig) -> Result<(ModelHandle, ModelMetadata)> {
        // Per-load override takes priority over backends.yaml routing.
        let choice = if let Some(ref hint) = config.backend_hint {
            match hint.as_str() {
                "llama_cpp" => BackendChoice::LlamaCpp,
                "native" => BackendChoice::Native,
                "lib_llama" => BackendChoice::LibLlama,
                "auto" => BackendChoice::Auto,
                other => anyhow::bail!(
                    "unknown backend hint '{}'; valid values: llama_cpp, native, lib_llama, auto",
                    other
                ),
            }
        } else {
            self.config.resolve(path)
        };
        let outer = next_router_handle();

        let (tag, inner, meta) = match choice {
            BackendChoice::Native => {
                info!("routing {} → native (ax-engine)", path.display());
                let (inner, meta) = try_native_load_model(&self.native, path, config)?;
                (BackendTag::Native, inner, meta)
            }
            BackendChoice::LlamaCpp => {
                info!("routing {} → llama.cpp", path.display());
                let (inner, meta) = self.llamacpp.load_model(path, config)?;
                (BackendTag::LlamaCpp, inner, meta)
            }
            BackendChoice::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    info!("routing {} → libllama (direct C API)", path.display());
                    let (inner, meta) = self.libllama.load_model(path, config)?;
                    (BackendTag::LibLlama, inner, meta)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    warn!(
                        "libllama feature not compiled; falling back to llama_cpp for {}",
                        path.display()
                    );
                    let (inner, meta) = self.llamacpp.load_model(path, config)?;
                    (BackendTag::LlamaCpp, inner, meta)
                }
            }
            BackendChoice::Auto => {
                info!("routing {} → auto (trying native first)", path.display());
                match try_native_load_model(&self.native, path, config.clone()) {
                    Ok((inner, meta)) => {
                        info!("auto-routing {} → native (success)", path.display());
                        (BackendTag::Native, inner, meta)
                    }
                    Err(e) if is_unsupported_model_error(&e) => {
                        // Native does not support this arch/quant: expected, fall back.
                        warn!(
                            "auto-routing {} → llama.cpp (native unsupported: {e})",
                            path.display()
                        );
                        let (inner, meta) = self.llamacpp.load_model(path, config)?;
                        (BackendTag::LlamaCpp, inner, meta)
                    }
                    Err(e) => {
                        // Real failure (OOM, corruption, permissions): propagate.
                        return Err(e);
                    }
                }
            }
        };

        self.entries
            .write()
            .unwrap()
            .insert(outer, RouterEntry { tag, inner });
        Ok((outer, meta))
    }

    fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let entry = self
            .entries
            .write()
            .unwrap()
            .remove(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;

        let result = match entry.tag {
            BackendTag::Native => self.native.unload_model(entry.inner),
            BackendTag::LlamaCpp => self.llamacpp.unload_model(entry.inner),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    self.libllama.unload_model(entry.inner)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    Err(anyhow::anyhow!("libllama feature not compiled"))
                }
            }
        };
        if result.is_err() {
            // Preserve routing state so callers can retry unload rather than
            // losing the inner handle mapping permanently.
            self.entries.write().unwrap().insert(handle, entry);
        }
        result
    }

    fn generate(
        &self,
        handle: ModelHandle,
        input: GenerateInput,
        params: GenerationParams,
        tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> Result<()> {
        let guard = self.entries.read().unwrap();
        let entry = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;

        match entry.tag {
            BackendTag::Native => self.native.generate(entry.inner, input, params, tx),
            BackendTag::LlamaCpp => self.llamacpp.generate(entry.inner, input, params, tx),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    self.libllama.generate(entry.inner, input, params, tx)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    Err(anyhow::anyhow!("libllama feature not compiled"))
                }
            }
        }
    }

    fn tokenize(&self, handle: ModelHandle, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let guard = self.entries.read().unwrap();
        let entry = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;

        match entry.tag {
            BackendTag::Native => self.native.tokenize(entry.inner, text, add_bos),
            BackendTag::LlamaCpp => self.llamacpp.tokenize(entry.inner, text, add_bos),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    self.libllama.tokenize(entry.inner, text, add_bos)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    Err(anyhow::anyhow!("libllama feature not compiled"))
                }
            }
        }
    }

    fn decode_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<String> {
        let guard = self.entries.read().unwrap();
        let entry = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;

        match entry.tag {
            BackendTag::Native => self.native.decode_tokens(entry.inner, tokens),
            BackendTag::LlamaCpp => self.llamacpp.decode_tokens(entry.inner, tokens),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    self.libllama.decode_tokens(entry.inner, tokens)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    Err(anyhow::anyhow!("libllama feature not compiled"))
                }
            }
        }
    }

    fn eos_tokens(&self, handle: ModelHandle) -> Result<Vec<u32>> {
        let guard = self.entries.read().unwrap();
        let entry = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;

        match entry.tag {
            BackendTag::Native => self.native.eos_tokens(entry.inner),
            BackendTag::LlamaCpp => self.llamacpp.eos_tokens(entry.inner),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    self.libllama.eos_tokens(entry.inner)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    Err(anyhow::anyhow!("libllama feature not compiled"))
                }
            }
        }
    }

    fn eval_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<u32> {
        let guard = self.entries.read().unwrap();
        let entry = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;

        match entry.tag {
            BackendTag::Native => self.native.eval_tokens(entry.inner, tokens),
            BackendTag::LlamaCpp => self.llamacpp.eval_tokens(entry.inner, tokens),
            BackendTag::LibLlama => Err(anyhow::anyhow!(
                "eval_tokens not supported by LibLlamaBackend"
            )),
        }
    }

    fn thermal_state(&self) -> ThermalState {
        let entries = self.entries.read().unwrap();
        let mut has_native = false;
        let mut has_llamacpp = false;
        let mut has_libllama = false;
        for entry in entries.values() {
            match entry.tag {
                BackendTag::Native => has_native = true,
                BackendTag::LlamaCpp => has_llamacpp = true,
                BackendTag::LibLlama => has_libllama = true,
            }
        }
        drop(entries);

        // Collect thermal states from active backends and return the worst.
        let mut worst = self.llamacpp.thermal_state(); // default for empty router
        if has_native {
            let t = self.native.thermal_state();
            if (t as u8) > (worst as u8) {
                worst = t;
            }
        }
        if has_llamacpp {
            let t = self.llamacpp.thermal_state();
            if (t as u8) > (worst as u8) {
                worst = t;
            }
        }
        #[cfg(feature = "libllama")]
        if has_libllama {
            let t = self.libllama.thermal_state();
            if (t as u8) > (worst as u8) {
                worst = t;
            }
        }
        let _ = has_libllama;
        worst
    }

    fn recommended_concurrency(&self) -> usize {
        let entries = self.entries.read().unwrap();
        let mut has_native = false;
        let mut has_llamacpp = false;
        let mut has_libllama = false;
        for entry in entries.values() {
            match entry.tag {
                BackendTag::Native => has_native = true,
                BackendTag::LlamaCpp => has_llamacpp = true,
                BackendTag::LibLlama => has_libllama = true,
            }
        }
        drop(entries);

        // Report the minimum concurrency across active backends (conservative).
        let mut min = self.llamacpp.recommended_concurrency(); // default
        if has_native {
            min = min.min(self.native.recommended_concurrency());
        }
        if has_llamacpp {
            min = min.min(self.llamacpp.recommended_concurrency());
        }
        #[cfg(feature = "libllama")]
        if has_libllama {
            min = min.min(self.libllama.recommended_concurrency());
        }
        let _ = has_libllama;
        min
    }

    fn embed(
        &self,
        handle: ModelHandle,
        inputs: &EmbedInput<'_>,
        config: &EmbedConfig,
    ) -> Result<EmbedResult> {
        let guard = self.entries.read().unwrap();
        let entry = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;

        match entry.tag {
            BackendTag::Native => self.native.embed(entry.inner, inputs, config),
            BackendTag::LlamaCpp => self.llamacpp.embed(entry.inner, inputs, config),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    self.libllama.embed(entry.inner, inputs, config)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    Err(anyhow::anyhow!("libllama feature not compiled"))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(families: &[(&str, BackendChoice)], default: BackendChoice) -> RoutingConfig {
        RoutingConfig {
            default_backend: default,
            families: families.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    // Filename detection returns the correct family name for known substrings.
    #[test]
    fn test_filename_fallback() {
        assert_eq!(
            detect_family_from_filename(Path::new("llama3-8b-Q4_K_M.gguf")),
            "llama"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("Qwen2-7B-Q4_K_M.gguf")),
            "qwen"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("gemma-3-4b-it-Q4_K_M.gguf")),
            "gemma"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("Phi-3-mini-4k-instruct-Q4_K_M.gguf")),
            "phi"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("mistral-7b-v0.1.gguf")),
            "mistral"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("falcon-7b-Q4_K_M.gguf")),
            "falcon"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("DeepSeek-R1-Distill-7B-Q4_K_M.gguf")),
            "deepseek"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")),
            "deepseek"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("GLM-4-9B-Q4_K_M.gguf")),
            "glm"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("starcoder2-7b-Q4_K_M.gguf")),
            "starcoder"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("MiniMax-Text-01-Q4_K_M.gguf")),
            "minimax"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("gpt-j-6b-Q4_K_M.gguf")),
            "gptj"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("gpt-neox-20b-Q4_K_M.gguf")),
            "gpt_neox"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("graphite-7b-Q4_K_M.gguf")),
            "unknown"
        );
        assert_eq!(
            detect_family_from_filename(Path::new("completely-unknown-model.gguf")),
            "unknown"
        );
    }

    // resolve() uses filename fallback on I/O error and matches exact family key.
    #[test]
    fn test_resolve_exact_match() {
        let cfg = make_config(&[("llama", BackendChoice::Native)], BackendChoice::LlamaCpp);
        // GGUF read fails for non-existent path → filename detection returns "llama"
        // → exact match in families → Native.
        let choice = cfg.resolve(Path::new("llama3-8b-Q4_K_M.gguf"));
        assert_eq!(choice, BackendChoice::Native);
    }

    // resolve() uses prefix match: key "llam" matches arch/family "llama".
    #[test]
    fn test_resolve_prefix_match() {
        // No exact key "llama"; key "llam" is a prefix of "llama".
        let cfg = make_config(&[("llam", BackendChoice::Native)], BackendChoice::LlamaCpp);
        // Filename detection: "llama3-8b.gguf" → "llama"
        // Exact match on "llam": fails.
        // Prefix match: "llama".starts_with("llam") → true → Native.
        let choice = cfg.resolve(Path::new("llama3-8b.gguf"));
        assert_eq!(choice, BackendChoice::Native);
    }

    // Overlapping prefixes must resolve deterministically to the longest match.
    #[test]
    fn test_resolve_longest_prefix_wins() {
        let cfg = make_config(
            &[
                ("q", BackendChoice::LlamaCpp),
                ("qwen", BackendChoice::Native),
            ],
            BackendChoice::LlamaCpp,
        );
        let choice = cfg.resolve(Path::new("qwen3-8b.gguf"));
        assert_eq!(choice, BackendChoice::Native);
    }

    // Normalized exact match supports alias separators (e.g. gpt_j vs gptj).
    #[test]
    fn test_resolve_normalized_exact_match() {
        let cfg = make_config(&[("gpt_j", BackendChoice::Native)], BackendChoice::LlamaCpp);
        let choice = cfg.resolve(Path::new("gptj-6b-Q4_K_M.gguf"));
        assert_eq!(choice, BackendChoice::Native);
    }

    // Multiple alias keys that normalize to the same family must resolve
    // deterministically, regardless of HashMap iteration order.
    #[test]
    fn test_resolve_normalized_exact_alias_tie_deterministic() {
        let cfg = make_config(
            &[
                ("gpt_j", BackendChoice::Native),
                ("gpt-j", BackendChoice::LlamaCpp),
            ],
            BackendChoice::Auto,
        );
        let choice = cfg.resolve(Path::new("gptj-6b-Q4_K_M.gguf"));
        // Lexical tie-break on raw key: "gpt-j" < "gpt_j".
        assert_eq!(choice, BackendChoice::LlamaCpp);
    }

    // Prefix ties at equal normalized length must also resolve deterministically.
    #[test]
    fn test_resolve_prefix_alias_tie_deterministic() {
        let cfg = make_config(
            &[
                ("qwen_", BackendChoice::Native),
                ("qwen-", BackendChoice::LlamaCpp),
            ],
            BackendChoice::Auto,
        );
        let choice = cfg.resolve(Path::new("qwen3-8b.gguf"));
        // Both normalize to "qwen"; lexical tie-break on raw key.
        assert_eq!(choice, BackendChoice::LlamaCpp);
    }

    // Unrecognised family falls back to default_backend.
    #[test]
    fn test_resolve_default_fallback() {
        let cfg = make_config(&[("llama", BackendChoice::Native)], BackendChoice::LlamaCpp);
        // "completely-unknown-model.gguf" → detect returns "unknown"
        // → no match → default LlamaCpp.
        let choice = cfg.resolve(Path::new("completely-unknown-model.gguf"));
        assert_eq!(choice, BackendChoice::LlamaCpp);
    }

    // is_unsupported_model_error: "unsupported architecture" → true,
    // infrastructure failures → false.
    #[test]
    fn test_is_unsupported_model_error() {
        let e_unsupported = anyhow::anyhow!("unsupported architecture: xyz");
        assert!(is_unsupported_model_error(&e_unsupported));

        let e_arch_variant = anyhow::anyhow!("architecture not supported: gptj");
        assert!(is_unsupported_model_error(&e_arch_variant));

        let e_quant = anyhow::anyhow!("unsupported quantization: q3_k");
        assert!(is_unsupported_model_error(&e_quant));

        let e_ggml_type = anyhow::anyhow!("unsupported ggml type 37 in tensor");
        assert!(is_unsupported_model_error(&e_ggml_type));

        let e_tokenizer = anyhow::anyhow!("tokenizer not supported for this model");
        assert!(is_unsupported_model_error(&e_tokenizer));

        let e_oom = anyhow::anyhow!("insufficient memory: need 32 GB, have 16 GB");
        assert!(!is_unsupported_model_error(&e_oom));

        let e_not_found = anyhow::anyhow!("model file not found: /models/x.gguf");
        assert!(!is_unsupported_model_error(&e_not_found));

        let e_random = anyhow::anyhow!("some unrelated error");
        assert!(!is_unsupported_model_error(&e_random));
    }

    // RoutingConfig round-trips through YAML serialization.
    #[test]
    fn test_routing_config_serde() {
        let yaml = r#"
default_backend: auto
families:
  llama: native
  gptj: llama_cpp
  qwen: native
"#;
        let cfg: RoutingConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.default_backend, BackendChoice::Auto);
        assert_eq!(cfg.families["llama"], BackendChoice::Native);
        assert_eq!(cfg.families["gptj"], BackendChoice::LlamaCpp);
        assert_eq!(cfg.families["qwen"], BackendChoice::Native);

        // Serialize and re-parse.
        let yaml2 = serde_yaml::to_string(&cfg).unwrap();
        let cfg2: RoutingConfig = serde_yaml::from_str(&yaml2).unwrap();
        assert_eq!(cfg2.default_backend, BackendChoice::Auto);
        assert_eq!(cfg2.families.len(), 3);
    }
}
