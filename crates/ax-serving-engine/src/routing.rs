//! Backend routing: dispatches inference to ax-engine (native) or llama.cpp.
//!
//! **Routing model**: ax-serving owns all routing decisions.  ax-engine is
//! never asked to load a model and then fall back on failure.  Instead,
//! `RoutingConfig` maintains a `native_families` allowlist of architectures
//! known to be supported by the current ax-engine version.  Any model whose
//! architecture is not in that list is sent directly to llama.cpp without
//! attempting a native load.
//!
//! # Config file (`backends.yaml`)
//!
//! ```yaml
//! # Default routing: consult native_families (see below).
//! # Options: native | llama_cpp | auto
//! #   native    — ax-engine only (fail if not supported)
//! #   llama_cpp — llama.cpp only (requires llama-server on PATH)
//! #   auto      — check native_families; route native if matched, else llama.cpp
//! default_backend: auto
//!
//! # Architectures ax-engine natively supports (prefix matching).
//! # Used only when routing to `auto`.  Override to add/remove families.
//! native_families:
//!   - llama
//!   - mistral
//!   - mixtral
//!   - qwen35
//!   - gemma3
//!   - gemma4
//!
//! # Per-family overrides.  Keys match `general.architecture` from GGUF metadata.
//! # Prefix matching: "qwen" matches "qwen35", etc.
//! # Exact match takes priority over prefix match.
//! # An explicit entry here overrides native_families for that family.
//! families:
//!   phi:     llama_cpp   # phi2, phi3, …
//!   glm:     llama_cpp
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

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{
    Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[cfg(feature = "libllama")]
use crate::libllama::LibLlamaBackend;
use crate::{
    CacheTelemetry, EmbedConfig, EmbedInput, EmbedResult, GenerateEvent, GenerateInput,
    GenerationParams, InferenceBackend, LoadConfig, ModelHandle, ModelMetadata, ThermalState,
    ax_engine::AxEngineBackend,
    llamacpp::{LlamaCppBackend, LlamaCppConfig},
    mlx::{MlxBackend, MlxConfig, is_mlx_model},
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
    /// Consult `RoutingConfig::native_families`: if the model's architecture
    /// prefix-matches an entry, route to native ax-engine; otherwise llama.cpp.
    /// No speculative probing — the decision is made before any load attempt.
    #[default]
    Auto,
    /// Use the MLX backend (`mlx_lm.server`).
    /// Only valid via `backend_hint = "mlx"` in `LoadConfig`.
    /// If the model path is not an MLX directory, falls back to llama.cpp.
    Mlx,
}

/// Top-level routing configuration, loaded from `backends.yaml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// Backend used when model family is not listed in `families`.
    pub default_backend: BackendChoice,
    /// Per-family overrides.  Keys are lowercase family names.
    pub families: HashMap<String, BackendChoice>,
    /// Architectures natively supported by ax-engine (used by `Auto` routing).
    /// Prefix matching applies — `"llama"` matches `"llama3"`, `"llama2"`, etc.
    /// An explicit `families` entry for the same arch takes priority.
    #[serde(default = "RoutingConfig::default_native_families")]
    pub native_families: Vec<String>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            // Default: auto — deterministic routing via native_families list.
            default_backend: BackendChoice::Auto,
            families: HashMap::new(),
            native_families: Self::default_native_families(),
        }
    }
}

impl RoutingConfig {
    /// Built-in ax-engine v3.x native architecture support list.
    /// qwen35 = Qwen 3.5 (hybrid attention+SSM); qwen2/qwen3 dense removed in v3.x.
    /// gemma3/gemma4 supported; gemma/gemma2 removed in v3.x.
    pub fn default_native_families() -> Vec<String> {
        ["llama", "mistral", "mixtral", "qwen35", "gemma3", "gemma4"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    /// Expand `Auto` to `Native` or `LlamaCpp` by checking `native_families`.
    /// Uses prefix matching on normalized (alphanumeric-only) arch strings.
    fn resolve_auto(&self, arch: &str) -> BackendChoice {
        let arch_norm = normalize_family_key(arch);
        for family in &self.native_families {
            let family_norm = normalize_family_key(family);
            if !family_norm.is_empty()
                && (arch_norm == family_norm || arch_norm.starts_with(&family_norm))
            {
                return BackendChoice::Native;
            }
        }
        BackendChoice::LlamaCpp
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
        if let Some(&choice) = self.families.get(&arch) {
            // Expand Auto here too — resolve() must never return Auto.
            return if matches!(choice, BackendChoice::Auto) {
                self.resolve_auto(&arch)
            } else {
                choice
            };
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
            let choice = exact_matches[0].1;
            // Expand Auto — resolve() must never return Auto.
            return if matches!(choice, BackendChoice::Auto) {
                self.resolve_auto(&arch)
            } else {
                choice
            };
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
        let choice = if let Some((_, _, choice)) = best_prefix {
            choice
        } else {
            self.default_backend
        };

        // Expand Auto → Native or LlamaCpp using the native_families list.
        // resolve() never returns Auto so callers always get a concrete backend.
        if matches!(choice, BackendChoice::Auto) {
            return self.resolve_auto(&arch);
        }
        choice
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
        // More-specific entries must come before their generic prefixes so that
        // e.g. "gemma3-4b.gguf" maps to "gemma3", not the broader "gemma".
        ("deepseek", "deepseek"),
        ("gpt2", "gpt2"),
        ("llama", "llama"),
        ("qwen35", "qwen35"),
        ("qwen3", "qwen3"),
        ("qwen", "qwen"),
        ("gemma4", "gemma4"),
        ("gemma-4", "gemma4"),
        ("gemma3", "gemma3"),
        ("gemma-3", "gemma3"),
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
    Mlx,
    /// Direct C API backend — only usable when `libllama` feature is compiled.
    #[cfg_attr(not(feature = "libllama"), allow(dead_code))]
    LibLlama,
}

struct RouterEntry {
    tag: BackendTag,
    inner: ModelHandle,
}

/// `InferenceBackend` that routes to ax-engine native, llama.cpp, mlx-lm, or
/// libllama based on `RoutingConfig` and per-load `backend_hint`.
pub struct RouterBackend {
    config: RoutingConfig,
    native: Arc<AxEngineBackend>,
    llamacpp: Arc<LlamaCppBackend>,
    mlx: Arc<MlxBackend>,
    #[cfg(feature = "libllama")]
    libllama: Arc<LibLlamaBackend>,
    entries: RwLock<HashMap<ModelHandle, RouterEntry>>,
}

impl RouterBackend {
    fn entries_read(
        &self,
    ) -> RwLockReadGuard<'_, std::collections::HashMap<ModelHandle, RouterEntry>> {
        self.entries.read().unwrap_or_else(|err| {
            warn!("router entries rwlock poisoned; recovering from poisoned read lock");
            err.into_inner()
        })
    }

    fn entries_write(
        &self,
    ) -> RwLockWriteGuard<'_, std::collections::HashMap<ModelHandle, RouterEntry>> {
        self.entries.write().unwrap_or_else(|err| {
            warn!("router entries rwlock poisoned; recovering from poisoned write lock");
            err.into_inner()
        })
    }

    pub fn new(routing: RoutingConfig, llamacpp: LlamaCppConfig, mlx: MlxConfig) -> Self {
        Self {
            config: routing,
            native: Arc::new(AxEngineBackend::new()),
            llamacpp: Arc::new(LlamaCppBackend::new(llamacpp)),
            mlx: Arc::new(MlxBackend::new(mlx)),
            #[cfg(feature = "libllama")]
            libllama: Arc::new(LibLlamaBackend::new()),
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Look up a router entry by handle, returning the backend tag and inner handle.
    fn resolve_entry(&self, handle: ModelHandle) -> Result<(BackendTag, ModelHandle)> {
        let guard = self.entries_read();
        let entry = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown router handle {:?}", handle))?;
        Ok((entry.tag, entry.inner))
    }

    /// Scan loaded entries and return which backend kinds have active models.
    fn active_backend_tags(&self) -> (bool, bool, bool, bool) {
        let entries = self.entries_read();
        let mut has_native = false;
        let mut has_llamacpp = false;
        let mut has_mlx = false;
        let mut has_libllama = false;
        for entry in entries.values() {
            match entry.tag {
                BackendTag::Native => has_native = true,
                BackendTag::LlamaCpp => has_llamacpp = true,
                BackendTag::Mlx => has_mlx = true,
                BackendTag::LibLlama => has_libllama = true,
            }
        }
        (has_native, has_llamacpp, has_mlx, has_libllama)
    }

    /// Convenience: create with defaults loaded from environment.
    pub fn from_env() -> Self {
        Self::new(
            RoutingConfig::load_default(),
            LlamaCppConfig::from_env(),
            MlxConfig::from_env(),
        )
    }
}

/// Dispatch a method call to the correct backend based on tag.
macro_rules! dispatch {
    ($self:expr, $tag:expr, $inner:expr, $method:ident $(, $arg:expr)*) => {
        match $tag {
            BackendTag::Native => $self.native.$method($inner, $($arg),*),
            BackendTag::LlamaCpp => $self.llamacpp.$method($inner, $($arg),*),
            BackendTag::Mlx => $self.mlx.$method($inner, $($arg),*),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                { $self.libllama.$method($inner, $($arg),*) }
                #[cfg(not(feature = "libllama"))]
                { Err(anyhow::anyhow!("libllama feature not compiled")) }
            }
        }
    };
}

impl InferenceBackend for RouterBackend {
    fn load_model(&self, path: &Path, config: LoadConfig) -> Result<(ModelHandle, ModelMetadata)> {
        // Per-load override takes priority over backends.yaml routing.
        let choice = if let Some(ref hint) = config.backend_hint {
            match hint.as_str() {
                "llama_cpp" => BackendChoice::LlamaCpp,
                "native" => BackendChoice::Native,
                "lib_llama" => BackendChoice::LibLlama,
                // "auto" hint: use standard routing table (same as no hint).
                // resolve() expands Auto via native_families, never returns Auto.
                "auto" => self.config.resolve(path),
                // "mlx" hint: detect MLX model directory; fall back to llama.cpp.
                "mlx" => {
                    if is_mlx_model(path) {
                        BackendChoice::Mlx
                    } else {
                        warn!(
                            "mlx backend requested but {} is not an MLX model directory \
                             (needs config.json + *.safetensors); routing to llama.cpp",
                            path.display()
                        );
                        BackendChoice::LlamaCpp
                    }
                }
                other => anyhow::bail!(
                    "unknown backend hint '{}'; valid values: llama_cpp, native, mlx, lib_llama, auto",
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
            BackendChoice::Mlx => {
                info!("routing {} → mlx-lm", path.display());
                let (inner, meta) = self.mlx.load_model(path, config)?;
                (BackendTag::Mlx, inner, meta)
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
                // resolve() always expands Auto before returning; this arm is
                // unreachable in normal operation.
                unreachable!(
                    "BackendChoice::Auto reached RouterBackend::load_model — \
                     RoutingConfig::resolve() must expand Auto before returning"
                );
            }
        };

        self.entries_write()
            .insert(outer, RouterEntry { tag, inner });
        Ok((outer, meta))
    }

    fn backend_name_for_handle(&self, handle: ModelHandle) -> Option<&'static str> {
        self.entries_read()
            .get(&handle)
            .map(|entry| match entry.tag {
                BackendTag::Native => "native",
                BackendTag::LlamaCpp => "llama_cpp",
                BackendTag::Mlx => "mlx",
                BackendTag::LibLlama => "lib_llama",
            })
    }

    fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        // BUG-061: peek without removing first so concurrent generate/tokenize
        // calls don't see "unknown router handle" while unload is in progress.
        // The entry is only removed after a successful backend unload.
        let (tag, inner) = self.resolve_entry(handle)?;

        let result = match tag {
            BackendTag::Native => self.native.unload_model(inner),
            BackendTag::LlamaCpp => self.llamacpp.unload_model(inner),
            BackendTag::Mlx => self.mlx.unload_model(inner),
            BackendTag::LibLlama => {
                #[cfg(feature = "libllama")]
                {
                    self.libllama.unload_model(inner)
                }
                #[cfg(not(feature = "libllama"))]
                {
                    Err(anyhow::anyhow!("libllama feature not compiled"))
                }
            }
        };
        if result.is_ok() {
            self.entries_write().remove(&handle);
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
        let (tag, inner) = self.resolve_entry(handle)?;
        dispatch!(self, tag, inner, generate, input, params, tx)
    }

    fn tokenize(&self, handle: ModelHandle, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let (tag, inner) = self.resolve_entry(handle)?;
        dispatch!(self, tag, inner, tokenize, text, add_bos)
    }

    fn decode_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<String> {
        let (tag, inner) = self.resolve_entry(handle)?;
        dispatch!(self, tag, inner, decode_tokens, tokens)
    }

    fn eos_tokens(&self, handle: ModelHandle) -> Result<Vec<u32>> {
        let (tag, inner) = self.resolve_entry(handle)?;
        dispatch!(self, tag, inner, eos_tokens)
    }

    fn bos_token(&self, handle: ModelHandle) -> Result<u32> {
        let (tag, inner) = self.resolve_entry(handle)?;
        dispatch!(self, tag, inner, bos_token)
    }

    fn eval_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<u32> {
        let (tag, inner) = self.resolve_entry(handle)?;
        match tag {
            BackendTag::Native => self.native.eval_tokens(inner, tokens),
            BackendTag::LlamaCpp => self.llamacpp.eval_tokens(inner, tokens),
            BackendTag::Mlx => Err(anyhow::anyhow!("eval_tokens not supported by MlxBackend")),
            BackendTag::LibLlama => Err(anyhow::anyhow!(
                "eval_tokens not supported by LibLlamaBackend"
            )),
        }
    }

    fn thermal_state(&self) -> ThermalState {
        let (has_native, has_llamacpp, has_mlx, has_libllama) = self.active_backend_tags();

        let mut worst = self.llamacpp.thermal_state(); // default for empty router
        let mut update = |t: ThermalState| {
            if (t as u8) > (worst as u8) {
                worst = t;
            }
        };
        if has_native {
            update(self.native.thermal_state());
        }
        if has_llamacpp {
            update(self.llamacpp.thermal_state());
        }
        if has_mlx {
            update(self.mlx.thermal_state());
        }
        #[cfg(feature = "libllama")]
        if has_libllama {
            update(self.libllama.thermal_state());
        }
        let _ = has_libllama;
        worst
    }

    fn recommended_concurrency(&self) -> usize {
        let (has_native, has_llamacpp, has_mlx, has_libllama) = self.active_backend_tags();

        let mut min = self.llamacpp.recommended_concurrency(); // default
        if has_native {
            min = min.min(self.native.recommended_concurrency());
        }
        if has_llamacpp {
            min = min.min(self.llamacpp.recommended_concurrency());
        }
        if has_mlx {
            min = min.min(self.mlx.recommended_concurrency());
        }
        #[cfg(feature = "libllama")]
        if has_libllama {
            min = min.min(self.libllama.recommended_concurrency());
        }
        let _ = has_libllama;
        min
    }

    fn cache_telemetry(&self) -> CacheTelemetry {
        let (has_native, has_llamacpp, has_mlx, has_libllama) = self.active_backend_tags();

        let mut agg = CacheTelemetry::default();
        let mut merge = |t: CacheTelemetry| {
            agg.kv_pages_used += t.kv_pages_used;
            agg.kv_pages_total += t.kv_pages_total;
            agg.prefix_reusable_tokens += t.prefix_reusable_tokens;
            agg.active_batch_size += t.active_batch_size;
            agg.max_batch_size += t.max_batch_size;
        };
        if has_native {
            merge(self.native.cache_telemetry());
        }
        if has_llamacpp {
            merge(self.llamacpp.cache_telemetry());
        }
        if has_mlx {
            merge(self.mlx.cache_telemetry());
        }
        #[cfg(feature = "libllama")]
        if has_libllama {
            merge(self.libllama.cache_telemetry());
        }
        let _ = has_libllama;
        agg
    }

    fn embed(
        &self,
        handle: ModelHandle,
        inputs: &EmbedInput<'_>,
        config: &EmbedConfig,
    ) -> Result<EmbedResult> {
        let (tag, inner) = self.resolve_entry(handle)?;
        dispatch!(self, tag, inner, embed, inputs, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(families: &[(&str, BackendChoice)], default: BackendChoice) -> RoutingConfig {
        RoutingConfig {
            default_backend: default,
            families: families.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            // Empty native_families: Auto with no list always resolves to LlamaCpp.
            native_families: vec![],
        }
    }

    fn make_config_with_native(
        families: &[(&str, BackendChoice)],
        default: BackendChoice,
        native: &[&str],
    ) -> RoutingConfig {
        RoutingConfig {
            default_backend: default,
            families: families.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            native_families: native.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_backend_name_for_handle() {
        let backend = RouterBackend::new(
            RoutingConfig::default(),
            LlamaCppConfig::from_env(),
            crate::mlx::MlxConfig::from_env(),
        );
        {
            let mut entries = backend.entries.write().unwrap();
            entries.insert(
                ModelHandle(10_001),
                RouterEntry {
                    tag: BackendTag::Native,
                    inner: ModelHandle(1),
                },
            );
            entries.insert(
                ModelHandle(10_002),
                RouterEntry {
                    tag: BackendTag::LlamaCpp,
                    inner: ModelHandle(2),
                },
            );
            #[cfg(feature = "libllama")]
            entries.insert(
                ModelHandle(10_003),
                RouterEntry {
                    tag: BackendTag::LibLlama,
                    inner: ModelHandle(3),
                },
            );
        }

        assert_eq!(
            backend.backend_name_for_handle(ModelHandle(10_001)),
            Some("native")
        );
        assert_eq!(
            backend.backend_name_for_handle(ModelHandle(10_002)),
            Some("llama_cpp")
        );
        #[cfg(feature = "libllama")]
        assert_eq!(
            backend.backend_name_for_handle(ModelHandle(10_003)),
            Some("lib_llama")
        );
        assert!(
            backend
                .backend_name_for_handle(ModelHandle(10_099))
                .is_none()
        );
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
            "gemma3"
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
            detect_family_from_filename(Path::new("gpt2-llama-compatible.gguf")),
            "gpt2"
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

    // Auto routing: arch in native_families → Native; unknown → LlamaCpp.
    #[test]
    fn test_auto_routes_native_arch_to_native() {
        let cfg = make_config_with_native(&[], BackendChoice::Auto, &["llama", "gemma3", "gemma4"]);
        // Filename detection: "llama3-8b.gguf" → family "llama" → in native_families → Native.
        assert_eq!(
            cfg.resolve(Path::new("llama3-8b.gguf")),
            BackendChoice::Native
        );
        // "gemma3-4b.gguf" → family "gemma" which prefix-matches "gemma3" → Native.
        assert_eq!(
            cfg.resolve(Path::new("gemma3-4b.gguf")),
            BackendChoice::Native
        );
    }

    // Auto routing: arch not in native_families → LlamaCpp (no probing).
    #[test]
    fn test_auto_routes_unknown_arch_to_llama_cpp() {
        let cfg = make_config_with_native(&[], BackendChoice::Auto, &["llama", "gemma3"]);
        // "phi-3.gguf" → family "phi" → not in native_families → LlamaCpp.
        assert_eq!(
            cfg.resolve(Path::new("phi-3-mini.gguf")),
            BackendChoice::LlamaCpp
        );
        // Unknown file → "unknown" → LlamaCpp.
        assert_eq!(
            cfg.resolve(Path::new("completely-unknown.gguf")),
            BackendChoice::LlamaCpp
        );
    }

    // An explicit families entry overrides native_families for that arch.
    #[test]
    fn test_families_override_beats_native_families() {
        // llama is in native_families but explicitly forced to LlamaCpp in families.
        let cfg = make_config_with_native(
            &[("llama", BackendChoice::LlamaCpp)],
            BackendChoice::Auto,
            &["llama", "gemma3"],
        );
        assert_eq!(
            cfg.resolve(Path::new("llama3-8b.gguf")),
            BackendChoice::LlamaCpp
        );
    }

    // RoutingConfig round-trips through YAML serialization.
    #[test]
    fn test_routing_config_serde() {
        let yaml = r#"
default_backend: auto
native_families:
  - llama
  - gemma3
  - gemma4
families:
  gptj: llama_cpp
  phi:  llama_cpp
"#;
        let cfg: RoutingConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.default_backend, BackendChoice::Auto);
        assert_eq!(cfg.native_families, vec!["llama", "gemma3", "gemma4"]);
        assert_eq!(cfg.families["gptj"], BackendChoice::LlamaCpp);
        assert_eq!(cfg.families["phi"], BackendChoice::LlamaCpp);

        // YAML without native_families uses the built-in default list.
        let yaml_no_native = "default_backend: auto\n";
        let cfg_default: RoutingConfig = serde_yaml::from_str(yaml_no_native).unwrap();
        assert_eq!(
            cfg_default.native_families,
            RoutingConfig::default_native_families()
        );

        // Serialize and re-parse.
        let yaml2 = serde_yaml::to_string(&cfg).unwrap();
        let cfg2: RoutingConfig = serde_yaml::from_str(&yaml2).unwrap();
        assert_eq!(cfg2.default_backend, BackendChoice::Auto);
        assert_eq!(cfg2.native_families.len(), 3);
        assert_eq!(cfg2.families.len(), 2);
    }
}
