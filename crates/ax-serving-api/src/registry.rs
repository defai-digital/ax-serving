//! Thread-safe model registry.
//!
//! Ported from ax-engine's `server/registry.rs`.
//! Uses Arc<RwLock<HashMap>> for concurrent reads during inference,
//! exclusive write lock only for load/unload (rare operations).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use anyhow::{Context, Result};
use ax_serving_engine::{InferenceBackend, LoadConfig, ModelHandle, ModelMetadata};
use tracing::{info, warn};

/// Typed errors returned by registry operations.
///
/// Wraps into `anyhow::Error` via the `?` operator so existing callers that
/// only use `e.to_string()` are unaffected.  Route handlers and the gRPC
/// service can call `e.downcast_ref::<RegistryError>()` to choose the correct
/// HTTP / gRPC status code without parsing error message strings.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("model already loaded: {0}")]
    AlreadyLoaded(String),
    #[error("model not loaded: {0}")]
    NotLoaded(String),
    #[error("model file not found: {0}")]
    FileNotFound(String),
    #[error("only .gguf models are supported; got: {0}")]
    InvalidFormat(String),
    #[error("max loaded models ({0}) reached; unload one first")]
    CapacityExceeded(usize),
    #[error("model path not allowed by AXS_MODEL_ALLOWED_DIRS: {0}")]
    PathNotAllowed(String),
    #[error("{0}")]
    InvalidModelId(String),
}

/// A single loaded model entry.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub id: String,
    pub path: PathBuf,
    pub handle: ModelHandle,
    pub metadata: ModelMetadata,
    /// Config used at load time — preserved for `reload()`.
    pub load_config: LoadConfig,
    pub loaded_at: Instant,
    /// Unix timestamp (ms) of the last `get()` call. Updated atomically.
    pub last_accessed_ms: Arc<AtomicU64>,
}

impl LoadedModel {
    /// Record the current time as the last-accessed timestamp.
    pub fn touch(&self) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_accessed_ms.store(now_ms, Ordering::Relaxed);
    }
}

/// Thread-safe model lifecycle manager.
///
/// Supports a warm pool (`AXS_MODEL_WARM_POOL_SIZE`): when the pool is full
/// and a new model is loaded, the oldest model (by `loaded_at`) is evicted
/// automatically.
///
/// `ModelRegistry` is `Clone` — clones share the same inner state (the `Arc`
/// wrapping is reused). Used to hand a reference to background tasks.
///
/// Read lock: many concurrent inference calls.
/// Write lock: load and unload (rare, and now only held for map mutations —
/// slow backend calls happen outside the lock).
#[derive(Clone)]
pub struct ModelRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
    /// Hard cap on simultaneously loaded models.
    max_loaded_models: usize,
    /// Maximum models to keep warm (None = no automatic eviction, hard cap applies).
    warm_pool_size: Option<usize>,
    /// Model IDs currently being loaded. Prevents two concurrent `load()` calls
    /// for the same model_id from both entering the (now lock-free) backend work
    /// phase, without holding the map write lock during the slow load.
    loading: Arc<Mutex<HashSet<String>>>,
}

/// RAII guard that removes a model ID from the `loading` set on drop.
/// Ensures the set is cleaned up on every exit path from `load()`.
struct LoadingGuard {
    loading: Arc<Mutex<HashSet<String>>>,
    id: String,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        if let Ok(mut g) = self.loading.lock() {
            g.remove(&self.id);
        }
    }
}

impl ModelRegistry {
    pub fn new(max_loaded_models: usize) -> Self {
        let warm_pool_size = std::env::var("AXS_MODEL_WARM_POOL_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0);
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            max_loaded_models,
            warm_pool_size,
            loading: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Load a model from a GGUF file, registering it under `model_id`.
    ///
    /// Validates:
    /// - File exists and has `.gguf` extension
    /// - `model_id` is 1–128 chars, alphanumeric/dash/underscore/dot
    /// - Not already loaded (checked under write lock to prevent TOCTOU)
    /// - Total loaded count < MAX_LOADED_MODELS
    ///
    /// The write lock is held only for fast map mutations. The slow backend
    /// calls (`unload_model` for warm-pool evictions and `load_model` for the
    /// new model) happen outside the lock so that concurrent `get()` / listing
    /// calls on other models are not stalled.
    pub fn load(
        &self,
        model_id: &str,
        path: &Path,
        config: LoadConfig,
        backend: &dyn InferenceBackend,
    ) -> Result<Arc<LoadedModel>> {
        validate_model_id(model_id)?;

        if !path.exists() {
            return Err(RegistryError::FileNotFound(path.display().to_string()).into());
        }
        let canonical_path = std::fs::canonicalize(path)
            .with_context(|| format!("failed to resolve model path {}", path.display()))?;

        if canonical_path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            return Err(RegistryError::InvalidFormat(canonical_path.display().to_string()).into());
        }

        if !is_allowed_model_path(&canonical_path)? {
            return Err(RegistryError::PathNotAllowed(canonical_path.display().to_string()).into());
        }

        // Reserve model_id in the loading set so two concurrent load() calls for
        // the same model_id cannot both proceed past this point.
        {
            let mut loading = self.loading.lock().unwrap();
            if loading.contains(model_id) {
                return Err(RegistryError::AlreadyLoaded(model_id.to_string()).into());
            }
            // Fast read to catch the already-loaded case without waiting for a write lock.
            if self.inner.read().unwrap().contains_key(model_id) {
                return Err(RegistryError::AlreadyLoaded(model_id.to_string()).into());
            }
            loading.insert(model_id.to_string());
        }
        // Clean up the loading reservation on every exit path (success or error).
        let _loading_guard = LoadingGuard {
            loading: Arc::clone(&self.loading),
            id: model_id.to_string(),
        };

        // Under write lock: TOCTOU checks + map mutations only (no backend calls).
        // Collect eviction candidates by removing them from the map; we'll call
        // backend.unload_model on them after releasing the lock.
        let eviction_candidates: Vec<Arc<LoadedModel>> = {
            let mut guard = self.inner.write().unwrap();

            // TOCTOU: re-check under write lock (fast).
            if guard.contains_key(model_id) {
                return Err(RegistryError::AlreadyLoaded(model_id.to_string()).into());
            }

            let mut candidates = Vec::new();
            if let Some(pool_size) = self.warm_pool_size {
                while guard.len() >= pool_size {
                    let oldest_id = guard
                        .iter()
                        .min_by_key(|(_, m)| m.last_accessed_ms.load(Ordering::Relaxed))
                        .map(|(id, _)| id.clone());
                    if let Some(id) = oldest_id {
                        let evicted = guard.remove(&id).unwrap();
                        candidates.push(evicted);
                    } else {
                        break;
                    }
                }
            }

            if guard.len() >= self.max_loaded_models {
                // Re-insert candidates; we cannot make room.
                for entry in candidates {
                    guard.insert(entry.id.clone(), entry);
                }
                return Err(RegistryError::CapacityExceeded(self.max_loaded_models).into());
            }

            candidates
        }; // write lock released here — `get()` and listing calls can proceed

        // Evict candidates outside the lock. On failure, re-insert to prevent a
        // handle leak, then bail.
        let mut failed_evictions: Vec<Arc<LoadedModel>> = Vec::new();
        for evicted in eviction_candidates {
            let evict_id = evicted.id.clone();
            match backend.unload_model(evicted.handle) {
                Ok(()) => info!("warm pool: evicted '{evict_id}' (oldest) to load '{model_id}'"),
                Err(e) => {
                    warn!("warm pool: failed to evict '{evict_id}' to make room for '{model_id}': {e}");
                    failed_evictions.push(evicted);
                }
            }
        }
        if !failed_evictions.is_empty() {
            let mut guard = self.inner.write().unwrap();
            for entry in failed_evictions {
                guard.insert(entry.id.clone(), entry);
            }
            anyhow::bail!("warm-pool eviction failed; cannot load '{model_id}'");
        }

        // Load model outside the write lock — this is the slow operation (seconds–minutes).
        let (handle, metadata) = backend
            .load_model(&canonical_path, config.clone())
            .with_context(|| format!("failed to load model from {}", canonical_path.display()))?;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let entry = Arc::new(LoadedModel {
            id: model_id.to_string(),
            path: canonical_path,
            handle,
            metadata,
            load_config: config,
            loaded_at: Instant::now(),
            last_accessed_ms: Arc::new(AtomicU64::new(now_ms)),
        });

        // Commit: insert under write lock. The LoadingGuard ensures no other
        // load() for this model_id is concurrently in progress.
        self.inner
            .write()
            .unwrap()
            .insert(model_id.to_string(), Arc::clone(&entry));

        // _loading_guard drops here, removing model_id from the loading set.
        info!("registered model '{model_id}' (handle={handle:?})");
        Ok(entry)
    }

    /// Unload a model by ID, freeing all associated resources.
    ///
    /// The write lock is held only long enough to remove the entry from the
    /// map. The slow `backend.unload_model` call happens after the lock is
    /// released so concurrent `get()` calls on other models are not stalled.
    pub fn unload(&self, model_id: &str, backend: &dyn InferenceBackend) -> Result<()> {
        // Remove from map under write lock (fast).
        let entry = {
            let mut guard = self.inner.write().unwrap();
            guard
                .remove(model_id)
                .ok_or_else(|| RegistryError::NotLoaded(model_id.to_string()))?
        }; // write lock released here

        // Backend unload outside the lock — slow operation.
        if let Err(e) = backend.unload_model(entry.handle) {
            // Re-insert to prevent a handle leak.
            self.inner
                .write()
                .unwrap()
                .insert(model_id.to_string(), entry);
            return Err(e).with_context(|| format!("backend failed to unload model '{model_id}'"));
        }

        info!("unloaded model '{model_id}'");
        Ok(())
    }

    /// Atomically reload a model: load a new handle first, then swap it into
    /// the registry, then unload the old handle.
    ///
    /// If the load phase fails, the old model remains live and registered —
    /// the caller sees an error but the model is still available.  This is
    /// safe against concurrent unload: if the model is removed between the
    /// read and write locks, the new handle is cleaned up before returning
    /// `NotLoaded`.
    pub fn reload(
        &self,
        model_id: &str,
        backend: &dyn InferenceBackend,
    ) -> Result<Arc<LoadedModel>> {
        // Step 1: read path + config under read lock, then release.
        let (path, load_config) = {
            let guard = self.inner.read().unwrap();
            let existing = guard
                .get(model_id)
                .ok_or_else(|| RegistryError::NotLoaded(model_id.to_string()))?;
            (existing.path.clone(), existing.load_config.clone())
        };

        // Step 2: load the new handle BEFORE touching the registry.
        // If this fails, the old model stays registered and live.
        let (new_handle, new_metadata) = backend
            .load_model(&path, load_config.clone())
            .with_context(|| format!("reload: load phase failed for '{model_id}'"))?;

        // Step 3: under write lock, swap the registry entry.
        let (new_entry, old_handle) = {
            let mut guard = self.inner.write().unwrap();

            let old_handle = match guard.get(model_id) {
                Some(e) => e.handle,
                None => {
                    // Model was unloaded concurrently while we were loading.
                    // Clean up the new handle to prevent a leak.
                    if let Err(e) = backend.unload_model(new_handle) {
                        warn!(
                            "reload: '{model_id}' concurrently unloaded; \
                             cleanup of new handle failed: {e}"
                        );
                    }
                    return Err(RegistryError::NotLoaded(model_id.to_string()).into());
                }
            };

            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let new_entry = Arc::new(LoadedModel {
                id: model_id.to_string(),
                path,
                handle: new_handle,
                metadata: new_metadata,
                load_config,
                loaded_at: Instant::now(),
                last_accessed_ms: Arc::new(AtomicU64::new(now_ms)),
            });
            guard.insert(model_id.to_string(), Arc::clone(&new_entry));
            (new_entry, old_handle)
        };

        // Step 4: unload the old handle outside the lock.
        // If this fails, log a warning — the new model is already live.
        if let Err(e) = backend.unload_model(old_handle) {
            warn!("reload: '{model_id}' reloaded but failed to free old handle: {e}");
        }
        info!("reloaded model '{model_id}' (new handle={new_handle:?})");
        Ok(new_entry)
    }

    /// Look up a model by ID (read lock). Updates `last_accessed_ms` on hit.
    pub fn get(&self, model_id: &str) -> Option<Arc<LoadedModel>> {
        let entry = self.inner.read().unwrap().get(model_id).cloned()?;
        entry.touch();
        Some(entry)
    }

    /// Unload all models whose `last_accessed_ms` is older than `idle_timeout_ms`.
    ///
    /// Called by the background eviction task spawned in `start_servers`.
    /// Returns the IDs of models that were unloaded.
    pub fn idle_evict_pass(
        &self,
        backend: &dyn InferenceBackend,
        idle_timeout_ms: u64,
    ) -> Vec<String> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Collect candidate IDs under read lock (IDs only, not handles).
        // We re-read the handle under write lock below to avoid TOCTOU.
        let candidates: Vec<String> = {
            let guard = self.inner.read().unwrap();
            guard
                .iter()
                .filter(|(_, m)| {
                    let last = m.last_accessed_ms.load(Ordering::Relaxed);
                    now_ms.saturating_sub(last) >= idle_timeout_ms
                })
                .map(|(id, _)| id.clone())
                .collect()
        };

        let mut evicted = Vec::new();
        for id in candidates {
            // Under write lock: re-check both existence and idle condition so
            // that a model accessed or reloaded between the two locks is not
            // incorrectly evicted with a stale handle.
            //
            // Capture the Arc<LoadedModel> (not just the handle) so we can
            // re-insert it into the map if backend.unload_model() fails,
            // preventing a permanent handle leak.
            let entry_to_evict: Option<Arc<LoadedModel>> = {
                let mut guard = self.inner.write().unwrap();
                let should_evict = if let Some(entry) = guard.get(&id) {
                    let last = entry.last_accessed_ms.load(Ordering::Relaxed);
                    now_ms.saturating_sub(last) >= idle_timeout_ms
                } else {
                    false // already unloaded
                };
                if should_evict {
                    guard.remove(&id) // returns the Arc<LoadedModel>
                } else {
                    None // accessed since our snapshot — skip
                }
            }; // write lock released here

            if let Some(entry) = entry_to_evict {
                if let Err(e) = backend.unload_model(entry.handle) {
                    warn!("idle eviction: failed to unload '{id}': {e}");
                    let mut guard = self.inner.write().unwrap();
                    guard.insert(id, entry); // re-insert: avoid handle leak
                } else {
                    evicted.push(id);
                }
            }
        }
        evicted
    }

    /// List all loaded model IDs.
    pub fn list_ids(&self) -> Vec<String> {
        self.inner.read().unwrap().keys().cloned().collect()
    }

    /// Return all loaded model entries without updating `last_accessed_ms`.
    ///
    /// Use this for read-only enumeration (e.g. list-models RPC) where the
    /// caller is not actually consuming the model — calling `get()` would
    /// incorrectly reset the idle eviction timer for every listed model.
    pub fn list_entries(&self) -> Vec<Arc<LoadedModel>> {
        self.inner.read().unwrap().values().cloned().collect()
    }

    /// Return `(model_id, metadata)` pairs for all currently loaded models.
    ///
    /// Does not update `last_accessed_ms` — safe for read-only enumeration
    /// such as emitting Prometheus per-model metrics.
    pub fn loaded_models_with_meta(&self) -> Vec<(String, ModelMetadata)> {
        self.inner
            .read()
            .unwrap()
            .iter()
            .map(|(id, m)| (id.clone(), m.metadata.clone()))
            .collect()
    }

    /// Number of currently loaded models.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn is_allowed_model_path(path: &Path) -> Result<bool> {
    let raw = match std::env::var("AXS_MODEL_ALLOWED_DIRS") {
        Ok(v) => v,
        Err(_) => return Ok(true),
    };

    let dirs: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();

    if dirs.is_empty() {
        return Ok(true);
    }

    for d in dirs {
        let root = std::fs::canonicalize(&d)
            .with_context(|| format!("invalid path in AXS_MODEL_ALLOWED_DIRS: {d}"))?;
        if path.starts_with(&root) {
            return Ok(true);
        }
    }

    Ok(false)
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Validate model ID format: 1–128 chars, alphanumeric / dash / underscore / dot.
fn validate_model_id(id: &str) -> Result<(), RegistryError> {
    if id.is_empty() {
        return Err(RegistryError::InvalidModelId(
            "model_id cannot be empty".into(),
        ));
    }
    if id.len() > 128 {
        return Err(RegistryError::InvalidModelId(
            "model_id too long (max 128 chars)".into(),
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(RegistryError::InvalidModelId(format!(
            "model_id must be alphanumeric with dashes, underscores, or dots; got: {id}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::Ordering;

    use ax_serving_engine::{
        GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig, ModelHandle,
        ModelMetadata, ThermalState,
    };

    use super::*;

    // ── Test backend ──────────────────────────────────────────────────────────

    struct NullBackend;

    impl InferenceBackend for NullBackend {
        fn load_model(
            &self,
            _path: &Path,
            _config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            Ok((
                ModelHandle(1),
                ModelMetadata {
                    architecture: "null".into(),
                    n_layers: 0,
                    n_heads: 0,
                    n_kv_heads: 0,
                    embedding_dim: 0,
                    vocab_size: 0,
                    context_length: 2048,
                    load_time_ms: 1,
                    peak_rss_bytes: 0,
                },
            ))
        }

        fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _: ModelHandle,
            _: GenerateInput,
            _: GenerationParams,
            _: tokio::sync::mpsc::Sender<GenerateEvent>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn tokenize(&self, _: ModelHandle, _: &str, _: bool) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn decode_tokens(&self, _: ModelHandle, _: &[u32]) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn eos_tokens(&self, _: ModelHandle) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn thermal_state(&self) -> ThermalState {
            ThermalState::Nominal
        }

        fn recommended_concurrency(&self) -> usize {
            4
        }
    }

    /// Backend whose `unload_model` always fails — used to test handle-leak prevention.
    struct FailingUnloadBackend;

    impl InferenceBackend for FailingUnloadBackend {
        fn load_model(
            &self,
            _path: &Path,
            _config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            Ok((
                ModelHandle(1),
                ModelMetadata {
                    architecture: "null".into(),
                    n_layers: 0,
                    n_heads: 0,
                    n_kv_heads: 0,
                    embedding_dim: 0,
                    vocab_size: 0,
                    context_length: 2048,
                    load_time_ms: 1,
                    peak_rss_bytes: 0,
                },
            ))
        }

        fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
            Err(anyhow::anyhow!("simulated backend unload failure"))
        }

        fn generate(
            &self,
            _: ModelHandle,
            _: GenerateInput,
            _: GenerationParams,
            _: tokio::sync::mpsc::Sender<GenerateEvent>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn tokenize(&self, _: ModelHandle, _: &str, _: bool) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn decode_tokens(&self, _: ModelHandle, _: &[u32]) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn eos_tokens(&self, _: ModelHandle) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn thermal_state(&self) -> ThermalState {
            ThermalState::Nominal
        }

        fn recommended_concurrency(&self) -> usize {
            4
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_gguf(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"dummy").unwrap();
        path
    }

    // ── model_id validation tests ─────────────────────────────────────────────

    #[test]
    fn valid_ids() {
        for id in &["llama3", "llama-3.1", "model_v2", "a"] {
            assert!(validate_model_id(id).is_ok(), "expected valid: {id}");
        }
    }

    #[test]
    fn invalid_ids() {
        assert!(validate_model_id("").is_err());
        assert!(validate_model_id("model id").is_err()); // space
        assert!(validate_model_id(&"a".repeat(129)).is_err()); // too long
    }

    // ── ModelRegistry lifecycle tests ─────────────────────────────────────────

    #[test]
    fn load_and_get_returns_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        let entry = reg
            .load("mymodel", &path, LoadConfig::default(), &backend)
            .unwrap();
        assert_eq!(entry.id, "mymodel");

        let got = reg.get("mymodel").unwrap();
        assert_eq!(got.id, "mymodel");
        assert_eq!(got.metadata.architecture, "null");
    }

    #[test]
    fn load_get_returns_none_for_unknown() {
        let reg = ModelRegistry::new(16);
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn load_already_loaded_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        reg.load("dup", &path, LoadConfig::default(), &backend)
            .unwrap();
        let err = reg
            .load("dup", &path, LoadConfig::default(), &backend)
            .unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::AlreadyLoaded(_))),
            "expected AlreadyLoaded, got: {err}"
        );
    }

    #[test]
    fn load_file_not_found_errors() {
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        let err = reg
            .load(
                "m",
                std::path::Path::new("/no/such/file.gguf"),
                LoadConfig::default(),
                &backend,
            )
            .unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::FileNotFound(_))),
            "expected FileNotFound, got: {err}"
        );
    }

    #[test]
    fn load_invalid_format_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        std::fs::write(&path, b"dummy").unwrap();

        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        let err = reg
            .load("m", &path, LoadConfig::default(), &backend)
            .unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::InvalidFormat(_))),
            "expected InvalidFormat, got: {err}"
        );
    }

    #[test]
    fn load_capacity_exceeded_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("m1.gguf");
        let path2 = dir.path().join("m2.gguf");
        std::fs::write(&path1, b"dummy").unwrap();
        std::fs::write(&path2, b"dummy").unwrap();

        let backend = NullBackend;
        let reg = ModelRegistry::new(1); // hard cap of 1

        reg.load("first", &path1, LoadConfig::default(), &backend)
            .unwrap();
        let err = reg
            .load("second", &path2, LoadConfig::default(), &backend)
            .unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::CapacityExceeded(_))),
            "expected CapacityExceeded, got: {err}"
        );
    }

    #[test]
    fn load_invalid_model_id_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        let err = reg
            .load("bad id!", &path, LoadConfig::default(), &backend)
            .unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::InvalidModelId(_))),
            "expected InvalidModelId, got: {err}"
        );
    }

    #[test]
    fn unload_removes_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        reg.load("m", &path, LoadConfig::default(), &backend)
            .unwrap();
        assert!(reg.get("m").is_some());

        reg.unload("m", &backend).unwrap();
        assert!(reg.get("m").is_none(), "model must be gone after unload");
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn unload_not_loaded_errors() {
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        let err = reg.unload("missing", &backend).unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::NotLoaded(_))),
            "expected NotLoaded, got: {err}"
        );
    }

    #[test]
    fn unload_backend_failure_reinserts_entry() {
        // If the backend fails to unload, the entry must be re-inserted to prevent a
        // handle leak. Subsequent `get()` must still find the model.
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = FailingUnloadBackend;
        let reg = ModelRegistry::new(16);

        reg.load("m", &path, LoadConfig::default(), &backend)
            .unwrap();
        let result = reg.unload("m", &backend);
        assert!(result.is_err(), "backend failure must propagate as error");
        assert!(
            reg.get("m").is_some(),
            "entry must be re-inserted after failed unload to prevent handle leak"
        );
    }

    #[test]
    fn reload_swaps_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        reg.load("m", &path, LoadConfig::default(), &backend)
            .unwrap();
        let reloaded = reg.reload("m", &backend).unwrap();
        assert_eq!(reloaded.id, "m");
        // Model must still be accessible after reload.
        assert!(reg.get("m").is_some());
    }

    #[test]
    fn reload_not_loaded_errors() {
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        let err = reg.reload("missing", &backend).unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::NotLoaded(_))),
            "expected NotLoaded, got: {err}"
        );
    }

    #[test]
    fn list_ids_returns_all_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("a.gguf");
        let p2 = dir.path().join("b.gguf");
        std::fs::write(&p1, b"x").unwrap();
        std::fs::write(&p2, b"x").unwrap();

        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        reg.load("alpha", &p1, LoadConfig::default(), &backend)
            .unwrap();
        reg.load("beta", &p2, LoadConfig::default(), &backend)
            .unwrap();

        let mut ids = reg.list_ids();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn list_entries_does_not_update_last_accessed_ms() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        reg.load("m", &path, LoadConfig::default(), &backend)
            .unwrap();

        // Set last_accessed_ms to a known value via list_entries (no touch).
        let before_ts = reg.list_entries()[0]
            .last_accessed_ms
            .load(Ordering::Relaxed);
        // Calling list_entries again must not change last_accessed_ms.
        let after_ts = reg.list_entries()[0]
            .last_accessed_ms
            .load(Ordering::Relaxed);
        assert_eq!(
            before_ts, after_ts,
            "list_entries must not update last_accessed_ms"
        );
    }

    #[test]
    fn get_updates_last_accessed_ms() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        reg.load("m", &path, LoadConfig::default(), &backend)
            .unwrap();

        // Set last_accessed_ms to 0 to simulate a very old access.
        let entry = reg.list_entries().into_iter().next().unwrap();
        entry.last_accessed_ms.store(0, Ordering::Relaxed);

        // get() must call touch() and update the timestamp.
        reg.get("m").unwrap();
        let ts = entry.last_accessed_ms.load(Ordering::Relaxed);
        assert!(ts > 0, "get() must update last_accessed_ms via touch()");
    }

    #[test]
    fn loaded_models_with_meta_returns_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        reg.load("arch-test", &path, LoadConfig::default(), &backend)
            .unwrap();

        let pairs = reg.loaded_models_with_meta();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "arch-test");
        assert_eq!(pairs[0].1.architecture, "null");
    }

    #[test]
    fn len_and_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);

        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);

        reg.load("m", &path, LoadConfig::default(), &backend)
            .unwrap();
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);

        reg.unload("m", &backend).unwrap();
        assert!(reg.is_empty());
    }

    #[test]
    fn idle_evict_pass_removes_stale_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        reg.load("stale", &path, LoadConfig::default(), &backend)
            .unwrap();

        // Force last_accessed_ms to epoch (0) so any idle_timeout_ms > 0 fires.
        let entry = reg.list_entries().into_iter().next().unwrap();
        entry.last_accessed_ms.store(0, Ordering::Relaxed);

        let evicted = reg.idle_evict_pass(&backend, 1 /* ms */);
        assert_eq!(evicted, vec!["stale"]);
        assert!(reg.is_empty(), "stale model must be evicted");
    }

    #[test]
    fn idle_evict_pass_skips_recently_accessed() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        reg.load("fresh", &path, LoadConfig::default(), &backend)
            .unwrap();

        // Ensure last_accessed_ms is recent by calling get().
        reg.get("fresh").unwrap();

        // idle_timeout is very large — recent access must not be evicted.
        let evicted = reg.idle_evict_pass(&backend, 10_000_000 /* 10 000 s */);
        assert!(evicted.is_empty(), "recently accessed model must not be evicted");
        assert!(!reg.is_empty());
    }

    #[test]
    fn idle_evict_backend_failure_reinserts_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = FailingUnloadBackend;
        let reg = ModelRegistry::new(16);
        reg.load("old", &path, LoadConfig::default(), &backend)
            .unwrap();

        let entry = reg.list_entries().into_iter().next().unwrap();
        entry.last_accessed_ms.store(0, Ordering::Relaxed);

        // Even though the backend fails, idle_evict_pass must re-insert the
        // entry rather than leaking the handle.
        let evicted = reg.idle_evict_pass(&backend, 1);
        assert!(evicted.is_empty(), "failed eviction must not appear in return list");
        assert!(
            reg.get("old").is_some(),
            "entry must be re-inserted after failed idle eviction"
        );
    }
}
