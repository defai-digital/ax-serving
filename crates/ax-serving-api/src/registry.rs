//! Thread-safe model registry.
//!
//! Ported from ax-engine's `server/registry.rs`.
//! Uses Arc<RwLock<HashMap>> for concurrent reads during inference,
//! exclusive write lock only for load/unload (rare operations).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use anyhow::{Context, Result};
use ax_serving_engine::{
    InferenceBackend, LoadConfig, ModelHandle, ModelMetadata, is_ax_engine_model_artifacts,
    is_mlx_model,
};
use tracing::{info, warn};

use crate::utils::time::unix_now_ms;

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
    #[error(
        "unsupported model path format; expected .gguf file, ax-engine artifact directory, or MLX directory: {0}"
    )]
    InvalidFormat(String),
    #[error("max loaded models ({0}) reached; unload one first")]
    CapacityExceeded(usize),
    #[error("model path not allowed by AXS_MODEL_ALLOWED_DIRS: {0}")]
    PathNotAllowed(String),
    #[error("model is busy: {0}")]
    Busy(String),
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
        self.last_accessed_ms
            .store(unix_now_ms(), Ordering::Relaxed);
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
/// Write lock: load and unload (rare). Most slow backend calls happen outside
/// the lock; final warm-pool eviction holds the write lock so an uncommitted
/// load is never exposed to readers before rollback is impossible.
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
        let mut g = match self.loading.lock() {
            Ok(g) => g,
            Err(err) => {
                warn!(%err, id = %self.id, "loading set lock poisoned during load guard drop; continuing with poisoned state");
                err.into_inner()
            }
        };
        if !self.id.is_empty() {
            g.remove(&self.id);
        }
    }
}

impl ModelRegistry {
    fn loading_lock(&self) -> MutexGuard<'_, HashSet<String>> {
        match self.loading.lock() {
            Ok(g) => g,
            Err(err) => {
                warn!(%err, "loading set lock poisoned; continuing with poisoned state");
                err.into_inner()
            }
        }
    }

    fn inner_read(&self) -> RwLockReadGuard<'_, HashMap<String, Arc<LoadedModel>>> {
        match self.inner.read() {
            Ok(g) => g,
            Err(err) => {
                warn!(%err, "model registry read lock poisoned; continuing with poisoned state");
                err.into_inner()
            }
        }
    }

    fn inner_write(&self) -> RwLockWriteGuard<'_, HashMap<String, Arc<LoadedModel>>> {
        match self.inner.write() {
            Ok(g) => g,
            Err(err) => {
                warn!(%err, "model registry write lock poisoned; continuing with poisoned state");
                err.into_inner()
            }
        }
    }

    pub fn new(max_loaded_models: usize) -> Self {
        match Self::try_new(max_loaded_models) {
            Ok(registry) => registry,
            Err(err) => {
                warn!(
                    %err,
                    "invalid AXS_MODEL_WARM_POOL_SIZE ignored by infallible registry constructor"
                );
                Self::new_with_warm_pool_size(max_loaded_models, None)
            }
        }
    }

    pub fn try_new(max_loaded_models: usize) -> Result<Self> {
        let warm_pool_size = warm_pool_size_from_env()?;
        Ok(Self::new_with_warm_pool_size(
            max_loaded_models,
            warm_pool_size,
        ))
    }

    fn new_with_warm_pool_size(max_loaded_models: usize, warm_pool_size: Option<usize>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            max_loaded_models,
            warm_pool_size,
            loading: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    fn collect_warm_pool_evictions(
        guard: &mut HashMap<String, Arc<LoadedModel>>,
        pool_size: usize,
        incoming_models: usize,
        model_id: &str,
    ) -> std::result::Result<Vec<Arc<LoadedModel>>, RegistryError> {
        let evictions_needed = guard
            .len()
            .saturating_add(incoming_models)
            .saturating_sub(pool_size);
        if evictions_needed == 0 {
            return Ok(Vec::new());
        }

        let mut evictable: Vec<(String, u64)> = guard
            .iter()
            .filter(|(_, model)| Arc::strong_count(model) == 1)
            .map(|(id, model)| (id.clone(), model.last_accessed_ms.load(Ordering::Relaxed)))
            .collect();
        if evictable.len() < evictions_needed {
            return Err(RegistryError::Busy(model_id.to_string()));
        }

        evictable.sort_by_key(|(_, last_accessed)| *last_accessed);
        Ok(evictable
            .into_iter()
            .take(evictions_needed)
            .filter_map(|(id, _)| guard.remove(&id))
            .collect())
    }

    /// Load a model file or supported artifact directory, registering it under `model_id`.
    ///
    /// Validates:
    /// - Path exists and is a supported model file/directory for the selected backend
    /// - `model_id` is 1–128 chars, alphanumeric/dash/underscore/dot
    /// - Not already loaded (checked under write lock to prevent TOCTOU)
    /// - Total loaded count < MAX_LOADED_MODELS
    ///
    /// The write lock is held only for fast map mutations. The slow backend
    /// calls happen outside the lock where possible so that concurrent `get()`
    /// / listing calls on other models are not stalled. Final warm-pool
    /// eviction is the exception: it holds the write lock until eviction
    /// succeeds so readers cannot observe a new handle that may still need
    /// rollback cleanup.
    pub fn load(
        &self,
        model_id: &str,
        path: &Path,
        mut config: LoadConfig,
        backend: &dyn InferenceBackend,
    ) -> Result<Arc<LoadedModel>> {
        validate_model_id(model_id)?;

        if !path.exists() {
            return Err(RegistryError::FileNotFound(path.display().to_string()).into());
        }
        let canonical_path = std::fs::canonicalize(path)
            .with_context(|| format!("failed to resolve model path {}", path.display()))?;

        validate_model_path_format(&canonical_path, &config)?;

        if !is_allowed_model_path(&canonical_path)? {
            return Err(RegistryError::PathNotAllowed(canonical_path.display().to_string()).into());
        }
        validate_optional_mmproj_path(&mut config)?;

        // Reserve model_id in the loading set so two concurrent load() calls for
        // the same model_id cannot both proceed past this point.
        {
            let mut loading = self.loading_lock();
            if loading.contains(model_id) {
                return Err(RegistryError::AlreadyLoaded(model_id.to_string()).into());
            }
            // Fast read to catch the already-loaded case without waiting for a write lock.
            if self.inner_read().contains_key(model_id) {
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
            let mut guard = self.inner_write();

            // TOCTOU: re-check under write lock (fast).
            if guard.contains_key(model_id) {
                return Err(RegistryError::AlreadyLoaded(model_id.to_string()).into());
            }

            let mut candidates = Vec::new();
            if let Some(pool_size) = self.warm_pool_size {
                candidates = Self::collect_warm_pool_evictions(&mut guard, pool_size, 1, model_id)?;
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
                    warn!(
                        "warm pool: failed to evict '{evict_id}' to make room for '{model_id}': {e}"
                    );
                    failed_evictions.push(evicted);
                }
            }
        }
        if !failed_evictions.is_empty() {
            let mut guard = self.inner_write();
            for entry in failed_evictions {
                guard.insert(entry.id.clone(), entry);
            }
            anyhow::bail!("warm-pool eviction failed; cannot load '{model_id}'");
        }

        // Load model outside the write lock — this is the slow operation (seconds–minutes).
        let (handle, metadata) = backend
            .load_model(&canonical_path, config.clone())
            .with_context(|| format!("failed to load model from {}", canonical_path.display()))?;

        let entry = Arc::new(LoadedModel {
            id: model_id.to_string(),
            path: canonical_path,
            handle,
            metadata,
            load_config: config,
            loaded_at: Instant::now(),
            last_accessed_ms: Arc::new(AtomicU64::new(unix_now_ms())),
        });

        // Commit: unload required final warm-pool evictions and then insert
        // under the write lock. The LoadingGuard ensures no other load() for
        // this model_id is concurrently in progress. Keeping the new entry out
        // of the map until evictions succeed avoids exposing a handle that may
        // need to be cleaned up on rollback.
        //
        // Re-apply warm-pool eviction here as well as before the slow load:
        // different model IDs may load concurrently, and each can pass the
        // pre-load warm-pool check before any of them commits.
        {
            let mut guard = self.inner_write();
            let mut candidates = Vec::new();
            if let Some(pool_size) = self.warm_pool_size {
                match Self::collect_warm_pool_evictions(&mut guard, pool_size, 1, model_id) {
                    Ok(evictions) => candidates = evictions,
                    Err(err) => {
                        drop(guard);
                        backend.unload_model(handle).with_context(|| {
                            format!(
                                "warm-pool finalization for '{model_id}' failed; cleanup failed"
                            )
                        })?;
                        return Err(err.into());
                    }
                }
            }

            if guard.len() >= self.max_loaded_models {
                for entry in candidates {
                    guard.insert(entry.id.clone(), entry);
                }
                drop(guard);
                backend.unload_model(handle).with_context(|| {
                    format!(
                        "capacity exceeded while finalizing load for '{model_id}'; cleanup failed"
                    )
                })?;
                return Err(RegistryError::CapacityExceeded(self.max_loaded_models).into());
            }

            let mut failed_final_evictions = Vec::new();
            for evicted in candidates {
                let evict_id = evicted.id.clone();
                match backend.unload_model(evicted.handle) {
                    Ok(()) => info!(
                        "warm pool: evicted '{evict_id}' during finalization to load '{model_id}'"
                    ),
                    Err(e) => {
                        warn!(
                            "warm pool: failed to finalize eviction of '{evict_id}' for '{model_id}': {e}"
                        );
                        failed_final_evictions.push(evicted);
                    }
                }
            }
            if !failed_final_evictions.is_empty() {
                for evicted in failed_final_evictions {
                    guard.insert(evicted.id.clone(), evicted);
                }
                drop(guard);
                if let Err(e) = backend.unload_model(handle) {
                    warn!(
                        "warm pool: failed to clean up newly loaded '{model_id}' after final eviction failure: {e}"
                    );
                }
                anyhow::bail!("warm-pool final eviction failed; cannot load '{model_id}'");
            }

            guard.insert(model_id.to_string(), Arc::clone(&entry));
        }

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
            let mut guard = self.inner_write();
            let entry = guard
                .remove(model_id)
                .ok_or_else(|| RegistryError::NotLoaded(model_id.to_string()))?;
            if Arc::strong_count(&entry) > 1 {
                guard.insert(model_id.to_string(), entry);
                return Err(RegistryError::Busy(model_id.to_string()).into());
            }
            entry
        }; // write lock released here

        // Backend unload outside the lock — slow operation.
        if let Err(e) = backend.unload_model(entry.handle) {
            // Re-insert to prevent a handle leak.
            {
                let mut guard = self.inner_write();
                guard.insert(model_id.to_string(), entry);
            }
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
            let guard = self.inner_read();
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
            let mut guard = self.inner_write();

            let old_handle = match guard.get(model_id) {
                Some(e) => {
                    if Arc::strong_count(e) > 1 {
                        drop(guard);
                        if let Err(err) = backend.unload_model(new_handle) {
                            warn!(
                                "reload: '{model_id}' is busy; cleanup of unused new handle failed: {err}"
                            );
                        }
                        return Err(RegistryError::Busy(model_id.to_string()).into());
                    }
                    e.handle
                }
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

            let new_entry = Arc::new(LoadedModel {
                id: model_id.to_string(),
                path,
                handle: new_handle,
                metadata: new_metadata,
                load_config,
                loaded_at: Instant::now(),
                last_accessed_ms: Arc::new(AtomicU64::new(unix_now_ms())),
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
        let entry = self.inner_read().get(model_id).cloned()?;
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
        let now_ms = unix_now_ms();

        // Collect candidate IDs under read lock (IDs only, not handles).
        // We re-read the handle under write lock below to avoid TOCTOU.
        let candidates: Vec<String> = {
            let guard = self.inner_read();
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
                let mut guard = self.inner_write();
                let should_evict = if let Some(entry) = guard.get(&id) {
                    let last = entry.last_accessed_ms.load(Ordering::Relaxed);
                    now_ms.saturating_sub(last) >= idle_timeout_ms && Arc::strong_count(entry) == 1
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
                    let mut guard = self.inner_write();
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
        self.inner_read().keys().cloned().collect()
    }

    /// Return all loaded model entries without updating `last_accessed_ms`.
    ///
    /// Use this for read-only enumeration (e.g. list-models RPC) where the
    /// caller is not actually consuming the model — calling `get()` would
    /// incorrectly reset the idle eviction timer for every listed model.
    pub fn list_entries(&self) -> Vec<Arc<LoadedModel>> {
        self.inner_read().values().cloned().collect()
    }

    /// Return `(model_id, metadata)` pairs for all currently loaded models.
    ///
    /// Does not update `last_accessed_ms` — safe for read-only enumeration
    /// such as emitting Prometheus per-model metrics.
    pub fn loaded_models_with_meta(&self) -> Vec<(String, ModelMetadata)> {
        self.inner_read()
            .iter()
            .map(|(id, m)| (id.clone(), m.metadata.clone()))
            .collect()
    }

    /// Number of currently loaded models.
    pub fn len(&self) -> usize {
        self.inner_read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn warm_pool_size_from_env() -> Result<Option<usize>> {
    match std::env::var("AXS_MODEL_WARM_POOL_SIZE") {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                let size = trimmed
                    .parse::<usize>()
                    .context("invalid AXS_MODEL_WARM_POOL_SIZE")?;
                Ok((size > 0).then_some(size))
            }
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(err).context("invalid AXS_MODEL_WARM_POOL_SIZE"),
    }
}

fn is_allowed_model_path(path: &Path) -> Result<bool> {
    #[cfg(test)]
    if let Some(dirs) = test_allowed_model_dirs() {
        return Ok(dirs.is_empty() || dirs.iter().any(|root| path.starts_with(root)));
    }

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

#[cfg(test)]
thread_local! {
    static TEST_ALLOWED_MODEL_DIRS: std::cell::RefCell<Option<Vec<PathBuf>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn test_allowed_model_dirs() -> Option<Vec<PathBuf>> {
    TEST_ALLOWED_MODEL_DIRS.with(|dirs| dirs.borrow().clone())
}

fn validate_model_path_format(path: &Path, config: &LoadConfig) -> Result<(), RegistryError> {
    if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        return Ok(());
    }

    let hint = config.backend_hint.as_deref();
    let directory_supported = match hint {
        Some("native") | Some("auto") | None => is_ax_engine_model_artifacts(path),
        Some("mlx") => is_mlx_model(path),
        Some("llama_cpp" | "lib_llama") => false,
        Some(_) => false,
    };

    if directory_supported {
        return Ok(());
    }

    Err(RegistryError::InvalidFormat(path.display().to_string()))
}

fn validate_optional_mmproj_path(config: &mut LoadConfig) -> Result<()> {
    let Some(raw) = config.mmproj_path.as_deref() else {
        return Ok(());
    };

    let path = Path::new(raw);
    if !path.exists() {
        return Err(RegistryError::FileNotFound(path.display().to_string()).into());
    }

    let canonical = std::fs::canonicalize(path)
        .map_err(|_| RegistryError::FileNotFound(path.display().to_string()))?;

    if !canonical.is_file() || canonical.extension().and_then(|e| e.to_str()) != Some("gguf") {
        return Err(RegistryError::InvalidFormat(canonical.display().to_string()).into());
    }

    if !is_allowed_model_path(&canonical)? {
        return Err(RegistryError::PathNotAllowed(canonical.display().to_string()).into());
    }

    config.mmproj_path = Some(canonical.to_string_lossy().into_owned());
    Ok(())
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
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(RegistryError::InvalidModelId(format!(
            "model_id must be alphanumeric with dashes, underscores, or dots; got: {id}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use ax_serving_engine::{
        GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig, ModelHandle,
        ModelMetadata, ThermalState,
    };

    use super::*;

    // ── Test backend ──────────────────────────────────────────────────────────

    struct EnvGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    struct AllowedDirsGuard {
        old: Option<Vec<PathBuf>>,
    }

    impl AllowedDirsGuard {
        fn set(paths: &[&Path]) -> Self {
            let canonical: Vec<PathBuf> = paths
                .iter()
                .map(|path| std::fs::canonicalize(path).unwrap())
                .collect();
            let old = TEST_ALLOWED_MODEL_DIRS.with(|dirs| dirs.replace(Some(canonical)));
            Self { old }
        }
    }

    impl Drop for AllowedDirsGuard {
        fn drop(&mut self) {
            TEST_ALLOWED_MODEL_DIRS.with(|dirs| {
                dirs.replace(self.old.take());
            });
        }
    }

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
                    resolved_backend: ax_serving_engine::BackendType::Auto,
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

    struct CountingBackend {
        next_handle: AtomicU64,
        unloads: AtomicU64,
    }

    impl CountingBackend {
        fn new() -> Self {
            Self {
                next_handle: AtomicU64::new(0),
                unloads: AtomicU64::new(0),
            }
        }
    }

    impl InferenceBackend for CountingBackend {
        fn load_model(
            &self,
            _path: &Path,
            _config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed) + 1;
            Ok((
                ModelHandle(handle),
                ModelMetadata {
                    architecture: "counting".into(),
                    n_layers: 0,
                    n_heads: 0,
                    n_kv_heads: 0,
                    embedding_dim: 0,
                    vocab_size: 0,
                    context_length: 2048,
                    load_time_ms: 1,
                    peak_rss_bytes: 0,
                    resolved_backend: ax_serving_engine::BackendType::Auto,
                },
            ))
        }

        fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
            self.unloads.fetch_add(1, Ordering::Relaxed);
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

    struct CaptureConfigBackend {
        observed: Arc<Mutex<Option<LoadConfig>>>,
    }

    impl InferenceBackend for CaptureConfigBackend {
        fn load_model(
            &self,
            _path: &Path,
            config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            *self.observed.lock().unwrap() = Some(config);
            Ok((
                ModelHandle(1),
                ModelMetadata {
                    architecture: "capture".into(),
                    n_layers: 0,
                    n_heads: 0,
                    n_kv_heads: 0,
                    embedding_dim: 0,
                    vocab_size: 0,
                    context_length: 2048,
                    load_time_ms: 1,
                    peak_rss_bytes: 0,
                    resolved_backend: ax_serving_engine::BackendType::Auto,
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
                    resolved_backend: ax_serving_engine::BackendType::Auto,
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

    fn make_ax_engine_artifacts(dir: &tempfile::TempDir) {
        std::fs::write(dir.path().join("model-manifest.json"), "{}").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
    }

    fn make_mlx_model(dir: &tempfile::TempDir) {
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"dummy").unwrap();
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

    #[test]
    fn warm_pool_size_from_env_defaults_when_unset() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::unset("AXS_MODEL_WARM_POOL_SIZE");

        assert_eq!(warm_pool_size_from_env().unwrap(), None);
    }

    #[test]
    fn warm_pool_size_from_env_parses_trimmed_value() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::set("AXS_MODEL_WARM_POOL_SIZE", " 2 ");

        assert_eq!(warm_pool_size_from_env().unwrap(), Some(2));
    }

    #[test]
    fn warm_pool_size_from_env_rejects_malformed_value() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::set("AXS_MODEL_WARM_POOL_SIZE", "many");

        let err = warm_pool_size_from_env().unwrap_err();
        assert!(err.to_string().contains("AXS_MODEL_WARM_POOL_SIZE"));
    }

    #[test]
    fn try_new_rejects_malformed_warm_pool_env() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::set("AXS_MODEL_WARM_POOL_SIZE", "many");

        let err = ModelRegistry::try_new(16)
            .err()
            .expect("malformed warm-pool env should fail");
        assert!(err.to_string().contains("AXS_MODEL_WARM_POOL_SIZE"));
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
    fn load_ax_engine_artifact_directory_is_allowed_for_auto_routing() {
        let dir = tempfile::tempdir().unwrap();
        make_ax_engine_artifacts(&dir);

        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        let config = LoadConfig {
            backend_hint: Some("auto".into()),
            ..LoadConfig::default()
        };

        let entry = reg
            .load("native-dir", dir.path(), config, &backend)
            .expect("AX Engine artifact directory should pass registry validation");

        assert_eq!(entry.path, std::fs::canonicalize(dir.path()).unwrap());
    }

    #[test]
    fn load_mlx_directory_is_allowed_for_mlx_hint() {
        let dir = tempfile::tempdir().unwrap();
        make_mlx_model(&dir);

        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        let config = LoadConfig {
            backend_hint: Some("mlx".into()),
            ..LoadConfig::default()
        };

        let entry = reg
            .load("mlx-dir", dir.path(), config, &backend)
            .expect("MLX model directory should pass registry validation for mlx hint");

        assert_eq!(entry.path, std::fs::canonicalize(dir.path()).unwrap());
    }

    #[test]
    fn load_directory_is_rejected_for_llama_cpp_hint() {
        let dir = tempfile::tempdir().unwrap();
        make_ax_engine_artifacts(&dir);

        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        let config = LoadConfig {
            backend_hint: Some("llama_cpp".into()),
            ..LoadConfig::default()
        };

        let err = reg.load("dir", dir.path(), config, &backend).unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::InvalidFormat(_))),
            "expected InvalidFormat, got: {err}"
        );
    }

    #[test]
    fn load_mmproj_path_is_canonicalized_before_backend_load() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = make_gguf(&dir);
        let mmproj_path = dir.path().join("mmproj.gguf");
        std::fs::write(&mmproj_path, b"projector").unwrap();

        let observed = Arc::new(Mutex::new(None));
        let backend = CaptureConfigBackend {
            observed: Arc::clone(&observed),
        };
        let reg = ModelRegistry::new(16);
        let config = LoadConfig {
            mmproj_path: Some(mmproj_path.display().to_string()),
            ..LoadConfig::default()
        };

        reg.load("with-mmproj", &model_path, config, &backend)
            .unwrap();

        let observed = observed.lock().unwrap().clone().unwrap();
        assert_eq!(
            observed.mmproj_path.as_deref(),
            Some(
                std::fs::canonicalize(&mmproj_path)
                    .unwrap()
                    .to_str()
                    .unwrap()
            )
        );
    }

    #[test]
    fn load_mmproj_path_rejects_non_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = make_gguf(&dir);
        let mmproj_path = dir.path().join("mmproj.bin");
        std::fs::write(&mmproj_path, b"projector").unwrap();

        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        let config = LoadConfig {
            mmproj_path: Some(mmproj_path.display().to_string()),
            ..LoadConfig::default()
        };

        let err = reg
            .load("bad-mmproj", &model_path, config, &backend)
            .unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::InvalidFormat(_))),
            "expected InvalidFormat, got: {err}"
        );
    }

    #[test]
    fn load_mmproj_path_respects_allowed_dirs() {
        let allowed_dir = tempfile::tempdir().unwrap();
        let blocked_dir = tempfile::tempdir().unwrap();
        let model_path = make_gguf(&allowed_dir);
        let mmproj_path = blocked_dir.path().join("mmproj.gguf");
        std::fs::write(&mmproj_path, b"projector").unwrap();
        let _allowed_dirs = AllowedDirsGuard::set(&[allowed_dir.path()]);

        let backend = NullBackend;
        let reg = ModelRegistry::new(16);
        let config = LoadConfig {
            mmproj_path: Some(mmproj_path.display().to_string()),
            ..LoadConfig::default()
        };

        let err = reg
            .load("blocked-mmproj", &model_path, config, &backend)
            .unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::PathNotAllowed(_))),
            "expected PathNotAllowed, got: {err}"
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

    struct BlockingLoadBackend {
        started: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
        next_handle: AtomicU64,
        unloads: Arc<AtomicU64>,
        fail_unloads: bool,
    }

    impl InferenceBackend for BlockingLoadBackend {
        fn load_model(
            &self,
            _path: &std::path::Path,
            _config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            self.started.wait();
            self.release.wait();
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed) + 1;
            Ok((
                ModelHandle(handle),
                ModelMetadata {
                    architecture: "blocking".into(),
                    n_layers: 1,
                    n_heads: 1,
                    n_kv_heads: 1,
                    embedding_dim: 1,
                    vocab_size: 1,
                    context_length: 1,
                    load_time_ms: 0,
                    peak_rss_bytes: 0,
                    resolved_backend: ax_serving_engine::BackendType::Cpu,
                },
            ))
        }

        fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
            self.unloads.fetch_add(1, Ordering::Relaxed);
            if self.fail_unloads {
                anyhow::bail!("simulated backend unload failure");
            }
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
            1
        }
    }

    struct BlockingFailFirstUnloadBackend {
        load_started: Arc<std::sync::Barrier>,
        release_first_load: Arc<std::sync::Barrier>,
        release_second_load: Arc<std::sync::Barrier>,
        next_handle: AtomicU64,
        unloads: AtomicU64,
        blocked_once: AtomicBool,
        unload_started: Arc<std::sync::Barrier>,
        release_unload: Arc<std::sync::Barrier>,
        immediate_path: PathBuf,
        first_path: PathBuf,
    }

    impl InferenceBackend for BlockingFailFirstUnloadBackend {
        fn load_model(
            &self,
            path: &Path,
            _config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            if path != self.immediate_path {
                self.load_started.wait();
                if path == self.first_path {
                    self.release_first_load.wait();
                } else {
                    self.release_second_load.wait();
                }
            }
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed) + 1;
            Ok((
                ModelHandle(handle),
                ModelMetadata {
                    architecture: "blocking-unload".into(),
                    n_layers: 1,
                    n_heads: 1,
                    n_kv_heads: 1,
                    embedding_dim: 1,
                    vocab_size: 1,
                    context_length: 1,
                    load_time_ms: 0,
                    peak_rss_bytes: 0,
                    resolved_backend: ax_serving_engine::BackendType::Cpu,
                },
            ))
        }

        fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
            self.unloads.fetch_add(1, Ordering::Relaxed);
            if !self.blocked_once.swap(true, Ordering::AcqRel) {
                self.unload_started.wait();
                self.release_unload.wait();
                anyhow::bail!("simulated final eviction failure");
            }
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
            1
        }
    }

    #[test]
    fn concurrent_loads_do_not_exceed_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("m1.gguf");
        let path2 = dir.path().join("m2.gguf");
        std::fs::write(&path1, b"dummy").unwrap();
        std::fs::write(&path2, b"dummy").unwrap();

        let backend = Arc::new(BlockingLoadBackend {
            started: Arc::new(std::sync::Barrier::new(2)),
            release: Arc::new(std::sync::Barrier::new(2)),
            next_handle: AtomicU64::new(0),
            unloads: Arc::new(AtomicU64::new(0)),
            fail_unloads: false,
        });
        let reg = ModelRegistry::new(1);

        let reg1 = reg.clone();
        let backend1 = Arc::clone(&backend);
        let path1_clone = path1.clone();
        let t1 = std::thread::spawn(move || {
            reg1.load(
                "first",
                &path1_clone,
                LoadConfig::default(),
                backend1.as_ref(),
            )
        });

        let reg2 = reg.clone();
        let backend2 = Arc::clone(&backend);
        let path2_clone = path2.clone();
        let t2 = std::thread::spawn(move || {
            reg2.load(
                "second",
                &path2_clone,
                LoadConfig::default(),
                backend2.as_ref(),
            )
        });

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        let successes = [r1.is_ok(), r2.is_ok()]
            .into_iter()
            .filter(|ok| *ok)
            .count();
        assert_eq!(successes, 1, "exactly one load must succeed at capacity 1");

        let capacity_errors = [r1.as_ref().err(), r2.as_ref().err()]
            .into_iter()
            .flatten()
            .filter(|err| {
                err.downcast_ref::<RegistryError>()
                    .is_some_and(|e| matches!(e, RegistryError::CapacityExceeded(_)))
            })
            .count();
        assert_eq!(
            capacity_errors, 1,
            "one concurrent load must fail with CapacityExceeded"
        );
        assert_eq!(reg.len(), 1, "registry must not exceed max_loaded_models");
        assert_eq!(
            backend.unloads.load(Ordering::Relaxed),
            1,
            "the losing concurrent load must clean up its backend handle"
        );
    }

    #[test]
    fn concurrent_loads_reject_busy_final_warm_pool_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("m1.gguf");
        let path2 = dir.path().join("m2.gguf");
        std::fs::write(&path1, b"dummy").unwrap();
        std::fs::write(&path2, b"dummy").unwrap();

        let backend = Arc::new(BlockingLoadBackend {
            started: Arc::new(std::sync::Barrier::new(2)),
            release: Arc::new(std::sync::Barrier::new(2)),
            next_handle: AtomicU64::new(0),
            unloads: Arc::new(AtomicU64::new(0)),
            fail_unloads: false,
        });
        let reg = ModelRegistry::new_with_warm_pool_size(16, Some(1));

        let reg1 = reg.clone();
        let backend1 = Arc::clone(&backend);
        let path1_clone = path1.clone();
        let t1 = std::thread::spawn(move || {
            reg1.load(
                "first",
                &path1_clone,
                LoadConfig::default(),
                backend1.as_ref(),
            )
        });

        let reg2 = reg.clone();
        let backend2 = Arc::clone(&backend);
        let path2_clone = path2.clone();
        let t2 = std::thread::spawn(move || {
            reg2.load(
                "second",
                &path2_clone,
                LoadConfig::default(),
                backend2.as_ref(),
            )
        });

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        let successes = [r1.is_ok(), r2.is_ok()]
            .into_iter()
            .filter(|ok| *ok)
            .count();
        assert_eq!(successes, 1, "exactly one concurrent load must succeed");

        let busy_errors = [r1.as_ref().err(), r2.as_ref().err()]
            .into_iter()
            .flatten()
            .filter(|err| {
                err.downcast_ref::<RegistryError>()
                    .is_some_and(|e| matches!(e, RegistryError::Busy(_)))
            })
            .count();
        assert_eq!(
            busy_errors, 1,
            "one concurrent load must fail instead of evicting an active model"
        );
        assert_eq!(
            reg.len(),
            1,
            "final insert must re-enforce AXS_MODEL_WARM_POOL_SIZE"
        );
        assert_eq!(
            backend.unloads.load(Ordering::Relaxed),
            1,
            "the rejected concurrent load must clean up its backend handle"
        );
    }

    #[test]
    fn concurrent_warm_pool_final_eviction_failure_rolls_back_new_load() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("m1.gguf");
        let path2 = dir.path().join("m2.gguf");
        std::fs::write(&path1, b"dummy").unwrap();
        std::fs::write(&path2, b"dummy").unwrap();

        let backend = Arc::new(BlockingLoadBackend {
            started: Arc::new(std::sync::Barrier::new(2)),
            release: Arc::new(std::sync::Barrier::new(2)),
            next_handle: AtomicU64::new(0),
            unloads: Arc::new(AtomicU64::new(0)),
            fail_unloads: true,
        });
        let reg = ModelRegistry::new_with_warm_pool_size(16, Some(1));

        let reg1 = reg.clone();
        let backend1 = Arc::clone(&backend);
        let path1_clone = path1.clone();
        let t1 = std::thread::spawn(move || {
            reg1.load(
                "first",
                &path1_clone,
                LoadConfig::default(),
                backend1.as_ref(),
            )
        });

        let reg2 = reg.clone();
        let backend2 = Arc::clone(&backend);
        let path2_clone = path2.clone();
        let t2 = std::thread::spawn(move || {
            reg2.load(
                "second",
                &path2_clone,
                LoadConfig::default(),
                backend2.as_ref(),
            )
        });

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        let successes = [r1.is_ok(), r2.is_ok()]
            .into_iter()
            .filter(|ok| *ok)
            .count();
        assert_eq!(
            successes, 1,
            "only the load that does not need failed final eviction may succeed"
        );
        assert_eq!(
            reg.len(),
            1,
            "failed final eviction must not leave warm pool above its configured size"
        );
        assert!(
            backend.unloads.load(Ordering::Relaxed) >= 1,
            "backend unload must have been attempted for final eviction"
        );
    }

    #[test]
    fn failed_final_warm_pool_eviction_does_not_expose_new_entry() {
        let dir = tempfile::tempdir().unwrap();
        let initial_path = dir.path().join("initial.gguf");
        let path1 = dir.path().join("m1.gguf");
        let path2 = dir.path().join("m2.gguf");
        std::fs::write(&initial_path, b"dummy").unwrap();
        std::fs::write(&path1, b"dummy").unwrap();
        std::fs::write(&path2, b"dummy").unwrap();

        let load_started = Arc::new(std::sync::Barrier::new(3));
        let release_first_load = Arc::new(std::sync::Barrier::new(2));
        let release_second_load = Arc::new(std::sync::Barrier::new(2));
        let unload_started = Arc::new(std::sync::Barrier::new(2));
        let release_unload = Arc::new(std::sync::Barrier::new(2));
        let backend = Arc::new(BlockingFailFirstUnloadBackend {
            load_started: Arc::clone(&load_started),
            release_first_load: Arc::clone(&release_first_load),
            release_second_load: Arc::clone(&release_second_load),
            next_handle: AtomicU64::new(0),
            unloads: AtomicU64::new(0),
            blocked_once: AtomicBool::new(false),
            unload_started: Arc::clone(&unload_started),
            release_unload: Arc::clone(&release_unload),
            immediate_path: std::fs::canonicalize(&initial_path).unwrap(),
            first_path: std::fs::canonicalize(&path1).unwrap(),
        });
        let reg = ModelRegistry::new_with_warm_pool_size(16, Some(2));

        let initial = reg
            .load(
                "initial",
                &initial_path,
                LoadConfig::default(),
                backend.as_ref(),
            )
            .unwrap();
        drop(initial);

        let reg_for_first = reg.clone();
        let backend_for_first = Arc::clone(&backend);
        let path1_for_load = path1.clone();
        let first_thread = std::thread::spawn(move || {
            reg_for_first
                .load(
                    "first",
                    &path1_for_load,
                    LoadConfig::default(),
                    backend_for_first.as_ref(),
                )
                .map(|_| ())
        });

        let reg_for_second = reg.clone();
        let backend_for_second = Arc::clone(&backend);
        let path2_for_load = path2.clone();
        let second_thread = std::thread::spawn(move || {
            reg_for_second
                .load(
                    "second",
                    &path2_for_load,
                    LoadConfig::default(),
                    backend_for_second.as_ref(),
                )
                .map(|_| ())
        });

        load_started.wait();
        release_first_load.wait();
        let first_result = first_thread.join().unwrap();
        assert!(first_result.is_ok());

        release_second_load.wait();
        unload_started.wait();

        let (tx, rx) = std::sync::mpsc::channel();
        let reg_for_get_second = reg.clone();
        let get_second_thread = std::thread::spawn(move || {
            tx.send(reg_for_get_second.get("second").is_some()).unwrap();
        });

        if let Ok(observed) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
            panic!("early registry read completed for uncommitted second load: {observed}");
        }

        release_unload.wait();
        let second_result = second_thread.join().unwrap();
        assert!(
            second_result.is_err(),
            "second load must fail when final eviction fails"
        );
        let observed = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        get_second_thread.join().unwrap();
        assert!(!observed, "failed second load must not become visible");
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn warm_pool_load_rejects_busy_eviction_candidate_without_unloading_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("m1.gguf");
        let path2 = dir.path().join("m2.gguf");
        std::fs::write(&path1, b"dummy").unwrap();
        std::fs::write(&path2, b"dummy").unwrap();

        let backend = CountingBackend::new();
        let reg = ModelRegistry::new_with_warm_pool_size(16, Some(1));

        let active = reg
            .load("first", &path1, LoadConfig::default(), &backend)
            .unwrap();
        let err = reg
            .load("second", &path2, LoadConfig::default(), &backend)
            .unwrap_err();

        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::Busy(_))),
            "expected Busy, got: {err}"
        );
        assert_eq!(
            backend.unloads.load(Ordering::Relaxed),
            0,
            "active warm-pool candidate must not be unloaded"
        );
        assert_eq!(
            backend.next_handle.load(Ordering::Relaxed),
            1,
            "new model must not be loaded when no safe warm-pool eviction exists"
        );
        assert!(reg.get("first").is_some());
        assert!(reg.get("second").is_none());

        drop(active);
        reg.load("second", &path2, LoadConfig::default(), &backend)
            .unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(backend.unloads.load(Ordering::Relaxed), 1);
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
    fn unload_busy_model_is_rejected_without_unloading_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = CountingBackend::new();
        let reg = ModelRegistry::new(16);

        reg.load("m", &path, LoadConfig::default(), &backend)
            .unwrap();
        let active = reg.get("m").unwrap();

        let err = reg.unload("m", &backend).unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::Busy(_))),
            "expected Busy, got: {err}"
        );
        assert_eq!(backend.unloads.load(Ordering::Relaxed), 0);
        assert!(reg.get("m").is_some());

        drop(active);
        reg.unload("m", &backend).unwrap();
        assert_eq!(backend.unloads.load(Ordering::Relaxed), 1);
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
    fn reload_busy_model_is_rejected_and_new_handle_is_cleaned_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = CountingBackend::new();
        let reg = ModelRegistry::new(16);

        let loaded = reg
            .load("m", &path, LoadConfig::default(), &backend)
            .unwrap();
        let original_handle = loaded.handle;
        drop(loaded);
        let active = reg.get("m").unwrap();

        let err = reg.reload("m", &backend).unwrap_err();
        assert!(
            err.downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::Busy(_))),
            "expected Busy, got: {err}"
        );
        assert_eq!(
            backend.unloads.load(Ordering::Relaxed),
            1,
            "unused new reload handle must be cleaned up"
        );
        assert_eq!(reg.get("m").unwrap().handle, original_handle);

        drop(active);
        reg.reload("m", &backend).unwrap();
        assert_eq!(backend.unloads.load(Ordering::Relaxed), 2);
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
        drop(entry);

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
        assert!(
            evicted.is_empty(),
            "recently accessed model must not be evicted"
        );
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
        drop(entry);

        // Even though the backend fails, idle_evict_pass must re-insert the
        // entry rather than leaking the handle.
        let evicted = reg.idle_evict_pass(&backend, 1);
        assert!(
            evicted.is_empty(),
            "failed eviction must not appear in return list"
        );
        assert!(
            reg.get("old").is_some(),
            "entry must be re-inserted after failed idle eviction"
        );
    }

    #[test]
    fn idle_evict_pass_skips_active_stale_entry_without_unloading_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_gguf(&dir);
        let backend = CountingBackend::new();
        let reg = ModelRegistry::new(16);
        reg.load("active", &path, LoadConfig::default(), &backend)
            .unwrap();

        let active = reg.get("active").unwrap();
        active.last_accessed_ms.store(0, Ordering::Relaxed);

        let evicted = reg.idle_evict_pass(&backend, 1);
        assert!(evicted.is_empty(), "active stale model must not be evicted");
        assert_eq!(backend.unloads.load(Ordering::Relaxed), 0);
        assert!(reg.get("active").is_some());

        drop(active);
        let entry = reg.get("active").unwrap();
        entry.last_accessed_ms.store(0, Ordering::Relaxed);
        drop(entry);
        let evicted = reg.idle_evict_pass(&backend, 1);
        assert_eq!(evicted, vec!["active"]);
        assert_eq!(backend.unloads.load(Ordering::Relaxed), 1);
    }
}
