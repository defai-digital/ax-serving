//! Model → worker secondary membership index.
//!
//! # Purpose
//!
//! Selection hot paths (`dispatch_workers_filtered*`, `eligible_model_endpoints`)
//! previously full-scanned `inner` on every request. This index answers
//! "which workers advertise model M?" so candidates are O(k) advertisers of M
//! rather than O(N) workers.
//!
//! # Semantics
//!
//! - **Membership only**, not eligibility. Live filters still fail-closed.
//! - Keys are post-normalization `capabilities.models` (the same set
//!   `dispatch_filter_matches` uses).
//! - Never treat index membership alone as ready without a live `inner` lookup
//!   and full filter application.
//!
//! # Concurrency
//!
//! Model-set diffs use **add-new-before-remove-old** so concurrent selection
//! may briefly over-include (safe: live filters drop mismatches) rather than
//! under-route (silent 503). Selection clones candidate `WorkerId`s from the
//! index before looking up `inner`.
//!
//! **Same-worker reindex runs under the `inner` entry write guard** so the
//! captured (old, new) pair is applied in the same critical section as the
//! entry mutation. Callers use [`reindex_into`] with a cloned `by_model` `Arc`
//! while still holding the entry guard — concurrent heartbeats/registers on
//! the same `WorkerId` cannot reorder diffs (DashMap per-key serialization).
//!
//! **Unindex is live-state aware** with a post-remove repair: never strip a
//! model a re-registered entry still advertises; if a strip races a reinsert,
//! membership is restored from the live entry.
//!
//! # Kill-switch
//!
//! Env-only (no `OrchestratorConfig` field):
//! - `AXS_WORKER_MODEL_INDEX` default **enabled** (unset / empty / `1` / `true` / `on`)
//! - Set to `0` / `false` / `off` / `no` to force full-scan selection for safety

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;

use super::types::{WorkerEntry, WorkerId};
use super::WorkerRegistry;

/// Whether the model→worker secondary index is used for selection.
///
/// Default **on**. Set `AXS_WORKER_MODEL_INDEX=0` (or `false`/`off`/`no`) to
/// force the pre-index full-scan path. Read on each selection so operators can
/// flip the kill-switch without restart (tests also rely on this).
pub(crate) fn worker_model_index_enabled() -> bool {
    match std::env::var("AXS_WORKER_MODEL_INDEX") {
        Ok(value) => {
            let v = value.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => true,
    }
}

/// Apply a model-set membership diff for `id` into `by_model`.
///
/// **Add new model keys before remove old** (prefer brief over-include).
/// Empty index sets are pruned when the last worker leaves a model key.
///
/// # Concurrency
///
/// Call this **while holding the `inner` write guard for `id`** (register
/// `and_modify` / `or_insert_with`, or heartbeat `get_mut`) so concurrent
/// same-worker mutations cannot apply diffs out of order. The free function
/// form takes `by_model` separately so the entry guard can keep borrowing
/// `inner` while the index is updated.
pub(crate) fn reindex_into(
    by_model: &DashMap<String, HashSet<WorkerId>>,
    id: WorkerId,
    old_models: &[String],
    new_models: &[String],
) {
    let old: HashSet<&str> = old_models.iter().map(String::as_str).collect();
    let new: HashSet<&str> = new_models.iter().map(String::as_str).collect();

    // Phase 1: add memberships not already present under old.
    for model in new.iter().filter(|m| !old.contains(*m)) {
        by_model
            .entry((*model).to_string())
            .or_default()
            .insert(id);
    }

    // Phase 2: remove memberships no longer advertised.
    // Safe under entry-write serialization: live models == new_models.
    for model in old.iter().filter(|m| !new.contains(*m)) {
        force_remove_worker_from_model(by_model, id, model);
    }
}

/// Drop model memberships for `id` (evict / remove paths).
///
/// Live-state aware: if `id` was re-registered and still advertises a model,
/// that membership is **not** stripped. After a force-remove, membership is
/// **repaired** if a concurrent re-register re-advertised the model between
/// the live check and the strip (check → remove → re-register → repair).
pub(crate) fn unindex_into(
    inner: &DashMap<WorkerId, WorkerEntry>,
    by_model: &DashMap<String, HashSet<WorkerId>>,
    id: WorkerId,
    models: &[String],
) {
    for model in models {
        // Do not strip if the live entry still advertises this model
        // (re-register / heartbeat after remove of a previous incarnation).
        if live_advertises(inner, id, model) {
            continue;
        }
        force_remove_worker_from_model(by_model, id, model);
        // Repair race: remove completed just as a re-register re-advertised `model`.
        if live_advertises(inner, id, model) {
            by_model
                .entry(model.clone())
                .or_default()
                .insert(id);
        }
    }
}

fn live_advertises(
    inner: &DashMap<WorkerId, WorkerEntry>,
    id: WorkerId,
    model: &str,
) -> bool {
    inner
        .get(&id)
        .is_some_and(|entry| entry.capabilities.models.iter().any(|m| m == model))
}

fn force_remove_worker_from_model(
    by_model: &DashMap<String, HashSet<WorkerId>>,
    id: WorkerId,
    model: &str,
) {
    let should_prune = if let Some(mut set) = by_model.get_mut(model) {
        set.remove(&id);
        set.is_empty()
    } else {
        false
    };
    if should_prune {
        // Race-safe prune: only remove if still empty when we hold the shard.
        by_model.remove_if(model, |_, set| set.is_empty());
    }
}

impl WorkerRegistry {
    /// Update membership for `id` after `capabilities.models` changed.
    ///
    /// Prefer [`reindex_into`] while holding the entry guard. This wrapper is
    /// for call sites that already serialized the mutation.
    pub(crate) fn reindex_worker(&self, id: WorkerId, old_models: &[String], new_models: &[String]) {
        reindex_into(&self.by_model, id, old_models, new_models);
    }

    /// Drop all model memberships for `id` (evict / remove paths).
    pub(crate) fn unindex_worker(&self, id: WorkerId, models: &[String]) {
        unindex_into(&self.inner, &self.by_model, id, models);
    }

    /// Clone of the index map for use while an `inner` entry guard is held.
    pub(crate) fn by_model_handle(&self) -> Arc<DashMap<String, HashSet<WorkerId>>> {
        Arc::clone(&self.by_model)
    }

    /// Clone candidate worker ids that advertise `model_id`.
    ///
    /// Callers must look up live entries in `inner` and apply full filters.
    /// Returns `None` when the kill-switch forces full scan (caller iterates
    /// `inner` instead).
    pub(crate) fn indexed_candidate_ids(&self, model_id: &str) -> Option<Vec<WorkerId>> {
        if !worker_model_index_enabled() {
            return None;
        }
        Some(
            self.by_model
                .get(model_id)
                .map(|set| set.iter().copied().collect())
                .unwrap_or_default(),
        )
    }

    /// Test/debug helper: assert `by_model` matches post-normalization
    /// `capabilities.models` across all live entries, with no empty sets.
    #[cfg(test)]
    pub(crate) fn assert_index_consistent(&self) {
        let mut expected: HashMap<String, HashSet<WorkerId>> = HashMap::new();
        for item in self.inner.iter() {
            let entry = item.value();
            for model in &entry.capabilities.models {
                expected.entry(model.clone()).or_default().insert(entry.id);
            }
        }

        let mut observed: HashMap<String, HashSet<WorkerId>> = HashMap::new();
        for item in self.by_model.iter() {
            let model = item.key().clone();
            let set = item.value().clone();
            assert!(
                !set.is_empty(),
                "by_model key {model:?} has empty worker set (should be pruned)"
            );
            observed.insert(model, set);
        }

        assert_eq!(
            expected, observed,
            "by_model index diverged from live capabilities.models"
        );
    }
}
