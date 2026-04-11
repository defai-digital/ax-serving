use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use dashmap::DashMap;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

const RESTART_INTERRUPTED_ERROR: &str = "job interrupted by orchestrator restart before completion";
const DEFAULT_COMPLETED_JOB_TTL_SECS: u64 = 24 * 60 * 60;
const DEFAULT_MAX_COMPLETED_JOBS: usize = 1_000;

fn completed_job_ttl_ms() -> u64 {
    std::env::var("AXS_JOB_TTL_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(DEFAULT_COMPLETED_JOB_TTL_SECS)
        .saturating_mul(1_000)
}

fn max_completed_jobs() -> usize {
    std::env::var("AXS_JOB_MAX_COMPLETED")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_COMPLETED_JOBS)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub status: JobStatus,
    pub created_at_ms: u128,
    pub updated_at_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobBatchRecord {
    pub batch_id: String,
    pub status: JobStatus,
    pub total_jobs: usize,
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub succeeded_jobs: usize,
    pub failed_jobs: usize,
    pub created_at_ms: u128,
    pub updated_at_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<u128>,
    pub jobs: Vec<JobRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobStoreSummary {
    pub total_jobs: usize,
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub succeeded_jobs: usize,
    pub failed_jobs: usize,
    pub total_batches: usize,
    pub pruned_total: u64,
}

#[derive(Default)]
pub struct JobStore {
    entries: DashMap<String, JobRecord>,
    next_id: AtomicU64,
    next_batch_id: AtomicU64,
    pruned_total: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedJobStoreState {
    entries: Vec<JobRecord>,
    next_id: u64,
    next_batch_id: u64,
    pruned_total: u64,
}

impl JobStore {
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn create(
        &self,
        kind: &str,
        model_id: Option<String>,
        batch_id: Option<String>,
    ) -> JobRecord {
        let now_ms = now_ms();
        let id = format!(
            "job_{:016x}",
            self.next_id.fetch_add(1, Ordering::Relaxed) + 1
        );
        let record = JobRecord {
            id: id.clone(),
            kind: kind.to_string(),
            batch_id,
            model_id,
            status: JobStatus::Queued,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            completed_at_ms: None,
            response_status: None,
            content_type: None,
            result: None,
            error: None,
        };
        self.entries.insert(id, record.clone());
        record
    }

    pub fn get(&self, id: &str) -> Option<JobRecord> {
        self.entries.get(id).map(|entry| entry.clone())
    }

    pub fn delete(&self, id: &str) -> Option<JobRecord> {
        self.entries.remove(id).map(|(_, record)| record)
    }

    pub fn new_batch_id(&self) -> String {
        format!(
            "batch_{:016x}",
            self.next_batch_id.fetch_add(1, Ordering::Relaxed) + 1
        )
    }

    pub fn list_recent(&self, limit: usize, batch_id: Option<&str>) -> Vec<JobRecord> {
        let mut jobs = self
            .entries
            .iter()
            .filter(|entry| {
                batch_id.is_none_or(|batch_id| entry.batch_id.as_deref() == Some(batch_id))
            })
            .map(|entry| entry.clone())
            .collect::<Vec<_>>();
        jobs.sort_by(|a, b| {
            b.created_at_ms
                .cmp(&a.created_at_ms)
                .then_with(|| b.updated_at_ms.cmp(&a.updated_at_ms))
                .then_with(|| b.id.cmp(&a.id))
        });
        jobs.truncate(limit);
        jobs
    }

    pub fn get_batch(&self, batch_id: &str) -> Option<JobBatchRecord> {
        let mut jobs = self
            .entries
            .iter()
            .filter(|entry| entry.batch_id.as_deref() == Some(batch_id))
            .map(|entry| entry.clone())
            .collect::<Vec<_>>();
        if jobs.is_empty() {
            return None;
        }
        jobs.sort_by(|a, b| {
            a.created_at_ms
                .cmp(&b.created_at_ms)
                .then_with(|| a.updated_at_ms.cmp(&b.updated_at_ms))
                .then_with(|| a.id.cmp(&b.id))
        });

        let total_jobs = jobs.len();
        let queued_jobs = jobs
            .iter()
            .filter(|job| job.status == JobStatus::Queued)
            .count();
        let running_jobs = jobs
            .iter()
            .filter(|job| job.status == JobStatus::Running)
            .count();
        let succeeded_jobs = jobs
            .iter()
            .filter(|job| job.status == JobStatus::Succeeded)
            .count();
        let failed_jobs = jobs
            .iter()
            .filter(|job| job.status == JobStatus::Failed)
            .count();
        let status = if queued_jobs == total_jobs {
            JobStatus::Queued
        } else if queued_jobs > 0 || running_jobs > 0 {
            JobStatus::Running
        } else if failed_jobs > 0 {
            JobStatus::Failed
        } else {
            JobStatus::Succeeded
        };
        let created_at_ms = jobs
            .iter()
            .map(|job| job.created_at_ms)
            .min()
            .unwrap_or_default();
        let updated_at_ms = jobs
            .iter()
            .map(|job| job.updated_at_ms)
            .max()
            .unwrap_or(created_at_ms);
        let completed_at_ms = if queued_jobs == 0 && running_jobs == 0 {
            jobs.iter().filter_map(|job| job.completed_at_ms).max()
        } else {
            None
        };

        Some(JobBatchRecord {
            batch_id: batch_id.to_string(),
            status,
            total_jobs,
            queued_jobs,
            running_jobs,
            succeeded_jobs,
            failed_jobs,
            created_at_ms,
            updated_at_ms,
            completed_at_ms,
            jobs,
        })
    }

    pub fn list_batches_recent(&self, limit: usize) -> Vec<JobBatchRecord> {
        let mut batch_ids = FxHashSet::default();
        for entry in self.entries.iter() {
            if let Some(batch_id) = entry.batch_id.as_deref() {
                batch_ids.insert(batch_id.to_string());
            }
        }

        let mut batches = batch_ids
            .into_iter()
            .filter_map(|batch_id| self.get_batch(&batch_id))
            .collect::<Vec<_>>();
        batches.sort_by(|a, b| {
            b.created_at_ms
                .cmp(&a.created_at_ms)
                .then_with(|| b.updated_at_ms.cmp(&a.updated_at_ms))
                .then_with(|| b.batch_id.cmp(&a.batch_id))
        });
        batches.truncate(limit);
        batches
    }

    pub fn summary(&self) -> JobStoreSummary {
        let mut total_jobs = 0usize;
        let mut queued_jobs = 0usize;
        let mut running_jobs = 0usize;
        let mut succeeded_jobs = 0usize;
        let mut failed_jobs = 0usize;
        let mut batch_ids = FxHashSet::default();

        for entry in self.entries.iter() {
            total_jobs += 1;
            match entry.status {
                JobStatus::Queued => queued_jobs += 1,
                JobStatus::Running => running_jobs += 1,
                JobStatus::Succeeded => succeeded_jobs += 1,
                JobStatus::Failed => failed_jobs += 1,
            }
            if let Some(batch_id) = entry.batch_id.as_deref() {
                batch_ids.insert(batch_id.to_string());
            }
        }

        JobStoreSummary {
            total_jobs,
            queued_jobs,
            running_jobs,
            succeeded_jobs,
            failed_jobs,
            total_batches: batch_ids.len(),
            pruned_total: self.pruned_total.load(Ordering::Relaxed),
        }
    }

    pub fn prune_completed(&self, completed_ttl_ms: u64, max_completed: usize) -> usize {
        let now_ms = now_ms();
        let ttl_cutoff_ms = now_ms.saturating_sub(completed_ttl_ms as u128);

        let mut completed = self
            .entries
            .iter()
            .filter_map(|entry| match (entry.status, entry.completed_at_ms) {
                (JobStatus::Succeeded | JobStatus::Failed, Some(completed_at_ms)) => {
                    Some((entry.id.clone(), completed_at_ms))
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        let mut remove_ids = FxHashSet::default();
        for (id, completed_at_ms) in &completed {
            if *completed_at_ms <= ttl_cutoff_ms {
                remove_ids.insert(id.clone());
            }
        }

        completed.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let retained_completed = completed.len().saturating_sub(remove_ids.len());
        if retained_completed > max_completed {
            let overflow = retained_completed - max_completed;
            let mut overflow_removed = 0usize;
            for (id, _) in completed {
                if overflow_removed >= overflow {
                    break;
                }
                if remove_ids.insert(id) {
                    overflow_removed += 1;
                }
            }
        }

        let removed = remove_ids
            .into_iter()
            .filter(|id| self.entries.remove(id).is_some())
            .count();
        if removed > 0 {
            self.pruned_total
                .fetch_add(removed as u64, Ordering::Relaxed);
        }
        removed
    }

    pub fn mark_running(&self, id: &str) {
        if let Some(mut entry) = self.entries.get_mut(id) {
            entry.status = JobStatus::Running;
            entry.updated_at_ms = now_ms();
            entry.completed_at_ms = None;
            entry.error = None;
        }
    }

    pub fn mark_finished(
        &self,
        id: &str,
        status: JobStatus,
        response_status: u16,
        content_type: Option<String>,
        result: Option<serde_json::Value>,
        error: Option<String>,
    ) {
        if let Some(mut entry) = self.entries.get_mut(id) {
            let now_ms = now_ms();
            entry.status = status;
            entry.updated_at_ms = now_ms;
            entry.completed_at_ms = Some(now_ms);
            entry.response_status = Some(response_status);
            entry.content_type = content_type;
            entry.result = result;
            entry.error = error;
        }
        self.prune_completed(completed_job_ttl_ms(), max_completed_jobs());
    }

    pub fn persist_to_path(&self, path: &Path) -> anyhow::Result<()> {
        let state = self.persisted_state();
        let raw = serde_json::to_vec_pretty(&state).context("failed to encode job store state")?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create async job persistence directory {}",
                    parent.display()
                )
            })?;
        }
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, raw).with_context(|| {
            format!(
                "failed to write async job persistence snapshot {}",
                tmp_path.display()
            )
        })?;
        std::fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "failed to move async job persistence snapshot into place {}",
                path.display()
            )
        })?;
        Ok(())
    }

    pub fn restore_from_path(&self, path: &Path) -> anyhow::Result<()> {
        let raw = match std::fs::read(path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "failed to read async job persistence snapshot {}",
                        path.display()
                    )
                });
            }
        };
        let state: PersistedJobStoreState = serde_json::from_slice(&raw).with_context(|| {
            format!(
                "failed to decode async job persistence snapshot {}",
                path.display()
            )
        })?;
        self.restore_state(state);
        Ok(())
    }

    fn persisted_state(&self) -> PersistedJobStoreState {
        let mut entries = self
            .entries
            .iter()
            .map(|entry| entry.clone())
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| {
            a.created_at_ms
                .cmp(&b.created_at_ms)
                .then_with(|| a.updated_at_ms.cmp(&b.updated_at_ms))
                .then_with(|| a.id.cmp(&b.id))
        });
        PersistedJobStoreState {
            entries,
            next_id: self.next_id.load(Ordering::Relaxed),
            next_batch_id: self.next_batch_id.load(Ordering::Relaxed),
            pruned_total: self.pruned_total.load(Ordering::Relaxed),
        }
    }

    fn restore_state(&self, state: PersistedJobStoreState) {
        let restored_at_ms = now_ms();
        self.entries.clear();
        for mut record in state.entries {
            normalize_restored_job_record(&mut record, restored_at_ms);
            self.entries.insert(record.id.clone(), record);
        }
        self.next_id.store(state.next_id, Ordering::Relaxed);
        self.next_batch_id
            .store(state.next_batch_id, Ordering::Relaxed);
        self.pruned_total
            .store(state.pruned_total, Ordering::Relaxed);
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn normalize_restored_job_record(record: &mut JobRecord, restored_at_ms: u128) {
    if !matches!(record.status, JobStatus::Queued | JobStatus::Running) {
        return;
    }

    record.status = JobStatus::Failed;
    record.updated_at_ms = restored_at_ms.max(record.updated_at_ms);
    record.completed_at_ms = Some(restored_at_ms.max(record.created_at_ms));
    record.response_status = Some(503);
    record.content_type = Some("application/json".to_string());
    record.result = Some(serde_json::json!({
        "error": RESTART_INTERRUPTED_ERROR,
    }));
    record.error = Some(RESTART_INTERRUPTED_ERROR.to_string());
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{
        JobRecord, JobStatus, JobStore, PersistedJobStoreState, RESTART_INTERRUPTED_ERROR,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // ── basic CRUD ────────────────────────────────────────────────────────────

    #[test]
    fn create_returns_queued_record_and_stores_it() {
        let store = JobStore::default();
        let job = store.create("chat", Some("m1".into()), None);
        assert_eq!(job.status, JobStatus::Queued);
        assert!(!job.id.is_empty());
        assert_eq!(job.kind, "chat");
        assert_eq!(job.model_id.as_deref(), Some("m1"));
        assert!(job.batch_id.is_none());
        assert!(store.get(&job.id).is_some());
    }

    #[test]
    fn get_returns_none_for_unknown_id() {
        let store = JobStore::default();
        assert!(store.get("no_such_job").is_none());
    }

    #[test]
    fn delete_removes_and_returns_record() {
        let store = JobStore::default();
        let job = store.create("completions", None, None);
        let removed = store.delete(&job.id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, job.id);
        assert!(store.get(&job.id).is_none());
    }

    #[test]
    fn delete_returns_none_for_missing_id() {
        let store = JobStore::default();
        assert!(store.delete("ghost").is_none());
    }

    #[test]
    fn mark_running_transitions_status_and_clears_error() {
        let store = JobStore::default();
        let job = store.create("chat", None, None);
        store.mark_running(&job.id);
        let updated = store.get(&job.id).unwrap();
        assert_eq!(updated.status, JobStatus::Running);
        assert!(updated.completed_at_ms.is_none());
        assert!(updated.error.is_none());
    }

    #[test]
    fn mark_finished_succeeded_sets_all_completion_fields() {
        let store = JobStore::default();
        let job = store.create("chat", None, None);
        let result = serde_json::json!({"content": "hello"});
        store.mark_finished(
            &job.id,
            JobStatus::Succeeded,
            200,
            Some("application/json".into()),
            Some(result.clone()),
            None,
        );
        let done = store.get(&job.id).unwrap();
        assert_eq!(done.status, JobStatus::Succeeded);
        assert_eq!(done.response_status, Some(200));
        assert_eq!(done.content_type.as_deref(), Some("application/json"));
        assert_eq!(done.result, Some(result));
        assert!(done.error.is_none());
        assert!(done.completed_at_ms.is_some());
    }

    #[test]
    fn mark_finished_failed_stores_error_message() {
        let store = JobStore::default();
        let job = store.create("chat", None, None);
        store.mark_finished(
            &job.id,
            JobStatus::Failed,
            500,
            None,
            None,
            Some("backend error".into()),
        );
        let done = store.get(&job.id).unwrap();
        assert_eq!(done.status, JobStatus::Failed);
        assert_eq!(done.error.as_deref(), Some("backend error"));
    }

    // ── list_recent ────────────────────────────────────────────────────────────

    #[test]
    fn list_recent_returns_all_jobs_newest_first() {
        let store = JobStore::default();
        let j1 = store.create("chat", None, None);
        let j2 = store.create("embed", None, None);
        let jobs = store.list_recent(10, None);
        assert_eq!(jobs.len(), 2);
        // Newest (j2) first
        assert_eq!(jobs[0].id, j2.id);
        assert_eq!(jobs[1].id, j1.id);
    }

    #[test]
    fn list_recent_respects_limit() {
        let store = JobStore::default();
        for _ in 0..5 {
            store.create("chat", None, None);
        }
        assert_eq!(store.list_recent(3, None).len(), 3);
    }

    #[test]
    fn list_recent_filters_by_batch_id() {
        let store = JobStore::default();
        let batch_id = store.new_batch_id();
        let in_batch = store.create("chat", None, Some(batch_id.clone()));
        let _out = store.create("chat", None, None);
        let results = store.list_recent(10, Some(&batch_id));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, in_batch.id);
    }

    // ── get_batch ──────────────────────────────────────────────────────────────

    #[test]
    fn get_batch_returns_none_for_unknown_batch() {
        let store = JobStore::default();
        assert!(store.get_batch("batch_nonexistent").is_none());
    }

    #[test]
    fn get_batch_all_queued_reports_queued_status() {
        let store = JobStore::default();
        let bid = store.new_batch_id();
        store.create("chat", None, Some(bid.clone()));
        store.create("chat", None, Some(bid.clone()));
        let batch = store.get_batch(&bid).unwrap();
        assert_eq!(batch.status, JobStatus::Queued);
        assert_eq!(batch.total_jobs, 2);
        assert_eq!(batch.queued_jobs, 2);
    }

    #[test]
    fn get_batch_mixed_queued_and_running_reports_running_status() {
        let store = JobStore::default();
        let bid = store.new_batch_id();
        let j1 = store.create("chat", None, Some(bid.clone()));
        let j2 = store.create("chat", None, Some(bid.clone()));
        store.mark_running(&j1.id);
        // j2 is still Queued
        let _ = j2;
        let batch = store.get_batch(&bid).unwrap();
        assert_eq!(batch.status, JobStatus::Running);
        assert_eq!(batch.running_jobs, 1);
        assert_eq!(batch.queued_jobs, 1);
    }

    #[test]
    fn get_batch_all_succeeded_reports_succeeded_status() {
        let store = JobStore::default();
        let bid = store.new_batch_id();
        let j1 = store.create("chat", None, Some(bid.clone()));
        let j2 = store.create("chat", None, Some(bid.clone()));
        store.mark_finished(&j1.id, JobStatus::Succeeded, 200, None, None, None);
        store.mark_finished(&j2.id, JobStatus::Succeeded, 200, None, None, None);
        let batch = store.get_batch(&bid).unwrap();
        assert_eq!(batch.status, JobStatus::Succeeded);
        assert_eq!(batch.succeeded_jobs, 2);
        assert!(batch.completed_at_ms.is_some());
    }

    #[test]
    fn get_batch_any_failed_reports_failed_status() {
        let store = JobStore::default();
        let bid = store.new_batch_id();
        let j1 = store.create("chat", None, Some(bid.clone()));
        let j2 = store.create("chat", None, Some(bid.clone()));
        store.mark_finished(&j1.id, JobStatus::Succeeded, 200, None, None, None);
        store.mark_finished(&j2.id, JobStatus::Failed, 500, None, None, Some("err".into()));
        let batch = store.get_batch(&bid).unwrap();
        assert_eq!(batch.status, JobStatus::Failed);
        assert_eq!(batch.failed_jobs, 1);
        assert_eq!(batch.succeeded_jobs, 1);
    }

    // ── summary ────────────────────────────────────────────────────────────────

    #[test]
    fn summary_counts_each_status_correctly() {
        let store = JobStore::default();
        let bid = store.new_batch_id();
        let j1 = store.create("chat", None, Some(bid.clone()));
        let j2 = store.create("chat", None, Some(bid.clone()));
        let j3 = store.create("chat", None, None);
        store.mark_running(&j1.id);
        store.mark_finished(&j2.id, JobStatus::Succeeded, 200, None, None, None);
        let _ = j3; // stays Queued

        let s = store.summary();
        assert_eq!(s.total_jobs, 3);
        assert_eq!(s.queued_jobs, 1);
        assert_eq!(s.running_jobs, 1);
        assert_eq!(s.succeeded_jobs, 1);
        assert_eq!(s.failed_jobs, 0);
        assert_eq!(s.total_batches, 1);
    }

    // ── new_batch_id uniqueness ────────────────────────────────────────────────

    #[test]
    fn new_batch_id_generates_unique_ids() {
        let store = JobStore::default();
        let ids: Vec<_> = (0..5).map(|_| store.new_batch_id()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 5, "batch IDs must be unique");
    }

    // ── persist / restore round-trip ──────────────────────────────────────────

    #[test]
    fn persist_to_path_and_restore_from_path_round_trips_all_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jobs.json");

        let original = JobStore::default();
        let j1 = original.create("chat", Some("m1".into()), None);
        let j2 = original.create("embed", None, None);
        original.mark_running(&j1.id);
        original.mark_finished(&j2.id, JobStatus::Succeeded, 200, None, None, None);
        original.persist_to_path(&path).unwrap();

        let restored = JobStore::default();
        restored.restore_from_path(&path).unwrap();

        // j1 was Running → should be normalized to Failed (restart-interrupted)
        let r1 = restored.get(&j1.id).unwrap();
        assert_eq!(r1.status, JobStatus::Failed);
        assert_eq!(r1.response_status, Some(503));

        // j2 was Succeeded → status preserved as-is
        let r2 = restored.get(&j2.id).unwrap();
        assert_eq!(r2.status, JobStatus::Succeeded);
        assert_eq!(r2.response_status, Some(200));
    }

    #[test]
    fn restore_from_path_is_noop_when_file_missing() {
        let store = JobStore::default();
        let result = store.restore_from_path(std::path::Path::new("/nonexistent/jobs.json"));
        assert!(result.is_ok());
        assert_eq!(store.summary().total_jobs, 0);
    }

    #[test]
    fn prune_completed_removes_expired_jobs_but_keeps_active_jobs() {
        let store = JobStore::default();
        let expired = store.create("completions", Some("expired".into()), None);
        store.mark_finished(&expired.id, JobStatus::Succeeded, 200, None, None, None);
        let active = store.create("completions", Some("active".into()), None);
        store.mark_running(&active.id);

        let expired_completed_at_ms = store
            .entries
            .get(&expired.id)
            .and_then(|entry| entry.completed_at_ms)
            .unwrap();
        if let Some(mut entry) = store.entries.get_mut(&expired.id) {
            entry.completed_at_ms = Some(expired_completed_at_ms.saturating_sub(5_000));
            entry.updated_at_ms = expired_completed_at_ms.saturating_sub(5_000);
        }

        let removed = store.prune_completed(1_000, 10);
        assert_eq!(removed, 1);
        assert!(store.get(&expired.id).is_none());
        assert!(store.get(&active.id).is_some());
        assert_eq!(store.summary().pruned_total, 1);
    }

    #[test]
    fn prune_completed_enforces_max_completed_cap() {
        let store = JobStore::default();
        let oldest = store.create("completions", Some("m1".into()), None);
        store.mark_finished(&oldest.id, JobStatus::Succeeded, 200, None, None, None);
        let middle = store.create("completions", Some("m2".into()), None);
        store.mark_finished(&middle.id, JobStatus::Succeeded, 200, None, None, None);
        let newest = store.create("completions", Some("m3".into()), None);
        store.mark_finished(
            &newest.id,
            JobStatus::Failed,
            500,
            None,
            None,
            Some("boom".into()),
        );

        let newest_completed_at_ms = store
            .entries
            .get(&newest.id)
            .and_then(|entry| entry.completed_at_ms)
            .unwrap();
        if let Some(mut entry) = store.entries.get_mut(&oldest.id) {
            entry.completed_at_ms = Some(newest_completed_at_ms.saturating_sub(3));
        }
        if let Some(mut entry) = store.entries.get_mut(&middle.id) {
            entry.completed_at_ms = Some(newest_completed_at_ms.saturating_sub(2));
        }
        if let Some(mut entry) = store.entries.get_mut(&newest.id) {
            entry.completed_at_ms = Some(newest_completed_at_ms.saturating_sub(1));
        }

        let removed = store.prune_completed(60_000, 2);
        assert_eq!(removed, 1);
        assert!(store.get(&oldest.id).is_none());
        assert!(store.get(&middle.id).is_some());
        assert!(store.get(&newest.id).is_some());
        assert_eq!(store.summary().total_jobs, 2);
        assert_eq!(store.summary().pruned_total, 1);
    }

    #[test]
    fn mark_finished_auto_prunes_completed_jobs() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_JOB_TTL_SECS", "3600") };
        unsafe { std::env::set_var("AXS_JOB_MAX_COMPLETED", "2") };

        let store = JobStore::default();
        let oldest = store.create("completions", Some("m1".into()), None);
        store.mark_finished(&oldest.id, JobStatus::Succeeded, 200, None, None, None);
        let middle = store.create("completions", Some("m2".into()), None);
        store.mark_finished(&middle.id, JobStatus::Succeeded, 200, None, None, None);
        let newest = store.create("completions", Some("m3".into()), None);
        store.mark_finished(&newest.id, JobStatus::Succeeded, 200, None, None, None);

        assert!(store.get(&oldest.id).is_none());
        assert!(store.get(&middle.id).is_some());
        assert!(store.get(&newest.id).is_some());
        assert_eq!(store.summary().pruned_total, 1);

        unsafe { std::env::remove_var("AXS_JOB_TTL_SECS") };
        unsafe { std::env::remove_var("AXS_JOB_MAX_COMPLETED") };
    }

    #[test]
    fn restore_state_marks_queued_jobs_as_restart_interrupted_failures() {
        let store = JobStore::default();
        let state = PersistedJobStoreState {
            entries: vec![JobRecord {
                id: "job_queued".into(),
                kind: "completions".into(),
                batch_id: Some("batch_1".into()),
                model_id: Some("model-a".into()),
                status: JobStatus::Queued,
                created_at_ms: 100,
                updated_at_ms: 100,
                completed_at_ms: None,
                response_status: None,
                content_type: None,
                result: None,
                error: None,
            }],
            next_id: 5,
            next_batch_id: 7,
            pruned_total: 3,
        };

        store.restore_state(state);

        let restored = store
            .get("job_queued")
            .expect("queued job should be restored");
        assert_eq!(restored.status, JobStatus::Failed);
        assert_eq!(restored.response_status, Some(503));
        assert_eq!(restored.error.as_deref(), Some(RESTART_INTERRUPTED_ERROR));
        assert_eq!(
            restored.result,
            Some(serde_json::json!({ "error": RESTART_INTERRUPTED_ERROR }))
        );
        assert!(restored.completed_at_ms.is_some());
        assert!(restored.updated_at_ms >= restored.created_at_ms);
        assert_eq!(store.summary().failed_jobs, 1);
        assert_eq!(store.summary().queued_jobs, 0);
    }

    #[test]
    fn restore_state_marks_running_jobs_as_restart_interrupted_failures() {
        let store = JobStore::default();
        let state = PersistedJobStoreState {
            entries: vec![JobRecord {
                id: "job_running".into(),
                kind: "completions".into(),
                batch_id: None,
                model_id: Some("model-b".into()),
                status: JobStatus::Running,
                created_at_ms: 200,
                updated_at_ms: 250,
                completed_at_ms: None,
                response_status: None,
                content_type: Some("application/json".into()),
                result: Some(serde_json::json!({ "partial": true })),
                error: None,
            }],
            next_id: 9,
            next_batch_id: 2,
            pruned_total: 0,
        };

        store.restore_state(state);

        let restored = store
            .get("job_running")
            .expect("running job should be restored");
        assert_eq!(restored.status, JobStatus::Failed);
        assert_eq!(restored.response_status, Some(503));
        assert_eq!(restored.content_type.as_deref(), Some("application/json"));
        assert_eq!(restored.error.as_deref(), Some(RESTART_INTERRUPTED_ERROR));
        assert_eq!(
            restored.result,
            Some(serde_json::json!({ "error": RESTART_INTERRUPTED_ERROR }))
        );
        assert!(restored.completed_at_ms.is_some());
        assert!(restored.updated_at_ms >= 250);
        assert_eq!(store.summary().running_jobs, 0);
        assert_eq!(store.summary().failed_jobs, 1);
    }

    // ── list_batches_recent ───────────────────────────────────────────────────

    #[test]
    fn list_batches_recent_returns_all_batches_when_below_limit() {
        let store = JobStore::default();
        let batch_a = store.new_batch_id();
        let batch_b = store.new_batch_id();

        store.create("chat", Some("m1".into()), Some(batch_a.clone()));
        store.create("chat", Some("m1".into()), Some(batch_b.clone()));

        let batches = store.list_batches_recent(10);
        assert_eq!(batches.len(), 2);
        let ids: std::collections::HashSet<_> = batches.iter().map(|b| b.batch_id.clone()).collect();
        assert!(ids.contains(&batch_a));
        assert!(ids.contains(&batch_b));
    }

    #[test]
    fn list_batches_recent_respects_limit() {
        let store = JobStore::default();
        for _ in 0..5 {
            let batch_id = store.new_batch_id();
            store.create("chat", Some("m1".into()), Some(batch_id));
        }

        let batches = store.list_batches_recent(3);
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn list_batches_recent_empty_store_returns_empty() {
        let store = JobStore::default();
        assert!(store.list_batches_recent(10).is_empty());
    }

    #[test]
    fn list_batches_recent_jobs_without_batch_id_are_not_counted() {
        let store = JobStore::default();
        // Two jobs without a batch ID.
        store.create("chat", Some("m1".into()), None);
        store.create("chat", Some("m1".into()), None);
        // One job with a batch ID.
        let batch_id = store.new_batch_id();
        store.create("chat", Some("m1".into()), Some(batch_id));

        let batches = store.list_batches_recent(10);
        assert_eq!(batches.len(), 1);
    }
}
