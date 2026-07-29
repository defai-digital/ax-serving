//! Retained operational evidence hooks for Mac cluster certification.
//!
//! Records are prompt-free and tensor-free. Physical multi-Mac soak output is
//! external evidence; this module defines the in-repo schema and capture API
//! so load/fault/restart/soak runs can be retained with stable identity.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use ax_serving_protocol::{CompatibilityManifestDigest, DomainId};

/// Kind of retained operational evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Load,
    Fault,
    Restart,
    Soak,
    Security,
    Value,
}

/// One bounded evidence record for a certified topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterEvidenceRecord {
    pub kind: EvidenceKind,
    pub domain_id: DomainId,
    pub generation: u64,
    pub manifest_digest: CompatibilityManifestDigest,
    pub topology_label: String,
    pub ax_engine_build: String,
    pub transport_kind: String,
    pub duration_secs: u64,
    pub success: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, u64>,
    #[serde(with = "time::serde::rfc3339")]
    pub recorded_at: OffsetDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvidenceError {
    #[error("evidence field '{0}' is empty or unbounded")]
    InvalidField(&'static str),
    #[error("evidence metric key is empty, too long, or invalid")]
    InvalidMetricKey,
    #[error("evidence journal exceeded its retention bound")]
    JournalFull,
}

impl ClusterEvidenceRecord {
    pub fn validate(&self) -> Result<(), EvidenceError> {
        for (field, value) in [
            ("topology_label", self.topology_label.as_str()),
            ("ax_engine_build", self.ax_engine_build.as_str()),
            ("transport_kind", self.transport_kind.as_str()),
        ] {
            if value.is_empty() || value.len() > 128 {
                return Err(EvidenceError::InvalidField(field));
            }
        }
        if self.generation == 0 {
            return Err(EvidenceError::InvalidField("generation"));
        }
        if self.metrics.len() > 32 {
            return Err(EvidenceError::InvalidField("metrics"));
        }
        for key in self.metrics.keys() {
            if key.is_empty()
                || key.len() > 64
                || !key
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
            {
                return Err(EvidenceError::InvalidMetricKey);
            }
        }
        Ok(())
    }
}

/// In-memory bounded journal used by adapters and tests.
#[derive(Debug, Default)]
pub struct EvidenceJournal {
    limit: usize,
    records: Mutex<Vec<ClusterEvidenceRecord>>,
}

impl EvidenceJournal {
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit: limit.max(1),
            records: Mutex::new(Vec::new()),
        })
    }

    pub fn record(&self, record: ClusterEvidenceRecord) -> Result<(), EvidenceError> {
        record.validate()?;
        let mut records = self
            .records
            .lock()
            .expect("evidence journal mutex is not poisoned");
        if records.len() >= self.limit {
            records.remove(0);
        }
        records.push(record);
        Ok(())
    }

    pub fn list(&self, kind: Option<EvidenceKind>) -> Vec<ClusterEvidenceRecord> {
        let records = self
            .records
            .lock()
            .expect("evidence journal mutex is not poisoned");
        records
            .iter()
            .filter(|record| kind.is_none_or(|wanted| record.kind == wanted))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_serving_protocol::{CompatibilityManifestDigest, DomainId};

    fn sample(kind: EvidenceKind) -> ClusterEvidenceRecord {
        ClusterEvidenceRecord {
            kind,
            domain_id: DomainId::new("mac-cluster-main").unwrap(),
            generation: 2,
            manifest_digest: CompatibilityManifestDigest::new(format!("sha256:{}", "a".repeat(64)))
                .unwrap(),
            topology_label: "2x-m3-ultra-tb5".into(),
            ax_engine_build: "6.12.0+deadbeef".into(),
            transport_kind: "tcp".into(),
            duration_secs: 3_600,
            success: true,
            metrics: BTreeMap::from([("ttft_p95_ms".into(), 120), ("tok_per_s".into(), 18)]),
            recorded_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn journal_retains_load_fault_restart_and_soak_hooks() {
        let journal = EvidenceJournal::new(8);
        for kind in [
            EvidenceKind::Load,
            EvidenceKind::Fault,
            EvidenceKind::Restart,
            EvidenceKind::Soak,
        ] {
            journal.record(sample(kind)).unwrap();
        }
        assert_eq!(journal.list(None).len(), 4);
        assert_eq!(journal.list(Some(EvidenceKind::Soak)).len(), 1);
        assert!(journal.list(Some(EvidenceKind::Soak))[0].success);
    }

    #[test]
    fn evidence_rejects_unbounded_or_secret_shaped_fields() {
        let mut bad = sample(EvidenceKind::Load);
        bad.topology_label.clear();
        assert!(bad.validate().is_err());
        let mut bad_metric = sample(EvidenceKind::Fault);
        bad_metric.metrics.insert("not valid".into(), 1);
        assert!(bad_metric.validate().is_err());
    }
}
