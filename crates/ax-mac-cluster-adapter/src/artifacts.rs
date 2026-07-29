//! Shard-aware artifact plan verification and preparation.
//!
//! Downloads only the files required by one rank's bootstrap plan, verifies
//! digests, reuses already-valid files, and publishes the rank store
//! atomically. Never infers identity from filenames alone.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{ArtifactFileKind, ArtifactFilePlan, Digest};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use tokio::io::AsyncWriteExt;

use crate::coordinator::RankBootstrapPlan;

/// Result of preparing one rank's certified artifact subset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RankArtifactStore {
    pub rank: u16,
    pub generation: u64,
    pub manifest_digest: String,
    pub root: PathBuf,
    pub files: Vec<PreparedArtifactFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedArtifactFile {
    pub relative_path: String,
    pub digest: Digest,
    pub size_bytes: u64,
    pub kind: ArtifactFileKind,
    pub reused: bool,
}

/// Validate that a bootstrap plan's artifact subset is exact and safe.
pub fn validate_rank_artifact_subset(plan: &RankBootstrapPlan) -> Result<()> {
    if plan.artifacts.is_empty() {
        bail!("rank bootstrap plan must include artifacts");
    }
    let required: BTreeSet<_> = plan.rank.required_weight_files.iter().collect();
    let weight_digests: BTreeSet<_> = plan
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == ArtifactFileKind::Weight)
        .map(|artifact| &artifact.digest)
        .collect();
    if weight_digests != required {
        bail!("rank artifact subset must equal the required weight digests exactly");
    }
    let mut paths = BTreeSet::new();
    for artifact in &plan.artifacts {
        validate_relative_path(&artifact.relative_path)?;
        if !paths.insert(artifact.relative_path.as_str()) {
            bail!("rank artifact paths must be unique");
        }
        if artifact.size_bytes == 0 {
            bail!("artifact size must be greater than zero");
        }
    }
    Ok(())
}

/// Prepare one rank store under `root`, reusing verified files when possible.
///
/// `fetch` supplies raw bytes for a relative path that is not already verified
/// on disk. Publication is atomic: files land in a temp directory then rename.
pub async fn prepare_rank_artifacts<F, Fut>(
    plan: &RankBootstrapPlan,
    root: &Path,
    mut fetch: F,
) -> Result<RankArtifactStore>
where
    F: FnMut(&ArtifactFilePlan) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<u8>>>,
{
    validate_rank_artifact_subset(plan)?;
    let rank_root = root
        .join(plan.cluster_id.as_str())
        .join(format!("gen-{}", plan.generation))
        .join(format!("rank-{}", plan.rank.rank));
    let staging = rank_root.with_extension("staging");
    if staging.exists() {
        tokio::fs::remove_dir_all(&staging)
            .await
            .context("failed to clear previous staging directory")?;
    }
    tokio::fs::create_dir_all(&staging)
        .await
        .context("failed to create staging directory")?;

    let mut prepared = Vec::with_capacity(plan.artifacts.len());
    for artifact in &plan.artifacts {
        let final_path = rank_root.join(&artifact.relative_path);
        let staging_path = staging.join(&artifact.relative_path);
        if let Some(parent) = staging_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("failed to create artifact parent directory")?;
        }

        let reused = if final_path.is_file() {
            match verify_file(&final_path, artifact).await {
                Ok(()) => true,
                Err(_) => false,
            }
        } else {
            false
        };

        if reused {
            // Copy the verified final file into staging for atomic publication.
            if let Some(parent) = staging_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::copy(&final_path, &staging_path)
                .await
                .context("failed to stage reused artifact")?;
        } else {
            let bytes = fetch(artifact)
                .await
                .with_context(|| format!("failed to fetch {}", artifact.relative_path))?;
            if bytes.len() as u64 != artifact.size_bytes {
                bail!(
                    "artifact {} size mismatch: expected {} got {}",
                    artifact.relative_path,
                    artifact.size_bytes,
                    bytes.len()
                );
            }
            let digest = sha256_hex(&bytes);
            if digest != artifact.digest.as_str() {
                bail!(
                    "artifact {} digest mismatch: expected {} got {}",
                    artifact.relative_path,
                    artifact.digest,
                    digest
                );
            }
            let mut file = tokio::fs::File::create(&staging_path)
                .await
                .context("failed to create staged artifact file")?;
            file.write_all(&bytes)
                .await
                .context("failed to write staged artifact")?;
            file.flush().await?;
        }

        prepared.push(PreparedArtifactFile {
            relative_path: artifact.relative_path.clone(),
            digest: artifact.digest.clone(),
            size_bytes: artifact.size_bytes,
            kind: artifact.kind,
            reused,
        });
    }

    if rank_root.exists() {
        tokio::fs::remove_dir_all(&rank_root)
            .await
            .context("failed to clear previous rank artifact root")?;
    }
    if let Some(parent) = rank_root.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::rename(&staging, &rank_root)
        .await
        .context("failed to publish rank artifact store atomically")?;

    Ok(RankArtifactStore {
        rank: plan.rank.rank,
        generation: plan.generation,
        manifest_digest: plan.manifest_digest.to_string(),
        root: rank_root,
        files: prepared,
    })
}

async fn verify_file(path: &Path, artifact: &ArtifactFilePlan) -> Result<()> {
    let bytes = tokio::fs::read(path)
        .await
        .context("failed to read existing artifact")?;
    if bytes.len() as u64 != artifact.size_bytes {
        bail!("size mismatch");
    }
    let digest = sha256_hex(&bytes);
    if digest != artifact.digest.as_str() {
        bail!("digest mismatch");
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn validate_relative_path(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 512
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.split(['/', '\\']).any(|part| {
            part.is_empty() || matches!(part, "." | "..") || part.chars().any(char::is_control)
        })
    {
        bail!("artifact path is unsafe: {value}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ValidatedManifest;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fixture_plan() -> RankBootstrapPlan {
        let validated = ValidatedManifest::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../config/mac-cluster-manifest.example.json"),
        )
        .unwrap();
        let coordinator = crate::coordinator::ClusterCoordinator::new(
            validated,
            1,
            std::time::Duration::from_secs(30),
        );
        coordinator.rank_bootstrap_plan(0).unwrap()
    }

    #[tokio::test]
    async fn prepare_downloads_only_required_files_and_reuses_verified_copies() {
        let mut plan = fixture_plan();
        let mut bodies = Vec::new();
        for artifact in &mut plan.artifacts {
            let body = format!("payload:{}", artifact.relative_path).into_bytes();
            let digest = sha256_hex(&body);
            artifact.digest = Digest::new(digest).unwrap();
            artifact.size_bytes = body.len() as u64;
            bodies.push((artifact.relative_path.clone(), body));
        }
        plan.rank.required_weight_files = plan
            .artifacts
            .iter()
            .filter(|artifact| artifact.kind == ArtifactFileKind::Weight)
            .map(|artifact| artifact.digest.clone())
            .collect();
        validate_rank_artifact_subset(&plan).unwrap();

        let root = tempfile::tempdir().unwrap();
        let fetches = AtomicUsize::new(0);
        let store = prepare_rank_artifacts(&plan, root.path(), |artifact| {
            fetches.fetch_add(1, Ordering::SeqCst);
            let body = bodies
                .iter()
                .find(|(path, _)| path == &artifact.relative_path)
                .map(|(_, body)| body.clone())
                .unwrap();
            async move { Ok(body) }
        })
        .await
        .unwrap();
        assert_eq!(store.rank, 0);
        assert!(store.files.iter().all(|file| !file.reused));
        assert_eq!(fetches.load(Ordering::SeqCst), plan.artifacts.len());
        assert!(store.root.is_dir());

        let fetches = AtomicUsize::new(0);
        let reused = prepare_rank_artifacts(&plan, root.path(), |_artifact| {
            fetches.fetch_add(1, Ordering::SeqCst);
            async move { bail!("should not fetch when reusing") }
        })
        .await
        .unwrap();
        assert!(reused.files.iter().all(|file| file.reused));
        assert_eq!(fetches.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn unknown_weight_subset_fails_closed() {
        let mut plan = fixture_plan();
        plan.artifacts
            .retain(|artifact| artifact.kind != ArtifactFileKind::Weight);
        assert!(validate_rank_artifact_subset(&plan).is_err());
    }
}
