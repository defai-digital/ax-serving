//! Integrity-bound parallelism manifest loading.

use std::path::Path;

use anyhow::{Context, Result};
use ax_serving_protocol::{CompatibilityManifestDigest, ParallelismManifestV1};
use sha2::{Digest as _, Sha256};

#[derive(Clone, Debug)]
pub struct ValidatedManifest {
    pub manifest: ParallelismManifestV1,
    pub digest: CompatibilityManifestDigest,
}

impl ValidatedManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read Mac cluster manifest {}", path.display()))?;
        let manifest = serde_json::from_slice::<ParallelismManifestV1>(&bytes)
            .with_context(|| format!("invalid Mac cluster manifest JSON {}", path.display()))?;
        manifest
            .validate()
            .context("Mac cluster manifest validation failed")?;
        let raw_digest = format!("sha256:{}", hex::encode(Sha256::digest(&bytes)));
        let digest = CompatibilityManifestDigest::new(raw_digest)
            .context("failed to construct Mac cluster manifest digest")?;
        Ok(Self { manifest, digest })
    }
}
