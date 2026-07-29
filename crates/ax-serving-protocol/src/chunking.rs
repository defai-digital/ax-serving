//! Profile-driven pipeline/tensor chunking contracts.
//!
//! Chunking profiles are measured execution evidence used by AX Engine. The
//! gateway and adapter only validate identity and bounds; they never execute
//! collectives or inspect activations.

use serde::{Deserialize, Serialize};

use crate::Digest;

/// Optional measured profile referenced by tensor/hybrid plans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkingProfileV1 {
    pub schema_version: u32,
    pub profile_digest: Digest,
    pub model_artifact_digest: Digest,
    pub stages: Vec<StageChunkProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageChunkProfile {
    pub stage: u16,
    pub preferred_chunk_tokens: u32,
    pub max_chunk_tokens: u32,
    pub measured_compute_micros: u64,
    pub measured_transfer_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChunkingProfileError {
    #[error("unsupported chunking profile schema version {0}")]
    UnsupportedSchema(u32),
    #[error("chunking profile must contain at least one stage")]
    EmptyStages,
    #[error("chunk token limits must be greater than zero and preferred <= max")]
    InvalidChunkLimits,
    #[error("stage ids must be contiguous from zero")]
    NonContiguousStages,
    #[error("measured stage durations must be greater than zero")]
    ZeroMeasurement,
}

impl ChunkingProfileV1 {
    pub fn validate(&self, expected_pp_size: u16) -> Result<(), ChunkingProfileError> {
        if self.schema_version != 1 {
            return Err(ChunkingProfileError::UnsupportedSchema(self.schema_version));
        }
        if self.stages.is_empty() {
            return Err(ChunkingProfileError::EmptyStages);
        }
        if self.stages.len() != usize::from(expected_pp_size) {
            return Err(ChunkingProfileError::NonContiguousStages);
        }
        for (index, stage) in self.stages.iter().enumerate() {
            if usize::from(stage.stage) != index {
                return Err(ChunkingProfileError::NonContiguousStages);
            }
            if stage.preferred_chunk_tokens == 0
                || stage.max_chunk_tokens == 0
                || stage.preferred_chunk_tokens > stage.max_chunk_tokens
            {
                return Err(ChunkingProfileError::InvalidChunkLimits);
            }
            if stage.measured_compute_micros == 0 || stage.measured_transfer_micros == 0 {
                return Err(ChunkingProfileError::ZeroMeasurement);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Digest {
        Digest::new(format!("sha256:{}", byte.to_string().repeat(64))).unwrap()
    }

    fn profile() -> ChunkingProfileV1 {
        ChunkingProfileV1 {
            schema_version: 1,
            profile_digest: digest('a'),
            model_artifact_digest: digest('b'),
            stages: vec![
                StageChunkProfile {
                    stage: 0,
                    preferred_chunk_tokens: 32,
                    max_chunk_tokens: 64,
                    measured_compute_micros: 100,
                    measured_transfer_micros: 20,
                },
                StageChunkProfile {
                    stage: 1,
                    preferred_chunk_tokens: 32,
                    max_chunk_tokens: 64,
                    measured_compute_micros: 120,
                    measured_transfer_micros: 25,
                },
            ],
        }
    }

    #[test]
    fn valid_profile_matches_pipeline_width() {
        profile().validate(2).unwrap();
    }

    #[test]
    fn invalid_limits_and_width_fail_closed() {
        let mut bad = profile();
        bad.stages[0].preferred_chunk_tokens = 128;
        assert_eq!(
            bad.validate(2),
            Err(ChunkingProfileError::InvalidChunkLimits)
        );
        assert_eq!(
            profile().validate(3),
            Err(ChunkingProfileError::NonContiguousStages)
        );
    }
}
