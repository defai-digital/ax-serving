//! Bounded pipeline micro-batch and asynchronous stage-transfer contracts.
//!
//! These types describe AX Engine-owned PP scheduling constraints. They never
//! carry activations, KV blocks, prompts, or transport credentials. Gateway
//! code may validate or retain them; it must not implement stage execution.

use serde::{Deserialize, Serialize};

use crate::Digest;

/// Immutable micro-batching contract for one certified pipeline generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicroBatchContract {
    /// Maximum number of micro-batches admitted for one logical request.
    pub max_micro_batches: u16,
    /// Maximum concurrently in-flight micro-batches across the pipeline.
    pub max_in_flight: u16,
    /// Whether cancellation tombstones must reach every stage before buffer reuse.
    pub ordered_cancellation: bool,
    /// Whether per-request micro-batch sequence IDs are required.
    pub require_sequence_ids: bool,
}

impl MicroBatchContract {
    /// Validate a bounded micro-batch contract without contacting ranks.
    pub fn validate(&self) -> Result<(), PipelineContractError> {
        if self.max_micro_batches == 0 {
            return Err(PipelineContractError::ZeroMicroBatches);
        }
        if self.max_in_flight == 0 {
            return Err(PipelineContractError::ZeroInFlight);
        }
        if self.max_in_flight > self.max_micro_batches {
            return Err(PipelineContractError::InFlightExceedsMicroBatches);
        }
        if !self.ordered_cancellation {
            return Err(PipelineContractError::UnorderedCancellationForbidden);
        }
        if !self.require_sequence_ids {
            return Err(PipelineContractError::SequenceIdsRequired);
        }
        Ok(())
    }
}

/// One asynchronous stage-to-stage transfer handle owned by AX Engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncStageTransfer {
    pub request_sequence: u64,
    pub micro_batch: u16,
    pub source_stage: u16,
    pub destination_stage: u16,
    pub payload_digest: Digest,
    pub byte_len: u64,
    pub cancelled: bool,
}

impl AsyncStageTransfer {
    /// Validate transfer metadata without inspecting tensor bytes.
    pub fn validate(
        &self,
        contract: &MicroBatchContract,
        pp_size: u16,
    ) -> Result<(), PipelineContractError> {
        contract.validate()?;
        if self.request_sequence == 0 {
            return Err(PipelineContractError::ZeroRequestSequence);
        }
        if self.micro_batch >= contract.max_micro_batches {
            return Err(PipelineContractError::MicroBatchOutOfRange);
        }
        if self.byte_len == 0 && !self.cancelled {
            return Err(PipelineContractError::EmptyNonCancelledPayload);
        }
        if self.source_stage + 1 != self.destination_stage {
            return Err(PipelineContractError::NonAdjacentStageTransfer);
        }
        if self.destination_stage >= pp_size {
            return Err(PipelineContractError::StageOutOfRange);
        }
        Ok(())
    }
}

/// Ordered commit gate that preserves per-request micro-batch order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicroBatchCommitGate {
    pub request_sequence: u64,
    pub next_expected_micro_batch: u16,
    pub max_micro_batches: u16,
    pub cancelled: bool,
}

impl MicroBatchCommitGate {
    pub fn new(
        request_sequence: u64,
        max_micro_batches: u16,
    ) -> Result<Self, PipelineContractError> {
        if request_sequence == 0 {
            return Err(PipelineContractError::ZeroRequestSequence);
        }
        if max_micro_batches == 0 {
            return Err(PipelineContractError::ZeroMicroBatches);
        }
        Ok(Self {
            request_sequence,
            next_expected_micro_batch: 0,
            max_micro_batches,
            cancelled: false,
        })
    }

    /// Admit the next ordered micro-batch or a cancellation tombstone.
    pub fn admit(
        &mut self,
        micro_batch: u16,
        cancelled: bool,
    ) -> Result<(), PipelineContractError> {
        if self.cancelled {
            return Err(PipelineContractError::RequestAlreadyCancelled);
        }
        if cancelled {
            self.cancelled = true;
            return Ok(());
        }
        if micro_batch != self.next_expected_micro_batch {
            return Err(PipelineContractError::OutOfOrderMicroBatch {
                expected: self.next_expected_micro_batch,
                actual: micro_batch,
            });
        }
        if micro_batch >= self.max_micro_batches {
            return Err(PipelineContractError::MicroBatchOutOfRange);
        }
        self.next_expected_micro_batch = self
            .next_expected_micro_batch
            .checked_add(1)
            .ok_or(PipelineContractError::MicroBatchOutOfRange)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PipelineContractError {
    #[error("max_micro_batches must be greater than zero")]
    ZeroMicroBatches,
    #[error("max_in_flight must be greater than zero")]
    ZeroInFlight,
    #[error("max_in_flight cannot exceed max_micro_batches")]
    InFlightExceedsMicroBatches,
    #[error("pipeline micro-batching requires ordered cancellation tombstones")]
    UnorderedCancellationForbidden,
    #[error("pipeline micro-batching requires stable per-request sequence ids")]
    SequenceIdsRequired,
    #[error("request sequence must be greater than zero")]
    ZeroRequestSequence,
    #[error("micro-batch index is outside the certified limit")]
    MicroBatchOutOfRange,
    #[error("non-cancelled stage transfer payload length must be greater than zero")]
    EmptyNonCancelledPayload,
    #[error("async pipeline transfers must move only to the adjacent next stage")]
    NonAdjacentStageTransfer,
    #[error("stage index is outside the certified pipeline width")]
    StageOutOfRange,
    #[error("request is already cancelled")]
    RequestAlreadyCancelled,
    #[error("micro-batch {actual} arrived out of order; expected {expected}")]
    OutOfOrderMicroBatch { expected: u16, actual: u16 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Digest;

    fn digest(byte: char) -> Digest {
        Digest::new(format!("sha256:{}", byte.to_string().repeat(64))).unwrap()
    }

    fn contract() -> MicroBatchContract {
        MicroBatchContract {
            max_micro_batches: 4,
            max_in_flight: 2,
            ordered_cancellation: true,
            require_sequence_ids: true,
        }
    }

    #[test]
    fn micro_batch_contract_fails_closed_on_unordered_cancellation() {
        let mut value = contract();
        value.ordered_cancellation = false;
        assert_eq!(
            value.validate(),
            Err(PipelineContractError::UnorderedCancellationForbidden)
        );
    }

    #[test]
    fn async_transfer_requires_adjacent_stage_and_identity() {
        let transfer = AsyncStageTransfer {
            request_sequence: 7,
            micro_batch: 0,
            source_stage: 0,
            destination_stage: 1,
            payload_digest: digest('a'),
            byte_len: 128,
            cancelled: false,
        };
        transfer.validate(&contract(), 2).unwrap();

        let mut skipped = transfer.clone();
        skipped.destination_stage = 2;
        assert_eq!(
            skipped.validate(&contract(), 3),
            Err(PipelineContractError::NonAdjacentStageTransfer)
        );
    }

    #[test]
    fn commit_gate_preserves_order_and_honors_cancellation() {
        let mut gate = MicroBatchCommitGate::new(9, 3).unwrap();
        gate.admit(0, false).unwrap();
        assert_eq!(
            gate.admit(2, false),
            Err(PipelineContractError::OutOfOrderMicroBatch {
                expected: 1,
                actual: 2
            })
        );
        gate.admit(1, false).unwrap();
        gate.admit(2, true).unwrap();
        assert_eq!(
            gate.admit(2, false),
            Err(PipelineContractError::RequestAlreadyCancelled)
        );
    }
}
