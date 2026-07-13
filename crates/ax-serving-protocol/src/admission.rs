use serde::{Deserialize, Serialize};

use crate::{AttemptId, RequestId};

pub const REQUEST_ID_HEADER: &str = "x-ax-request-id";
pub const ATTEMPT_ID_HEADER: &str = "x-ax-attempt-id";
pub const ADMISSION_STATE_HEADER: &str = "x-ax-admission-state";
pub const DISPATCH_TOKEN_HEADER: &str = "x-ax-dispatch-token";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionState {
    NotAdmitted,
    Admitted,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionPhase {
    Authentication,
    Admission,
    EndpointSelection,
    Connecting,
    PreAdmission,
    PostAdmission,
    ResponseCommitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitmentState {
    Uncommitted,
    HeadersCommitted,
    BodyCommitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    RetryDifferentWorker,
    DoNotRetry,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub phase: AdmissionPhase,
    pub admission_state: AdmissionState,
    pub request_id: RequestId,
    pub attempt_id: AttemptId,
}

impl AdmissionError {
    pub fn retry_decision(
        &self,
        commitment: CommitmentState,
        attempt_number: u8,
        maximum_attempts: u8,
    ) -> RetryDecision {
        if commitment != CommitmentState::Uncommitted
            || attempt_number >= maximum_attempts
            || !self.retryable
            || self.admission_state != AdmissionState::NotAdmitted
            || self.phase != AdmissionPhase::PreAdmission
        {
            return RetryDecision::DoNotRetry;
        }
        RetryDecision::RetryDifferentWorker
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    pub code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxErrorMetadata {
    pub request_id: RequestId,
    pub retryable: bool,
    pub phase: AdmissionPhase,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxErrorEnvelope {
    pub error: ErrorBody,
    pub ax: AxErrorMetadata,
}

#[cfg(test)]
mod tests {
    use super::{AdmissionError, AdmissionPhase, AdmissionState, CommitmentState, RetryDecision};
    use crate::{AttemptId, RequestId};

    fn not_admitted() -> AdmissionError {
        AdmissionError {
            code: "AXS_WORKER_DRAINING".into(),
            message: "worker is draining".into(),
            retryable: true,
            phase: AdmissionPhase::PreAdmission,
            admission_state: AdmissionState::NotAdmitted,
            request_id: RequestId::new(),
            attempt_id: AttemptId::new(),
        }
    }

    #[test]
    fn retry_requires_typed_pre_admission_and_uncommitted_response() {
        let error = not_admitted();
        assert_eq!(
            error.retry_decision(CommitmentState::Uncommitted, 1, 2),
            RetryDecision::RetryDifferentWorker
        );
        assert_eq!(
            error.retry_decision(CommitmentState::HeadersCommitted, 1, 2),
            RetryDecision::DoNotRetry
        );
    }

    #[test]
    fn retry_stops_at_maximum_attempts() {
        assert_eq!(
            not_admitted().retry_decision(CommitmentState::Uncommitted, 2, 2),
            RetryDecision::DoNotRetry
        );
    }
}
