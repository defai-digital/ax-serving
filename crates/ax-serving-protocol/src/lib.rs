//! Runtime-neutral wire contracts for AX Serving.
//!
//! This crate intentionally contains data types, validation, and protocol
//! semantics only. It must remain independent of async runtimes, HTTP stacks,
//! accelerator libraries, and inference-runtime SDKs.

pub mod admission;
pub mod deployment;
pub mod ids;
pub mod lifecycle;
pub mod operation;
pub mod version;
pub mod worker;

pub use admission::{
    ADMISSION_STATE_HEADER, ATTEMPT_ID_HEADER, AdmissionError, AdmissionPhase, AdmissionState,
    AxErrorEnvelope, AxErrorMetadata, CommitmentState, DISPATCH_TOKEN_HEADER, ErrorBody,
    REQUEST_ID_HEADER, RetryDecision,
};
pub use deployment::{
    DeploymentIdentity, DeploymentSpec, Digest, EquivalencePolicy, IdentityField, IdentityPolicy,
    PoolSpec, RuntimeModelDescriptor,
};
pub use ids::{
    AttemptId, DeploymentId, EquivalenceClassId, JobId, LogicalModelId, PoolId, RegistrationId,
    RequestId, RuntimeModelId, TenantId, TrustDomainId, WorkerId, WorkerInstanceId,
};
pub use lifecycle::{
    DeploymentCommand, DeploymentControlRecord, DeploymentDesiredState, DeploymentJobAction,
    DeploymentJobObservation, DeploymentJobRecord, DeploymentJobStatus, DeploymentObservedState,
};
pub use operation::{Operation, ProtocolCapability};
pub use version::{
    CURRENT_PROTOCOL, NegotiatedProtocol, ProtocolDescriptor, ProtocolError, ProtocolVersion,
    negotiate_protocol,
};
pub use worker::{
    AgentDescriptor, CapacityObservation, DrainDirective, HardwareDescriptor, HeartbeatRequest,
    HeartbeatResponse, LeaseToken, RegisterWorkerRequest, RegisterWorkerResponse,
    RuntimeDescriptor, RuntimeObservation, RuntimeState, RuntimeStatus, WorkerDescriptor,
    WorkerState,
};
