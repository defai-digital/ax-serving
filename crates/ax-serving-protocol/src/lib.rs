//! Runtime-neutral wire contracts for AX Serving.
//!
//! This crate intentionally contains data types, validation, and protocol
//! semantics only. It must remain independent of async runtimes, HTTP stacks,
//! accelerator libraries, and inference-runtime SDKs.

pub mod adaptive;
pub mod admission;
pub mod cluster;
pub mod decision;
pub mod deployment;
pub mod domain;
pub mod ids;
pub mod lifecycle;
pub mod operation;
pub mod pipeline;
pub mod version;
pub mod worker;

pub use adaptive::{
    AdaptiveFederationPolicyV1, AdaptivePolicyError, AdaptiveSelection, DomainCostSignal,
};
pub use admission::{
    ADMISSION_STATE_HEADER, ATTEMPT_ID_HEADER, AdmissionError, AdmissionPhase, AdmissionState,
    AxErrorEnvelope, AxErrorMetadata, CommitmentState, DISPATCH_TOKEN_HEADER, ErrorBody,
    REQUEST_ID_HEADER, RetryDecision,
};
pub use cluster::{
    ArtifactFileKind, ArtifactFilePlan, ClusterLifecycleState, ClusterManifestError,
    ClusterModelSpec, ClusterRankObservation, ClusterRuntimeSpec, LayerRange, ParallelismKind,
    ParallelismManifestV1, ParallelismPlan, RankMemoryPlan, RankPlan, TransportPlan,
};
pub use decision::{
    CandidateDecision, CandidateRejectionReason, DecisionProfileV1, DecisionReasonCode,
    DecisionRecordV1, DecisionValidationError, PolicyMode,
};
pub use pipeline::{
    AsyncStageTransfer, MicroBatchCommitGate, MicroBatchContract, PipelineContractError,
};
pub use deployment::{
    DeploymentIdentity, DeploymentSpec, Digest, DigestError, EquivalencePolicy, IdentityField,
    IdentityPolicy, PoolSpec, RuntimeModelDescriptor,
};
pub use domain::{
    CompatibilityManifestDigest, DomainObservation, DomainSpec, DomainValidationError,
    EndpointScope, ExecutionDomainDescriptor, ExecutionDomainKind, QualificationState,
};
pub use ids::{
    AttemptId, DeploymentId, DomainId, EquivalenceClassId, JobId, LogicalModelId, PolicyId,
    PolicyVersion, PoolId, RegistrationId, RequestId, RuntimeModelId, TenantId, TrustDomainId,
    WorkerId, WorkerInstanceId,
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
    AgentDescriptor, CapacityError, CapacityObservation, DomainContractError, DrainDirective,
    HardwareDescriptor, HeartbeatRequest, HeartbeatResponse, LeaseToken, RegisterWorkerRequest,
    RegisterWorkerResponse, RuntimeDescriptor, RuntimeObservation, RuntimeState, RuntimeStatus,
    WorkerDescriptor, WorkerState,
};
