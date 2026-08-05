use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use anyhow::{Context, Result};
use ax_serving_protocol::{
    AgentDescriptor, CapacityObservation, DeploymentIdentity, DomainObservation, EndpointScope,
    ExecutionDomainDescriptor, ExecutionDomainKind, HardwareDescriptor,
    HeartbeatRequest as ProtocolHeartbeatRequest, HeartbeatResponse as ProtocolHeartbeatResponse,
    LeaseToken, Operation, PoolId, ProtocolCapability, ProtocolDescriptor, RegisterWorkerRequest,
    RegisterWorkerResponse, RegistrationId, RuntimeDescriptor, RuntimeModelDescriptor,
    RuntimeModelId, RuntimeObservation, RuntimeStatus, TrustDomainId, WorkerDescriptor,
    WorkerId as ProtocolWorkerId, WorkerInstanceId,
};
use tokio::sync::RwLock;

fn current_rss_bytes() -> u64 {
    // Read RSS from /proc/self/status on macOS via sysctl (same approach as
    // ax-serving-engine metrics).  Returns 0 on failure rather than panicking.
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn getpid() -> i32;
        }
        // SAFETY: libc call with correct argument structure.
        let pid = unsafe { getpid() };
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output();
        if let Ok(out) = output
            && let Ok(s) = std::str::from_utf8(&out.stdout)
            && let Ok(kb) = s.trim().parse::<u64>()
        {
            return kb * 1024;
        }
        0
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status.lines().find_map(|line| {
                    let value = line.strip_prefix("VmRSS:")?.trim();
                    let kib = value.split_whitespace().next()?.parse::<u64>().ok()?;
                    Some(kib.saturating_mul(1024))
                })
            })
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

use crate::config::ThorConfig;
use crate::sglang;

const CONTROL_PLANE_REQUEST_TIMEOUT_SECS: u64 = 10;

fn control_plane_request_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(CONTROL_PLANE_REQUEST_TIMEOUT_SECS)
}

fn with_internal_token(
    req: reqwest::RequestBuilder,
    token: Option<&String>,
) -> reqwest::RequestBuilder {
    match token {
        Some(t) => req.header("X-Internal-Token", t),
        None => req,
    }
}

#[derive(Debug, Clone)]
pub struct WorkerSession {
    pub worker_id: ProtocolWorkerId,
    pub instance_id: WorkerInstanceId,
    pub registration_id: RegistrationId,
    pub lease_token: LeaseToken,
    pub heartbeat_interval_ms: u64,
    pub sequence: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
pub struct RegistrationState {
    pub session: WorkerSession,
    pub models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SharedRuntime {
    pub inflight: Arc<AtomicUsize>,
    pub session: Arc<RwLock<Option<WorkerSession>>>,
    pub models: Arc<RwLock<Vec<String>>>,
    pub instance_id: WorkerInstanceId,
    pub inventory_generation: Arc<AtomicU64>,
    pub draining: Arc<AtomicBool>,
}

impl SharedRuntime {
    pub fn new() -> Self {
        Self {
            inflight: Arc::new(AtomicUsize::new(0)),
            session: Arc::new(RwLock::new(None)),
            models: Arc::new(RwLock::new(Vec::new())),
            instance_id: WorkerInstanceId::new(),
            inventory_generation: Arc::new(AtomicU64::new(1)),
            draining: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Default for SharedRuntime {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn register(
    control_client: &reqwest::Client,
    runtime_client: &reqwest::Client,
    config: &ThorConfig,
    instance_id: WorkerInstanceId,
) -> Result<RegistrationState> {
    let model_info = sglang::get_model_info(runtime_client, &config.runtime_url).await?;
    let models: Vec<String> = model_info.iter().map(|m| m.id.clone()).collect();
    let body = registration_body(config, &model_info, instance_id)?;

    let req = with_internal_token(
        control_client
            .post(format!(
                "{}/internal/workers/register",
                config.control_plane_url
            ))
            .timeout(control_plane_request_timeout())
            .json(&body),
        config.worker_token.as_ref(),
    );

    let response: RegisterWorkerResponse = req
        .send()
        .await
        .context("runtime-node agent registration request failed")?
        .error_for_status()
        .context("runtime-node agent registration rejected")?
        .json()
        .await
        .context("failed to parse runtime-node agent registration response")?;

    let heartbeat_interval_ms = response.heartbeat_interval_ms.clamp(1_000, 300_000);
    Ok(RegistrationState {
        session: WorkerSession {
            worker_id: ProtocolWorkerId::new(config.worker_id.clone())?,
            instance_id,
            registration_id: response.registration_id,
            lease_token: response.lease_token,
            heartbeat_interval_ms,
            sequence: Arc::new(AtomicU64::new(0)),
        },
        models,
    })
}

fn registration_body(
    config: &ThorConfig,
    model_info: &[sglang::ModelInfo],
    instance_id: WorkerInstanceId,
) -> Result<RegisterWorkerRequest> {
    let runtime_models = model_info
        .iter()
        .map(|model| protocol_model_descriptor(model, config))
        .collect::<Result<Vec<_>>>()?;
    let pool = config
        .worker_pool
        .clone()
        .unwrap_or_else(|| format!("{}-{}", config.runtime, config.hardware_class));
    let pool = PoolId::new(normalize_identifier(&pool))?;
    let worker_id = ProtocolWorkerId::new(config.worker_id.clone())?;
    let trust_domain = TrustDomainId::new(config.trust_domain.clone())?;
    let mut labels = std::collections::BTreeMap::new();
    labels.insert("node_class".into(), config.node_class.clone());
    if let Some(friendly_name) = &config.friendly_name {
        labels.insert("friendly_name".into(), friendly_name.clone());
    }
    if let Some(chip_model) = &config.chip_model {
        labels.insert("chip_model".into(), chip_model.clone());
    }

    let observed_at = time::OffsetDateTime::now_utc();
    let runtime_status = RuntimeStatus::ready();
    let capacity = CapacityObservation {
        active_requests: Some(0),
        max_concurrent_requests: Some(config.max_inflight as u64),
        ..Default::default()
    };
    let domain = execution_domain_descriptor(config, &pool, &trust_domain)?;
    let domain_observation = domain.as_ref().map(|descriptor| DomainObservation {
        observed_at,
        generation: 1,
        ready: runtime_status.ready,
        state: runtime_status.state,
        reason_code: runtime_status.reason_code.clone(),
        frontend_instances_ready: Some(1),
        aggregate_capacity: Some(capacity.clone()),
        manifest_digest: descriptor.compatibility_manifest.clone(),
        models: runtime_models.clone(),
    });

    Ok(RegisterWorkerRequest {
        protocol: ProtocolDescriptor::current(protocol_capabilities(domain.is_some())),
        agent: AgentDescriptor {
            name: "ax-runtime-agent".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_sha: option_env!("AXS_BUILD_SHA").map(ToOwned::to_owned),
        },
        worker: WorkerDescriptor {
            id: worker_id,
            instance_id,
            advertise_url: config.advertised_url.clone(),
            pool_id: pool,
            trust_domain,
            labels,
        },
        runtime: RuntimeDescriptor {
            kind: normalize_runtime_kind(&config.runtime),
            version: config.runtime_version.clone(),
            api: "openai-v1".into(),
        },
        hardware: HardwareDescriptor {
            platform: std::env::consts::OS.into(),
            accelerator: accelerator_name(config),
            device_count: 1,
            memory_bytes: None,
            hardware_class: Some(config.hardware_class.clone()),
        },
        domain,
        domain_observation,
        observation: RuntimeObservation {
            observed_at,
            runtime: runtime_status,
            inventory_generation: 1,
            models: runtime_models,
            capacity: Some(capacity),
        },
    })
}

fn protocol_capabilities(domain_enabled: bool) -> Vec<ProtocolCapability> {
    let mut capabilities = vec![
        ProtocolCapability::CONTROL_DRAIN,
        ProtocolCapability::CONTROL_INVENTORY_DELTA,
        ProtocolCapability::DISPATCH_CANCEL,
        ProtocolCapability::DISPATCH_TYPED_ADMISSION,
        ProtocolCapability::TELEMETRY_CAPACITY,
        ProtocolCapability::TELEMETRY_KV_CACHE,
        ProtocolCapability::TELEMETRY_PREFIX_CACHE,
    ];
    if domain_enabled {
        capabilities.push(ProtocolCapability::CONTROL_EXECUTION_DOMAIN);
        capabilities.push(ProtocolCapability::TELEMETRY_DOMAIN_CAPACITY);
    }
    capabilities
        .into_iter()
        .map(|capability| ProtocolCapability::new(capability).expect("static protocol capability"))
        .collect()
}

fn execution_domain_descriptor(
    config: &ThorConfig,
    pool: &PoolId,
    trust_domain: &TrustDomainId,
) -> Result<Option<ExecutionDomainDescriptor>> {
    let Some(domain) = config.execution_domain.as_ref() else {
        return Ok(None);
    };
    let runtime_kind = normalize_runtime_kind(&config.runtime);
    let (kind, execution_owner) = if runtime_kind == "ax_engine" {
        (ExecutionDomainKind::MacAxEngine, "ax_engine".to_string())
    } else {
        (
            ExecutionDomainKind::CompatibilityRuntimeEndpoint,
            runtime_kind,
        )
    };
    let descriptor = ExecutionDomainDescriptor {
        id: domain.id.clone(),
        kind,
        endpoint_scope: EndpointScope::Node,
        execution_owner,
        qualification: domain.qualification,
        pool_id: pool.clone(),
        trust_domain: trust_domain.clone(),
        hardware_class: config.hardware_class.clone(),
        architecture: normalized_architecture().into(),
        compatibility_manifest: domain.compatibility_manifest.clone(),
        labels: Default::default(),
    };
    descriptor
        .validate()
        .context("invalid protocol-v1.1 execution-domain descriptor")?;
    Ok(Some(descriptor))
}

fn normalized_architecture() -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" => "arm64",
        architecture => architecture,
    }
}

fn protocol_model_descriptor(
    model: &sglang::ModelInfo,
    config: &ThorConfig,
) -> Result<RuntimeModelDescriptor> {
    let supported = model_supported_operations(model, config);
    let mut operations = std::collections::BTreeSet::new();
    let mut capabilities = std::collections::BTreeSet::new();
    if supported.iter().any(|operation| operation == "llm") {
        operations.insert(Operation::chat_completions());
        operations.insert(Operation::text_completions());
    }
    if supported.iter().any(|operation| operation == "embedding") {
        operations.insert(Operation::embeddings());
    }
    if supported.iter().any(|operation| operation == "vision") {
        capabilities.insert(ProtocolCapability::new("inference.vision")?);
    }
    capabilities.extend(config.model_identity.capabilities.iter().cloned());

    Ok(RuntimeModelDescriptor {
        runtime_model_id: RuntimeModelId::new(model.id.clone())?,
        identity: DeploymentIdentity {
            runtime_kind: normalize_runtime_kind(&config.runtime),
            runtime_version: Some(config.runtime_version.clone()),
            revision: config.model_identity.revision.clone(),
            artifact_digest: config.model_identity.artifact_digest.clone(),
            tokenizer_digest: config.model_identity.tokenizer_digest.clone(),
            template_digest: config.model_identity.template_digest.clone(),
            quantization: config
                .model_identity
                .quantization
                .clone()
                .or_else(|| model.quantization.clone()),
        },
        operations,
        capabilities,
        max_context_tokens: config.max_context.or(model.max_model_len).map(u64::from),
        max_output_tokens: config
            .model_identity
            .max_output_tokens
            .or(model.max_output_tokens.map(u64::from)),
    })
}

fn normalize_runtime_kind(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "axengine" | "ax_engine" | "native" => "ax_engine".into(),
        "v_llm" | "vllm" => "vllm".into(),
        "sg_lang" | "sglang" => "sglang".into(),
        other => other.to_string(),
    }
}

fn normalize_identifier(raw: &str) -> String {
    let mut normalized = raw
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | ':' | '-') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    normalized.truncate(128);
    if normalized.is_empty() {
        "default".into()
    } else {
        normalized
    }
}

fn accelerator_name(config: &ThorConfig) -> String {
    // Match on the normalized runtime kind so aliases/casing accepted by
    // `normalize_runtime_kind` (e.g. "axengine", "native", "vLLM", "sg-lang")
    // resolve to the same accelerator as their canonical spellings.
    match normalize_runtime_kind(&config.runtime).as_str() {
        "ax_engine" => "apple-gpu".into(),
        "vllm" | "sglang" => "nvidia-gpu".into(),
        _ if config.hardware_class.to_ascii_lowercase().contains("cuda") => "nvidia-gpu".into(),
        _ => "unknown".into(),
    }
}

pub async fn heartbeat_loop(
    control_client: reqwest::Client,
    runtime_client: reqwest::Client,
    config: ThorConfig,
    runtime: SharedRuntime,
) {
    loop {
        let session = {
            let guard = runtime.session.read().await;
            guard.clone()
        };

        let Some(session) = session else {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            continue;
        };

        let readiness = sglang::probe_runtime(&runtime_client, &config.runtime_url).await;
        let (models, runtime_models, runtime_ready, runtime_status_reason) = match readiness {
            Ok(()) => match sglang::get_model_info(&runtime_client, &config.runtime_url).await {
                Ok(model_info) => {
                    let models = model_info.iter().map(|m| m.id.clone()).collect::<Vec<_>>();
                    let runtime_models = model_info
                        .iter()
                        .map(|model| protocol_model_descriptor(model, &config))
                        .collect::<Result<Vec<_>>>();
                    match runtime_models {
                        Ok(runtime_models) => (models, runtime_models, true, None),
                        Err(error) => {
                            tracing::warn!(%error, "runtime inventory violates protocol contract");
                            (
                                Vec::new(),
                                Vec::new(),
                                false,
                                Some("runtime_inventory_invalid"),
                            )
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(%err, "failed to refresh runtime model list for heartbeat");
                    runtime.models.write().await.clear();
                    (
                        Vec::new(),
                        Vec::new(),
                        false,
                        Some("runtime_inventory_failed"),
                    )
                }
            },
            Err(err) => {
                tracing::warn!(%err, "runtime readiness probe failed");
                runtime.models.write().await.clear();
                (Vec::new(), Vec::new(), false, Some("runtime_health_failed"))
            }
        };

        let inventory_generation = {
            let mut current_models = runtime.models.write().await;
            if *current_models != models {
                *current_models = models;
                runtime.inventory_generation.fetch_add(1, Ordering::AcqRel) + 1
            } else {
                runtime.inventory_generation.load(Ordering::Acquire)
            }
        };

        let current_inflight = runtime.inflight.load(Ordering::Relaxed);
        let rss_bytes = current_rss_bytes();
        let telemetry = if runtime_ready {
            match sglang::get_runtime_telemetry(&runtime_client, &config.runtime_url).await {
                Ok(telemetry) => telemetry,
                Err(err) => {
                    tracing::debug!(%err, "runtime metrics unavailable; using heartbeat defaults");
                    sglang::RuntimeTelemetry::default()
                }
            }
        } else {
            sglang::RuntimeTelemetry::default()
        };
        let kv_cache_used_ratio = telemetry.kv_utilization.or_else(|| {
            match (telemetry.kv_pages_used, telemetry.kv_pages_total) {
                (Some(used), Some(total)) if total > 0 => Some(used as f64 / total as f64),
                _ => None,
            }
        });
        let capacity = CapacityObservation {
            active_requests: Some(
                telemetry
                    .active_sequences
                    .unwrap_or(current_inflight)
                    .min(u64::MAX as usize) as u64,
            ),
            max_concurrent_requests: Some(config.max_inflight as u64),
            waiting_requests: Some(telemetry.queue_depth.unwrap_or(0).min(u64::MAX as usize) as u64),
            process_rss_bytes: Some(rss_bytes),
            recent_error_rate: telemetry.error_rate,
            kv_cache_used_ratio,
            prefix_cache_hit_ratio: None,
            batch_token_capacity: telemetry.max_batch_size.map(u64::from),
            batch_tokens_in_use: telemetry.active_batch_size.map(u64::from),
            ttft_ewma_ms: telemetry.ttft_p95_ms.map(|value| value as f64),
            inter_token_ewma_ms: None,
            generated_tokens_per_second: telemetry.decode_tok_per_sec,
            observation_window_ms: None,
        };
        let sequence = session.sequence.fetch_add(1, Ordering::AcqRel) + 1;
        let observed_at = time::OffsetDateTime::now_utc();
        let runtime_status = if runtime_ready {
            RuntimeStatus::ready()
        } else {
            RuntimeStatus::unavailable(runtime_status_reason.unwrap_or("runtime_unavailable"))
        };
        let domain_observation = config
            .execution_domain
            .as_ref()
            .map(|domain| DomainObservation {
                observed_at,
                generation: sequence,
                ready: runtime_status.ready,
                state: runtime_status.state,
                reason_code: runtime_status.reason_code.clone(),
                frontend_instances_ready: Some(u32::from(runtime_status.ready)),
                aggregate_capacity: Some(capacity.clone()),
                manifest_digest: domain.compatibility_manifest.clone(),
                models: runtime_models.clone(),
            });
        let body = ProtocolHeartbeatRequest {
            registration_id: session.registration_id,
            instance_id: session.instance_id,
            sequence,
            observed_at,
            runtime: runtime_status,
            inventory_generation,
            models: Some(runtime_models),
            capacity: Some(capacity),
            domain_observation,
            deployment_jobs: Vec::new(),
        };

        // BUG-096: use a short per-request timeout for control-plane calls so a
        // slow/unresponsive orchestrator doesn't stall the heartbeat loop for 300s.
        let req = with_internal_token(
            control_client
                .post(format!(
                    "{}/internal/workers/{}/heartbeat",
                    config.control_plane_url, session.worker_id
                ))
                .timeout(control_plane_request_timeout())
                .header("x-ax-lease-token", session.lease_token.expose())
                .json(&body),
            config.worker_token.as_ref(),
        );

        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<ProtocolHeartbeatResponse>().await {
                    Ok(directive) if directive.reregister => {
                        tracing::warn!("control plane requested runtime-node re-registration");
                        *runtime.session.write().await = None;
                    }
                    Ok(directive)
                        if directive.drain == ax_serving_protocol::DrainDirective::Begin =>
                    {
                        runtime.draining.store(true, Ordering::Release);
                        tracing::info!("control plane requested runtime-node drain");
                        if runtime.inflight.load(Ordering::Acquire) == 0 {
                            match drain_complete(&control_client, &config, &runtime).await {
                                Ok(()) => {
                                    *runtime.session.write().await = None;
                                    tracing::info!("runtime-node drain completed");
                                }
                                Err(error) => {
                                    tracing::warn!(%error, "runtime-node drain completion failed");
                                }
                            }
                        }
                    }
                    Ok(directive)
                        if directive.drain == ax_serving_protocol::DrainDirective::Complete =>
                    {
                        runtime.draining.store(true, Ordering::Release);
                        *runtime.session.write().await = None;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(%error, "invalid control-plane heartbeat response");
                    }
                }
            }
            Ok(resp) if matches!(resp.status().as_u16(), 404 | 409 | 410) => {
                tracing::warn!(status = %resp.status(), "runtime-node agent evicted, re-registering");
                match register(
                    &control_client,
                    &runtime_client,
                    &config,
                    runtime.instance_id,
                )
                .await
                {
                    Ok(registration) => {
                        install_registration(&runtime, registration).await;
                    }
                    Err(err) => {
                        tracing::warn!(%err, "runtime-node agent re-registration failed, clearing stale session");
                        *runtime.session.write().await = None;
                    }
                }
            }
            Ok(resp) => tracing::warn!(status = %resp.status(), "runtime-node heartbeat rejected"),
            Err(err) => tracing::warn!(%err, "runtime-node heartbeat failed"),
        }

        tokio::time::sleep(std::time::Duration::from_millis(
            session.heartbeat_interval_ms,
        ))
        .await;
    }
}

fn model_supported_operations(model: &sglang::ModelInfo, config: &ThorConfig) -> Vec<String> {
    let mut operations = model
        .supported_operations
        .iter()
        .filter_map(|operation| normalize_operation(operation))
        .collect::<Vec<_>>();
    apply_operation_override(&mut operations, "embedding", config.embedding);
    apply_operation_override(&mut operations, "vision", config.vision);
    operations.sort();
    operations.dedup();
    operations
}

fn apply_operation_override(
    operations: &mut Vec<String>,
    operation: &'static str,
    override_value: Option<bool>,
) {
    match override_value {
        Some(true) => operations.push(operation.to_string()),
        Some(false) => operations.retain(|op| !operation_matches(op, operation)),
        None => {}
    }
}

fn operation_matches(raw: &str, operation: &str) -> bool {
    normalize_operation(raw).as_deref() == Some(operation)
}

fn normalize_operation(raw: &str) -> Option<String> {
    let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
    let operation = match normalized.as_str() {
        "" => return None,
        "embedding" | "embeddings" => "embedding",
        "vision" | "image" | "multimodal" => "vision",
        "llm" | "text" | "chat" | "completion" | "completions" => "llm",
        _ => normalized.as_str(),
    };
    Some(operation.to_string())
}

/// Install a (re-)registration into the shared runtime state.
///
/// A successful registration means the control plane has (re-)admitted this
/// node, so any stale drain state from a prior session must be cleared and the
/// inventory generation restarts at 1, matching `registration_body`.
async fn install_registration(runtime: &SharedRuntime, registration: RegistrationState) {
    *runtime.models.write().await = registration.models;
    *runtime.session.write().await = Some(registration.session);
    runtime.draining.store(false, Ordering::Release);
    runtime.inventory_generation.store(1, Ordering::Release);
}

pub async fn drain(
    client: &reqwest::Client,
    config: &ThorConfig,
    runtime: &SharedRuntime,
) -> Result<()> {
    let session = runtime
        .session
        .read()
        .await
        .clone()
        .context("runtime-node agent has no active worker session")?;
    with_internal_token(
        client
            .post(format!(
                "{}/internal/workers/{}/drain",
                config.control_plane_url, session.worker_id
            ))
            .header("x-ax-lease-token", session.lease_token.expose())
            .timeout(control_plane_request_timeout()),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("runtime-node drain request failed")?
    .error_for_status()
    .context("runtime-node drain request rejected")?;
    Ok(())
}

pub async fn drain_complete(
    client: &reqwest::Client,
    config: &ThorConfig,
    runtime: &SharedRuntime,
) -> Result<()> {
    let session = runtime
        .session
        .read()
        .await
        .clone()
        .context("runtime-node agent has no active worker session")?;
    with_internal_token(
        client
            .post(format!(
                "{}/internal/workers/{}/drain-complete",
                config.control_plane_url, session.worker_id
            ))
            .header("x-ax-lease-token", session.lease_token.expose())
            .timeout(control_plane_request_timeout()),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("runtime-node drain-complete request failed")?
    .error_for_status()
    .context("runtime-node drain-complete request rejected")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CONTROL_PLANE_REQUEST_TIMEOUT_SECS, SharedRuntime, control_plane_request_timeout,
        registration_body,
    };
    use crate::config::{ExecutionDomainConfig, ThorConfig};
    use crate::sglang::ModelInfo;
    use ax_serving_protocol::{
        DomainId, ExecutionDomainKind, Operation, ProtocolCapability, QualificationState,
        WorkerInstanceId,
    };

    fn test_config() -> ThorConfig {
        ThorConfig {
            control_plane_url: "http://127.0.0.1:18080".into(),
            worker_token: None,
            runtime_url: "http://127.0.0.1:8000".into(),
            runtime_api_key: None,
            dispatch_token: None,
            tls_profile: "loopback_dev".into(),
            runtime: "vllm".into(),
            runtime_version: "test".into(),
            worker_id: "worker-test".into(),
            trust_domain: "test".into(),
            listen_addr: "127.0.0.1:18081".parse().unwrap(),
            advertised_url: "http://127.0.0.1:18081".into(),
            max_inflight: 8,
            worker_pool: None,
            node_class: "thor".into(),
            hardware_class: "thor".into(),
            execution_domain: None,
            friendly_name: None,
            chip_model: None,
            shutdown_timeout_secs: None,
            max_context: None,
            embedding: None,
            vision: None,
            model_identity: Default::default(),
        }
    }

    #[tokio::test]
    async fn shared_runtime_starts_with_empty_model_cache() {
        let runtime = SharedRuntime::new();
        assert!(runtime.models.read().await.is_empty());
    }

    #[test]
    fn control_plane_request_timeout_is_short_enough_for_shutdown_paths() {
        assert_eq!(
            control_plane_request_timeout(),
            std::time::Duration::from_secs(CONTROL_PLANE_REQUEST_TIMEOUT_SECS)
        );
        assert!(control_plane_request_timeout() <= std::time::Duration::from_secs(10));
    }

    #[test]
    fn registration_body_advertises_only_safe_node_domain_kinds() {
        let mut config = test_config();
        config.runtime = "ax_engine".into();
        config.worker_pool = Some("mac-mlx".into());
        config.hardware_class = "apple-silicon".into();
        config.execution_domain = Some(ExecutionDomainConfig {
            id: DomainId::new("mac-studio-1").unwrap(),
            qualification: QualificationState::Certified,
            compatibility_manifest: None,
        });
        let models = [ModelInfo {
            id: "qwen-main".into(),
            max_model_len: Some(8192),
            max_output_tokens: Some(2048),
            quantization: None,
            artifact_format: None,
            modalities: Vec::new(),
            supported_operations: vec!["llm".into()],
        }];

        let mac = registration_body(&config, &models, WorkerInstanceId::new()).unwrap();
        assert_eq!(
            mac.domain.as_ref().map(|domain| domain.kind),
            Some(ExecutionDomainKind::MacAxEngine)
        );
        assert!(
            mac.domain_observation
                .as_ref()
                .is_some_and(|value| value.ready)
        );
        assert!(mac.protocol.capabilities.iter().any(|capability| {
            capability.as_str() == ProtocolCapability::CONTROL_EXECUTION_DOMAIN
        }));

        config.runtime = "vllm".into();
        let compatibility = registration_body(&config, &models, WorkerInstanceId::new()).unwrap();
        assert_eq!(
            compatibility.domain.as_ref().map(|domain| domain.kind),
            Some(ExecutionDomainKind::CompatibilityRuntimeEndpoint)
        );
    }

    #[test]
    fn registration_body_derives_embedding_capability_from_runtime_metadata() {
        let config = test_config();
        let body = registration_body(
            &config,
            &[ModelInfo {
                id: "embed-main".into(),
                max_model_len: Some(8192),
                max_output_tokens: Some(2048),
                quantization: None,
                artifact_format: None,
                modalities: Vec::new(),
                supported_operations: vec!["embedding".into()],
            }],
            WorkerInstanceId::new(),
        )
        .unwrap();

        let model = &body.observation.models[0];
        assert_eq!(model.runtime_model_id.as_str(), "embed-main");
        assert!(model.operations.contains(&Operation::embeddings()));
        assert!(!model.operations.contains(&Operation::chat_completions()));
        assert_eq!(model.max_output_tokens, Some(2048));
    }

    #[test]
    fn registration_body_env_override_can_disable_metadata_embedding() {
        let mut config = test_config();
        config.embedding = Some(false);
        let body = registration_body(
            &config,
            &[ModelInfo {
                id: "embed-main".into(),
                max_model_len: Some(8192),
                max_output_tokens: None,
                quantization: None,
                artifact_format: None,
                modalities: Vec::new(),
                supported_operations: vec!["embedding".into()],
            }],
            WorkerInstanceId::new(),
        )
        .unwrap();

        assert!(body.observation.models[0].operations.is_empty());
    }

    #[test]
    fn registration_body_env_override_updates_model_inventory_operations() {
        let mut config = test_config();
        config.embedding = Some(true);
        config.vision = Some(false);
        let body = registration_body(
            &config,
            &[ModelInfo {
                id: "mixed-main".into(),
                max_model_len: Some(8192),
                max_output_tokens: None,
                quantization: None,
                artifact_format: None,
                modalities: Vec::new(),
                supported_operations: vec!["Completion".into(), "Image".into()],
            }],
            WorkerInstanceId::new(),
        )
        .unwrap();

        let model = &body.observation.models[0];
        assert!(model.operations.contains(&Operation::embeddings()));
        assert!(model.operations.contains(&Operation::chat_completions()));
        assert!(model.capabilities.is_empty());
    }

    #[test]
    fn registration_body_normalizes_runtime_operation_aliases() {
        let config = test_config();
        let body = registration_body(
            &config,
            &[ModelInfo {
                id: "embed-main".into(),
                max_model_len: Some(8192),
                max_output_tokens: None,
                quantization: None,
                artifact_format: None,
                modalities: Vec::new(),
                supported_operations: vec!["Embeddings".into()],
            }],
            WorkerInstanceId::new(),
        )
        .unwrap();

        assert!(
            body.observation.models[0]
                .operations
                .contains(&Operation::embeddings())
        );
    }

    #[test]
    fn accelerator_name_uses_normalized_runtime_kind() {
        use super::accelerator_name;

        let mut config = test_config();
        for alias in ["ax_engine", "axengine", "AxEngine", "native"] {
            config.runtime = alias.into();
            assert_eq!(
                accelerator_name(&config),
                "apple-gpu",
                "runtime alias {alias} must map to apple-gpu"
            );
        }
        for alias in ["vllm", "vLLM", "sglang", "sg-lang"] {
            config.runtime = alias.into();
            assert_eq!(
                accelerator_name(&config),
                "nvidia-gpu",
                "runtime alias {alias} must map to nvidia-gpu"
            );
        }
    }

    #[tokio::test]
    async fn install_registration_resets_drain_state_and_inventory_generation() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        use super::{RegistrationState, WorkerSession, install_registration};
        use ax_serving_protocol::{LeaseToken, RegistrationId, WorkerId as ProtocolWorkerId};

        let runtime = SharedRuntime::new();
        runtime.draining.store(true, Ordering::Release);
        runtime.inventory_generation.store(7, Ordering::Release);

        install_registration(
            &runtime,
            RegistrationState {
                session: WorkerSession {
                    worker_id: ProtocolWorkerId::new("worker-1").unwrap(),
                    instance_id: WorkerInstanceId::new(),
                    registration_id: RegistrationId::new(),
                    lease_token: LeaseToken::new("0123456789abcdef").unwrap(),
                    heartbeat_interval_ms: 5_000,
                    sequence: Arc::new(AtomicU64::new(0)),
                },
                models: vec!["model-a".to_string()],
            },
        )
        .await;

        assert!(
            !runtime.draining.load(Ordering::Acquire),
            "re-registration re-admits the node; stale drain state must be cleared"
        );
        assert_eq!(
            runtime.inventory_generation.load(Ordering::Acquire),
            1,
            "inventory generation must restart at 1 for a fresh registration"
        );
        assert!(runtime.session.read().await.is_some());
        assert_eq!(runtime.models.read().await.as_slice(), ["model-a"]);
    }
}
