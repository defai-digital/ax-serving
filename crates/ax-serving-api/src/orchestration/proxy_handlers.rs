use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use ax_serving_protocol::{LogicalModelId, Operation, PoolId, ProtocolCapability, TenantId};
use axum::{
    Json,
    body::{Body, BodyDataStream, Bytes},
    extract::{ConnectInfo, Extension, Path, Query, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{Html, IntoResponse},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tracing::warn;

use super::OrchestratorLayer;
use super::deployment::DeploymentMode;
use super::error::ax_error_response;
use super::queue::{AcquireResult, QueuePriority};
use super::registry::{BackendKind, RuntimeKind};
use super::request_profile::{PriorityClass, RequestProfile, validate_unique_routing_fields};
use crate::auth::{AxRequestId, RequestId};
use crate::project_policy;
use crate::rest::schema::{
    EmbeddingsInput, InputMessage, MAX_CONTENT_BYTES, MAX_EMBEDDING_INPUTS,
    MAX_EMBEDDING_TOTAL_BYTES, MAX_EMBEDDING_TOTAL_TOKENS, MAX_MAX_TOKENS, MAX_MESSAGES,
    MessageContent,
};
use crate::utils::request_meta::{audit_actor, default_audit_limit};

// ── Shared inference proxy ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ProxyRequestMeta {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    runtime: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    messages: Vec<InputMessage>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
}

async fn proxy_inference(
    layer: Arc<OrchestratorLayer>,
    peer_addr: Option<SocketAddr>,
    req_headers: HeaderMap,
    body: Bytes,
    worker_path: &'static str,
    request_id: ax_serving_protocol::RequestId,
) -> axum::response::Response {
    let Some(admission_guard) = super::gateway_ops::AcceptedRequestGuard::try_admit(&layer.ops)
    else {
        let mut response = ax_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            request_id,
            "gateway_draining",
            "gateway is draining and not accepting new inference requests",
            true,
            ax_serving_protocol::AdmissionPhase::Admission,
        );
        if let Ok(value) = HeaderValue::from_str(&layer.retry_after_secs.to_string()) {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response.headers_mut().insert(
            HeaderName::from_static("x-ax-admission-state"),
            HeaderValue::from_static("not-admitted"),
        );
        return response;
    };
    if let Err(error) = validate_unique_routing_fields(&body) {
        return ax_error_response(
            StatusCode::BAD_REQUEST,
            request_id,
            "AXS_INVALID_REQUEST_JSON",
            error.to_string(),
            false,
            ax_serving_protocol::AdmissionPhase::Admission,
        );
    }
    let requested_pool = req_headers
        .get("x-ax-worker-pool")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let meta = match serde_json::from_slice::<ProxyRequestMeta>(&body) {
        Ok(meta) => meta,
        Err(_) => {
            return ax_error_response(
                StatusCode::BAD_REQUEST,
                request_id,
                "AXS_INVALID_REQUEST_JSON",
                "invalid JSON body",
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };
    let model_id = match validate_proxy_model_id(meta.model.clone()) {
        Ok(model_id) => model_id,
        Err((status, error)) => {
            return ax_error_response(
                status,
                request_id,
                "AXS_INVALID_MODEL",
                error,
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };
    let backend_hint = match validate_dispatch_hint(meta.runtime.clone().or(meta.backend.clone())) {
        Ok(hint) => hint,
        Err(error) => {
            return ax_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                request_id,
                "AXS_INVALID_RUNTIME_HINT",
                error,
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };
    let stream = meta.stream;
    let max_tokens = match (meta.max_tokens, meta.max_completion_tokens) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    };
    let _embedding_input = match validate_proxy_request_shape(worker_path, &meta) {
        Ok(input) => input,
        Err((status, error)) => {
            return ax_error_response(
                status,
                request_id,
                "AXS_INVALID_REQUEST_SHAPE",
                error,
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };
    let min_context = match declared_minimum_context(&req_headers) {
        Ok(value) => value,
        Err(error) => {
            return ax_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                request_id,
                "AXS_INVALID_CONTEXT_REQUIREMENT",
                error,
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };
    let request_timeout = match declared_request_timeout(
        &req_headers,
        std::time::Duration::from_secs(layer.config.request_timeout_secs),
    ) {
        Ok(value) => value,
        Err(error) => {
            return ax_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                request_id,
                "AXS_INVALID_REQUEST_DEADLINE",
                error,
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };

    let resolved_policy =
        match project_policy::enforce(&req_headers, &model_id, max_tokens, &layer.project_policy) {
            Ok(v) => v,
            Err(resp) => return resp.into_response(),
        };
    let policy_pool = resolved_policy
        .as_ref()
        .and_then(|v| v.worker_pool.as_deref());
    let required_pool = match policy_pool
        .map(|pool| PoolId::new(pool.to_string()))
        .transpose()
    {
        Ok(value) => value,
        Err(error) => {
            return ax_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                request_id,
                "AXS_POLICY_CONFIGURATION",
                format!("invalid policy pool: {error}"),
                false,
                ax_serving_protocol::AdmissionPhase::Authentication,
            );
        }
    };
    let preferred_pool = match requested_pool
        .map(|pool| PoolId::new(pool.to_string()))
        .transpose()
    {
        Ok(value) => value,
        Err(error) => {
            return ax_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                request_id,
                "AXS_INVALID_POOL_CONSTRAINT",
                format!("invalid worker pool: {error}"),
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };
    let priority = match PriorityClass::parse(
        req_headers
            .get("x-ax-priority")
            .and_then(|value| value.to_str().ok()),
    ) {
        Ok(value) => value,
        Err(error) => {
            return ax_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                request_id,
                "AXS_INVALID_PRIORITY",
                error.to_string(),
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };
    let profile = match build_request_profile(
        request_id,
        worker_path,
        &model_id,
        &meta,
        body.len(),
        max_tokens,
        min_context,
        backend_hint.clone(),
        required_pool.clone(),
        preferred_pool.clone(),
        priority,
        &req_headers,
        layer
            .config
            .cache_affinity_secret
            .as_ref()
            .map(|secret| secret.expose()),
        request_timeout,
    ) {
        Ok(profile) => profile,
        Err(error) => {
            return ax_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                request_id,
                "AXS_INVALID_REQUEST_PROFILE",
                error.to_string(),
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
    };

    let tenant_permit = if layer.config.tenant_max_concurrent > 0 {
        match layer.tenant_limiter.try_acquire(
            profile.tenant_id.as_str(),
            layer.config.tenant_max_concurrent,
        ) {
            Some(permit) => Some(permit),
            None => {
                return ax_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    request_id,
                    "AXS_TENANT_QUOTA_EXCEEDED",
                    "tenant concurrent-request quota exceeded",
                    true,
                    ax_serving_protocol::AdmissionPhase::Admission,
                );
            }
        }
    } else {
        None
    };
    let queue_priority = match profile.priority {
        PriorityClass::Low => QueuePriority::Low,
        PriorityClass::Normal => QueuePriority::Normal,
        PriorityClass::High => QueuePriority::High,
    };

    // Admission control: acquire a queue slot before dispatching.
    let permit = match tokio::time::timeout_at(
        profile.deadline,
        layer
            .queue
            .acquire_with_priority(fairness_client_key(&req_headers, peer_addr), queue_priority),
    )
    .await
    {
        Err(_) => {
            return ax_error_response(
                StatusCode::GATEWAY_TIMEOUT,
                request_id,
                "AXS_REQUEST_DEADLINE",
                "request deadline expired while waiting for gateway admission",
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
        Ok(outcome) => match outcome {
            AcquireResult::Permit(p) => p,

            AcquireResult::Rejected => {
                let queued = layer.queue.queued();
                let mut response = ax_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    request_id,
                    "AXS_GATEWAY_OVERLOADED",
                    "gateway concurrency limit exceeded",
                    true,
                    ax_serving_protocol::AdmissionPhase::Admission,
                );
                if let Ok(value) = HeaderValue::from_str(&queued.to_string()) {
                    response.headers_mut().insert("x-queue-depth", value);
                }
                if let Ok(value) = HeaderValue::from_str(&layer.retry_after_secs.to_string()) {
                    response.headers_mut().insert("retry-after", value);
                }
                return response;
            }

            AcquireResult::Shed => {
                let mut resp = ax_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    request_id,
                    "AXS_REQUEST_SHED",
                    "request shed by gateway overload policy",
                    true,
                    ax_serving_protocol::AdmissionPhase::Admission,
                );
                resp.headers_mut().insert(
                    HeaderName::from_static("x-reason"),
                    HeaderValue::from_static("request_shed"),
                );
                return resp;
            }

            AcquireResult::Timeout => {
                let mut resp = ax_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    request_id,
                    "AXS_ADMISSION_DEADLINE",
                    "request deadline expired while waiting for gateway admission",
                    true,
                    ax_serving_protocol::AdmissionPhase::Admission,
                );
                resp.headers_mut().insert(
                    HeaderName::from_static("x-reason"),
                    HeaderValue::from_static("queue_timeout"),
                );
                return resp;
            }
        },
    };

    let deployment_catalog = layer.deployment_catalog.snapshot();
    let mut resp = if deployment_catalog.mode() == DeploymentMode::Explicit {
        layer
            .dispatcher
            .forward_profile(
                &layer.registry,
                &deployment_catalog,
                layer.policy.as_ref(),
                &profile,
                worker_path,
                body,
                layer.config.telemetry_stale_ms,
                layer.config.max_dispatch_attempts,
            )
            .await
    } else {
        layer
            .dispatcher
            .forward_kind_until(
                &layer.registry,
                layer.policy.as_ref(),
                &model_id,
                profile.request_kind(),
                backend_hint.as_deref(),
                min_context.map(|value| value.min(u64::from(u32::MAX)) as u32),
                stream,
                required_pool
                    .as_ref()
                    .or(preferred_pool.as_ref())
                    .map(|pool| pool.as_str()),
                required_pool.is_some(),
                worker_path,
                body,
                Some(profile.deadline),
            )
            .await
    };
    layer
        .dispatcher
        .record_request_result(resp.status().is_success());

    // Add X-Reason header for dispatcher-level errors (PRD §FR-3.3).
    if !resp.headers().contains_key("x-reason") {
        let reason: Option<&'static str> = match resp.status() {
            StatusCode::SERVICE_UNAVAILABLE => Some("no_eligible_worker"),
            StatusCode::BAD_GATEWAY => Some("worker_crash"),
            _ => None,
        };
        if let Some(r) = reason {
            resp.headers_mut().insert(
                HeaderName::from_static("x-reason"),
                HeaderValue::from_static(r),
            );
        }
    }

    // For streaming responses the body is delivered lazily after this handler
    // returns. Carry the queue/tenant permits AND the drain admission guard
    // inside the body stream so both concurrency accounting and drain
    // inflight stay held until the stream ends or the client disconnects.
    // Non-streaming path: forward() buffers the full body before returning, so
    // dropping permits here still covers the full inference duration.
    if stream {
        let (parts, old_body) = resp.into_parts();
        let guarded = futures::stream::unfold(
            (
                old_body.into_data_stream(),
                Some((permit, tenant_permit, admission_guard)),
            ),
            |(mut data_stream, permits): (BodyDataStream, Option<_>)| async move {
                use futures::StreamExt as _;
                match data_stream.next().await {
                    Some(chunk) => Some((chunk, (data_stream, permits))),
                    None => {
                        drop(permits);
                        None
                    }
                }
            },
        );
        axum::response::Response::from_parts(parts, Body::from_stream(guarded))
    } else {
        drop(permit);
        drop(tenant_permit);
        drop(admission_guard);
        resp
    }
}

fn request_has_images(messages: &[InputMessage]) -> bool {
    messages
        .iter()
        .any(|msg| msg.content.as_ref().is_some_and(MessageContent::has_images))
}

fn declared_minimum_context(headers: &HeaderMap) -> Result<Option<u64>, String> {
    const MAX_DECLARED_CONTEXT_TOKENS: u64 = 16 * 1024 * 1024;
    let Some(raw) = headers
        .get("x-ax-minimum-context-tokens")
        .and_then(|value| value.to_str().ok())
    else {
        return Ok(None);
    };
    let value = raw
        .trim()
        .parse::<u64>()
        .map_err(|_| "x-ax-minimum-context-tokens must be a positive integer".to_string())?;
    if value == 0 || value > MAX_DECLARED_CONTEXT_TOKENS {
        return Err(format!(
            "x-ax-minimum-context-tokens must be between 1 and {MAX_DECLARED_CONTEXT_TOKENS}"
        ));
    }
    Ok(Some(value))
}

fn declared_request_timeout(
    headers: &HeaderMap,
    configured_maximum: std::time::Duration,
) -> Result<std::time::Duration, String> {
    let Some(raw) = headers
        .get("x-ax-request-timeout-ms")
        .and_then(|value| value.to_str().ok())
    else {
        return Ok(configured_maximum);
    };
    let milliseconds = raw
        .trim()
        .parse::<u64>()
        .map_err(|_| "x-ax-request-timeout-ms must be a positive integer".to_string())?;
    if milliseconds == 0 {
        return Err("x-ax-request-timeout-ms must be greater than zero".into());
    }
    Ok(std::time::Duration::from_millis(milliseconds).min(configured_maximum))
}

fn derive_cache_affinity_key(
    headers: &HeaderMap,
    tenant: &str,
    secret: Option<&str>,
) -> Result<Option<u64>, String> {
    const MAX_AFFINITY_HINT_BYTES: usize = 256;
    let Some(raw) = headers.get("x-ax-cache-affinity") else {
        return Ok(None);
    };
    let hint = raw
        .to_str()
        .map_err(|_| "x-ax-cache-affinity must be valid visible ASCII".to_string())?
        .trim();
    if hint.is_empty() || hint.len() > MAX_AFFINITY_HINT_BYTES {
        return Err(format!(
            "x-ax-cache-affinity must contain 1 to {MAX_AFFINITY_HINT_BYTES} bytes"
        ));
    }
    let secret = secret.ok_or_else(|| {
        "cache affinity is disabled; configure AXS_CACHE_AFFINITY_SECRET".to_string()
    })?;

    let mut digest = Sha256::new();
    digest.update(b"ax-serving-cache-affinity-v1\0");
    digest.update((secret.len() as u64).to_be_bytes());
    digest.update(secret.as_bytes());
    digest.update((tenant.len() as u64).to_be_bytes());
    digest.update(tenant.as_bytes());
    digest.update((hint.len() as u64).to_be_bytes());
    digest.update(hint.as_bytes());
    let digest = digest.finalize();
    Ok(Some(u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 has at least 8 bytes"),
    )))
}

#[allow(clippy::too_many_arguments)]
fn build_request_profile(
    request_id: ax_serving_protocol::RequestId,
    worker_path: &str,
    model_id: &str,
    meta: &ProxyRequestMeta,
    body_bytes: usize,
    max_output_tokens: Option<u32>,
    minimum_context_tokens: Option<u64>,
    runtime_hint: Option<String>,
    required_pool: Option<PoolId>,
    preferred_pool: Option<PoolId>,
    priority: PriorityClass,
    headers: &HeaderMap,
    cache_affinity_secret: Option<&str>,
    request_timeout: std::time::Duration,
) -> anyhow::Result<RequestProfile> {
    let operation = match worker_path {
        "/v1/chat/completions" => Operation::chat_completions(),
        "/v1/completions" => Operation::text_completions(),
        "/v1/embeddings" => Operation::embeddings(),
        _ => anyhow::bail!("unsupported inference operation"),
    };
    let logical_model = LogicalModelId::new(model_id.to_string())?;
    let tenant = headers
        .get("x-ax-project")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("default");
    let tenant_id = TenantId::new(tenant.to_string())?;
    let cache_affinity_key = derive_cache_affinity_key(headers, tenant, cache_affinity_secret)
        .map_err(anyhow::Error::msg)?;

    let mut modalities = BTreeSet::from(["text".to_string()]);
    let mut required_capabilities = BTreeSet::new();
    if request_has_images(&meta.messages) {
        modalities.insert("image".into());
        required_capabilities.insert(ProtocolCapability::new("inference.vision")?);
    }
    if meta.tools.as_ref().is_some_and(|tools| {
        tools.as_array().is_none_or(|values| !values.is_empty()) && !tools.is_null()
    }) {
        required_capabilities.insert(ProtocolCapability::new("inference.tools")?);
    }
    if meta.response_format.as_ref().is_some_and(|format| {
        format
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind != "text")
    }) {
        required_capabilities.insert(ProtocolCapability::new("inference.structured-output")?);
    }

    Ok(RequestProfile {
        request_id,
        operation,
        logical_model,
        stream: meta.stream,
        max_output_tokens: max_output_tokens.map(u64::from),
        body_bytes,
        message_count: (!meta.messages.is_empty()).then_some(meta.messages.len()),
        modalities,
        required_capabilities,
        minimum_context_tokens,
        tenant_id,
        priority,
        cache_affinity_key,
        required_pool,
        preferred_pool,
        decision: Default::default(),
        runtime_hint,
        deadline: tokio::time::Instant::now() + request_timeout,
    })
}

fn validate_proxy_request_shape(
    worker_path: &str,
    meta: &ProxyRequestMeta,
) -> Result<Option<EmbeddingsInput>, (StatusCode, String)> {
    match worker_path {
        "/v1/chat/completions" => {
            validate_proxy_max_tokens(meta.max_tokens)?;
            validate_proxy_chat_messages(&meta.messages)?;
            Ok(None)
        }
        "/v1/completions" => {
            validate_proxy_max_tokens(meta.max_tokens)?;
            validate_proxy_prompt(meta.prompt.as_deref())?;
            Ok(None)
        }
        "/v1/embeddings" => {
            let Some(input) = meta.input.as_ref() else {
                return Err((StatusCode::BAD_REQUEST, "missing field: input".to_string()));
            };
            let input = serde_json::from_value::<EmbeddingsInput>(input.clone()).map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    "invalid embedding input".to_string(),
                )
            })?;
            validate_proxy_embeddings_input(&input)?;
            Ok(Some(input))
        }
        _ => Ok(None),
    }
}

fn validate_proxy_max_tokens(max_tokens: Option<u32>) -> Result<(), (StatusCode, String)> {
    if matches!(max_tokens, Some(0)) {
        return Err((
            StatusCode::BAD_REQUEST,
            "max_tokens must be >= 1".to_string(),
        ));
    }
    if matches!(max_tokens, Some(n) if n > MAX_MAX_TOKENS) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("max_tokens exceeds limit ({MAX_MAX_TOKENS})"),
        ));
    }
    Ok(())
}

fn validate_proxy_chat_messages(messages: &[InputMessage]) -> Result<(), (StatusCode, String)> {
    if messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "messages must not be empty".to_string(),
        ));
    }
    if messages.len() > MAX_MESSAGES {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("too many messages (max {MAX_MESSAGES})"),
        ));
    }
    for message in messages {
        if message.content.is_none()
            && !(message.role.eq_ignore_ascii_case("assistant") && message.tool_calls.is_some())
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "message content is required unless assistant tool_calls are present".to_string(),
            ));
        }
        if message
            .content
            .as_ref()
            .is_some_and(|content| content.byte_len() > MAX_CONTENT_BYTES)
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "message content exceeds 32 KB limit".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_proxy_prompt(prompt: Option<&str>) -> Result<(), (StatusCode, String)> {
    let Some(prompt) = prompt else {
        return Err((
            StatusCode::BAD_REQUEST,
            "prompt must not be empty".to_string(),
        ));
    };
    if prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "prompt must not be empty".to_string(),
        ));
    }
    if prompt.len() > MAX_CONTENT_BYTES {
        return Err((
            StatusCode::BAD_REQUEST,
            "prompt exceeds 32 KB limit".to_string(),
        ));
    }
    Ok(())
}

fn validate_proxy_embeddings_input(input: &EmbeddingsInput) -> Result<(), (StatusCode, String)> {
    match input {
        EmbeddingsInput::One(text) => validate_proxy_embedding_text(text, 0),
        EmbeddingsInput::Many(texts) => {
            if texts.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "input must not be empty".to_string(),
                ));
            }
            if texts.len() > MAX_EMBEDDING_INPUTS {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("too many embedding inputs (max {MAX_EMBEDDING_INPUTS})"),
                ));
            }
            let mut total_bytes = 0usize;
            for (idx, text) in texts.iter().enumerate() {
                validate_proxy_embedding_text(text, idx)?;
                total_bytes = total_bytes.saturating_add(text.len());
            }
            if total_bytes > MAX_EMBEDDING_TOTAL_BYTES {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "embedding input text exceeds total limit of {MAX_EMBEDDING_TOTAL_BYTES} bytes"
                    ),
                ));
            }
            Ok(())
        }
        EmbeddingsInput::OneTokens(tokens) => validate_proxy_embedding_tokens(tokens, 0),
        EmbeddingsInput::ManyTokens(seqs) => {
            if seqs.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "input must not be empty".to_string(),
                ));
            }
            if seqs.len() > MAX_EMBEDDING_INPUTS {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("too many embedding inputs (max {MAX_EMBEDDING_INPUTS})"),
                ));
            }
            let mut total_tokens = 0usize;
            for (idx, tokens) in seqs.iter().enumerate() {
                validate_proxy_embedding_tokens(tokens, idx)?;
                total_tokens = total_tokens.saturating_add(tokens.len());
            }
            if total_tokens > MAX_EMBEDDING_TOTAL_TOKENS {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "embedding token input exceeds total limit of {MAX_EMBEDDING_TOTAL_TOKENS}"
                    ),
                ));
            }
            Ok(())
        }
    }
}

fn validate_proxy_embedding_text(text: &str, index: usize) -> Result<(), (StatusCode, String)> {
    if text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("embedding input at index {index} must not be empty"),
        ));
    }
    if text.len() > MAX_CONTENT_BYTES {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("embedding input at index {index} exceeds {MAX_CONTENT_BYTES} bytes"),
        ));
    }
    Ok(())
}

fn validate_proxy_embedding_tokens(
    tokens: &[u32],
    index: usize,
) -> Result<(), (StatusCode, String)> {
    if tokens.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("embedding token input at index {index} must not be empty"),
        ));
    }
    if tokens.len() > MAX_EMBEDDING_TOTAL_TOKENS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "embedding token input at index {index} exceeds {MAX_EMBEDDING_TOTAL_TOKENS} tokens"
            ),
        ));
    }
    Ok(())
}

fn validate_proxy_model_id(model: Option<String>) -> Result<String, (StatusCode, String)> {
    let Some(model) = model else {
        return Err((StatusCode::BAD_REQUEST, "missing field: model".to_string()));
    };
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "model must not be empty".to_string(),
        ));
    }
    if model != trimmed {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "model contains unsupported whitespace".to_string(),
        ));
    }
    LogicalModelId::new(model.clone()).map_err(|error| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("invalid model identifier: {error}"),
        )
    })?;
    Ok(model)
}

fn validate_dispatch_hint(hint: Option<String>) -> Result<Option<String>, String> {
    let Some(raw) = hint else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }

    if BackendKind::parse(trimmed) != BackendKind::Auto
        || RuntimeKind::parse(trimmed) != RuntimeKind::Unknown
    {
        return Ok(Some(trimmed.to_ascii_lowercase()));
    }

    Err(format!(
        "invalid backend/runtime hint; expected native, ax_engine, llama_cpp, sglang, vllm, or auto but got {trimmed}"
    ))
}

fn fairness_client_key(headers: &HeaderMap, peer_addr: Option<SocketAddr>) -> String {
    if let Some(auth) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        auth.hash(&mut hasher);
        return format!("auth:{:016x}", hasher.finish());
    }
    if let Some(peer_ip) = peer_addr.map(|addr| addr.ip()) {
        return format!("ip:{peer_ip}");
    }
    "anonymous".to_string()
}

#[derive(Deserialize)]
pub(super) struct AuditQuery {
    #[serde(default = "default_audit_limit")]
    limit: usize,
}

pub(super) fn orchestrator_startup_report_value(
    layer: &Arc<OrchestratorLayer>,
) -> serde_json::Value {
    serde_json::json!({
        "service": "orchestrator",
        "status": "ok",
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "license": layer.license.to_json(),
        "runtime": {
            "host": layer.config.host,
            "port": layer.config.port,
            "internal_bind_addr": layer.config.internal_bind_addr,
            "allowed_node_cidrs": layer.config.allowed_node_cidrs,
            "internal_port": layer.config.internal_port,
            "dispatch_policy": layer.config.dispatch_policy,
            "deployment_mode": layer.config.deployment_mode,
            "telemetry_stale_ms": layer.config.telemetry_stale_ms,
            "max_dispatch_attempts": layer.config.max_dispatch_attempts,
            "dispatch_auth_configured": layer.config.dispatch_token.is_some(),
            "tls_profile": layer.config.tls_profile,
            "gateway_id": layer.config.gateway_id,
            "fleet_store": layer.fleet_store.kind(),
            "worker_heartbeat_ms": layer.config.worker_heartbeat_ms,
            "worker_ttl_ms": layer.config.worker_ttl_ms,
            "request_timeout_secs": layer.config.request_timeout_secs,
            "first_byte_timeout_ms": layer.config.first_byte_timeout_ms,
            "stream_idle_timeout_ms": layer.config.stream_idle_timeout_ms,
            "global_queue_max": layer.config.global_queue_max,
            "global_queue_depth": layer.config.global_queue_depth,
            "global_queue_wait_ms": layer.config.global_queue_wait_ms,
        },
        "dispatch_runtime": {},
        "project_policy": project_policy::summary_json(&layer.project_policy),
        "governance": {
            "project_policy_enabled": layer.project_policy.enabled,
        }
    })
}

// ── Route handlers ────────────────────────────────────────────────────────────

pub(super) async fn proxy_chat_completions(
    State(layer): State<Arc<OrchestratorLayer>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    request_id: Option<Extension<AxRequestId>>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(
        layer,
        Some(peer_addr),
        headers,
        body,
        "/v1/chat/completions",
        request_id
            .map(|Extension(value)| value.0)
            .unwrap_or_default(),
    )
    .await
}

pub(super) async fn proxy_completions(
    State(layer): State<Arc<OrchestratorLayer>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    request_id: Option<Extension<AxRequestId>>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(
        layer,
        Some(peer_addr),
        headers,
        body,
        "/v1/completions",
        request_id
            .map(|Extension(value)| value.0)
            .unwrap_or_default(),
    )
    .await
}

pub(super) async fn proxy_embeddings(
    State(layer): State<Arc<OrchestratorLayer>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    request_id: Option<Extension<AxRequestId>>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(
        layer,
        Some(peer_addr),
        headers,
        body,
        "/v1/embeddings",
        request_id
            .map(|Extension(value)| value.0)
            .unwrap_or_default(),
    )
    .await
}

pub(super) async fn proxy_models(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
    let deployment_catalog = layer.deployment_catalog.snapshot();
    let mut models: Vec<String> = if deployment_catalog.mode() == DeploymentMode::Explicit {
        deployment_catalog
            .logical_models()
            .into_iter()
            .map(|model| model.id.to_string())
            .collect()
    } else {
        layer
            .registry
            .list_all()
            .iter()
            // Mirror dispatch eligibility: only healthy, non-draining workers.
            .filter(|worker| !worker.drain && worker.health == "healthy")
            .flat_map(|worker| worker.capability_descriptor.models.iter().cloned())
            .collect()
    };
    models.sort_unstable();
    models.dedup();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(serde_json::json!({
        "object": "list",
        "data": models.iter().map(|id| serde_json::json!({
            "id": id,
            "object": "model",
            "created": now,
            "owned_by": "ax-serving",
        })).collect::<Vec<_>>()
    }))
}

pub(super) async fn proxy_health(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    // "ok" only when at least one worker can actually accept requests:
    // healthy AND not draining. A fully-draining pool shows as "degraded" even
    // if all workers are technically Healthy, because the dispatcher would
    // return 503 for every request.
    let eligible = layer.registry.eligible_healthy_count();
    let status = if eligible > 0 { "ok" } else { "degraded" };
    let qm = &layer.queue.metrics;
    Json(serde_json::json!({
        "status": status,
        "workers": {
            "total": healthy + unhealthy,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "draining": draining,
            "eligible": eligible,
        },
        "queue": {
            "active": layer.queue.active(),
            "queued": layer.queue.queued(),
            "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
            "shed_total": qm.shed_total.load(Ordering::Relaxed),
            "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
        }
    }))
}

pub(super) async fn proxy_liveness(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(layer.ops.live_response())
}

pub(super) async fn proxy_readiness(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    let eligible = layer.registry.eligible_healthy_count();
    let assessment = layer
        .ops
        .ready_assessment(super::fleet_state::unix_time_millis(), eligible);
    let mut body = serde_json::json!({
        "status": assessment.status,
        "fleet_store": assessment.fleet_store,
        "draining": assessment.draining,
    });
    if let Some(reason) = &assessment.reason {
        body["reason"] = serde_json::json!(reason);
    }
    if let Some(retry) = assessment.retry_after_seconds {
        body["retry_after_seconds"] = serde_json::json!(retry);
    }
    // Compatibility field for legacy consumers during migration.
    body["eligible_workers"] = serde_json::json!(eligible);
    let mut response = (
        if assessment.ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(body),
    )
        .into_response();
    if let Some(retry) = assessment.retry_after_seconds
        && let Ok(value) = HeaderValue::from_str(&retry.to_string())
    {
        response.headers_mut().insert(header::RETRY_AFTER, value);
    }
    response
}

pub(super) async fn proxy_routability(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    let eligible = layer.registry.eligible_healthy_count();
    let assessment = layer.ops.routable_assessment(eligible);
    // Unauthenticated summary intentionally omits worker IDs and model details.
    let body = if assessment.routable {
        serde_json::json!({ "status": "routable" })
    } else {
        serde_json::json!({
            "status": "not_routable",
            "retry_after_seconds": assessment.retry_after_seconds.unwrap_or(5),
        })
    };
    let mut response = (
        if assessment.routable {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(body),
    )
        .into_response();
    if let Some(retry) = assessment.retry_after_seconds
        && let Ok(value) = HeaderValue::from_str(&retry.to_string())
    {
        response.headers_mut().insert(header::RETRY_AFTER, value);
    }
    response
}

pub(super) async fn proxy_metrics(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    let workers = layer.registry.list_all();
    let total_inflight: usize = workers.iter().map(|w| w.inflight).sum();
    let qm = &layer.queue.metrics;
    let dispatch = layer.dispatcher.metrics();

    Json(serde_json::json!({
        "mode": "direct",
        "policy": layer.config.dispatch_policy,
        "workers": {
            "healthy": healthy,
            "unhealthy": unhealthy,
            "draining": draining,
        },
        "total_inflight": total_inflight,
        "reroute_total": layer.dispatcher.reroutes(),
        "requests": {
            "total": dispatch.requests_total,
            "attempts_total": dispatch.attempts_total,
            "completed_total": dispatch.completed_total,
            "failed_total": dispatch.failed_total,
            "cancelled_total": dispatch.cancelled_total,
            "retried_total": dispatch.retries_total,
        },
        "queue": {
            "active": layer.queue.active(),
            "queued": layer.queue.queued(),
            "permit_total": qm.permit_total.load(Ordering::Relaxed),
            "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
            "shed_total": qm.shed_total.load(Ordering::Relaxed),
            "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
        },
        "worker_detail": workers,
    }))
}

fn prometheus_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn prometheus_metric(buf: &mut String, name: &str, help: &str, kind: &str, value: u64) {
    use std::fmt::Write as _;

    let _ = writeln!(buf, "# HELP {name} {help}");
    let _ = writeln!(buf, "# TYPE {name} {kind}");
    let _ = writeln!(buf, "{name} {value}");
}

fn prometheus_latency_histogram(
    buf: &mut String,
    name: &str,
    help: &str,
    snapshot: &super::direct::LatencyHistogramSnapshot,
) {
    use std::fmt::Write as _;

    let _ = writeln!(buf, "# HELP {name} {help}");
    let _ = writeln!(buf, "# TYPE {name} histogram");
    for (upper_us, count) in super::direct::GATEWAY_LATENCY_BUCKETS_US
        .iter()
        .zip(snapshot.cumulative_buckets.iter())
    {
        if *upper_us == u64::MAX {
            let _ = writeln!(buf, "{name}_bucket{{le=\"+Inf\"}} {count}");
        } else {
            let upper_seconds = *upper_us as f64 / 1_000_000.0;
            let _ = writeln!(buf, "{name}_bucket{{le=\"{upper_seconds:.6}\"}} {count}");
        }
    }
    let _ = writeln!(
        buf,
        "{name}_sum {:.6}",
        snapshot.sum_us as f64 / 1_000_000.0
    );
    let _ = writeln!(buf, "{name}_count {}", snapshot.count);
}

pub(super) async fn proxy_prometheus_metrics(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    use std::fmt::Write as _;

    let (healthy, unhealthy, draining) = layer.registry.counts();
    let eligible = layer.registry.eligible_healthy_count();
    let workers = layer.registry.list_all();
    let total_inflight = workers.iter().map(|worker| worker.inflight).sum::<usize>();
    let queue = &layer.queue.metrics;
    let dispatch = layer.dispatcher.metrics();
    let mut body = String::with_capacity(4096);

    let _ = writeln!(
        body,
        "# HELP axs_gateway_info Static gateway build and configuration identity.\n# TYPE axs_gateway_info gauge\naxs_gateway_info{{gateway_id=\"{}\",fleet_store=\"{}\",policy=\"{}\"}} 1",
        prometheus_label(&layer.config.gateway_id),
        prometheus_label(layer.fleet_store.kind()),
        prometheus_label(&layer.config.dispatch_policy),
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_requests_total",
        "Inference requests that reached gateway dispatch.",
        "counter",
        dispatch.requests_total,
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_dispatch_attempts_total",
        "Gateway-to-agent dispatch attempts.",
        "counter",
        dispatch.attempts_total,
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_requests_completed_total",
        "Successful responses consumed through completion.",
        "counter",
        dispatch.completed_total,
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_requests_failed_total",
        "Requests ending in an error response or stream failure.",
        "counter",
        dispatch.failed_total,
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_requests_cancelled_total",
        "Successful response streams dropped before completion.",
        "counter",
        dispatch.cancelled_total,
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_retries_total",
        "Safe pre-commit retries after connect failure or typed non-admission.",
        "counter",
        dispatch.retries_total,
    );
    prometheus_latency_histogram(
        &mut body,
        "axs_gateway_endpoint_selection_duration_seconds",
        "Time spent resolving and selecting an eligible runtime endpoint.",
        &dispatch.endpoint_selection,
    );
    prometheus_latency_histogram(
        &mut body,
        "axs_gateway_response_headers_duration_seconds",
        "Gateway-to-agent dispatch latency through upstream response headers.",
        &dispatch.response_headers,
    );
    prometheus_latency_histogram(
        &mut body,
        "axs_gateway_attempt_duration_seconds",
        "Dispatch-attempt duration through complete blocking body or streaming termination.",
        &dispatch.attempt_duration,
    );
    prometheus_latency_histogram(
        &mut body,
        "axs_gateway_time_to_first_byte_seconds",
        "Dispatch-attempt latency through the first streamed response bytes.",
        &dispatch.time_to_first_byte,
    );
    prometheus_latency_histogram(
        &mut body,
        "axs_gateway_stream_duration_seconds",
        "Streaming dispatch-attempt duration through completion, failure, or cancellation.",
        &dispatch.stream_duration,
    );
    let _ = writeln!(
        body,
        "# HELP axs_gateway_endpoint_selections_total Endpoint selection outcomes with bounded reason labels.\n# TYPE axs_gateway_endpoint_selections_total counter\naxs_gateway_endpoint_selections_total{{outcome=\"selected\"}} {}\naxs_gateway_endpoint_selections_total{{outcome=\"no_candidate\"}} {}\naxs_gateway_endpoint_selections_total{{outcome=\"at_capacity\"}} {}\naxs_gateway_endpoint_selections_total{{outcome=\"error\"}} {}",
        dispatch.selection_selected_total,
        dispatch.selection_no_candidate_total,
        dispatch.selection_at_capacity_total,
        dispatch.selection_error_total,
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_admitted_total",
        "Requests admitted by the global concurrency queue.",
        "counter",
        queue.permit_total.load(Ordering::Relaxed),
    );
    let _ = writeln!(
        body,
        "# HELP axs_gateway_rejected_total Requests rejected before dispatch.\n# TYPE axs_gateway_rejected_total counter\naxs_gateway_rejected_total{{reason=\"queue_full\"}} {}\naxs_gateway_rejected_total{{reason=\"shed\"}} {}\naxs_gateway_rejected_total{{reason=\"queue_timeout\"}} {}",
        queue.rejected_total.load(Ordering::Relaxed),
        queue.shed_total.load(Ordering::Relaxed),
        queue.timeout_total.load(Ordering::Relaxed),
    );
    let _ = writeln!(
        body,
        "# HELP axs_gateway_queue_requests Current gateway admission state.\n# TYPE axs_gateway_queue_requests gauge\naxs_gateway_queue_requests{{state=\"active\"}} {}\naxs_gateway_queue_requests{{state=\"queued\"}} {}",
        layer.queue.active(),
        layer.queue.queued(),
    );
    let _ = writeln!(
        body,
        "# HELP axs_gateway_workers Current worker membership by bounded state.\n# TYPE axs_gateway_workers gauge\naxs_gateway_workers{{state=\"healthy\"}} {healthy}\naxs_gateway_workers{{state=\"unhealthy\"}} {unhealthy}\naxs_gateway_workers{{state=\"draining\"}} {draining}\naxs_gateway_workers{{state=\"eligible\"}} {eligible}"
    );
    prometheus_metric(
        &mut body,
        "axs_gateway_worker_inflight",
        "Aggregate requests currently reserved on workers.",
        "gauge",
        total_inflight as u64,
    );

    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

pub(super) async fn proxy_admin_status(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<axum::extract::Extension<crate::auth::RequestId>>,
) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    let workers = layer.registry.list_all();
    let total_workers = workers.len();
    let total_inflight: usize = workers.iter().map(|w| w.inflight).sum();
    let total_active_sequences: usize = workers.iter().map(|w| w.active_sequences).sum();
    let runtime_buckets = runtime_fleet_buckets(&workers);
    let eligible = layer.registry.eligible_healthy_count();
    let qm = &layer.queue.metrics;

    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "mode": "direct",
        "status": if eligible > 0 { "ok" } else { "degraded" },
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "dispatch_policy": layer.config.dispatch_policy,
        "gateway": {
            "id": layer.config.gateway_id,
            "fleet_store": layer.fleet_store.kind(),
            "tls_profile": layer.config.tls_profile,
        },
        "license": layer.license.to_json(),
        "workers": {
            "total": total_workers,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "draining": draining,
            "eligible": eligible,
            "total_inflight": total_inflight,
            "total_active_sequences": total_active_sequences,
            "runtimes": runtime_buckets,
        },
        "queue": {
            "active": layer.queue.active(),
            "queued": layer.queue.queued(),
            "permit_total": qm.permit_total.load(Ordering::Relaxed),
            "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
            "shed_total": qm.shed_total.load(Ordering::Relaxed),
            "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
        },
        "dispatcher": {
            "reroute_total": layer.dispatcher.reroutes(),
            "request_timeout_secs": layer.config.request_timeout_secs,
            "retry_after_secs": layer.retry_after_secs,
        }
    }))
}

pub(super) async fn proxy_admin_startup_report(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(orchestrator_startup_report_value(&layer))
}

pub(super) async fn proxy_admin_diagnostics(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    let workers = layer.registry.list_all();
    let total_inflight: usize = workers.iter().map(|w| w.inflight).sum();
    let total_active_sequences: usize = workers.iter().map(|w| w.active_sequences).sum();
    let runtime_buckets = runtime_fleet_buckets(&workers);
    let runtime_diagnostics = runtime_diagnostics(&workers);
    let eligible = layer.registry.eligible_healthy_count();
    let qm = &layer.queue.metrics;
    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "startup_report": orchestrator_startup_report_value(&layer),
        "health": {
            "status": if eligible > 0 { "ok" } else { "degraded" },
            "workers": {
                "total": healthy + unhealthy,
                "healthy": healthy,
                "unhealthy": unhealthy,
                "draining": draining,
                "eligible": eligible,
            },
            "queue": {
                "active": layer.queue.active(),
                "queued": layer.queue.queued(),
                "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
                "shed_total": qm.shed_total.load(Ordering::Relaxed),
                "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
            }
        },
        "metrics": {
            "mode": "direct",
            "policy": layer.config.dispatch_policy,
            "workers": {
                "healthy": healthy,
                "unhealthy": unhealthy,
                "draining": draining,
                "eligible": eligible,
                "total_inflight": total_inflight,
                "total_active_sequences": total_active_sequences,
                "runtimes": runtime_buckets,
            },
            "runtime_diagnostics": runtime_diagnostics,
            "reroute_total": layer.dispatcher.reroutes(),
            "queue": {
                "active": layer.queue.active(),
                "queued": layer.queue.queued(),
                "permit_total": qm.permit_total.load(Ordering::Relaxed),
                "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
                "shed_total": qm.shed_total.load(Ordering::Relaxed),
                "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
            }
        },
        "runtime_diagnostics": runtime_diagnostics,
        "workers": workers,
        "audit_tail": layer.audit.tail(50),
    }))
}

pub(super) async fn proxy_admin_policy(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(project_policy::summary_json(&layer.project_policy))
}

pub(super) async fn proxy_admin_audit(
    State(layer): State<Arc<OrchestratorLayer>>,
    Query(query): Query<AuditQuery>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "events": layer.audit.tail(query.limit.clamp(1, 200)),
    }))
}

pub(super) async fn proxy_admin_decisions(
    State(layer): State<Arc<OrchestratorLayer>>,
    Query(query): Query<AuditQuery>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "records": layer.dispatcher.decision_records(query.limit),
    }))
}

pub(super) async fn proxy_admin_fleet(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    let workers = layer.registry.list_all();
    let mut pools = serde_json::Map::new();
    let mut node_classes = serde_json::Map::new();
    let mut backends = serde_json::Map::new();
    let mut runtimes = serde_json::Map::new();

    for worker in &workers {
        accumulate_fleet_bucket(
            &mut pools,
            worker.worker_pool.as_deref().unwrap_or("default"),
            worker,
        );
        accumulate_fleet_bucket(
            &mut node_classes,
            worker.node_class.as_deref().unwrap_or("unknown"),
            worker,
        );
        accumulate_fleet_bucket(&mut backends, &worker.backend, worker);
        accumulate_fleet_bucket(&mut runtimes, &worker.runtime, worker);
    }

    Json(serde_json::json!({
        "gateway_id": layer.config.gateway_id,
        "fleet_store": layer.fleet_store.kind(),
        "total_workers": workers.len(),
        "eligible_workers": layer.registry.eligible_healthy_count(),
        "pools": pools,
        "node_classes": node_classes,
        "backends": backends,
        "runtimes": runtimes,
        "workers": workers,
    }))
}

pub(super) async fn proxy_admin_deployments(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    let catalog = layer.deployment_catalog.snapshot();
    Json(serde_json::json!({
        "mode": catalog.mode(),
        "logical_models": catalog.logical_models(),
        "pools": catalog.pools().collect::<Vec<_>>(),
        "domains": catalog.domains().collect::<Vec<_>>(),
        "deployments": catalog.deployments().collect::<Vec<_>>(),
        "equivalence_classes": catalog.equivalence_classes().collect::<Vec<_>>(),
    }))
}

fn accumulate_fleet_bucket(
    buckets: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    worker: &super::registry::WorkerSnapshot,
) {
    let entry = buckets.entry(key.to_string()).or_insert_with(|| {
        serde_json::json!({
            "workers": 0usize,
            "healthy": 0usize,
            "draining": 0usize,
            "eligible": 0usize,
            "total_inflight": 0usize,
            "total_active_sequences": 0usize,
            "total_queue_depth": 0usize,
            "max_error_rate": 0.0_f64,
        })
    });
    if let Some(obj) = entry.as_object_mut() {
        increment_bucket(obj, "workers", 1_u64);
        if worker.health == "healthy" {
            increment_bucket(obj, "healthy", 1_u64);
        }
        if worker.drain {
            increment_bucket(obj, "draining", 1_u64);
        }
        if worker.health == "healthy" && !worker.drain {
            increment_bucket(obj, "eligible", 1_u64);
        }
        increment_bucket(obj, "total_inflight", worker.inflight as u64);
        increment_bucket(
            obj,
            "total_active_sequences",
            worker.active_sequences as u64,
        );
        increment_bucket(obj, "total_queue_depth", worker.queue_depth as u64);
        let current_max = obj
            .get("max_error_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        if worker.error_rate > current_max {
            obj.insert(
                "max_error_rate".to_string(),
                serde_json::Value::from(worker.error_rate),
            );
        }
    } else {
        warn!(key = %key, "unexpected non-object fleet bucket encountered");
    }
}

fn runtime_fleet_buckets(
    workers: &[super::registry::WorkerSnapshot],
) -> serde_json::Map<String, serde_json::Value> {
    let mut runtimes = serde_json::Map::new();
    for worker in workers {
        accumulate_fleet_bucket(&mut runtimes, &worker.runtime, worker);
    }
    runtimes
}

const RUNTIME_ERROR_RATE_WARN_THRESHOLD: f64 = 0.05;
const RUNTIME_KV_PRESSURE_WARN_THRESHOLD: f64 = 0.90;
const RUNTIME_BATCH_PRESSURE_WARN_THRESHOLD: f64 = 0.90;

#[derive(Default)]
struct RuntimeDiagnostic {
    workers: usize,
    healthy: usize,
    unhealthy: usize,
    draining: usize,
    eligible: usize,
    total_inflight: usize,
    total_active_sequences: usize,
    total_queue_depth: usize,
    max_error_rate: f64,
    models: BTreeSet<String>,
    model_inventory: Vec<serde_json::Value>,
    hardware_classes: BTreeMap<String, usize>,
    node_classes: BTreeMap<String, usize>,
    worker_pools: BTreeMap<String, usize>,
    runtime_modes: BTreeMap<String, usize>,
    supported_operations: BTreeSet<String>,
    runtime_endpoints: BTreeSet<String>,
    missing_runtime_endpoint_workers: Vec<String>,
    unhealthy_workers: Vec<String>,
    draining_workers: Vec<String>,
    compatibility_workers: Vec<String>,
    unknown_runtime_workers: Vec<String>,
    empty_model_inventory_workers: Vec<String>,
    unexpected_hardware_class_workers: Vec<String>,
    high_error_rate_workers: Vec<String>,
    queue_backlog_workers: Vec<String>,
    high_kv_pressure_workers: Vec<String>,
    high_batch_pressure_workers: Vec<String>,
}

impl RuntimeDiagnostic {
    fn observe(&mut self, worker: &super::registry::WorkerSnapshot) {
        self.workers += 1;
        if worker.health == "healthy" {
            self.healthy += 1;
        } else {
            self.unhealthy += 1;
            self.unhealthy_workers.push(worker.id.to_string());
        }
        if worker.drain {
            self.draining += 1;
            self.draining_workers.push(worker.id.to_string());
        }
        if worker.health == "healthy" && !worker.drain {
            self.eligible += 1;
        }
        self.total_inflight += worker.inflight;
        self.total_active_sequences += worker.active_sequences;
        self.total_queue_depth += worker.queue_depth;
        self.max_error_rate = self.max_error_rate.max(worker.error_rate);
        if worker.error_rate >= RUNTIME_ERROR_RATE_WARN_THRESHOLD {
            self.high_error_rate_workers.push(worker.id.to_string());
        }
        if worker.queue_depth >= worker.max_inflight.max(1) {
            self.queue_backlog_workers.push(worker.id.to_string());
        }
        if worker
            .kv_utilization
            .is_some_and(|value| value >= RUNTIME_KV_PRESSURE_WARN_THRESHOLD)
        {
            self.high_kv_pressure_workers.push(worker.id.to_string());
        }
        if worker
            .batch_utilization
            .is_some_and(|value| value >= RUNTIME_BATCH_PRESSURE_WARN_THRESHOLD)
            || (worker.max_batch_size > 0 && worker.active_batch_size >= worker.max_batch_size)
        {
            self.high_batch_pressure_workers.push(worker.id.to_string());
        }

        if worker.capabilities.is_empty() {
            self.empty_model_inventory_workers
                .push(worker.id.to_string());
        }
        for model in &worker.capabilities {
            self.models.insert(model.clone());
        }
        for model in &worker.model_inventory {
            self.model_inventory.push(serde_json::json!({
                "worker_id": worker.id,
                "model_id": model.id.as_str(),
                "runtime": worker.runtime.as_str(),
                "node_class": worker.node_class.as_deref(),
                "hardware_class": worker.hardware_class.as_deref(),
                "max_context": model.max_context,
                "quantization": model.quantization.as_deref(),
                "artifact_format": model.artifact_format.as_deref(),
                "modalities": &model.modalities,
                "supported_operations": &model.supported_operations,
            }));
        }
        if let Some(hardware_class) = worker.hardware_class.as_deref() {
            increment_count(&mut self.hardware_classes, hardware_class);
            if let Some(expected) = expected_hardware_classes(worker.runtime.as_str())
                && !expected.contains(&hardware_class)
            {
                self.unexpected_hardware_class_workers
                    .push(worker.id.to_string());
            }
        } else if expected_hardware_classes(worker.runtime.as_str()).is_some() {
            self.unexpected_hardware_class_workers
                .push(worker.id.to_string());
        }
        if let Some(node_class) = worker.node_class.as_deref() {
            increment_count(&mut self.node_classes, node_class);
        }
        if let Some(worker_pool) = worker.worker_pool.as_deref() {
            increment_count(&mut self.worker_pools, worker_pool);
        }
        if let Some(runtime_mode) = worker.runtime_mode.as_deref() {
            increment_count(&mut self.runtime_modes, runtime_mode);
        }
        for operation in &worker.supported_operations {
            self.supported_operations.insert(operation.clone());
        }
        if let Some(endpoint) = worker.runtime_endpoint.as_deref() {
            self.runtime_endpoints.insert(endpoint.to_string());
        } else {
            self.missing_runtime_endpoint_workers
                .push(worker.id.to_string());
        }

        if worker.runtime == "unknown" {
            self.unknown_runtime_workers.push(worker.id.to_string());
        }
        if worker.runtime_mode.as_deref() == Some("embedded")
            || (worker.runtime == "ax_engine"
                && worker.runtime_mode.is_none()
                && matches!(
                    worker.backend.as_str(),
                    "native" | "auto" | "llama_cpp" | "mlx"
                ))
        {
            self.compatibility_workers.push(worker.id.to_string());
        }
    }

    fn issues(&self) -> Vec<serde_json::Value> {
        let mut issues = Vec::new();
        if self.eligible == 0 {
            issues.push(serde_json::json!({
                "code": "no_eligible_workers",
                "severity": "error",
                "message": "runtime has no healthy non-draining workers"
            }));
        }
        if !self.unhealthy_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "unhealthy_workers",
                "severity": "warning",
                "workers": self.unhealthy_workers
            }));
        }
        if !self.draining_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "draining_workers",
                "severity": "info",
                "workers": self.draining_workers
            }));
        }
        if !self.unknown_runtime_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "unknown_runtime",
                "severity": "warning",
                "workers": self.unknown_runtime_workers
            }));
        }
        if !self.missing_runtime_endpoint_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "missing_runtime_endpoint",
                "severity": "warning",
                "workers": self.missing_runtime_endpoint_workers
            }));
        }
        if !self.empty_model_inventory_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "empty_model_inventory",
                "severity": "warning",
                "workers": self.empty_model_inventory_workers
            }));
        }
        if !self.unexpected_hardware_class_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "unexpected_hardware_class",
                "severity": "warning",
                "workers": self.unexpected_hardware_class_workers
            }));
        }
        if !self.high_error_rate_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "high_runtime_error_rate",
                "severity": "warning",
                "workers": self.high_error_rate_workers,
                "threshold": RUNTIME_ERROR_RATE_WARN_THRESHOLD
            }));
        }
        if !self.queue_backlog_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "runtime_queue_backlog",
                "severity": "warning",
                "workers": self.queue_backlog_workers
            }));
        }
        if !self.high_kv_pressure_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "high_runtime_kv_pressure",
                "severity": "warning",
                "workers": self.high_kv_pressure_workers,
                "threshold": RUNTIME_KV_PRESSURE_WARN_THRESHOLD
            }));
        }
        if !self.high_batch_pressure_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "high_runtime_batch_pressure",
                "severity": "info",
                "workers": self.high_batch_pressure_workers,
                "threshold": RUNTIME_BATCH_PRESSURE_WARN_THRESHOLD
            }));
        }
        if !self.compatibility_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "embedded_compatibility_path",
                "severity": "info",
                "workers": self.compatibility_workers
            }));
        }
        issues
    }

    fn recommended_actions(&self, runtime: &str) -> Vec<serde_json::Value> {
        let mut actions = Vec::new();
        if self.eligible == 0 {
            actions.push(serde_json::json!({
                "action": "restore_runtime_capacity",
                "runtime": runtime,
                "priority": "high",
                "reason": "runtime has no eligible workers",
                "operator_hint": "Start or recover at least one healthy non-draining runtime node for this runtime."
            }));
        }
        if !self.unhealthy_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "replace_unhealthy_workers",
                "runtime": runtime,
                "priority": "high",
                "worker_ids": self.unhealthy_workers,
                "suggested_commands": worker_replacement_commands(&self.unhealthy_workers),
                "operator_hint": "Drain or remove unhealthy workers, restart the runtime node, then verify registration and heartbeat."
            }));
        }
        if !self.draining_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "complete_drain_when_idle",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.draining_workers,
                "suggested_commands": worker_drain_complete_commands(&self.draining_workers),
                "operator_hint": "Wait for inflight requests to reach zero, then call drain-complete before replacement."
            }));
        }
        if !self.missing_runtime_endpoint_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "fix_runtime_endpoint_registration",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.missing_runtime_endpoint_workers,
                "operator_hint": "Restart the adapter with AXS_NODE_RUNTIME_URL or AXS_WORKER_RUNTIME_ENDPOINT set."
            }));
        }
        if !self.empty_model_inventory_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "refresh_model_inventory",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.empty_model_inventory_workers,
                "operator_hint": "Check the runtime /v1/models endpoint and restart the adapter after the model is loaded."
            }));
        }
        if !self.unknown_runtime_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "fix_runtime_class",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.unknown_runtime_workers,
                "operator_hint": "Register the worker with runtime ax_engine or vllm."
            }));
        }
        if !self.unexpected_hardware_class_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "fix_hardware_class",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.unexpected_hardware_class_workers,
                "expected_hardware_classes": expected_hardware_classes(runtime).unwrap_or(&[]),
                "operator_hint": "Restart the adapter with the hardware class expected for this runtime."
            }));
        }
        if !self.high_error_rate_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "investigate_runtime_errors",
                "runtime": runtime,
                "priority": "high",
                "worker_ids": self.high_error_rate_workers,
                "suggested_commands": worker_inspection_commands(&self.high_error_rate_workers),
                "operator_hint": "Check runtime logs and recent failed requests before returning these workers to normal routing."
            }));
        }
        if !self.queue_backlog_workers.is_empty()
            || !self.high_kv_pressure_workers.is_empty()
            || !self.high_batch_pressure_workers.is_empty()
        {
            let pressure_workers = unique_worker_ids([
                &self.queue_backlog_workers,
                &self.high_kv_pressure_workers,
                &self.high_batch_pressure_workers,
            ]);
            actions.push(serde_json::json!({
                "action": "relieve_runtime_pressure",
                "runtime": runtime,
                "priority": "medium",
                "queue_backlog_worker_ids": self.queue_backlog_workers,
                "high_kv_pressure_worker_ids": self.high_kv_pressure_workers,
                "high_batch_pressure_worker_ids": self.high_batch_pressure_workers,
                "suggested_commands": worker_replacement_commands(&pressure_workers),
                "operator_hint": "Reduce admission pressure, add runtime capacity, or drain and replace overloaded nodes."
            }));
        }
        if !self.compatibility_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "migrate_embedded_compatibility_path",
                "runtime": runtime,
                "priority": "low",
                "worker_ids": self.compatibility_workers,
                "suggested_commands": [
                    "ax-serving status --diagnostics --url <gateway-url>",
                    "AXS_EMBEDDED_RUNTIME_POLICY=deny ax-serving-api"
                ],
                "operator_hint": "Move inference to ax-runtime-agent plus ax-engine or vLLM, then set AXS_EMBEDDED_RUNTIME_POLICY=deny in production."
            }));
        }
        actions
    }

    fn to_json(&self, runtime: &str) -> serde_json::Value {
        serde_json::json!({
            "workers": self.workers,
            "healthy": self.healthy,
            "unhealthy": self.unhealthy,
            "draining": self.draining,
            "eligible": self.eligible,
            "total_inflight": self.total_inflight,
            "total_active_sequences": self.total_active_sequences,
            "total_queue_depth": self.total_queue_depth,
            "max_error_rate": self.max_error_rate,
            "models": self.models,
            "model_inventory": self.model_inventory,
            "hardware_classes": self.hardware_classes,
            "node_classes": self.node_classes,
            "worker_pools": self.worker_pools,
            "runtime_modes": self.runtime_modes,
            "supported_operations": self.supported_operations,
            "runtime_endpoints": self.runtime_endpoints,
            "unhealthy_workers": self.unhealthy_workers,
            "draining_workers": self.draining_workers,
            "high_error_rate_workers": self.high_error_rate_workers,
            "queue_backlog_workers": self.queue_backlog_workers,
            "high_kv_pressure_workers": self.high_kv_pressure_workers,
            "high_batch_pressure_workers": self.high_batch_pressure_workers,
            "issues": self.issues(),
            "recommended_actions": self.recommended_actions(runtime),
            "runtime_guidance": runtime_guidance(runtime),
        })
    }
}

fn unique_worker_ids<'a>(groups: impl IntoIterator<Item = &'a Vec<String>>) -> Vec<String> {
    groups
        .into_iter()
        .flat_map(|group| group.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn worker_inspection_commands(worker_ids: &[String]) -> Vec<String> {
    worker_ids
        .iter()
        .map(|id| format!("ax-serving workers get {id} --url <gateway-url>"))
        .collect()
}

fn worker_drain_complete_commands(worker_ids: &[String]) -> Vec<String> {
    worker_ids
        .iter()
        .map(|id| format!("ax-serving workers drain {id} --complete-when-idle --url <gateway-url>"))
        .collect()
}

fn worker_replacement_commands(worker_ids: &[String]) -> Vec<String> {
    worker_ids
        .iter()
        .flat_map(|id| {
            [
                format!("ax-serving workers drain {id} --complete-when-idle --url <gateway-url>"),
                "start or restart the replacement ax-runtime-agent node".to_string(),
                "ax-serving status --diagnostics --url <gateway-url>".to_string(),
            ]
        })
        .collect()
}

fn expected_hardware_classes(runtime: &str) -> Option<&'static [&'static str]> {
    match runtime {
        "ax_engine" => Some(&["mac"]),
        "vllm" => Some(&["pc-cuda", "thor"]),
        _ => None,
    }
}

fn runtime_guidance(runtime: &str) -> serde_json::Value {
    match runtime {
        "ax_engine" => serde_json::json!({
            "runtime_owner": "ax-engine",
            "expected_hardware_classes": ["mac"],
            "adapter": "ax-runtime-agent",
            "required_registration": {
                "runtime": "ax_engine",
                "hardware_class": "mac"
            },
            "operator_checks": [
                "runtime endpoint exposes /health",
                "runtime endpoint exposes /v1/models",
                "adapter reports ax_runtime_* metrics when available",
                "embedded compatibility workers should be migrated before production"
            ]
        }),
        "vllm" => serde_json::json!({
            "runtime_owner": "vLLM",
            "expected_hardware_classes": ["pc-cuda", "thor"],
            "adapter": "ax-runtime-agent",
            "required_registration": {
                "runtime": "vllm",
                "hardware_class": "pc-cuda or thor"
            },
            "operator_checks": [
                "vLLM OpenAI-compatible endpoint exposes /health",
                "vLLM OpenAI-compatible endpoint exposes /v1/models",
                "adapter reports runtime endpoint and supported operations",
                "PC CUDA and Thor placement should be represented by hardware_class and worker_pool"
            ]
        }),
        _ => serde_json::json!({
            "runtime_owner": "unknown",
            "expected_hardware_classes": [],
            "adapter": "unknown",
            "operator_checks": [
                "register the node with runtime ax_engine or vllm",
                "verify the adapter follows the AX Serving node contract"
            ]
        }),
    }
}

fn increment_count(counts: &mut BTreeMap<String, usize>, key: &str) {
    *counts.entry(key.to_string()).or_default() += 1;
}

fn runtime_diagnostics(workers: &[super::registry::WorkerSnapshot]) -> serde_json::Value {
    let mut diagnostics = BTreeMap::<String, RuntimeDiagnostic>::new();
    for worker in workers {
        diagnostics
            .entry(worker.runtime.clone())
            .or_default()
            .observe(worker);
    }

    let mut runtimes = serde_json::Map::new();
    let mut issues = Vec::new();
    let mut recommended_actions = Vec::new();
    if workers.is_empty() {
        issues.push(serde_json::json!({
            "code": "no_workers_registered",
            "severity": "error",
            "message": "no runtime nodes are registered"
        }));
        recommended_actions.push(serde_json::json!({
            "action": "register_runtime_nodes",
            "priority": "high",
            "reason": "no runtime nodes are registered",
            "operator_hint": "Start ax-serving-api and register ax-runtime-agent nodes for ax_engine or vllm."
        }));
    }
    for (runtime, diagnostic) in diagnostics {
        let runtime_issues = diagnostic.issues();
        for issue in &runtime_issues {
            issues.push(serde_json::json!({
                "runtime": runtime,
                "code": issue["code"],
                "severity": issue["severity"],
            }));
        }
        recommended_actions.extend(diagnostic.recommended_actions(&runtime));
        runtimes.insert(runtime.clone(), diagnostic.to_json(&runtime));
    }

    serde_json::json!({
        "runtimes": runtimes,
        "issues": issues,
        "recommended_actions": recommended_actions,
    })
}

fn increment_bucket(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    amount: impl Into<u64>,
) {
    let amount = amount.into();
    let current = obj.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    obj.insert(key.to_string(), serde_json::json!(current + amount));
}

pub(super) async fn proxy_list_workers(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "workers": layer.registry.list_all(),
    }))
}

pub(super) async fn proxy_get_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let Some(id) = WorkerId::parse(&id_str) else {
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    match layer.registry.get_snapshot(id) {
        Some(worker) => Json(worker).into_response(),
        None => (StatusCode::NOT_FOUND, "worker not found").into_response(),
    }
}

pub(super) async fn proxy_drain_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let actor = audit_actor(req_id);
    let Some(id) = WorkerId::parse(&id_str) else {
        layer.audit.record(
            actor,
            "worker_drain",
            "worker",
            Some(id_str),
            "error",
            Some(serde_json::json!({"error": "invalid worker id"})),
        );
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    if !layer.registry.mark_drain(id) {
        layer.audit.record(
            actor,
            "worker_drain",
            "worker",
            Some(id.to_string()),
            "error",
            Some(serde_json::json!({"error": "worker not found"})),
        );
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    layer.audit.record(
        actor,
        "worker_drain",
        "worker",
        Some(id.to_string()),
        "ok",
        None,
    );
    tracing::info!(%id, "worker marked for drain via public API");
    StatusCode::OK.into_response()
}

pub(super) async fn proxy_drain_complete_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let actor = audit_actor(req_id);
    let Some(id) = WorkerId::parse(&id_str) else {
        layer.audit.record(
            actor,
            "worker_drain_complete",
            "worker",
            Some(id_str),
            "error",
            Some(serde_json::json!({"error": "invalid worker id"})),
        );
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    if layer.registry.get_snapshot(id).is_none() {
        layer.audit.record(
            actor,
            "worker_drain_complete",
            "worker",
            Some(id.to_string()),
            "error",
            Some(serde_json::json!({"error": "worker not found"})),
        );
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    layer.registry.evict(id);
    layer.audit.record(
        actor,
        "worker_drain_complete",
        "worker",
        Some(id.to_string()),
        "ok",
        None,
    );
    tracing::info!(%id, "worker drain complete via public API");
    StatusCode::NO_CONTENT.into_response()
}

pub(super) async fn proxy_delete_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let actor = audit_actor(req_id);
    let Some(id) = WorkerId::parse(&id_str) else {
        layer.audit.record(
            actor,
            "worker_delete",
            "worker",
            Some(id_str),
            "error",
            Some(serde_json::json!({"error": "invalid worker id"})),
        );
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    // mark_drain returns false when the worker does not exist.
    if !layer.registry.mark_drain(id) {
        layer.audit.record(
            actor,
            "worker_delete",
            "worker",
            Some(id.to_string()),
            "error",
            Some(serde_json::json!({"error": "worker not found"})),
        );
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    layer.registry.evict(id);
    layer.audit.record(
        actor,
        "worker_delete",
        "worker",
        Some(id.to_string()),
        "ok",
        None,
    );
    tracing::info!(%id, "worker force-removed via public API");
    StatusCode::NO_CONTENT.into_response()
}

pub(super) async fn proxy_dashboard() -> impl IntoResponse {
    Html(include_str!("../dashboard.html"))
}

pub(super) async fn proxy_get_license(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(layer.license.to_json())
}

pub(super) async fn proxy_set_license(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let Some(key) = body.get("key").and_then(|v| v.as_str()) else {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "missing field: key"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "missing field: key"})),
        )
            .into_response();
    };
    let key = key.trim().to_string();
    if key.is_empty() {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "key must not be empty"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "key must not be empty"})),
        )
            .into_response();
    }
    match layer.license.set_key(key) {
        Ok(()) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "ok",
                None,
            );
            Json(layer.license.to_json()).into_response()
        }
        Err(e) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "error",
                Some(serde_json::json!({"error": e.to_string()})),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::{derive_cache_affinity_key, fairness_client_key};

    #[test]
    fn fairness_client_key_hashes_authorization_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-test-secret"),
        );

        let key = fairness_client_key(&headers, Some("10.0.0.8:443".parse().unwrap()));
        assert!(key.starts_with("auth:"));
        assert!(!key.contains("sk-test-secret"));
    }

    #[test]
    fn fairness_client_key_uses_peer_addr_not_forwarded_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", HeaderValue::from_static("203.0.113.1"));
        headers.insert("x-real-ip", HeaderValue::from_static("203.0.113.2"));

        let key = fairness_client_key(&headers, Some("10.0.0.9:1234".parse().unwrap()));
        assert_eq!(key, "ip:10.0.0.9");
    }

    #[test]
    fn cache_affinity_is_keyed_and_tenant_scoped() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ax-cache-affinity",
            HeaderValue::from_static("conversation-42"),
        );

        let a = derive_cache_affinity_key(&headers, "tenant-a", Some(&"a".repeat(32)))
            .unwrap()
            .unwrap();
        let b = derive_cache_affinity_key(&headers, "tenant-b", Some(&"a".repeat(32)))
            .unwrap()
            .unwrap();
        let different_secret =
            derive_cache_affinity_key(&headers, "tenant-a", Some(&"b".repeat(32)))
                .unwrap()
                .unwrap();

        assert_ne!(a, b);
        assert_ne!(a, different_secret);
    }

    #[test]
    fn cache_affinity_requires_an_operator_secret() {
        let mut headers = HeaderMap::new();
        headers.insert("x-ax-cache-affinity", HeaderValue::from_static("session"));

        let error = derive_cache_affinity_key(&headers, "tenant-a", None).unwrap_err();
        assert!(error.contains("AXS_CACHE_AFFINITY_SECRET"));
    }
}
