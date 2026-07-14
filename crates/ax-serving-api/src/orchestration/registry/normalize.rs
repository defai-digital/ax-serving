//! Pure registry helpers: normalizers, snapshots, and protocol converters.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use rustc_hash::FxHashSet;
use sha2::{Digest as _, Sha256};

use ax_serving_protocol::{
    HeartbeatRequest as ProtocolHeartbeatRequest, ProtocolVersion, RuntimeModelDescriptor,
};

use super::MAX_WORKER_INFLIGHT;
use super::types::{
    BackendKind, HeartbeatRequest, ModelInventoryEntry, RuntimeKind, WorkerCapabilities,
    WorkerEntry, WorkerSnapshot, WorkerStatus,
};

pub(super) fn backend_filter_from_hint(hint: Option<&str>) -> Option<BackendKind> {
    let raw = hint?.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return None;
    }
    match BackendKind::parse(raw) {
        BackendKind::Auto => None,
        kind => Some(kind),
    }
}

pub(super) fn runtime_filter_from_hint(hint: Option<&str>) -> Option<RuntimeKind> {
    let raw = hint?.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return None;
    }
    match RuntimeKind::parse(raw) {
        RuntimeKind::Unknown => None,
        kind => Some(kind),
    }
}

pub(super) fn protocol_model_inventory(
    models: &[RuntimeModelDescriptor],
) -> Vec<ModelInventoryEntry> {
    models
        .iter()
        .map(|model| ModelInventoryEntry {
            id: model.runtime_model_id.to_string(),
            max_context: model
                .max_context_tokens
                .map(|value| value.min(u64::from(u32::MAX)) as u32),
            quantization: model.identity.quantization.clone(),
            artifact_format: None,
            modalities: protocol_model_modalities(model),
            supported_operations: model
                .operations
                .iter()
                .filter_map(protocol_operation_to_legacy)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            protocol_capabilities: model.capabilities.iter().map(ToString::to_string).collect(),
            revision: model.identity.revision.clone(),
            artifact_digest: model
                .identity
                .artifact_digest
                .as_ref()
                .map(ToString::to_string),
            tokenizer_digest: model
                .identity
                .tokenizer_digest
                .as_ref()
                .map(ToString::to_string),
            template_digest: model
                .identity
                .template_digest
                .as_ref()
                .map(ToString::to_string),
            runtime_kind: Some(model.identity.runtime_kind.clone()),
            runtime_version: model.identity.runtime_version.clone(),
            max_output_tokens: model.max_output_tokens,
        })
        .collect()
}

pub(super) fn protocol_model_modalities(model: &RuntimeModelDescriptor) -> Vec<String> {
    let mut modalities = BTreeSet::from(["text".to_string()]);
    if model
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "inference.vision")
    {
        modalities.insert("image".into());
    }
    modalities.into_iter().collect()
}

pub(super) fn protocol_operation_to_legacy(
    operation: &ax_serving_protocol::Operation,
) -> Option<String> {
    match operation.as_str() {
        ax_serving_protocol::Operation::CHAT_COMPLETIONS
        | ax_serving_protocol::Operation::TEXT_COMPLETIONS => Some("llm".into()),
        ax_serving_protocol::Operation::EMBEDDINGS => Some("embedding".into()),
        _ => None,
    }
}

pub(super) fn protocol_supported_operations(models: &[RuntimeModelDescriptor]) -> Vec<String> {
    let mut operations = models
        .iter()
        .flat_map(|model| model.operations.iter())
        .filter_map(protocol_operation_to_legacy)
        .collect::<BTreeSet<_>>();
    if models.iter().any(|model| {
        model
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == "inference.vision")
    }) {
        operations.insert("vision".into());
    }
    operations.into_iter().collect()
}

pub(super) fn legacy_heartbeat_from_observation(
    observation: &ax_serving_protocol::RuntimeObservation,
    protocol_version: ProtocolVersion,
    agent_version: &str,
) -> HeartbeatRequest {
    let model_inventory = protocol_model_inventory(&observation.models);
    HeartbeatRequest {
        model_ids: model_inventory
            .iter()
            .map(|model| model.id.clone())
            .collect(),
        model_inventory,
        runtime_ready: Some(observation.runtime.ready),
        runtime_state: Some(format!("{:?}", observation.runtime.state).to_ascii_lowercase()),
        runtime_status_reason: observation.runtime.reason_code.clone(),
        observed_at_unix_ms: offset_datetime_millis(observation.observed_at),
        protocol_version: Some(protocol_version),
        agent_version: Some(agent_version.to_string()),
        ..legacy_capacity_heartbeat(observation.capacity.as_ref())
    }
}

pub(super) fn legacy_heartbeat_from_protocol(
    request: &ProtocolHeartbeatRequest,
    model_inventory: &[ModelInventoryEntry],
    protocol_version: ProtocolVersion,
    agent_version: &str,
) -> HeartbeatRequest {
    HeartbeatRequest {
        model_ids: model_inventory
            .iter()
            .map(|model| model.id.clone())
            .collect(),
        model_inventory: model_inventory.to_vec(),
        runtime_ready: Some(request.runtime.ready),
        runtime_state: Some(format!("{:?}", request.runtime.state).to_ascii_lowercase()),
        runtime_status_reason: request.runtime.reason_code.clone(),
        observed_at_unix_ms: offset_datetime_millis(request.observed_at),
        protocol_version: Some(protocol_version),
        agent_version: Some(agent_version.to_string()),
        ..legacy_capacity_heartbeat(request.capacity.as_ref())
    }
}

pub(super) fn legacy_capacity_heartbeat(
    capacity: Option<&ax_serving_protocol::CapacityObservation>,
) -> HeartbeatRequest {
    let Some(capacity) = capacity else {
        return HeartbeatRequest::default();
    };
    HeartbeatRequest {
        inflight: capacity
            .active_requests
            .unwrap_or(0)
            .min(MAX_WORKER_INFLIGHT as u64) as usize,
        active_sequences: capacity
            .active_requests
            .unwrap_or(0)
            .min(MAX_WORKER_INFLIGHT as u64) as usize,
        decode_tok_per_sec: capacity.generated_tokens_per_second.unwrap_or(0.0),
        ttft_p95_ms: capacity
            .ttft_ewma_ms
            .unwrap_or(0.0)
            .round()
            .clamp(0.0, u64::MAX as f64) as u64,
        queue_depth: capacity
            .waiting_requests
            .unwrap_or(0)
            .min(MAX_WORKER_INFLIGHT as u64) as usize,
        rss_bytes: capacity.process_rss_bytes.unwrap_or(0),
        error_rate: capacity.recent_error_rate.unwrap_or(0.0),
        kv_utilization: capacity.kv_cache_used_ratio,
        batch_utilization: match (capacity.batch_tokens_in_use, capacity.batch_token_capacity) {
            (Some(in_use), Some(total)) if total > 0 => Some(in_use as f64 / total as f64),
            _ => None,
        },
        ..Default::default()
    }
}

pub(super) fn offset_datetime_millis(value: time::OffsetDateTime) -> Option<u64> {
    let nanos = value.unix_timestamp_nanos();
    if nanos < 0 {
        return None;
    }
    Some((nanos / 1_000_000).min(i128::from(u64::MAX)) as u64)
}

pub(super) fn lease_token_digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

pub(super) fn constant_time_digest_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

pub(super) fn snapshot_of(e: &WorkerEntry) -> WorkerSnapshot {
    let inflight = effective_inflight(e);
    let kv_utilization = worker_kv_utilization(e);
    let batch_utilization = worker_batch_utilization(e);
    WorkerSnapshot {
        id: e.id,
        addr: e.addr.to_string(),
        capabilities: e.capabilities.models.clone(),
        model_inventory: e.model_inventory.clone(),
        capability_descriptor: e.capabilities.clone(),
        backend: e.backend.as_str().to_string(),
        runtime: e.runtime.as_str().to_string(),
        runtime_mode: e.runtime_mode.clone(),
        runtime_version: e.runtime_version.clone(),
        hardware_class: e.hardware_class.clone(),
        runtime_endpoint: e.runtime_endpoint.clone(),
        protocol_worker_id: e.protocol_worker_id.clone(),
        worker_instance_id: e.worker_instance_id.clone(),
        registration_id: e.registration_id.clone(),
        trust_domain: e.trust_domain.clone(),
        agent_name: e.agent_name.clone(),
        supported_operations: e.supported_operations.clone(),
        max_inflight: e.max_inflight,
        inflight,
        saturation: inflight as f64 / e.max_inflight.max(1) as f64,
        health: e.health.as_str().to_string(),
        runtime_ready: e.runtime_ready,
        runtime_state: e.runtime_state.clone(),
        runtime_status_reason: e.runtime_status_reason.clone(),
        observed_at_unix_ms: e.observed_at_unix_ms,
        protocol_version: e.protocol_version,
        agent_version: e.agent_version.clone(),
        drain: e.drain,
        last_heartbeat_age_ms: e.last_heartbeat.elapsed().as_millis() as u64,
        thermal_state: e.thermal_state.clone(),
        rss_bytes: e.rss_bytes,
        friendly_name: e.friendly_name.clone(),
        chip_model: e.chip_model.clone(),
        worker_pool: e.worker_pool.clone(),
        node_class: e.node_class.clone(),
        active_sequences: e.active_sequences,
        decode_tok_per_sec: e.decode_tok_per_sec,
        ttft_p95_ms: e.ttft_p95_ms,
        queue_depth: e.queue_depth,
        error_rate: e.error_rate,
        kv_pages_used: e.kv_pages_used,
        kv_pages_total: e.kv_pages_total,
        kv_utilization,
        prefix_reusable_tokens: e.prefix_reusable_tokens,
        active_batch_size: e.active_batch_size,
        max_batch_size: e.max_batch_size,
        batch_utilization,
    }
}

pub(super) fn supported_operations_from_capabilities(
    capabilities: &WorkerCapabilities,
) -> Vec<String> {
    let mut operations = Vec::new();
    if capabilities.llm {
        operations.push("llm".to_string());
    }
    if capabilities.embedding {
        operations.push("embedding".to_string());
    }
    if capabilities.vision {
        operations.push("vision".to_string());
    }
    operations
}

pub(super) fn supported_operations_from_model_inventory(
    inventory: &[ModelInventoryEntry],
) -> Vec<String> {
    let mut seen = FxHashSet::default();
    let mut operations = Vec::new();
    for operation in inventory
        .iter()
        .flat_map(|model| model.supported_operations.iter())
    {
        if seen.insert(operation.clone()) {
            operations.push(operation.clone());
        }
    }
    operations
}

pub(super) fn max_context_from_model_inventory(inventory: &[ModelInventoryEntry]) -> Option<u32> {
    inventory.iter().filter_map(|model| model.max_context).max()
}

pub(super) fn refresh_capabilities_from_inventory_summary(
    capabilities: &mut WorkerCapabilities,
    operations: &[String],
    max_context: Option<u32>,
    clear_missing_max_context: bool,
) {
    if !operations.is_empty() {
        capabilities.llm = operations.iter().any(|op| op == "llm");
        capabilities.embedding = operations.iter().any(|op| op == "embedding");
        capabilities.vision = operations.iter().any(|op| op == "vision");
    }
    if max_context.is_some() || clear_missing_max_context {
        capabilities.max_context = max_context;
    }
}

pub(super) fn refresh_capabilities_from_operation_summary(
    capabilities: &mut WorkerCapabilities,
    operations: &[String],
) {
    if !operations.is_empty() {
        capabilities.llm = operations.iter().any(|op| op == "llm");
        capabilities.embedding = operations.iter().any(|op| op == "embedding");
        capabilities.vision = operations.iter().any(|op| op == "vision");
    }
}

pub(super) fn normalize_supported_operations(operations: Vec<String>) -> Vec<String> {
    let mut seen = FxHashSet::default();
    operations
        .into_iter()
        .filter_map(|op| normalize_supported_operation(&op))
        .filter(|op| seen.insert(op.clone()))
        .collect()
}

pub(super) fn normalize_supported_operation(operation: &str) -> Option<String> {
    let normalized = operation.trim().to_ascii_lowercase().replace('-', "_");
    let canonical = match normalized.as_str() {
        "" => return None,
        "embedding" | "embeddings" => "embedding",
        "vision" | "image" | "multimodal" => "vision",
        "llm" | "text" | "chat" | "completion" | "completions" => "llm",
        _ => normalized.as_str(),
    };
    Some(canonical.to_string())
}

pub(super) fn normalize_model_inventory(
    model_ids: &[String],
    inventory: Vec<ModelInventoryEntry>,
) -> Vec<ModelInventoryEntry> {
    let mut by_id = std::collections::BTreeMap::<String, ModelInventoryEntry>::new();
    for mut item in inventory {
        item.id = item.id.trim().to_string();
        if item.id.is_empty() {
            continue;
        }
        item.modalities.sort();
        item.modalities.dedup();
        item.supported_operations = normalize_supported_operations(item.supported_operations);
        by_id.insert(item.id.clone(), item);
    }
    for id in model_ids {
        let id = id.trim();
        if !id.is_empty() {
            by_id
                .entry(id.to_string())
                .or_insert_with(|| ModelInventoryEntry {
                    id: id.to_string(),
                    ..Default::default()
                });
        }
    }
    by_id.into_values().collect()
}

pub(super) fn retain_model_inventory_for_ids(
    previous: &[ModelInventoryEntry],
    model_ids: &[String],
) -> Vec<ModelInventoryEntry> {
    let retained = previous
        .iter()
        .filter(|entry| model_ids.iter().any(|id| id.trim() == entry.id))
        .cloned()
        .collect();
    normalize_model_inventory(model_ids, retained)
}

pub(super) fn normalize_runtime_mode(mode: Option<String>) -> Option<String> {
    mode.and_then(|value| {
        let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    })
}

pub(super) fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

pub(super) fn ratio_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub(super) fn non_negative_finite(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

pub(super) fn worker_status_of(e: &WorkerEntry) -> WorkerStatus {
    let kv_utilization = worker_kv_utilization(e);
    let batch_headroom = worker_batch_utilization(e).map(|value| 1.0 - value);
    WorkerStatus {
        id: e.id,
        addr: e.addr.clone(),
        inflight: effective_inflight(e),
        max_inflight: e.max_inflight,
        active_sequences: e.active_sequences,
        ttft_p95_ms: e.ttft_p95_ms,
        kv_utilization,
        batch_headroom,
        queue_depth: e.protocol_version.map(|_| e.queue_depth),
        error_rate: e.protocol_version.map(|_| e.error_rate),
        decode_tok_per_sec: e
            .protocol_version
            .and((e.decode_tok_per_sec > 0.0).then_some(e.decode_tok_per_sec)),
        telemetry_age_ms: e
            .protocol_version
            .and(e.observed_at_unix_ms)
            .map(observation_age_ms),
    }
}

pub(super) fn observation_age_ms(observed_at_unix_ms: u64) -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    now.saturating_sub(observed_at_unix_ms)
}

pub(super) fn effective_inflight(e: &WorkerEntry) -> usize {
    e.inflight.load(Ordering::Relaxed).max(e.reported_inflight)
}

pub(super) fn worker_kv_utilization(e: &WorkerEntry) -> Option<f64> {
    if e.kv_pages_total > 0 {
        Some((e.kv_pages_used as f64 / e.kv_pages_total as f64).clamp(0.0, 1.0))
    } else {
        e.kv_utilization
    }
}

pub(super) fn worker_batch_utilization(e: &WorkerEntry) -> Option<f64> {
    if e.max_batch_size > 0 {
        Some((e.active_batch_size as f64 / e.max_batch_size as f64).clamp(0.0, 1.0))
    } else {
        e.batch_utilization
    }
}
