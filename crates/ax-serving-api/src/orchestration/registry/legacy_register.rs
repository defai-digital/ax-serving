//! Legacy worker registration, heartbeat, drain, and inflight helpers.
//!
//! Mutation entry points for the pre-protocol registry path. Protocol-v1
//! sessions live in [`super::protocol_session`]; health eviction lives in
//! [`super::health_tick`].

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use tracing::warn;

use super::super::worker_endpoint::WorkerEndpoint;
use super::index::reindex_into;
use super::MAX_WORKER_INFLIGHT;
use super::WorkerRegistry;
use super::normalize::{
    max_context_from_model_inventory, non_negative_finite, normalize_model_inventory,
    normalize_optional_string, normalize_runtime_mode, normalize_supported_operations,
    ratio_or_zero, refresh_capabilities_from_inventory_summary,
    refresh_capabilities_from_operation_summary, retain_model_inventory_for_ids,
    supported_operations_from_capabilities, supported_operations_from_model_inventory,
};
use super::types::{
    BackendKind, CapabilitySource, HeartbeatRequest, RegisterRequest, RegisterResponse,
    RuntimeKind, WorkerEntry, WorkerHealth, WorkerId,
};

impl WorkerRegistry {
    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Register (or re-register) a worker.  Returns the assigned `WorkerId`.
    pub fn register(&self, req: RegisterRequest, heartbeat_interval_ms: u64) -> RegisterResponse {
        let RegisterRequest {
            worker_id,
            addr: raw_addr,
            capabilities,
            model_inventory,
            backend,
            runtime,
            runtime_mode,
            runtime_version,
            hardware_class,
            runtime_endpoint,
            supported_operations,
            max_inflight,
            friendly_name,
            chip_model,
            worker_pool,
            node_class,
        } = req;
        let id = worker_id
            .as_deref()
            .and_then(WorkerId::parse)
            .unwrap_or_default();

        // Sentinel loopback endpoint so a malformed registration cannot be
        // selected for real dispatch but still appears in diagnostics.
        let addr = match WorkerEndpoint::parse(&raw_addr) {
            Ok(addr) => addr,
            Err(err) => {
                warn!(
                    raw_addr = %raw_addr,
                    err = %err,
                    "worker registered with unparseable address; it will never receive traffic"
                );
                WorkerEndpoint::parse("http://127.0.0.1:1").expect("static sentinel endpoint")
            }
        };
        let max_inflight = max_inflight.clamp(1, MAX_WORKER_INFLIGHT);
        let backend = BackendKind::parse(&backend);
        let runtime = runtime
            .as_deref()
            .map(RuntimeKind::parse)
            .filter(|runtime| *runtime != RuntimeKind::Unknown)
            .unwrap_or_else(|| RuntimeKind::from_backend(&backend));
        let runtime_mode = normalize_runtime_mode(runtime_mode);
        let runtime_version = normalize_optional_string(runtime_version);
        let hardware_class = normalize_optional_string(hardware_class);
        let runtime_endpoint = normalize_optional_string(runtime_endpoint);
        let friendly_name = normalize_optional_string(friendly_name);
        let chip_model = normalize_optional_string(chip_model);
        let worker_pool = normalize_optional_string(worker_pool);
        let node_class = normalize_optional_string(node_class);
        let (mut capabilities, capability_source) = capabilities.into_parts();
        let incoming_model_inventory = model_inventory;
        let incoming_model_inventory_empty = incoming_model_inventory.is_empty();
        let model_inventory =
            normalize_model_inventory(&capabilities.models, incoming_model_inventory);
        let inventory_supported_operations = if incoming_model_inventory_empty {
            Vec::new()
        } else {
            let operations = supported_operations_from_model_inventory(&model_inventory);
            refresh_capabilities_from_inventory_summary(
                &mut capabilities,
                &operations,
                max_context_from_model_inventory(&model_inventory),
                false,
            );
            operations
        };
        capabilities.models = model_inventory
            .iter()
            .map(|model| model.id.clone())
            .collect();
        let explicit_supported_operations_empty = supported_operations.is_empty();
        let supported_operations = if explicit_supported_operations_empty {
            if !inventory_supported_operations.is_empty() {
                inventory_supported_operations
            } else {
                match capability_source {
                    CapabilitySource::Legacy => Vec::new(),
                    CapabilitySource::Structured => {
                        supported_operations_from_capabilities(&capabilities)
                    }
                }
            }
        } else {
            normalize_supported_operations(supported_operations)
        };
        refresh_capabilities_from_operation_summary(&mut capabilities, &supported_operations);

        // Reindex under the entry write guard so concurrent same-id register /
        // heartbeat diffs cannot be applied out of order (permanent under-index).
        // Clone by_model Arc first so the free function does not re-borrow `self`.
        let by_model = self.by_model_handle();
        self.inner
            .entry(id)
            .and_modify(|existing| {
                let old_models = existing.capabilities.models.clone();
                let mut updated_capabilities = capabilities.clone();
                // Idempotent re-registration: update mutable fields, reset health.
                existing.addr = addr.clone();
                existing.model_inventory = if incoming_model_inventory_empty {
                    retain_model_inventory_for_ids(
                        &existing.model_inventory,
                        &updated_capabilities.models,
                    )
                } else {
                    model_inventory.clone()
                };
                let retained_inventory_supported_operations =
                    if incoming_model_inventory_empty && explicit_supported_operations_empty {
                        let operations =
                            supported_operations_from_model_inventory(&existing.model_inventory);
                        refresh_capabilities_from_inventory_summary(
                            &mut updated_capabilities,
                            &operations,
                            max_context_from_model_inventory(&existing.model_inventory),
                            false,
                        );
                        operations
                    } else {
                        Vec::new()
                    };
                existing.capabilities = updated_capabilities;
                existing.capability_source = capability_source;
                existing.backend = backend.clone();
                existing.runtime = runtime.clone();
                existing.runtime_mode = runtime_mode
                    .clone()
                    .or_else(|| existing.runtime_mode.clone());
                existing.max_inflight = max_inflight;
                existing.health = WorkerHealth::Healthy;
                existing.last_heartbeat = Instant::now();
                existing.runtime_ready = None;
                existing.runtime_state = None;
                existing.runtime_status_reason = None;
                existing.observed_at_unix_ms = None;
                existing.protocol_version = None;
                existing.agent_version = None;
                existing.drain = false;
                existing.supported_operations =
                    if retained_inventory_supported_operations.is_empty() {
                        supported_operations.clone()
                    } else {
                        retained_inventory_supported_operations
                    };
                existing.supported_operations_explicit = !explicit_supported_operations_empty;
                existing.runtime_version = runtime_version.clone();
                existing.hardware_class = hardware_class.clone();
                existing.runtime_endpoint = runtime_endpoint.clone();
                existing.protocol_worker_id = None;
                existing.worker_instance_id = None;
                existing.registration_id = None;
                existing.trust_domain = None;
                existing.agent_name = None;
                existing.friendly_name = friendly_name.clone();
                existing.chip_model = chip_model.clone();
                existing.worker_pool = worker_pool.clone();
                existing.node_class = node_class.clone();
                let new_models = existing.capabilities.models.clone();
                reindex_into(&by_model, id, &old_models, &new_models);
            })
            .or_insert_with(|| {
                let new_models = capabilities.models.clone();
                reindex_into(&by_model, id, &[], &new_models);
                WorkerEntry {
                    id,
                    addr,
                    capabilities,
                    model_inventory,
                    capability_source,
                    backend,
                    runtime,
                    runtime_mode,
                    runtime_version,
                    hardware_class,
                    runtime_endpoint,
                    protocol_worker_id: None,
                    worker_instance_id: None,
                    registration_id: None,
                    trust_domain: None,
                    agent_name: None,
                    supported_operations,
                    supported_operations_explicit: !explicit_supported_operations_empty,
                    max_inflight,
                    inflight: Arc::new(AtomicUsize::new(0)),
                    reported_inflight: 0,
                    health: WorkerHealth::Healthy,
                    last_heartbeat: Instant::now(),
                    runtime_ready: None,
                    runtime_state: None,
                    runtime_status_reason: None,
                    observed_at_unix_ms: None,
                    protocol_version: None,
                    agent_version: None,
                    drain: false,
                    thermal_state: String::new(),
                    rss_bytes: 0,
                    friendly_name,
                    chip_model,
                    worker_pool,
                    node_class,
                    active_sequences: 0,
                    decode_tok_per_sec: 0.0,
                    ttft_p95_ms: 0,
                    queue_depth: 0,
                    error_rate: 0.0,
                    kv_pages_used: 0,
                    kv_pages_total: 0,
                    kv_utilization: None,
                    prefix_reusable_tokens: 0,
                    active_batch_size: 0,
                    max_batch_size: 0,
                    batch_utilization: None,
                }
            });

        RegisterResponse {
            worker_id: id.to_string(),
            heartbeat_interval_ms,
        }
    }

    /// Record a heartbeat.  Returns `false` if the worker is not registered.
    pub fn heartbeat(&self, id: WorkerId, req: HeartbeatRequest) -> bool {
        // Reindex under the entry write guard so concurrent same-id heartbeats
        // cannot apply (old, new) diffs out of order.
        let by_model = self.by_model_handle();
        match self.inner.get_mut(&id) {
            Some(mut e) => {
                let old_models = e.capabilities.models.clone();
                let runtime_ready = req.runtime_ready;
                let runtime_state = normalize_optional_string(req.runtime_state);
                let runtime_status_reason = normalize_optional_string(req.runtime_status_reason);
                let observed_at_unix_ms = req.observed_at_unix_ms;
                let protocol_version = req.protocol_version;
                let agent_version = normalize_optional_string(req.agent_version);
                e.last_heartbeat = Instant::now();
                e.health = if runtime_ready == Some(false) {
                    WorkerHealth::Unhealthy { missed: 1 }
                } else {
                    WorkerHealth::Healthy
                };
                e.runtime_ready = runtime_ready;
                e.runtime_state = runtime_state;
                e.runtime_status_reason = runtime_status_reason;
                e.observed_at_unix_ms = observed_at_unix_ms;
                e.protocol_version = protocol_version;
                e.agent_version = agent_version;
                e.reported_inflight = req.inflight.min(MAX_WORKER_INFLIGHT);
                e.thermal_state = req.thermal_state;
                e.rss_bytes = req.rss_bytes;
                // Authoritative capability snapshot from worker heartbeat.
                // Empty model_ids means the worker currently has no models.
                let mut model_ids = req.model_ids;
                let heartbeat_has_inventory = !req.model_inventory.is_empty();
                if !req.model_inventory.is_empty() {
                    model_ids.extend(req.model_inventory.iter().map(|model| model.id.clone()));
                }
                e.model_inventory = if req.model_inventory.is_empty() {
                    retain_model_inventory_for_ids(&e.model_inventory, &model_ids)
                } else {
                    normalize_model_inventory(&model_ids, req.model_inventory)
                };
                e.capabilities.models = e
                    .model_inventory
                    .iter()
                    .map(|model| model.id.clone())
                    .collect();
                let retained_inventory_has_operations = e
                    .model_inventory
                    .iter()
                    .any(|model| !model.supported_operations.is_empty());
                if heartbeat_has_inventory
                    || (retained_inventory_has_operations && !e.supported_operations_explicit)
                {
                    let operations = supported_operations_from_model_inventory(&e.model_inventory);
                    let max_context = max_context_from_model_inventory(&e.model_inventory);
                    refresh_capabilities_from_inventory_summary(
                        &mut e.capabilities,
                        &operations,
                        max_context,
                        true,
                    );
                    e.supported_operations = operations;
                    e.supported_operations_explicit = false;
                }
                // Token-cost dispatch telemetry — graceful defaults for legacy workers.
                // active_sequences == 0 and inflight != 0 means the worker doesn't send
                // the extended field; TokenCostPolicy falls back to inflight ratio.
                e.active_sequences = req.active_sequences.min(MAX_WORKER_INFLIGHT);
                e.decode_tok_per_sec = non_negative_finite(req.decode_tok_per_sec);
                e.ttft_p95_ms = req.ttft_p95_ms;
                e.queue_depth = req.queue_depth.min(MAX_WORKER_INFLIGHT);
                e.error_rate = ratio_or_zero(req.error_rate);
                e.kv_pages_used = req.kv_pages_used;
                e.kv_pages_total = req.kv_pages_total;
                e.kv_utilization = req.kv_utilization.map(ratio_or_zero);
                e.prefix_reusable_tokens = req.prefix_reusable_tokens;
                e.active_batch_size = req.active_batch_size;
                e.max_batch_size = req.max_batch_size;
                e.batch_utilization = req.batch_utilization.map(ratio_or_zero);
                let new_models = e.capabilities.models.clone();
                reindex_into(&by_model, id, &old_models, &new_models);
                true
            }
            None => false,
        }
    }

    /// Start graceful drain.  Returns `false` if worker not found.
    pub fn mark_drain(&self, id: WorkerId) -> bool {
        match self.inner.get_mut(&id) {
            Some(mut e) => {
                e.drain = true;
                true
            }
            None => false,
        }
    }

    /// Mark a worker as unhealthy after a failed dispatch.
    ///
    /// No-op if the worker is already unhealthy, dead, or not found.
    /// The health ticker will re-evaluate on the next tick.
    pub fn mark_unhealthy(&self, id: WorkerId) {
        if let Some(mut entry) = self.inner.get_mut(&id)
            && matches!(entry.health, WorkerHealth::Healthy)
        {
            entry.health = WorkerHealth::Unhealthy { missed: 1 };
        }
    }

    /// Shared inflight counter for a specific worker.
    ///
    /// This is used only after dispatch policy selection so the hot-path
    /// candidate list does not clone the counter `Arc` for every worker.
    pub fn inflight_counter(&self, id: WorkerId) -> Option<Arc<AtomicUsize>> {
        self.inner
            .get(&id)
            .map(|entry| Arc::clone(&entry.value().inflight))
    }
}
