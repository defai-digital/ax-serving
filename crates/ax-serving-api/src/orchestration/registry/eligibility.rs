//! Registry eligibility and dispatch candidate queries.
//!
//! Full-scan over the worker map (no secondary index yet). Fail-closed
//! filters live here so a later index PR can plug in without re-touching
//! mutation paths.

use std::collections::BTreeSet;

use super::WorkerRegistry;
use super::normalize::{
    backend_filter_from_hint, effective_inflight, runtime_filter_from_hint, worker_status_of,
};
use super::types::{
    BackendKind, CapabilitySource, ModelInventoryEntry, RequestKind, RuntimeKind, WorkerEntry,
    WorkerHealth, WorkerId, WorkerModelEndpoint, WorkerStatus,
};

impl WorkerRegistry {
    // ── Queries ───────────────────────────────────────────────────────────────

    /// Returns workers eligible to receive a request for `model_id`:
    /// healthy, not draining, and has `model_id` in capabilities.
    pub fn eligible_workers(&self, model_id: &str) -> Vec<WorkerStatus> {
        self.eligible_workers_filtered(model_id, RequestKind::Llm, None, None)
    }

    /// Returns workers eligible to receive a request for `model_id` and request kind:
    /// healthy, not draining, advertises the model, and supports the request kind.
    pub fn eligible_workers_for(
        &self,
        model_id: &str,
        request_kind: RequestKind,
    ) -> Vec<WorkerStatus> {
        self.eligible_workers_filtered(model_id, request_kind, None, None)
    }

    pub fn eligible_workers_filtered(
        &self,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
    ) -> Vec<WorkerStatus> {
        self.dispatch_workers_filtered(
            model_id,
            request_kind,
            backend_hint,
            min_context,
            None,
            None,
        )
    }

    pub fn dispatch_workers_filtered(
        &self,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
        preferred_pool: Option<&str>,
        excluded_id: Option<WorkerId>,
    ) -> Vec<WorkerStatus> {
        self.dispatch_workers_filtered_with_pool_mode(
            model_id,
            request_kind,
            backend_hint,
            min_context,
            preferred_pool,
            false,
            excluded_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_workers_filtered_with_pool_mode(
        &self,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
        preferred_pool: Option<&str>,
        require_preferred_pool: bool,
        excluded_id: Option<WorkerId>,
    ) -> Vec<WorkerStatus> {
        let backend_filter = backend_filter_from_hint(backend_hint);
        let runtime_filter = runtime_filter_from_hint(backend_hint);
        let preferred_pool = preferred_pool
            .map(str::trim)
            .filter(|pool| !pool.is_empty());
        let Some(preferred_pool) = preferred_pool else {
            return self
                .inner
                .iter()
                .filter_map(|r| {
                    let e = r.value();
                    dispatch_filter_matches(
                        e,
                        model_id,
                        request_kind,
                        backend_filter.as_ref(),
                        runtime_filter.as_ref(),
                        min_context,
                        excluded_id,
                    )
                    .then(|| worker_status_of(e))
                })
                .collect();
        };
        let mut preferred_workers = Vec::new();
        let mut fallback_workers = Vec::new();

        for r in self.inner.iter() {
            let e = r.value();
            let in_preferred_pool = e.worker_pool.as_deref() == Some(preferred_pool);
            let matches_without_exclusion = dispatch_filter_matches(
                e,
                model_id,
                request_kind,
                backend_filter.as_ref(),
                runtime_filter.as_ref(),
                min_context,
                None,
            );

            if !matches_without_exclusion {
                continue;
            }
            if excluded_id == Some(e.id) {
                continue;
            }

            let worker = worker_status_of(e);
            if in_preferred_pool {
                preferred_workers.push(worker);
            } else {
                fallback_workers.push(worker);
            }
        }

        if require_preferred_pool || !preferred_workers.is_empty() {
            preferred_workers
        } else {
            fallback_workers
        }
    }

    /// Build strict model-scoped endpoint snapshots for explicit deployment routing.
    ///
    /// Unlike the legacy compatibility queries, unknown context, output, modality,
    /// or protocol-capability limits fail closed when the request declares them.
    #[allow(clippy::too_many_arguments)]
    pub fn eligible_model_endpoints(
        &self,
        runtime_model_id: &str,
        request_kind: RequestKind,
        runtime_hint: Option<&str>,
        minimum_context_tokens: Option<u64>,
        max_output_tokens: Option<u64>,
        required_modalities: &BTreeSet<String>,
        required_capabilities: &BTreeSet<ax_serving_protocol::ProtocolCapability>,
        excluded_id: Option<WorkerId>,
    ) -> Vec<WorkerModelEndpoint> {
        let backend_filter = backend_filter_from_hint(runtime_hint);
        let runtime_filter = runtime_filter_from_hint(runtime_hint);
        self.inner
            .iter()
            .filter_map(|item| {
                let entry = item.value();
                if !dispatch_filter_matches(
                    entry,
                    runtime_model_id,
                    request_kind,
                    backend_filter.as_ref(),
                    runtime_filter.as_ref(),
                    None,
                    excluded_id,
                ) || effective_inflight(entry) >= entry.max_inflight
                {
                    return None;
                }
                let model = entry
                    .model_inventory
                    .iter()
                    .find(|model| model.id == runtime_model_id)?;
                if minimum_context_tokens.is_some_and(|required| {
                    model
                        .max_context
                        .map(u64::from)
                        .is_none_or(|limit| limit < required)
                }) || max_output_tokens.is_some_and(|required| {
                    model.max_output_tokens.is_none_or(|limit| limit < required)
                }) || !required_modalities.iter().all(|required| {
                    model
                        .modalities
                        .iter()
                        .any(|observed| observed.eq_ignore_ascii_case(required))
                }) || !required_capabilities.iter().all(|required| {
                    model
                        .protocol_capabilities
                        .iter()
                        .any(|observed| observed == required.as_str())
                }) {
                    return None;
                }

                Some(WorkerModelEndpoint {
                    worker: worker_status_of(entry),
                    worker_pool: entry.worker_pool.clone(),
                    node_class: entry.node_class.clone(),
                    hardware_class: entry.hardware_class.clone(),
                    runtime_kind: entry.runtime.as_str().to_string(),
                    runtime_version: entry.runtime_version.clone(),
                    trust_domain: entry.trust_domain.clone(),
                    protocol_worker_id: entry.protocol_worker_id.clone(),
                    worker_instance_id: entry.worker_instance_id.clone(),
                    registration_id: entry.registration_id.clone(),
                    model: model.clone(),
                })
            })
            .collect()
    }

    /// Conservative compatibility guard for retries in legacy deployment mode.
    ///
    /// Compatibility mode has no operator-certified equivalence graph, so a
    /// retry may stay only inside the same runtime/pool cohort and must not
    /// cross any observed model-identity difference.
    pub fn legacy_retry_compatible(
        &self,
        source_id: WorkerId,
        target_id: WorkerId,
        model_id: &str,
    ) -> bool {
        let Some(source) = self.inner.get(&source_id) else {
            return false;
        };
        let Some(target) = self.inner.get(&target_id) else {
            return false;
        };
        if source.runtime != target.runtime
            || source.worker_pool != target.worker_pool
            || (source.trust_domain.is_some() || target.trust_domain.is_some())
                && source.trust_domain != target.trust_domain
        {
            return false;
        }
        let source_model = source
            .model_inventory
            .iter()
            .find(|model| model.id == model_id);
        let target_model = target
            .model_inventory
            .iter()
            .find(|model| model.id == model_id);
        match (source_model, target_model) {
            (Some(source), Some(target)) => legacy_model_identity_matches(source, target),
            (None, None) => true,
            _ => false,
        }
    }
}

fn dispatch_filter_matches(
    entry: &WorkerEntry,
    model_id: &str,
    request_kind: RequestKind,
    backend_filter: Option<&BackendKind>,
    runtime_filter: Option<&RuntimeKind>,
    min_context: Option<u32>,
    excluded_id: Option<WorkerId>,
) -> bool {
    excluded_id != Some(entry.id)
        && !entry.drain
        && matches!(entry.health, WorkerHealth::Healthy)
        && entry.capabilities.models.iter().any(|c| c == model_id)
        && supports_request_kind(entry, request_kind)
        && model_inventory_supports_request_kind(entry, model_id, request_kind)
        && backend_filter.is_none_or(|kind| &entry.backend == kind)
        && runtime_filter.is_none_or(|kind| &entry.runtime == kind)
        && model_context_supports_request(entry, model_id, min_context)
}

fn model_context_supports_request(
    entry: &WorkerEntry,
    model_id: &str,
    min_context: Option<u32>,
) -> bool {
    let Some(required) = min_context else {
        return true;
    };

    let worker_context_ok = entry
        .capabilities
        .max_context
        .is_none_or(|worker_max| worker_max >= required);
    if !worker_context_ok {
        return false;
    }

    entry
        .model_inventory
        .iter()
        .find(|model| model.id == model_id)
        .and_then(|model| model.max_context)
        .is_none_or(|model_max| model_max >= required)
}

fn model_inventory_supports_request_kind(
    entry: &WorkerEntry,
    model_id: &str,
    request_kind: RequestKind,
) -> bool {
    entry
        .model_inventory
        .iter()
        .find(|model| model.id == model_id)
        .is_none_or(|model| {
            model.supported_operations.is_empty()
                || model
                    .supported_operations
                    .iter()
                    .any(|operation| operation == request_kind.as_operation())
        })
}

fn legacy_model_identity_matches(
    source: &ModelInventoryEntry,
    target: &ModelInventoryEntry,
) -> bool {
    let fields_present = source.revision.is_some()
        || target.revision.is_some()
        || source.artifact_digest.is_some()
        || target.artifact_digest.is_some()
        || source.tokenizer_digest.is_some()
        || target.tokenizer_digest.is_some()
        || source.template_digest.is_some()
        || target.template_digest.is_some()
        || source.quantization.is_some()
        || target.quantization.is_some();
    !fields_present
        || (source.revision == target.revision
            && source.artifact_digest == target.artifact_digest
            && source.tokenizer_digest == target.tokenizer_digest
            && source.template_digest == target.template_digest
            && source.quantization == target.quantization)
}

fn supports_request_kind(entry: &WorkerEntry, request_kind: RequestKind) -> bool {
    if !entry.supported_operations.is_empty()
        && !entry
            .supported_operations
            .iter()
            .any(|operation| operation == request_kind.as_operation())
    {
        return false;
    }

    match entry.capability_source {
        // Compatibility path: legacy workers historically routed by model-id only.
        CapabilitySource::Legacy => true,
        CapabilitySource::Structured => match request_kind {
            RequestKind::Llm => entry.capabilities.llm,
            RequestKind::Embedding => entry.capabilities.embedding,
            RequestKind::Vision => entry.capabilities.vision,
        },
    }
}
