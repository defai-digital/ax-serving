//! Protocol-v1 session map: register, heartbeat, lease, restore, and export.
//!
//! Bridges protocol workers onto the legacy [`WorkerRegistry`] entry store
//! while retaining stable worker identity, leases, and shared-state restore.

use std::collections::BTreeSet;
use std::time::Instant;

use uuid::Uuid;

use ax_serving_protocol::{
    HeartbeatRequest as ProtocolHeartbeatRequest, HeartbeatResponse as ProtocolHeartbeatResponse,
    LeaseToken, NegotiatedProtocol, ProtocolDescriptor,
    RegisterWorkerRequest as ProtocolRegisterRequest,
    RegisterWorkerResponse as ProtocolRegisterResponse, RegistrationId,
    WorkerId as ProtocolWorkerId,
};

use super::super::fleet_state::{unix_time_millis, SharedWorkerRecord};
use super::super::worker_endpoint::WorkerEndpoint;
use super::normalize::{
    constant_time_digest_eq, lease_token_digest, legacy_heartbeat_from_observation,
    legacy_heartbeat_from_protocol, protocol_model_inventory, protocol_supported_operations,
};
use super::types::{
    ProtocolRegistryError, ProtocolSession, RegisterCapabilities, RegisterRequest,
    WorkerCapabilities, WorkerId,
};
use super::WorkerRegistry;
use super::MAX_WORKER_INFLIGHT;

impl WorkerRegistry {
    /// Register a protocol-v1 runtime agent while retaining the legacy
    /// registry as the endpoint-picker storage during migration.
    pub fn register_protocol(
        &self,
        request: ProtocolRegisterRequest,
        addr: WorkerEndpoint,
        negotiated: NegotiatedProtocol,
        heartbeat_interval_ms: u64,
        lease_ttl_ms: u64,
    ) -> Result<ProtocolRegisterResponse, ProtocolRegistryError> {
        request
            .observation
            .validate()
            .map_err(|error| ProtocolRegistryError::InvalidObservation(error.to_string()))?;

        let registration = request.clone();
        let stable_worker_id = request.worker.id.clone();
        let existing_internal_id = self
            .protocol_sessions
            .get(&stable_worker_id)
            .map(|session| session.internal_id);
        let model_inventory = protocol_model_inventory(&request.observation.models);
        let models = model_inventory
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>();
        let operations = protocol_supported_operations(&request.observation.models);
        let max_context = request
            .observation
            .models
            .iter()
            .filter_map(|model| model.max_context_tokens)
            .max()
            .map(|value| value.min(u64::from(u32::MAX)) as u32);
        let capabilities = WorkerCapabilities {
            llm: operations.iter().any(|operation| operation == "llm"),
            embedding: operations.iter().any(|operation| operation == "embedding"),
            vision: operations.iter().any(|operation| operation == "vision"),
            models,
            max_context,
        };

        let legacy_request = RegisterRequest {
            worker_id: existing_internal_id.map(|id| id.to_string()),
            addr: addr.to_string(),
            capabilities: RegisterCapabilities::Structured(capabilities),
            model_inventory,
            backend: request.runtime.kind.clone(),
            runtime: Some(request.runtime.kind.clone()),
            runtime_mode: Some("adapter".into()),
            runtime_version: Some(request.runtime.version.clone()),
            hardware_class: request.hardware.hardware_class.clone(),
            runtime_endpoint: None,
            supported_operations: operations,
            max_inflight: request
                .observation
                .capacity
                .as_ref()
                .and_then(|capacity| capacity.max_concurrent_requests)
                .unwrap_or(1)
                .min(MAX_WORKER_INFLIGHT as u64) as usize,
            friendly_name: request.worker.labels.get("friendly_name").cloned(),
            chip_model: request.worker.labels.get("chip_model").cloned(),
            worker_pool: Some(request.worker.pool_id.to_string()),
            node_class: request.worker.labels.get("node_class").cloned(),
        };
        let registered = self.register(legacy_request, heartbeat_interval_ms);
        let internal_id = WorkerId::parse(&registered.worker_id)
            .ok_or(ProtocolRegistryError::InternalRegistration)?;

        let heartbeat = legacy_heartbeat_from_observation(
            &request.observation,
            negotiated.version,
            &request.agent.version,
        );
        if !self.heartbeat(internal_id, heartbeat) {
            return Err(ProtocolRegistryError::InternalRegistration);
        }

        let registration_id = RegistrationId::new();
        let raw_token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
        let lease_token = LeaseToken::new(raw_token.clone())
            .map_err(|_| ProtocolRegistryError::InternalRegistration)?;
        let descriptor: ProtocolDescriptor = negotiated.clone().into();
        self.protocol_sessions.insert(
            stable_worker_id.clone(),
            ProtocolSession {
                internal_id,
                instance_id: request.worker.instance_id,
                registration_id,
                lease_token_digest: lease_token_digest(&raw_token),
                negotiated: descriptor.clone(),
                agent: request.agent.clone(),
                last_sequence: 0,
                inventory_generation: request.observation.inventory_generation,
                heartbeat_interval_ms,
                lease_ttl_ms,
                registration,
            },
        );
        if let Some(mut entry) = self.inner.get_mut(&internal_id) {
            entry.protocol_worker_id = Some(stable_worker_id.to_string());
            entry.worker_instance_id = Some(request.worker.instance_id.to_string());
            entry.registration_id = Some(registration_id.to_string());
            entry.trust_domain = Some(request.worker.trust_domain.to_string());
            entry.agent_name = Some(request.agent.name);
        }

        Ok(ProtocolRegisterResponse {
            registration_id,
            lease_token,
            protocol: descriptor,
            heartbeat_interval_ms,
            lease_ttl_ms,
            inventory_resync: false,
        })
    }

    pub fn heartbeat_protocol(
        &self,
        worker_id: &ProtocolWorkerId,
        lease_token: &str,
        request: ProtocolHeartbeatRequest,
    ) -> Result<ProtocolHeartbeatResponse, ProtocolRegistryError> {
        if !request.runtime.is_consistent() {
            return Err(ProtocolRegistryError::InvalidObservation(
                "runtime ready flag and state are inconsistent".into(),
            ));
        }
        if let Some(capacity) = &request.capacity {
            capacity
                .validate()
                .map_err(|error| ProtocolRegistryError::InvalidObservation(error.to_string()))?;
        }

        let mut session = self
            .protocol_sessions
            .get_mut(worker_id)
            .ok_or(ProtocolRegistryError::NotRegistered)?;
        if session.instance_id != request.instance_id {
            return Err(ProtocolRegistryError::InstanceMismatch);
        }
        if session.registration_id != request.registration_id {
            return Err(ProtocolRegistryError::RegistrationMismatch);
        }
        if !constant_time_digest_eq(
            &session.lease_token_digest,
            &lease_token_digest(lease_token),
        ) {
            return Err(ProtocolRegistryError::InvalidLeaseToken);
        }
        if request.sequence < session.last_sequence {
            return Err(ProtocolRegistryError::ReplayedHeartbeat {
                received: request.sequence,
                accepted: session.last_sequence,
            });
        }
        if request.sequence == session.last_sequence && session.last_sequence != 0 {
            return Ok(ProtocolHeartbeatResponse::default());
        }

        let inventory_resync = request.models.is_none()
            && request.inventory_generation != session.inventory_generation;
        let model_inventory = match &request.models {
            Some(models) => protocol_model_inventory(models),
            None => self
                .inner
                .get(&session.internal_id)
                .map(|entry| entry.model_inventory.clone())
                .unwrap_or_default(),
        };
        let heartbeat = legacy_heartbeat_from_protocol(
            &request,
            &model_inventory,
            session.negotiated.version,
            &session.agent.version,
        );
        if !self.heartbeat(session.internal_id, heartbeat) {
            return Err(ProtocolRegistryError::NotRegistered);
        }

        session.last_sequence = request.sequence;
        session.registration.observation.observed_at = request.observed_at;
        session.registration.observation.runtime = request.runtime.clone();
        session.registration.observation.capacity = request.capacity.clone();
        session.registration.observation.inventory_generation = request.inventory_generation;
        if request.models.is_some() {
            session.inventory_generation = request.inventory_generation;
            session.registration.observation.models = request.models.clone().unwrap_or_default();
        }

        Ok(ProtocolHeartbeatResponse {
            drain: if self
                .inner
                .get(&session.internal_id)
                .is_some_and(|entry| entry.drain)
            {
                ax_serving_protocol::DrainDirective::Begin
            } else {
                ax_serving_protocol::DrainDirective::None
            },
            inventory_resync,
            reregister: false,
            deployment_commands: Vec::new(),
        })
    }

    /// Resolve either the legacy gateway-assigned UUID or a protocol-v1 stable
    /// worker identifier to the current internal registry identity.
    pub fn resolve_worker_id(&self, raw: &str) -> Option<WorkerId> {
        WorkerId::parse(raw).or_else(|| {
            let stable = raw.parse::<ProtocolWorkerId>().ok()?;
            self.protocol_sessions
                .get(&stable)
                .map(|session| session.internal_id)
        })
    }

    pub fn validate_protocol_lease(
        &self,
        worker_id: &ProtocolWorkerId,
        lease_token: &str,
    ) -> Result<WorkerId, ProtocolRegistryError> {
        let session = self
            .protocol_sessions
            .get(worker_id)
            .ok_or(ProtocolRegistryError::NotRegistered)?;
        if !constant_time_digest_eq(
            &session.lease_token_digest,
            &lease_token_digest(lease_token),
        ) {
            return Err(ProtocolRegistryError::InvalidLeaseToken);
        }
        Ok(session.internal_id)
    }

    pub fn export_protocol_record(
        &self,
        worker_id: &ProtocolWorkerId,
    ) -> Option<SharedWorkerRecord> {
        let session = self.protocol_sessions.get(worker_id)?;
        let entry = self.inner.get(&session.internal_id)?;
        Some(SharedWorkerRecord {
            worker_id: worker_id.clone(),
            instance_id: session.instance_id,
            registration_id: session.registration_id,
            lease_token_digest: session.lease_token_digest,
            protocol: session.negotiated.clone(),
            agent: session.agent.clone(),
            registration: session.registration.clone(),
            addr: entry.addr.clone(),
            last_sequence: session.last_sequence,
            inventory_generation: session.inventory_generation,
            heartbeat_interval_ms: session.heartbeat_interval_ms,
            lease_ttl_ms: session.lease_ttl_ms,
            updated_at_unix_ms: unix_time_millis(),
            draining: entry.drain,
        })
    }

    /// Stable protocol worker ids currently mirrored by this gateway replica.
    pub fn protocol_worker_ids(&self) -> BTreeSet<ProtocolWorkerId> {
        self.protocol_sessions
            .iter()
            .map(|session| session.key().clone())
            .collect()
    }

    /// Return the shared lease identity associated with an internal routing id.
    pub fn protocol_identity_for_internal(
        &self,
        internal_id: WorkerId,
    ) -> Option<(ProtocolWorkerId, RegistrationId)> {
        self.protocol_sessions.iter().find_map(|session| {
            (session.internal_id == internal_id)
                .then(|| (session.key().clone(), session.registration_id))
        })
    }

    /// Drop the local mirror for a stable protocol worker id.
    pub fn evict_protocol(&self, worker_id: &ProtocolWorkerId) -> bool {
        let Some((_, session)) = self.protocol_sessions.remove(worker_id) else {
            return false;
        };
        // register_protocol/restore already reindexed via register/heartbeat;
        // unindex here so the secondary index does not retain a dead WorkerId.
        if let Some((_, entry)) = self.inner.remove(&session.internal_id) {
            self.unindex_worker(session.internal_id, &entry.capabilities.models);
        }
        true
    }

    /// Restore only when shared state is authoritative and newer than this
    /// replica's local mirror. This avoids resetting local inflight counters on
    /// every reconciliation pass.
    pub fn restore_protocol_record_if_newer(
        &self,
        record: SharedWorkerRecord,
    ) -> Result<bool, ProtocolRegistryError> {
        let should_restore = match self.protocol_sessions.get(&record.worker_id) {
            None => true,
            Some(session)
                if session.registration_id != record.registration_id
                    || session.instance_id != record.instance_id =>
            {
                true
            }
            Some(session) if record.last_sequence > session.last_sequence => true,
            Some(session) if record.last_sequence == session.last_sequence => self
                .inner
                .get(&session.internal_id)
                .is_none_or(|entry| entry.drain != record.draining),
            Some(_) => false,
        };
        if should_restore {
            self.restore_protocol_record(record)?;
        }
        Ok(should_restore)
    }

    /// Reconcile a protocol worker record created by another gateway replica.
    pub fn restore_protocol_record(
        &self,
        record: SharedWorkerRecord,
    ) -> Result<WorkerId, ProtocolRegistryError> {
        let shared_age = std::time::Duration::from_millis(
            unix_time_millis().saturating_sub(record.updated_at_unix_ms),
        );
        record
            .registration
            .observation
            .validate()
            .map_err(|error| ProtocolRegistryError::InvalidObservation(error.to_string()))?;
        if record.worker_id != record.registration.worker.id
            || record.instance_id != record.registration.worker.instance_id
        {
            return Err(ProtocolRegistryError::InstanceMismatch);
        }

        let existing_internal_id = self
            .protocol_sessions
            .get(&record.worker_id)
            .map(|session| session.internal_id);
        let model_inventory = protocol_model_inventory(&record.registration.observation.models);
        let operations = protocol_supported_operations(&record.registration.observation.models);
        let capabilities = WorkerCapabilities {
            llm: operations.iter().any(|operation| operation == "llm"),
            embedding: operations.iter().any(|operation| operation == "embedding"),
            vision: operations.iter().any(|operation| operation == "vision"),
            models: model_inventory
                .iter()
                .map(|model| model.id.clone())
                .collect(),
            max_context: record
                .registration
                .observation
                .models
                .iter()
                .filter_map(|model| model.max_context_tokens)
                .max()
                .map(|value| value.min(u64::from(u32::MAX)) as u32),
        };
        let request = &record.registration;
        let registered = self.register(
            RegisterRequest {
                worker_id: existing_internal_id.map(|id| id.to_string()),
                addr: record.addr.to_string(),
                capabilities: RegisterCapabilities::Structured(capabilities),
                model_inventory,
                backend: request.runtime.kind.clone(),
                runtime: Some(request.runtime.kind.clone()),
                runtime_mode: Some("adapter".into()),
                runtime_version: Some(request.runtime.version.clone()),
                hardware_class: request.hardware.hardware_class.clone(),
                runtime_endpoint: None,
                supported_operations: operations,
                max_inflight: request
                    .observation
                    .capacity
                    .as_ref()
                    .and_then(|capacity| capacity.max_concurrent_requests)
                    .unwrap_or(1)
                    .min(MAX_WORKER_INFLIGHT as u64) as usize,
                friendly_name: request.worker.labels.get("friendly_name").cloned(),
                chip_model: request.worker.labels.get("chip_model").cloned(),
                worker_pool: Some(request.worker.pool_id.to_string()),
                node_class: request.worker.labels.get("node_class").cloned(),
            },
            record.heartbeat_interval_ms,
        );
        let internal_id = WorkerId::parse(&registered.worker_id)
            .ok_or(ProtocolRegistryError::InternalRegistration)?;
        if !self.heartbeat(
            internal_id,
            legacy_heartbeat_from_observation(
                &request.observation,
                record.protocol.version,
                &record.agent.version,
            ),
        ) {
            return Err(ProtocolRegistryError::InternalRegistration);
        }

        self.protocol_sessions.insert(
            record.worker_id.clone(),
            ProtocolSession {
                internal_id,
                instance_id: record.instance_id,
                registration_id: record.registration_id,
                lease_token_digest: record.lease_token_digest,
                negotiated: record.protocol.clone(),
                agent: record.agent.clone(),
                last_sequence: record.last_sequence,
                inventory_generation: record.inventory_generation,
                heartbeat_interval_ms: record.heartbeat_interval_ms,
                lease_ttl_ms: record.lease_ttl_ms,
                registration: record.registration.clone(),
            },
        );
        if let Some(mut entry) = self.inner.get_mut(&internal_id) {
            entry.protocol_worker_id = Some(record.worker_id.to_string());
            entry.worker_instance_id = Some(record.instance_id.to_string());
            entry.registration_id = Some(record.registration_id.to_string());
            entry.trust_domain = Some(request.worker.trust_domain.to_string());
            entry.agent_name = Some(record.agent.name);
            entry.drain = record.draining;
            entry.last_heartbeat = Instant::now()
                .checked_sub(shared_age)
                .unwrap_or_else(Instant::now);
        }
        Ok(internal_id)
    }
}
