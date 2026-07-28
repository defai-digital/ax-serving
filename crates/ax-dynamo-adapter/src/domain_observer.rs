//! Authoritative readiness and aggregate observation for one Dynamo domain.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use anyhow::{Context, Result};
use ax_serving_adapter_core::openai_runtime;
use ax_serving_protocol::{
    CapacityObservation, DomainObservation, RuntimeModelDescriptor, RuntimeObservation,
    RuntimeState, RuntimeStatus,
};
use tokio::sync::RwLock;

use crate::config::DynamoAdapterConfig;
use crate::inventory::protocol_models;
use crate::manifest::ValidatedManifest;

#[derive(Clone, Debug)]
pub struct ObservationSnapshot {
    pub runtime: RuntimeObservation,
    pub domain: DomainObservation,
}

#[derive(Clone)]
pub struct DomainState {
    pub inflight: Arc<AtomicUsize>,
    pub draining: Arc<AtomicBool>,
    pub ready: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
    inventory_generation: Arc<AtomicU64>,
    snapshot: Arc<RwLock<ObservationSnapshot>>,
}

impl DomainState {
    pub fn new(manifest: &ValidatedManifest, max_inflight: usize) -> Self {
        let observed_at = time::OffsetDateTime::now_utc();
        let capacity = CapacityObservation {
            active_requests: Some(0),
            max_concurrent_requests: Some(max_inflight as u64),
            ..Default::default()
        };
        let runtime_status = RuntimeStatus {
            ready: false,
            state: RuntimeState::Starting,
            reason_code: Some("dynamo_starting".into()),
            message: None,
            probe_latency_ms: None,
        };
        let runtime = RuntimeObservation {
            observed_at,
            runtime: runtime_status.clone(),
            inventory_generation: 1,
            models: Vec::new(),
            capacity: Some(capacity.clone()),
        };
        let domain = DomainObservation {
            observed_at,
            generation: 1,
            ready: false,
            state: RuntimeState::Starting,
            reason_code: runtime_status.reason_code.clone(),
            frontend_instances_ready: Some(0),
            aggregate_capacity: Some(capacity),
            manifest_digest: Some(manifest.digest.clone()),
            models: Vec::new(),
        };
        Self {
            inflight: Arc::new(AtomicUsize::new(0)),
            draining: Arc::new(AtomicBool::new(false)),
            ready: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicU64::new(1)),
            inventory_generation: Arc::new(AtomicU64::new(1)),
            snapshot: Arc::new(RwLock::new(ObservationSnapshot { runtime, domain })),
        }
    }

    pub async fn snapshot(&self) -> ObservationSnapshot {
        self.snapshot.read().await.clone()
    }

    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::Release);
        self.ready.store(false, Ordering::Release);
    }

    pub async fn observe(
        &self,
        client: &reqwest::Client,
        config: &DynamoAdapterConfig,
        manifest: &ValidatedManifest,
    ) -> Result<ObservationSnapshot> {
        let started = std::time::Instant::now();
        let previous = self.snapshot().await;
        let observation_result = openai_runtime::get_model_info(client, &config.frontend_url)
            .await
            .and_then(|models| {
                if models.is_empty() {
                    anyhow::bail!("Dynamo frontend reported no models");
                }
                protocol_models(&models, manifest)
            });
        let observed_at = time::OffsetDateTime::now_utc();
        let probe_latency_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let draining = self.draining.load(Ordering::Acquire);

        let (models, runtime_status) = match observation_result {
            Ok(models) if !draining => (
                models,
                RuntimeStatus {
                    ready: true,
                    state: RuntimeState::Ready,
                    reason_code: None,
                    message: None,
                    probe_latency_ms: Some(probe_latency_ms),
                },
            ),
            Ok(models) => (
                models,
                RuntimeStatus {
                    ready: false,
                    state: RuntimeState::Draining,
                    reason_code: Some("dynamo_adapter_draining".into()),
                    message: None,
                    probe_latency_ms: Some(probe_latency_ms),
                },
            ),
            Err(error) => {
                tracing::warn!(%error, "Dynamo frontend observation failed");
                (
                    previous.runtime.models.clone(),
                    RuntimeStatus {
                        ready: false,
                        state: if draining {
                            RuntimeState::Draining
                        } else {
                            RuntimeState::Unavailable
                        },
                        reason_code: Some(
                            if draining {
                                "dynamo_adapter_draining"
                            } else {
                                "dynamo_frontend_unavailable"
                            }
                            .into(),
                        ),
                        message: None,
                        probe_latency_ms: Some(probe_latency_ms),
                    },
                )
            }
        };

        let inventory_generation = if previous.runtime.models == models {
            self.inventory_generation.load(Ordering::Acquire)
        } else {
            self.inventory_generation.fetch_add(1, Ordering::AcqRel) + 1
        };
        let capacity = self.capacity(config.max_inflight);
        let runtime = RuntimeObservation {
            observed_at,
            runtime: runtime_status.clone(),
            inventory_generation,
            models: models.clone(),
            capacity: Some(capacity.clone()),
        };
        let domain = DomainObservation {
            observed_at,
            generation,
            ready: runtime_status.ready,
            state: runtime_status.state,
            reason_code: runtime_status.reason_code.clone(),
            frontend_instances_ready: Some(u32::from(runtime_status.ready)),
            aggregate_capacity: Some(capacity),
            manifest_digest: Some(manifest.digest.clone()),
            models,
        };
        runtime
            .validate()
            .context("Dynamo runtime observation violates protocol contract")?;
        domain
            .validate()
            .context("Dynamo domain observation violates protocol contract")?;

        self.ready.store(runtime_status.ready, Ordering::Release);
        let snapshot = ObservationSnapshot { runtime, domain };
        *self.snapshot.write().await = snapshot.clone();
        Ok(snapshot)
    }

    fn capacity(&self, max_inflight: usize) -> CapacityObservation {
        CapacityObservation {
            active_requests: Some(
                self.inflight.load(Ordering::Acquire).min(u64::MAX as usize) as u64
            ),
            max_concurrent_requests: Some(max_inflight as u64),
            // No Dynamo-internal or per-worker metrics are inferred here.
            ..Default::default()
        }
    }
}

pub fn model_ids(models: &[RuntimeModelDescriptor]) -> Vec<String> {
    models
        .iter()
        .map(|model| model.runtime_model_id.to_string())
        .collect()
}
