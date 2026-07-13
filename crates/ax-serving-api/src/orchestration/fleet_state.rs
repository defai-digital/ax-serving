//! Shared fleet-state storage for active gateway replicas.
//!
//! Only protocol-v1 worker lease and observation state is shared. Per-request
//! streams, queue waiters, and HTTP connection state remain local to the
//! gateway that accepted the client request.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ax_serving_protocol::{
    AgentDescriptor, AttemptId, DeploymentControlRecord, DeploymentId, DeploymentJobRecord, JobId,
    ProtocolDescriptor, RegisterWorkerRequest, RegistrationId, WorkerId, WorkerInstanceId,
};
use dashmap::DashMap;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};

use crate::config::OrchestratorConfig;

pub type StoreFuture<'a, T> = Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send + 'a>>;

/// Result of a lease-fenced shared-state mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FleetMutationResult {
    /// The mutation was committed.
    Applied,
    /// The worker lease no longer exists in shared state.
    Missing,
    /// A different registration or worker instance owns the stable worker id.
    Fenced,
    /// Shared state has already accepted a newer heartbeat sequence.
    StaleSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationResult {
    Reserved,
    Saturated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedWorkerRecord {
    pub worker_id: WorkerId,
    pub instance_id: WorkerInstanceId,
    pub registration_id: RegistrationId,
    pub lease_token_digest: [u8; 32],
    pub protocol: ProtocolDescriptor,
    pub agent: AgentDescriptor,
    pub registration: RegisterWorkerRequest,
    pub addr: SocketAddr,
    pub last_sequence: u64,
    pub inventory_generation: u64,
    pub heartbeat_interval_ms: u64,
    pub lease_ttl_ms: u64,
    pub updated_at_unix_ms: u64,
    pub draining: bool,
}

impl SharedWorkerRecord {
    pub fn is_fresh(&self, now_unix_ms: u64) -> bool {
        now_unix_ms.saturating_sub(self.updated_at_unix_ms) <= self.lease_ttl_ms
    }
}

pub trait FleetStateStore: Send + Sync {
    fn kind(&self) -> &'static str;
    /// Publish a new registration. A later registration for the same stable
    /// worker id supersedes the previous lease.
    fn put<'a>(&'a self, record: &'a SharedWorkerRecord) -> StoreFuture<'a, ()>;
    /// Refresh heartbeat or drain state only while the registration still owns
    /// the shared lease and its heartbeat sequence is monotonic.
    fn compare_and_put<'a>(
        &'a self,
        record: &'a SharedWorkerRecord,
    ) -> StoreFuture<'a, FleetMutationResult>;
    fn get<'a>(&'a self, worker_id: &'a WorkerId) -> StoreFuture<'a, Option<SharedWorkerRecord>>;
    fn list(&self) -> StoreFuture<'_, Vec<SharedWorkerRecord>>;
    fn remove<'a>(&'a self, worker_id: &'a WorkerId) -> StoreFuture<'a, ()>;
    /// Remove a lease only if the caller still owns its registration.
    fn remove_if_registration<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        registration_id: RegistrationId,
    ) -> StoreFuture<'a, FleetMutationResult>;
    /// Reserve one advertised worker slot across all gateway replicas. Calling
    /// this again with the same attempt id renews the reservation lease.
    fn try_reserve<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        attempt_id: AttemptId,
        max_concurrent: usize,
        ttl_ms: u64,
    ) -> StoreFuture<'a, ReservationResult>;
    fn release_reservation<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        attempt_id: AttemptId,
    ) -> StoreFuture<'a, ()>;
    /// Acquire or renew short-lived ownership of an active health probe. In HA
    /// mode this prevents every gateway replica from probing the same worker.
    fn try_acquire_probe_lease<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        owner: &'a str,
        ttl_ms: u64,
    ) -> StoreFuture<'a, bool>;
    fn put_deployment_if_generation<'a>(
        &'a self,
        record: &'a DeploymentControlRecord,
        expected_generation: Option<u64>,
    ) -> StoreFuture<'a, FleetMutationResult>;
    fn get_deployment<'a>(
        &'a self,
        deployment_id: &'a DeploymentId,
    ) -> StoreFuture<'a, Option<DeploymentControlRecord>>;
    fn list_deployments(&self) -> StoreFuture<'_, Vec<DeploymentControlRecord>>;
    fn put_deployment_job<'a>(
        &'a self,
        record: &'a DeploymentJobRecord,
        ttl_ms: u64,
    ) -> StoreFuture<'a, ()>;
    fn get_deployment_job<'a>(
        &'a self,
        job_id: JobId,
    ) -> StoreFuture<'a, Option<DeploymentJobRecord>>;
    fn list_deployment_jobs(&self) -> StoreFuture<'_, Vec<DeploymentJobRecord>>;
}

#[derive(Clone)]
struct MemoryValue {
    record: SharedWorkerRecord,
    expires_at: Instant,
}

#[derive(Clone)]
struct MemoryJobValue {
    record: DeploymentJobRecord,
    expires_at: Instant,
}

#[derive(Default)]
pub struct MemoryFleetStateStore {
    records: DashMap<WorkerId, MemoryValue>,
    reservations: DashMap<WorkerId, std::collections::BTreeMap<AttemptId, Instant>>,
    probe_leases: DashMap<WorkerId, (String, Instant)>,
    deployments: DashMap<DeploymentId, DeploymentControlRecord>,
    deployment_jobs: DashMap<JobId, MemoryJobValue>,
}

impl MemoryFleetStateStore {
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn prune_expired(&self) {
        let now = Instant::now();
        self.records.retain(|_, value| value.expires_at > now);
    }
}

impl FleetStateStore for MemoryFleetStateStore {
    fn kind(&self) -> &'static str {
        "memory"
    }

    fn put<'a>(&'a self, record: &'a SharedWorkerRecord) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            let ttl = Duration::from_millis(record.lease_ttl_ms.max(1));
            self.records.insert(
                record.worker_id.clone(),
                MemoryValue {
                    record: record.clone(),
                    expires_at: Instant::now() + ttl,
                },
            );
            Ok(())
        })
    }

    fn compare_and_put<'a>(
        &'a self,
        record: &'a SharedWorkerRecord,
    ) -> StoreFuture<'a, FleetMutationResult> {
        Box::pin(async move {
            self.prune_expired();
            let Some(mut current) = self.records.get_mut(&record.worker_id) else {
                return Ok(FleetMutationResult::Missing);
            };
            if current.record.registration_id != record.registration_id
                || current.record.instance_id != record.instance_id
            {
                return Ok(FleetMutationResult::Fenced);
            }
            if current.record.last_sequence > record.last_sequence {
                return Ok(FleetMutationResult::StaleSequence);
            }
            *current = MemoryValue {
                record: record.clone(),
                expires_at: Instant::now() + Duration::from_millis(record.lease_ttl_ms.max(1)),
            };
            Ok(FleetMutationResult::Applied)
        })
    }

    fn get<'a>(&'a self, worker_id: &'a WorkerId) -> StoreFuture<'a, Option<SharedWorkerRecord>> {
        Box::pin(async move {
            self.prune_expired();
            Ok(self
                .records
                .get(worker_id)
                .map(|value| value.record.clone()))
        })
    }

    fn list(&self) -> StoreFuture<'_, Vec<SharedWorkerRecord>> {
        Box::pin(async move {
            self.prune_expired();
            Ok(self
                .records
                .iter()
                .map(|value| value.record.clone())
                .collect())
        })
    }

    fn remove<'a>(&'a self, worker_id: &'a WorkerId) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            self.records.remove(worker_id);
            Ok(())
        })
    }

    fn remove_if_registration<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        registration_id: RegistrationId,
    ) -> StoreFuture<'a, FleetMutationResult> {
        Box::pin(async move {
            self.prune_expired();
            let Some(current) = self.records.get(worker_id) else {
                return Ok(FleetMutationResult::Missing);
            };
            if current.record.registration_id != registration_id {
                return Ok(FleetMutationResult::Fenced);
            }
            drop(current);
            self.records.remove(worker_id);
            Ok(FleetMutationResult::Applied)
        })
    }

    fn try_reserve<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        attempt_id: AttemptId,
        max_concurrent: usize,
        ttl_ms: u64,
    ) -> StoreFuture<'a, ReservationResult> {
        Box::pin(async move {
            let now = Instant::now();
            let mut reservations = self.reservations.entry(worker_id.clone()).or_default();
            reservations.retain(|_, expires_at| *expires_at > now);
            if let std::collections::btree_map::Entry::Occupied(mut entry) =
                reservations.entry(attempt_id)
            {
                entry.insert(now + Duration::from_millis(ttl_ms.max(1)));
                return Ok(ReservationResult::Reserved);
            }
            if reservations.len() >= max_concurrent.max(1) {
                return Ok(ReservationResult::Saturated);
            }
            reservations.insert(attempt_id, now + Duration::from_millis(ttl_ms.max(1)));
            Ok(ReservationResult::Reserved)
        })
    }

    fn release_reservation<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        attempt_id: AttemptId,
    ) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            let mut remove_bucket = false;
            if let Some(mut reservations) = self.reservations.get_mut(worker_id) {
                reservations.remove(&attempt_id);
                remove_bucket = reservations.is_empty();
            }
            if remove_bucket {
                self.reservations.remove(worker_id);
            }
            Ok(())
        })
    }

    fn try_acquire_probe_lease<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        owner: &'a str,
        ttl_ms: u64,
    ) -> StoreFuture<'a, bool> {
        Box::pin(async move {
            let now = Instant::now();
            let expires_at = now + Duration::from_millis(ttl_ms.max(1));
            match self.probe_leases.get_mut(worker_id) {
                Some(current) if current.1 > now && current.0 != owner => Ok(false),
                Some(mut current) => {
                    *current = (owner.to_string(), expires_at);
                    Ok(true)
                }
                None => {
                    self.probe_leases
                        .insert(worker_id.clone(), (owner.to_string(), expires_at));
                    Ok(true)
                }
            }
        })
    }

    fn put_deployment_if_generation<'a>(
        &'a self,
        record: &'a DeploymentControlRecord,
        expected_generation: Option<u64>,
    ) -> StoreFuture<'a, FleetMutationResult> {
        Box::pin(async move {
            match self.deployments.entry(record.deployment.id.clone()) {
                dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                    let Some(expected) = expected_generation else {
                        return Ok(FleetMutationResult::Fenced);
                    };
                    if entry.get().generation != expected {
                        return Ok(FleetMutationResult::Fenced);
                    }
                    entry.insert(record.clone());
                    Ok(FleetMutationResult::Applied)
                }
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    if expected_generation.is_some() {
                        return Ok(FleetMutationResult::Missing);
                    }
                    entry.insert(record.clone());
                    Ok(FleetMutationResult::Applied)
                }
            }
        })
    }

    fn get_deployment<'a>(
        &'a self,
        deployment_id: &'a DeploymentId,
    ) -> StoreFuture<'a, Option<DeploymentControlRecord>> {
        Box::pin(async move {
            Ok(self
                .deployments
                .get(deployment_id)
                .map(|entry| entry.clone()))
        })
    }

    fn list_deployments(&self) -> StoreFuture<'_, Vec<DeploymentControlRecord>> {
        Box::pin(async move { Ok(self.deployments.iter().map(|entry| entry.clone()).collect()) })
    }

    fn put_deployment_job<'a>(
        &'a self,
        record: &'a DeploymentJobRecord,
        ttl_ms: u64,
    ) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            self.deployment_jobs.insert(
                record.id,
                MemoryJobValue {
                    record: record.clone(),
                    expires_at: Instant::now() + Duration::from_millis(ttl_ms.max(1)),
                },
            );
            Ok(())
        })
    }

    fn get_deployment_job<'a>(
        &'a self,
        job_id: JobId,
    ) -> StoreFuture<'a, Option<DeploymentJobRecord>> {
        Box::pin(async move {
            let now = Instant::now();
            if self
                .deployment_jobs
                .get(&job_id)
                .is_some_and(|entry| entry.expires_at <= now)
            {
                self.deployment_jobs.remove(&job_id);
            }
            Ok(self
                .deployment_jobs
                .get(&job_id)
                .map(|entry| entry.record.clone()))
        })
    }

    fn list_deployment_jobs(&self) -> StoreFuture<'_, Vec<DeploymentJobRecord>> {
        Box::pin(async move {
            let now = Instant::now();
            self.deployment_jobs
                .retain(|_, value| value.expires_at > now);
            Ok(self
                .deployment_jobs
                .iter()
                .map(|entry| entry.record.clone())
                .collect())
        })
    }
}

pub struct RedisFleetStateStore {
    client: redis::Client,
    key_prefix: String,
}

impl RedisFleetStateStore {
    pub fn new(url: &str, key_prefix: &str) -> anyhow::Result<Self> {
        let key_prefix = key_prefix.trim().trim_end_matches(':');
        anyhow::ensure!(
            !key_prefix.is_empty(),
            "fleet-state key prefix must not be empty"
        );
        anyhow::ensure!(
            key_prefix.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
            }),
            "fleet-state key prefix contains invalid characters"
        );
        Ok(Self {
            client: redis::Client::open(url)?,
            key_prefix: key_prefix.to_string(),
        })
    }

    fn record_key(&self, worker_id: &WorkerId) -> String {
        format!("{}:worker:{}", self.key_prefix, worker_id)
    }

    fn index_key(&self) -> String {
        format!("{}:workers", self.key_prefix)
    }

    fn reservation_key(&self, worker_id: &WorkerId) -> String {
        format!("{}:reservations:{}", self.key_prefix, worker_id)
    }

    fn probe_lease_key(&self, worker_id: &WorkerId) -> String {
        format!("{}:probe-owner:{}", self.key_prefix, worker_id)
    }

    fn deployment_key(&self, deployment_id: &DeploymentId) -> String {
        format!("{}:deployment:{}", self.key_prefix, deployment_id)
    }

    fn deployment_index_key(&self) -> String {
        format!("{}:deployments", self.key_prefix)
    }

    fn deployment_job_key(&self, job_id: JobId) -> String {
        format!("{}:deployment-job:{}", self.key_prefix, job_id)
    }

    fn deployment_job_index_key(&self) -> String {
        format!("{}:deployment-jobs", self.key_prefix)
    }

    async fn connection(&self) -> anyhow::Result<redis::aio::MultiplexedConnection> {
        Ok(self.client.get_multiplexed_async_connection().await?)
    }
}

impl FleetStateStore for RedisFleetStateStore {
    fn kind(&self) -> &'static str {
        "redis"
    }

    fn put<'a>(&'a self, record: &'a SharedWorkerRecord) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let body = serde_json::to_string(record)?;
            let record_key = self.record_key(&record.worker_id);
            let index_key = self.index_key();
            redis::pipe()
                .atomic()
                .cmd("SET")
                .arg(&record_key)
                .arg(body)
                .arg("PX")
                .arg(record.lease_ttl_ms.max(1))
                .ignore()
                .cmd("SADD")
                .arg(index_key)
                .arg(record.worker_id.to_string())
                .ignore()
                .query_async::<()>(&mut connection)
                .await?;
            Ok(())
        })
    }

    fn compare_and_put<'a>(
        &'a self,
        record: &'a SharedWorkerRecord,
    ) -> StoreFuture<'a, FleetMutationResult> {
        Box::pin(async move {
            const COMPARE_AND_PUT: &str = r#"
                local current_json = redis.call('GET', KEYS[1])
                if not current_json then
                    return 0
                end
                local current = cjson.decode(current_json)
                local incoming = cjson.decode(ARGV[1])
                if current.registration_id ~= incoming.registration_id
                    or current.instance_id ~= incoming.instance_id then
                    return -1
                end
                if tonumber(current.last_sequence) > tonumber(incoming.last_sequence) then
                    return -2
                end
                redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
                redis.call('SADD', KEYS[2], ARGV[3])
                return 1
            "#;
            let mut connection = self.connection().await?;
            let body = serde_json::to_string(record)?;
            let result: i64 = redis::Script::new(COMPARE_AND_PUT)
                .key(self.record_key(&record.worker_id))
                .key(self.index_key())
                .arg(body)
                .arg(record.lease_ttl_ms.max(1))
                .arg(record.worker_id.to_string())
                .invoke_async(&mut connection)
                .await?;
            Ok(match result {
                1 => FleetMutationResult::Applied,
                0 => FleetMutationResult::Missing,
                -1 => FleetMutationResult::Fenced,
                -2 => FleetMutationResult::StaleSequence,
                other => anyhow::bail!("unexpected Redis fleet mutation result {other}"),
            })
        })
    }

    fn get<'a>(&'a self, worker_id: &'a WorkerId) -> StoreFuture<'a, Option<SharedWorkerRecord>> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let body: Option<String> = connection.get(self.record_key(worker_id)).await?;
            body.map(|body| serde_json::from_str(&body).map_err(anyhow::Error::from))
                .transpose()
        })
    }

    fn list(&self) -> StoreFuture<'_, Vec<SharedWorkerRecord>> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let worker_ids: Vec<String> = connection.smembers(self.index_key()).await?;
            let mut records = Vec::with_capacity(worker_ids.len());
            let mut stale = Vec::new();
            for raw_id in worker_ids {
                let Ok(worker_id) = WorkerId::new(raw_id.clone()) else {
                    stale.push(raw_id);
                    continue;
                };
                match connection
                    .get::<_, Option<String>>(self.record_key(&worker_id))
                    .await?
                {
                    Some(body) => records.push(serde_json::from_str(&body)?),
                    None => stale.push(raw_id),
                }
            }
            if !stale.is_empty() {
                let _: usize = connection.srem(self.index_key(), stale).await?;
            }
            Ok(records)
        })
    }

    fn remove<'a>(&'a self, worker_id: &'a WorkerId) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            redis::pipe()
                .atomic()
                .cmd("DEL")
                .arg(self.record_key(worker_id))
                .ignore()
                .cmd("SREM")
                .arg(self.index_key())
                .arg(worker_id.to_string())
                .ignore()
                .query_async::<()>(&mut connection)
                .await?;
            Ok(())
        })
    }

    fn remove_if_registration<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        registration_id: RegistrationId,
    ) -> StoreFuture<'a, FleetMutationResult> {
        Box::pin(async move {
            const REMOVE_IF_REGISTRATION: &str = r#"
                local current_json = redis.call('GET', KEYS[1])
                if not current_json then
                    return 0
                end
                local current = cjson.decode(current_json)
                if current.registration_id ~= ARGV[1] then
                    return -1
                end
                redis.call('DEL', KEYS[1])
                redis.call('SREM', KEYS[2], ARGV[2])
                return 1
            "#;
            let mut connection = self.connection().await?;
            let result: i64 = redis::Script::new(REMOVE_IF_REGISTRATION)
                .key(self.record_key(worker_id))
                .key(self.index_key())
                .arg(registration_id.to_string())
                .arg(worker_id.to_string())
                .invoke_async(&mut connection)
                .await?;
            Ok(match result {
                1 => FleetMutationResult::Applied,
                0 => FleetMutationResult::Missing,
                -1 => FleetMutationResult::Fenced,
                other => anyhow::bail!("unexpected Redis fleet removal result {other}"),
            })
        })
    }

    fn try_reserve<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        attempt_id: AttemptId,
        max_concurrent: usize,
        ttl_ms: u64,
    ) -> StoreFuture<'a, ReservationResult> {
        Box::pin(async move {
            const TRY_RESERVE: &str = r#"
                local now = redis.call('TIME')
                local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
                redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms)
                local existing = redis.call('ZSCORE', KEYS[1], ARGV[1])
                local expires_at = now_ms + tonumber(ARGV[3])
                if existing then
                    redis.call('ZADD', KEYS[1], expires_at, ARGV[1])
                    redis.call('PEXPIRE', KEYS[1], ARGV[3] * 2)
                    return 1
                end
                if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then
                    return 0
                end
                redis.call('ZADD', KEYS[1], expires_at, ARGV[1])
                redis.call('PEXPIRE', KEYS[1], ARGV[3] * 2)
                return 1
            "#;
            let mut connection = self.connection().await?;
            let result: i64 = redis::Script::new(TRY_RESERVE)
                .key(self.reservation_key(worker_id))
                .arg(attempt_id.to_string())
                .arg(max_concurrent.max(1))
                .arg(ttl_ms.max(1))
                .invoke_async(&mut connection)
                .await?;
            Ok(if result == 1 {
                ReservationResult::Reserved
            } else {
                ReservationResult::Saturated
            })
        })
    }

    fn release_reservation<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        attempt_id: AttemptId,
    ) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let _: usize = connection
                .zrem(self.reservation_key(worker_id), attempt_id.to_string())
                .await?;
            Ok(())
        })
    }

    fn try_acquire_probe_lease<'a>(
        &'a self,
        worker_id: &'a WorkerId,
        owner: &'a str,
        ttl_ms: u64,
    ) -> StoreFuture<'a, bool> {
        Box::pin(async move {
            const ACQUIRE_PROBE_LEASE: &str = r#"
                local current = redis.call('GET', KEYS[1])
                if current and current ~= ARGV[1] then
                    return 0
                end
                redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
                return 1
            "#;
            let mut connection = self.connection().await?;
            let result: i64 = redis::Script::new(ACQUIRE_PROBE_LEASE)
                .key(self.probe_lease_key(worker_id))
                .arg(owner)
                .arg(ttl_ms.max(1))
                .invoke_async(&mut connection)
                .await?;
            Ok(result == 1)
        })
    }

    fn put_deployment_if_generation<'a>(
        &'a self,
        record: &'a DeploymentControlRecord,
        expected_generation: Option<u64>,
    ) -> StoreFuture<'a, FleetMutationResult> {
        Box::pin(async move {
            const PUT_DEPLOYMENT: &str = r#"
                local current_json = redis.call('GET', KEYS[1])
                if ARGV[2] == '' then
                    if current_json then return -1 end
                else
                    if not current_json then return 0 end
                    local current = cjson.decode(current_json)
                    if tonumber(current.generation) ~= tonumber(ARGV[2]) then return -1 end
                end
                redis.call('SET', KEYS[1], ARGV[1])
                redis.call('SADD', KEYS[2], ARGV[3])
                return 1
            "#;
            let mut connection = self.connection().await?;
            let result: i64 = redis::Script::new(PUT_DEPLOYMENT)
                .key(self.deployment_key(&record.deployment.id))
                .key(self.deployment_index_key())
                .arg(serde_json::to_string(record)?)
                .arg(
                    expected_generation
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                )
                .arg(record.deployment.id.to_string())
                .invoke_async(&mut connection)
                .await?;
            Ok(match result {
                1 => FleetMutationResult::Applied,
                0 => FleetMutationResult::Missing,
                -1 => FleetMutationResult::Fenced,
                other => anyhow::bail!("unexpected Redis deployment mutation result {other}"),
            })
        })
    }

    fn get_deployment<'a>(
        &'a self,
        deployment_id: &'a DeploymentId,
    ) -> StoreFuture<'a, Option<DeploymentControlRecord>> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let body: Option<String> = connection.get(self.deployment_key(deployment_id)).await?;
            body.map(|body| serde_json::from_str(&body).map_err(anyhow::Error::from))
                .transpose()
        })
    }

    fn list_deployments(&self) -> StoreFuture<'_, Vec<DeploymentControlRecord>> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let ids: Vec<String> = connection.smembers(self.deployment_index_key()).await?;
            let mut records = Vec::with_capacity(ids.len());
            let mut stale = Vec::new();
            for raw_id in ids {
                let Ok(id) = DeploymentId::new(raw_id.clone()) else {
                    stale.push(raw_id);
                    continue;
                };
                match connection
                    .get::<_, Option<String>>(self.deployment_key(&id))
                    .await?
                {
                    Some(body) => records.push(serde_json::from_str(&body)?),
                    None => stale.push(raw_id),
                }
            }
            if !stale.is_empty() {
                let _: usize = connection.srem(self.deployment_index_key(), stale).await?;
            }
            Ok(records)
        })
    }

    fn put_deployment_job<'a>(
        &'a self,
        record: &'a DeploymentJobRecord,
        ttl_ms: u64,
    ) -> StoreFuture<'a, ()> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            redis::pipe()
                .atomic()
                .cmd("SET")
                .arg(self.deployment_job_key(record.id))
                .arg(serde_json::to_string(record)?)
                .arg("PX")
                .arg(ttl_ms.max(1))
                .ignore()
                .cmd("SADD")
                .arg(self.deployment_job_index_key())
                .arg(record.id.to_string())
                .ignore()
                .query_async::<()>(&mut connection)
                .await?;
            Ok(())
        })
    }

    fn get_deployment_job<'a>(
        &'a self,
        job_id: JobId,
    ) -> StoreFuture<'a, Option<DeploymentJobRecord>> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let body: Option<String> = connection.get(self.deployment_job_key(job_id)).await?;
            body.map(|body| serde_json::from_str(&body).map_err(anyhow::Error::from))
                .transpose()
        })
    }

    fn list_deployment_jobs(&self) -> StoreFuture<'_, Vec<DeploymentJobRecord>> {
        Box::pin(async move {
            let mut connection = self.connection().await?;
            let ids: Vec<String> = connection.smembers(self.deployment_job_index_key()).await?;
            let mut records = Vec::with_capacity(ids.len());
            let mut stale = Vec::new();
            for raw_id in ids {
                let Ok(id) = raw_id.parse::<JobId>() else {
                    stale.push(raw_id);
                    continue;
                };
                match connection
                    .get::<_, Option<String>>(self.deployment_job_key(id))
                    .await?
                {
                    Some(body) => records.push(serde_json::from_str(&body)?),
                    None => stale.push(raw_id),
                }
            }
            if !stale.is_empty() {
                let _: usize = connection
                    .srem(self.deployment_job_index_key(), stale)
                    .await?;
            }
            Ok(records)
        })
    }
}

pub fn store_from_config(config: &OrchestratorConfig) -> anyhow::Result<Arc<dyn FleetStateStore>> {
    match config.fleet_store.trim().to_ascii_lowercase().as_str() {
        "memory" => Ok(MemoryFleetStateStore::shared()),
        "redis" | "valkey" => {
            let url = config
                .fleet_redis_url
                .as_ref()
                .map(|value| value.expose())
                .ok_or_else(|| anyhow::anyhow!("fleet_store=redis requires AXS_REDIS_URL"))?;
            Ok(Arc::new(RedisFleetStateStore::new(
                url,
                &config.fleet_key_prefix,
            )?))
        }
        other => anyhow::bail!("unknown fleet store {other:?}; expected memory or redis"),
    }
}

pub fn unix_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ax_serving_protocol::{
        AgentDescriptor, HardwareDescriptor, PoolId, ProtocolDescriptor, RegisterWorkerRequest,
        RuntimeDescriptor, RuntimeObservation, RuntimeStatus, TrustDomainId, WorkerDescriptor,
    };

    use super::{FleetStateStore, MemoryFleetStateStore, SharedWorkerRecord, unix_time_millis};

    fn record() -> SharedWorkerRecord {
        let worker_id = ax_serving_protocol::WorkerId::new("worker-1").unwrap();
        let instance_id = ax_serving_protocol::WorkerInstanceId::new();
        let registration = RegisterWorkerRequest {
            protocol: ProtocolDescriptor::current(BTreeSet::new()),
            agent: AgentDescriptor {
                name: "agent".into(),
                version: "1".into(),
                build_sha: None,
            },
            worker: WorkerDescriptor {
                id: worker_id.clone(),
                instance_id,
                advertise_url: "http://127.0.0.1:18081".into(),
                pool_id: PoolId::new("default").unwrap(),
                trust_domain: TrustDomainId::new("local").unwrap(),
                labels: BTreeMap::new(),
            },
            runtime: RuntimeDescriptor {
                kind: "vllm".into(),
                version: "1".into(),
                api: "openai-http".into(),
            },
            hardware: HardwareDescriptor {
                platform: "linux".into(),
                accelerator: "cuda".into(),
                device_count: 1,
                memory_bytes: None,
                hardware_class: Some("cuda".into()),
            },
            observation: RuntimeObservation {
                observed_at: time::OffsetDateTime::now_utc(),
                runtime: RuntimeStatus::ready(),
                inventory_generation: 1,
                models: Vec::new(),
                capacity: None,
            },
        };
        SharedWorkerRecord {
            worker_id,
            instance_id,
            registration_id: ax_serving_protocol::RegistrationId::new(),
            lease_token_digest: [7; 32],
            protocol: registration.protocol.clone(),
            agent: registration.agent.clone(),
            registration,
            addr: "127.0.0.1:18081".parse().unwrap(),
            last_sequence: 1,
            inventory_generation: 1,
            heartbeat_interval_ms: 5_000,
            lease_ttl_ms: 15_000,
            updated_at_unix_ms: unix_time_millis(),
            draining: false,
        }
    }

    #[tokio::test]
    async fn memory_store_round_trips_and_removes_records() {
        let store = MemoryFleetStateStore::default();
        let record = record();
        store.put(&record).await.unwrap();
        assert_eq!(
            store
                .get(&record.worker_id)
                .await
                .unwrap()
                .unwrap()
                .registration_id,
            record.registration_id
        );
        assert_eq!(store.list().await.unwrap().len(), 1);
        store.remove(&record.worker_id).await.unwrap();
        assert!(store.get(&record.worker_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn memory_store_fences_registration_and_sequence_mutations() {
        let store = MemoryFleetStateStore::default();
        let mut record = record();
        assert_eq!(
            store.compare_and_put(&record).await.unwrap(),
            super::FleetMutationResult::Missing
        );

        store.put(&record).await.unwrap();
        record.last_sequence = 2;
        assert_eq!(
            store.compare_and_put(&record).await.unwrap(),
            super::FleetMutationResult::Applied
        );

        let mut stale = record.clone();
        stale.last_sequence = 1;
        assert_eq!(
            store.compare_and_put(&stale).await.unwrap(),
            super::FleetMutationResult::StaleSequence
        );

        let mut fenced = record.clone();
        fenced.registration_id = ax_serving_protocol::RegistrationId::new();
        assert_eq!(
            store.compare_and_put(&fenced).await.unwrap(),
            super::FleetMutationResult::Fenced
        );
        assert_eq!(
            store
                .remove_if_registration(&record.worker_id, fenced.registration_id)
                .await
                .unwrap(),
            super::FleetMutationResult::Fenced
        );
        assert_eq!(
            store
                .remove_if_registration(&record.worker_id, record.registration_id)
                .await
                .unwrap(),
            super::FleetMutationResult::Applied
        );
    }

    #[tokio::test]
    async fn memory_store_reservations_are_bounded_idempotent_and_releasable() {
        let store = MemoryFleetStateStore::default();
        let worker_id = ax_serving_protocol::WorkerId::new("worker-reservation").unwrap();
        let first = ax_serving_protocol::AttemptId::new();
        let second = ax_serving_protocol::AttemptId::new();

        assert_eq!(
            store
                .try_reserve(&worker_id, first, 1, 1_000)
                .await
                .unwrap(),
            super::ReservationResult::Reserved
        );
        assert_eq!(
            store
                .try_reserve(&worker_id, first, 1, 1_000)
                .await
                .unwrap(),
            super::ReservationResult::Reserved
        );
        assert_eq!(
            store
                .try_reserve(&worker_id, second, 1, 1_000)
                .await
                .unwrap(),
            super::ReservationResult::Saturated
        );

        store.release_reservation(&worker_id, first).await.unwrap();
        assert_eq!(
            store
                .try_reserve(&worker_id, second, 1, 1_000)
                .await
                .unwrap(),
            super::ReservationResult::Reserved
        );
    }

    #[tokio::test]
    async fn memory_probe_lease_has_one_owner_and_expires() {
        let store = MemoryFleetStateStore::default();
        let worker_id = ax_serving_protocol::WorkerId::new("worker-probe").unwrap();

        assert!(
            store
                .try_acquire_probe_lease(&worker_id, "gateway-a", 5)
                .await
                .unwrap()
        );
        assert!(
            !store
                .try_acquire_probe_lease(&worker_id, "gateway-b", 5)
                .await
                .unwrap()
        );
        assert!(
            store
                .try_acquire_probe_lease(&worker_id, "gateway-a", 5)
                .await
                .unwrap()
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(
            store
                .try_acquire_probe_lease(&worker_id, "gateway-b", 5)
                .await
                .unwrap()
        );
    }
}
