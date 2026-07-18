//! Direct dispatcher: HTTP reverse proxy from orchestrator to worker.
//!
//! After the dispatch policy selects a worker, this module:
//! 1. Increments `inflight` atomically on the worker's shared counter.
//! 2. Forwards the full request body to `http://{worker_addr}{path}`.
//! 3. Streams or buffers the response back to the client.
//! 4. Decrements `inflight` via RAII guard on completion or error.
//!
//! A second attempt is allowed only for a proven connection failure or a
//! trusted typed pre-admission rejection. Arbitrary runtime 5xx responses are
//! never retried because the runtime may already have admitted the request.
//!
//! Streaming responses (SSE, `text/event-stream`) are forwarded chunk-by-chunk
//! without buffering so time-to-first-token is not impacted.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, TryStreamExt as _};
use opentelemetry::propagation::Injector;
use reqwest::Client;
use tracing::{Instrument as _, debug, error, info_span, warn};
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use ax_serving_protocol::{
    CandidateDecision, DecisionReasonCode, DecisionRecordV1, PolicyId, PolicyMode, PolicyVersion,
};

use super::deployment::{DeploymentCatalog, RouteCandidate};
use super::error::ax_error_response;
use super::fleet_state::{FleetStateStore, ReservationResult};
use super::policy::{DispatchContext, DispatchPolicy};
use super::registry::{RequestKind, WorkerId, WorkerRegistry};
use super::request_profile::{RequestProfile, rewrite_runtime_model};

struct TraceHeaderInjector(HeaderMap);

impl Injector for TraceHeaderInjector {
    fn set(&mut self, key: &str, value: String) {
        if !matches!(key, "traceparent" | "tracestate" | "baggage") || value.len() > 1024 {
            return;
        }
        let Ok(name) = HeaderName::from_bytes(key.as_bytes()) else {
            return;
        };
        let Ok(value) = HeaderValue::from_str(&value) else {
            return;
        };
        self.0.insert(name, value);
    }
}

fn current_trace_headers() -> HeaderMap {
    let context = tracing::Span::current().context();
    let mut injector = TraceHeaderInjector(HeaderMap::new());
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut injector);
    });
    injector.0
}

// ── InflightGuard ─────────────────────────────────────────────────────────────

/// RAII guard: increments a counter on creation, decrements on drop.
struct InflightGuard(Arc<AtomicUsize>);

impl InflightGuard {
    #[cfg(test)]
    fn acquire(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(Arc::clone(counter))
    }

    fn try_acquire(counter: &Arc<AtomicUsize>, max_inflight: usize) -> Option<Self> {
        let max_inflight = max_inflight.max(1);
        let mut current = counter.load(Ordering::Acquire);
        loop {
            if current >= max_inflight {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Self(Arc::clone(counter))),
                Err(actual) => current = actual,
            }
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let _ = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            });
    }
}

struct SharedReservationGuard {
    store: Arc<dyn FleetStateStore>,
    worker_id: ax_serving_protocol::WorkerId,
    attempt_id: ax_serving_protocol::AttemptId,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

impl SharedReservationGuard {
    fn new(
        store: Arc<dyn FleetStateStore>,
        worker_id: ax_serving_protocol::WorkerId,
        attempt_id: ax_serving_protocol::AttemptId,
        max_concurrent: usize,
        ttl_ms: u64,
    ) -> Self {
        let (stop, mut stopped) = tokio::sync::oneshot::channel();
        let renew_store = Arc::clone(&store);
        let renew_worker = worker_id.clone();
        let renew_every = std::time::Duration::from_millis((ttl_ms / 3).max(250));
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(renew_every) => {
                        match renew_store
                            .try_reserve(
                                &renew_worker,
                                attempt_id,
                                max_concurrent,
                                ttl_ms,
                            )
                            .await
                        {
                            Ok(ReservationResult::Reserved) => {}
                            Ok(ReservationResult::Saturated) => {
                                warn!(%renew_worker, %attempt_id, "shared dispatch reservation renewal was fenced");
                                break;
                            }
                            Err(error) => {
                                warn!(%renew_worker, %attempt_id, %error, "shared dispatch reservation renewal failed");
                            }
                        }
                    }
                    _ = &mut stopped => break,
                }
            }
        });
        Self {
            store,
            worker_id,
            attempt_id,
            stop: Some(stop),
        }
    }
}

impl Drop for SharedReservationGuard {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let store = Arc::clone(&self.store);
        let worker_id = self.worker_id.clone();
        let attempt_id = self.attempt_id;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(error) = store.release_reservation(&worker_id, attempt_id).await {
                    warn!(%worker_id, %attempt_id, %error, "shared dispatch reservation release failed");
                }
            });
        }
    }
}

struct AttemptGuard {
    _inflight: InflightGuard,
    _reservation: Option<SharedReservationGuard>,
}

#[derive(Debug, thiserror::Error)]
enum ReservationAcquireError {
    #[error("worker reservation capacity exhausted")]
    Saturated,
    #[error("shared fleet state is unavailable")]
    Store(#[source] anyhow::Error),
}

// ── DirectDispatcher ──────────────────────────────────────────────────────────

/// HTTP proxy dispatcher for direct (no-broker) mode.
///
/// Holds a shared `reqwest::Client` (connection-pool enabled).
/// Per-request routing state comes from `WorkerRegistry` and the policy. Only a
/// bounded, prompt-free decision journal is retained for operator diagnostics.
#[derive(Clone)]
pub struct DirectDispatcher {
    client: Client,
    reroute_total: Arc<AtomicU64>,
    metrics: Arc<DispatchMetrics>,
    dispatch_token: Option<HeaderValue>,
    first_byte_timeout: std::time::Duration,
    stream_idle_timeout: std::time::Duration,
    fleet_store: Option<Arc<dyn FleetStateStore>>,
    reservation_ttl_ms: u64,
    decision_journal: Arc<DecisionJournal>,
}

const DEFAULT_DECISION_JOURNAL_CAPACITY: usize = 256;
const MAX_RECORDED_DECISION_CANDIDATES: usize = 128;

#[derive(Debug)]
struct DecisionJournal {
    capacity: usize,
    records: Mutex<VecDeque<DecisionRecordV1>>,
}

impl DecisionJournal {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            records: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }

    fn record(&self, record: DecisionRecordV1) {
        let mut records = self.records_lock();
        if records.len() >= self.capacity {
            records.pop_front();
        }
        records.push_back(record);
    }

    fn tail(&self, limit: usize) -> Vec<DecisionRecordV1> {
        let records = self.records_lock();
        let take = limit.min(records.len());
        records
            .iter()
            .skip(records.len().saturating_sub(take))
            .cloned()
            .collect()
    }

    fn records_lock(&self) -> MutexGuard<'_, VecDeque<DecisionRecordV1>> {
        match self.records.lock() {
            Ok(records) => records,
            Err(error) => {
                warn!(%error, "decision journal lock poisoned; continuing with retained records");
                error.into_inner()
            }
        }
    }
}

#[derive(Debug, Default)]
struct DispatchMetrics {
    requests_total: AtomicU64,
    attempts_total: AtomicU64,
    completed_total: AtomicU64,
    failed_total: AtomicU64,
    cancelled_total: AtomicU64,
    endpoint_selection: AtomicLatencyHistogram,
    response_headers: AtomicLatencyHistogram,
    attempt_duration: AtomicLatencyHistogram,
    time_to_first_byte: AtomicLatencyHistogram,
    stream_duration: AtomicLatencyHistogram,
    selection_selected_total: AtomicU64,
    selection_no_candidate_total: AtomicU64,
    selection_at_capacity_total: AtomicU64,
    selection_error_total: AtomicU64,
}

/// Fixed, cumulative microsecond buckets keep the dispatch hot path lock-free
/// and prevent unbounded label/cardinality growth.
pub const GATEWAY_LATENCY_BUCKETS_US: [u64; 9] =
    [50, 100, 250, 500, 1_000, 2_000, 5_000, 15_000, u64::MAX];

#[derive(Debug)]
struct AtomicLatencyHistogram {
    count: AtomicU64,
    sum_us: AtomicU64,
    cumulative_buckets: [AtomicU64; GATEWAY_LATENCY_BUCKETS_US.len()],
}

impl Default for AtomicLatencyHistogram {
    fn default() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            cumulative_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl AtomicLatencyHistogram {
    fn record(&self, elapsed: std::time::Duration) {
        let elapsed_us = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .sum_us
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(elapsed_us))
            });
        for (index, upper_bound) in GATEWAY_LATENCY_BUCKETS_US.iter().enumerate() {
            if elapsed_us <= *upper_bound {
                self.cumulative_buckets[index].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn snapshot(&self) -> LatencyHistogramSnapshot {
        LatencyHistogramSnapshot {
            count: self.count.load(Ordering::Relaxed),
            sum_us: self.sum_us.load(Ordering::Relaxed),
            cumulative_buckets: std::array::from_fn(|index| {
                self.cumulative_buckets[index].load(Ordering::Relaxed)
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyHistogramSnapshot {
    pub count: u64,
    pub sum_us: u64,
    pub cumulative_buckets: [u64; GATEWAY_LATENCY_BUCKETS_US.len()],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionOutcome {
    Selected,
    NoCandidate,
    AtCapacity,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchMetricsSnapshot {
    pub requests_total: u64,
    pub attempts_total: u64,
    pub completed_total: u64,
    pub failed_total: u64,
    pub cancelled_total: u64,
    pub retries_total: u64,
    pub endpoint_selection: LatencyHistogramSnapshot,
    pub response_headers: LatencyHistogramSnapshot,
    pub attempt_duration: LatencyHistogramSnapshot,
    pub time_to_first_byte: LatencyHistogramSnapshot,
    pub stream_duration: LatencyHistogramSnapshot,
    pub selection_selected_total: u64,
    pub selection_no_candidate_total: u64,
    pub selection_at_capacity_total: u64,
    pub selection_error_total: u64,
}

struct DispatchOutcomeGuard {
    metrics: Arc<DispatchMetrics>,
    resolved: bool,
    successful_response: bool,
    attempt_started: std::time::Instant,
    first_byte_recorded: bool,
}

impl DispatchOutcomeGuard {
    fn new(
        metrics: Arc<DispatchMetrics>,
        successful_response: bool,
        attempt_started: std::time::Instant,
    ) -> Self {
        Self {
            metrics,
            resolved: false,
            successful_response,
            attempt_started,
            first_byte_recorded: false,
        }
    }

    fn first_byte(&mut self) {
        if !self.first_byte_recorded {
            self.metrics
                .time_to_first_byte
                .record(self.attempt_started.elapsed());
            self.first_byte_recorded = true;
        }
    }

    fn finish_timing(&self) {
        let elapsed = self.attempt_started.elapsed();
        self.metrics.attempt_duration.record(elapsed);
        self.metrics.stream_duration.record(elapsed);
    }

    fn completed(&mut self) {
        if !self.resolved {
            self.finish_timing();
            if self.successful_response {
                self.metrics.completed_total.fetch_add(1, Ordering::Relaxed);
            }
            self.resolved = true;
        }
    }

    fn failed(&mut self) {
        if !self.resolved {
            self.finish_timing();
            if self.successful_response {
                self.metrics.failed_total.fetch_add(1, Ordering::Relaxed);
            }
            self.resolved = true;
        }
    }
}

impl Drop for DispatchOutcomeGuard {
    fn drop(&mut self) {
        if !self.resolved {
            self.finish_timing();
            if self.successful_response {
                self.metrics.cancelled_total.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// TCP connect timeout for the dispatcher's reqwest client.
/// Short enough to fail fast on unreachable workers without blocking the queue.
const DISPATCHER_CONNECT_TIMEOUT_SECS: u64 = 5;
/// Default pool size and request timeout matching serving.example.yaml defaults.
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 8;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;
/// Maximum buffered non-streaming worker response body.
const MAX_WORKER_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
enum WorkerBodyError {
    TooLarge,
    Read(reqwest::Error),
}

fn response_declares_oversize(content_length: Option<u64>, max_bytes: usize) -> bool {
    content_length.is_some_and(|len| len > max_bytes as u64)
}

fn append_limited_body_chunk(
    body: &mut Vec<u8>,
    chunk: &[u8],
    max_bytes: usize,
) -> Result<(), WorkerBodyError> {
    let next_len = body
        .len()
        .checked_add(chunk.len())
        .ok_or(WorkerBodyError::TooLarge)?;
    if next_len > max_bytes {
        return Err(WorkerBodyError::TooLarge);
    }
    body.extend_from_slice(chunk);
    Ok(())
}

fn add_limited_body_len(
    current_len: usize,
    chunk_len: usize,
    max_bytes: usize,
) -> Result<usize, WorkerBodyError> {
    let next_len = current_len
        .checked_add(chunk_len)
        .ok_or(WorkerBodyError::TooLarge)?;
    if next_len > max_bytes {
        return Err(WorkerBodyError::TooLarge);
    }
    Ok(next_len)
}

async fn read_worker_response_body_limited(
    resp: reqwest::Response,
    max_bytes: usize,
) -> Result<Bytes, WorkerBodyError> {
    if response_declares_oversize(resp.content_length(), max_bytes) {
        return Err(WorkerBodyError::TooLarge);
    }

    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        append_limited_body_chunk(&mut body, &chunk.map_err(WorkerBodyError::Read)?, max_bytes)?;
    }
    Ok(Bytes::from(body))
}

fn should_forward_worker_header(name: &HeaderName, include_content_length: bool) -> bool {
    let name = name.as_str();
    !matches!(
        name,
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "set-cookie"
            | "www-authenticate"
            | "x-ax-admission-state"
            | "x-ax-attempt-id"
            | "x-ax-dispatch-token"
            | "x-ax-deployment-id"
            | "x-ax-domain-id"
            | "x-ax-pool-id"
    ) && (include_content_length || name != header::CONTENT_LENGTH.as_str())
}

fn response_builder_with_worker_headers(
    status: StatusCode,
    headers: &HeaderMap,
    include_content_length: bool,
) -> axum::http::response::Builder {
    let mut builder = axum::response::Response::builder().status(status);
    for (name, value) in headers {
        if should_forward_worker_header(name, include_content_length) {
            builder = builder.header(name, value);
        }
    }
    if !headers.contains_key(header::CONTENT_TYPE) {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }
    builder
}

async fn drain_worker_error_response(resp: reqwest::Response, url: &str) {
    if response_declares_oversize(resp.content_length(), MAX_WORKER_RESPONSE_BODY_BYTES) {
        warn!(
            %url,
            content_length = resp.content_length().unwrap_or_default(),
            limit = MAX_WORKER_RESPONSE_BODY_BYTES,
            "skipping oversized worker error response drain"
        );
        return;
    }

    let mut total_len = 0usize;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                match add_limited_body_len(total_len, chunk.len(), MAX_WORKER_RESPONSE_BODY_BYTES) {
                    Ok(next_len) => total_len = next_len,
                    Err(WorkerBodyError::TooLarge) => {
                        warn!(
                            %url,
                            limit = MAX_WORKER_RESPONSE_BODY_BYTES,
                            "stopping oversized worker error response drain"
                        );
                        return;
                    }
                    Err(WorkerBodyError::Read(_)) => {
                        unreachable!("length accounting does not read")
                    }
                }
            }
            Err(err) => {
                warn!(%url, err = %err, "draining worker error response failed");
                return;
            }
        }
    }
}

impl DirectDispatcher {
    pub fn new(pool_max_idle_per_host: usize, request_timeout_secs: u64) -> Self {
        Self::try_new(pool_max_idle_per_host, request_timeout_secs, None)
            .expect("dispatcher without a credential has valid configuration")
    }

    pub fn try_new(
        pool_max_idle_per_host: usize,
        request_timeout_secs: u64,
        dispatch_token: Option<&str>,
    ) -> anyhow::Result<Self> {
        Self::try_new_with_timeouts(
            pool_max_idle_per_host,
            request_timeout_secs,
            120_000,
            30_000,
            dispatch_token,
        )
    }

    pub fn try_new_with_timeouts(
        pool_max_idle_per_host: usize,
        request_timeout_secs: u64,
        first_byte_timeout_ms: u64,
        stream_idle_timeout_ms: u64,
        dispatch_token: Option<&str>,
    ) -> anyhow::Result<Self> {
        let client = match Client::builder()
            .pool_max_idle_per_host(pool_max_idle_per_host)
            .connect_timeout(std::time::Duration::from_secs(
                DISPATCHER_CONNECT_TIMEOUT_SECS,
            ))
            .timeout(std::time::Duration::from_secs(request_timeout_secs))
            .build()
        {
            Ok(client) => client,
            Err(err) => {
                warn!(
                    %err,
                    pool_max_idle_per_host,
                    request_timeout_secs,
                    "failed to build tuned reqwest client; falling back to default client"
                );
                Client::new()
            }
        };

        let dispatch_token = dispatch_token
            .map(|token| {
                let mut value = HeaderValue::from_str(token).map_err(|_| {
                    anyhow::anyhow!("AXS_DISPATCH_TOKEN is not a valid HTTP header")
                })?;
                value.set_sensitive(true);
                Ok::<_, anyhow::Error>(value)
            })
            .transpose()?;

        Ok(Self {
            client,
            reroute_total: Arc::new(AtomicU64::new(0)),
            metrics: Arc::new(DispatchMetrics::default()),
            dispatch_token,
            first_byte_timeout: std::time::Duration::from_millis(first_byte_timeout_ms.max(1)),
            stream_idle_timeout: std::time::Duration::from_millis(stream_idle_timeout_ms.max(1)),
            fleet_store: None,
            reservation_ttl_ms: 15_000,
            decision_journal: Arc::new(DecisionJournal::new(DEFAULT_DECISION_JOURNAL_CAPACITY)),
        })
    }

    pub fn with_fleet_state(
        mut self,
        fleet_store: Arc<dyn FleetStateStore>,
        reservation_ttl_ms: u64,
    ) -> Self {
        self.fleet_store = Some(fleet_store);
        self.reservation_ttl_ms = reservation_ttl_ms.max(1_000);
        self
    }

    async fn reserve_attempt(
        &self,
        worker_id: Option<ax_serving_protocol::WorkerId>,
        attempt_id: ax_serving_protocol::AttemptId,
        max_concurrent: usize,
    ) -> Result<Option<SharedReservationGuard>, ReservationAcquireError> {
        let (Some(store), Some(worker_id)) = (&self.fleet_store, worker_id) else {
            return Ok(None);
        };
        match store
            .try_reserve(
                &worker_id,
                attempt_id,
                max_concurrent,
                self.reservation_ttl_ms,
            )
            .await
            .map_err(ReservationAcquireError::Store)?
        {
            ReservationResult::Reserved => Ok(Some(SharedReservationGuard::new(
                Arc::clone(store),
                worker_id,
                attempt_id,
                max_concurrent,
                self.reservation_ttl_ms,
            ))),
            ReservationResult::Saturated => Err(ReservationAcquireError::Saturated),
        }
    }

    fn attach_dispatch_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.dispatch_token.as_ref() {
            Some(value) => {
                builder.header(ax_serving_protocol::DISPATCH_TOKEN_HEADER, value.clone())
            }
            None => builder,
        }
    }

    /// Total number of reroutes performed since startup.
    pub fn reroutes(&self) -> u64 {
        self.reroute_total.load(Ordering::Relaxed)
    }

    pub fn record_request_result(&self, success: bool) {
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.metrics.failed_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Return the most recent prompt-free domain decisions in chronological order.
    pub fn decision_records(&self, limit: usize) -> Vec<DecisionRecordV1> {
        self.decision_journal.tail(limit.clamp(1, 200))
    }

    fn record_domain_decision(
        &self,
        profile: &RequestProfile,
        candidates: &[RouteCandidate],
        selected: &RouteCandidate,
    ) {
        let Some(selected_domain) = selected.domain.as_ref() else {
            return;
        };

        let mut seen = BTreeSet::new();
        let mut candidate_summary = Vec::new();
        for candidate in std::iter::once(selected).chain(candidates.iter()) {
            let Some(domain) = candidate.domain.as_ref() else {
                continue;
            };
            let key = (domain.id.clone(), candidate.deployment.id.clone());
            if !seen.insert(key) {
                continue;
            }
            candidate_summary.push(CandidateDecision {
                domain: domain.id.clone(),
                deployment: candidate.deployment.id.clone(),
                eligible: true,
                rejection_reasons: BTreeSet::new(),
                normalized_score_microunits: None,
            });
            if candidate_summary.len() == MAX_RECORDED_DECISION_CANDIDATES {
                break;
            }
        }

        let mut observation_generations = BTreeMap::new();
        for candidate in candidates {
            if let Some(domain) = candidate.domain.as_ref() {
                let generation = candidate
                    .endpoint
                    .domain_observation
                    .as_ref()
                    .map_or(0, |observation| observation.generation);
                observation_generations
                    .entry(domain.id.clone())
                    .and_modify(|accepted: &mut u64| *accepted = (*accepted).max(generation))
                    .or_insert(generation);
            }
        }

        let mut reason_codes = BTreeSet::from([DecisionReasonCode::ExplicitDeployment]);
        if profile.decision.required_domain.as_ref() == Some(&selected_domain.id) {
            reason_codes.insert(DecisionReasonCode::RequiredDomain);
        } else if profile.decision.preferred_domain.as_ref() == Some(&selected_domain.id) {
            reason_codes.insert(DecisionReasonCode::PreferredDomain);
        }
        if candidate_summary.len() == 1 {
            reason_codes.insert(DecisionReasonCode::OnlyEligible);
        }

        let record = DecisionRecordV1 {
            request_id: profile.request_id,
            operation: profile.operation.clone(),
            logical_model: profile.logical_model.clone(),
            routing_profile: profile.decision.routing_profile.clone(),
            policy_id: PolicyId::new("explicit-catalog")
                .expect("static decision policy id is valid"),
            policy_version: PolicyVersion::new(env!("CARGO_PKG_VERSION"))
                .expect("static decision policy version is valid"),
            policy_mode: PolicyMode::Active,
            candidate_summary,
            selected_domain: selected_domain.id.clone(),
            selected_deployment: selected.deployment.id.clone(),
            reason_codes,
            observation_generations,
            predicted_cost_microusd: None,
            predicted_latency_ms: None,
            decided_at: time::OffsetDateTime::now_utc(),
        };
        match record.validate() {
            Ok(()) => self.decision_journal.record(record),
            Err(error) => warn!(%error, "discarding invalid bounded domain decision record"),
        }
    }

    fn record_selection(&self, started: std::time::Instant, outcome: SelectionOutcome) {
        self.metrics.endpoint_selection.record(started.elapsed());
        let counter = match outcome {
            SelectionOutcome::Selected => &self.metrics.selection_selected_total,
            SelectionOutcome::NoCandidate => &self.metrics.selection_no_candidate_total,
            SelectionOutcome::AtCapacity => &self.metrics.selection_at_capacity_total,
            SelectionOutcome::Error => &self.metrics.selection_error_total,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn metrics(&self) -> DispatchMetricsSnapshot {
        DispatchMetricsSnapshot {
            requests_total: self.metrics.requests_total.load(Ordering::Relaxed),
            attempts_total: self.metrics.attempts_total.load(Ordering::Relaxed),
            completed_total: self.metrics.completed_total.load(Ordering::Relaxed),
            failed_total: self.metrics.failed_total.load(Ordering::Relaxed),
            cancelled_total: self.metrics.cancelled_total.load(Ordering::Relaxed),
            retries_total: self.reroutes(),
            endpoint_selection: self.metrics.endpoint_selection.snapshot(),
            response_headers: self.metrics.response_headers.snapshot(),
            attempt_duration: self.metrics.attempt_duration.snapshot(),
            time_to_first_byte: self.metrics.time_to_first_byte.snapshot(),
            stream_duration: self.metrics.stream_duration.snapshot(),
            selection_selected_total: self
                .metrics
                .selection_selected_total
                .load(Ordering::Relaxed),
            selection_no_candidate_total: self
                .metrics
                .selection_no_candidate_total
                .load(Ordering::Relaxed),
            selection_at_capacity_total: self
                .metrics
                .selection_at_capacity_total
                .load(Ordering::Relaxed),
            selection_error_total: self.metrics.selection_error_total.load(Ordering::Relaxed),
        }
    }

    /// Dispatch through the explicit logical-model catalog.
    ///
    /// The only network retry conditions are a proven connection failure or
    /// the trusted agent's typed pre-admission rejection. Generic runtime 5xx
    /// responses are returned to the caller without a second attempt.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_profile(
        &self,
        registry: &WorkerRegistry,
        catalog: &DeploymentCatalog,
        policy: &dyn DispatchPolicy,
        profile: &RequestProfile,
        path: &str,
        body: Bytes,
        telemetry_stale_ms: u64,
        max_dispatch_attempts: u8,
    ) -> Response {
        let request_hash = request_hash(profile.request_id);
        let ctx = DispatchContext {
            model_id: profile.logical_model.as_str(),
            stream: profile.stream,
            preferred_pool: profile.preferred_pool.as_ref().map(|pool| pool.as_str()),
            request_hash,
            cache_affinity_key: profile.cache_affinity_key,
            telemetry_stale_ms,
        };
        let maximum_attempts = max_dispatch_attempts.clamp(1, 2);
        let mut attempt_number = 0u8;
        let mut excluded_id = None;
        let mut retry_source = None;

        loop {
            let Some(remaining) = profile.remaining() else {
                return ax_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    profile.request_id,
                    "AXS_REQUEST_DEADLINE",
                    "request deadline expired before dispatch completed",
                    false,
                    ax_serving_protocol::AdmissionPhase::Admission,
                );
            };
            let selection_started = std::time::Instant::now();
            let candidates = match catalog.route_candidates(
                registry,
                profile,
                excluded_id,
                retry_source.as_ref(),
            ) {
                Ok(candidates) => candidates,
                Err(error) => {
                    self.record_selection(selection_started, SelectionOutcome::Error);
                    let unknown_model = error.to_string().starts_with("unknown logical model");
                    return trace_response(
                        ax_error_response(
                            if unknown_model {
                                StatusCode::NOT_FOUND
                            } else {
                                StatusCode::SERVICE_UNAVAILABLE
                            },
                            profile.request_id,
                            if unknown_model {
                                "AXS_UNKNOWN_MODEL"
                            } else {
                                "AXS_DEPLOYMENT_RESOLUTION"
                            },
                            if unknown_model {
                                error.to_string()
                            } else {
                                "deployment resolution is unavailable".to_string()
                            },
                            false,
                            ax_serving_protocol::AdmissionPhase::EndpointSelection,
                        ),
                        0,
                        None,
                        "deployment_resolution_failed",
                    );
                }
            };
            if candidates.is_empty() {
                self.record_selection(selection_started, SelectionOutcome::NoCandidate);
                let reason = if attempt_number == 0 {
                    "no_compatible_deployment"
                } else {
                    "safe_retry_no_candidate"
                };
                return trace_response(
                    ax_error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        profile.request_id,
                        "AXS_NO_COMPATIBLE_DEPLOYMENT",
                        format!(
                            "no compatible ready deployment for logical model '{}'",
                            profile.logical_model
                        ),
                        true,
                        ax_serving_protocol::AdmissionPhase::EndpointSelection,
                    ),
                    0,
                    None,
                    reason,
                );
            }

            let statuses = candidates
                .iter()
                .map(|candidate| candidate.endpoint.worker.clone())
                .collect::<Vec<_>>();
            let Some(selected) = policy.select(&statuses, &ctx) else {
                self.record_selection(selection_started, SelectionOutcome::AtCapacity);
                return trace_response(
                    ax_error_response(
                        StatusCode::TOO_MANY_REQUESTS,
                        profile.request_id,
                        "AXS_DEPLOYMENT_CAPACITY",
                        format!(
                            "all compatible deployments for '{}' are at capacity",
                            profile.logical_model
                        ),
                        true,
                        ax_serving_protocol::AdmissionPhase::Admission,
                    ),
                    candidates.len(),
                    None,
                    "all_at_capacity",
                );
            };
            self.record_selection(selection_started, SelectionOutcome::Selected);
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.endpoint.worker.id == selected.id)
                .expect("policy selection must come from the candidate snapshot")
                .clone();
            let selected_id = candidate.endpoint.worker.id;
            let Some(inflight_counter) = registry.inflight_counter(selected_id) else {
                excluded_id = Some(selected_id);
                continue;
            };
            let Some(guard) = InflightGuard::try_acquire(
                &inflight_counter,
                candidate.endpoint.worker.max_inflight,
            ) else {
                excluded_id = Some(selected_id);
                continue;
            };

            let attempt_id = ax_serving_protocol::AttemptId::new();
            let protocol_worker_id = candidate
                .endpoint
                .protocol_worker_id
                .as_deref()
                .and_then(|value| value.parse::<ax_serving_protocol::WorkerId>().ok());
            let reservation = match self
                .reserve_attempt(
                    protocol_worker_id,
                    attempt_id,
                    candidate.endpoint.worker.max_inflight,
                )
                .await
            {
                Ok(reservation) => reservation,
                Err(ReservationAcquireError::Saturated) => {
                    drop(guard);
                    excluded_id = Some(selected_id);
                    continue;
                }
                Err(ReservationAcquireError::Store(error)) => {
                    warn!(%error, "shared fleet state rejected dispatch reservation");
                    drop(guard);
                    return ax_error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        profile.request_id,
                        "AXS_FLEET_STATE_UNAVAILABLE",
                        "shared fleet state is temporarily unavailable",
                        true,
                        ax_serving_protocol::AdmissionPhase::Admission,
                    );
                }
            };
            let attempt_guard = AttemptGuard {
                _inflight: guard,
                _reservation: reservation,
            };
            let rewritten_body = match rewrite_runtime_model(
                &body,
                candidate.deployment.runtime_model_id.as_str(),
            ) {
                Ok(body) => body,
                Err(error) => {
                    drop(attempt_guard);
                    return ax_error_response(
                        StatusCode::BAD_REQUEST,
                        profile.request_id,
                        "AXS_INVALID_ROUTING_FIELD",
                        format!("invalid model routing field: {error}"),
                        false,
                        ax_serving_protocol::AdmissionPhase::Admission,
                    );
                }
            };
            self.record_domain_decision(profile, &candidates, &candidate);
            attempt_number = attempt_number.saturating_add(1);
            let url = worker_url(&candidate.endpoint.worker.addr, path);
            debug!(
                request_id = %profile.request_id,
                %attempt_id,
                attempt_number,
                worker_id = %selected_id,
                deployment_id = %candidate.deployment.id,
                pool_id = %candidate.pool.id,
                domain_id = candidate.domain.as_ref().map(|domain| domain.id.as_str()),
                runtime_kind = %candidate.endpoint.runtime_kind,
                telemetry_age_ms = candidate.endpoint.worker.telemetry_age_ms,
                "dispatch attempt selected"
            );
            self.metrics.attempts_total.fetch_add(1, Ordering::Relaxed);
            let dispatch_span = info_span!(
                "axs.dispatch",
                otel.kind = "client",
                axs.request.id = %profile.request_id,
                axs.attempt.id = %attempt_id,
                axs.attempt.number = attempt_number,
                axs.deployment.id = %candidate.deployment.id,
                axs.pool.id = %candidate.pool.id,
                axs.domain.id = candidate
                    .domain
                    .as_ref()
                    .map(|domain| domain.id.as_str())
                    .unwrap_or(""),
                axs.runtime.kind = %candidate.endpoint.runtime_kind,
                http.response.status_code = tracing::field::Empty,
                otel.status_code = tracing::field::Empty,
            );
            let mut request = self
                .attach_dispatch_auth(self.client.post(&url))
                .header("content-type", "application/json")
                .header(
                    ax_serving_protocol::REQUEST_ID_HEADER,
                    profile.request_id.to_string(),
                )
                .header(
                    ax_serving_protocol::ATTEMPT_ID_HEADER,
                    attempt_id.to_string(),
                )
                .header("x-ax-deployment-id", candidate.deployment.id.to_string())
                .header("x-ax-pool-id", candidate.pool.id.to_string())
                .timeout(remaining)
                .body(rewritten_body);
            if let Some(domain) = candidate.domain.as_ref() {
                request = request.header("x-ax-domain-id", domain.id.to_string());
            }
            let response_headers_started = std::time::Instant::now();
            let result = async move { request.headers(current_trace_headers()).send().await }
                .instrument(dispatch_span.clone())
                .await;
            self.metrics
                .response_headers
                .record(response_headers_started.elapsed());
            match &result {
                Ok(response) => {
                    dispatch_span.record("http.response.status_code", response.status().as_u16());
                }
                Err(_) => {
                    dispatch_span.record("otel.status_code", "ERROR");
                }
            };

            let retryable_connect_failure = result
                .as_ref()
                .err()
                .is_some_and(|error| error.is_connect() && !error.is_timeout());
            let retryable_not_admitted = result.as_ref().ok().is_some_and(is_typed_not_admitted);
            if attempt_number < maximum_attempts
                && (retryable_connect_failure || retryable_not_admitted)
            {
                if let Ok(response) = result {
                    drain_worker_error_response(response, &url).await;
                } else {
                    registry.mark_unhealthy(selected_id);
                }
                drop(attempt_guard);
                excluded_id = Some(selected_id);
                retry_source = Some(candidate.deployment.clone());
                self.reroute_total.fetch_add(1, Ordering::Relaxed);
                debug!(
                    request_id = %profile.request_id,
                    %attempt_id,
                    worker_id = %selected_id,
                    retry_reason = if retryable_connect_failure {
                        "connect_failure"
                    } else {
                        "not_admitted"
                    },
                    "dispatch attempt eligible for one safe retry"
                );
                continue;
            }

            if retryable_not_admitted {
                if let Ok(response) = result {
                    drain_worker_error_response(response, &url).await;
                }
                drop(attempt_guard);
                return trace_response(
                    ax_error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        profile.request_id,
                        "AXS_WORKER_NOT_ADMITTED",
                        "all selected workers rejected the request before admission",
                        true,
                        ax_serving_protocol::AdmissionPhase::PreAdmission,
                    ),
                    candidates.len(),
                    Some(selected_id),
                    "not_admitted",
                );
            }
            if let Err(error) = result {
                let connect_failure = error.is_connect();
                let deadline = error.is_timeout();
                drop(attempt_guard);
                return trace_response(
                    ax_error_response(
                        if deadline {
                            StatusCode::GATEWAY_TIMEOUT
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        },
                        profile.request_id,
                        if deadline {
                            "AXS_REQUEST_DEADLINE"
                        } else if connect_failure {
                            "AXS_WORKER_CONNECT_FAILED"
                        } else {
                            "AXS_RUNTIME_TRANSPORT"
                        },
                        if deadline {
                            "request deadline expired during runtime dispatch"
                        } else if connect_failure {
                            "no selected worker accepted a connection"
                        } else {
                            "the runtime transport failed after dispatch"
                        },
                        connect_failure && !deadline,
                        if deadline {
                            ax_serving_protocol::AdmissionPhase::PostAdmission
                        } else if connect_failure {
                            ax_serving_protocol::AdmissionPhase::Connecting
                        } else {
                            ax_serving_protocol::AdmissionPhase::PostAdmission
                        },
                    ),
                    candidates.len(),
                    Some(selected_id),
                    "transport_failed",
                );
            }

            if matches!(&result, Ok(response) if response.status().is_success()) {
                policy.record_dispatch_context(selected_id, &ctx);
            }
            let reason = if attempt_number > 1 {
                "safe_retry"
            } else {
                "primary"
            };
            return trace_response(
                self.build_response(
                    result,
                    url,
                    profile.stream,
                    attempt_guard,
                    response_headers_started,
                )
                .await,
                candidates.len(),
                Some(selected_id),
                reason,
            );
        }
    }

    /// Forward a request to the selected worker.
    ///
    /// The legacy `auth_header` parameter is ignored. Public credentials
    /// terminate at the gateway and are never valid worker credentials.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward(
        &self,
        registry: &WorkerRegistry,
        policy: &dyn DispatchPolicy,
        model_id: &str,
        stream: bool,
        preferred_pool: Option<&str>,
        path: &str,
        body: Bytes,
        _auth_header: Option<&HeaderValue>,
    ) -> Response {
        self.forward_kind(
            registry,
            policy,
            model_id,
            RequestKind::Llm,
            None,
            None,
            stream,
            preferred_pool,
            false,
            path,
            body,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn forward_kind(
        &self,
        registry: &WorkerRegistry,
        policy: &dyn DispatchPolicy,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
        stream: bool,
        preferred_pool: Option<&str>,
        require_preferred_pool: bool,
        path: &str,
        body: Bytes,
        _auth_header: Option<&HeaderValue>,
    ) -> Response {
        self.forward_kind_until(
            registry,
            policy,
            model_id,
            request_kind,
            backend_hint,
            min_context,
            stream,
            preferred_pool,
            require_preferred_pool,
            path,
            body,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn forward_kind_until(
        &self,
        registry: &WorkerRegistry,
        policy: &dyn DispatchPolicy,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
        stream: bool,
        preferred_pool: Option<&str>,
        require_preferred_pool: bool,
        path: &str,
        body: Bytes,
        deadline: Option<tokio::time::Instant>,
    ) -> Response {
        let request_id = ax_serving_protocol::RequestId::new();
        if deadline.is_some_and(|deadline| deadline <= tokio::time::Instant::now()) {
            return ax_error_response(
                StatusCode::GATEWAY_TIMEOUT,
                request_id,
                "AXS_REQUEST_DEADLINE",
                "request deadline expired before dispatch",
                false,
                ax_serving_protocol::AdmissionPhase::Admission,
            );
        }
        let selection_started = std::time::Instant::now();
        let workers = registry.dispatch_workers_filtered_with_pool_mode(
            model_id,
            request_kind,
            backend_hint,
            min_context,
            preferred_pool,
            require_preferred_pool,
            None,
        );
        if workers.is_empty() {
            self.record_selection(selection_started, SelectionOutcome::NoCandidate);
            return trace_response(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no eligible workers for model '{model_id}'"),
                )
                    .into_response(),
                0,
                None,
                "no_eligible_worker",
            );
        }
        let candidate_count = workers.len();

        let ctx = DispatchContext {
            model_id,
            stream,
            preferred_pool,
            request_hash: 0,
            cache_affinity_key: None,
            telemetry_stale_ms: 10_000,
        };
        let selected = match policy.select(&workers, &ctx) {
            Some(w) => w,
            None => {
                self.record_selection(selection_started, SelectionOutcome::AtCapacity);
                return trace_response(
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("all workers for '{model_id}' are at capacity"),
                    )
                        .into_response(),
                    candidate_count,
                    None,
                    "all_at_capacity",
                );
            }
        };
        self.record_selection(selection_started, SelectionOutcome::Selected);

        let selected_id = selected.id;
        let url = worker_url(&selected.addr, path);
        // `Bytes::clone` is shallow, and soft pool preferences can hide fallback
        // candidates from the initial selection set. Keep a retry body so the
        // reroute pass can relax soft preferences after excluding the failed worker.
        let retry_body = body.clone();
        let Some(inflight_counter) = registry.inflight_counter(selected_id) else {
            warn!(
                worker_id = %selected_id,
                "selected worker disappeared before dispatch"
            );
            self.reroute_total.fetch_add(1, Ordering::Relaxed);
            return self
                .reroute(
                    registry,
                    policy,
                    &ctx,
                    request_kind,
                    backend_hint,
                    min_context,
                    path,
                    retry_body,
                    selected_id,
                    require_preferred_pool,
                    request_id,
                    deadline,
                    None,
                )
                .await;
        };
        let Some(guard) = InflightGuard::try_acquire(&inflight_counter, selected.max_inflight)
        else {
            warn!(
                worker_id = %selected_id,
                max_inflight = selected.max_inflight,
                "selected worker reached capacity before dispatch"
            );
            self.reroute_total.fetch_add(1, Ordering::Relaxed);
            return self
                .reroute(
                    registry,
                    policy,
                    &ctx,
                    request_kind,
                    backend_hint,
                    min_context,
                    path,
                    retry_body,
                    selected_id,
                    require_preferred_pool,
                    request_id,
                    deadline,
                    None,
                )
                .await;
        };

        let attempt_id = ax_serving_protocol::AttemptId::new();
        let reservation_worker_id = registry
            .protocol_identity_for_internal(selected_id)
            .map(|(worker_id, _)| worker_id);
        let reservation = match self
            .reserve_attempt(reservation_worker_id, attempt_id, selected.max_inflight)
            .await
        {
            Ok(reservation) => reservation,
            Err(ReservationAcquireError::Saturated) => {
                drop(guard);
                return self
                    .reroute(
                        registry,
                        policy,
                        &ctx,
                        request_kind,
                        backend_hint,
                        min_context,
                        path,
                        retry_body,
                        selected_id,
                        require_preferred_pool,
                        request_id,
                        deadline,
                        None,
                    )
                    .await;
            }
            Err(ReservationAcquireError::Store(error)) => {
                warn!(%error, "shared fleet state rejected dispatch reservation");
                drop(guard);
                return ax_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    request_id,
                    "AXS_FLEET_STATE_UNAVAILABLE",
                    "shared fleet state is temporarily unavailable",
                    true,
                    ax_serving_protocol::AdmissionPhase::Admission,
                );
            }
        };
        let attempt_guard = AttemptGuard {
            _inflight: guard,
            _reservation: reservation,
        };

        let mut request = self
            .attach_dispatch_auth(self.client.post(&url))
            .header("content-type", "application/json")
            .header(
                ax_serving_protocol::REQUEST_ID_HEADER,
                request_id.to_string(),
            )
            .header(
                ax_serving_protocol::ATTEMPT_ID_HEADER,
                attempt_id.to_string(),
            )
            .body(body);
        if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(tokio::time::Instant::now())
            else {
                drop(attempt_guard);
                return ax_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    request_id,
                    "AXS_REQUEST_DEADLINE",
                    "request deadline expired before worker dispatch",
                    false,
                    ax_serving_protocol::AdmissionPhase::Connecting,
                );
            };
            request = request.timeout(remaining);
        }
        self.metrics.attempts_total.fetch_add(1, Ordering::Relaxed);
        let dispatch_span = info_span!(
            "axs.dispatch",
            otel.kind = "client",
            axs.request.id = %request_id,
            axs.attempt.id = %attempt_id,
            axs.attempt.number = 1_u8,
            axs.runtime.model = model_id,
            http.response.status_code = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
        );
        let response_headers_started = std::time::Instant::now();
        let result = async move { request.headers(current_trace_headers()).send().await }
            .instrument(dispatch_span.clone())
            .await;
        self.metrics
            .response_headers
            .record(response_headers_started.elapsed());
        match &result {
            Ok(response) => {
                dispatch_span.record("http.response.status_code", response.status().as_u16());
            }
            Err(_) => {
                dispatch_span.record("otel.status_code", "ERROR");
            }
        };

        let retryable_connect_failure = result
            .as_ref()
            .err()
            .is_some_and(|error| error.is_connect() && !error.is_timeout());
        let retryable_not_admitted = result.as_ref().ok().is_some_and(is_typed_not_admitted);

        if retryable_connect_failure || retryable_not_admitted {
            match &result {
                Err(error) => warn!(%url, err = %error, "worker connect failed, rerouting"),
                Ok(response) => warn!(
                    %url,
                    status = response.status().as_u16(),
                    "worker rejected before admission, rerouting"
                ),
            }
            // Drain the error response body so the connection can be reused
            // from the pool instead of being discarded.
            if let Ok(resp) = result {
                drain_worker_error_response(resp, &url).await;
            }
            drop(attempt_guard);
            if retryable_connect_failure {
                registry.mark_unhealthy(selected_id);
            }
            self.reroute_total.fetch_add(1, Ordering::Relaxed);

            return self
                .reroute(
                    registry,
                    policy,
                    &ctx,
                    request_kind,
                    backend_hint,
                    min_context,
                    path,
                    retry_body,
                    selected_id,
                    require_preferred_pool,
                    request_id,
                    deadline,
                    None,
                )
                .await;
        }

        if result
            .as_ref()
            .err()
            .is_some_and(reqwest::Error::is_timeout)
        {
            drop(attempt_guard);
            return ax_error_response(
                StatusCode::GATEWAY_TIMEOUT,
                request_id,
                "AXS_REQUEST_DEADLINE",
                "request deadline expired during worker dispatch",
                false,
                ax_serving_protocol::AdmissionPhase::PostAdmission,
            );
        }

        // Record affinity only on 2xx — not on 4xx, to avoid biasing future
        // dispatch towards workers that returned client errors.
        if matches!(&result, Ok(r) if r.status().is_success()) {
            policy.record_dispatch_context(selected_id, &ctx);
        }
        trace_response(
            self.build_response(result, url, stream, attempt_guard, response_headers_started)
                .await,
            candidate_count,
            Some(selected_id),
            "primary",
        )
    }

    /// Try once more with a different worker (excluding `excluded_id`).
    #[allow(clippy::too_many_arguments)]
    async fn reroute(
        &self,
        registry: &WorkerRegistry,
        policy: &dyn DispatchPolicy,
        ctx: &DispatchContext<'_>,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
        path: &str,
        body: Bytes,
        excluded_id: WorkerId,
        require_preferred_pool: bool,
        request_id: ax_serving_protocol::RequestId,
        deadline: Option<tokio::time::Instant>,
        _auth_header: Option<&HeaderValue>,
    ) -> Response {
        let selection_started = std::time::Instant::now();
        let candidates = registry
            .dispatch_workers_filtered_with_pool_mode(
                ctx.model_id,
                request_kind,
                backend_hint,
                min_context,
                ctx.preferred_pool,
                require_preferred_pool,
                Some(excluded_id),
            )
            .into_iter()
            .filter(|candidate| {
                registry.legacy_retry_compatible(excluded_id, candidate.id, ctx.model_id)
            })
            .collect::<Vec<_>>();

        let selected2 = match policy.select(&candidates, ctx) {
            Some(w) => w,
            None => {
                self.record_selection(
                    selection_started,
                    if candidates.is_empty() {
                        SelectionOutcome::NoCandidate
                    } else {
                        SelectionOutcome::AtCapacity
                    },
                );
                return trace_response(
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("no alternative worker for '{}' after reroute", ctx.model_id),
                    )
                        .into_response(),
                    candidates.len(),
                    None,
                    "reroute_no_candidate",
                );
            }
        };
        self.record_selection(selection_started, SelectionOutcome::Selected);

        let selected2_id = selected2.id;
        let url2 = worker_url(&selected2.addr, path);
        let Some(inflight_counter2) = registry.inflight_counter(selected2_id) else {
            warn!(
                worker_id = %selected2_id,
                "reroute worker disappeared before dispatch"
            );
            return trace_response(
                worker_failure_response("all workers failed for this request"),
                candidates.len(),
                Some(selected2_id),
                "reroute_target_unavailable",
            );
        };
        let Some(guard2) = InflightGuard::try_acquire(&inflight_counter2, selected2.max_inflight)
        else {
            warn!(
                worker_id = %selected2_id,
                max_inflight = selected2.max_inflight,
                "reroute worker reached capacity before dispatch"
            );
            return trace_response(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no alternative worker for '{}' after reroute", ctx.model_id),
                )
                    .into_response(),
                candidates.len(),
                Some(selected2_id),
                "reroute_target_at_capacity",
            );
        };

        let attempt_id = ax_serving_protocol::AttemptId::new();
        let reservation_worker_id = registry
            .protocol_identity_for_internal(selected2_id)
            .map(|(worker_id, _)| worker_id);
        let reservation = match self
            .reserve_attempt(reservation_worker_id, attempt_id, selected2.max_inflight)
            .await
        {
            Ok(reservation) => reservation,
            Err(ReservationAcquireError::Saturated) => {
                drop(guard2);
                return trace_response(
                    ax_error_response(
                        StatusCode::TOO_MANY_REQUESTS,
                        request_id,
                        "AXS_DEPLOYMENT_CAPACITY",
                        "safe retry target reached shared capacity",
                        true,
                        ax_serving_protocol::AdmissionPhase::Admission,
                    ),
                    candidates.len(),
                    Some(selected2_id),
                    "reroute_target_at_capacity",
                );
            }
            Err(ReservationAcquireError::Store(error)) => {
                warn!(%error, "shared fleet state rejected safe-retry reservation");
                drop(guard2);
                return ax_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    request_id,
                    "AXS_FLEET_STATE_UNAVAILABLE",
                    "shared fleet state is temporarily unavailable",
                    true,
                    ax_serving_protocol::AdmissionPhase::Admission,
                );
            }
        };
        let attempt_guard2 = AttemptGuard {
            _inflight: guard2,
            _reservation: reservation,
        };

        let mut request = self
            .attach_dispatch_auth(self.client.post(&url2))
            .header("content-type", "application/json")
            .header(
                ax_serving_protocol::REQUEST_ID_HEADER,
                request_id.to_string(),
            )
            .header(
                ax_serving_protocol::ATTEMPT_ID_HEADER,
                attempt_id.to_string(),
            )
            .body(body);
        if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(tokio::time::Instant::now())
            else {
                drop(attempt_guard2);
                return ax_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    request_id,
                    "AXS_REQUEST_DEADLINE",
                    "request deadline expired before safe retry",
                    false,
                    ax_serving_protocol::AdmissionPhase::Connecting,
                );
            };
            request = request.timeout(remaining);
        }
        self.metrics.attempts_total.fetch_add(1, Ordering::Relaxed);
        let dispatch_span = info_span!(
            "axs.dispatch",
            otel.kind = "client",
            axs.request.id = %request_id,
            axs.attempt.id = %attempt_id,
            axs.attempt.number = 2_u8,
            axs.runtime.model = ctx.model_id,
            http.response.status_code = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
        );
        let response_headers_started = std::time::Instant::now();
        let result2 = async move { request.headers(current_trace_headers()).send().await }
            .instrument(dispatch_span.clone())
            .await;
        self.metrics
            .response_headers
            .record(response_headers_started.elapsed());
        match &result2 {
            Ok(response) => {
                dispatch_span.record("http.response.status_code", response.status().as_u16());
            }
            Err(_) => {
                dispatch_span.record("otel.status_code", "ERROR");
            }
        };

        if result2
            .as_ref()
            .err()
            .is_some_and(reqwest::Error::is_connect)
        {
            registry.mark_unhealthy(selected2_id);
        }
        if result2
            .as_ref()
            .err()
            .is_some_and(reqwest::Error::is_timeout)
        {
            drop(attempt_guard2);
            return ax_error_response(
                StatusCode::GATEWAY_TIMEOUT,
                request_id,
                "AXS_REQUEST_DEADLINE",
                "request deadline expired during safe retry",
                false,
                ax_serving_protocol::AdmissionPhase::PostAdmission,
            );
        }

        // Record affinity only on 2xx — not on 4xx, to avoid biasing future
        // dispatch towards workers that returned client errors.
        if matches!(&result2, Ok(r) if r.status().is_success()) {
            policy.record_dispatch_context(selected2_id, ctx);
        }

        trace_response(
            self.build_response(
                result2,
                url2,
                ctx.stream,
                attempt_guard2,
                response_headers_started,
            )
            .await,
            candidates.len(),
            Some(selected2_id),
            "reroute",
        )
    }

    /// Build an axum `Response` from a reqwest result.
    ///
    /// For streaming responses the `guard` lives inside the stream and is
    /// dropped when the stream is exhausted or the client disconnects.
    async fn build_response(
        &self,
        result: reqwest::Result<reqwest::Response>,
        url: String,
        stream: bool,
        guard: AttemptGuard,
        attempt_started: std::time::Instant,
    ) -> Response {
        match result {
            Err(e) => {
                self.metrics
                    .attempt_duration
                    .record(attempt_started.elapsed());
                warn!(%url, err = %e, "dispatch request failed");
                worker_failure_response(e.to_string())
            }
            Ok(resp) => {
                let status = StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let headers = resp.headers().clone();

                if stream {
                    type GuardedResponseStream =
                        std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>;
                    let byte_stream: GuardedResponseStream =
                        Box::pin(resp.bytes_stream().map_err(std::io::Error::other));
                    let stream_url = url.clone();
                    let first_byte_timeout = self.first_byte_timeout;
                    let stream_idle_timeout = self.stream_idle_timeout;
                    let outcome = Some(DispatchOutcomeGuard::new(
                        Arc::clone(&self.metrics),
                        status.is_success(),
                        attempt_started,
                    ));
                    let guarded = futures::stream::unfold(
                        (byte_stream, Some(guard), outcome, 0usize, false, true),
                        move |(mut inner, guard, mut outcome, total_len, done, first_byte): (
                            GuardedResponseStream,
                            Option<AttemptGuard>,
                            Option<DispatchOutcomeGuard>,
                            usize,
                            bool,
                            bool,
                        )| {
                            let stream_url = stream_url.clone();
                            async move {
                                if done {
                                    drop(guard);
                                    drop(outcome);
                                    return None;
                                }
                                let wait = if first_byte {
                                    first_byte_timeout
                                } else {
                                    stream_idle_timeout
                                };
                                match tokio::time::timeout(wait, inner.next()).await {
                                    Err(_) => {
                                        error!(
                                            url = %stream_url,
                                            timeout_ms = wait.as_millis(),
                                            first_byte,
                                            "worker streaming response deadline expired"
                                        );
                                        drop(guard);
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.failed();
                                        }
                                        Some((
                                            Err(std::io::Error::new(
                                                std::io::ErrorKind::TimedOut,
                                                if first_byte {
                                                    "worker first-byte deadline expired"
                                                } else {
                                                    "worker stream idle deadline expired"
                                                },
                                            )),
                                            (inner, None, None, total_len, true, false),
                                        ))
                                    }
                                    Ok(Some(Ok(chunk))) => {
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.first_byte();
                                        }
                                        match add_limited_body_len(
                                            total_len,
                                            chunk.len(),
                                            MAX_WORKER_RESPONSE_BODY_BYTES,
                                        ) {
                                            Ok(next_len) => Some((
                                                Ok(chunk),
                                                (inner, guard, outcome, next_len, false, false),
                                            )),
                                            Err(WorkerBodyError::TooLarge) => {
                                                error!(
                                                    url = %stream_url,
                                                    limit = MAX_WORKER_RESPONSE_BODY_BYTES,
                                                    "worker streaming response body exceeded size limit"
                                                );
                                                drop(guard);
                                                if let Some(outcome) = outcome.as_mut() {
                                                    outcome.failed();
                                                }
                                                Some((
                                                    Err(std::io::Error::new(
                                                        std::io::ErrorKind::InvalidData,
                                                        "worker streaming response body exceeded size limit",
                                                    )),
                                                    (inner, None, None, total_len, true, false),
                                                ))
                                            }
                                            Err(WorkerBodyError::Read(_)) => {
                                                unreachable!("length accounting does not read")
                                            }
                                        }
                                    }
                                    Ok(Some(Err(err))) => {
                                        drop(guard);
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.failed();
                                        }
                                        Some((
                                            Err(err),
                                            (inner, None, None, total_len, true, false),
                                        ))
                                    }
                                    Ok(None) => {
                                        drop(guard);
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.completed();
                                        }
                                        None
                                    }
                                }
                            }
                        },
                    );

                    response_builder_with_worker_headers(status, &headers, false)
                        .body(Body::from_stream(guarded))
                        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                } else {
                    let body_result =
                        read_worker_response_body_limited(resp, MAX_WORKER_RESPONSE_BODY_BYTES)
                            .await;
                    self.metrics
                        .attempt_duration
                        .record(attempt_started.elapsed());
                    match body_result {
                        Err(WorkerBodyError::TooLarge) => {
                            error!(
                                %url,
                                limit = MAX_WORKER_RESPONSE_BODY_BYTES,
                                "worker response body exceeded size limit"
                            );
                            worker_failure_response(format!(
                                "worker response body exceeded {} byte limit",
                                MAX_WORKER_RESPONSE_BODY_BYTES
                            ))
                        }
                        Ok(bytes) => {
                            if status.is_success() {
                                self.metrics.completed_total.fetch_add(1, Ordering::Relaxed);
                            }
                            response_builder_with_worker_headers(status, &headers, true)
                                .body(Body::from(bytes))
                                .unwrap_or_else(|_| {
                                    StatusCode::INTERNAL_SERVER_ERROR.into_response()
                                })
                        }
                        Err(WorkerBodyError::Read(e)) => {
                            error!(%url, err = %e, "reading worker response body failed");
                            worker_failure_response(e.to_string())
                        }
                    }
                }
            }
        }
    }
}

fn request_hash(request_id: ax_serving_protocol::RequestId) -> u64 {
    let bytes = request_id.as_uuid().into_bytes();
    u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("UUID has at least eight bytes"),
    )
}

fn is_typed_not_admitted(response: &reqwest::Response) -> bool {
    !response.status().is_success()
        && response
            .headers()
            .get(ax_serving_protocol::ADMISSION_STATE_HEADER)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.eq_ignore_ascii_case("not-admitted"))
}

impl Default for DirectDispatcher {
    fn default() -> Self {
        Self::new(DEFAULT_POOL_MAX_IDLE_PER_HOST, DEFAULT_REQUEST_TIMEOUT_SECS)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn worker_url(addr: &super::worker_endpoint::WorkerEndpoint, path: &str) -> String {
    addr.join_path(path)
}

fn routing_trace_enabled() -> bool {
    std::env::var("AXS_ROUTING_TRACE")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}

fn trace_response(
    mut response: Response,
    candidates: usize,
    selected: Option<WorkerId>,
    reason: &'static str,
) -> Response {
    if !routing_trace_enabled() {
        return response;
    }

    let selected = selected
        .map(|id| id.to_string())
        .unwrap_or_else(|| "none".to_string());
    let value = format!("candidates={candidates},selected={selected},reason={reason}");
    if let Ok(header) = HeaderValue::from_str(&value) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-ax-routing-trace"), header);
    }
    response
}

fn worker_failure_response(_internal_message: impl Into<String>) -> Response {
    let mut resp = (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::Json(serde_json::json!({
            "error": {
                "message": "worker transport is unavailable",
                "type": "server_error",
                "param": null,
                "code": "AXS_WORKER_UNAVAILABLE"
            }
        })),
    )
        .into_response();
    resp.headers_mut().insert(
        HeaderName::from_static("x-reason"),
        HeaderValue::from_static("worker_crash"),
    );
    resp
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::body;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::{
        AtomicLatencyHistogram, AttemptGuard, DirectDispatcher, GATEWAY_LATENCY_BUCKETS_US,
        InflightGuard, InflightGuard as Guard, add_limited_body_len, append_limited_body_chunk,
        response_declares_oversize, worker_url,
    };

    #[test]
    fn latency_histogram_uses_cumulative_bounded_buckets() {
        let histogram = AtomicLatencyHistogram::default();
        histogram.record(std::time::Duration::from_micros(75));
        histogram.record(std::time::Duration::from_micros(3_000));

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.count, 2);
        assert_eq!(snapshot.sum_us, 3_075);
        assert_eq!(snapshot.cumulative_buckets[0], 0);
        assert_eq!(snapshot.cumulative_buckets[1], 1);
        assert_eq!(
            snapshot.cumulative_buckets[GATEWAY_LATENCY_BUCKETS_US.len() - 1],
            2
        );
    }

    #[test]
    fn worker_url_with_leading_slash() {
        let addr = super::super::worker_endpoint::WorkerEndpoint::parse("127.0.0.1:8081").unwrap();
        assert_eq!(
            worker_url(&addr, "/v1/chat/completions"),
            "http://127.0.0.1:8081/v1/chat/completions"
        );
    }

    #[test]
    fn worker_url_without_leading_slash_adds_root() {
        let addr = super::super::worker_endpoint::WorkerEndpoint::parse("127.0.0.1:8081").unwrap();
        assert_eq!(
            worker_url(&addr, "v1/completions"),
            "http://127.0.0.1:8081/v1/completions"
        );
    }

    #[test]
    fn inflight_guard_increments_on_acquire_decrements_on_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        let guard = InflightGuard::acquire(&counter);
        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "must increment on acquire"
        );

        drop(guard);
        assert_eq!(counter.load(Ordering::Relaxed), 0, "must decrement on drop");
    }

    #[test]
    fn inflight_guard_multiple_concurrent_guards() {
        let counter = Arc::new(AtomicUsize::new(0));
        let g1 = InflightGuard::acquire(&counter);
        let g2 = InflightGuard::acquire(&counter);
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        drop(g1);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        drop(g2);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn inflight_guard_try_acquire_respects_max_inflight() {
        let counter = Arc::new(AtomicUsize::new(0));
        let g1 = InflightGuard::try_acquire(&counter, 1).expect("first slot");
        assert!(InflightGuard::try_acquire(&counter, 1).is_none());
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        drop(g1);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        let g2 = InflightGuard::try_acquire(&counter, 1).expect("slot after release");
        drop(g2);
    }

    #[test]
    fn inflight_guard_drop_does_not_underflow_zero_counter() {
        let counter = Arc::new(AtomicUsize::new(0));
        drop(InflightGuard(Arc::clone(&counter)));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn response_declares_oversize_rejects_only_declared_excess() {
        assert!(response_declares_oversize(Some(11), 10));
        assert!(!response_declares_oversize(Some(10), 10));
        assert!(!response_declares_oversize(None, 10));
    }

    #[test]
    fn append_limited_body_chunk_rejects_incremental_excess() {
        let mut body = Vec::new();
        append_limited_body_chunk(&mut body, b"12345", 8).expect("first chunk fits");
        assert!(append_limited_body_chunk(&mut body, b"6789", 8).is_err());
        assert_eq!(body, b"12345");
    }

    #[test]
    fn add_limited_body_len_rejects_incremental_excess() {
        assert_eq!(add_limited_body_len(5, 3, 8).unwrap(), 8);
        assert!(add_limited_body_len(5, 4, 8).is_err());
        assert!(add_limited_body_len(usize::MAX, 1, usize::MAX).is_err());
    }

    #[tokio::test]
    async fn build_response_rejects_oversized_content_length_before_buffering() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = socket.read(&mut request).await.unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                super::MAX_WORKER_RESPONSE_BODY_BYTES + 1
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
        });

        let reqwest_resp = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .body("{}")
            .send()
            .await
            .unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let guard: Guard = InflightGuard::acquire(&counter);

        let response = DirectDispatcher::default()
            .build_response(
                Ok(reqwest_resp),
                format!("http://{addr}"),
                false,
                AttemptGuard {
                    _inflight: guard,
                    _reservation: None,
                },
                std::time::Instant::now(),
            )
            .await;

        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        let body = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("AXS_WORKER_UNAVAILABLE"));
        assert!(!body.contains(&addr.to_string()));

        server.await.unwrap();
    }

    #[tokio::test]
    async fn build_response_preserves_worker_response_headers() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = socket.read(&mut request).await.unwrap();
            let response = concat!(
                "HTTP/1.1 200 OK\r\n",
                "content-type: application/json\r\n",
                "x-request-id: worker-request-123\r\n",
                "x-runtime: ax-engine\r\n",
                "connection: close\r\n",
                "\r\n",
                "{\"ok\":true}"
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
        });

        let reqwest_resp = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .body("{}")
            .send()
            .await
            .unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let guard: Guard = InflightGuard::acquire(&counter);

        let response = DirectDispatcher::default()
            .build_response(
                Ok(reqwest_resp),
                format!("http://{addr}"),
                false,
                AttemptGuard {
                    _inflight: guard,
                    _reservation: None,
                },
                std::time::Instant::now(),
            )
            .await;

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers().get("x-request-id").unwrap(),
            "worker-request-123"
        );
        assert_eq!(response.headers().get("x-runtime").unwrap(), "ax-engine");
        assert!(!response.headers().contains_key("connection"));

        server.await.unwrap();
    }
}
