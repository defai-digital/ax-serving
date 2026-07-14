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

mod attempt;
mod client;
mod headers;
mod metrics;
mod reservation;
mod retry_policy;
mod stream_proxy;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Bytes;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use reqwest::Client;
use tracing::{Instrument as _, debug, info_span, warn};

use self::attempt::{AttemptGuard, InflightGuard};
use self::client::{
    DEFAULT_POOL_MAX_IDLE_PER_HOST, DEFAULT_REQUEST_TIMEOUT_SECS, build_dispatcher_client,
    parse_dispatch_token, worker_url,
};
use self::headers::{attach_dispatch_auth, current_trace_headers};
use self::metrics::{DispatchMetrics, SelectionOutcome};
use self::reservation::{ReservationAcquireError, reserve_attempt};
use self::retry_policy::{is_retryable_connect_failure, is_typed_not_admitted};
use self::stream_proxy::drain_worker_error_response;
use super::deployment::DeploymentCatalog;
use super::error::ax_error_response;
use super::fleet_state::FleetStateStore;
use super::policy::{DispatchContext, DispatchPolicy};
use super::registry::{RequestKind, WorkerId, WorkerRegistry};
use super::request_profile::{RequestProfile, rewrite_runtime_model};

pub use self::metrics::{
    DispatchMetricsSnapshot, GATEWAY_LATENCY_BUCKETS_US, LatencyHistogramSnapshot,
};

// ── DirectDispatcher ──────────────────────────────────────────────────────────

/// HTTP proxy dispatcher for direct (no-broker) mode.
///
/// Holds a shared `reqwest::Client` (connection-pool enabled).
/// Stateless — all per-request state comes from `WorkerRegistry` and the policy.
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
        let client = build_dispatcher_client(pool_max_idle_per_host, request_timeout_secs);
        let dispatch_token = parse_dispatch_token(dispatch_token)?;

        Ok(Self {
            client,
            reroute_total: Arc::new(AtomicU64::new(0)),
            metrics: Arc::new(DispatchMetrics::default()),
            dispatch_token,
            first_byte_timeout: std::time::Duration::from_millis(first_byte_timeout_ms.max(1)),
            stream_idle_timeout: std::time::Duration::from_millis(stream_idle_timeout_ms.max(1)),
            fleet_store: None,
            reservation_ttl_ms: 15_000,
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
            reservation_renew_tasks: self.metrics.reservation_renew_tasks.load(Ordering::Relaxed),
            reservation_renew_ok_total: self
                .metrics
                .reservation_renew_ok_total
                .load(Ordering::Relaxed),
            reservation_renew_fenced_total: self
                .metrics
                .reservation_renew_fenced_total
                .load(Ordering::Relaxed),
            reservation_renew_error_total: self
                .metrics
                .reservation_renew_error_total
                .load(Ordering::Relaxed),
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
        // At most one safe retry (attempts clamped to 1..=2).
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
            let reservation = match reserve_attempt(
                self.fleet_store.as_ref(),
                &self.metrics,
                self.reservation_ttl_ms,
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
            attempt_number = attempt_number.saturating_add(1);
            let url = worker_url(&candidate.endpoint.worker.addr, path);
            debug!(
                request_id = %profile.request_id,
                %attempt_id,
                attempt_number,
                worker_id = %selected_id,
                deployment_id = %candidate.deployment.id,
                pool_id = %candidate.pool.id,
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
                axs.runtime.kind = %candidate.endpoint.runtime_kind,
                http.response.status_code = tracing::field::Empty,
                otel.status_code = tracing::field::Empty,
            );
            let request =
                attach_dispatch_auth(self.client.post(&url), self.dispatch_token.as_ref())
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
                .is_some_and(is_retryable_connect_failure);
            let retryable_not_admitted = result.as_ref().ok().is_some_and(is_typed_not_admitted);
            // Safe retry only before commitment: connect fail or typed pre-admission.
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
        let reservation = match reserve_attempt(
            self.fleet_store.as_ref(),
            &self.metrics,
            self.reservation_ttl_ms,
            reservation_worker_id,
            attempt_id,
            selected.max_inflight,
        )
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

        let mut request =
            attach_dispatch_auth(self.client.post(&url), self.dispatch_token.as_ref())
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
            .is_some_and(is_retryable_connect_failure);
        let retryable_not_admitted = result.as_ref().ok().is_some_and(is_typed_not_admitted);

        // At most one safe retry: connect fail or typed not-admitted only.
        // Never retry after commitment (headers/stream already building below).
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
        let reservation = match reserve_attempt(
            self.fleet_store.as_ref(),
            &self.metrics,
            self.reservation_ttl_ms,
            reservation_worker_id,
            attempt_id,
            selected2.max_inflight,
        )
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

        let mut request =
            attach_dispatch_auth(self.client.post(&url2), self.dispatch_token.as_ref())
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
}

impl Default for DirectDispatcher {
    fn default() -> Self {
        Self::new(DEFAULT_POOL_MAX_IDLE_PER_HOST, DEFAULT_REQUEST_TIMEOUT_SECS)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn request_hash(request_id: ax_serving_protocol::RequestId) -> u64 {
    let bytes = request_id.as_uuid().into_bytes();
    u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("UUID has at least eight bytes"),
    )
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

    use super::metrics::AtomicLatencyHistogram;
    use super::stream_proxy::{
        MAX_WORKER_RESPONSE_BODY_BYTES, add_limited_body_len, append_limited_body_chunk,
        response_declares_oversize,
    };
    use super::{
        AttemptGuard, DirectDispatcher, GATEWAY_LATENCY_BUCKETS_US, InflightGuard,
        InflightGuard as Guard, worker_url,
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
        let addr =
            crate::orchestration::worker_endpoint::WorkerEndpoint::parse("127.0.0.1:8081").unwrap();
        assert_eq!(
            worker_url(&addr, "/v1/chat/completions"),
            "http://127.0.0.1:8081/v1/chat/completions"
        );
    }

    #[test]
    fn worker_url_without_leading_slash_adds_root() {
        let addr =
            crate::orchestration::worker_endpoint::WorkerEndpoint::parse("127.0.0.1:8081").unwrap();
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
                MAX_WORKER_RESPONSE_BODY_BYTES + 1
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
