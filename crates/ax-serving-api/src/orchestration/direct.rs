//! Direct dispatcher: HTTP reverse proxy from orchestrator to worker.
//!
//! After the dispatch policy selects a worker, this module:
//! 1. Increments `inflight` atomically on the worker's shared counter.
//! 2. Forwards the full request body to `http://{worker_addr}{path}`.
//! 3. Streams or buffers the response back to the client.
//! 4. Decrements `inflight` via RAII guard on completion or error.
//!
//! On network error or 5xx from the selected worker the dispatcher
//! marks it unhealthy and retries once with the next eligible worker
//! (TASK-MW-008 reroute logic).
//!
//! Streaming responses (SSE, `text/event-stream`) are forwarded chunk-by-chunk
//! without buffering so time-to-first-token is not impacted.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use axum::body::{Body, Bytes};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, TryStreamExt as _};
use reqwest::Client;
use tracing::{error, warn};

use super::policy::{DispatchContext, DispatchPolicy};
use super::registry::{RequestKind, WorkerId, WorkerRegistry};

// ── InflightGuard ─────────────────────────────────────────────────────────────

/// RAII guard: increments a counter on creation, decrements on drop.
struct InflightGuard(Arc<AtomicUsize>);

impl InflightGuard {
    fn acquire(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(Arc::clone(counter))
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

// ── DirectDispatcher ──────────────────────────────────────────────────────────

/// HTTP proxy dispatcher for direct (no-broker) mode.
///
/// Holds a shared `reqwest::Client` (connection-pool enabled).
/// Stateless — all per-request state comes from `WorkerRegistry` and the policy.
#[derive(Clone)]
pub struct DirectDispatcher {
    client: Client,
    reroute_total: Arc<AtomicU64>,
}

/// Attach an `Authorization` header to a request builder when one is provided.
fn attach_auth(
    builder: reqwest::RequestBuilder,
    auth: Option<&HeaderValue>,
) -> reqwest::RequestBuilder {
    match auth.and_then(|v| v.to_str().ok()) {
        Some(v) => builder.header("authorization", v),
        None => builder,
    }
}

/// TCP connect timeout for the dispatcher's reqwest client.
/// Short enough to fail fast on unreachable workers without blocking the queue.
const DISPATCHER_CONNECT_TIMEOUT_SECS: u64 = 5;
/// Default pool size and request timeout matching serving.example.yaml defaults.
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 8;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

impl DirectDispatcher {
    pub fn new(pool_max_idle_per_host: usize, request_timeout_secs: u64) -> Self {
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

        Self {
            client,
            reroute_total: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Total number of reroutes performed since startup.
    pub fn reroutes(&self) -> u64 {
        self.reroute_total.load(Ordering::Relaxed)
    }

    /// Forward a request to the selected worker.
    ///
    /// `auth_header` is the value of the client's `Authorization` header (if
    /// present).  It is forwarded verbatim to the worker so that deployments
    /// where workers also require bearer-token auth continue to work when
    /// `AXS_API_KEY` is set on both the orchestrator and the workers.
    ///
    /// On network error or 5xx the worker is marked unhealthy and the request
    /// is retried once with the next eligible worker.
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
        auth_header: Option<&HeaderValue>,
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
            path,
            body,
            auth_header,
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
        path: &str,
        body: Bytes,
        auth_header: Option<&HeaderValue>,
    ) -> Response {
        let workers = registry.dispatch_workers_filtered(
            model_id,
            request_kind,
            backend_hint,
            min_context,
            preferred_pool,
            None,
        );
        if workers.is_empty() {
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
        };
        let selected = match policy.select(&workers, &ctx) {
            Some(w) => w,
            None => {
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

        let selected_id = selected.id;
        let url = worker_url(selected.addr, path);
        // Only pre-clone the body when reroute is possible.
        let can_reroute = candidate_count > 1;
        let retry_body = if can_reroute {
            Some(body.clone())
        } else {
            None
        };
        let Some(inflight_counter) = registry.inflight_counter(selected_id) else {
            warn!(
                worker_id = %selected_id,
                "selected worker disappeared before dispatch"
            );
            let Some(retry_body) = retry_body else {
                return trace_response(
                    worker_failure_response(format!(
                        "selected worker unavailable for '{}'",
                        ctx.model_id
                    )),
                    candidate_count,
                    Some(selected_id),
                    "selected_worker_unavailable",
                );
            };
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
                    auth_header,
                )
                .await;
        };
        let guard = InflightGuard::acquire(&inflight_counter);

        let result = attach_auth(
            self.client
                .post(&url)
                .header("content-type", "application/json")
                .body(body),
            auth_header,
        )
        .send()
        .await;

        // Reroute on network error or 5xx.
        let is_err = result.is_err();
        let is_5xx = matches!(&result, Ok(r) if r.status().is_server_error());

        if is_err || is_5xx {
            match &result {
                Err(e) => warn!(%url, err = %e, "dispatch failed, rerouting"),
                Ok(r) => {
                    warn!(%url, status = r.status().as_u16(), "worker returned 5xx, rerouting")
                }
            }
            // Drain the error response body so the connection can be reused
            // from the pool instead of being discarded.
            if let Ok(resp) = result {
                let _ = resp.bytes().await;
            }
            drop(guard);
            registry.mark_unhealthy(selected_id);
            self.reroute_total.fetch_add(1, Ordering::Relaxed);

            let Some(retry_body) = retry_body else {
                return trace_response(
                    worker_failure_response(format!(
                        "no alternative worker for '{}' after reroute",
                        ctx.model_id
                    )),
                    candidate_count,
                    Some(selected_id),
                    "reroute_exhausted",
                );
            };

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
                    auth_header,
                )
                .await;
        }

        // Record affinity only on 2xx — not on 4xx, to avoid biasing future
        // dispatch towards workers that returned client errors.
        if matches!(&result, Ok(r) if r.status().is_success()) {
            policy.record_dispatch(selected_id, model_id);
        }
        trace_response(
            self.build_response(result, url, stream, guard).await,
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
        auth_header: Option<&HeaderValue>,
    ) -> Response {
        let candidates = registry.dispatch_workers_filtered(
            ctx.model_id,
            request_kind,
            backend_hint,
            min_context,
            ctx.preferred_pool,
            Some(excluded_id),
        );

        let selected2 = match policy.select(&candidates, ctx) {
            Some(w) => w,
            None => {
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

        let selected2_id = selected2.id;
        let url2 = worker_url(selected2.addr, path);
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
        let guard2 = InflightGuard::acquire(&inflight_counter2);

        let result2 = attach_auth(
            self.client
                .post(&url2)
                .header("content-type", "application/json")
                .body(body),
            auth_header,
        )
        .send()
        .await;

        // If the reroute also failed, return 503 rather than passing the worker's
        // internal error status through — the orchestrator owns the failure signal.
        let is_err2 = result2.is_err();
        let is_5xx2 = matches!(&result2, Ok(r) if r.status().is_server_error());
        if is_err2 || is_5xx2 {
            match &result2 {
                Err(e) => warn!(%url2, err = %e, "reroute also failed"),
                Ok(r) => {
                    warn!(%url2, status = r.status().as_u16(), "reroute worker also returned 5xx")
                }
            }
            // Drain the error response body so the connection can be reused.
            if let Ok(resp) = result2 {
                let _ = resp.bytes().await;
            }
            registry.mark_unhealthy(selected2_id);
            return trace_response(
                worker_failure_response("all workers failed for this request"),
                candidates.len(),
                Some(selected2_id),
                "reroute_failed",
            );
        }

        // Record affinity only on 2xx — not on 4xx, to avoid biasing future
        // dispatch towards workers that returned client errors.
        if matches!(&result2, Ok(r) if r.status().is_success()) {
            policy.record_dispatch(selected2_id, ctx.model_id);
        }

        trace_response(
            self.build_response(result2, url2, ctx.stream, guard2).await,
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
        guard: InflightGuard,
    ) -> Response {
        match result {
            Err(e) => {
                warn!(%url, err = %e, "dispatch request failed");
                worker_failure_response(e.to_string())
            }
            Ok(resp) => {
                let status = StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                let content_type = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/json")
                    .to_string();

                if stream {
                    type GuardedResponseStream =
                        std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>;
                    let byte_stream: GuardedResponseStream =
                        Box::pin(resp.bytes_stream().map_err(std::io::Error::other));
                    let guarded = futures::stream::unfold(
                        (byte_stream, Some(guard)),
                        |(mut inner, guard): (GuardedResponseStream, Option<InflightGuard>)| async move {
                            match inner.next().await {
                                Some(item) => Some((item, (inner, guard))),
                                None => {
                                    drop(guard);
                                    None
                                }
                            }
                        },
                    );

                    axum::response::Response::builder()
                        .status(status)
                        .header("content-type", content_type)
                        .body(Body::from_stream(guarded))
                        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                } else {
                    match resp.bytes().await {
                        Ok(bytes) => axum::response::Response::builder()
                            .status(status)
                            .header("content-type", content_type)
                            .body(Body::from(bytes))
                            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
                        Err(e) => {
                            error!(%url, err = %e, "reading worker response body failed");
                            worker_failure_response(e.to_string())
                        }
                    }
                }
            }
        }
    }
}

impl Default for DirectDispatcher {
    fn default() -> Self {
        Self::new(DEFAULT_POOL_MAX_IDLE_PER_HOST, DEFAULT_REQUEST_TIMEOUT_SECS)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn worker_url(addr: std::net::SocketAddr, path: &str) -> String {
    if path.starts_with('/') {
        format!("http://{addr}{path}")
    } else {
        // Ensure the path is always rooted — callers should always pass a
        // leading slash, but guard against that to avoid silently routing to "/".
        format!("http://{addr}/{path}")
    }
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

fn worker_failure_response(message: impl Into<String>) -> Response {
    let mut resp = (StatusCode::SERVICE_UNAVAILABLE, message.into()).into_response();
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

    use super::{InflightGuard, worker_url};

    #[test]
    fn worker_url_with_leading_slash() {
        let addr: std::net::SocketAddr = "127.0.0.1:8081".parse().unwrap();
        assert_eq!(
            worker_url(addr, "/v1/chat/completions"),
            "http://127.0.0.1:8081/v1/chat/completions"
        );
    }

    #[test]
    fn worker_url_without_leading_slash_adds_root() {
        let addr: std::net::SocketAddr = "127.0.0.1:8081".parse().unwrap();
        assert_eq!(
            worker_url(addr, "v1/completions"),
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
}
