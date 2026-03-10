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
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use reqwest::Client;
use tracing::{error, warn};

use super::policy::{DispatchContext, DispatchPolicy};
use super::registry::{WorkerId, WorkerRegistry};

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

impl DirectDispatcher {
    pub fn new(pool_max_idle_per_host: usize, request_timeout_secs: u64) -> Self {
        Self {
            client: Client::builder()
                .pool_max_idle_per_host(pool_max_idle_per_host)
                .connect_timeout(std::time::Duration::from_secs(5))
                .timeout(std::time::Duration::from_secs(request_timeout_secs))
                .build()
                .expect("failed to build reqwest client"),
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
        path: &str,
        body: Bytes,
        auth_header: Option<&HeaderValue>,
    ) -> Response {
        let workers = registry.eligible_workers(model_id);
        if workers.is_empty() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("no eligible workers for model '{model_id}'"),
            )
                .into_response();
        }

        let ctx = DispatchContext { model_id, stream };
        let selected = match policy.select(&workers, &ctx) {
            Some(w) => w,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("all workers for '{model_id}' are at capacity"),
                )
                    .into_response();
            }
        };

        let selected_id = selected.id;
        let url = worker_url(selected.addr, path);
        let guard = InflightGuard::acquire(&selected.inflight_counter);
        // Only pre-clone the body when reroute is possible.
        let can_reroute = workers.iter().any(|w| w.id != selected_id);
        let retry_body = if can_reroute {
            Some(body.clone())
        } else {
            None
        };

        let mut req_builder = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .body(body);
        if let Some(auth) = auth_header
            && let Ok(v) = auth.to_str()
        {
            req_builder = req_builder.header("authorization", v);
        }
        let result = req_builder.send().await;

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
            drop(guard);
            registry.mark_unhealthy(selected_id);
            self.reroute_total.fetch_add(1, Ordering::Relaxed);

            let Some(retry_body) = retry_body else {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no alternative worker for '{}' after reroute", ctx.model_id),
                )
                    .into_response();
            };

            return self
                .reroute(
                    registry,
                    policy,
                    &ctx,
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
        self.build_response(result, url, stream, guard).await
    }

    /// Try once more with a different worker (excluding `excluded_id`).
    #[allow(clippy::too_many_arguments)]
    async fn reroute(
        &self,
        registry: &WorkerRegistry,
        policy: &dyn DispatchPolicy,
        ctx: &DispatchContext<'_>,
        path: &str,
        body: Bytes,
        excluded_id: WorkerId,
        auth_header: Option<&HeaderValue>,
    ) -> Response {
        let workers2 = registry.eligible_workers(ctx.model_id);
        let candidates: Vec<_> = workers2
            .into_iter()
            .filter(|w| w.id != excluded_id)
            .collect();

        let selected2 = match policy.select(&candidates, ctx) {
            Some(w) => w,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no alternative worker for '{}' after reroute", ctx.model_id),
                )
                    .into_response();
            }
        };

        let selected2_id = selected2.id;
        let url2 = worker_url(selected2.addr, path);
        let guard2 = InflightGuard::acquire(&selected2.inflight_counter);

        let mut req_builder2 = self
            .client
            .post(&url2)
            .header("content-type", "application/json")
            .body(body);
        if let Some(auth) = auth_header
            && let Ok(v) = auth.to_str()
        {
            req_builder2 = req_builder2.header("authorization", v);
        }
        let result2 = req_builder2.send().await;

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
            registry.mark_unhealthy(selected2_id);
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "all workers failed for this request",
            )
                .into_response();
        }

        // Record affinity only on 2xx — not on 4xx, to avoid biasing future
        // dispatch towards workers that returned client errors.
        if matches!(&result2, Ok(r) if r.status().is_success()) {
            policy.record_dispatch(selected2_id, ctx.model_id);
        }

        self.build_response(result2, url2, ctx.stream, guard2).await
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
                (StatusCode::BAD_GATEWAY, e.to_string()).into_response()
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
                    let byte_stream = resp.bytes_stream();
                    let guarded = futures::stream::unfold(
                        (Box::pin(byte_stream), Some(guard)),
                        |(mut inner, guard)| async move {
                            match inner.next().await {
                                Some(item) => {
                                    let mapped = item.map_err(std::io::Error::other);
                                    Some((mapped, (inner, guard)))
                                }
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
                            (StatusCode::BAD_GATEWAY, e.to_string()).into_response()
                        }
                    }
                }
            }
        }
    }
}

impl Default for DirectDispatcher {
    fn default() -> Self {
        Self::new(8, 300)
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
        assert_eq!(counter.load(Ordering::Relaxed), 1, "must increment on acquire");

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
