use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use futures::StreamExt as _;

use crate::config::ThorConfig;

#[derive(Clone)]
pub struct ProxyState {
    pub client: reqwest::Client,
    pub sglang_url: String,
    pub inflight: Arc<AtomicUsize>,
    pub max_inflight: usize,
}

struct InflightGuard(Arc<AtomicUsize>);

impl InflightGuard {
    /// Try to acquire an inflight slot. Returns `None` if at capacity.
    fn try_acquire(counter: &Arc<AtomicUsize>, max: usize) -> Option<Self> {
        loop {
            let current = counter.load(Ordering::Relaxed);
            if current >= max {
                return None;
            }
            if counter
                .compare_exchange_weak(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Some(Self(Arc::clone(counter)));
            }
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn router(config: &ThorConfig, client: reqwest::Client, inflight: Arc<AtomicUsize>) -> Router {
    let state = ProxyState {
        client,
        sglang_url: config.sglang_url.clone(),
        inflight,
        max_inflight: config.max_inflight,
    };

    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(proxy_chat))
        .route("/v1/completions", post(proxy_completions))
        .route("/v1/embeddings", post(proxy_embeddings))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({ "status": "ok" }))
}

async fn proxy_chat(
    State(state): State<ProxyState>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_to(&state, "/v1/chat/completions", &headers, body).await
}

async fn proxy_completions(
    State(state): State<ProxyState>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_to(&state, "/v1/completions", &headers, body).await
}

async fn proxy_embeddings(
    State(state): State<ProxyState>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_to(&state, "/v1/embeddings", &headers, body).await
}

/// Headers forwarded from the client request to sglang.
const FORWARDED_HEADERS: &[&str] = &["authorization", "x-request-id", "content-type"];

async fn proxy_to(
    state: &ProxyState,
    path: &str,
    client_headers: &axum::http::HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let Some(_guard) = InflightGuard::try_acquire(&state.inflight, state.max_inflight) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "max inflight requests reached",
        )
            .into_response();
    };
    let url = format!("{}{}", state.sglang_url, path);
    let mut req = state.client.post(url);
    // Forward client headers first; fall back to application/json for content-type.
    let mut has_content_type = false;
    for &name in FORWARDED_HEADERS {
        if let Some(val) = client_headers.get(name) {
            req = req.header(name, val.clone());
            if name == "content-type" {
                has_content_type = true;
            }
        }
    }
    if !has_content_type {
        req = req.header("content-type", "application/json");
    }
    match req.body(body).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let resp_headers = resp.headers().clone();
            let content_type = resp_headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();
            let is_event_stream = content_type
                .to_ascii_lowercase()
                .starts_with("text/event-stream");

            if is_event_stream {
                let byte_stream = resp.bytes_stream();
                let guarded = futures::stream::unfold(
                    (Box::pin(byte_stream), Some(_guard)),
                    |(mut stream, guard)| async move {
                        match stream.next().await {
                            Some(item) => {
                                let mapped = item.map_err(axum::Error::new);
                                Some((mapped, (stream, guard)))
                            }
                            None => {
                                drop(guard);
                                None
                            }
                        }
                    },
                );

                let mut builder = axum::response::Response::builder().status(status);
                for (name, value) in &resp_headers {
                    builder = builder.header(name, value);
                }
                return builder
                    .body(Body::from_stream(guarded))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
            }

            // Limit buffered response bodies to 64 MiB to prevent OOM from a
            // misbehaving upstream (BUG-033).
            const MAX_RESPONSE_BODY: usize = 64 * 1024 * 1024;
            match resp.bytes().await {
                Ok(bytes) if bytes.len() > MAX_RESPONSE_BODY => (
                    StatusCode::BAD_GATEWAY,
                    "upstream response body exceeded 64 MiB limit",
                )
                    .into_response(),
                Ok(bytes) => {
                    let mut builder = axum::response::Response::builder().status(status);
                    for (name, value) in &resp_headers {
                        builder = builder.header(name, value);
                    }
                    builder
                        .body(axum::body::Body::from(bytes))
                        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                }
                Err(err) => (
                    StatusCode::BAD_GATEWAY,
                    format!("failed to read sglang response: {err}"),
                )
                    .into_response(),
            }
        }
        Err(err) => (
            StatusCode::BAD_GATEWAY,
            format!("sglang proxy error: {err}"),
        )
            .into_response(),
    }
}
