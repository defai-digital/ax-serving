use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::HeaderValue,
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
}

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

pub fn router(config: &ThorConfig, client: reqwest::Client, inflight: Arc<AtomicUsize>) -> Router {
    let state = ProxyState {
        client,
        sglang_url: config.sglang_url.clone(),
        inflight,
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

async fn proxy_chat(State(state): State<ProxyState>, body: Bytes) -> impl IntoResponse {
    proxy_to(&state, "/v1/chat/completions", body).await
}

async fn proxy_completions(State(state): State<ProxyState>, body: Bytes) -> impl IntoResponse {
    proxy_to(&state, "/v1/completions", body).await
}

async fn proxy_embeddings(State(state): State<ProxyState>, body: Bytes) -> impl IntoResponse {
    proxy_to(&state, "/v1/embeddings", body).await
}

async fn proxy_to(state: &ProxyState, path: &str, body: Bytes) -> axum::response::Response {
    let _guard = InflightGuard::acquire(&state.inflight);
    let url = format!("{}{}", state.sglang_url, path);
    match state
        .client
        .post(url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();
            let is_event_stream = content_type.starts_with("text/event-stream");

            if is_event_stream {
                let byte_stream = resp.bytes_stream();
                let guarded = futures::stream::unfold(
                    (Box::pin(byte_stream), Some(_guard)),
                    |(mut stream, guard)| async move {
                        match stream.next().await {
                            Some(Ok(chunk)) => Some((Ok::<Bytes, axum::Error>(chunk), (stream, guard))),
                            Some(Err(err)) => Some((
                                Err(axum::Error::new(err)),
                                (stream, None),
                            )),
                            None => None,
                        }
                    },
                );

                return axum::response::Response::builder()
                    .status(status)
                    .header(
                        "content-type",
                        HeaderValue::from_str(&content_type)
                            .unwrap_or_else(|_| HeaderValue::from_static("text/event-stream")),
                    )
                    .body(Body::from_stream(guarded))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
            }

            match resp.bytes().await {
                Ok(bytes) => axum::response::Response::builder()
                    .status(status)
                    .header("content-type", content_type)
                    .body(axum::body::Body::from(bytes))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
                Err(err) => (
                    StatusCode::BAD_GATEWAY,
                    format!("failed to read sglang response: {err}"),
                )
                    .into_response(),
            }
        }
        Err(err) => (StatusCode::BAD_GATEWAY, format!("sglang proxy error: {err}")).into_response(),
    }
}
