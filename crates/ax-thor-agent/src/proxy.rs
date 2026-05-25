use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, HeaderName, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use futures::StreamExt as _;

use crate::config::ThorConfig;

const MAX_PROXY_REQUEST_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_PROXY_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct ProxyState {
    pub client: reqwest::Client,
    pub runtime_url: String,
    pub inflight: Arc<AtomicUsize>,
    pub max_inflight: usize,
}

struct InflightGuard(Arc<AtomicUsize>);

#[derive(Debug)]
enum ProxyBodyError {
    TooLarge,
    Read(reqwest::Error),
}

impl InflightGuard {
    /// Try to acquire an inflight slot. Returns `None` if at capacity.
    fn try_acquire(counter: &Arc<AtomicUsize>, max: usize) -> Option<Self> {
        let mut spins = 0usize;
        loop {
            let current = counter.load(Ordering::Acquire);
            if current >= max {
                return None;
            }
            if counter
                .compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(Self(Arc::clone(counter)));
            }
            spins += 1;
            std::hint::spin_loop();
            if spins.is_multiple_of(16) {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

fn response_declares_oversize(content_length: Option<u64>, max_bytes: usize) -> bool {
    content_length.is_some_and(|len| len > max_bytes as u64)
}

fn append_limited_body_chunk(
    body: &mut Vec<u8>,
    chunk: &[u8],
    max_bytes: usize,
) -> Result<(), ProxyBodyError> {
    let next_len = body
        .len()
        .checked_add(chunk.len())
        .ok_or(ProxyBodyError::TooLarge)?;
    if next_len > max_bytes {
        return Err(ProxyBodyError::TooLarge);
    }
    body.extend_from_slice(chunk);
    Ok(())
}

fn add_limited_body_len(
    current_len: usize,
    chunk_len: usize,
    max_bytes: usize,
) -> Result<usize, ProxyBodyError> {
    let next_len = current_len
        .checked_add(chunk_len)
        .ok_or(ProxyBodyError::TooLarge)?;
    if next_len > max_bytes {
        return Err(ProxyBodyError::TooLarge);
    }
    Ok(next_len)
}

async fn read_response_body_limited(
    resp: reqwest::Response,
    max_bytes: usize,
) -> Result<Bytes, ProxyBodyError> {
    if response_declares_oversize(resp.content_length(), max_bytes) {
        return Err(ProxyBodyError::TooLarge);
    }

    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        append_limited_body_chunk(&mut body, &chunk.map_err(ProxyBodyError::Read)?, max_bytes)?;
    }

    Ok(Bytes::from(body))
}

fn should_forward_runtime_header(name: &HeaderName, include_content_length: bool) -> bool {
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
    ) && (include_content_length || name != header::CONTENT_LENGTH.as_str())
}

fn response_builder_with_runtime_headers(
    status: StatusCode,
    headers: &HeaderMap,
    include_content_length: bool,
) -> axum::http::response::Builder {
    let mut builder = axum::response::Response::builder().status(status);
    for (name, value) in headers {
        if should_forward_runtime_header(name, include_content_length) {
            builder = builder.header(name, value);
        }
    }
    if !headers.contains_key(header::CONTENT_TYPE) {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }
    builder
}

pub fn router(config: &ThorConfig, client: reqwest::Client, inflight: Arc<AtomicUsize>) -> Router {
    let state = ProxyState {
        client,
        runtime_url: config.runtime_url.clone(),
        inflight,
        max_inflight: config.max_inflight,
    };

    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(proxy_chat))
        .route("/v1/completions", post(proxy_completions))
        .route("/v1/embeddings", post(proxy_embeddings))
        .layer(DefaultBodyLimit::max(MAX_PROXY_REQUEST_BODY_BYTES))
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

/// Headers forwarded from the client request to the runtime endpoint.
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
    let url = format!("{}{}", state.runtime_url, path);
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
                    (Box::pin(byte_stream), Some(_guard), 0usize, false),
                    |(mut stream, guard, total_len, done)| async move {
                        if done {
                            drop(guard);
                            return None;
                        }
                        match stream.next().await {
                            Some(Ok(chunk)) => {
                                match add_limited_body_len(
                                    total_len,
                                    chunk.len(),
                                    MAX_PROXY_RESPONSE_BODY_BYTES,
                                ) {
                                    Ok(next_len) => {
                                        Some((Ok(chunk), (stream, guard, next_len, false)))
                                    }
                                    Err(ProxyBodyError::TooLarge) => {
                                        drop(guard);
                                        Some((
                                            Err(axum::Error::new(std::io::Error::new(
                                                std::io::ErrorKind::InvalidData,
                                                "upstream streaming response body exceeded 64 MiB limit",
                                            ))),
                                            (stream, None, total_len, true),
                                        ))
                                    }
                                    Err(ProxyBodyError::Read(_)) => {
                                        unreachable!("length accounting does not read")
                                    }
                                }
                            }
                            Some(Err(err)) => {
                                let mapped = Err(axum::Error::new(err));
                                Some((mapped, (stream, guard, total_len, false)))
                            }
                            None => {
                                drop(guard);
                                None
                            }
                        }
                    },
                );

                return response_builder_with_runtime_headers(status, &resp_headers, false)
                    .body(Body::from_stream(guarded))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
            }

            match read_response_body_limited(resp, MAX_PROXY_RESPONSE_BODY_BYTES).await {
                Err(ProxyBodyError::TooLarge) => (
                    StatusCode::BAD_GATEWAY,
                    "upstream response body exceeded 64 MiB limit",
                )
                    .into_response(),
                Ok(bytes) => response_builder_with_runtime_headers(status, &resp_headers, true)
                    .body(axum::body::Body::from(bytes))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
                Err(ProxyBodyError::Read(err)) => (
                    StatusCode::BAD_GATEWAY,
                    format!("failed to read runtime response: {err}"),
                )
                    .into_response(),
            }
        }
        Err(err) => (
            StatusCode::BAD_GATEWAY,
            format!("runtime proxy error: {err}"),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use axum::http::{HeaderMap, HeaderValue};

    use super::{
        InflightGuard, add_limited_body_len, append_limited_body_chunk,
        response_builder_with_runtime_headers, response_declares_oversize,
    };

    #[test]
    fn inflight_guard_stops_at_capacity() {
        let counter = Arc::new(AtomicUsize::new(0));
        let g1 = InflightGuard::try_acquire(&counter, 2);
        let g2 = InflightGuard::try_acquire(&counter, 2);
        let g3 = InflightGuard::try_acquire(&counter, 2);

        assert!(g1.is_some());
        assert!(g2.is_some());
        assert!(g3.is_none());
    }

    #[test]
    fn inflight_guard_releases_slot_on_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let g1 = InflightGuard::try_acquire(&counter, 1).expect("first permit");
        assert!(InflightGuard::try_acquire(&counter, 1).is_none());
        drop(g1);
        assert!(InflightGuard::try_acquire(&counter, 1).is_some());
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

    #[test]
    fn response_builder_strips_hop_by_hop_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
        headers.insert("upgrade", HeaderValue::from_static("websocket"));
        headers.insert("x-runtime", HeaderValue::from_static("vllm"));

        let response =
            response_builder_with_runtime_headers(axum::http::StatusCode::OK, &headers, false)
                .body(axum::body::Body::empty())
                .expect("response should build");

        assert_eq!(response.headers().get("x-runtime").unwrap(), "vllm");
        assert!(!response.headers().contains_key("connection"));
        assert!(!response.headers().contains_key("transfer-encoding"));
        assert!(!response.headers().contains_key("upgrade"));
    }

    #[test]
    fn response_builder_omits_content_length_for_streaming_bodies() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("content-length", HeaderValue::from_static("999"));

        let streaming =
            response_builder_with_runtime_headers(axum::http::StatusCode::OK, &headers, false)
                .body(axum::body::Body::empty())
                .expect("streaming response should build");
        assert!(!streaming.headers().contains_key("content-length"));

        let buffered =
            response_builder_with_runtime_headers(axum::http::StatusCode::OK, &headers, true)
                .body(axum::body::Body::empty())
                .expect("buffered response should build");
        assert_eq!(buffered.headers().get("content-length").unwrap(), "999");
    }
}
