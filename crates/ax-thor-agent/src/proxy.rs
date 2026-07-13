use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::StreamExt as _;
use opentelemetry::propagation::{Extractor, Injector};
use tracing::Instrument as _;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use ax_serving_protocol::{
    ADMISSION_STATE_HEADER, ATTEMPT_ID_HEADER, AttemptId, DISPATCH_TOKEN_HEADER, REQUEST_ID_HEADER,
    RequestId,
};

use crate::config::ThorConfig;

const MAX_PROXY_REQUEST_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_PROXY_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct ProxyState {
    pub client: reqwest::Client,
    pub runtime_url: String,
    pub inflight: Arc<AtomicUsize>,
    pub max_inflight: usize,
    pub draining: Arc<AtomicBool>,
}

#[derive(Clone)]
struct DispatchAuthState {
    token: Option<Arc<str>>,
}

struct HeaderExtractor<'a>(&'a HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(HeaderName::as_str).collect()
    }
}

struct HeaderInjector(HeaderMap);

impl Injector for HeaderInjector {
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
    let mut injector = HeaderInjector(HeaderMap::new());
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut injector);
    });
    injector.0
}

struct InflightGuard(Arc<AtomicUsize>);

#[derive(Debug)]
enum ProxyBodyError {
    TooLarge,
    Read(reqwest::Error),
}

impl InflightGuard {
    /// Try to acquire an inflight slot. Returns `None` if at capacity.
    ///
    /// Uses `fetch_update` (a bounded CAS retry that makes progress on every
    /// iteration) rather than a spin-loop. The previous implementation called
    /// `std::hint::spin_loop()` / `std::thread::yield_now()` on contention,
    /// which yields to the OS scheduler — not the tokio runtime — and starves
    /// every other async task on the same worker thread (BUG-152).
    fn try_acquire(counter: &Arc<AtomicUsize>, max: usize) -> Option<Self> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < max).then_some(current + 1)
            })
            .ok()
            .map(|_| Self(Arc::clone(counter)))
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

fn sanitize_runtime_error_body(bytes: &[u8]) -> Bytes {
    let parsed = serde_json::from_slice::<serde_json::Value>(bytes).ok();
    let error = parsed.as_ref().and_then(|value| value.get("error"));
    let message = error
        .and_then(|value| {
            value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .or_else(|| value.as_str())
        })
        .map(|value| bounded_safe_text(value, 512))
        .unwrap_or_else(|| "runtime rejected the request".into());
    let code = error
        .and_then(|value| value.get("code"))
        .and_then(serde_json::Value::as_str)
        .filter(|value| safe_error_token(value))
        .unwrap_or("AXS_RUNTIME_ERROR");
    let error_type = error
        .and_then(|value| value.get("type"))
        .and_then(serde_json::Value::as_str)
        .filter(|value| safe_error_token(value))
        .unwrap_or("server_error");
    Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": code,
            }
        }))
        .unwrap_or_else(|_| {
            br#"{"error":{"message":"runtime rejected the request","type":"server_error","param":null,"code":"AXS_RUNTIME_ERROR"}}"#.to_vec()
        }),
    )
}

fn bounded_safe_text(value: &str, limit: usize) -> String {
    let mut characters = value.chars().map(|character| {
        if character.is_control() {
            ' '
        } else {
            character
        }
    });
    let prefix = characters.by_ref().take(limit).collect::<String>();
    if characters.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

fn safe_error_token(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
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
            | "set-cookie"
            | "www-authenticate"
            | "x-ax-admission-state"
            | "x-ax-attempt-id"
            | "x-ax-dispatch-token"
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

pub fn router(
    config: &ThorConfig,
    client: reqwest::Client,
    inflight: Arc<AtomicUsize>,
    draining: Arc<AtomicBool>,
) -> Router {
    let state = ProxyState {
        client,
        runtime_url: config.runtime_url.clone(),
        inflight,
        max_inflight: config.max_inflight,
        draining,
    };

    let dispatch_auth = DispatchAuthState {
        token: config.dispatch_token.clone().map(Arc::from),
    };
    let inference = Router::new()
        .route("/v1/chat/completions", post(proxy_chat))
        .route("/v1/completions", post(proxy_completions))
        .route("/v1/embeddings", post(proxy_embeddings))
        .route_layer(middleware::from_fn_with_state(
            dispatch_auth,
            dispatch_auth_middleware,
        ))
        .route_layer(middleware::from_fn(trace_agent_request_middleware));

    Router::new()
        .route("/livez", get(livez))
        .route("/health", get(health))
        .merge(inference)
        .layer(DefaultBodyLimit::max(MAX_PROXY_REQUEST_BODY_BYTES))
        .with_state(state)
}

async fn trace_agent_request_middleware(request: Request, next: Next) -> Response {
    let path = request.uri().path().to_string();
    let request_id = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let attempt_id = request
        .headers()
        .get(ATTEMPT_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let parent = opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(request.headers()))
    });
    let span = tracing::info_span!(
        "axs.agent.dispatch",
        otel.kind = "server",
        url.path = %path,
        axs.request.id = %request_id,
        axs.attempt.id = %attempt_id,
        http.response.status_code = tracing::field::Empty,
    );
    let _ = span.set_parent(parent);
    let response = next.run(request).instrument(span.clone()).await;
    span.record("http.response.status_code", response.status().as_u16());
    response
}

async fn dispatch_auth_middleware(
    State(state): State<DispatchAuthState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.token.as_deref() else {
        return next.run(request).await;
    };
    let authorized = request
        .headers()
        .get(DISPATCH_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|candidate| constant_time_eq(candidate, expected));
    if !authorized {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({
                "error": {
                    "code": "AXS_DISPATCH_UNAUTHORIZED",
                    "message": "invalid gateway dispatch credential"
                }
            })),
        )
            .into_response();
    }
    let ids_are_valid = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<RequestId>().ok())
        .is_some()
        && request
            .headers()
            .get(ATTEMPT_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<AttemptId>().ok())
            .is_some();
    if !ids_are_valid {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": {
                    "code": "AXS_DISPATCH_IDENTITY_INVALID",
                    "message": "valid request and attempt IDs are required"
                }
            })),
        )
            .into_response();
    }
    next.run(request).await
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut difference = left.len() ^ right.len();
    let length = left.len().max(right.len());
    for index in 0..length {
        let left = left.get(index).copied().unwrap_or_default();
        let right = right.get(index).copied().unwrap_or_default();
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

async fn livez() -> impl IntoResponse {
    axum::Json(serde_json::json!({ "agent_live": true }))
}

async fn health(State(state): State<ProxyState>) -> axum::response::Response {
    let observed_at = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());
    let request = state
        .client
        .get(format!("{}/health", state.runtime_url))
        .timeout(std::time::Duration::from_secs(5));

    match request.send().await {
        Ok(response) if response.status().is_success() => (
            StatusCode::OK,
            axum::Json(serde_json::json!({
                "agent_live": true,
                "runtime_ready": true,
                "runtime_state": "ready",
                "observed_at": observed_at,
            })),
        )
            .into_response(),
        Ok(response) => (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({
                "agent_live": true,
                "runtime_ready": false,
                "runtime_state": "unavailable",
                "reason_code": "runtime_health_rejected",
                "runtime_status": response.status().as_u16(),
                "observed_at": observed_at,
            })),
        )
            .into_response(),
        Err(error) => {
            tracing::warn!(%error, "agent runtime health probe failed");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({
                    "agent_live": true,
                    "runtime_ready": false,
                    "runtime_state": "unavailable",
                    "reason_code": "runtime_connect_failed",
                    "observed_at": observed_at,
                })),
            )
                .into_response()
        }
    }
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

/// Headers forwarded from the trusted gateway request to the runtime endpoint.
///
/// Public client credentials terminate at the gateway. Runtime authentication
/// is injected through the runtime client's default headers and never copied
/// from the inbound request.
const FORWARDED_HEADERS: &[&str] = &[
    "accept",
    "content-type",
    "content-encoding",
    "x-ax-request-id",
    "x-ax-attempt-id",
];

fn pre_admission_response(
    status: StatusCode,
    code: &'static str,
    message: &'static str,
    headers: &HeaderMap,
) -> axum::response::Response {
    let request_id = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown");
    let attempt_id = headers
        .get(ATTEMPT_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown");
    let mut response = (
        status,
        axum::Json(serde_json::json!({
            "error": {
                "code": code,
                "message": message,
                "retryable": true,
                "phase": "pre_admission",
                "admission_state": "not_admitted",
                "request_id": request_id,
                "attempt_id": attempt_id,
            }
        })),
    )
        .into_response();
    response.headers_mut().insert(
        HeaderName::from_static(ADMISSION_STATE_HEADER),
        HeaderValue::from_static("not-admitted"),
    );
    response
}

async fn proxy_to(
    state: &ProxyState,
    path: &str,
    client_headers: &axum::http::HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    if state.draining.load(Ordering::Acquire) {
        return pre_admission_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "AXS_WORKER_DRAINING",
            "worker is draining and not accepting new requests",
            client_headers,
        );
    }
    let Some(_guard) = InflightGuard::try_acquire(&state.inflight, state.max_inflight) else {
        return pre_admission_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "AXS_WORKER_CAPACITY",
            "worker is not accepting additional requests",
            client_headers,
        );
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
    match req.headers(current_trace_headers()).body(body).send().await {
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
                Ok(bytes) => {
                    let sanitize = status.is_client_error() || status.is_server_error();
                    let bytes = if sanitize {
                        sanitize_runtime_error_body(&bytes)
                    } else {
                        bytes
                    };
                    let mut headers = resp_headers;
                    if sanitize {
                        headers.remove(header::CONTENT_LENGTH);
                        headers.insert(
                            header::CONTENT_TYPE,
                            HeaderValue::from_static("application/json"),
                        );
                    }
                    response_builder_with_runtime_headers(status, &headers, !sanitize)
                        .body(axum::body::Body::from(bytes))
                        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                }
                Err(ProxyBodyError::Read(err)) => {
                    tracing::warn!(%err, "agent failed to read runtime response");
                    (
                        StatusCode::BAD_GATEWAY,
                        axum::Json(serde_json::json!({
                            "error": {
                                "code": "AXS_RUNTIME_RESPONSE_READ",
                                "message": "failed to read runtime response"
                            }
                        })),
                    )
                        .into_response()
                }
            }
        }
        Err(err) => {
            tracing::warn!(%err, "agent runtime proxy transport failed");
            (
                StatusCode::BAD_GATEWAY,
                axum::Json(serde_json::json!({
                    "error": {
                        "code": "AXS_RUNTIME_TRANSPORT",
                        "message": "runtime transport is unavailable"
                    }
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    use super::{
        FORWARDED_HEADERS, InflightGuard, ProxyState, add_limited_body_len,
        append_limited_body_chunk, pre_admission_response, proxy_to,
        response_builder_with_runtime_headers, response_declares_oversize,
        sanitize_runtime_error_body, should_forward_runtime_header,
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
    fn public_authorization_is_not_forwarded_to_runtime() {
        assert!(!FORWARDED_HEADERS.contains(&"authorization"));
        assert!(!FORWARDED_HEADERS.contains(&"cookie"));
        assert!(FORWARDED_HEADERS.contains(&"x-ax-request-id"));
    }

    #[test]
    fn runtime_cannot_forge_internal_admission_header() {
        let name = axum::http::HeaderName::from_static("x-ax-admission-state");
        assert!(!should_forward_runtime_header(&name, true));
    }

    #[test]
    fn runtime_errors_are_bounded_and_drop_internal_detail() {
        let body = serde_json::json!({
            "error": {
                "message": format!("bad request {}", "x".repeat(700)),
                "type": "invalid_request_error",
                "code": "invalid_model",
                "stack": "/srv/runtime/model.py:42",
                "runtime_url": "http://10.0.0.8:8000"
            }
        });
        let sanitized = sanitize_runtime_error_body(&serde_json::to_vec(&body).unwrap());
        let text = String::from_utf8(sanitized.to_vec()).unwrap();
        assert!(text.contains("invalid_model"));
        assert!(!text.contains("model.py"));
        assert!(!text.contains("10.0.0.8"));
        assert!(text.len() < 800);
    }

    #[test]
    fn local_capacity_rejection_is_typed_not_admitted() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ax-request-id",
            HeaderValue::from_static("87f5aa94-4042-43b5-bb45-f36e6b559da6"),
        );
        headers.insert(
            "x-ax-attempt-id",
            HeaderValue::from_static("ae7a8560-66cf-4d4a-962a-570021e71fca"),
        );
        let response = pre_admission_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "AXS_WORKER_CAPACITY",
            "worker full",
            &headers,
        );
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["x-ax-admission-state"], "not-admitted");
    }

    #[tokio::test]
    async fn draining_worker_rejects_before_runtime_dispatch() {
        let state = ProxyState {
            client: reqwest::Client::new(),
            runtime_url: "http://127.0.0.1:1".into(),
            inflight: Arc::new(AtomicUsize::new(0)),
            max_inflight: 1,
            draining: Arc::new(AtomicBool::new(true)),
        };
        let response = proxy_to(
            &state,
            "/v1/chat/completions",
            &HeaderMap::new(),
            axum::body::Bytes::from_static(b"{}"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["x-ax-admission-state"], "not-admitted");
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
