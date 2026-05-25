//! NATS JetStream dispatcher (feature = `nats-dispatch`, TASK-MW-015).
//!
//! Publishes inference requests to a durable JetStream stream
//! (`axs.requests.<model_id>`).  Workers pull from the stream via a
//! per-worker pull consumer, process the request, and publish the response
//! back to a per-request ephemeral reply subject.
//!
//! # Message protocol
//!
//! **Request** (published to `axs.requests.<model_id>` in JetStream):
//! ```json
//! {
//!   "request_id": "<uuid>",
//!   "reply_subject": "axs.replies.<uuid>",
//!   "model_id": "llama3-8b",
//!   "stream": false,
//!   "path": "/v1/chat/completions",
//!   "body_hex": "<hex-encoded request body>",
//!   "authorization": "Bearer <token>"
//! }
//! ```
//!
//! **Response** (core-NATS publish to `axs.replies.<uuid>`):
//! ```json
//! {
//!   "request_id": "<uuid>",
//!   "status": 200,
//!   "content_type": "application/json",
//!   "done": true,
//!   "data_hex": "<hex-encoded response body>"
//! }
//! ```
//!
//! For streaming responses `done` is `false` for intermediate chunks and
//! `true` for the final (possibly empty) sentinel.
//!
//! # Error handling
//!
//! If a worker returns a 5xx response it publishes a response with
//! `"status": 5xx` and `"done": true`.  The orchestrator observes this and
//! may retry (nack triggers JetStream redelivery up to `AXS_NATS_MAX_DELIVER`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Context as _;
use async_nats::jetstream;
use axum::body::{Body, Bytes};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::{SinkExt as _, StreamExt as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{error, warn};
use uuid::Uuid;

// ── NatsConfig ────────────────────────────────────────────────────────────────

/// Configuration for the NATS JetStream dispatcher.
#[derive(Debug, Clone)]
pub struct NatsConfig {
    /// NATS server URL, e.g. `nats://127.0.0.1:4222`.
    pub nats_url: String,
    /// JetStream stream name (default `ax-serving`).
    pub stream_name: String,
    /// Max redelivery attempts before the message is dead-lettered.
    pub max_deliver: i64,
    /// Per-request reply timeout in milliseconds.
    pub wait_ms: u64,
}

const DEFAULT_NATS_URL: &str = "nats://127.0.0.1:4222";
const DEFAULT_NATS_STREAM: &str = "ax-serving";
pub(super) const MAX_NATS_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;

impl Default for NatsConfig {
    fn default() -> Self {
        Self {
            nats_url: DEFAULT_NATS_URL.into(),
            stream_name: DEFAULT_NATS_STREAM.into(),
            max_deliver: 3,
            wait_ms: 10_000,
        }
    }
}

impl NatsConfig {
    /// Read configuration from `AXS_NATS_*` environment variables.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("AXS_NATS_URL") {
            cfg.nats_url = v;
        }
        if let Ok(v) = std::env::var("AXS_NATS_STREAM") {
            cfg.stream_name = v;
        }
        if let Ok(v) = std::env::var("AXS_NATS_MAX_DELIVER")
            && let Ok(n) = v.parse::<i64>()
        {
            cfg.max_deliver = n.max(1);
        }
        if let Ok(v) = std::env::var("AXS_GLOBAL_QUEUE_WAIT_MS")
            && let Ok(ms) = v.parse::<u64>()
        {
            cfg.wait_ms = ms.max(1);
        }
        cfg
    }
}

// ── Wire protocol types ───────────────────────────────────────────────────────

/// Request message written to JetStream by the orchestrator.
#[derive(Debug, Serialize, Deserialize)]
pub struct NatsRequest {
    pub request_id: String,
    /// Core-NATS subject the worker must publish the response to.
    pub reply_subject: String,
    pub model_id: String,
    pub stream: bool,
    pub path: String,
    /// Hex-encoded raw HTTP request body bytes.
    pub body_hex: String,
    /// Optional client Authorization header forwarded to the local worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,
}

/// Response message sent by the worker via core-NATS to the reply subject.
#[derive(Debug, Serialize, Deserialize)]
pub struct NatsResponse {
    pub request_id: String,
    pub status: u16,
    pub content_type: String,
    /// `true` = this is the final (or only) message for this request.
    pub done: bool,
    /// Hex-encoded response body bytes (non-streaming) or SSE chunk (streaming).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_hex: Option<String>,
    /// Error message from the worker (non-retryable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl NatsResponse {
    /// A complete (non-streaming) successful response.
    pub fn complete(request_id: String, status: u16, content_type: String, body: &[u8]) -> Self {
        Self {
            request_id,
            status,
            content_type,
            done: true,
            data_hex: Some(hex::encode(body)),
            error: None,
        }
    }

    /// A single streaming chunk (not the final message).
    pub fn streaming_chunk(request_id: String, status: u16, chunk: &[u8]) -> Self {
        Self {
            request_id,
            status,
            content_type: "text/event-stream".into(),
            done: false,
            data_hex: Some(hex::encode(chunk)),
            error: None,
        }
    }

    /// The done sentinel that closes a streaming response.
    pub fn streaming_done(request_id: String, status: u16, error: Option<String>) -> Self {
        Self {
            request_id,
            status,
            content_type: "text/event-stream".into(),
            done: true,
            data_hex: None,
            error,
        }
    }

    /// An error response (status 503 by default; pass a specific status if available).
    pub fn error_response(request_id: String, status: u16, message: String) -> Self {
        Self {
            request_id,
            status,
            content_type: "application/json".into(),
            done: true,
            data_hex: None,
            error: Some(message),
        }
    }

    /// Serialize to a NATS payload, falling back to an empty vec on encode failure.
    pub fn to_payload(&self) -> Bytes {
        serde_json::to_vec(self).unwrap_or_default().into()
    }
}

#[derive(Debug)]
enum NatsBodyDecodeError {
    InvalidHex(hex::FromHexError),
    TooLarge,
}

fn decode_body_hex_limited(
    data_hex: Option<&str>,
    limit_bytes: usize,
) -> Result<Vec<u8>, NatsBodyDecodeError> {
    let Some(data_hex) = data_hex else {
        return Ok(Vec::new());
    };

    if data_hex.len() > limit_bytes.saturating_mul(2) {
        return Err(NatsBodyDecodeError::TooLarge);
    }
    let body = hex::decode(data_hex).map_err(NatsBodyDecodeError::InvalidHex)?;
    if body.len() > limit_bytes {
        return Err(NatsBodyDecodeError::TooLarge);
    }
    Ok(body)
}

fn decode_stream_body_hex_limited(
    data_hex: Option<&str>,
    total_bytes: &mut usize,
    limit_bytes: usize,
) -> Result<Vec<u8>, NatsBodyDecodeError> {
    let remaining = limit_bytes
        .checked_sub(*total_bytes)
        .ok_or(NatsBodyDecodeError::TooLarge)?;
    let body = decode_body_hex_limited(data_hex, remaining)?;
    *total_bytes = total_bytes
        .checked_add(body.len())
        .ok_or(NatsBodyDecodeError::TooLarge)?;
    Ok(body)
}

fn nats_status(status: u16) -> StatusCode {
    StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

fn is_event_stream(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
}

fn build_complete_nats_response(request_id: &str, resp: NatsResponse) -> Response {
    let status = nats_status(resp.status);
    let content_type = resp.content_type.clone();
    let body_bytes =
        match decode_body_hex_limited(resp.data_hex.as_deref(), MAX_NATS_RESPONSE_BODY_BYTES) {
            Ok(body) => body,
            Err(NatsBodyDecodeError::InvalidHex(e)) => {
                error!(%request_id, %e, "NATS: failed to decode response body hex");
                return (StatusCode::BAD_GATEWAY, "NATS: bad response body encoding")
                    .into_response();
            }
            Err(NatsBodyDecodeError::TooLarge) => {
                error!(
                    %request_id,
                    limit = MAX_NATS_RESPONSE_BODY_BYTES,
                    "NATS: response body exceeded size limit"
                );
                return (
                    StatusCode::BAD_GATEWAY,
                    "NATS response body exceeded size limit",
                )
                    .into_response();
            }
        };

    let body = if let Some(error) = resp.error {
        if body_bytes.is_empty() {
            Bytes::from(
                serde_json::to_vec(&serde_json::json!({ "error": error }))
                    .unwrap_or_else(|_| b"{\"error\":\"worker error\"}".to_vec()),
            )
        } else {
            Bytes::from(body_bytes)
        }
    } else {
        Bytes::from(body_bytes)
    };

    axum::response::Response::builder()
        .status(status)
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn stream_frames_from_nats_response(
    request_id: &str,
    resp: NatsResponse,
    total_bytes: &mut usize,
    reroute_total: &Arc<AtomicU64>,
) -> Result<(Vec<Bytes>, bool), std::io::Error> {
    if resp.request_id != request_id {
        let err = format!(
            "NATS: response ID mismatch (want {request_id}, got {})",
            resp.request_id
        );
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, err));
    }

    if resp.status >= 500 {
        reroute_total.fetch_add(1, Ordering::Relaxed);
    }
    if let Some(ref err_msg) = resp.error {
        warn!(
            request_id = %resp.request_id,
            %err_msg,
            status = resp.status,
            "NATS: worker returned streaming error"
        );
    }

    let chunk = match decode_stream_body_hex_limited(
        resp.data_hex.as_deref(),
        total_bytes,
        MAX_NATS_RESPONSE_BODY_BYTES,
    ) {
        Ok(b) => b,
        Err(NatsBodyDecodeError::InvalidHex(e)) => {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
        }
        Err(NatsBodyDecodeError::TooLarge) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "NATS streaming response body exceeded size limit",
            ));
        }
    };

    let mut frames = Vec::new();
    if !chunk.is_empty() {
        frames.push(Bytes::from(chunk));
    }
    if let Some(err_msg) = resp.error.as_deref() {
        let payload = serde_json::to_string(&serde_json::json!({ "error": err_msg }))
            .unwrap_or_else(|_| "{\"error\":\"serialization failure\"}".to_string());
        frames.push(Bytes::from(format!("event: error\ndata: {payload}\n\n")));
    }

    Ok((frames, resp.done))
}

// ── NatsDispatcher ────────────────────────────────────────────────────────────

/// Orchestrator-side NATS dispatcher.
///
/// Publishes requests to JetStream and waits for responses on per-request
/// ephemeral core-NATS subscriptions.
///
/// Cloning is cheap — all fields are `Arc`-backed.
#[derive(Clone)]
pub struct NatsDispatcher {
    client: async_nats::Client,
    jetstream: jetstream::Context,
    config: Arc<NatsConfig>,
    reroute_total: Arc<AtomicU64>,
}

impl NatsDispatcher {
    /// Connect to NATS and ensure the request stream exists.
    pub async fn connect(config: NatsConfig) -> anyhow::Result<Self> {
        let client = async_nats::connect(&config.nats_url)
            .await
            .with_context(|| format!("failed to connect to NATS at {}", config.nats_url))?;

        let jetstream = jetstream::new(client.clone());

        // Idempotent stream creation — subjects `axs.requests.*`.
        // Note: max_deliver is a consumer-level setting, not a stream setting.
        jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: config.stream_name.clone(),
                subjects: vec!["axs.requests.>".to_string()],
                ..Default::default()
            })
            .await
            .context("failed to create JetStream request stream")?;

        Ok(Self {
            client,
            jetstream,
            config: Arc::new(config),
            reroute_total: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Total number of reroutes (5xx responses from workers).
    pub fn reroutes(&self) -> u64 {
        self.reroute_total.load(Ordering::Relaxed)
    }

    /// Forward a client request to a worker via NATS JetStream.
    pub async fn forward(&self, model_id: &str, stream: bool, path: &str, body: Bytes) -> Response {
        self.forward_with_auth(model_id, stream, path, body, None)
            .await
    }

    /// Forward a client request to a worker via NATS JetStream, preserving the
    /// Authorization header for local workers that also enforce bearer auth.
    pub async fn forward_with_auth(
        &self,
        model_id: &str,
        stream: bool,
        path: &str,
        body: Bytes,
        auth_header: Option<&HeaderValue>,
    ) -> Response {
        let request_id = Uuid::new_v4().to_string();
        let reply_subject = format!("axs.replies.{}", request_id);

        // Subscribe to the ephemeral reply subject BEFORE publishing so we
        // cannot miss the response.
        let sub = match self.client.subscribe(reply_subject.clone()).await {
            Ok(s) => s,
            Err(e) => {
                error!(%e, "NATS: failed to subscribe to reply subject");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("NATS subscribe failed: {e}"),
                )
                    .into_response();
            }
        };

        // Build and publish the request message to JetStream.
        let req = NatsRequest {
            request_id: request_id.clone(),
            reply_subject,
            model_id: model_id.to_string(),
            stream,
            path: path.to_string(),
            body_hex: hex::encode(&body),
            authorization: auth_header
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned),
        };
        let payload = match serde_json::to_vec(&req) {
            Ok(v) => v,
            Err(e) => {
                error!(%e, "NATS: failed to serialize request");
                return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
            }
        };

        let subject = format!("axs.requests.{}", sanitize_subject_component(model_id));
        // publish() returns Ok(PublishAckFuture) — we must await the future a
        // second time to get JetStream's persistence ACK.  Dropping it without
        // awaiting means we proceed even if the stream rejected the message.
        let ack_future = match self
            .jetstream
            .publish(subject.clone(), payload.into())
            .await
        {
            Ok(f) => f,
            Err(e) => {
                error!(%subject, %e, "NATS: JetStream publish failed");
                return (StatusCode::BAD_GATEWAY, format!("NATS publish failed: {e}"))
                    .into_response();
            }
        };
        if let Err(e) = ack_future.await {
            error!(%subject, %e, "NATS: JetStream publish not acknowledged by server");
            return (
                StatusCode::BAD_GATEWAY,
                format!("NATS publish not acknowledged: {e}"),
            )
                .into_response();
        }

        let timeout_dur = Duration::from_millis(self.config.wait_ms);

        if !stream {
            self.recv_non_streaming(sub, timeout_dur, &request_id).await
        } else {
            self.recv_streaming(sub, timeout_dur, &request_id).await
        }
    }

    /// Receive a single non-streaming response.
    async fn recv_non_streaming(
        &self,
        mut sub: async_nats::Subscriber,
        timeout: Duration,
        request_id: &str,
    ) -> Response {
        match tokio::time::timeout(timeout, sub.next()).await {
            Err(_) => {
                warn!(%request_id, "NATS: timed out waiting for response");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "NATS request timed out waiting for worker response",
                )
                    .into_response()
            }
            Ok(None) => {
                warn!(%request_id, "NATS: reply subscription closed unexpectedly");
                (StatusCode::BAD_GATEWAY, "NATS reply subscription closed").into_response()
            }
            Ok(Some(msg)) => {
                let resp: NatsResponse = match serde_json::from_slice(&msg.payload) {
                    Ok(r) => r,
                    Err(e) => {
                        error!(%request_id, %e, "NATS: failed to deserialize response");
                        return (StatusCode::BAD_GATEWAY, e.to_string()).into_response();
                    }
                };

                if resp.request_id != request_id {
                    error!(
                        %request_id,
                        got = %resp.request_id,
                        "NATS: response request_id mismatch"
                    );
                    return (StatusCode::BAD_GATEWAY, "NATS: response ID mismatch").into_response();
                }

                if let Some(err) = resp.error {
                    warn!(%request_id, %err, "NATS: worker returned error");
                    self.reroute_total.fetch_add(1, Ordering::Relaxed);
                    return (StatusCode::BAD_GATEWAY, err).into_response();
                }

                let status =
                    StatusCode::from_u16(resp.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                if status.is_server_error() {
                    self.reroute_total.fetch_add(1, Ordering::Relaxed);
                }

                let body_bytes = match decode_body_hex_limited(
                    resp.data_hex.as_deref(),
                    MAX_NATS_RESPONSE_BODY_BYTES,
                ) {
                    Ok(body) => body,
                    Err(NatsBodyDecodeError::InvalidHex(e)) => {
                        error!(%request_id, %e, "NATS: failed to decode response body hex");
                        return (StatusCode::BAD_GATEWAY, "NATS: bad response body encoding")
                            .into_response();
                    }
                    Err(NatsBodyDecodeError::TooLarge) => {
                        error!(
                            %request_id,
                            limit = MAX_NATS_RESPONSE_BODY_BYTES,
                            "NATS: response body exceeded size limit"
                        );
                        return (
                            StatusCode::BAD_GATEWAY,
                            "NATS response body exceeded size limit",
                        )
                            .into_response();
                    }
                };

                axum::response::Response::builder()
                    .status(status)
                    .header("content-type", resp.content_type)
                    .body(Body::from(body_bytes))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
            }
        }
    }

    /// Receive a streaming response — collect chunks until `done = true`.
    async fn recv_streaming(
        &self,
        mut sub: async_nats::Subscriber,
        timeout: Duration,
        request_id: &str,
    ) -> Response {
        let first = match tokio::time::timeout(timeout, sub.next()).await {
            Err(_) => {
                warn!(%request_id, "NATS: timed out waiting for first streaming response");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "NATS request timed out waiting for worker response",
                )
                    .into_response();
            }
            Ok(None) => {
                warn!(%request_id, "NATS: reply subscription closed unexpectedly");
                return (StatusCode::BAD_GATEWAY, "NATS reply subscription closed").into_response();
            }
            Ok(Some(msg)) => {
                let resp: NatsResponse = match serde_json::from_slice(&msg.payload) {
                    Ok(r) => r,
                    Err(e) => {
                        error!(%request_id, %e, "NATS: failed to deserialize response");
                        return (StatusCode::BAD_GATEWAY, e.to_string()).into_response();
                    }
                };
                if resp.request_id != request_id {
                    error!(
                        %request_id,
                        got = %resp.request_id,
                        "NATS: response request_id mismatch"
                    );
                    return (StatusCode::BAD_GATEWAY, "NATS: response ID mismatch").into_response();
                }
                resp
            }
        };

        if first.done && (first.error.is_some() || !is_event_stream(&first.content_type)) {
            if first.status >= 500 {
                self.reroute_total.fetch_add(1, Ordering::Relaxed);
            }
            return build_complete_nats_response(request_id, first);
        }

        let (tx, rx) = futures::channel::mpsc::channel::<Result<Bytes, std::io::Error>>(32);
        let reroute_total = Arc::clone(&self.reroute_total);
        let request_id = request_id.to_string();

        tokio::spawn(async move {
            let mut tx = tx;
            let deadline = tokio::time::Instant::now() + timeout;
            let mut total_bytes = 0usize;
            let mut pending = Some(first);
            while let Some(remaining) = deadline.checked_duration_since(tokio::time::Instant::now())
            {
                let resp = if let Some(resp) = pending.take() {
                    resp
                } else {
                    let msg = match tokio::time::timeout(remaining, sub.next()).await {
                        Ok(Some(msg)) => msg,
                        _ => break,
                    };
                    match serde_json::from_slice(&msg.payload) {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = tx
                                .send(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                                .await;
                            break;
                        }
                    }
                };

                match stream_frames_from_nats_response(
                    &request_id,
                    resp,
                    &mut total_bytes,
                    &reroute_total,
                ) {
                    Ok((frames, done)) => {
                        for frame in frames {
                            if tx.send(Ok(frame)).await.is_err() {
                                return;
                            }
                        }
                        if done {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = tx.send(Err(error)).await;
                        break;
                    }
                }
            }
        });

        axum::response::Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(rx))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    }
}

pub(super) fn sanitize_subject_component(model_id: &str) -> String {
    let digest = Sha256::digest(model_id.as_bytes());
    hex::encode(&digest[..16])
}

impl std::fmt::Debug for NatsDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NatsDispatcher")
            .field("nats_url", &self.config.nats_url)
            .field("stream_name", &self.config.stream_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use axum::body::{self, Bytes};
    use axum::http::StatusCode;

    use super::{
        NatsBodyDecodeError, NatsConfig, NatsRequest, NatsResponse, build_complete_nats_response,
        decode_body_hex_limited, decode_stream_body_hex_limited, is_event_stream,
        sanitize_subject_component, stream_frames_from_nats_response,
    };

    #[test]
    fn sanitize_subject_component_preserves_allowed_chars() {
        let out = sanitize_subject_component("llama3_8b-instruct");
        assert_eq!(out.len(), 32);
        assert!(out.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sanitize_subject_component_distinguishes_colliding_plaintext_forms() {
        assert_ne!(
            sanitize_subject_component("org/model.v1"),
            sanitize_subject_component("org_model.v1")
        );
    }

    #[test]
    fn nats_config_clamps_zero_wait_ms_from_env() {
        let _guard = crate::test_env::lock();
        unsafe { std::env::set_var("AXS_GLOBAL_QUEUE_WAIT_MS", "0") };
        let cfg = NatsConfig::from_env();
        unsafe { std::env::remove_var("AXS_GLOBAL_QUEUE_WAIT_MS") };

        assert_eq!(cfg.wait_ms, 1);
    }

    #[test]
    fn decode_body_hex_limited_rejects_oversized_payload_before_decode() {
        let err = decode_body_hex_limited(Some("000000"), 2).unwrap_err();
        assert!(matches!(err, NatsBodyDecodeError::TooLarge));
    }

    #[test]
    fn decode_body_hex_limited_rejects_invalid_hex() {
        let err = decode_body_hex_limited(Some("not-hex"), 64).unwrap_err();
        assert!(matches!(err, NatsBodyDecodeError::InvalidHex(_)));
    }

    #[test]
    fn decode_body_hex_limited_accepts_empty_and_valid_payloads() {
        assert_eq!(decode_body_hex_limited(None, 64).unwrap(), Vec::<u8>::new());
        assert_eq!(decode_body_hex_limited(Some("6869"), 64).unwrap(), b"hi");
    }

    #[test]
    fn decode_stream_body_hex_limited_tracks_cumulative_limit() {
        let mut total = 0usize;
        assert_eq!(
            decode_stream_body_hex_limited(Some("3132"), &mut total, 4).unwrap(),
            b"12"
        );
        assert_eq!(total, 2);
        assert_eq!(
            decode_stream_body_hex_limited(Some("3334"), &mut total, 4).unwrap(),
            b"34"
        );
        assert_eq!(total, 4);
        let err = decode_stream_body_hex_limited(Some("35"), &mut total, 4).unwrap_err();
        assert!(matches!(err, NatsBodyDecodeError::TooLarge));
        assert_eq!(total, 4);
    }

    #[test]
    fn nats_request_authorization_is_optional_for_backward_compatibility() {
        let payload = serde_json::json!({
            "request_id": "req-1",
            "reply_subject": "axs.replies.req-1",
            "model_id": "m1",
            "stream": false,
            "path": "/v1/chat/completions",
            "body_hex": "7b7d"
        });

        let req: NatsRequest = serde_json::from_value(payload).unwrap();

        assert_eq!(req.authorization, None);
    }

    #[test]
    fn nats_request_authorization_round_trips() {
        let req = NatsRequest {
            request_id: "req-1".into(),
            reply_subject: "axs.replies.req-1".into(),
            model_id: "m1".into(),
            stream: false,
            path: "/v1/chat/completions".into(),
            body_hex: "7b7d".into(),
            authorization: Some("Bearer secret".into()),
        };

        let encoded = serde_json::to_vec(&req).unwrap();
        let decoded: NatsRequest = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded.authorization.as_deref(), Some("Bearer secret"));
    }

    #[test]
    fn event_stream_content_type_allows_parameters() {
        assert!(is_event_stream("text/event-stream"));
        assert!(is_event_stream("text/event-stream; charset=utf-8"));
        assert!(!is_event_stream("application/json"));
    }

    #[tokio::test]
    async fn complete_nats_response_preserves_pre_stream_error_status() {
        let resp = NatsResponse::complete(
            "req-1".to_string(),
            400,
            "application/json".to_string(),
            br#"{"error":"bad request"}"#,
        );

        let response = build_complete_nats_response("req-1", resp);

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], br#"{"error":"bad request"}"#);
    }

    #[test]
    fn stream_frames_from_nats_response_emits_initial_chunk() {
        let reroutes = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut total = 0usize;
        let resp = NatsResponse::streaming_chunk("req-1".to_string(), 200, b"data: hello\n\n");

        let (frames, done) =
            stream_frames_from_nats_response("req-1", resp, &mut total, &reroutes).unwrap();

        assert!(!done);
        assert_eq!(frames, vec![Bytes::from_static(b"data: hello\n\n")]);
        assert_eq!(total, "data: hello\n\n".len());
    }
}
