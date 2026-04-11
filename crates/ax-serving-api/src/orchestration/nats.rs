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
//!   "body_hex": "<hex-encoded request body>"
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
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::{SinkExt as _, StreamExt as _};
use serde::{Deserialize, Serialize};
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
            cfg.wait_ms = ms;
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
    pub fn streaming_done(request_id: String, status: u16) -> Self {
        Self {
            request_id,
            status,
            content_type: "text/event-stream".into(),
            done: true,
            data_hex: None,
            error: None,
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

                let body_bytes: Vec<u8> =
                    match resp.data_hex.as_deref().map(hex::decode).transpose() {
                        Ok(b) => b.unwrap_or_default(),
                        Err(e) => {
                            error!(%request_id, %e, "NATS: failed to decode response body hex");
                            return (StatusCode::BAD_GATEWAY, "NATS: bad response body encoding")
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
        sub: async_nats::Subscriber,
        timeout: Duration,
        request_id: &str,
    ) -> Response {
        let (tx, rx) = futures::channel::mpsc::channel::<Result<Bytes, std::io::Error>>(32);
        let reroute_total = Arc::clone(&self.reroute_total);
        let request_id = request_id.to_string();

        tokio::spawn(async move {
            let mut sub = sub;
            let mut tx = tx;
            let deadline = tokio::time::Instant::now() + timeout;
            while let Some(remaining) = deadline.checked_duration_since(tokio::time::Instant::now())
            {
                let msg = match tokio::time::timeout(remaining, sub.next()).await {
                    Ok(Some(msg)) => msg,
                    _ => break,
                };
                let resp: NatsResponse = match serde_json::from_slice(&msg.payload) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx
                            .send(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                            .await;
                        break;
                    }
                };

                if resp.request_id != request_id {
                    let err = format!(
                        "NATS: response ID mismatch (want {request_id}, got {})",
                        resp.request_id
                    );
                    let _ = tx
                        .send(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            err,
                        )))
                        .await;
                    break;
                }

                if resp.status >= 500 {
                    reroute_total.fetch_add(1, Ordering::Relaxed);
                }
                // Log worker-side errors so they are not silently dropped.
                // (HTTP headers are already sent as 200 for streaming, so we
                // cannot change the status code — at minimum we surface the error
                // in logs for operator visibility.)
                if let Some(ref err_msg) = resp.error {
                    warn!(
                        request_id = %resp.request_id,
                        %err_msg,
                        status = resp.status,
                        "NATS: worker returned streaming error"
                    );
                }

                let chunk: Vec<u8> = match resp.data_hex.as_deref().map(hex::decode).transpose() {
                    Ok(b) => b.unwrap_or_default(),
                    Err(e) => {
                        let _ = tx
                            .send(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                            .await;
                        break;
                    }
                };

                if !chunk.is_empty() && tx.send(Ok(Bytes::from(chunk))).await.is_err() {
                    break;
                }
                if resp.done {
                    break;
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
    model_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
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
    use super::sanitize_subject_component;

    #[test]
    fn sanitize_subject_component_preserves_allowed_chars() {
        assert_eq!(
            sanitize_subject_component("llama3_8b-instruct"),
            "llama3_8b-instruct"
        );
    }

    #[test]
    fn sanitize_subject_component_replaces_disallowed_chars() {
        assert_eq!(
            sanitize_subject_component("org/model.v1:beta"),
            "org_model_v1_beta"
        );
    }
}
