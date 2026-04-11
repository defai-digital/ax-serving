//! NATS JetStream worker sidecar (feature = `nats-dispatch`, TASK-MW-015).
//!
//! Runs alongside a local `ax-serving serve` instance and bridges NATS
//! JetStream messages to the local HTTP inference endpoint.
//!
//! # Lifecycle
//!
//! 1. Connect to NATS and ensure the request stream exists.
//! 2. For each `model_id` the worker serves, create (or attach to) a durable
//!    pull consumer that filters on `axs.requests.<model_id>`.
//! 3. Pull messages in a loop, forward to the local HTTP endpoint, publish
//!    the response to the request's `reply_subject`.
//! 4. **Ack** the message on success.
//! 5. **Nack** the message on worker error (HTTP 5xx or network failure) so
//!    that JetStream redelivers to another worker (up to `max_deliver` times).
//!
//! # Streaming
//!
//! For streaming requests (`stream: true`) the worker forwards the request to
//! the local `/v1/chat/completions` endpoint with `stream: true`, reads the
//! SSE response line by line, and publishes one `NatsResponse` per SSE chunk
//! with `done: false`, followed by a final sentinel with `done: true`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use async_nats::jetstream;
use futures::StreamExt as _;
use reqwest::Client;
use tokio::sync::watch;
use tracing::{error, info, warn};
use uuid::Uuid;

use super::nats::{NatsConfig, NatsRequest, NatsResponse};

// ── NatsWorker ────────────────────────────────────────────────────────────────

/// NATS sidecar that bridges JetStream messages to a local HTTP inference server.
///
/// Constructed via [`NatsWorker::connect`] and run with [`NatsWorker::run`].
pub struct NatsWorker {
    /// Unique ID for this worker instance (used to name JetStream consumers).
    worker_id: String,
    /// Address of the local HTTP inference server (e.g. `127.0.0.1:8082`).
    local_addr: SocketAddr,
    /// Model IDs this worker has loaded and will serve.
    model_ids: Vec<String>,
    /// NATS client (core + JetStream).
    client: async_nats::Client,
    /// JetStream context.
    jetstream: jetstream::Context,
    /// Validated config.
    config: Arc<NatsConfig>,
    /// HTTP client for forwarding to the local inference server.
    http_client: Client,
}

impl NatsWorker {
    /// Connect to NATS and prepare the worker sidecar.
    pub async fn connect(
        config: NatsConfig,
        local_addr: SocketAddr,
        model_ids: Vec<String>,
    ) -> anyhow::Result<Self> {
        let client = async_nats::connect(&config.nats_url)
            .await
            .with_context(|| {
                format!(
                    "NatsWorker: failed to connect to NATS at {}",
                    config.nats_url
                )
            })?;

        let jetstream = jetstream::new(client.clone());

        // Ensure the request stream exists (idempotent).
        // Note: max_deliver is a consumer-level setting, not a stream setting.
        jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: config.stream_name.clone(),
                subjects: vec!["axs.requests.>".to_string()],
                ..Default::default()
            })
            .await
            .context("NatsWorker: failed to get/create JetStream stream")?;

        let http_client = Client::builder()
            .connect_timeout(Duration::from_millis(config.wait_ms.max(1)))
            .pool_max_idle_per_host(4)
            .build()
            .context("NatsWorker: failed to build HTTP client")?;

        Ok(Self {
            worker_id: Uuid::new_v4().to_string(),
            local_addr,
            model_ids,
            client,
            jetstream,
            config: Arc::new(config),
            http_client,
        })
    }

    /// Run the sidecar until `shutdown` emits `true`.
    ///
    /// Spawns one task per model_id, each running a pull-consumer message loop.
    pub async fn run(self, mut shutdown: watch::Receiver<bool>) -> anyhow::Result<()> {
        info!(
            worker_id = %self.worker_id,
            local_addr = %self.local_addr,
            model_ids = ?self.model_ids,
            "NatsWorker started"
        );

        let worker_id = Arc::new(self.worker_id.clone());
        let local_addr = self.local_addr;
        let config = Arc::clone(&self.config);
        let http_client = self.http_client.clone();
        let client = self.client.clone();
        let jetstream = self.jetstream.clone();

        let mut handles = Vec::new();
        for model_id in &self.model_ids {
            let model_id = model_id.clone();
            let worker_id = Arc::clone(&worker_id);
            let config = Arc::clone(&config);
            let http_client = http_client.clone();
            let client = client.clone();
            let js = jetstream.clone();
            let shutdown_rx = shutdown.clone();

            handles.push(tokio::spawn(async move {
                run_model_loop(
                    js,
                    client,
                    http_client,
                    config,
                    worker_id,
                    local_addr,
                    model_id,
                    shutdown_rx,
                )
                .await
            }));
        }

        // Wait for shutdown signal.
        let _ = shutdown.changed().await;
        info!("NatsWorker: shutdown signal received, stopping");

        // Cancel all model loops (they observe the watch channel).
        for h in handles {
            h.abort();
        }

        Ok(())
    }
}

// ── Per-model pull-consumer loop ──────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn run_model_loop(
    jetstream: jetstream::Context,
    client: async_nats::Client,
    http_client: Client,
    config: Arc<NatsConfig>,
    worker_id: Arc<String>,
    local_addr: SocketAddr,
    model_id: String,
    mut shutdown: watch::Receiver<bool>,
) {
    let consumer_name = format!(
        "worker-{}-{}",
        worker_id.as_str(),
        super::nats::sanitize_subject_component(&model_id)
    );
    let filter_subject = format!(
        "axs.requests.{}",
        super::nats::sanitize_subject_component(&model_id)
    );

    // Get the stream so we can attach a pull consumer.
    let stream = match jetstream.get_stream(config.stream_name.as_str()).await {
        Ok(s) => s,
        Err(e) => {
            error!(%model_id, %e, "NatsWorker: failed to get stream");
            return;
        }
    };

    // Create or reattach to a durable pull consumer.
    // max_deliver limits JetStream redelivery attempts before dead-lettering.
    let consumer = match stream
        .get_or_create_consumer(
            &consumer_name,
            jetstream::consumer::pull::Config {
                durable_name: Some(consumer_name.clone()),
                filter_subject: filter_subject.clone(),
                max_deliver: config.max_deliver,
                ..Default::default()
            },
        )
        .await
    {
        Ok(c) => c,
        Err(e) => {
            error!(%model_id, %e, "NatsWorker: failed to create pull consumer");
            return;
        }
    };

    info!(%model_id, %consumer_name, "NatsWorker: pull consumer ready");

    loop {
        // Check shutdown before blocking.
        if *shutdown.borrow() {
            break;
        }

        // Fetch a batch of messages (up to 8, non-blocking if none available).
        let batch = consumer
            .fetch()
            .max_messages(8)
            .expires(Duration::from_millis(500))
            .messages()
            .await;

        let mut messages = match batch {
            Ok(m) => m,
            Err(e) => {
                warn!(%model_id, %e, "NatsWorker: fetch failed, retrying");
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
        };

        while let Some(result) = tokio::select! {
            msg = messages.next() => msg,
            _ = shutdown.changed() => None,
        } {
            match result {
                Ok(msg) => {
                    let req: NatsRequest = match serde_json::from_slice(&msg.message.payload) {
                        Ok(r) => r,
                        Err(e) => {
                            warn!(%model_id, %e, "NatsWorker: failed to parse request");
                            if let Err(e) = msg.ack().await {
                                warn!(%model_id, %e, "NatsWorker: ack failed for malformed request");
                            }
                            continue;
                        }
                    };

                    if req.model_id != model_id {
                        warn!(
                            expected_model = %model_id,
                            got_model = %req.model_id,
                            request_id = %req.request_id,
                            "NatsWorker: request model_id did not match consumer model"
                        );
                        publish_error(
                            &client,
                            &req,
                            "worker received request for a different model",
                        )
                        .await;
                        if let Err(e) = msg.ack().await {
                            warn!(%model_id, %e, "NatsWorker: ack failed for mismatched model");
                        }
                        continue;
                    }

                    let succeeded = dispatch_to_local(
                        &http_client,
                        &client,
                        local_addr,
                        &req,
                        Duration::from_millis(config.wait_ms.max(1)),
                    )
                    .await;

                    if succeeded {
                        if let Err(e) = msg.ack().await {
                            warn!(%model_id, %e, "NatsWorker: ack failed");
                        }
                    } else {
                        // Nack triggers JetStream redelivery.
                        if let Err(e) = msg.ack_with(jetstream::AckKind::Nak(None)).await {
                            warn!(%model_id, %e, "NatsWorker: nack failed");
                        }
                    }
                }
                Err(e) => {
                    warn!(%model_id, %e, "NatsWorker: message error");
                }
            }
        }
    }
}

// ── Local HTTP dispatch ───────────────────────────────────────────────────────

/// Forward one NATS request to the local HTTP inference server and publish
/// the response back to the `reply_subject`.
///
/// Returns `true` on success (ack), `false` on worker error (nack).
async fn dispatch_to_local(
    http_client: &Client,
    nats_client: &async_nats::Client,
    local_addr: SocketAddr,
    req: &NatsRequest,
    request_timeout: Duration,
) -> bool {
    let url = format!("http://{}{}", local_addr, req.path);

    let body_bytes = match hex::decode(&req.body_hex) {
        Ok(b) => b,
        Err(e) => {
            error!(request_id = %req.request_id, %e, "NatsWorker: bad body hex");
            // Publish error response so the orchestrator isn't left hanging.
            publish_error(nats_client, req, "bad body hex encoding").await;
            return true; // ack — not retryable
        }
    };

    if !req.stream {
        dispatch_non_streaming(
            http_client,
            nats_client,
            &url,
            req,
            body_bytes,
            request_timeout,
        )
        .await
    } else {
        dispatch_streaming(
            http_client,
            nats_client,
            &url,
            req,
            body_bytes,
            request_timeout,
        )
        .await
    }
}

/// POST JSON to a local HTTP endpoint and return the response.
async fn http_post_json(
    client: &Client,
    url: &str,
    body: Vec<u8>,
) -> reqwest::Result<reqwest::Response> {
    client
        .post(url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
}

async fn dispatch_non_streaming(
    http_client: &Client,
    nats_client: &async_nats::Client,
    url: &str,
    req: &NatsRequest,
    body_bytes: Vec<u8>,
    request_timeout: Duration,
) -> bool {
    let result = match tokio::time::timeout(
        request_timeout,
        http_post_json(http_client, url, body_bytes),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => {
            warn!(
                request_id = %req.request_id,
                timeout_ms = request_timeout.as_millis(),
                "NatsWorker: local HTTP request timed out"
            );
            publish_error(nats_client, req, "local HTTP request timed out").await;
            return false;
        }
    };

    match result {
        Err(e) => {
            warn!(request_id = %req.request_id, %e, "NatsWorker: local HTTP request failed");
            publish_error(nats_client, req, &e.to_string()).await;
            false // nack — network error, retry with another worker
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();

            match tokio::time::timeout(request_timeout, resp.bytes()).await {
                Err(_) => {
                    warn!(
                        request_id = %req.request_id,
                        timeout_ms = request_timeout.as_millis(),
                        "NatsWorker: reading response body timed out"
                    );
                    publish_error(nats_client, req, "reading local response body timed out").await;
                    false
                }
                Ok(result) => match result {
                    Err(e) => {
                        warn!(request_id = %req.request_id, %e, "NatsWorker: reading response body failed");
                        publish_error(nats_client, req, &e.to_string()).await;
                        false
                    }
                    Ok(body) => {
                        let payload = NatsResponse::complete(
                            req.request_id.clone(),
                            status,
                            content_type,
                            &body,
                        )
                        .to_payload();
                        if let Err(e) = nats_client
                            .publish(req.reply_subject.clone(), payload)
                            .await
                        {
                            error!(request_id = %req.request_id, %e, "NatsWorker: reply publish failed");
                        }
                        status < 500 // ack on success; nack on 5xx (allows redelivery)
                    }
                },
            }
        }
    }
}

async fn dispatch_streaming(
    http_client: &Client,
    nats_client: &async_nats::Client,
    url: &str,
    req: &NatsRequest,
    body_bytes: Vec<u8>,
    request_timeout: Duration,
) -> bool {
    let resp = match tokio::time::timeout(
        request_timeout,
        http_post_json(http_client, url, body_bytes),
    )
    .await
    {
        Err(_) => {
            warn!(
                request_id = %req.request_id,
                timeout_ms = request_timeout.as_millis(),
                "NatsWorker: streaming local HTTP request timed out"
            );
            publish_error(nats_client, req, "streaming local HTTP request timed out").await;
            return false;
        }
        Ok(result) => match result {
            Err(e) => {
                warn!(request_id = %req.request_id, %e, "NatsWorker: streaming local HTTP failed");
                publish_error(nats_client, req, &e.to_string()).await;
                return false;
            }
            Ok(r) => r,
        },
    };

    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    if status >= 500 {
        // Publish a 5xx response and nack so JetStream retries.
        let payload = NatsResponse::error_response(
            req.request_id.clone(),
            status,
            format!("worker returned {status}"),
        )
        .to_payload();
        let _ = nats_client
            .publish(req.reply_subject.clone(), payload)
            .await;
        return false;
    }

    if !resp.status().is_success() {
        match resp.bytes().await {
            Err(e) => {
                warn!(request_id = %req.request_id, %e, "NatsWorker: reading error response body failed");
                publish_error(nats_client, req, &e.to_string()).await;
                return false;
            }
            Ok(body) => {
                let payload =
                    NatsResponse::complete(req.request_id.clone(), status, content_type, &body)
                        .to_payload();
                if let Err(e) = nats_client
                    .publish(req.reply_subject.clone(), payload)
                    .await
                {
                    error!(request_id = %req.request_id, %e, "NatsWorker: reply publish failed");
                }
                return true;
            }
        }
    }

    // Stream chunks to the reply subject.
    let mut byte_stream = resp.bytes_stream();
    let mut stream_error = None;
    loop {
        match tokio::time::timeout(request_timeout, byte_stream.next()).await {
            Err(_) => {
                warn!(
                    request_id = %req.request_id,
                    timeout_ms = request_timeout.as_millis(),
                    "NatsWorker: stream read timed out"
                );
                stream_error = Some("local stream read timed out".to_string());
                break;
            }
            Ok(None) => break,
            Ok(Some(Err(e))) => {
                warn!(request_id = %req.request_id, %e, "NatsWorker: stream read error");
                stream_error = Some(format!("local stream read failed: {e}"));
                break;
            }
            Ok(Some(Ok(chunk))) => {
                let payload = NatsResponse::streaming_chunk(req.request_id.clone(), status, &chunk)
                    .to_payload();
                if let Err(e) = nats_client
                    .publish(req.reply_subject.clone(), payload)
                    .await
                {
                    error!(request_id = %req.request_id, %e, "NatsWorker: chunk publish failed");
                    break;
                }
            }
        }
    }

    // Send the done sentinel.
    let done_status = if stream_error.is_some() { 502 } else { status };
    let payload = NatsResponse::streaming_done(req.request_id.clone(), done_status, stream_error)
        .to_payload();
    let _ = nats_client
        .publish(req.reply_subject.clone(), payload)
        .await;

    true // ack — streaming completed
}

async fn publish_error(nats_client: &async_nats::Client, req: &NatsRequest, message: &str) {
    let payload =
        NatsResponse::error_response(req.request_id.clone(), 503, message.to_string()).to_payload();
    let _ = nats_client
        .publish(req.reply_subject.clone(), payload)
        .await;
}
