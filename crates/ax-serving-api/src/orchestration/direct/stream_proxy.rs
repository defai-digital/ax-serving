//! SSE/byte streaming and buffered response construction for worker proxies.
//!
//! Streaming responses are forwarded chunk-by-chunk without full buffering so
//! time-to-first-token is not impacted. First-byte and stream-idle deadlines
//! apply only on the streaming path.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, TryStreamExt as _};
use tracing::error;

use super::attempt::AttemptGuard;
use super::headers::should_forward_worker_header;
use super::metrics::DispatchOutcomeGuard;
use super::{DirectDispatcher, worker_failure_response};

/// Maximum buffered non-streaming worker response body.
pub(super) const MAX_WORKER_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub(super) enum WorkerBodyError {
    TooLarge,
    Read(reqwest::Error),
}

pub(super) fn response_declares_oversize(content_length: Option<u64>, max_bytes: usize) -> bool {
    content_length.is_some_and(|len| len > max_bytes as u64)
}

pub(super) fn append_limited_body_chunk(
    body: &mut Vec<u8>,
    chunk: &[u8],
    max_bytes: usize,
) -> Result<(), WorkerBodyError> {
    let next_len = body
        .len()
        .checked_add(chunk.len())
        .ok_or(WorkerBodyError::TooLarge)?;
    if next_len > max_bytes {
        return Err(WorkerBodyError::TooLarge);
    }
    body.extend_from_slice(chunk);
    Ok(())
}

pub(super) fn add_limited_body_len(
    current_len: usize,
    chunk_len: usize,
    max_bytes: usize,
) -> Result<usize, WorkerBodyError> {
    let next_len = current_len
        .checked_add(chunk_len)
        .ok_or(WorkerBodyError::TooLarge)?;
    if next_len > max_bytes {
        return Err(WorkerBodyError::TooLarge);
    }
    Ok(next_len)
}

async fn read_worker_response_body_limited(
    resp: reqwest::Response,
    max_bytes: usize,
) -> Result<Bytes, WorkerBodyError> {
    if response_declares_oversize(resp.content_length(), max_bytes) {
        return Err(WorkerBodyError::TooLarge);
    }

    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        append_limited_body_chunk(&mut body, &chunk.map_err(WorkerBodyError::Read)?, max_bytes)?;
    }
    Ok(Bytes::from(body))
}

fn response_builder_with_worker_headers(
    status: StatusCode,
    headers: &HeaderMap,
    include_content_length: bool,
) -> axum::http::response::Builder {
    let mut builder = axum::response::Response::builder().status(status);
    for (name, value) in headers {
        if should_forward_worker_header(name, include_content_length) {
            builder = builder.header(name, value);
        }
    }
    if !headers.contains_key(header::CONTENT_TYPE) {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }
    builder
}

/// Drain an error response body so the connection can be reused from the pool.
pub(super) async fn drain_worker_error_response(resp: reqwest::Response, url: &str) {
    if response_declares_oversize(resp.content_length(), MAX_WORKER_RESPONSE_BODY_BYTES) {
        tracing::warn!(
            %url,
            content_length = resp.content_length().unwrap_or_default(),
            limit = MAX_WORKER_RESPONSE_BODY_BYTES,
            "skipping oversized worker error response drain"
        );
        return;
    }

    let mut total_len = 0usize;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                match add_limited_body_len(total_len, chunk.len(), MAX_WORKER_RESPONSE_BODY_BYTES) {
                    Ok(next_len) => total_len = next_len,
                    Err(WorkerBodyError::TooLarge) => {
                        tracing::warn!(
                            %url,
                            limit = MAX_WORKER_RESPONSE_BODY_BYTES,
                            "stopping oversized worker error response drain"
                        );
                        return;
                    }
                    Err(WorkerBodyError::Read(_)) => {
                        unreachable!("length accounting does not read")
                    }
                }
            }
            Err(err) => {
                tracing::warn!(%url, err = %err, "draining worker error response failed");
                return;
            }
        }
    }
}

impl DirectDispatcher {
    /// Build an axum `Response` from a reqwest result.
    ///
    /// For streaming responses the `guard` lives inside the stream and is
    /// dropped when the stream is exhausted or the client disconnects.
    /// Retry decisions must already have been made before calling this method
    /// (no retry after commitment of response headers / stream body).
    pub(super) async fn build_response(
        &self,
        result: reqwest::Result<reqwest::Response>,
        url: String,
        stream: bool,
        guard: AttemptGuard,
        attempt_started: std::time::Instant,
    ) -> Response {
        match result {
            Err(e) => {
                self.metrics
                    .attempt_duration
                    .record(attempt_started.elapsed());
                tracing::warn!(%url, err = %e, "dispatch request failed");
                worker_failure_response(e.to_string())
            }
            Ok(resp) => {
                let status = StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let headers = resp.headers().clone();

                if stream {
                    type GuardedResponseStream =
                        std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>;
                    let byte_stream: GuardedResponseStream =
                        Box::pin(resp.bytes_stream().map_err(std::io::Error::other));
                    let stream_url = url.clone();
                    let first_byte_timeout = self.first_byte_timeout;
                    let stream_idle_timeout = self.stream_idle_timeout;
                    let outcome = Some(DispatchOutcomeGuard::new(
                        Arc::clone(&self.metrics),
                        status.is_success(),
                        attempt_started,
                    ));
                    let guarded = futures::stream::unfold(
                        (byte_stream, Some(guard), outcome, 0usize, false, true),
                        move |(mut inner, guard, mut outcome, total_len, done, first_byte): (
                            GuardedResponseStream,
                            Option<AttemptGuard>,
                            Option<DispatchOutcomeGuard>,
                            usize,
                            bool,
                            bool,
                        )| {
                            let stream_url = stream_url.clone();
                            async move {
                                if done {
                                    drop(guard);
                                    drop(outcome);
                                    return None;
                                }
                                let wait = if first_byte {
                                    first_byte_timeout
                                } else {
                                    stream_idle_timeout
                                };
                                match tokio::time::timeout(wait, inner.next()).await {
                                    Err(_) => {
                                        error!(
                                            url = %stream_url,
                                            timeout_ms = wait.as_millis(),
                                            first_byte,
                                            "worker streaming response deadline expired"
                                        );
                                        drop(guard);
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.failed();
                                        }
                                        Some((
                                            Err(std::io::Error::new(
                                                std::io::ErrorKind::TimedOut,
                                                if first_byte {
                                                    "worker first-byte deadline expired"
                                                } else {
                                                    "worker stream idle deadline expired"
                                                },
                                            )),
                                            (inner, None, None, total_len, true, false),
                                        ))
                                    }
                                    Ok(Some(Ok(chunk))) => {
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.first_byte();
                                        }
                                        match add_limited_body_len(
                                            total_len,
                                            chunk.len(),
                                            MAX_WORKER_RESPONSE_BODY_BYTES,
                                        ) {
                                            Ok(next_len) => Some((
                                                Ok(chunk),
                                                (inner, guard, outcome, next_len, false, false),
                                            )),
                                            Err(WorkerBodyError::TooLarge) => {
                                                error!(
                                                    url = %stream_url,
                                                    limit = MAX_WORKER_RESPONSE_BODY_BYTES,
                                                    "worker streaming response body exceeded size limit"
                                                );
                                                drop(guard);
                                                if let Some(outcome) = outcome.as_mut() {
                                                    outcome.failed();
                                                }
                                                Some((
                                                    Err(std::io::Error::new(
                                                        std::io::ErrorKind::InvalidData,
                                                        "worker streaming response body exceeded size limit",
                                                    )),
                                                    (inner, None, None, total_len, true, false),
                                                ))
                                            }
                                            Err(WorkerBodyError::Read(_)) => {
                                                unreachable!("length accounting does not read")
                                            }
                                        }
                                    }
                                    Ok(Some(Err(err))) => {
                                        drop(guard);
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.failed();
                                        }
                                        Some((
                                            Err(err),
                                            (inner, None, None, total_len, true, false),
                                        ))
                                    }
                                    Ok(None) => {
                                        drop(guard);
                                        if let Some(outcome) = outcome.as_mut() {
                                            outcome.completed();
                                        }
                                        None
                                    }
                                }
                            }
                        },
                    );

                    response_builder_with_worker_headers(status, &headers, false)
                        .body(Body::from_stream(guarded))
                        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                } else {
                    let body_result =
                        read_worker_response_body_limited(resp, MAX_WORKER_RESPONSE_BODY_BYTES)
                            .await;
                    self.metrics
                        .attempt_duration
                        .record(attempt_started.elapsed());
                    match body_result {
                        Err(WorkerBodyError::TooLarge) => {
                            error!(
                                %url,
                                limit = MAX_WORKER_RESPONSE_BODY_BYTES,
                                "worker response body exceeded size limit"
                            );
                            worker_failure_response(format!(
                                "worker response body exceeded {} byte limit",
                                MAX_WORKER_RESPONSE_BODY_BYTES
                            ))
                        }
                        Ok(bytes) => {
                            if status.is_success() {
                                self.metrics
                                    .completed_total
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            response_builder_with_worker_headers(status, &headers, true)
                                .body(Body::from(bytes))
                                .unwrap_or_else(|_| {
                                    StatusCode::INTERNAL_SERVER_ERROR.into_response()
                                })
                        }
                        Err(WorkerBodyError::Read(e)) => {
                            error!(%url, err = %e, "reading worker response body failed");
                            worker_failure_response(e.to_string())
                        }
                    }
                }
            }
        }
    }
}
