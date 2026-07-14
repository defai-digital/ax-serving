//! In-process mock worker HTTP servers for orchestration integration tests.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::{Router, routing::post};

/// Spawn a minimal axum mock worker on an ephemeral port.
///
/// Returns `None` if the loopback socket cannot be bound (e.g. in restricted
/// sandbox environments). Tests that receive `None` must skip via
/// `skip_if_no_socket!`.
///
/// Every POST to `/v1/chat/completions` returns the given `status` and `body`.
/// The server runs until the test process exits.
pub async fn spawn_mock_worker(status: u16, body: &'static str) -> Option<SocketAddr> {
    let response = move || async move {
        axum::response::Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .unwrap()
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(response))
        .route("/v1/completions", post(response))
        .route("/v1/embeddings", post(response));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}

pub async fn spawn_not_admitted_worker() -> Option<SocketAddr> {
    let response = || async move {
        axum::response::Response::builder()
            .status(503)
            .header("content-type", "application/json")
            .header("x-ax-admission-state", "not-admitted")
            .body(axum::body::Body::from(
                r#"{"error":{"code":"AXS_WORKER_CAPACITY","retryable":true,"phase":"pre_admission"}}"#,
            ))
            .unwrap()
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(response))
        .route("/v1/completions", post(response))
        .route("/v1/embeddings", post(response));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}

#[derive(Default)]
pub struct EchoWorkerState {
    pub models: Mutex<Vec<String>>,
    pub public_authorization_seen: Mutex<bool>,
}

pub async fn spawn_echo_model_worker() -> Option<(SocketAddr, Arc<EchoWorkerState>)> {
    use axum::Json;
    use axum::extract::State;
    use axum::http::HeaderMap;

    async fn echo(
        State(state): State<Arc<EchoWorkerState>>,
        headers: HeaderMap,
        Json(body): Json<serde_json::Value>,
    ) -> Json<serde_json::Value> {
        let model = body["model"].as_str().unwrap_or_default().to_string();
        state.models.lock().unwrap().push(model.clone());
        if headers.contains_key(axum::http::header::AUTHORIZATION) {
            *state.public_authorization_seen.lock().unwrap() = true;
        }
        Json(serde_json::json!({
            "id": "echo",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]
        }))
    }

    let state = Arc::new(EchoWorkerState::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(echo))
        .with_state(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some((addr, state))
}

/// Spawn a mock worker that returns a streaming SSE response.
///
/// Emits `tokens` chunks. If `drop_after` is `Some(n)`, the response body
/// ends cleanly after `n` chunks (no `[DONE]`), simulating a mid-stream drop.
/// If `drop_after` is `None`, `tokens` chunks are emitted followed by `[DONE]`.
pub async fn spawn_sse_worker(
    tokens: usize,
    drop_after: Option<usize>,
) -> Option<std::net::SocketAddr> {
    let app = Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move || async move {
            let emit_count = drop_after.unwrap_or(tokens);
            let mut chunks: Vec<Result<axum::body::Bytes, std::io::Error>> = (0..emit_count)
                .map(|i| {
                    let s = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"tok{i}\"}}}}]}}\n\n"
                    );
                    Ok(axum::body::Bytes::from(s))
                })
                .collect();
            if drop_after.is_none() {
                chunks.push(Ok(axum::body::Bytes::from("data: [DONE]\n\n")));
            }
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from_stream(futures::stream::iter(chunks)))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}

pub async fn spawn_delayed_first_byte_worker(delay: std::time::Duration) -> Option<SocketAddr> {
    let handler = move || {
        let delay = delay;
        async move {
            let stream = futures::stream::once(async move {
                tokio::time::sleep(delay).await;
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from_static(
                    b"data: {\"choices\":[]}\n\n",
                ))
            });
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from_stream(stream))
                .unwrap()
        }
    };
    let app = Router::new().route("/v1/chat/completions", post(handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}
