use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use ax_thor_agent::agent::{self, SharedRuntime};
use ax_thor_agent::config::ThorConfig;
use ax_thor_agent::proxy;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
};
use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify};

#[derive(Default)]
struct ControlPlaneState {
    registrations: Mutex<Vec<Value>>,
    heartbeats: Mutex<Vec<(String, Value)>>,
    drains: Mutex<Vec<String>>,
    drain_completes: Mutex<Vec<String>>,
    heartbeat_notify: Notify,
}

#[derive(Default)]
struct SgLangState {
    chats: Mutex<Vec<Value>>,
}

#[tokio::test]
async fn thor_agent_registers_heartbeats_and_proxies_chat() -> Result<()> {
    let control_state = Arc::new(ControlPlaneState::default());
    let sglang_state = Arc::new(SgLangState::default());

    let (control_base, _control_task) = spawn_server(control_plane_router(control_state.clone())).await?;
    let (sglang_base, _sglang_task) = spawn_server(sglang_router(sglang_state.clone())).await?;

    let config = ThorConfig {
        control_plane_url: control_base.clone(),
        worker_token: Some("secret".into()),
        sglang_url: sglang_base.clone(),
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        advertised_addr: "127.0.0.1:18081".parse().unwrap(),
        max_inflight: 8,
        worker_pool: Some("thor".into()),
        node_class: "thor".into(),
        friendly_name: Some("thor-01".into()),
        chip_model: Some("RTX".into()),
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("failed to build reqwest client")?;

    let runtime = SharedRuntime::new();
    let registration = agent::register(&client, &config).await?;
    {
        *runtime.models.write().await = registration.models;
        *runtime.session.write().await = Some(registration.session);
    }

    let registrations = control_state.registrations.lock().await;
    assert_eq!(registrations.len(), 1);
    assert_eq!(registrations[0]["backend"], "sglang");
    assert_eq!(registrations[0]["addr"], "127.0.0.1:18081");
    assert_eq!(registrations[0]["capabilities"]["models"], json!(["qwen2-72b"]));
    drop(registrations);

    let heartbeat_task = tokio::spawn(agent::heartbeat_loop(
        client.clone(),
        config.clone(),
        runtime.clone(),
    ));

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if !control_state.heartbeats.lock().await.is_empty() {
                break;
            }
            control_state.heartbeat_notify.notified().await;
        }
    })
    .await
    .context("timed out waiting for thor heartbeat")?;

    let heartbeats = control_state.heartbeats.lock().await;
    assert!(!heartbeats.is_empty());
    assert_eq!(heartbeats[0].0, "worker-1");
    assert_eq!(heartbeats[0].1["model_ids"], json!(["qwen2-72b"]));
    drop(heartbeats);

    let (proxy_base, _proxy_task) = spawn_server(proxy::router(
        &config,
        client.clone(),
        runtime.inflight.clone(),
    ))
    .await?;

    let response: Value = client
        .post(format!("{proxy_base}/v1/chat/completions"))
        .json(&json!({
            "model": "qwen2-72b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .context("failed to call thor proxy")?
        .error_for_status()
        .context("thor proxy returned error")?
        .json()
        .await
        .context("failed to parse thor proxy response")?;
    assert_eq!(response["model"], "qwen2-72b");
    assert_eq!(response["choices"][0]["message"]["content"], "hello from sglang");

    let chats = sglang_state.chats.lock().await;
    assert_eq!(chats.len(), 1);
    assert_eq!(chats[0]["model"], "qwen2-72b");
    drop(chats);

    agent::drain(&client, &config, &runtime).await?;
    agent::drain_complete(&client, &config, &runtime).await?;

    let drains = control_state.drains.lock().await;
    assert_eq!(drains.as_slice(), ["worker-1"]);
    drop(drains);
    let drain_completes = control_state.drain_completes.lock().await;
    assert_eq!(drain_completes.as_slice(), ["worker-1"]);

    heartbeat_task.abort();
    Ok(())
}

fn control_plane_router(state: Arc<ControlPlaneState>) -> Router {
    Router::new()
        .route("/internal/workers/register", post(handle_register))
        .route("/internal/workers/{id}/heartbeat", post(handle_heartbeat))
        .route("/internal/workers/{id}/drain", post(handle_drain))
        .route(
            "/internal/workers/{id}/drain-complete",
            post(handle_drain_complete),
        )
        .with_state(state)
}

fn sglang_router(state: Arc<SgLangState>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data": [{"id": "qwen2-72b"}]})) }),
        )
        .route("/v1/chat/completions", post(handle_chat))
        .with_state(state)
}

async fn handle_register(
    State(state): State<Arc<ControlPlaneState>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    state.registrations.lock().await.push(body);
    (
        StatusCode::OK,
        Json(json!({
            "worker_id": "worker-1",
            "heartbeat_interval_ms": 25
        })),
    )
}

async fn handle_heartbeat(
    State(state): State<Arc<ControlPlaneState>>,
    Path(worker_id): Path<String>,
    Json(body): Json<Value>,
) -> StatusCode {
    state.heartbeats.lock().await.push((worker_id, body));
    state.heartbeat_notify.notify_waiters();
    StatusCode::OK
}

async fn handle_drain(
    State(state): State<Arc<ControlPlaneState>>,
    Path(worker_id): Path<String>,
) -> StatusCode {
    state.drains.lock().await.push(worker_id);
    StatusCode::OK
}

async fn handle_drain_complete(
    State(state): State<Arc<ControlPlaneState>>,
    Path(worker_id): Path<String>,
) -> StatusCode {
    state.drain_completes.lock().await.push(worker_id);
    StatusCode::OK
}

async fn handle_chat(
    State(state): State<Arc<SgLangState>>,
    body: Bytes,
) -> Result<(StatusCode, Json<Value>), StatusCode> {
    let parsed: Value = serde_json::from_slice(&body).map_err(|_| StatusCode::BAD_REQUEST)?;
    state.chats.lock().await.push(parsed);
    Ok((
        StatusCode::OK,
        Json(json!({
            "id": "chatcmpl-thor",
            "object": "chat.completion",
            "model": "qwen2-72b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello from sglang"},
                "finish_reason": "stop"
            }]
        })),
    ))
}

async fn spawn_server(app: Router) -> Result<(String, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind test listener")?;
    let addr = listener.local_addr().context("missing listener addr")?;
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("test server failed");
    });
    Ok((format!("http://{}", display_addr(addr)), handle))
}

fn display_addr(addr: SocketAddr) -> String {
    match addr {
        SocketAddr::V4(_) => addr.to_string(),
        SocketAddr::V6(_) => format!("[{}]:{}", addr.ip(), addr.port()),
    }
}
