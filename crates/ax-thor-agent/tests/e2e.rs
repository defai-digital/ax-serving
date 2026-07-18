use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use ax_serving_protocol::{DomainId, QualificationState};
use ax_thor_agent::agent::{self, SharedRuntime};
use ax_thor_agent::config::{ExecutionDomainConfig, ThorConfig};
use ax_thor_agent::proxy;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode, header},
    routing::{get, post},
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
    authorization_headers: Mutex<Vec<Option<String>>>,
}

#[tokio::test]
async fn thor_agent_registers_heartbeats_and_proxies_chat() -> Result<()> {
    let control_state = Arc::new(ControlPlaneState::default());
    let sglang_state = Arc::new(SgLangState::default());

    let (control_base, _control_task) =
        spawn_server(control_plane_router(control_state.clone())).await?;
    let (sglang_base, _sglang_task) = spawn_server(sglang_router(sglang_state.clone())).await?;

    let config = ThorConfig {
        control_plane_url: control_base.clone(),
        worker_token: Some("secret".into()),
        runtime_url: sglang_base.clone(),
        runtime_api_key: None,
        dispatch_token: Some("gateway-dispatch-secret".into()),
        tls_profile: "loopback_dev".into(),
        runtime: "sglang".into(),
        runtime_version: "test".into(),
        worker_id: "worker-test".into(),
        trust_domain: "test".into(),
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        advertised_url: "http://127.0.0.1:18081".into(),
        max_inflight: 8,
        worker_pool: Some("thor".into()),
        node_class: "thor".into(),
        hardware_class: "thor".into(),
        execution_domain: Some(ExecutionDomainConfig {
            id: DomainId::new("thor-sglang-compat").unwrap(),
            qualification: QualificationState::Experimental,
            compatibility_manifest: None,
        }),
        friendly_name: Some("thor-01".into()),
        chip_model: Some("RTX".into()),
        shutdown_timeout_secs: None,
        max_context: None,
        embedding: None,
        vision: None,
        model_identity: Default::default(),
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("failed to build reqwest client")?;

    let runtime = SharedRuntime::new();
    let registration = agent::register(&client, &client, &config, runtime.instance_id).await?;
    {
        *runtime.models.write().await = registration.models;
        *runtime.session.write().await = Some(registration.session);
    }

    let registrations = control_state.registrations.lock().await;
    assert_eq!(registrations.len(), 1);
    assert_eq!(registrations[0]["protocol"]["version"]["major"], 1);
    assert_eq!(registrations[0]["worker"]["id"], "worker-test");
    assert_eq!(registrations[0]["runtime"]["kind"], "sglang");
    assert_eq!(
        registrations[0]["domain"]["kind"],
        "compatibility_runtime_endpoint"
    );
    assert_eq!(registrations[0]["domain"]["id"], "thor-sglang-compat");
    assert_eq!(registrations[0]["domain_observation"]["ready"], true);
    assert_eq!(registrations[0]["hardware"]["hardware_class"], "thor");
    assert_eq!(
        registrations[0]["worker"]["advertise_url"],
        "http://127.0.0.1:18081"
    );
    assert_eq!(
        registrations[0]["observation"]["models"][0]["runtime_model_id"],
        json!("qwen2-72b")
    );
    assert_eq!(
        registrations[0]["observation"]["models"][0]["identity"]["quantization"],
        json!("awq")
    );
    assert_eq!(
        registrations[0]["observation"]["models"][0]["operations"],
        json!(["chat_completions", "text_completions"])
    );
    assert_eq!(
        registrations[0]["observation"]["models"][0]["capabilities"],
        json!(["inference.vision"])
    );
    drop(registrations);

    let heartbeat_task = tokio::spawn(agent::heartbeat_loop(
        client.clone(),
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
    assert_eq!(heartbeats[0].0, "worker-test");
    assert_eq!(
        heartbeats[0].1["models"][0]["runtime_model_id"],
        json!("qwen2-72b")
    );
    assert_eq!(heartbeats[0].1["capacity"]["active_requests"], json!(4));
    assert_eq!(heartbeats[0].1["capacity"]["waiting_requests"], json!(3));
    assert_eq!(
        heartbeats[0].1["capacity"]["generated_tokens_per_second"],
        json!(42.5)
    );
    assert_eq!(heartbeats[0].1["capacity"]["ttft_ewma_ms"], json!(118.0));
    assert_eq!(heartbeats[0].1["runtime"]["ready"], json!(true));
    assert_eq!(heartbeats[0].1["runtime"]["state"], json!("ready"));
    assert_eq!(heartbeats[0].1["domain_observation"]["ready"], true);
    assert_eq!(
        heartbeats[0].1["domain_observation"]["aggregate_capacity"]["waiting_requests"],
        json!(3)
    );
    drop(heartbeats);

    let (proxy_base, _proxy_task) = spawn_server(proxy::router(
        &config,
        client.clone(),
        runtime.inflight.clone(),
        runtime.draining.clone(),
    ))
    .await?;

    let response: Value = client
        .post(format!("{proxy_base}/v1/chat/completions"))
        .header(header::AUTHORIZATION, "Bearer public-client-secret")
        .header("x-ax-dispatch-token", "gateway-dispatch-secret")
        .header("x-ax-request-id", uuid::Uuid::new_v4().to_string())
        .header("x-ax-attempt-id", uuid::Uuid::new_v4().to_string())
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
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "hello from sglang"
    );

    let chats = sglang_state.chats.lock().await;
    assert_eq!(chats.len(), 1);
    assert_eq!(chats[0]["model"], "qwen2-72b");
    drop(chats);
    assert_eq!(
        sglang_state.authorization_headers.lock().await.as_slice(),
        [None],
        "public client credentials must terminate before the runtime trust zone"
    );

    let health: Value = client
        .get(format!("{proxy_base}/health"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(health["agent_live"], json!(true));
    assert_eq!(health["runtime_ready"], json!(true));

    agent::drain(&client, &config, &runtime).await?;
    agent::drain_complete(&client, &config, &runtime).await?;

    let drains = control_state.drains.lock().await;
    assert_eq!(drains.as_slice(), ["worker-test"]);
    drop(drains);
    let drain_completes = control_state.drain_completes.lock().await;
    assert_eq!(drain_completes.as_slice(), ["worker-test"]);

    heartbeat_task.abort();
    Ok(())
}

#[tokio::test]
async fn thor_proxy_rejects_oversized_runtime_content_length_without_buffering() -> Result<()> {
    let (runtime_base, _runtime_task) = spawn_raw_oversized_runtime_response().await?;

    let config = ThorConfig {
        control_plane_url: "http://127.0.0.1:1".into(),
        worker_token: None,
        runtime_url: runtime_base,
        runtime_api_key: None,
        dispatch_token: None,
        tls_profile: "loopback_dev".into(),
        runtime: "sglang".into(),
        runtime_version: "test".into(),
        worker_id: "worker-test".into(),
        trust_domain: "test".into(),
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        advertised_url: "http://127.0.0.1:18081".into(),
        max_inflight: 8,
        worker_pool: None,
        node_class: "thor".into(),
        hardware_class: "thor".into(),
        execution_domain: None,
        friendly_name: None,
        chip_model: None,
        shutdown_timeout_secs: None,
        max_context: None,
        embedding: None,
        vision: None,
        model_identity: Default::default(),
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .context("failed to build reqwest client")?;
    let runtime = SharedRuntime::new();
    let (proxy_base, _proxy_task) = spawn_server(proxy::router(
        &config,
        client.clone(),
        runtime.inflight.clone(),
        runtime.draining.clone(),
    ))
    .await?;

    let response = client
        .post(format!("{proxy_base}/v1/chat/completions"))
        .json(&json!({
            "model": "qwen2-72b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .context("failed to call thor proxy")?;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response
        .text()
        .await
        .context("failed to read proxy error response")?;
    assert!(body.contains("exceeded 64 MiB limit"));

    Ok(())
}

#[tokio::test]
async fn thor_proxy_accepts_valid_body_above_axum_default_limit() -> Result<()> {
    let sglang_state = Arc::new(SgLangState::default());
    let (sglang_base, _sglang_task) = spawn_server(sglang_router(sglang_state.clone())).await?;

    let config = ThorConfig {
        control_plane_url: "http://127.0.0.1:1".into(),
        worker_token: None,
        runtime_url: sglang_base,
        runtime_api_key: None,
        dispatch_token: None,
        tls_profile: "loopback_dev".into(),
        runtime: "sglang".into(),
        runtime_version: "test".into(),
        worker_id: "worker-test".into(),
        trust_domain: "test".into(),
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        advertised_url: "http://127.0.0.1:18081".into(),
        max_inflight: 8,
        worker_pool: None,
        node_class: "thor".into(),
        hardware_class: "thor".into(),
        execution_domain: None,
        friendly_name: None,
        chip_model: None,
        shutdown_timeout_secs: None,
        max_context: None,
        embedding: None,
        vision: None,
        model_identity: Default::default(),
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("failed to build reqwest client")?;
    let runtime = SharedRuntime::new();
    let (proxy_base, _proxy_task) = spawn_server(proxy::router(
        &config,
        client.clone(),
        runtime.inflight.clone(),
        runtime.draining.clone(),
    ))
    .await?;

    let large_content = "a".repeat(2 * 1024 * 1024 + 4096);
    let body = serde_json::to_vec(&json!({
        "model": "qwen2-72b",
        "messages": [{"role": "user", "content": large_content}],
        "stream": false
    }))
    .context("failed to serialize large request")?;

    let response = client
        .post(format!("{proxy_base}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .context("failed to call thor proxy")?;

    assert_eq!(response.status(), StatusCode::OK);
    let chats = sglang_state.chats.lock().await;
    assert_eq!(chats.len(), 1);
    assert_eq!(
        chats[0]["messages"][0]["content"].as_str().unwrap().len(),
        2 * 1024 * 1024 + 4096
    );

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
            get(|| async {
                Json(json!({
                    "data": [
                        {
                            "id": "qwen2-72b",
                            "max_model_len": 32768,
                            "quantization": "awq",
                            "model_format": "safetensors",
                            "modalities": ["text", "vision"],
                            "supported_operations": ["llm", "vision"]
                        }
                    ]
                }))
            }),
        )
        .route("/metrics", get(runtime_metrics))
        .route("/v1/chat/completions", post(handle_chat))
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .with_state(state)
}

async fn runtime_metrics() -> &'static str {
    r#"
ax_runtime_active_sequences 4
ax_runtime_queue_depth 3
ax_runtime_decode_tok_per_sec 42.5
ax_runtime_ttft_p95_ms 118
ax_runtime_error_rate 0.02
ax_runtime_kv_pages_used 12
ax_runtime_kv_pages_total 128
"#
}

async fn handle_register(
    State(state): State<Arc<ControlPlaneState>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let protocol = body["protocol"].clone();
    state.registrations.lock().await.push(body);
    (
        StatusCode::OK,
        Json(json!({
            "registration_id": "9fe2f234-591f-43df-a900-cfd68e5600bd",
            "lease_token": "0123456789abcdef-test-lease",
            "protocol": protocol,
            "heartbeat_interval_ms": 1000,
            "lease_ttl_ms": 15000,
            "inventory_resync": false
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
    headers: HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, Json<Value>), StatusCode> {
    let parsed: Value = serde_json::from_slice(&body).map_err(|_| StatusCode::BAD_REQUEST)?;
    state.chats.lock().await.push(parsed);
    state.authorization_headers.lock().await.push(
        headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned),
    );
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
        axum::serve(listener, app)
            .await
            .expect("test server failed");
    });
    Ok((format!("http://{}", display_addr(addr)), handle))
}

async fn spawn_raw_oversized_runtime_response() -> Result<(String, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind raw runtime listener")?;
    let addr = listener.local_addr().context("missing listener addr")?;
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept raw runtime request");
        let mut request = [0_u8; 1024];
        let _ = socket.read(&mut request).await.expect("read proxy request");
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
            64 * 1024 * 1024 + 1
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write oversized response headers");
        tokio::time::sleep(Duration::from_secs(2)).await;
    });
    Ok((format!("http://{}", display_addr(addr)), handle))
}

fn display_addr(addr: SocketAddr) -> String {
    match addr {
        SocketAddr::V4(_) => addr.to_string(),
        SocketAddr::V6(_) => format!("[{}]:{}", addr.ip(), addr.port()),
    }
}
