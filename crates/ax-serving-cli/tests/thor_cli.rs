use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

#[derive(Default)]
struct MockServerState {
    actions: Mutex<Vec<String>>,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thor_status_require_ready_exits_with_registration_mismatch_code() -> Result<()> {
    let (base_url, listen_addr, server_task) =
        spawn_mock_server("[]", Arc::new(MockServerState::default())).await?;
    let config_path = temp_path("thor-status", "env");
    let worker_id_path = temp_path("thor-status-worker", "id");
    std::fs::write(&worker_id_path, "worker-missing\n")
        .with_context(|| format!("failed to write {}", worker_id_path.display()))?;
    write_thor_env(&config_path, &base_url, listen_addr, &worker_id_path)?;

    let config_arg = config_path.to_str().unwrap().to_string();
    let output = tokio::task::spawn_blocking(move || {
        Command::new(ax_serving_bin())
            .args(["thor", "status", "--config", &config_arg, "--require-ready"])
            .output()
    })
    .await
    .context("thor status task join failed")?
    .context("failed to run ax-serving thor status")?;

    server_task.abort();

    assert_eq!(output.status.code(), Some(24));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("overall_state=registration_mismatch"));
    assert!(stderr.contains("thor status overall_state=registration_mismatch"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thor_wait_ready_honors_timeout_without_sleeping_full_poll_interval() -> Result<()> {
    let (base_url, listen_addr, server_task) =
        spawn_mock_server("[]", Arc::new(MockServerState::default())).await?;
    let config_path = temp_path("thor-wait-ready", "env");
    let worker_id_path = temp_path("thor-wait-ready-worker", "id");
    write_thor_env(&config_path, &base_url, listen_addr, &worker_id_path)?;

    let started_at = Instant::now();
    let config_arg = config_path.to_str().unwrap().to_string();
    let output = tokio::task::spawn_blocking(move || {
        Command::new(ax_serving_bin())
            .args([
                "thor",
                "wait-ready",
                "--config",
                &config_arg,
                "--timeout-secs",
                "1",
                "--poll-interval-ms",
                "5000",
            ])
            .output()
    })
    .await
    .context("thor wait-ready task join failed")?
    .context("failed to run ax-serving thor wait-ready")?;
    let elapsed = started_at.elapsed();

    server_task.abort();

    assert_eq!(output.status.code(), Some(24));
    assert!(
        elapsed < Duration::from_secs(5),
        "wait-ready overslept timeout: {elapsed:?}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("overall_state=registration_mismatch"));
    assert!(stderr.contains("thor wait-ready timed out after 1s"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thor_drain_complete_when_idle_calls_drain_and_drain_complete() -> Result<()> {
    let state = Arc::new(MockServerState::default());
    let worker_list = r#"[{"id":"worker-1","addr":"127.0.0.1:18081","backend":"sglang","health":"healthy","drain":false}]"#;
    let (base_url, listen_addr, server_task) =
        spawn_mock_server(worker_list, state.clone()).await?;
    let config_path = temp_path("thor-drain", "env");
    let worker_id_path = temp_path("thor-drain-worker", "id");
    std::fs::write(&worker_id_path, "worker-1\n")
        .with_context(|| format!("failed to write {}", worker_id_path.display()))?;
    write_thor_env(&config_path, &base_url, listen_addr, &worker_id_path)?;

    let config_arg = config_path.to_str().unwrap().to_string();
    let output = tokio::task::spawn_blocking(move || {
        Command::new(ax_serving_bin())
            .args([
                "thor",
                "drain",
                "--config",
                &config_arg,
                "--complete-when-idle",
                "--idle-timeout-secs",
                "1",
            ])
            .output()
    })
    .await
    .context("thor drain task join failed")?
    .context("failed to run ax-serving thor drain")?;

    server_task.abort();

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("thor drain requested for worker_id=worker-1"));
    assert!(stdout.contains("thor drain-complete sent for worker_id=worker-1"));

    let actions = state.actions.lock().await;
    assert_eq!(
        actions.as_slice(),
        [
            "POST /internal/workers/worker-1/drain",
            "POST /internal/workers/worker-1/drain-complete",
        ]
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thor_status_require_ready_detects_live_agent_capability_mismatch() -> Result<()> {
    let state = Arc::new(MockServerState::default());
    let worker_list = r#"[{"id":"worker-1","addr":"127.0.0.1:18081","backend":"sglang","health":"healthy","drain":false}]"#;
    let health_json = r#"{"status":"ok","backend":"sglang","capabilities":{"llm":true,"embedding":false,"vision":false},"max_context":null,"model_ids":["qwen2-72b"],"inflight":0,"queue_depth":0}"#;
    let (base_url, listen_addr, server_task) =
        spawn_mock_server_with_health(worker_list, health_json, state).await?;
    let config_path = temp_path("thor-status-config-drift", "env");
    let worker_id_path = temp_path("thor-status-config-drift-worker", "id");
    std::fs::write(&worker_id_path, "worker-1\n")
        .with_context(|| format!("failed to write {}", worker_id_path.display()))?;
    write_thor_env(&config_path, &base_url, listen_addr, &worker_id_path)?;

    let config_arg = config_path.to_str().unwrap().to_string();
    let output = tokio::task::spawn_blocking(move || {
        Command::new(ax_serving_bin())
            .args(["thor", "status", "--config", &config_arg, "--require-ready"])
            .output()
    })
    .await
    .context("thor status config-drift task join failed")?
    .context("failed to run ax-serving thor status for config drift")?;

    server_task.abort();

    assert_eq!(output.status.code(), Some(24));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("thor agent config: capability_mismatch"));
    assert!(stdout.contains("overall_state=registration_mismatch"));
    assert!(stderr.contains("thor status overall_state=registration_mismatch"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thor_status_require_ready_detects_live_agent_max_context_mismatch() -> Result<()> {
    let state = Arc::new(MockServerState::default());
    let worker_list = r#"[{"id":"worker-1","addr":"127.0.0.1:18081","backend":"sglang","health":"healthy","drain":false}]"#;
    let health_json = r#"{"status":"ok","backend":"sglang","capabilities":{"llm":true,"embedding":true,"vision":false},"max_context":16384,"model_ids":["qwen2-72b"],"inflight":0,"queue_depth":0}"#;
    let (base_url, listen_addr, server_task) =
        spawn_mock_server_with_health(worker_list, health_json, state).await?;
    let config_path = temp_path("thor-status-max-context-drift", "env");
    let worker_id_path = temp_path("thor-status-max-context-drift-worker", "id");
    std::fs::write(&worker_id_path, "worker-1\n")
        .with_context(|| format!("failed to write {}", worker_id_path.display()))?;
    write_thor_env(&config_path, &base_url, listen_addr, &worker_id_path)?;
    append_to_env(&config_path, "AXS_THOR_MAX_CONTEXT=32768\n")?;

    let config_arg = config_path.to_str().unwrap().to_string();
    let output = tokio::task::spawn_blocking(move || {
        Command::new(ax_serving_bin())
            .args(["thor", "status", "--config", &config_arg, "--require-ready"])
            .output()
    })
    .await
    .context("thor status max-context drift task join failed")?
    .context("failed to run ax-serving thor status for max-context drift")?;

    server_task.abort();

    assert_eq!(output.status.code(), Some(24));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("thor agent config: max_context_mismatch"));
    assert!(stdout.contains("overall_state=registration_mismatch"));
    assert!(stderr.contains("thor status overall_state=registration_mismatch"));

    Ok(())
}

fn ax_serving_bin() -> &'static str {
    env!("CARGO_BIN_EXE_ax-serving")
}

fn temp_path(prefix: &str, ext: &str) -> PathBuf {
    let unique = format!(
        "{prefix}-{}-{}.{ext}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos()
    );
    std::env::temp_dir().join(unique)
}

fn write_thor_env(
    path: &Path,
    base_url: &str,
    listen_addr: std::net::SocketAddr,
    worker_id_path: &Path,
) -> Result<()> {
    let contents = format!(
        "\
AXS_CONTROL_PLANE_URL={base_url}
AXS_THOR_BACKEND=sglang
AXS_THOR_RUNTIME_URL={base_url}
AXS_THOR_LISTEN_ADDR={listen_addr}
AXS_THOR_ADVERTISED_ADDR={listen_addr}
AXS_THOR_WORKER_ID_PATH={}
",
        worker_id_path.display()
    );
    std::fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn append_to_env(path: &Path, content: &str) -> Result<()> {
    use std::io::Write;

    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {} for append", path.display()))?;
    file.write_all(content.as_bytes())
        .with_context(|| format!("failed to append {}", path.display()))
}

async fn spawn_mock_server(
    worker_list_json: &str,
    state: Arc<MockServerState>,
) -> Result<(String, std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    spawn_mock_server_with_health(
        worker_list_json,
        r#"{"status":"ok","backend":"sglang","capabilities":{"llm":true,"embedding":true,"vision":false},"max_context":null,"model_ids":["qwen2-72b"],"inflight":0,"queue_depth":0}"#,
        state,
    )
    .await
}

async fn spawn_mock_server_with_health(
    worker_list_json: &str,
    health_json: &str,
    state: Arc<MockServerState>,
) -> Result<(String, std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind mock thor server")?;
    let addr = listener
        .local_addr()
        .context("mock thor server missing local addr")?;
    let worker_list_json = worker_list_json.to_string();
    let health_json = health_json.to_string();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let worker_list_json = worker_list_json.clone();
            let health_json = health_json.clone();
            let state = state.clone();
            tokio::spawn(async move {
                let _ = serve_connection(stream, &worker_list_json, &health_json, state).await;
            });
        }
    });
    Ok((format!("http://{addr}"), addr, task))
}

async fn serve_connection(
    mut stream: TcpStream,
    worker_list_json: &str,
    health_json: &str,
    state: Arc<MockServerState>,
) -> Result<()> {
    let mut buf = [0u8; 4096];
    let mut request = Vec::new();
    loop {
        let read = stream
            .read(&mut buf)
            .await
            .context("failed to read mock request")?;
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buf[..read]);
        if request.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    let request = String::from_utf8_lossy(&request);
    let request_line = request.lines().next().unwrap_or("GET / HTTP/1.1");
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or("GET");
    let path = request_parts.next().unwrap_or("/");
    if path.starts_with("/internal/workers/") && method == "POST" {
        state.actions.lock().await.push(format!("{method} {path}"));
    }
    let (status, content_type, body) = match path {
        "/health" => ("200 OK", "application/json", health_json.to_string()),
        "/v1/models" => ("200 OK", "application/json", r#"{"data":[]}"#.to_string()),
        "/internal/workers" => ("200 OK", "application/json", worker_list_json.to_string()),
        p if p.ends_with("/drain") => ("200 OK", "application/json", r#"{"ok":true}"#.to_string()),
        p if p.ends_with("/drain-complete") => {
            ("200 OK", "application/json", r#"{"ok":true}"#.to_string())
        }
        _ => ("404 Not Found", "text/plain", "not found".to_string()),
    };
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .await
        .context("failed to write mock response")?;
    stream
        .flush()
        .await
        .context("failed to flush mock response")?;
    Ok(())
}
