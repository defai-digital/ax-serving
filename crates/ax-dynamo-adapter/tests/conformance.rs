use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use ax_serving_adapter_core::proxy::{ProxyConfig, router};
use ax_serving_protocol::{DISPATCH_TOKEN_HEADER, DomainId};
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::State,
    http::{Request, StatusCode, header},
    response::Response,
    routing::post,
};
use tokio::sync::Mutex;
use tower::ServiceExt as _;

const REQUEST_ID: &str = "87f5aa94-4042-43b5-bb45-f36e6b559da6";
const ATTEMPT_ID: &str = "ae7a8560-66cf-4d4a-962a-570021e71fca";

#[derive(Clone, Default)]
struct UpstreamState {
    bodies: Arc<Mutex<Vec<Bytes>>>,
    calls: Arc<AtomicUsize>,
}

async fn spawn_upstream() -> (String, UpstreamState) {
    let state = UpstreamState::default();
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(
                |State(state): State<UpstreamState>, body: Bytes| async move {
                    state.calls.fetch_add(1, Ordering::AcqRel);
                    state.bodies.lock().await.push(body);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            r#"{"id":"exact","choices":[{"message":{"content":"ok"}}]}"#,
                        ))
                        .unwrap()
                },
            ),
        )
        .route(
            "/v1/completions",
            post(|| async {
                Response::builder()
                    .status(StatusCode::SERVICE_UNAVAILABLE)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"error":{"message":"ambiguous upstream failure"}}"#,
                    ))
                    .unwrap()
            }),
        )
        .route(
            "/v1/embeddings",
            post(|| async {
                let chunks = futures::stream::iter([
                    Ok::<_, std::io::Error>(Bytes::from_static(b"data: {\"index\":0}\n")),
                    Ok(Bytes::from_static(b"\ndata: [DONE]\n\n")),
                ]);
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from_stream(chunks))
                    .unwrap()
            }),
        )
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), state)
}

fn adapter(upstream_url: String, ready: bool) -> Router {
    router(
        ProxyConfig {
            upstream_url,
            upstream_health_path: "/v1/models".into(),
            dispatch_token: Some("dispatch-secret".into()),
            max_inflight: 8,
            expected_domain_id: Some(DomainId::new("nvidia-pc-main").unwrap()),
            require_dispatch_identity: true,
        },
        reqwest::Client::new(),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicBool::new(false)),
        Some(Arc::new(AtomicBool::new(ready))),
    )
}

fn request(path: &str, body: &'static str, domain: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .header(DISPATCH_TOKEN_HEADER, "dispatch-secret")
        .header("x-ax-request-id", REQUEST_ID)
        .header("x-ax-attempt-id", ATTEMPT_ID)
        .header("x-ax-domain-id", domain)
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test]
async fn preserves_rewritten_openai_body_and_unknown_fields() {
    let (upstream, state) = spawn_upstream().await;
    let body = r#"{"model":"runtime-model","messages":[],"vendor_extension":{"x":1}}"#;
    let response = adapter(upstream, true)
        .oneshot(request("/v1/chat/completions", body, "nvidia-pc-main"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(state.bodies.lock().await.as_slice(), &[Bytes::from(body)]);
}

#[tokio::test]
async fn rejects_wrong_domain_before_contacting_dynamo() {
    let (upstream, state) = spawn_upstream().await;
    let response = adapter(upstream, true)
        .oneshot(request("/v1/chat/completions", "{}", "nvidia-thor-lab"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(state.calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn generic_dynamo_failure_is_never_typed_not_admitted() {
    let (upstream, _) = spawn_upstream().await;
    let response = adapter(upstream, true)
        .oneshot(request("/v1/completions", "{}", "nvidia-pc-main"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(!response.headers().contains_key("x-ax-admission-state"));
}

#[tokio::test]
async fn preserves_fragmented_sse_bytes() {
    let (upstream, _) = spawn_upstream().await;
    let response = adapter(upstream, true)
        .oneshot(request("/v1/embeddings", "{}", "nvidia-pc-main"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024).await.unwrap();
    assert_eq!(
        body,
        Bytes::from_static(b"data: {\"index\":0}\n\ndata: [DONE]\n\n")
    );
}

#[tokio::test]
async fn not_ready_domain_rejects_before_contacting_dynamo() {
    let (upstream, state) = spawn_upstream().await;
    let response = adapter(upstream, false)
        .oneshot(request("/v1/chat/completions", "{}", "nvidia-pc-main"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()["x-ax-admission-state"], "not-admitted");
    assert_eq!(state.calls.load(Ordering::Acquire), 0);
}
