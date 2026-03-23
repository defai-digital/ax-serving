//! API key authentication middleware and security response headers.

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::Request;
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use uuid::Uuid;

/// Request correlation ID inserted by middleware for downstream handlers.
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// Load API keys from the `AXS_API_KEY` environment variable (comma-separated).
///
/// Returns an empty set when `AXS_API_KEY` is unset or empty.  The caller
/// (`start_servers`) enforces that an empty set is only permitted when
/// `AXS_ALLOW_NO_AUTH=true` is explicitly set.
pub fn load_api_keys() -> Arc<HashSet<String>> {
    let keys: HashSet<String> = std::env::var("AXS_API_KEY")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    Arc::new(keys)
}

/// Constant-time string equality helper for secret comparisons.
///
/// Compares over the maximum length and folds length mismatch into the diff,
/// avoiding early-return timing leaks from byte-wise equality.
pub fn constant_time_eq_str(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut diff = a_bytes.len() ^ b_bytes.len();
    let max = a_bytes.len().max(b_bytes.len());

    for i in 0..max {
        let av = if i < a_bytes.len() { a_bytes[i] } else { 0 };
        let bv = if i < b_bytes.len() { b_bytes[i] } else { 0 };
        diff |= (av ^ bv) as usize;
    }

    diff == 0
}

fn has_valid_api_key(candidate: &str, keys: &HashSet<String>) -> bool {
    keys.iter()
        .any(|expected| constant_time_eq_str(candidate, expected))
}

/// Returns `true` if the given path is exempt from authentication.
///
/// Monitoring endpoints remain unauthenticated so Prometheus scrapers,
/// load-balancer health probes, and the embedded dashboard work without
/// Bearer tokens.  `/v1/metrics` is also exempt: the dashboard fetches it
/// on every poll tick and it is the JSON counterpart of the already-exempt
/// Prometheus `/metrics` endpoint.
///
/// Only read-only license state (`GET /v1/license`) is exempt; mutating
/// endpoints (`POST /v1/license`, `DELETE /v1/workers/{id}`) require auth.
fn is_exempt(method: &Method, path: &str) -> bool {
    path == "/health"
        || path == "/metrics"
        || path == "/v1/metrics"
        || path == "/dashboard"
        || (*method == Method::GET && path == "/v1/license")
}

/// Axum middleware: validates `Authorization: Bearer <key>` on every request.
///
/// Skips auth when:
/// - `AXS_API_KEY` was not set at startup and `AXS_ALLOW_NO_AUTH=true` is set
///   (key set is empty — explicitly opted-in development mode).
/// - The request targets an exempt path (`/health`, `/metrics`).
///
/// Returns HTTP 401 with a JSON error body on missing or invalid credentials.
pub async fn auth_middleware(
    axum::extract::State(keys): axum::extract::State<Arc<HashSet<String>>>,
    request: Request,
    next: Next,
) -> Response {
    if keys.is_empty() {
        return next.run(request).await;
    }

    if is_exempt(request.method(), request.uri().path()) {
        return next.run(request).await;
    }

    let authorized = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|key| has_valid_api_key(key.trim(), &keys))
        .unwrap_or(false);

    if authorized {
        next.run(request).await
    } else {
        // RFC 7235 §4.1: a 401 response MUST include WWW-Authenticate so
        // conforming clients know the expected authentication scheme.
        let mut resp = (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({
                "error": "unauthorized: missing or invalid API key"
            })),
        )
            .into_response();
        resp.headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        resp
    }
}

/// Axum middleware: generates/propagates `X-Request-ID` and adds security headers.
///
/// - Forwards the client's `X-Request-ID` if present; otherwise generates a UUID v4.
/// - Adds `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY` to every response.
pub async fn request_id_and_headers_middleware(request: Request, next: Next) -> Response {
    let request_id = request
        .headers()
        .get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let mut request = request;
    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));

    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    if let Ok(v) = HeaderValue::from_str(&request_id) {
        headers.insert("x-request-id", v);
    } else {
        headers.insert("x-request-id", HeaderValue::from_static("unknown"));
    }
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exempt_paths_match_exactly() {
        assert!(is_exempt(&Method::GET, "/health"));
        assert!(is_exempt(&Method::GET, "/metrics"));
        assert!(is_exempt(&Method::GET, "/v1/metrics"));
        assert!(is_exempt(&Method::GET, "/dashboard"));
        assert!(is_exempt(&Method::GET, "/v1/license"));
        assert!(!is_exempt(&Method::POST, "/v1/license"));
        assert!(!is_exempt(&Method::DELETE, "/v1/workers/abc-123"));
        assert!(!is_exempt(&Method::GET, "/v1/workers"));
        assert!(!is_exempt(&Method::GET, "/v1/chat/completions"));
        assert!(!is_exempt(&Method::GET, "/v1/models"));
        assert!(!is_exempt(&Method::GET, "/health/extra"));
    }

    #[test]
    fn bearer_token_trim_after_prefix_strip() {
        // Simulate what auth_middleware does: strip prefix then trim.
        let cases = [
            ("Bearer secret", Some("secret")),   // normal
            ("Bearer  secret", Some("secret")),  // double space after Bearer
            ("Bearer secret ", Some("secret")),  // trailing space on token
            ("Bearer  secret ", Some("secret")), // both
            ("Token secret", None),              // wrong scheme
            ("secret", None),                    // no scheme
        ];
        let mut keys = std::collections::HashSet::new();
        keys.insert("secret".to_string());
        for (header, expected_key) in cases {
            let result = header.strip_prefix("Bearer ").map(|k| k.trim().to_string());
            assert_eq!(result.as_deref(), expected_key, "header: {header:?}");
            if let Some(k) = result {
                assert!(
                    has_valid_api_key(&k, &keys),
                    "trimmed key should be found in set"
                );
            }
        }
    }

    #[test]
    fn constant_time_eq_str_works() {
        assert!(constant_time_eq_str("abc", "abc"));
        assert!(!constant_time_eq_str("abc", "abd"));
        assert!(!constant_time_eq_str("abc", "ab"));
        assert!(!constant_time_eq_str("", "x"));
    }

    #[test]
    fn load_api_keys_filters_empty_segments() {
        // Simulate a value with leading/trailing commas and spaces.
        // We can't set env vars safely in parallel tests, so test the parsing logic
        // by re-implementing it inline.
        let raw = " key1 , , key2 ,";
        let keys: HashSet<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains("key1"));
        assert!(keys.contains("key2"));
    }

    // ── auth_middleware tests via minimal router ───────────────────────────────

    fn make_keys(raw: &[&str]) -> Arc<HashSet<String>> {
        Arc::new(raw.iter().map(|s| s.to_string()).collect())
    }

    /// Build a minimal one-route app layered with auth_middleware.
    fn auth_app(keys: Arc<HashSet<String>>) -> axum::Router {
        axum::Router::new()
            .route("/v1/models", axum::routing::get(|| async { "ok" }))
            .route("/health", axum::routing::get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(keys, auth_middleware))
    }

    #[tokio::test]
    async fn auth_middleware_empty_key_set_allows_all() {
        use tower::ServiceExt;
        let app = auth_app(make_keys(&[]));
        let req = axum::http::Request::builder()
            .uri("/v1/models")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_middleware_valid_key_returns_200() {
        use tower::ServiceExt;
        let app = auth_app(make_keys(&["correct-key"]));
        let req = axum::http::Request::builder()
            .uri("/v1/models")
            .header("Authorization", "Bearer correct-key")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_middleware_missing_header_returns_401() {
        use tower::ServiceExt;
        let app = auth_app(make_keys(&["secret"]));
        let req = axum::http::Request::builder()
            .uri("/v1/models")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        // RFC 7235 §4.1 requires WWW-Authenticate on every 401.
        assert_eq!(
            resp.headers()
                .get(header::WWW_AUTHENTICATE)
                .expect("WWW-Authenticate header must be present"),
            "Bearer"
        );
    }

    #[tokio::test]
    async fn auth_middleware_wrong_key_returns_401_with_www_authenticate() {
        use tower::ServiceExt;
        let app = auth_app(make_keys(&["correct-key"]));
        let req = axum::http::Request::builder()
            .uri("/v1/models")
            .header("Authorization", "Bearer wrong-key")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(resp.headers().contains_key(header::WWW_AUTHENTICATE));
    }

    #[tokio::test]
    async fn auth_middleware_exempt_health_bypasses_key_check() {
        use tower::ServiceExt;
        // Even with a required key, /health is always accessible.
        let app = auth_app(make_keys(&["secret"]));
        let req = axum::http::Request::builder()
            .uri("/health")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── request_id_and_headers_middleware tests ────────────────────────────────

    fn headers_app() -> axum::Router {
        axum::Router::new()
            .route("/test", axum::routing::get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(request_id_and_headers_middleware))
    }

    #[tokio::test]
    async fn security_headers_always_added_to_response() {
        use tower::ServiceExt;
        let app = headers_app();
        let req = axum::http::Request::builder()
            .uri("/test")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.headers()
                .get("x-content-type-options")
                .and_then(|v| v.to_str().ok()),
            Some("nosniff")
        );
        assert_eq!(
            resp.headers()
                .get("x-frame-options")
                .and_then(|v| v.to_str().ok()),
            Some("DENY")
        );
        assert!(
            resp.headers().contains_key("x-request-id"),
            "x-request-id header must be present"
        );
    }

    #[tokio::test]
    async fn request_id_forwarded_when_client_provides_header() {
        use tower::ServiceExt;
        let app = headers_app();
        let req = axum::http::Request::builder()
            .uri("/test")
            .header("X-Request-ID", "my-correlation-id-42")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok()),
            Some("my-correlation-id-42"),
            "client-provided X-Request-ID must be echoed back"
        );
    }

    #[tokio::test]
    async fn request_id_generated_as_uuid_when_absent() {
        use tower::ServiceExt;
        let app = headers_app();
        let req = axum::http::Request::builder()
            .uri("/test")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id must be set");
        assert!(
            Uuid::parse_str(id).is_ok(),
            "generated X-Request-ID must be a valid UUID: {id}"
        );
    }
}
