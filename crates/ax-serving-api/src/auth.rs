//! API key authentication middleware and security response headers.

use std::collections::HashSet;
use std::sync::Arc;

use ax_serving_protocol::RequestId as ProtocolRequestId;
use axum::extract::Request;
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

/// Request correlation ID inserted by middleware for downstream handlers.
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// Gateway-owned request identity used by the worker protocol and dispatch attempts.
///
/// This ID is always generated at the trust boundary. A caller-provided
/// `X-Request-ID` remains available as a correlation value but cannot choose
/// the internal request identity used for retries and fencing.
#[derive(Clone, Copy, Debug)]
pub struct AxRequestId(pub ProtocolRequestId);

/// Load API keys from the `AXS_API_KEY` environment variable (comma-separated).
///
/// Returns an empty set when `AXS_API_KEY` is unset or empty.  The caller
/// (`start_servers`) enforces that an empty set is only permitted when
/// `AXS_ALLOW_NO_AUTH=true` is explicitly set.
pub fn load_api_keys() -> Arc<HashSet<String>> {
    load_key_set("AXS_API_KEY")
}

/// Load operator credentials. Public client credentials are intentionally not
/// accepted by the gateway's admin and worker-management routes.
pub fn load_admin_api_keys() -> Arc<HashSet<String>> {
    load_key_set("AXS_ADMIN_API_KEY")
}

fn load_key_set(name: &str) -> Arc<HashSet<String>> {
    let keys: HashSet<String> = std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    Arc::new(keys)
}

#[derive(Clone)]
pub struct GatewayAuthState {
    pub public_keys: Arc<HashSet<String>>,
    pub admin_keys: Arc<HashSet<String>>,
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

pub(crate) fn has_valid_api_key(candidate: &str, keys: &HashSet<String>) -> bool {
    keys.iter()
        .any(|expected| constant_time_eq_str(candidate, expected))
}

/// Extract a bearer token from an Authorization header value.
///
/// HTTP authentication schemes are case-insensitive. Require at least one
/// whitespace separator between the scheme and token.
pub(crate) fn bearer_token_from_authorization(value: &str) -> Option<&str> {
    let mut parts = value.trim_start().splitn(2, char::is_whitespace);
    let scheme = parts.next()?;
    let token = parts.next()?.trim();
    if scheme.eq_ignore_ascii_case("Bearer") && !token.is_empty() {
        Some(token)
    } else {
        None
    }
}

/// Returns `true` if the given path is exempt from authentication.
///
/// Minimal load-balancer probes remain unauthenticated. Metrics, diagnostics,
/// and dashboards are operator surfaces and use the independent admin key.
///
/// Only read-only license state (`GET /v1/license`) is exempt; mutating
/// endpoints (`POST /v1/license`, `DELETE /v1/workers/{id}`) require auth.
fn is_exempt(method: &Method, path: &str) -> bool {
    path == "/health"
        || path == "/livez"
        || path == "/readyz"
        || path == "/routablez"
        || (*method == Method::GET && path == "/v1/license")
}

fn is_admin_path(method: &Method, path: &str) -> bool {
    path.starts_with("/admin/v1/")
        || path == "/admin/v1/deployments"
        || path.starts_with("/v1/admin/")
        || path == "/v1/workers"
        || path.starts_with("/v1/workers/")
        || path == "/metrics"
        || path == "/v1/metrics"
        || path == "/dashboard"
        || (path == "/v1/license" && *method != Method::GET)
}

pub async fn gateway_auth_middleware(
    axum::extract::State(state): axum::extract::State<GatewayAuthState>,
    request: Request,
    next: Next,
) -> Response {
    if is_exempt(request.method(), request.uri().path()) {
        return next.run(request).await;
    }
    let admin_path = is_admin_path(request.method(), request.uri().path());
    let required_keys = if admin_path {
        &state.admin_keys
    } else {
        &state.public_keys
    };
    if required_keys.is_empty()
        && (!admin_path || state.public_keys.is_empty() && state.admin_keys.is_empty())
    {
        return next.run(request).await;
    }
    let authorized = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(bearer_token_from_authorization)
        .is_some_and(|candidate| has_valid_api_key(candidate.trim(), required_keys));
    if authorized {
        return next.run(request).await;
    }

    let request_id = request
        .extensions()
        .get::<AxRequestId>()
        .map(|value| value.0)
        .unwrap_or_default();
    let mut response = crate::orchestration::error::ax_error_response(
        StatusCode::UNAUTHORIZED,
        request_id,
        if admin_path {
            "AXS_ADMIN_UNAUTHORIZED"
        } else {
            "AXS_UNAUTHORIZED"
        },
        "missing or invalid credential",
        false,
        ax_serving_protocol::AdmissionPhase::Authentication,
    );
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
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
        .and_then(bearer_token_from_authorization)
        .map(|key| has_valid_api_key(key.trim(), &keys))
        .unwrap_or(false);

    if authorized {
        next.run(request).await
    } else {
        let request_id = request
            .extensions()
            .get::<AxRequestId>()
            .map(|value| value.0)
            .unwrap_or_default();
        let mut resp = crate::orchestration::error::ax_error_response(
            StatusCode::UNAUTHORIZED,
            request_id,
            "AXS_UNAUTHORIZED",
            "missing or invalid API key",
            false,
            ax_serving_protocol::AdmissionPhase::Authentication,
        );
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

    let ax_request_id = ProtocolRequestId::new();
    let mut request = request;
    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));
    request.extensions_mut().insert(AxRequestId(ax_request_id));

    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    if let Ok(v) = HeaderValue::from_str(&request_id) {
        headers.insert("x-request-id", v);
    } else {
        headers.insert("x-request-id", HeaderValue::from_static("unknown"));
    }
    if let Ok(value) = HeaderValue::from_str(&ax_request_id.to_string()) {
        headers.insert(ax_serving_protocol::REQUEST_ID_HEADER, value);
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
        assert!(is_exempt(&Method::GET, "/livez"));
        assert!(is_exempt(&Method::GET, "/readyz"));
        assert!(is_exempt(&Method::GET, "/routablez"));
        assert!(!is_exempt(&Method::GET, "/metrics"));
        assert!(!is_exempt(&Method::GET, "/v1/metrics"));
        assert!(!is_exempt(&Method::GET, "/dashboard"));
        assert!(is_exempt(&Method::GET, "/v1/license"));
        assert!(!is_exempt(&Method::POST, "/v1/license"));
        assert!(!is_exempt(&Method::DELETE, "/v1/workers/abc-123"));
        assert!(!is_exempt(&Method::GET, "/v1/workers"));
        assert!(!is_exempt(&Method::GET, "/v1/chat/completions"));
        assert!(!is_exempt(&Method::GET, "/v1/models"));
        assert!(!is_exempt(&Method::GET, "/health/extra"));
    }

    #[test]
    fn bearer_token_parser_accepts_case_insensitive_scheme_and_whitespace() {
        let cases = [
            ("Bearer secret", Some("secret")),
            ("bearer secret", Some("secret")),
            ("BEARER secret", Some("secret")),
            ("Bearer  secret", Some("secret")),
            ("Bearer\tsecret", Some("secret")),
            ("Bearer secret ", Some("secret")),
            (" Bearer secret", Some("secret")),
            ("Token secret", None),
            ("Bearer", None),
            ("Bearer ", None),
            ("Bearersecret", None),
            ("secret", None),
        ];
        let mut keys = std::collections::HashSet::new();
        keys.insert("secret".to_string());
        for (header, expected_key) in cases {
            let result = bearer_token_from_authorization(header);
            assert_eq!(result, expected_key, "header: {header:?}");
            if let Some(k) = result {
                assert!(
                    has_valid_api_key(k, &keys),
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

    fn gateway_auth_app(state: GatewayAuthState) -> axum::Router {
        axum::Router::new()
            .route("/v1/models", axum::routing::get(|| async { "models" }))
            .route(
                "/admin/v1/deployments",
                axum::routing::get(|| async { "admin" }),
            )
            .route("/metrics", axum::routing::get(|| async { "metrics" }))
            .layer(axum::middleware::from_fn_with_state(
                state,
                gateway_auth_middleware,
            ))
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
    async fn auth_middleware_accepts_lowercase_bearer_scheme() {
        use tower::ServiceExt;
        let app = auth_app(make_keys(&["correct-key"]));
        let req = axum::http::Request::builder()
            .uri("/v1/models")
            .header("Authorization", "bearer correct-key")
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
    async fn gateway_metrics_require_admin_key_not_public_key() {
        use tower::ServiceExt;

        let app = gateway_auth_app(GatewayAuthState {
            public_keys: make_keys(&["public-key"]),
            admin_keys: make_keys(&["admin-key"]),
        });
        let public_response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/metrics")
                    .header("Authorization", "Bearer public-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(public_response.status(), StatusCode::UNAUTHORIZED);

        let admin_response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/metrics")
                    .header("Authorization", "Bearer admin-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(admin_response.status(), StatusCode::OK);
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

    #[tokio::test]
    async fn gateway_auth_separates_public_and_admin_credentials() {
        use tower::ServiceExt;

        let app = gateway_auth_app(GatewayAuthState {
            public_keys: make_keys(&["public-key"]),
            admin_keys: make_keys(&["admin-key"]),
        });
        let public_models = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/models")
                    .header("authorization", "Bearer public-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(public_models.status(), StatusCode::OK);

        let public_admin = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/admin/v1/deployments")
                    .header("authorization", "Bearer public-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(public_admin.status(), StatusCode::UNAUTHORIZED);

        let admin = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/admin/v1/deployments")
                    .header("authorization", "Bearer admin-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(admin.status(), StatusCode::OK);
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
