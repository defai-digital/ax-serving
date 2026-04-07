//! Dashboard and license route handlers.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::ServingLayer;
use crate::auth::RequestId;
use crate::utils::request_meta::audit_actor;

/// `GET /dashboard` — embedded monitoring dashboard (no auth required).
pub async fn dashboard() -> impl IntoResponse {
    axum::response::Html(include_str!("../dashboard.html"))
}

/// `GET /v1/license` — current license state (no auth required).
pub async fn get_license(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    Json(layer.license.to_json())
}

/// `POST /v1/license` — activate a license key (no auth required).
///
/// Body: `{"key": "<license-key>"}`
pub async fn set_license(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let Some(key) = body.get("key").and_then(|v| v.as_str()) else {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "missing field: key"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "missing field: key"})),
        )
            .into_response();
    };
    let key = key.trim().to_string();
    if key.is_empty() {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "key must not be empty"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "key must not be empty"})),
        )
            .into_response();
    }
    match layer.license.set_key(key) {
        Ok(()) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "ok",
                None,
            );
            Json(layer.license.to_json()).into_response()
        }
        Err(e) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "error",
                Some(serde_json::json!({"error": e.to_string()})),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}
