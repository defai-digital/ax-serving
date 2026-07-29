//! Dashboard and license route handlers.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::ServingLayer;

/// `GET /dashboard` — embedded monitoring dashboard (operator-authenticated when configured).
pub async fn dashboard() -> impl IntoResponse {
    axum::response::Html(include_str!("../dashboard.html"))
}

/// `GET /v1/license` — immutable Apache-2.0 license identity.
pub async fn get_license(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    Json(layer.license.to_json())
}
