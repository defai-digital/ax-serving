//! Model management route handlers.

use std::sync::Arc;

use ax_serving_engine::{BackendType, LoadConfig};
use axum::Json;
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::routes::unix_now;
use super::schema::*;
use super::validation::validate_model_identifier;
use crate::ServingLayer;
use crate::auth::RequestId;
use crate::registry::RegistryError;
use crate::utils::request_meta::audit_actor;

/// GET /v1/models
pub async fn list_models(State(layer): State<Arc<ServingLayer>>) -> Json<ModelsResponse> {
    let ids = layer.registry.list_ids();
    let now = unix_now();
    Json(ModelsResponse {
        object: "list",
        data: ids
            .into_iter()
            .map(|id| ModelEntry {
                id,
                object: "model",
                created: now,
                owned_by: "ax-serving",
            })
            .collect(),
    })
}

/// POST /v1/models — load a model from a GGUF file.
pub async fn rest_load_model(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    if let Some(response) = validate_model_identifier(&req.model_id, "model_id") {
        return response;
    }

    let pooling_type = match req.pooling_type.as_deref() {
        Some(raw) => match normalize_pooling_type(raw) {
            Some(v) => Some(v),
            None => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(serde_json::json!({
                        "error": "invalid pooling_type; expected one of: none, mean, cls, last, rank"
                    })),
                )
                    .into_response();
            }
        },
        None => None,
    };

    let path = std::path::PathBuf::from(&req.path);

    // Resolve backend hint: explicit `backend` field takes priority; then `mlx: true`
    // flag; then default to "auto" (routing config decides).
    let raw_hint = if let Some(ref b) = req.backend {
        b.to_ascii_lowercase()
    } else if req.mlx == Some(true) {
        "mlx".to_string()
    } else {
        "auto".to_string()
    };
    let backend_hint = match raw_hint.as_str() {
        "native" | "llama_cpp" | "mlx" | "lib_llama" | "auto" => raw_hint,
        other => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "error": format!(
                        "invalid backend; expected native, llama_cpp, mlx, lib_llama, or auto but got {other}"
                    )
                })),
            )
                .into_response();
        }
    };
    let config = LoadConfig {
        context_length: req.context_length.unwrap_or(0),
        backend_type: BackendType::Auto,
        llama_cpp_n_gpu_layers: req.n_gpu_layers,
        mmproj_path: req.mmproj_path.clone(),
        backend_hint: Some(backend_hint),
        enable_embeddings: req.enable_embeddings,
        pooling_type,
    };
    let model_id = req.model_id.clone();
    let layer_for_load = Arc::clone(&layer);

    let result = tokio::task::spawn_blocking(move || {
        layer_for_load
            .registry
            .load(&model_id, &path, config, layer_for_load.backend.as_ref())
    })
    .await;

    match result {
        Ok(Ok(entry)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_load",
                "model",
                Some(entry.id.clone()),
                "ok",
                Some(serde_json::json!({
                    "path": entry.path.display().to_string(),
                    "architecture": entry.metadata.architecture,
                })),
            );
            let (ready, model_available, loaded_model_count) = lifecycle_snapshot(&layer);
            (
                StatusCode::CREATED,
                Json(LoadModelResponse {
                    model_id: entry.id.clone(),
                    state: "loaded",
                    ready,
                    model_available,
                    loaded_model_count,
                    architecture: entry.metadata.architecture.clone(),
                    context_length: entry.metadata.context_length,
                    load_time_ms: entry.metadata.load_time_ms,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_load",
                "model",
                Some(req.model_id.clone()),
                "error",
                Some(serde_json::json!({
                    "path": req.path,
                    "error": e.to_string(),
                })),
            );
            let status = match e.downcast_ref::<RegistryError>() {
                Some(RegistryError::AlreadyLoaded(_)) => StatusCode::CONFLICT,
                Some(
                    RegistryError::FileNotFound(_)
                    | RegistryError::InvalidFormat(_)
                    | RegistryError::InvalidModelId(_),
                ) => StatusCode::UNPROCESSABLE_ENTITY,
                Some(RegistryError::PathNotAllowed(_)) => StatusCode::FORBIDDEN,
                Some(RegistryError::CapacityExceeded(_)) => StatusCode::SERVICE_UNAVAILABLE,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, Json(serde_json::json!({"error": e.to_string()}))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

pub(crate) fn is_valid_pooling_type(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "none" | "mean" | "cls" | "last" | "rank"
    )
}

fn normalize_pooling_type(s: &str) -> Option<String> {
    let canonical = s.trim().to_ascii_lowercase();
    if is_valid_pooling_type(&canonical) {
        Some(canonical)
    } else {
        None
    }
}

fn lifecycle_snapshot(layer: &Arc<ServingLayer>) -> (bool, bool, usize) {
    let loaded_model_count = layer.registry.list_ids().len();
    let model_available = loaded_model_count > 0;
    let ready = !matches!(
        layer.backend.thermal_state(),
        ax_serving_engine::ThermalState::Critical
    );
    (ready, model_available, loaded_model_count)
}

/// DELETE /v1/models/:id — unload a loaded model.
pub async fn rest_unload_model(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(model_id): Path<String>,
) -> Response {
    if let Some(response) = validate_model_identifier(&model_id, "model_id") {
        return response;
    }

    let id_for_response = model_id.clone();
    let layer_for_unload = Arc::clone(&layer);
    let result = tokio::task::spawn_blocking(move || {
        layer_for_unload
            .registry
            .unload(&model_id, layer_for_unload.backend.as_ref())
    })
    .await;

    match result {
        Ok(Ok(())) => {
            layer.per_model_scheduler.remove(&id_for_response);
            layer.audit.record(
                audit_actor(req_id),
                "model_unload",
                "model",
                Some(id_for_response.clone()),
                "ok",
                None,
            );
            let (ready, model_available, loaded_model_count) = lifecycle_snapshot(&layer);
            (
                StatusCode::OK,
                Json(UnloadModelResponse {
                    model_id: id_for_response,
                    state: "unloaded",
                    ready,
                    model_available,
                    loaded_model_count,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_unload",
                "model",
                Some(id_for_response.clone()),
                "error",
                Some(serde_json::json!({ "error": e.to_string() })),
            );
            let status = match e.downcast_ref::<RegistryError>() {
                Some(RegistryError::NotLoaded(_)) => StatusCode::NOT_FOUND,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, Json(serde_json::json!({"error": e.to_string()}))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// POST /v1/models/:id/reload — atomically reload from the same path and config.
pub async fn rest_reload_model(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(model_id): Path<String>,
) -> Response {
    if let Some(response) = validate_model_identifier(&model_id, "model_id") {
        return response;
    }

    let layer_for_reload = Arc::clone(&layer);
    let reload_id = model_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        layer_for_reload
            .registry
            .reload(&reload_id, layer_for_reload.backend.as_ref())
    })
    .await;

    match result {
        Ok(Ok(entry)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_reload",
                "model",
                Some(entry.id.clone()),
                "ok",
                Some(serde_json::json!({
                    "path": entry.path.display().to_string(),
                    "architecture": entry.metadata.architecture,
                })),
            );
            let (ready, model_available, loaded_model_count) = lifecycle_snapshot(&layer);
            (
                StatusCode::OK,
                Json(ReloadModelResponse {
                    model_id: entry.id.clone(),
                    state: "loaded",
                    ready,
                    model_available,
                    loaded_model_count,
                    architecture: entry.metadata.architecture.clone(),
                    load_time_ms: entry.metadata.load_time_ms,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_reload",
                "model",
                Some(model_id.clone()),
                "error",
                Some(serde_json::json!({ "error": e.to_string() })),
            );
            let status = match e.downcast_ref::<RegistryError>() {
                Some(RegistryError::NotLoaded(_)) => StatusCode::NOT_FOUND,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, Json(serde_json::json!({"error": e.to_string()}))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
