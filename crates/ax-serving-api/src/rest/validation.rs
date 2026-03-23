//! Request validation utilities.

use ax_serving_engine::GenerationParams;
use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::schema::{MAX_MAX_TOKENS, MAX_MODEL_ID_BYTES, ResponseFormat};

/// Validate a model identifier field (e.g., `model` or `model_id`).
///
/// Returns `Some(Response)` when the ID is empty/whitespace-only, exceeds
/// the API limit, or contains unsupported characters.
pub fn validate_model_identifier(model: &str, field_name: &str) -> Option<Response> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("{field_name} must not be empty")})),
            )
                .into_response(),
        );
    }

    if model != trimmed {
        return Some(
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "error": format!("{field_name} contains unsupported whitespace")
                })),
            )
                .into_response(),
        );
    }
    if model.chars().count() > MAX_MODEL_ID_BYTES {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("{field_name} exceeds max length of {MAX_MODEL_ID_BYTES}")
                })),
            )
                .into_response(),
        );
    }
    if !model
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Some(
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "error": format!("{field_name} must be alphanumeric with '-', '_', or '.'")
                })),
            )
                .into_response(),
        );
    }
    None
}

/// Validate sampling parameters common to chat and text completion requests.
///
/// Returns `Some(Response)` with 422 Unprocessable Entity if any parameter is out of range.
/// Returns `None` when all parameters are within acceptable bounds.
#[allow(clippy::too_many_arguments)]
pub fn validate_sampling_params(
    temperature: f32,
    top_p: f32,
    min_p: Option<f32>,
    top_k: Option<u32>,
    repeat_penalty: f32,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
    mirostat: Option<u8>,
) -> Option<Response> {
    macro_rules! reject {
        ($msg:expr) => {
            return Some(
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(serde_json::json!({"error": $msg})),
                )
                    .into_response(),
            )
        };
    }
    if !(0.0..=2.0).contains(&temperature) {
        reject!("temperature must be in [0, 2]");
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        reject!("top_p must be in (0, 1]");
    }
    if let Some(mp) = min_p
        && !(0.0..=1.0).contains(&mp)
    {
        reject!("min_p must be in [0, 1]");
    }
    if matches!(top_k, Some(0)) {
        reject!("top_k must be > 0");
    }
    if !(repeat_penalty > 0.0 && repeat_penalty <= 10.0) {
        reject!("repeat_penalty must be in (0, 10]");
    }
    if let Some(fp) = frequency_penalty
        && !(-2.0..=2.0).contains(&fp)
    {
        reject!("frequency_penalty must be in [-2, 2]");
    }
    if let Some(pp) = presence_penalty
        && !(-2.0..=2.0).contains(&pp)
    {
        reject!("presence_penalty must be in [-2, 2]");
    }
    if matches!(top_logprobs, Some(n) if n > 20) {
        reject!("top_logprobs must be <= 20");
    }
    if top_logprobs.is_some() && logprobs != Some(true) {
        reject!("top_logprobs requires logprobs=true");
    }
    if matches!(mirostat, Some(m) if m > 2) {
        reject!("mirostat must be 0, 1, or 2");
    }
    None
}

/// Validate max_tokens parameter.
///
/// Returns `Some(Response)` with 400 Bad Request if max_tokens is invalid.
/// Returns `None` when max_tokens is valid or None.
pub fn validate_max_tokens(max_tokens: Option<u32>) -> Option<Response> {
    if matches!(max_tokens, Some(0)) {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "max_tokens must be >= 1"})),
            )
                .into_response(),
        );
    }
    if matches!(max_tokens, Some(n) if n > MAX_MAX_TOKENS) {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("max_tokens exceeds limit ({MAX_MAX_TOKENS})")
                })),
            )
                .into_response(),
        );
    }
    None
}

/// Resolve grammar: explicit `grammar` string wins; `response_format: json_object`
/// maps to the `"__json__"` sentinel recognised by all backends.
pub fn resolve_grammar(
    grammar: Option<String>,
    response_format: Option<&ResponseFormat>,
) -> Option<String> {
    grammar.or_else(|| {
        response_format
            .filter(|rf| rf.format_type == "json_object")
            .map(|_| "__json__".to_string())
    })
}

/// Compute effective logprobs flags from request fields.
///
/// Returns `(req_logprobs, req_top_logprobs)` where `req_top_logprobs` is
/// non-zero only when `logprobs` is true.
pub fn resolve_logprobs(logprobs: Option<bool>, top_logprobs: Option<u32>) -> (bool, u32) {
    let enabled = logprobs.unwrap_or(false);
    let top = if enabled {
        top_logprobs.unwrap_or(0)
    } else {
        0
    };
    (enabled, top)
}

/// Build [`GenerationParams`] from sampling fields shared by both
/// chat-completions and text-completions requests.
///
/// `tools` and `tool_choice` default to `None`; the chat handler sets them
/// after calling this function.
#[allow(clippy::too_many_arguments)]
pub fn build_generation_params(
    stream: bool,
    temperature: f32,
    top_p: f32,
    min_p: Option<f32>,
    top_k: Option<u32>,
    effective_max_tokens: Option<u32>,
    stop_seqs: Vec<String>,
    seed: Option<u64>,
    repeat_penalty: f32,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    grammar: Option<String>,
    response_format: Option<&ResponseFormat>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    req_logprobs: bool,
    req_top_logprobs: u32,
) -> GenerationParams {
    GenerationParams {
        stream,
        temperature: if temperature == 0.0 {
            None
        } else {
            Some(temperature as f64)
        },
        top_p: Some(top_p as f64),
        min_p: min_p.map(|v| v as f64),
        top_k: top_k.map(|k| k as usize),
        max_tokens: effective_max_tokens.map(|n| n as usize),
        stop_seqs,
        seed,
        repeat_penalty: Some(repeat_penalty as f64),
        frequency_penalty: frequency_penalty.map(|v| v as f64),
        presence_penalty: presence_penalty.map(|v| v as f64),
        grammar,
        response_format: response_format.map(|rf| rf.format_type.clone()),
        mirostat,
        mirostat_tau: mirostat_tau.map(|v| v as f64),
        mirostat_eta: mirostat_eta.map(|v| v as f64),
        logprobs: if req_logprobs { Some(true) } else { None },
        top_logprobs: if req_top_logprobs > 0 {
            Some(req_top_logprobs)
        } else {
            None
        },
        tools: None,
        tool_choice: None,
    }
}

/// Build a 400 Bad Request response for an invalid `cache_ttl` value.
pub fn cache_ttl_err(e: impl std::fmt::Display) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": format!("invalid cache_ttl: {e}")})),
    )
        .into_response()
}

/// Validate response format parameter.
///
/// Returns `Some(Response)` with 422 Unprocessable Entity if format is invalid.
/// Returns `None` when response_format is valid or None.
pub fn validate_response_format(response_format: Option<&ResponseFormat>) -> Option<Response> {
    let response_format = response_format?;
    match response_format.format_type.as_str() {
        "text" | "json_object" => None,
        other => Some(
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "error": format!(
                        "invalid response_format.type '{other}'; expected 'text' or 'json_object'"
                    )
                })),
            )
                .into_response(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::validate_model_identifier;

    #[test]
    fn validate_model_identifier_rejects_empty_or_whitespace() {
        let resp = validate_model_identifier("   ", "model").unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_model_identifier_rejects_disallowed_characters() {
        let resp = validate_model_identifier("bad model", "model").unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn validate_model_identifier_rejects_trailing_whitespace() {
        let resp = validate_model_identifier("model ", "model").unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn validate_model_identifier_rejects_too_long() {
        let resp = validate_model_identifier(&"a".repeat(129), "model").unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
