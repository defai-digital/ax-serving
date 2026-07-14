//! Portable pure validation of public OpenAI request shape.
//!
//! Shared by the gateway proxy path (and available for embedded adapters).
//! **No** `ax_serving_engine` imports — safe on the default gateway feature graph.
//!
//! # Parity policy
//!
//! Proxy vs embedded surfaces already differ on some status codes and messages.
//! This module encodes the **proxy-facing** pure checks (and model-id trim /
//! charset rules shared via `LogicalModelId`). Callers map [`ShapeError`] into
//! their surface-specific error response types; do not "unify for beauty"
//! without an explicit parity-matrix decision.

use ax_serving_protocol::LogicalModelId;
use axum::http::StatusCode;

use crate::openai_schema::{
    EmbeddingsInput, InputMessage, MAX_CONTENT_BYTES, MAX_EMBEDDING_INPUTS,
    MAX_EMBEDDING_TOTAL_BYTES, MAX_EMBEDDING_TOTAL_TOKENS, MAX_MAX_TOKENS, MAX_MESSAGES,
};

/// Portable shape-validation failure (status + client-visible message).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError {
    pub status: StatusCode,
    pub message: String,
}

impl ShapeError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unprocessable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message: message.into(),
        }
    }
}

impl From<ShapeError> for (StatusCode, String) {
    fn from(error: ShapeError) -> Self {
        (error.status, error.message)
    }
}

/// Validate `max_tokens` is unset, or in `[1, MAX_MAX_TOKENS]`.
pub fn validate_max_tokens(max_tokens: Option<u32>) -> Result<(), ShapeError> {
    if matches!(max_tokens, Some(0)) {
        return Err(ShapeError::bad_request("max_tokens must be >= 1"));
    }
    if matches!(max_tokens, Some(n) if n > MAX_MAX_TOKENS) {
        return Err(ShapeError::bad_request(format!(
            "max_tokens exceeds limit ({MAX_MAX_TOKENS})"
        )));
    }
    Ok(())
}

/// Validate chat `messages` non-empty, under count/byte limits, and content rules.
///
/// Assistant messages with `tool_calls` may omit `content` (proxy allow path).
pub fn validate_chat_messages(messages: &[InputMessage]) -> Result<(), ShapeError> {
    if messages.is_empty() {
        return Err(ShapeError::bad_request("messages must not be empty"));
    }
    if messages.len() > MAX_MESSAGES {
        return Err(ShapeError::bad_request(format!(
            "too many messages (max {MAX_MESSAGES})"
        )));
    }
    for message in messages {
        if message.content.is_none()
            && !(message.role.eq_ignore_ascii_case("assistant") && message.tool_calls.is_some())
        {
            return Err(ShapeError::bad_request(
                "message content is required unless assistant tool_calls are present",
            ));
        }
        if message
            .content
            .as_ref()
            .is_some_and(|content| content.byte_len() > MAX_CONTENT_BYTES)
        {
            return Err(ShapeError::bad_request(
                "message content exceeds 32 KB limit",
            ));
        }
    }
    Ok(())
}

/// Validate completions `prompt` is present, non-empty, and under byte limit.
pub fn validate_prompt(prompt: Option<&str>) -> Result<(), ShapeError> {
    let Some(prompt) = prompt else {
        return Err(ShapeError::bad_request("prompt must not be empty"));
    };
    if prompt.is_empty() {
        return Err(ShapeError::bad_request("prompt must not be empty"));
    }
    if prompt.len() > MAX_CONTENT_BYTES {
        return Err(ShapeError::bad_request("prompt exceeds 32 KB limit"));
    }
    Ok(())
}

/// Validate embeddings `input` variants (text / token sequences, aggregate caps).
pub fn validate_embeddings_input(input: &EmbeddingsInput) -> Result<(), ShapeError> {
    match input {
        EmbeddingsInput::One(text) => validate_embedding_text(text, 0),
        EmbeddingsInput::Many(texts) => {
            if texts.is_empty() {
                return Err(ShapeError::bad_request("input must not be empty"));
            }
            if texts.len() > MAX_EMBEDDING_INPUTS {
                return Err(ShapeError::bad_request(format!(
                    "too many embedding inputs (max {MAX_EMBEDDING_INPUTS})"
                )));
            }
            let mut total_bytes = 0usize;
            for (idx, text) in texts.iter().enumerate() {
                validate_embedding_text(text, idx)?;
                total_bytes = total_bytes.saturating_add(text.len());
            }
            if total_bytes > MAX_EMBEDDING_TOTAL_BYTES {
                return Err(ShapeError::bad_request(format!(
                    "embedding input text exceeds total limit of {MAX_EMBEDDING_TOTAL_BYTES} bytes"
                )));
            }
            Ok(())
        }
        EmbeddingsInput::OneTokens(tokens) => validate_embedding_tokens(tokens, 0),
        EmbeddingsInput::ManyTokens(seqs) => {
            if seqs.is_empty() {
                return Err(ShapeError::bad_request("input must not be empty"));
            }
            if seqs.len() > MAX_EMBEDDING_INPUTS {
                return Err(ShapeError::bad_request(format!(
                    "too many embedding inputs (max {MAX_EMBEDDING_INPUTS})"
                )));
            }
            let mut total_tokens = 0usize;
            for (idx, tokens) in seqs.iter().enumerate() {
                validate_embedding_tokens(tokens, idx)?;
                total_tokens = total_tokens.saturating_add(tokens.len());
            }
            if total_tokens > MAX_EMBEDDING_TOTAL_TOKENS {
                return Err(ShapeError::bad_request(format!(
                    "embedding token input exceeds total limit of {MAX_EMBEDDING_TOTAL_TOKENS}"
                )));
            }
            Ok(())
        }
    }
}

fn validate_embedding_text(text: &str, index: usize) -> Result<(), ShapeError> {
    if text.is_empty() {
        return Err(ShapeError::bad_request(format!(
            "embedding input at index {index} must not be empty"
        )));
    }
    if text.len() > MAX_CONTENT_BYTES {
        return Err(ShapeError::bad_request(format!(
            "embedding input at index {index} exceeds {MAX_CONTENT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_embedding_tokens(tokens: &[u32], index: usize) -> Result<(), ShapeError> {
    if tokens.is_empty() {
        return Err(ShapeError::bad_request(format!(
            "embedding token input at index {index} must not be empty"
        )));
    }
    if tokens.len() > MAX_EMBEDDING_TOTAL_TOKENS {
        return Err(ShapeError::bad_request(format!(
            "embedding token input at index {index} exceeds {MAX_EMBEDDING_TOTAL_TOKENS} tokens"
        )));
    }
    Ok(())
}

/// Validate proxy-facing model identifier (missing / empty / whitespace / charset).
///
/// Preserves proxy surface messages and status codes. Embedded path keeps its
/// own `validate_model_identifier` adapter (parity matrix: surface-specific).
pub fn validate_model_id(model: Option<String>) -> Result<String, ShapeError> {
    let Some(model) = model else {
        return Err(ShapeError::bad_request("missing field: model"));
    };
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err(ShapeError::bad_request("model must not be empty"));
    }
    if model != trimmed {
        return Err(ShapeError::unprocessable(
            "model contains unsupported whitespace",
        ));
    }
    LogicalModelId::new(model.clone())
        .map_err(|error| ShapeError::unprocessable(format!("invalid model identifier: {error}")))?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_schema::MessageContent;

    fn user_msg(content: &str) -> InputMessage {
        InputMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn max_tokens_rejects_zero_and_over_limit() {
        let err = validate_max_tokens(Some(0)).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "max_tokens must be >= 1");

        let err = validate_max_tokens(Some(MAX_MAX_TOKENS + 1)).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains(&MAX_MAX_TOKENS.to_string()));

        assert!(validate_max_tokens(None).is_ok());
        assert!(validate_max_tokens(Some(1)).is_ok());
        assert!(validate_max_tokens(Some(MAX_MAX_TOKENS)).is_ok());
    }

    #[test]
    fn model_id_missing_empty_whitespace_and_charset() {
        let err = validate_model_id(None).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "missing field: model");

        let err = validate_model_id(Some("   ".into())).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "model must not be empty");

        let err = validate_model_id(Some("model ".into())).unwrap_err();
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.message, "model contains unsupported whitespace");

        let err = validate_model_id(Some("bad model".into())).unwrap_err();
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(err.message.starts_with("invalid model identifier:"));

        assert_eq!(
            validate_model_id(Some("public/qwen".into())).unwrap(),
            "public/qwen"
        );
    }

    #[test]
    fn chat_messages_empty_oversize_and_tool_calls_without_content() {
        let err = validate_chat_messages(&[]).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "messages must not be empty");

        let too_many: Vec<_> = (0..=MAX_MESSAGES).map(|_| user_msg("hi")).collect();
        let err = validate_chat_messages(&too_many).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("too many messages"));

        let oversize = vec![user_msg(&"x".repeat(MAX_CONTENT_BYTES + 1))];
        let err = validate_chat_messages(&oversize).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "message content exceeds 32 KB limit");

        let missing_content = vec![InputMessage {
            role: "user".into(),
            content: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let err = validate_chat_messages(&missing_content).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("message content is required"));

        // Proxy allow path: assistant tool_calls without content.
        let assistant_tools = vec![InputMessage {
            role: "assistant".into(),
            content: None,
            name: None,
            tool_calls: Some(serde_json::json!([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"}
            }])),
            tool_call_id: None,
        }];
        assert!(validate_chat_messages(&assistant_tools).is_ok());

        assert!(validate_chat_messages(&[user_msg("hello")]).is_ok());
    }

    #[test]
    fn prompt_rejects_missing_empty_and_oversize() {
        let err = validate_prompt(None).unwrap_err();
        assert_eq!(err.message, "prompt must not be empty");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);

        let err = validate_prompt(Some("")).unwrap_err();
        assert_eq!(err.message, "prompt must not be empty");

        let big = "x".repeat(MAX_CONTENT_BYTES + 1);
        let err = validate_prompt(Some(&big)).unwrap_err();
        assert_eq!(err.message, "prompt exceeds 32 KB limit");

        assert!(validate_prompt(Some("ok")).is_ok());
    }

    #[test]
    fn embeddings_input_empty_and_aggregate_limits() {
        let err = validate_embeddings_input(&EmbeddingsInput::One(String::new())).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("must not be empty"));

        let err = validate_embeddings_input(&EmbeddingsInput::Many(vec![])).unwrap_err();
        assert_eq!(err.message, "input must not be empty");

        let too_many = vec!["x".to_string(); MAX_EMBEDDING_INPUTS + 1];
        let err = validate_embeddings_input(&EmbeddingsInput::Many(too_many)).unwrap_err();
        assert!(err.message.contains("too many embedding inputs"));

        let oversize_one = "a".repeat(MAX_CONTENT_BYTES + 1);
        let err = validate_embeddings_input(&EmbeddingsInput::One(oversize_one)).unwrap_err();
        assert!(err.message.contains("exceeds"));

        // Aggregate text byte cap: MAX_CONTENT_BYTES * MAX_MESSAGES items of max size.
        let aggregate = vec!["a".repeat(MAX_CONTENT_BYTES); MAX_MESSAGES + 1];
        // Each item is at limit (ok individually) but total exceeds MAX_EMBEDDING_TOTAL_BYTES
        // only when we can pack more than MAX_MESSAGES at full size — capped by per-item
        // limit first if single item is over. Use many medium strings that sum over cap.
        if aggregate.len() <= MAX_EMBEDDING_INPUTS {
            let total: usize = aggregate.iter().map(|s| s.len()).sum();
            if total > MAX_EMBEDDING_TOTAL_BYTES {
                let err = validate_embeddings_input(&EmbeddingsInput::Many(aggregate)).unwrap_err();
                assert!(err.message.contains("total limit"));
            }
        }

        let empty_tokens =
            validate_embeddings_input(&EmbeddingsInput::OneTokens(vec![])).unwrap_err();
        assert!(empty_tokens.message.contains("must not be empty"));

        let too_many_tokens = vec![1u32; MAX_EMBEDDING_TOTAL_TOKENS + 1];
        let err =
            validate_embeddings_input(&EmbeddingsInput::OneTokens(too_many_tokens)).unwrap_err();
        assert!(err.message.contains("exceeds"));

        assert!(validate_embeddings_input(&EmbeddingsInput::One("hello".into())).is_ok());
        assert!(
            validate_embeddings_input(&EmbeddingsInput::Many(vec!["a".into(), "b".into()])).is_ok()
        );
        assert!(validate_embeddings_input(&EmbeddingsInput::OneTokens(vec![1, 2, 3])).is_ok());
    }
}
