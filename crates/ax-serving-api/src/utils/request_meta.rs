//! Shared request-shaping helpers used by both the public REST surface and the
//! multi-worker orchestrator.

use axum::extract::Extension;

use crate::auth::RequestId;
use crate::rest::schema::InputMessage;

/// Characters per token for the heuristic fallback estimator.
const CHARS_PER_TOKEN: u64 = 4;
/// Tokens added per chat message for role/separator framing.
const MESSAGE_FRAMING_TOKENS: u64 = 4;

fn estimated_tokens_from_text(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    chars.saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

/// Estimate prompt tokens for a chat-style request.
pub fn estimate_chat_prompt_tokens_u64(messages: &[InputMessage]) -> u64 {
    messages
        .iter()
        .map(|msg| {
            let role_tokens = estimated_tokens_from_text(&msg.role);
            let name_tokens = msg
                .name
                .as_deref()
                .map(estimated_tokens_from_text)
                .unwrap_or(0);
            let content_tokens = estimated_tokens_from_text(&msg.content.as_text());
            role_tokens
                .saturating_add(name_tokens)
                .saturating_add(content_tokens)
                .saturating_add(MESSAGE_FRAMING_TOKENS)
        })
        .sum::<u64>()
        .max(1)
}

/// Estimate prompt tokens for a text-prompt request.
pub fn estimate_text_prompt_tokens_u64(prompt: &str) -> u64 {
    estimated_tokens_from_text(prompt).max(1)
}

/// `u32` variant used by orchestrator request routing.
pub fn estimate_chat_prompt_tokens_u32(messages: &[InputMessage]) -> u32 {
    estimate_chat_prompt_tokens_u64(messages).min(u32::MAX as u64) as u32
}

/// `u32` variant used by orchestrator request routing.
pub fn estimate_text_prompt_tokens_u32(prompt: &str) -> u32 {
    estimate_text_prompt_tokens_u64(prompt).min(u32::MAX as u64) as u32
}

/// Default audit listing limit shared by serving and orchestrator admin paths.
pub fn default_audit_limit() -> usize {
    50
}

/// Convert the optional request extension into a stable audit actor string.
pub fn audit_actor(req_id: Option<Extension<RequestId>>) -> String {
    req_id
        .map(|id| format!("request:{}", id.0.0))
        .unwrap_or_else(|| "request:unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::schema::{InputMessage, MessageContent};

    fn msg(role: &str, content: &str) -> InputMessage {
        InputMessage {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
            name: None,
        }
    }

    #[test]
    fn chat_prompt_estimate_is_at_least_one() {
        assert_eq!(estimate_chat_prompt_tokens_u64(&[]), 1);
    }

    #[test]
    fn chat_prompt_estimate_counts_role_and_content() {
        let estimate = estimate_chat_prompt_tokens_u64(&[msg("user", "hello there")]);
        assert!(estimate >= 4, "unexpected estimate: {estimate}");
    }

    #[test]
    fn text_prompt_estimate_is_at_least_one() {
        assert_eq!(estimate_text_prompt_tokens_u64(""), 1);
    }

    #[test]
    fn u32_estimators_clamp_large_values() {
        let huge = "x".repeat(64);
        assert!(estimate_text_prompt_tokens_u32(&huge) >= 1);
        assert!(estimate_chat_prompt_tokens_u32(&[msg("user", &huge)]) >= 1);
    }

    #[test]
    fn default_audit_limit_is_stable() {
        assert_eq!(default_audit_limit(), 50);
    }
}
