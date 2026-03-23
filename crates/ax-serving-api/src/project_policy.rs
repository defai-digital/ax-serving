use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::config::ProjectPolicyConfig;

pub const PROJECT_HEADER: &str = "x-ax-project";

#[derive(Clone, Debug)]
pub struct ResolvedProjectPolicy {
    pub project: String,
    pub worker_pool: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProjectPolicyError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for ProjectPolicyError {
    fn into_response(self) -> Response {
        (
            self.status,
            axum::Json(serde_json::json!({
                "error": self.message,
                "policy_header": PROJECT_HEADER,
            })),
        )
            .into_response()
    }
}

pub fn summary_json(config: &ProjectPolicyConfig) -> serde_json::Value {
    serde_json::json!({
        "enabled": config.enabled,
        "header": PROJECT_HEADER,
        "default_project": config.default_project,
        "rules": config.rules.iter().map(|rule| {
            serde_json::json!({
                "project": rule.project,
                "allowed_models": rule.allowed_models,
                "max_tokens_limit": rule.max_tokens_limit,
                "worker_pool": rule.worker_pool,
            })
        }).collect::<Vec<_>>()
    })
}

pub fn enforce(
    headers: &HeaderMap,
    model: &str,
    requested_max_tokens: Option<u32>,
    config: &ProjectPolicyConfig,
) -> Result<Option<ResolvedProjectPolicy>, ProjectPolicyError> {
    if !config.enabled {
        return Ok(None);
    }

    let project_name = headers
        .get(PROJECT_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or_else(|| config.default_project.clone())
        .ok_or_else(|| ProjectPolicyError {
            status: StatusCode::BAD_REQUEST,
            message: format!("missing required header '{}'", PROJECT_HEADER),
        })?;

    let rule = config
        .rules
        .iter()
        .find(|rule| rule.project == project_name)
        .ok_or_else(|| ProjectPolicyError {
            status: StatusCode::FORBIDDEN,
            message: format!(
                "project '{}' is not allowed by runtime policy",
                project_name
            ),
        })?;

    if !model_allowed(&rule.allowed_models, model) {
        return Err(ProjectPolicyError {
            status: StatusCode::FORBIDDEN,
            message: format!(
                "project '{}' is not allowed to use model '{}'",
                project_name, model
            ),
        });
    }

    if let Some(limit) = rule.max_tokens_limit
        && let Some(requested) = requested_max_tokens
        && requested > limit
    {
        return Err(ProjectPolicyError {
            status: StatusCode::FORBIDDEN,
            message: format!(
                "project '{}' max_tokens {} exceeds policy limit {}",
                project_name, requested, limit
            ),
        });
    }

    Ok(Some(ResolvedProjectPolicy {
        project: project_name,
        worker_pool: rule.worker_pool.clone(),
    }))
}

fn model_allowed(allowed_models: &[String], model: &str) -> bool {
    allowed_models
        .iter()
        .any(|pattern| model_matches(pattern, model))
}

fn model_matches(pattern: &str, model: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return model.starts_with(prefix);
    }
    pattern == model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProjectRuleConfig;
    use axum::http::HeaderValue;

    fn sample_config() -> ProjectPolicyConfig {
        ProjectPolicyConfig {
            enabled: true,
            default_project: Some("fabric".into()),
            rules: vec![
                ProjectRuleConfig {
                    project: "fabric".into(),
                    allowed_models: vec!["embed-*".into(), "chat-main".into()],
                    max_tokens_limit: Some(2048),
                    worker_pool: Some("fabric".into()),
                },
                ProjectRuleConfig {
                    project: "ops".into(),
                    allowed_models: vec!["*".into()],
                    max_tokens_limit: None,
                    worker_pool: None,
                },
            ],
        }
    }

    #[test]
    fn enforce_uses_default_project() {
        let headers = HeaderMap::new();
        let resolved = enforce(&headers, "embed-small", Some(128), &sample_config())
            .unwrap()
            .unwrap();
        assert_eq!(resolved.project, "fabric");
        assert_eq!(resolved.worker_pool.as_deref(), Some("fabric"));
    }

    #[test]
    fn enforce_rejects_unknown_project() {
        let mut headers = HeaderMap::new();
        headers.insert(PROJECT_HEADER, HeaderValue::from_static("unknown"));
        let resp = enforce(&headers, "chat-main", Some(64), &sample_config()).unwrap_err();
        assert_eq!(resp.status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn enforce_rejects_disallowed_model() {
        let mut headers = HeaderMap::new();
        headers.insert(PROJECT_HEADER, HeaderValue::from_static("fabric"));
        let resp = enforce(&headers, "secret-model", Some(64), &sample_config()).unwrap_err();
        assert_eq!(resp.status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn enforce_rejects_max_tokens_limit() {
        let mut headers = HeaderMap::new();
        headers.insert(PROJECT_HEADER, HeaderValue::from_static("fabric"));
        let resp = enforce(&headers, "chat-main", Some(4096), &sample_config()).unwrap_err();
        assert_eq!(resp.status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn enforce_disabled_always_returns_none() {
        let mut cfg = sample_config();
        cfg.enabled = false;
        let mut headers = HeaderMap::new();
        headers.insert(PROJECT_HEADER, HeaderValue::from_static("fabric"));
        // Policy disabled → always Ok(None), regardless of model or tokens.
        let result = enforce(&headers, "secret-model", Some(99999), &cfg).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn enforce_trims_header_whitespace() {
        // Header value with surrounding whitespace should still resolve to "fabric".
        let mut headers = HeaderMap::new();
        headers.insert(PROJECT_HEADER, HeaderValue::from_static("  fabric  "));
        let resolved = enforce(&headers, "chat-main", Some(64), &sample_config())
            .unwrap()
            .unwrap();
        assert_eq!(resolved.project, "fabric");
    }

    #[test]
    fn enforce_accepts_exact_max_tokens_limit() {
        // max_tokens == limit must be allowed (only > triggers rejection).
        let mut headers = HeaderMap::new();
        headers.insert(PROJECT_HEADER, HeaderValue::from_static("fabric"));
        // fabric max_tokens_limit = Some(2048)
        let result = enforce(&headers, "chat-main", Some(2048), &sample_config());
        assert!(result.is_ok(), "tokens exactly at limit should be allowed");
    }

    #[test]
    fn enforce_missing_header_no_default_returns_bad_request() {
        let cfg = ProjectPolicyConfig {
            enabled: true,
            default_project: None,
            rules: vec![ProjectRuleConfig {
                project: "fabric".into(),
                allowed_models: vec!["*".into()],
                max_tokens_limit: None,
                worker_pool: None,
            }],
        };
        let headers = HeaderMap::new();
        let err = enforce(&headers, "any-model", None, &cfg).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn summary_json_has_expected_shape() {
        let json = summary_json(&sample_config());
        assert_eq!(json["enabled"], true);
        assert_eq!(json["header"], PROJECT_HEADER);
        assert_eq!(json["default_project"], "fabric");
        let rules = json["rules"].as_array().unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0]["project"], "fabric");
        assert!(rules[0]["allowed_models"].is_array());
        assert_eq!(rules[0]["max_tokens_limit"], 2048);
        assert_eq!(rules[0]["worker_pool"], "fabric");
        // Second rule has no worker_pool.
        assert_eq!(rules[1]["project"], "ops");
        assert!(rules[1]["worker_pool"].is_null());
    }

    // ── model_matches edge cases ──────────────────────────────────────────────

    #[test]
    fn model_matches_exact_string() {
        assert!(model_matches("chat-main", "chat-main"));
        assert!(!model_matches("chat-main", "chat-other"));
    }

    #[test]
    fn model_matches_wildcard_star_accepts_anything() {
        assert!(model_matches("*", "any-model"));
        assert!(model_matches("*", ""));
    }

    #[test]
    fn model_matches_prefix_glob() {
        assert!(model_matches("embed-*", "embed-small"));
        assert!(model_matches("embed-*", "embed-large-v3"));
        // Prefix must match from the start; "embeddings" doesn't start with "embed-".
        assert!(!model_matches("embed-*", "embeddings"));
        assert!(!model_matches("embed-*", "chat-main"));
    }

    #[test]
    fn model_matches_no_suffix_star_is_exact_match() {
        // "embed-" (no trailing *) is a literal exact match, not a glob.
        assert!(model_matches("embed-", "embed-"));
        assert!(
            !model_matches("embed-", "embed-small"),
            "no star → not a prefix glob"
        );
        assert!(!model_matches("embed-", "embed-large"));
    }
}
