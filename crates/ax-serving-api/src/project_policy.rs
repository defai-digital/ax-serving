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
            message: format!("project '{}' is not allowed by runtime policy", project_name),
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
    allowed_models.iter().any(|pattern| model_matches(pattern, model))
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
    use axum::http::HeaderValue;
    use crate::config::ProjectRuleConfig;

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
}
