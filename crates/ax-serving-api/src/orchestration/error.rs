//! Stable public AX error envelope for gateway-generated failures.

use ax_serving_protocol::{AdmissionPhase, AxErrorEnvelope, AxErrorMetadata, ErrorBody, RequestId};
use axum::Json;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

const MAX_PUBLIC_ERROR_CHARS: usize = 512;

pub fn ax_error_response(
    status: StatusCode,
    request_id: RequestId,
    code: impl Into<String>,
    message: impl AsRef<str>,
    retryable: bool,
    phase: AdmissionPhase,
) -> Response {
    let code = code.into();
    let envelope = AxErrorEnvelope {
        error: ErrorBody {
            message: bounded_public_message(message.as_ref()),
            error_type: error_type(status).to_string(),
            param: None,
            code: code.clone(),
        },
        ax: AxErrorMetadata {
            request_id,
            retryable,
            phase,
        },
    };
    let mut response = (status, Json(envelope)).into_response();
    if let Ok(value) = HeaderValue::from_str(&request_id.to_string()) {
        response
            .headers_mut()
            .insert(ax_serving_protocol::REQUEST_ID_HEADER, value);
    }
    if let Ok(value) = HeaderValue::from_str(&code) {
        response.headers_mut().insert("x-ax-error-code", value);
    }
    response
}

fn error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "authentication_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.is_client_error() => "invalid_request_error",
        _ => "server_error",
    }
}

fn bounded_public_message(message: &str) -> String {
    let mut characters = message.chars();
    let prefix = characters
        .by_ref()
        .take(MAX_PUBLIC_ERROR_CHARS)
        .collect::<String>();
    if characters.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use ax_serving_protocol::{AdmissionPhase, RequestId};
    use axum::http::StatusCode;

    use super::ax_error_response;

    #[tokio::test]
    async fn envelope_is_openai_compatible_and_carries_ax_metadata() {
        let request_id = RequestId::new();
        let response = ax_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            request_id,
            "AXS_NO_COMPATIBLE_DEPLOYMENT",
            "no compatible deployment",
            true,
            AdmissionPhase::EndpointSelection,
        );
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "AXS_NO_COMPATIBLE_DEPLOYMENT");
        assert_eq!(value["ax"]["request_id"], request_id.to_string());
        assert_eq!(value["ax"]["phase"], "endpoint_selection");
    }
}
