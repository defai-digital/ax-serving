//! Trace context injection and worker-response header filtering for dispatch.

use axum::http::{HeaderMap, HeaderName, HeaderValue, header};
use opentelemetry::propagation::Injector;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

struct TraceHeaderInjector(HeaderMap);

impl Injector for TraceHeaderInjector {
    fn set(&mut self, key: &str, value: String) {
        if !matches!(key, "traceparent" | "tracestate" | "baggage") || value.len() > 1024 {
            return;
        }
        let Ok(name) = HeaderName::from_bytes(key.as_bytes()) else {
            return;
        };
        let Ok(value) = HeaderValue::from_str(&value) else {
            return;
        };
        self.0.insert(name, value);
    }
}

pub(super) fn current_trace_headers() -> HeaderMap {
    let context = tracing::Span::current().context();
    let mut injector = TraceHeaderInjector(HeaderMap::new());
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut injector);
    });
    injector.0
}

/// Attach the gateway→worker dispatch credential when configured.
pub(super) fn attach_dispatch_auth(
    builder: reqwest::RequestBuilder,
    dispatch_token: Option<&HeaderValue>,
) -> reqwest::RequestBuilder {
    match dispatch_token {
        Some(value) => builder.header(ax_serving_protocol::DISPATCH_TOKEN_HEADER, value.clone()),
        None => builder,
    }
}

pub(super) fn should_forward_worker_header(
    name: &HeaderName,
    include_content_length: bool,
) -> bool {
    let name = name.as_str();
    !matches!(
        name,
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "set-cookie"
            | "www-authenticate"
            | "x-ax-admission-state"
            | "x-ax-attempt-id"
            | "x-ax-dispatch-token"
            | "x-ax-deployment-id"
            | "x-ax-pool-id"
    ) && (include_content_length || name != header::CONTENT_LENGTH.as_str())
}
