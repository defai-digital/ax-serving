//! Shared structured logging and OTLP/HTTP JSON trace initialization.
//!
//! Trace export is opt-in through standard `OTEL_EXPORTER_OTLP_ENDPOINT` or
//! `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` variables. This crate implements the
//! OTLP/JSON encoding directly over the OpenTelemetry SDK so the portable
//! gateway does not acquire tonic or prost dependencies.

use std::future::ready;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use opentelemetry::global;
use opentelemetry::trace::{SpanKind, Status, TracerProvider as _};
use opentelemetry::{Array, KeyValue, Value};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value as JsonValue, json};
use tracing_subscriber::prelude::*;

pub struct TelemetryGuard {
    provider: Option<SdkTracerProvider>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            let _ = provider.shutdown();
        }
    }
}

struct OtlpJsonExporter {
    client: reqwest::blocking::Client,
    endpoint: String,
    headers: HeaderMap,
    resource: Resource,
}

impl std::fmt::Debug for OtlpJsonExporter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OtlpJsonExporter")
            .field("endpoint", &"[REDACTED]")
            .field("headers", &"[REDACTED]")
            .finish()
    }
}

impl OtlpJsonExporter {
    fn from_env() -> anyhow::Result<Option<Self>> {
        let Some(endpoint) = trace_endpoint()? else {
            return Ok(None);
        };
        validate_protocol()?;
        let timeout = exporter_timeout()?;
        let headers = exporter_headers()?;
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(timeout.min(Duration::from_secs(5)))
            .timeout(timeout)
            .build()
            .context("failed to build OTLP HTTP client")?;
        Ok(Some(Self {
            client,
            endpoint,
            headers,
            resource: Resource::builder_empty().build(),
        }))
    }

    fn export_sync(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        if batch.is_empty() {
            return Ok(());
        }
        let body = encode_export_request(&self.resource, batch);
        let response = self
            .client
            .post(&self.endpoint)
            .headers(self.headers.clone())
            .json(&body)
            .send()
            .map_err(|_| OTelSdkError::InternalFailure("OTLP HTTP export failed".into()))?;
        if !response.status().is_success() {
            return Err(OTelSdkError::InternalFailure(format!(
                "OTLP collector returned HTTP {}",
                response.status().as_u16()
            )));
        }
        Ok(())
    }
}

impl SpanExporter for OtlpJsonExporter {
    fn export(
        &self,
        batch: Vec<SpanData>,
    ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
        ready(self.export_sync(batch))
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.resource = resource.clone();
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn trace_endpoint() -> anyhow::Result<Option<String>> {
    let endpoint = if let Some(endpoint) = nonempty_env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") {
        endpoint
    } else if let Some(endpoint) = nonempty_env("OTEL_EXPORTER_OTLP_ENDPOINT") {
        format!("{}/v1/traces", endpoint.trim_end_matches('/'))
    } else {
        return Ok(None);
    };
    let parsed = reqwest::Url::parse(&endpoint).context("invalid OTLP trace endpoint")?;
    anyhow::ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "OTLP trace endpoint must use http or https"
    );
    anyhow::ensure!(
        parsed.username().is_empty() && parsed.password().is_none(),
        "OTLP endpoint credentials must use OTEL_EXPORTER_OTLP_HEADERS"
    );
    Ok(Some(endpoint))
}

fn validate_protocol() -> anyhow::Result<()> {
    let protocol = nonempty_env("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        .or_else(|| nonempty_env("OTEL_EXPORTER_OTLP_PROTOCOL"));
    anyhow::ensure!(
        protocol
            .as_deref()
            .is_none_or(|value| value.eq_ignore_ascii_case("http/json")),
        "portable AX Serving trace export supports only OTLP http/json"
    );
    Ok(())
}

fn exporter_timeout() -> anyhow::Result<Duration> {
    let value = nonempty_env("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT")
        .or_else(|| nonempty_env("OTEL_EXPORTER_OTLP_TIMEOUT"));
    let milliseconds = value
        .map(|value| {
            value
                .parse::<u64>()
                .context("OTLP timeout must be milliseconds")
        })
        .transpose()?
        .unwrap_or(10_000)
        .clamp(100, 60_000);
    Ok(Duration::from_millis(milliseconds))
}

fn exporter_headers() -> anyhow::Result<HeaderMap> {
    let raw = nonempty_env("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
        .or_else(|| nonempty_env("OTEL_EXPORTER_OTLP_HEADERS"));
    let mut headers = HeaderMap::new();
    let Some(raw) = raw else {
        return Ok(headers);
    };
    for entry in raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (name, value) = entry
            .split_once('=')
            .context("OTLP headers must use name=value entries")?;
        let name =
            HeaderName::from_bytes(name.trim().as_bytes()).context("invalid OTLP header name")?;
        anyhow::ensure!(
            !matches!(
                name.as_str(),
                "host" | "content-length" | "transfer-encoding" | "connection"
            ),
            "OTLP headers contain a forbidden transport header"
        );
        let decoded = percent_decode(value.trim()).context("invalid OTLP header encoding")?;
        let mut value = HeaderValue::from_str(&decoded).context("invalid OTLP header value")?;
        value.set_sensitive(true);
        headers.insert(name, value);
    }
    Ok(headers)
}

fn percent_decode(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                let high = hex_value(bytes[index + 1])?;
                let low = hex_value(bytes[index + 2])?;
                output.push((high << 4) | low);
                index += 3;
            }
            b'%' => return None,
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(output).ok()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn encode_export_request(resource: &Resource, spans: Vec<SpanData>) -> JsonValue {
    let mut resource_attributes = resource
        .iter()
        .map(|(key, value)| encode_key_value(key.as_str(), value))
        .collect::<Vec<_>>();
    resource_attributes.sort_by(|left, right| left["key"].as_str().cmp(&right["key"].as_str()));
    let scope_spans = spans
        .into_iter()
        .map(|span| {
            let scope = &span.instrumentation_scope;
            let mut scope_json = json!({
                "name": scope.name(),
                "attributes": scope
                    .attributes()
                    .map(|value| encode_key_value(value.key.as_str(), &value.value))
                    .collect::<Vec<_>>(),
            });
            if let Some(version) = scope.version() {
                scope_json["version"] = json!(version);
            }
            if let Some(schema_url) = scope.schema_url() {
                scope_json["schemaUrl"] = json!(schema_url);
            }
            json!({"scope": scope_json, "spans": [encode_span(span)]})
        })
        .collect::<Vec<_>>();
    let mut resource_spans = json!({
        "resource": {"attributes": resource_attributes},
        "scopeSpans": scope_spans,
    });
    if let Some(schema_url) = resource.schema_url() {
        resource_spans["schemaUrl"] = json!(schema_url);
    }
    json!({"resourceSpans": [resource_spans]})
}

fn encode_span(span: SpanData) -> JsonValue {
    let context = span.span_context;
    let mut encoded = json!({
        "traceId": context.trace_id().to_string(),
        "spanId": context.span_id().to_string(),
        "name": span.name,
        "kind": span_kind_code(span.span_kind),
        "startTimeUnixNano": unix_nanos(span.start_time).to_string(),
        "endTimeUnixNano": unix_nanos(span.end_time).to_string(),
        "attributes": encode_attributes(&span.attributes),
        "droppedAttributesCount": span.dropped_attributes_count,
        "events": span.events.events.into_iter().map(encode_event).collect::<Vec<_>>(),
        "droppedEventsCount": span.events.dropped_count,
        "links": span.links.links.into_iter().map(encode_link).collect::<Vec<_>>(),
        "droppedLinksCount": span.links.dropped_count,
        "status": encode_status(span.status),
        "flags": context.trace_flags().to_u8(),
    });
    if span.parent_span_id != opentelemetry::SpanId::INVALID {
        encoded["parentSpanId"] = json!(span.parent_span_id.to_string());
    }
    let trace_state = context.trace_state().header();
    if !trace_state.is_empty() {
        encoded["traceState"] = json!(trace_state);
    }
    encoded
}

fn encode_event(event: opentelemetry::trace::Event) -> JsonValue {
    json!({
        "timeUnixNano": unix_nanos(event.timestamp).to_string(),
        "name": event.name,
        "attributes": encode_attributes(&event.attributes),
        "droppedAttributesCount": event.dropped_attributes_count,
    })
}

fn encode_link(link: opentelemetry::trace::Link) -> JsonValue {
    let context = link.span_context;
    let mut encoded = json!({
        "traceId": context.trace_id().to_string(),
        "spanId": context.span_id().to_string(),
        "attributes": encode_attributes(&link.attributes),
        "droppedAttributesCount": link.dropped_attributes_count,
        "flags": context.trace_flags().to_u8(),
    });
    let trace_state = context.trace_state().header();
    if !trace_state.is_empty() {
        encoded["traceState"] = json!(trace_state);
    }
    encoded
}

fn encode_attributes(attributes: &[KeyValue]) -> Vec<JsonValue> {
    attributes
        .iter()
        .map(|attribute| encode_key_value(attribute.key.as_str(), &attribute.value))
        .collect()
}

fn encode_key_value(key: &str, value: &Value) -> JsonValue {
    json!({"key": key, "value": encode_value(value)})
}

fn encode_value(value: &Value) -> JsonValue {
    match value {
        Value::Bool(value) => json!({"boolValue": value}),
        Value::I64(value) => json!({"intValue": value.to_string()}),
        Value::F64(value) => json!({"doubleValue": value}),
        Value::String(value) => json!({"stringValue": value.as_str()}),
        Value::Array(value) => json!({"arrayValue": {"values": encode_array(value)}}),
        _ => json!({"stringValue": "unsupported"}),
    }
}

fn encode_array(array: &Array) -> Vec<JsonValue> {
    match array {
        Array::Bool(values) => values
            .iter()
            .map(|value| json!({"boolValue": value}))
            .collect(),
        Array::I64(values) => values
            .iter()
            .map(|value| json!({"intValue": value.to_string()}))
            .collect(),
        Array::F64(values) => values
            .iter()
            .map(|value| json!({"doubleValue": value}))
            .collect(),
        Array::String(values) => values
            .iter()
            .map(|value| json!({"stringValue": value.as_str()}))
            .collect(),
        _ => Vec::new(),
    }
}

fn span_kind_code(kind: SpanKind) -> u8 {
    match kind {
        SpanKind::Internal => 1,
        SpanKind::Server => 2,
        SpanKind::Client => 3,
        SpanKind::Producer => 4,
        SpanKind::Consumer => 5,
    }
}

fn encode_status(status: Status) -> JsonValue {
    match status {
        Status::Unset => json!({"code": 0}),
        Status::Ok => json!({"code": 1}),
        Status::Error { description } => json!({"code": 2, "message": description}),
    }
}

fn unix_nanos(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn env_filter(default_level: tracing::Level) -> tracing_subscriber::EnvFilter {
    // `with_default_directive` applies only when AXS_LOG yields no directives;
    // `from_env("AXS_LOG").add_directive(...)` would overwrite a bare global
    // directive (e.g. "debug") because Directive equality ignores the level.
    tracing_subscriber::EnvFilter::builder()
        .with_env_var("AXS_LOG")
        .with_default_directive(default_level.into())
        .from_env_lossy()
}

fn log_format() -> String {
    std::env::var("AXS_LOG_FORMAT")
        .unwrap_or_else(|_| "text".into())
        .trim()
        .to_ascii_lowercase()
}

/// Initialize one process-global tracing subscriber.
///
/// Prompt, output, tool arguments, image data, model paths, and credentials are
/// never added by this initializer. Call sites must use bounded attributes.
pub fn init(
    service_name: &'static str,
    default_level: tracing::Level,
) -> anyhow::Result<TelemetryGuard> {
    let format = log_format();
    anyhow::ensure!(
        matches!(format.as_str(), "" | "text" | "json"),
        "AXS_LOG_FORMAT must be text or json"
    );

    // Trace-context continuity is part of the gateway/agent protocol and must
    // not depend on whether this process happens to export spans. Install an
    // SDK provider even without an exporter so incoming W3C parents produce a
    // valid child context that can be injected into the next hop.
    global::set_text_map_propagator(TraceContextPropagator::new());
    let provider_builder = SdkTracerProvider::builder()
        .with_resource(Resource::builder().with_service_name(service_name).build());
    let provider = match OtlpJsonExporter::from_env()? {
        Some(exporter) => provider_builder.with_batch_exporter(exporter).build(),
        None => provider_builder.build(),
    };
    global::set_tracer_provider(provider.clone());
    let telemetry = tracing_opentelemetry::layer().with_tracer(provider.tracer(service_name));

    if format == "json" {
        tracing_subscriber::registry()
            .with(env_filter(default_level))
            .with(telemetry)
            .with(tracing_subscriber::fmt::layer().json())
            .try_init()
            .context("failed to initialize JSON tracing and OTLP subscriber")?;
    } else {
        tracing_subscriber::registry()
            .with(env_filter(default_level))
            .with(telemetry)
            .with(tracing_subscriber::fmt::layer())
            .try_init()
            .context("failed to initialize tracing and OTLP subscriber")?;
    }

    Ok(TelemetryGuard {
        provider: Some(provider),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::process::Command;

    use opentelemetry::propagation::{Extractor, Injector};
    use opentelemetry::trace::TraceContextExt as _;
    use opentelemetry::{Array, Value};
    use tracing_opentelemetry::OpenTelemetrySpanExt as _;

    use super::{encode_array, percent_decode};

    #[test]
    fn percent_decodes_otel_headers_without_logging_them() {
        assert_eq!(
            percent_decode("Bearer%20secret"),
            Some("Bearer secret".into())
        );
        assert_eq!(percent_decode("bad%2"), None);
    }

    #[test]
    fn integer_arrays_follow_otlp_json_string_encoding() {
        let encoded = encode_array(&Array::I64(vec![1, 2]));
        assert_eq!(encoded[0]["intValue"], "1");
        assert_eq!(super::encode_value(&Value::I64(42))["intValue"], "42");
    }

    struct MapCarrier(HashMap<String, String>);

    impl Extractor for MapCarrier {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).map(String::as_str)
        }

        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(String::as_str).collect()
        }
    }

    impl Injector for MapCarrier {
        fn set(&mut self, key: &str, value: String) {
            self.0.insert(key.to_string(), value);
        }
    }

    #[test]
    fn trace_context_continues_without_exporter() {
        const CHILD_ENV: &str = "AXS_OBSERVABILITY_PROPAGATION_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "tests::trace_context_continues_without_exporter",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                // Must not silence spans: the env filter gates the telemetry
                // layer too, so "off" would disable the info_span! below.
                .env_remove("AXS_LOG")
                .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
                .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "child trace test failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }

        let _guard = super::init("ax-serving-observability-test", tracing::Level::INFO).unwrap();
        let incoming_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
        let incoming = MapCarrier(HashMap::from([(
            "traceparent".into(),
            format!("00-{incoming_trace_id}-00f067aa0ba902b7-01"),
        )]));
        let parent = opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.extract(&incoming)
        });
        let span = tracing::info_span!("propagation-child");
        span.set_parent(parent).unwrap();

        let context = span.context();
        let span_context = context.span().span_context().clone();
        assert!(span_context.is_valid());
        assert_eq!(span_context.trace_id().to_string(), incoming_trace_id);
        assert_ne!(span_context.span_id().to_string(), "00f067aa0ba902b7");

        let mut outgoing = MapCarrier(HashMap::new());
        opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&context, &mut outgoing);
        });
        assert!(
            outgoing
                .0
                .get("traceparent")
                .is_some_and(|value| value.starts_with(&format!("00-{incoming_trace_id}-")))
        );
    }
}
