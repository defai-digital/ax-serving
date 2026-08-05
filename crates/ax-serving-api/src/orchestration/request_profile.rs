//! Runtime-neutral request classification used for admission and routing.
//!
//! The profile deliberately contains only bounded routing metadata. Prompt
//! rendering and tokenization stay inside the selected inference runtime.

use std::collections::BTreeSet;

use ax_serving_protocol::{
    DecisionProfileV1, LogicalModelId, Operation, PoolId, ProtocolCapability, RequestId, TenantId,
};
use serde::Serialize;

use super::registry::RequestKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorityClass {
    Low,
    Normal,
    High,
}

impl PriorityClass {
    pub fn parse(value: Option<&str>) -> anyhow::Result<Self> {
        match value.map(str::trim).filter(|value| !value.is_empty()) {
            None | Some("normal") => Ok(Self::Normal),
            Some("low") => Ok(Self::Low),
            Some("high") => Ok(Self::High),
            Some(other) => {
                anyhow::bail!("invalid priority {other:?}; expected low, normal, or high")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestProfile {
    pub request_id: RequestId,
    pub operation: Operation,
    pub logical_model: LogicalModelId,
    pub stream: bool,
    pub max_output_tokens: Option<u64>,
    pub body_bytes: usize,
    pub message_count: Option<usize>,
    pub modalities: BTreeSet<String>,
    pub required_capabilities: BTreeSet<ProtocolCapability>,
    /// Optional client-declared requirement. This is never inferred by
    /// tokenizing or rendering the prompt in the gateway.
    pub minimum_context_tokens: Option<u64>,
    pub tenant_id: TenantId,
    pub priority: PriorityClass,
    /// Tenant-scoped digest of an optional client affinity hint. The raw hint
    /// never leaves request classification and is not logged or forwarded.
    pub cache_affinity_key: Option<u64>,
    pub required_pool: Option<PoolId>,
    pub preferred_pool: Option<PoolId>,
    /// Versioned domain-policy inputs. P0 construction leaves client-facing
    /// cost/quality hints unset until tenant policy can authenticate them.
    pub decision: DecisionProfileV1,
    /// Compatibility constraint for clients that explicitly pin a runtime.
    pub runtime_hint: Option<String>,
    /// Absolute gateway deadline shared by admission, all attempts, and streaming.
    pub deadline: tokio::time::Instant,
}

impl RequestProfile {
    pub fn request_kind(&self) -> RequestKind {
        match self.operation.as_str() {
            Operation::EMBEDDINGS => RequestKind::Embedding,
            _ if self.modalities.contains("image") => RequestKind::Vision,
            _ => RequestKind::Llm,
        }
    }

    pub fn remaining(&self) -> Option<std::time::Duration> {
        self.deadline
            .checked_duration_since(tokio::time::Instant::now())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RequestBodyError {
    #[error("request JSON must be a top-level object")]
    NotObject,
    #[error("request JSON is malformed")]
    Malformed,
    #[error("request must contain exactly one top-level model field")]
    ModelFieldCount,
    #[error("request model field must be a string")]
    ModelNotString,
    #[error("duplicate top-level routing field '{0}'")]
    DuplicateRoutingField(String),
    #[error("failed to encode runtime model id")]
    EncodeModel(#[source] serde_json::Error),
}

const ROUTING_FIELDS: &[&str] = &[
    "model",
    "backend",
    "runtime",
    "stream",
    "max_tokens",
    "max_completion_tokens",
    "tools",
    "response_format",
];

#[derive(Debug)]
struct TopLevelMember {
    key: String,
    key_start: usize,
    delimiter: Option<usize>,
}

/// Reject duplicate fields whose ambiguity could alter admission or routing.
pub fn validate_unique_routing_fields(body: &[u8]) -> Result<(), RequestBodyError> {
    scan_top_level_members(body).map(|_| ())
}

/// Remove AX Serving-only top-level routing hints before runtime dispatch.
///
/// The scanner preserves every byte outside the removed `backend` and
/// `runtime` members, including unknown extension fields, nested fields, and
/// number formatting. It also enforces the same duplicate-routing-field guard
/// as [`validate_unique_routing_fields`]. `None` means no rewrite was needed.
pub fn strip_ax_routing_hints(body: &[u8]) -> Result<Option<Vec<u8>>, RequestBodyError> {
    let (members, close_brace) = scan_top_level_members(body)?;
    if !members
        .iter()
        .any(|member| matches!(member.key.as_str(), "backend" | "runtime"))
    {
        return Ok(None);
    }

    let mut removal_ranges = Vec::new();
    let mut index = 0;
    while index < members.len() {
        if !matches!(members[index].key.as_str(), "backend" | "runtime") {
            index += 1;
            continue;
        }

        let run_start = index;
        while index + 1 < members.len()
            && matches!(members[index + 1].key.as_str(), "backend" | "runtime")
        {
            index += 1;
        }
        let run_end = index;

        let (start, end) = if run_end + 1 < members.len() {
            (
                members[run_start].key_start,
                members[run_end]
                    .delimiter
                    .expect("non-final top-level member has a delimiter")
                    + 1,
            )
        } else if run_start > 0 {
            (
                members[run_start - 1]
                    .delimiter
                    .expect("member before a final run has a delimiter"),
                close_brace,
            )
        } else {
            (members[run_start].key_start, close_brace)
        };
        removal_ranges.push((start, end));
        index += 1;
    }

    let removed_bytes = removal_ranges
        .iter()
        .map(|(start, end)| end - start)
        .sum::<usize>();
    let mut sanitized = Vec::with_capacity(body.len() - removed_bytes);
    let mut copied_through = 0;
    for (start, end) in removal_ranges {
        sanitized.extend_from_slice(&body[copied_through..start]);
        copied_through = end;
    }
    sanitized.extend_from_slice(&body[copied_through..]);
    Ok(Some(sanitized))
}

fn scan_top_level_members(body: &[u8]) -> Result<(Vec<TopLevelMember>, usize), RequestBodyError> {
    let mut cursor = skip_whitespace(body, 0);
    if body.get(cursor) != Some(&b'{') {
        return Err(RequestBodyError::NotObject);
    }
    cursor += 1;
    let mut seen = BTreeSet::new();
    let mut members = Vec::new();
    let close_brace;
    loop {
        cursor = skip_whitespace(body, cursor);
        if body.get(cursor) == Some(&b'}') {
            close_brace = cursor;
            cursor += 1;
            break;
        }
        let key_start = cursor;
        let key_end = scan_string(body, cursor)?;
        let key: String = serde_json::from_slice(&body[key_start..key_end])
            .map_err(|_| RequestBodyError::Malformed)?;
        if ROUTING_FIELDS.contains(&key.as_str()) && !seen.insert(key.clone()) {
            return Err(RequestBodyError::DuplicateRoutingField(key));
        }
        cursor = skip_whitespace(body, key_end);
        if body.get(cursor) != Some(&b':') {
            return Err(RequestBodyError::Malformed);
        }
        cursor = skip_whitespace(body, cursor + 1);
        cursor = scan_value(body, cursor, 0)?;
        cursor = skip_whitespace(body, cursor);
        let delimiter = match body.get(cursor) {
            Some(b',') => {
                let delimiter = cursor;
                cursor += 1;
                Some(delimiter)
            }
            Some(b'}') => {
                close_brace = cursor;
                cursor += 1;
                members.push(TopLevelMember {
                    key,
                    key_start,
                    delimiter: None,
                });
                break;
            }
            _ => return Err(RequestBodyError::Malformed),
        };
        members.push(TopLevelMember {
            key,
            key_start,
            delimiter,
        });
    }
    if skip_whitespace(body, cursor) != body.len() {
        return Err(RequestBodyError::Malformed);
    }
    Ok((members, close_brace))
}

/// Replace only the top-level `model` string while preserving all other
/// request bytes verbatim, including unknown fields and number formatting.
pub fn rewrite_runtime_model(
    body: &[u8],
    runtime_model_id: &str,
) -> Result<Vec<u8>, RequestBodyError> {
    let mut cursor = skip_whitespace(body, 0);
    if body.get(cursor) != Some(&b'{') {
        return Err(RequestBodyError::NotObject);
    }
    cursor += 1;
    let mut model_span = None;

    loop {
        cursor = skip_whitespace(body, cursor);
        if body.get(cursor) == Some(&b'}') {
            cursor += 1;
            break;
        }
        let key_start = cursor;
        let key_end = scan_string(body, cursor)?;
        let key: String = serde_json::from_slice(&body[key_start..key_end])
            .map_err(|_| RequestBodyError::Malformed)?;
        cursor = skip_whitespace(body, key_end);
        if body.get(cursor) != Some(&b':') {
            return Err(RequestBodyError::Malformed);
        }
        cursor = skip_whitespace(body, cursor + 1);
        let value_start = cursor;
        let value_end = scan_value(body, cursor, 0)?;
        if key == "model" {
            if model_span.is_some() {
                return Err(RequestBodyError::ModelFieldCount);
            }
            serde_json::from_slice::<String>(&body[value_start..value_end])
                .map_err(|_| RequestBodyError::ModelNotString)?;
            model_span = Some((value_start, value_end));
        }
        cursor = skip_whitespace(body, value_end);
        match body.get(cursor) {
            Some(b',') => cursor += 1,
            Some(b'}') => {
                cursor += 1;
                break;
            }
            _ => return Err(RequestBodyError::Malformed),
        }
    }

    if skip_whitespace(body, cursor) != body.len() {
        return Err(RequestBodyError::Malformed);
    }
    let (start, end) = model_span.ok_or(RequestBodyError::ModelFieldCount)?;
    let encoded = serde_json::to_vec(runtime_model_id).map_err(RequestBodyError::EncodeModel)?;
    let mut rewritten = Vec::with_capacity(body.len() - (end - start) + encoded.len());
    rewritten.extend_from_slice(&body[..start]);
    rewritten.extend_from_slice(&encoded);
    rewritten.extend_from_slice(&body[end..]);
    Ok(rewritten)
}

fn skip_whitespace(body: &[u8], mut cursor: usize) -> usize {
    while body
        .get(cursor)
        .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
    {
        cursor += 1;
    }
    cursor
}

fn scan_string(body: &[u8], start: usize) -> Result<usize, RequestBodyError> {
    if body.get(start) != Some(&b'"') {
        return Err(RequestBodyError::Malformed);
    }
    let mut cursor = start + 1;
    while let Some(byte) = body.get(cursor).copied() {
        match byte {
            b'"' => return Ok(cursor + 1),
            b'\\' => {
                cursor += 1;
                let escaped = body.get(cursor).ok_or(RequestBodyError::Malformed)?;
                if *escaped == b'u' {
                    let end = cursor.checked_add(5).ok_or(RequestBodyError::Malformed)?;
                    let digits = body
                        .get(cursor + 1..end)
                        .ok_or(RequestBodyError::Malformed)?;
                    if !digits.iter().all(u8::is_ascii_hexdigit) {
                        return Err(RequestBodyError::Malformed);
                    }
                    cursor = end;
                    continue;
                }
                if !matches!(
                    *escaped,
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
                ) {
                    return Err(RequestBodyError::Malformed);
                }
            }
            0..=0x1f => return Err(RequestBodyError::Malformed),
            _ => {}
        }
        cursor += 1;
    }
    Err(RequestBodyError::Malformed)
}

fn scan_value(body: &[u8], start: usize, depth: u8) -> Result<usize, RequestBodyError> {
    if depth > 64 {
        return Err(RequestBodyError::Malformed);
    }
    match body.get(start) {
        Some(b'"') => scan_string(body, start),
        Some(b'{') => scan_object(body, start, depth + 1),
        Some(b'[') => scan_array(body, start, depth + 1),
        Some(_) => scan_primitive(body, start),
        None => Err(RequestBodyError::Malformed),
    }
}

fn scan_object(body: &[u8], start: usize, depth: u8) -> Result<usize, RequestBodyError> {
    let mut cursor = start + 1;
    loop {
        cursor = skip_whitespace(body, cursor);
        if body.get(cursor) == Some(&b'}') {
            return Ok(cursor + 1);
        }
        cursor = scan_string(body, cursor)?;
        cursor = skip_whitespace(body, cursor);
        if body.get(cursor) != Some(&b':') {
            return Err(RequestBodyError::Malformed);
        }
        cursor = skip_whitespace(body, cursor + 1);
        cursor = scan_value(body, cursor, depth)?;
        cursor = skip_whitespace(body, cursor);
        match body.get(cursor) {
            Some(b',') => cursor += 1,
            Some(b'}') => return Ok(cursor + 1),
            _ => return Err(RequestBodyError::Malformed),
        }
    }
}

fn scan_array(body: &[u8], start: usize, depth: u8) -> Result<usize, RequestBodyError> {
    let mut cursor = start + 1;
    loop {
        cursor = skip_whitespace(body, cursor);
        if body.get(cursor) == Some(&b']') {
            return Ok(cursor + 1);
        }
        cursor = scan_value(body, cursor, depth)?;
        cursor = skip_whitespace(body, cursor);
        match body.get(cursor) {
            Some(b',') => cursor += 1,
            Some(b']') => return Ok(cursor + 1),
            _ => return Err(RequestBodyError::Malformed),
        }
    }
}

fn scan_primitive(body: &[u8], start: usize) -> Result<usize, RequestBodyError> {
    let mut cursor = start;
    while body
        .get(cursor)
        .is_some_and(|byte| !matches!(byte, b' ' | b'\n' | b'\r' | b'\t' | b',' | b']' | b'}'))
    {
        cursor += 1;
    }
    if cursor == start {
        Err(RequestBodyError::Malformed)
    } else {
        Ok(cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PriorityClass, rewrite_runtime_model, strip_ax_routing_hints,
        validate_unique_routing_fields,
    };

    #[test]
    fn priority_is_bounded_and_defaults_to_normal() {
        assert_eq!(PriorityClass::parse(None).unwrap(), PriorityClass::Normal);
        assert_eq!(
            PriorityClass::parse(Some("high")).unwrap(),
            PriorityClass::High
        );
        assert!(PriorityClass::parse(Some("urgent")).is_err());
    }

    #[test]
    fn runtime_model_rewrite_preserves_unknown_bytes() {
        let body =
            br#"{ "model" : "public/model", "extension": 1.2300, "nested": {"model":"keep"} }"#;
        let rewritten = rewrite_runtime_model(body, "Qwen/Qwen3-32B").unwrap();
        assert_eq!(
            String::from_utf8(rewritten).unwrap(),
            r#"{ "model" : "Qwen/Qwen3-32B", "extension": 1.2300, "nested": {"model":"keep"} }"#
        );
    }

    #[test]
    fn duplicate_top_level_model_is_rejected() {
        let body = br#"{"model":"one","model":"two"}"#;
        assert!(rewrite_runtime_model(body, "runtime").is_err());
    }

    #[test]
    fn duplicate_security_sensitive_routing_field_is_rejected() {
        let body = br#"{"model":"one","stream":false,"stream":true}"#;
        assert!(validate_unique_routing_fields(body).is_err());
        let harmless = br#"{"model":"one","metadata":1,"other":2}"#;
        validate_unique_routing_fields(harmless).unwrap();
    }

    #[test]
    fn ax_routing_hints_are_stripped_without_reencoding_runtime_fields() {
        let body = br#"{ "runtime":"tensorrt_llm", "model":"m", "extension":1.2300, "nested":{"runtime":"keep","backend":"keep"} }"#;
        let sanitized = strip_ax_routing_hints(body).unwrap().unwrap();
        assert_eq!(
            String::from_utf8(sanitized).unwrap(),
            r#"{  "model":"m", "extension":1.2300, "nested":{"runtime":"keep","backend":"keep"} }"#
        );
    }

    #[test]
    fn adjacent_ax_routing_hints_are_removed_from_every_object_position() {
        for (body, expected) in [
            (
                br#"{"runtime":"r","backend":"auto","model":"m"}"#.as_slice(),
                br#"{"model":"m"}"#.as_slice(),
            ),
            (
                br#"{"model":"m","runtime":"r","backend":"auto","stream":false}"#.as_slice(),
                br#"{"model":"m","stream":false}"#.as_slice(),
            ),
            (
                br#"{"model":"m","runtime":"r","backend":"auto"}"#.as_slice(),
                br#"{"model":"m"}"#.as_slice(),
            ),
            (
                br#"{"runtime":"r","backend":"auto"}"#.as_slice(),
                br#"{}"#.as_slice(),
            ),
        ] {
            assert_eq!(strip_ax_routing_hints(body).unwrap().unwrap(), expected);
        }
    }

    #[test]
    fn separated_and_escaped_ax_routing_hints_are_removed() {
        let body = br#"{"\u0072untime":"r","model":"m","backend":"auto","stream":false}"#;
        assert_eq!(
            strip_ax_routing_hints(body).unwrap().unwrap(),
            br#"{"model":"m","stream":false}"#
        );
    }

    #[test]
    fn routing_sanitizer_rejects_duplicates_and_skips_unmodified_bodies() {
        assert!(strip_ax_routing_hints(br#"{"model":"m","runtime":"a","runtime":"b"}"#).is_err());
        assert_eq!(
            strip_ax_routing_hints(br#"{"model":"m","metadata":{"runtime":"nested"}}"#).unwrap(),
            None
        );
    }
}
