//! Inference route handlers: chat completions, text completions, embeddings.

use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::UNIX_EPOCH;

use ax_serving_engine::{ChatMessage, GenerateEvent, GenerateInput};
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{self, Stream};
use serde::Serialize;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::schema::*;
use super::validation::{
    build_generation_params, cache_ttl_err, effective_max_tokens, map_stop_reason, resolve_grammar,
    resolve_logprobs, validate_max_tokens, validate_model_identifier,
    validate_multimodal_backend_support, validate_response_format, validate_sampling_params,
};
use crate::ServingLayer;
use crate::cache::{CacheInflightEnter, CacheInflightLeaderGuard, CachePreference};
use crate::project_policy;
use crate::scheduler::SchedulerPermit;
use crate::utils::request_meta::{
    estimate_chat_prompt_tokens_u64, estimate_text_prompt_tokens_u64,
};
use tokio::sync::OwnedSemaphorePermit;

use super::routes::{
    cache_hit_response, record_cache_error, scheduler_error_status, unix_now, with_timing,
    write_cache_and_record,
};

#[derive(Serialize)]
struct StreamTopLogprob {
    token: String,
    logprob: f32,
    bytes: Vec<u8>,
}

#[derive(Serialize)]
struct StreamTokenLogprob {
    token: String,
    logprob: f32,
    bytes: Vec<u8>,
    top_logprobs: Vec<StreamTopLogprob>,
}

#[derive(Serialize)]
struct StreamLogprobs {
    content: Vec<StreamTokenLogprob>,
}

#[derive(Serialize)]
struct StreamChatToolFunction<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize)]
struct StreamChatToolCall<'a> {
    index: u32,
    id: &'a str,
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: StreamChatToolFunction<'a>,
}

#[derive(Serialize)]
struct StreamChatDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<StreamChatToolCall<'a>>>,
}

#[derive(Serialize)]
struct StreamChatChoice<'a> {
    index: u32,
    delta: StreamChatDelta<'a>,
    finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<StreamLogprobs>,
}

#[derive(Serialize)]
struct StreamUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct StreamChatChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<StreamChatChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<StreamUsage>,
}

#[derive(Serialize)]
struct StreamTextChoice<'a> {
    text: &'a str,
    index: u32,
    finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<StreamLogprobs>,
}

#[derive(Serialize)]
struct StreamTextChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<StreamTextChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<StreamUsage>,
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: &'a str,
}

fn sse_json_event<T: Serialize>(value: &T) -> Event {
    match serde_json::to_string(value) {
        Ok(s) => Event::default().data(s),
        Err(_) => Event::default()
            .event("error")
            .data("{\"error\":\"serialization failure\"}"),
    }
}

/// Capacity of the mpsc channel used to stream `GenerateEvent` tokens from the
/// backend to the HTTP response handler.  Large enough to buffer a full short
/// response without back-pressure on the inference thread.
const GENERATE_CHANNEL_CAPACITY: usize = 512;

/// Build a logprobs payload from token-level probability data.
fn build_logprobs_payload(
    logprobs_enabled: bool,
    lp_data: Option<(f32, Vec<(String, f32)>)>,
    token_text: &str,
) -> Option<StreamLogprobs> {
    if !logprobs_enabled {
        return None;
    }
    let content = if let Some((lp, top)) = lp_data {
        let top_logprobs = top
            .iter()
            .map(|(t, l)| StreamTopLogprob {
                token: t.clone(),
                logprob: *l,
                bytes: t.as_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        vec![StreamTokenLogprob {
            token: token_text.to_string(),
            logprob: lp,
            bytes: token_text.as_bytes().to_vec(),
            top_logprobs,
        }]
    } else {
        Vec::new()
    };
    Some(StreamLogprobs { content })
}

fn record_generation_stats(
    metrics: &crate::metrics::MetricsStore,
    stats: &ax_serving_engine::GenerationStats,
) {
    metrics.record_generation_stats(stats);
}

/// Normalized form of a single chat message used exclusively for cache-key
/// construction.  Role is lowercased and content is whitespace-trimmed so
/// that minor client-side formatting differences never cause a false miss.
#[derive(Serialize)]
pub(crate) struct NormalizedMessage {
    role: String,
    content: String,
}

fn model_cache_fingerprint(path: &Path) -> String {
    let Ok(metadata) = std::fs::metadata(path) else {
        return "missing".to_string();
    };
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("size={};mtime_ns={modified_ns}", metadata.len())
}

/// Cache key payload.
///
/// # Normalization rules
/// - `requested_model_id` is dropped: the resolved path already identifies the
///   loaded model, so model-alias changes no longer bust the cache.
/// - `model_fingerprint` includes file metadata so cache entries invalidate
///   when weights are replaced in place at the same path.
/// - `messages`: role lowercased, content trimmed (leading/trailing whitespace).
/// - Floating-point params serialized with 4-decimal precision to absorb f32
///   representation noise (`0.6999999` and `0.7` both become `"0.7000"`).
#[derive(Serialize)]
pub(crate) struct CacheKeyPayload<'a> {
    version: &'static str,
    resolved_model_path: &'a str,
    model_fingerprint: String,
    resolved_model_arch: &'a str,
    messages: Vec<NormalizedMessage>,
    temperature: String,
    top_p: String,
    min_p: Option<String>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
    repeat_penalty: String,
    stop: Option<Vec<String>>,
    frequency_penalty: Option<String>,
    presence_penalty: Option<String>,
    grammar: Option<&'a str>,
    response_format: Option<&'a ResponseFormat>,
    mirostat: Option<u8>,
    mirostat_tau: Option<String>,
    mirostat_eta: Option<String>,
    tools: Option<&'a Vec<Tool>>,
    tool_choice: Option<&'a serde_json::Value>,
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
}

/// Build the raw bytes that are fed into the SHA-256 cache key.
///
/// `effective_max_tokens` should already have the server-side default applied
/// (so requests without `max_tokens` and requests with `max_tokens=default`
/// share the same cache entry).
pub(crate) fn build_cache_key(
    req: &ChatCompletionRequest,
    resolved_model_path: &str,
    resolved_model_arch: &str,
    effective_max_tokens: Option<u32>,
) -> anyhow::Result<Vec<u8>> {
    let model_fingerprint = model_cache_fingerprint(Path::new(resolved_model_path));
    let messages = req
        .messages
        .iter()
        .map(|m| NormalizedMessage {
            role: m.role.to_lowercase(),
            content: m.content.as_text().trim().to_string(),
        })
        .collect();

    let payload = CacheKeyPayload {
        version: "v4",
        resolved_model_path,
        model_fingerprint,
        resolved_model_arch,
        messages,
        temperature: format!("{:.4}", req.temperature),
        top_p: format!("{:.4}", req.top_p),
        min_p: req.min_p.map(|v| format!("{v:.4}")),
        top_k: req.top_k,
        max_tokens: effective_max_tokens,
        seed: req.seed,
        repeat_penalty: format!("{:.4}", req.repeat_penalty),
        stop: req.stop.as_ref().map(|s| {
            let mut v = s.clone().into_vec();
            v.sort();
            v
        }),
        frequency_penalty: req.frequency_penalty.map(|v| format!("{v:.4}")),
        presence_penalty: req.presence_penalty.map(|v| format!("{v:.4}")),
        grammar: req.grammar.as_deref(),
        response_format: req.response_format.as_ref(),
        mirostat: req.mirostat,
        mirostat_tau: req.mirostat_tau.map(|v| format!("{v:.4}")),
        mirostat_eta: req.mirostat_eta.map(|v| format!("{v:.4}")),
        tools: req.tools.as_ref(),
        tool_choice: req.tool_choice.as_ref(),
        logprobs: req.logprobs,
        top_logprobs: req.top_logprobs,
    };
    serde_json::to_vec(&payload).map_err(anyhow::Error::from)
}

/// Cache key payload for text completions (`POST /v1/completions`).
///
/// Normalisation mirrors `CacheKeyPayload`: prompt is trimmed, floats at 4dp.
#[derive(Serialize)]
pub(crate) struct TextCacheKeyPayload<'a> {
    version: &'static str,
    kind: &'static str,
    resolved_model_path: &'a str,
    model_fingerprint: String,
    resolved_model_arch: &'a str,
    prompt: &'a str,
    temperature: String,
    top_p: String,
    min_p: Option<String>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
    repeat_penalty: String,
    stop: Option<Vec<String>>,
    frequency_penalty: Option<String>,
    presence_penalty: Option<String>,
    grammar: Option<&'a str>,
    response_format: Option<&'a ResponseFormat>,
    mirostat: Option<u8>,
    mirostat_tau: Option<String>,
    mirostat_eta: Option<String>,
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
}

/// Build the raw bytes for the SHA-256 text-completion cache key.
pub(crate) fn build_text_cache_key(
    req: &CompletionRequest,
    resolved_model_path: &str,
    resolved_model_arch: &str,
    effective_max_tokens: Option<u32>,
) -> anyhow::Result<Vec<u8>> {
    let model_fingerprint = model_cache_fingerprint(Path::new(resolved_model_path));
    let payload = TextCacheKeyPayload {
        version: "v3",
        kind: "text_completion",
        resolved_model_path,
        model_fingerprint,
        resolved_model_arch,
        prompt: req.prompt.trim(),
        temperature: format!("{:.4}", req.temperature),
        top_p: format!("{:.4}", req.top_p),
        min_p: req.min_p.map(|v| format!("{v:.4}")),
        top_k: req.top_k,
        max_tokens: effective_max_tokens,
        seed: req.seed,
        repeat_penalty: format!("{:.4}", req.repeat_penalty),
        stop: req.stop.as_ref().map(|s| {
            let mut v = s.clone().into_vec();
            v.sort();
            v
        }),
        frequency_penalty: req.frequency_penalty.map(|v| format!("{v:.4}")),
        presence_penalty: req.presence_penalty.map(|v| format!("{v:.4}")),
        grammar: req.grammar.as_deref(),
        response_format: req.response_format.as_ref(),
        mirostat: req.mirostat,
        mirostat_tau: req.mirostat_tau.map(|v| format!("{v:.4}")),
        mirostat_eta: req.mirostat_eta.map(|v| format!("{v:.4}")),
        logprobs: req.logprobs,
        top_logprobs: req.top_logprobs,
    };
    serde_json::to_vec(&payload).map_err(anyhow::Error::from)
}

/// POST /v1/chat/completions
///
/// Supports both streaming (SSE) and non-streaming responses.
pub async fn chat_completions(
    State(layer): State<Arc<ServingLayer>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Input validation.
    if let Some(response) = validate_model_identifier(&req.model, "model") {
        return response;
    }
    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "messages must not be empty"})),
        )
            .into_response();
    }
    if req.messages.len() > MAX_MESSAGES {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("too many messages (max {MAX_MESSAGES})")})),
        )
            .into_response();
    }
    for msg in &req.messages {
        if msg.content.byte_len() > MAX_CONTENT_BYTES {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "message content exceeds 32 KB limit"})),
            )
                .into_response();
        }
    }
    if let Some(r) = validate_max_tokens(req.max_tokens) {
        return r;
    }
    if let Some(r) = validate_sampling_params(
        req.temperature,
        req.top_p,
        req.min_p,
        req.top_k,
        req.repeat_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        req.logprobs,
        req.top_logprobs,
        req.mirostat,
    ) {
        return r;
    }
    if let Some(r) = validate_response_format(req.response_format.as_ref()) {
        return r;
    }

    // Apply the server-side default when the client omits max_tokens.
    // A configured default of 0 means "no cap" (pass None to the backend).
    let effective_max_tokens = effective_max_tokens(req.max_tokens, layer.default_max_tokens);
    if let Err(resp) = project_policy::enforce(
        &headers,
        &req.model,
        effective_max_tokens,
        &layer.config.project_policy,
    ) {
        return resp.into_response();
    }

    // Look up the model in the registry.
    let entry = match layer.registry.get(&req.model) {
        Some(e) => e,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": format!("model '{}' not loaded", req.model)})),
            )
                .into_response();
        }
    };

    let has_image_input = req.messages.iter().any(|msg| msg.content.has_images());
    if let Some(resp) = validate_multimodal_backend_support(
        has_image_input,
        layer.backend.backend_name_for_handle(entry.handle),
    ) {
        return resp;
    }

    let handle = entry.handle;
    let model_name = req.model.clone();
    let cache_requested = req.cache.unwrap_or(CachePreference::Enable);
    let cache_active = cache_requested == CachePreference::Enable
        && !req.stream
        && !has_image_input
        && layer.cache.is_some();
    let mut cache_key = None::<String>;
    let mut cache_ttl = None;
    let mut cache_leader_guard: Option<CacheInflightLeaderGuard> = None;

    // Phase 1 — pre-permit cache fast path.
    //
    // If we enter the inflight deduplication tracker as the Leader *and* Redis
    // has an immediate cache hit, we return here without acquiring any admission
    // permits.  True cache hits are free: they consume no scheduler capacity.
    //
    // In every other case (Leader+miss, Follower) we record state and continue.
    // Followers wait in the pre-permit path below, so true deduplicated waiters
    // do not consume scheduler capacity unless they later promote to leader.
    let mut pending_follower: Option<(String, tokio::sync::broadcast::Receiver<()>)> = None;

    if cache_active && let Some(cache) = &layer.cache {
        match build_cache_key(
            &req,
            &entry.path.display().to_string(),
            &entry.metadata.architecture,
            effective_max_tokens,
        ) {
            Ok(payload) => {
                let key = cache.make_key(&payload);
                match layer.cache_inflight.enter(&key) {
                    CacheInflightEnter::Leader(leader) => {
                        match cache.get(&key).await {
                            Ok(Some(hit_json)) => {
                                if serde_json::from_str::<serde_json::Value>(&hit_json).is_ok() {
                                    // Immediate cache hit — bypass admission entirely.
                                    layer.metrics.record_exact_cache_hit();
                                    return cache_hit_response(hit_json);
                                }
                                record_cache_error(
                                    &layer.cache_metrics,
                                    "invalid cached JSON payload",
                                );
                            }
                            Ok(None) => {}
                            Err(e) => {
                                record_cache_error(&layer.cache_metrics, format_args!("read: {e}"));
                            }
                        }
                        // Leader, no immediate hit — set up for inference + caching.
                        match cache.ttl_for_request(req.cache_ttl.as_deref()) {
                            Ok(ttl) => {
                                cache_key = Some(key);
                                cache_ttl = Some(ttl);
                                cache_leader_guard = Some(leader);
                            }
                            Err(e) => {
                                return cache_ttl_err(e);
                            }
                        }
                    }
                    CacheInflightEnter::Follower(rx) => {
                        // Defer the follower wait until after permits are acquired.
                        pending_follower = Some((key, rx));
                    }
                }
            }
            Err(e) => {
                record_cache_error(&layer.cache_metrics, format_args!("key generation: {e}"));
            }
        }
    }

    // Phase 1.5 — follower wait WITHOUT admission permits (WS3).
    //
    // Followers wait for leader completion before acquiring any scheduler slot.
    // A follower that receives a cache hit returns immediately without ever
    // holding a permit — it consumes zero execution capacity.  Only if the
    // leader fails (cache miss, guard dropped early, max retries) does the
    // follower promote to leader and proceed to permit acquisition below.
    //
    // `axs_cache_follower_waiting` tracks the current count of pre-permit
    // waiters for observability (WS5).
    if let Some((key, mut rx)) = pending_follower
        && let Some(cache) = &layer.cache
    {
        layer
            .scheduler
            .metrics
            .cache_follower_waiting
            .fetch_add(1, Ordering::Relaxed);

        let mut attempts = 0usize;
        let early_return = loop {
            attempts += 1;
            let _ = rx.recv().await;
            match cache.get(&key).await {
                Ok(Some(hit_json)) => {
                    if serde_json::from_str::<serde_json::Value>(&hit_json).is_ok() {
                        // Cache hit — no permits ever needed.
                        layer.metrics.record_cache_follower_hit();
                        layer
                            .scheduler
                            .metrics
                            .cache_follower_waiting
                            .fetch_sub(1, Ordering::Relaxed);
                        return cache_hit_response(hit_json);
                    }
                    record_cache_error(&layer.cache_metrics, "invalid cached JSON payload");
                }
                Ok(None) => {}
                Err(e) => {
                    record_cache_error(&layer.cache_metrics, &e);
                }
            }
            if attempts >= layer.cache_inflight_max_retries {
                break false; // fall through to inference without a cache guard
            }
            // Re-enter the inflight tracker; we may become leader if the
            // previous leader dropped its guard without writing to cache.
            match layer.cache_inflight.enter(&key) {
                CacheInflightEnter::Leader(leader) => {
                    match cache.ttl_for_request(req.cache_ttl.as_deref()) {
                        Ok(ttl) => {
                            cache_key = Some(key);
                            cache_ttl = Some(ttl);
                            cache_leader_guard = Some(leader);
                        }
                        Err(e) => {
                            layer
                                .scheduler
                                .metrics
                                .cache_follower_waiting
                                .fetch_sub(1, Ordering::Relaxed);
                            return cache_ttl_err(e);
                        }
                    }
                    break false; // Proceed to inference as new leader.
                }
                CacheInflightEnter::Follower(new_rx) => {
                    rx = new_rx;
                    continue;
                }
            }
        };
        let _ = early_return; // loop always breaks with false; kept for clarity
        layer
            .scheduler
            .metrics
            .cache_follower_waiting
            .fetch_sub(1, Ordering::Relaxed);
    }

    // Acquire per-model slot first: if this model is at capacity we fail fast
    // without consuming a global admission slot, avoiding head-of-line blocking
    // across unrelated models.
    let pm_permit = match layer
        .per_model_scheduler
        .acquire(&req.model, layer.scheduler.config().max_wait_ms)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    // Split-scheduler prompt estimation is only needed when the feature is enabled.
    let estimated_prompt_tokens = if layer.scheduler.split_enabled {
        estimate_chat_prompt_tokens_u64(&req.messages)
    } else {
        0
    };
    // Acquire global admission slot — returns 429 on queue-full, 503 on timeout/throttle.
    let permit = match layer
        .scheduler
        .acquire_with_tokens(estimated_prompt_tokens)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            return (
                scheduler_error_status(&e),
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    // Preserve structured chat messages so backend applies the model template.
    // Multipart content (vision) is serialised as a JSON array so llama-server
    // receives the full image_url parts unchanged.
    let chat_messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: serde_json::to_value(&m.content).unwrap_or_default(),
        })
        .collect();

    let stop_seqs = req.stop.clone().map(|s| s.into_vec()).unwrap_or_default();
    let grammar = resolve_grammar(req.grammar.clone(), req.response_format.as_ref());
    let (req_logprobs, req_top_logprobs) = resolve_logprobs(req.logprobs, req.top_logprobs);

    let mut params = build_generation_params(
        req.stream,
        req.temperature,
        req.top_p,
        req.min_p,
        req.top_k,
        effective_max_tokens,
        stop_seqs,
        req.seed,
        req.repeat_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        grammar,
        req.response_format.as_ref(),
        req.mirostat,
        req.mirostat_tau,
        req.mirostat_eta,
        req_logprobs,
        req_top_logprobs,
    );
    params.tools = req
        .tools
        .as_ref()
        .map(|t| serde_json::to_value(t).unwrap_or_default());
    params.tool_choice = req.tool_choice.clone();

    let (tx, rx) = mpsc::channel::<GenerateEvent>(GENERATE_CHANNEL_CAPACITY);

    if let Err(e) = layer
        .backend
        .generate(handle, GenerateInput::Chat(chat_messages), params, tx)
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response();
    }
    layer.metrics.record_cold_request();

    // Both permits are carried into the stream/blocking response and dropped
    // only when the full response is sent — keeping inflight slots occupied.
    // Extract queue_wait_us before permit is moved.
    let queue_wait_us = permit.queue_wait_us();
    if req.stream {
        stream_response(
            rx,
            model_name,
            req_logprobs,
            Arc::clone(&layer.metrics),
            permit,
            pm_permit,
        )
        .into_response()
    } else {
        blocking_response(
            rx,
            model_name,
            layer.cache.as_ref(),
            cache_key,
            cache_ttl,
            layer.cache_metrics.as_ref(),
            layer.metrics.as_ref(),
            cache_leader_guard,
            permit,
            pm_permit,
            queue_wait_us,
            req_logprobs,
        )
        .await
    }
}

/// Return a streaming SSE response, forwarding tokens as OpenAI delta chunks.
///
/// Conforms to the OpenAI streaming protocol:
/// - `role: "assistant"` is emitted **only** in the first content chunk's delta.
/// - `choices[0].logprobs` is populated per-token when `logprobs=true`.
/// - A final `data: [DONE]` sentinel is sent after the stop chunk.
fn stream_response(
    rx: mpsc::Receiver<GenerateEvent>,
    model: String,
    logprobs: bool,
    metrics: Arc<crate::metrics::MetricsStore>,
    permit: SchedulerPermit,
    pm_permit: OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let created = unix_now();
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());

    // State: (rx, id, model, created, phase, first_token, permit, pm_permit, logprobs, pending)
    //
    // phase 0 — normal streaming (recv from channel)
    // phase 1 — stop chunk emitted; send `data: [DONE]` sentinel next
    // phase 2 — done; return None to end the stream
    //
    // `pending` holds an event consumed while reading a paired TokenLogprob that
    // turned out to be something else (Done/Error). It is processed on the next step.
    let event_stream = stream::unfold(
        (
            rx,
            id,
            model,
            created,
            0u8,
            true,
            Some(permit),
            Some(pm_permit),
            logprobs,
            None::<GenerateEvent>,
            metrics,
        ),
        |(
            mut rx,
            id,
            model,
            created,
            phase,
            first_token,
            permit,
            pm,
            logprobs,
            pending,
            metrics,
        )| async move {
            match phase {
                2 => None,
                1 => {
                    // OpenAI requires `data: [DONE]` as the final SSE frame.
                    let ev = Event::default().data("[DONE]");
                    Some((
                        Ok(ev),
                        (
                            rx, id, model, created, 2, false, None, None, logprobs, None, metrics,
                        ),
                    ))
                }
                _ => {
                    // Use a pending event from the previous step (if any), otherwise
                    // read from the channel, skipping orphaned TokenLogprob events.
                    let real_event = if let Some(ev) = pending {
                        ev
                    } else {
                        loop {
                            match rx.recv().await {
                                Some(GenerateEvent::TokenLogprob { .. }) => continue,
                                ev => break ev.unwrap_or(GenerateEvent::Done(Default::default())),
                            }
                        }
                    };

                    match real_event {
                        GenerateEvent::Token(text) => {
                            // Record TTFT on the first token (WS5 observability).
                            if first_token && let Some(ref p) = permit {
                                p.record_ttft_now();
                            }

                            // If logprobs requested, try to read the paired TokenLogprob.
                            let (lp_data, next_pending) = if logprobs {
                                match rx.recv().await {
                                    Some(GenerateEvent::TokenLogprob { logprob, top }) => {
                                        (Some((logprob, top)), None)
                                    }
                                    // Unexpected event — store as pending for the next step.
                                    other => (None, other),
                                }
                            } else {
                                (None, None)
                            };

                            let logprobs_payload = build_logprobs_payload(logprobs, lp_data, &text);

                            let delta = StreamChatDelta {
                                role: if first_token { Some("assistant") } else { None },
                                content: Some(&text),
                                tool_calls: None,
                            };
                            let chunk = StreamChatChunk {
                                id: &id,
                                object: "chat.completion.chunk",
                                created,
                                model: &model,
                                choices: vec![StreamChatChoice {
                                    index: 0,
                                    delta,
                                    finish_reason: None,
                                    logprobs: logprobs_payload,
                                }],
                                usage: None,
                            };
                            let ev = sse_json_event(&chunk);
                            Some((
                                Ok(ev),
                                (
                                    rx,
                                    id,
                                    model,
                                    created,
                                    0,
                                    false,
                                    permit,
                                    pm,
                                    logprobs,
                                    next_pending,
                                    metrics,
                                ),
                            ))
                        }
                        GenerateEvent::ToolCall {
                            id: call_id,
                            name,
                            arguments,
                        } => {
                            let chunk = StreamChatChunk {
                                id: &id,
                                object: "chat.completion.chunk",
                                created,
                                model: &model,
                                choices: vec![StreamChatChoice {
                                    index: 0,
                                    delta: StreamChatDelta {
                                        role: Some("assistant"),
                                        content: None,
                                        tool_calls: Some(vec![StreamChatToolCall {
                                            index: 0,
                                            id: &call_id,
                                            tool_type: "function",
                                            function: StreamChatToolFunction {
                                                name: &name,
                                                arguments: &arguments,
                                            },
                                        }]),
                                    },
                                    finish_reason: Some("tool_calls"),
                                    logprobs: None,
                                }],
                                usage: None,
                            };
                            let ev = sse_json_event(&chunk);
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 0, false, permit, pm, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                        GenerateEvent::Done(stats) => {
                            record_generation_stats(metrics.as_ref(), &stats);
                            let chunk = StreamChatChunk {
                                id: &id,
                                object: "chat.completion.chunk",
                                created,
                                model: &model,
                                choices: vec![StreamChatChoice {
                                    index: 0,
                                    delta: StreamChatDelta {
                                        role: None,
                                        content: None,
                                        tool_calls: None,
                                    },
                                    finish_reason: Some(map_stop_reason(&stats.stop_reason)),
                                    logprobs: None,
                                }],
                                usage: Some(StreamUsage {
                                    prompt_tokens: stats.prompt_tokens,
                                    completion_tokens: stats.completion_tokens,
                                    total_tokens: stats.prompt_tokens + stats.completion_tokens,
                                }),
                            };
                            let ev = sse_json_event(&chunk);
                            // Drop both permits; next phase emits [DONE] then ends.
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 1, false, None, None, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                        GenerateEvent::Error(e) => {
                            let env = ErrorEnvelope { error: &e };
                            let ev = Event::default().event("error").data(
                                serde_json::to_string(&env).unwrap_or_else(|_| {
                                    "{\"error\":\"serialization failure\"}".to_string()
                                }),
                            );
                            // Transition to phase 1 (not 2) so the next
                            // iteration emits the `data: [DONE]` sentinel
                            // required by the OpenAI SSE protocol.
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 1, false, None, None, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                        GenerateEvent::TokenLogprob { .. } => {
                            tracing::warn!(
                                "rest/routes: unexpected TokenLogprob event in chat stream"
                            );
                            let env = ErrorEnvelope {
                                error: "unexpected token logprob event",
                            };
                            let ev = Event::default().event("error").data(
                                serde_json::to_string(&env).unwrap_or_else(|_| {
                                    "{\"error\":\"serialization failure\"}".to_string()
                                }),
                            );
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 1, false, None, None, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                    }
                }
            }
        },
    );

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}

/// Collect all tokens and return a complete (non-streaming) response.
///
/// Adds `X-Ax-Stage-Timing: queue_wait_us=<N>` on every response (success and
/// error) so callers can correlate latency with queue wait time.
#[allow(clippy::too_many_arguments)]
async fn blocking_response(
    mut rx: mpsc::Receiver<GenerateEvent>,
    model: String,
    cache: Option<&crate::cache::ResponseCache>,
    cache_key: Option<String>,
    cache_ttl: Option<std::time::Duration>,
    cache_metrics: &crate::cache::CacheMetrics,
    metrics: &crate::metrics::MetricsStore,
    _cache_leader_guard: Option<crate::cache::CacheInflightLeaderGuard>,
    permit: SchedulerPermit,
    pm_permit: OwnedSemaphorePermit,
    queue_wait_us: u64,
    collect_logprobs: bool,
) -> Response {
    use super::schema::{ChoiceLogprobs, LogprobContent, TopLogprob};
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = Usage::default();
    let mut generation_done = false;
    let mut finish_reason = FinishReason::Stop;
    // Logprob accumulation: parallel vecs — logprob_tokens[i] corresponds to token_texts[i].
    let mut token_texts: Vec<String> = Vec::new();
    let mut logprob_entries: Vec<(f32, Vec<(String, f32)>)> = Vec::new();
    let mut pending_token: Option<String> = None;

    while let Some(event) = rx.recv().await {
        match event {
            GenerateEvent::Token(text) => {
                content.push_str(&text);
                if collect_logprobs {
                    pending_token = Some(text);
                }
            }
            GenerateEvent::TokenLogprob { logprob, top } => {
                if collect_logprobs && let Some(tok) = pending_token.take() {
                    token_texts.push(tok);
                    logprob_entries.push((logprob, top));
                }
            }
            GenerateEvent::ToolCall {
                id: call_id,
                name,
                arguments,
            } => {
                pending_token = None;
                let index = tool_calls.len() as u32;
                tool_calls.push(ToolCall {
                    index,
                    id: call_id,
                    tool_type: "function".into(),
                    function: ToolCallFunction { name, arguments },
                });
                finish_reason = FinishReason::ToolCalls;
            }
            GenerateEvent::Done(stats) => {
                record_generation_stats(metrics, &stats);
                usage = Usage {
                    prompt_tokens: stats.prompt_tokens as u32,
                    completion_tokens: stats.completion_tokens as u32,
                    total_tokens: (stats.prompt_tokens + stats.completion_tokens) as u32,
                };
                // Only update if not already set by a ToolCall event.
                if finish_reason == FinishReason::Stop {
                    finish_reason = match stats.stop_reason.as_str() {
                        "length" => FinishReason::Length,
                        "content_filter" => FinishReason::ContentFilter,
                        _ => FinishReason::Stop,
                    };
                }
                generation_done = true;
                break;
            }
            GenerateEvent::Error(e) => {
                return with_timing(
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ChatCompletionResponse {
                            id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
                            object: "chat.completion",
                            created: unix_now(),
                            model,
                            choices: vec![Choice {
                                index: 0,
                                message: Some(Message {
                                    role: "assistant".into(),
                                    content: format!("[error: {e}]"),
                                    tool_calls: None,
                                }),
                                delta: None,
                                finish_reason: Some(FinishReason::Stop),
                                logprobs: None,
                            }],
                            usage: Some(usage),
                        }),
                    )
                        .into_response(),
                    queue_wait_us,
                );
            }
        }
    }

    if !generation_done {
        return with_timing(
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ChatCompletionResponse {
                    id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
                    object: "chat.completion",
                    created: unix_now(),
                    model,
                    choices: vec![Choice {
                        index: 0,
                        message: Some(Message {
                            role: "assistant".into(),
                            content: "[error: generation ended unexpectedly]".into(),
                            tool_calls: None,
                        }),
                        delta: None,
                        finish_reason: Some(FinishReason::Stop),
                        logprobs: None,
                    }],
                    usage: Some(usage),
                }),
            )
                .into_response(),
            queue_wait_us,
        );
    }

    let tool_calls_opt = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let choice_logprobs = if collect_logprobs && !token_texts.is_empty() {
        let content_lp: Vec<LogprobContent> = token_texts
            .into_iter()
            .zip(logprob_entries)
            .map(|(tok, (lp, top))| LogprobContent {
                bytes: Some(tok.as_bytes().to_vec()),
                token: tok,
                logprob: lp,
                top_logprobs: top
                    .into_iter()
                    .map(|(t, l)| TopLogprob {
                        bytes: Some(t.as_bytes().to_vec()),
                        token: t,
                        logprob: l,
                    })
                    .collect(),
            })
            .collect();
        Some(ChoiceLogprobs {
            content: content_lp,
        })
    } else {
        None
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
        object: "chat.completion",
        created: unix_now(),
        model,
        choices: vec![Choice {
            index: 0,
            message: Some(Message {
                role: "assistant".into(),
                content,
                tool_calls: tool_calls_opt,
            }),
            delta: None,
            finish_reason: Some(finish_reason),
            logprobs: choice_logprobs,
        }],
        usage: Some(usage),
    };

    // Release both inflight slots (global + per-model) before optional cache
    // I/O.  Redis writes can be slow (network latency); holding either permit
    // during that window would unnecessarily block the next request from
    // entering the scheduler.
    drop(pm_permit);
    drop(permit);

    if let (Some(cache), Some(key), Some(ttl)) = (cache, cache_key, cache_ttl) {
        write_cache_and_record(cache, &key, &response, ttl, cache_metrics, metrics).await;
    }

    with_timing(
        (StatusCode::OK, Json(response)).into_response(),
        queue_wait_us,
    )
}

/// POST /v1/completions
///
/// OpenAI-compatible text completions endpoint.  Accepts a raw `prompt`
/// string and returns generated text.  Supports both streaming (SSE) and
/// non-streaming responses.
pub async fn text_completions(
    State(layer): State<Arc<ServingLayer>>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    // Input validation.
    if let Some(response) = validate_model_identifier(&req.model, "model") {
        return response;
    }
    if req.prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt must not be empty"})),
        )
            .into_response();
    }
    if req.prompt.len() > MAX_CONTENT_BYTES {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt exceeds 32 KB limit"})),
        )
            .into_response();
    }
    if let Some(r) = validate_max_tokens(req.max_tokens) {
        return r;
    }
    if let Some(r) = validate_sampling_params(
        req.temperature,
        req.top_p,
        req.min_p,
        req.top_k,
        req.repeat_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        req.logprobs,
        req.top_logprobs,
        req.mirostat,
    ) {
        return r;
    }
    if let Some(r) = validate_response_format(req.response_format.as_ref()) {
        return r;
    }

    let effective_max_tokens = effective_max_tokens(req.max_tokens, layer.default_max_tokens);
    if let Err(resp) = project_policy::enforce(
        &headers,
        &req.model,
        effective_max_tokens,
        &layer.config.project_policy,
    ) {
        return resp.into_response();
    }

    let entry = match layer.registry.get(&req.model) {
        Some(e) => e,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": format!("model '{}' not loaded", req.model)})),
            )
                .into_response();
        }
    };

    let handle = entry.handle;
    let model_name = req.model.clone();

    let cache_requested = req.cache.unwrap_or(CachePreference::Enable);
    let cache_active =
        cache_requested == CachePreference::Enable && !req.stream && layer.cache.is_some();
    let mut cache_key = None::<String>;
    let mut cache_ttl = None;
    let mut cache_leader_guard: Option<CacheInflightLeaderGuard> = None;
    let mut pending_follower: Option<(String, tokio::sync::broadcast::Receiver<()>)> = None;

    // Phase 1 — pre-permit cache fast path.
    if cache_active && let Some(cache) = &layer.cache {
        match build_text_cache_key(
            &req,
            &entry.path.display().to_string(),
            &entry.metadata.architecture,
            effective_max_tokens,
        ) {
            Ok(payload) => {
                let key = cache.make_key(&payload);
                match layer.cache_inflight.enter(&key) {
                    CacheInflightEnter::Leader(leader) => {
                        match cache.get(&key).await {
                            Ok(Some(hit_json)) => {
                                if serde_json::from_str::<serde_json::Value>(&hit_json).is_ok() {
                                    layer.metrics.record_exact_cache_hit();
                                    return cache_hit_response(hit_json);
                                }
                                record_cache_error(
                                    &layer.cache_metrics,
                                    "invalid cached JSON payload",
                                );
                            }
                            Ok(None) => {}
                            Err(e) => {
                                record_cache_error(&layer.cache_metrics, format_args!("read: {e}"));
                            }
                        }
                        match cache.ttl_for_request(req.cache_ttl.as_deref()) {
                            Ok(ttl) => {
                                cache_key = Some(key);
                                cache_ttl = Some(ttl);
                                cache_leader_guard = Some(leader);
                            }
                            Err(e) => {
                                return cache_ttl_err(e);
                            }
                        }
                    }
                    CacheInflightEnter::Follower(rx) => {
                        pending_follower = Some((key, rx));
                    }
                }
            }
            Err(e) => {
                record_cache_error(&layer.cache_metrics, format_args!("key generation: {e}"));
            }
        }
    }

    // Phase 1.5 — follower wait WITHOUT admission permits (WS3).
    // Same logic as chat_completions: followers return immediately on cache hit
    // without ever acquiring a scheduler slot.
    if let Some((key, mut rx)) = pending_follower
        && let Some(cache) = &layer.cache
    {
        layer
            .scheduler
            .metrics
            .cache_follower_waiting
            .fetch_add(1, Ordering::Relaxed);

        let mut attempts = 0usize;
        let early_return = loop {
            attempts += 1;
            let _ = rx.recv().await;
            match cache.get(&key).await {
                Ok(Some(hit_json)) => {
                    if serde_json::from_str::<serde_json::Value>(&hit_json).is_ok() {
                        layer.metrics.record_cache_follower_hit();
                        layer
                            .scheduler
                            .metrics
                            .cache_follower_waiting
                            .fetch_sub(1, Ordering::Relaxed);
                        return cache_hit_response(hit_json);
                    }
                    record_cache_error(&layer.cache_metrics, "invalid cached JSON payload");
                }
                Ok(None) => {}
                Err(e) => {
                    record_cache_error(&layer.cache_metrics, &e);
                }
            }
            if attempts >= layer.cache_inflight_max_retries {
                break false;
            }
            match layer.cache_inflight.enter(&key) {
                CacheInflightEnter::Leader(leader) => {
                    match cache.ttl_for_request(req.cache_ttl.as_deref()) {
                        Ok(ttl) => {
                            cache_key = Some(key);
                            cache_ttl = Some(ttl);
                            cache_leader_guard = Some(leader);
                        }
                        Err(e) => {
                            layer
                                .scheduler
                                .metrics
                                .cache_follower_waiting
                                .fetch_sub(1, Ordering::Relaxed);
                            return cache_ttl_err(e);
                        }
                    }
                    break false;
                }
                CacheInflightEnter::Follower(new_rx) => {
                    rx = new_rx;
                    continue;
                }
            }
        };
        let _ = early_return;
        layer
            .scheduler
            .metrics
            .cache_follower_waiting
            .fetch_sub(1, Ordering::Relaxed);
    }

    // Per-model first: fail fast without holding a global slot if this model is saturated.
    let pm_permit = match layer
        .per_model_scheduler
        .acquire(&req.model, layer.scheduler.config().max_wait_ms)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let estimated_prompt_tokens = if layer.scheduler.split_enabled {
        estimate_text_prompt_tokens_u64(&req.prompt)
    } else {
        0
    };
    let permit = match layer
        .scheduler
        .acquire_with_tokens(estimated_prompt_tokens)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            return (
                scheduler_error_status(&e),
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let stop_seqs = req.stop.clone().map(|s| s.into_vec()).unwrap_or_default();
    let grammar = resolve_grammar(req.grammar.clone(), req.response_format.as_ref());
    let (req_logprobs, req_top_logprobs) = resolve_logprobs(req.logprobs, req.top_logprobs);

    let params = build_generation_params(
        req.stream,
        req.temperature,
        req.top_p,
        req.min_p,
        req.top_k,
        effective_max_tokens,
        stop_seqs,
        req.seed,
        req.repeat_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        grammar,
        req.response_format.as_ref(),
        req.mirostat,
        req.mirostat_tau,
        req.mirostat_eta,
        req_logprobs,
        req_top_logprobs,
    );

    let (tx, rx) = mpsc::channel::<GenerateEvent>(GENERATE_CHANNEL_CAPACITY);

    if let Err(e) = layer
        .backend
        .generate(handle, GenerateInput::Text(req.prompt), params, tx)
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response();
    }
    layer.metrics.record_cold_request();

    let queue_wait_us = permit.queue_wait_us();
    if req.stream {
        text_stream_response(
            rx,
            model_name,
            req_logprobs,
            Arc::clone(&layer.metrics),
            permit,
            pm_permit,
        )
        .into_response()
    } else {
        text_blocking_response(
            rx,
            model_name,
            req_logprobs,
            layer.cache.as_ref(),
            cache_key,
            cache_ttl,
            layer.cache_metrics.as_ref(),
            layer.metrics.as_ref(),
            cache_leader_guard,
            permit,
            pm_permit,
            queue_wait_us,
        )
        .await
    }
}

/// Return a streaming SSE response for text completions, forwarding tokens as
/// OpenAI `text_completion` chunks.
fn text_stream_response(
    rx: mpsc::Receiver<GenerateEvent>,
    model: String,
    logprobs: bool,
    metrics: Arc<crate::metrics::MetricsStore>,
    permit: SchedulerPermit,
    pm_permit: OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let created = unix_now();
    let id = format!("cmpl-{}", Uuid::new_v4().simple());

    let event_stream = stream::unfold(
        (
            rx,
            id,
            model,
            created,
            0u8,
            true, // first_token
            Some(permit),
            Some(pm_permit),
            logprobs,
            None::<GenerateEvent>,
            metrics,
        ),
        |(
            mut rx,
            id,
            model,
            created,
            phase,
            first_token,
            permit,
            pm,
            logprobs,
            pending,
            metrics,
        )| async move {
            match phase {
                2 => None,
                1 => {
                    let ev = Event::default().data("[DONE]");
                    Some((
                        Ok(ev),
                        (
                            rx, id, model, created, 2, false, None, None, logprobs, None, metrics,
                        ),
                    ))
                }
                _ => {
                    // Use pending event from previous step, or read a fresh one
                    // (skipping orphaned TokenLogprob events).
                    let real_event = if let Some(ev) = pending {
                        ev
                    } else {
                        loop {
                            match rx.recv().await {
                                Some(GenerateEvent::TokenLogprob { .. }) => continue,
                                ev => break ev.unwrap_or(GenerateEvent::Done(Default::default())),
                            }
                        }
                    };

                    match real_event {
                        GenerateEvent::Token(text) => {
                            // Record TTFT on the first token (WS5 observability).
                            if first_token && let Some(ref p) = permit {
                                p.record_ttft_now();
                            }

                            // If logprobs requested, try to read the paired TokenLogprob.
                            let (lp_data, next_pending) = if logprobs {
                                match rx.recv().await {
                                    Some(GenerateEvent::TokenLogprob { logprob, top }) => {
                                        (Some((logprob, top)), None)
                                    }
                                    other => (None, other),
                                }
                            } else {
                                (None, None)
                            };

                            let logprobs_payload = build_logprobs_payload(logprobs, lp_data, &text);

                            let chunk = StreamTextChunk {
                                id: &id,
                                object: "text_completion",
                                created,
                                model: &model,
                                choices: vec![StreamTextChoice {
                                    text: &text,
                                    index: 0,
                                    finish_reason: None,
                                    logprobs: logprobs_payload,
                                }],
                                usage: None,
                            };
                            let ev = sse_json_event(&chunk);
                            Some((
                                Ok(ev),
                                (
                                    rx,
                                    id,
                                    model,
                                    created,
                                    0,
                                    false, // first_token: no longer first
                                    permit,
                                    pm,
                                    logprobs,
                                    next_pending,
                                    metrics,
                                ),
                            ))
                        }
                        GenerateEvent::Done(stats) => {
                            record_generation_stats(metrics.as_ref(), &stats);
                            let chunk = StreamTextChunk {
                                id: &id,
                                object: "text_completion",
                                created,
                                model: &model,
                                choices: vec![StreamTextChoice {
                                    text: "",
                                    index: 0,
                                    finish_reason: Some(map_stop_reason(&stats.stop_reason)),
                                    logprobs: None,
                                }],
                                usage: Some(StreamUsage {
                                    prompt_tokens: stats.prompt_tokens,
                                    completion_tokens: stats.completion_tokens,
                                    total_tokens: stats.prompt_tokens + stats.completion_tokens,
                                }),
                            };
                            let ev = sse_json_event(&chunk);
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 1, false, None, None, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                        GenerateEvent::Error(e) => {
                            let env = ErrorEnvelope { error: &e };
                            let ev = Event::default().event("error").data(
                                serde_json::to_string(&env).unwrap_or_else(|_| {
                                    "{\"error\":\"serialization failure\"}".to_string()
                                }),
                            );
                            // Transition to phase 1 (not 2) so the next
                            // iteration emits the `data: [DONE]` sentinel
                            // required by the OpenAI SSE protocol.
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 1, false, None, None, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                        // Tool calls are not part of the text completions protocol.
                        GenerateEvent::ToolCall { .. } => Some((
                            Ok(Event::default().comment("")),
                            (
                                rx,
                                id,
                                model,
                                created,
                                0,
                                first_token,
                                permit,
                                pm,
                                logprobs,
                                None,
                                metrics,
                            ),
                        )),
                        GenerateEvent::TokenLogprob { .. } => {
                            tracing::warn!(
                                "rest/routes: unexpected TokenLogprob event in text stream"
                            );
                            let env = ErrorEnvelope {
                                error: "unexpected token logprob event",
                            };
                            let ev = Event::default().event("error").data(
                                serde_json::to_string(&env).unwrap_or_else(|_| {
                                    "{\"error\":\"serialization failure\"}".to_string()
                                }),
                            );
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 1, false, None, None, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                    }
                }
            }
        },
    );

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}

/// Collect all tokens and return a complete text completion response.
#[allow(clippy::too_many_arguments)]
async fn text_blocking_response(
    mut rx: mpsc::Receiver<GenerateEvent>,
    model: String,
    collect_logprobs: bool,
    cache: Option<&crate::cache::ResponseCache>,
    cache_key: Option<String>,
    cache_ttl: Option<std::time::Duration>,
    cache_metrics: &crate::cache::CacheMetrics,
    metrics: &crate::metrics::MetricsStore,
    _cache_leader_guard: Option<crate::cache::CacheInflightLeaderGuard>,
    permit: SchedulerPermit,
    pm_permit: OwnedSemaphorePermit,
    queue_wait_us: u64,
) -> Response {
    use super::schema::{ChoiceLogprobs, LogprobContent, TopLogprob};
    let mut text = String::new();
    let mut usage = Usage::default();
    let mut generation_done = false;
    let mut finish_reason = FinishReason::Stop;
    let mut token_texts: Vec<String> = Vec::new();
    let mut logprob_entries: Vec<(f32, Vec<(String, f32)>)> = Vec::new();
    let mut pending_token: Option<String> = None;

    while let Some(event) = rx.recv().await {
        match event {
            GenerateEvent::Token(t) => {
                text.push_str(&t);
                if collect_logprobs {
                    pending_token = Some(t);
                }
            }
            GenerateEvent::TokenLogprob { logprob, top } => {
                if collect_logprobs && let Some(tok) = pending_token.take() {
                    token_texts.push(tok);
                    logprob_entries.push((logprob, top));
                }
            }
            GenerateEvent::ToolCall { .. } => {}
            GenerateEvent::Done(stats) => {
                record_generation_stats(metrics, &stats);
                usage = Usage {
                    prompt_tokens: stats.prompt_tokens as u32,
                    completion_tokens: stats.completion_tokens as u32,
                    total_tokens: (stats.prompt_tokens + stats.completion_tokens) as u32,
                };
                finish_reason = match stats.stop_reason.as_str() {
                    "length" => FinishReason::Length,
                    "content_filter" => FinishReason::ContentFilter,
                    _ => FinishReason::Stop,
                };
                generation_done = true;
                break;
            }
            GenerateEvent::Error(e) => {
                return with_timing(
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": e})),
                    )
                        .into_response(),
                    queue_wait_us,
                );
            }
        }
    }

    if !generation_done {
        return with_timing(
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "generation ended unexpectedly"})),
            )
                .into_response(),
            queue_wait_us,
        );
    }

    let choice_logprobs = if collect_logprobs && !token_texts.is_empty() {
        let content_lp: Vec<LogprobContent> = token_texts
            .into_iter()
            .zip(logprob_entries)
            .map(|(tok, (lp, top))| LogprobContent {
                bytes: Some(tok.as_bytes().to_vec()),
                token: tok,
                logprob: lp,
                top_logprobs: top
                    .into_iter()
                    .map(|(t, l)| TopLogprob {
                        bytes: Some(t.as_bytes().to_vec()),
                        token: t,
                        logprob: l,
                    })
                    .collect(),
            })
            .collect();
        Some(ChoiceLogprobs {
            content: content_lp,
        })
    } else {
        None
    };

    let response = CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4().simple()),
        object: "text_completion",
        created: unix_now(),
        model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: Some(finish_reason),
            logprobs: choice_logprobs,
        }],
        usage: Some(usage),
    };

    // Release both inflight slots before cache I/O (same as blocking_response).
    drop(pm_permit);
    drop(permit);

    if let (Some(cache), Some(key), Some(ttl)) = (cache, cache_key, cache_ttl) {
        write_cache_and_record(cache, &key, &response, ttl, cache_metrics, metrics).await;
    }

    with_timing(
        (StatusCode::OK, Json(response)).into_response(),
        queue_wait_us,
    )
}

/// POST /v1/embeddings — generate embedding vectors for one or more strings.
///
/// Supports the `llama_cpp` subprocess backend (batched POST to llama-server's
/// `/v1/embeddings`) and the `LibLlamaBackend`. Returns 501 if the backend does
/// not implement embeddings (e.g. the native ax-engine backend).
pub async fn embeddings(
    State(layer): State<Arc<ServingLayer>>,
    headers: HeaderMap,
    Json(req): Json<EmbeddingsRequest>,
) -> Response {
    use ax_serving_engine::{EmbedConfig, EmbedInput};
    use base64::{Engine as _, engine::general_purpose};

    if let Some(response) = validate_model_identifier(&req.model, "model") {
        return response;
    }
    match req.encoding_format.as_deref() {
        None | Some("float" | "base64") => {}
        Some(other) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unsupported encoding_format '{other}'; expected 'float' or 'base64'"
                    )
                })),
            )
                .into_response();
        }
    }
    if let Err(resp) =
        project_policy::enforce(&headers, &req.model, None, &layer.config.project_policy)
    {
        return resp.into_response();
    }
    let entry = match layer.registry.get(&req.model) {
        Some(e) => e,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": format!("model '{}' not loaded", req.model)})),
            )
                .into_response();
        }
    };

    let handle = entry.handle;
    let model_name = req.model.clone();
    let base64_out = req.encoding_format.as_deref() == Some("base64");
    let config = EmbedConfig {
        normalize: req.normalize.unwrap_or(true),
        truncate: req.truncate.unwrap_or(true),
    };

    // Resolve input into owned containers so they can cross the spawn_blocking boundary.
    let (strings_owned, tokens_owned) = match req.input {
        EmbeddingsInput::One(s) => (Some(vec![s]), None),
        EmbeddingsInput::Many(v) => (Some(v), None),
        EmbeddingsInput::OneTokens(t) => (None, Some(vec![t])),
        EmbeddingsInput::ManyTokens(t) => (None, Some(t)),
    };

    let result = tokio::task::spawn_blocking(move || match (strings_owned, tokens_owned) {
        (Some(texts), None) => layer
            .backend
            .embed(handle, &EmbedInput::Strings(&texts), &config),
        (None, Some(seqs)) => layer
            .backend
            .embed(handle, &EmbedInput::Tokens(&seqs), &config),
        _ => Err(anyhow::anyhow!(
            "embedding request payload had unsupported input state"
        )),
    })
    .await;

    match result {
        Ok(Ok(embed_result)) => {
            let data: Vec<EmbeddingObject> = embed_result
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(i, vec)| {
                    let embedding = if base64_out {
                        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
                        serde_json::Value::String(general_purpose::STANDARD.encode(&bytes))
                    } else {
                        serde_json::to_value(vec).unwrap_or(serde_json::Value::Array(vec![]))
                    };
                    EmbeddingObject {
                        object: "embedding",
                        index: i as u32,
                        embedding,
                    }
                })
                .collect();
            let pt = embed_result.prompt_tokens;
            Json(EmbeddingsResponse {
                object: "list",
                model: model_name,
                data,
                usage: EmbeddingUsage {
                    prompt_tokens: pt,
                    total_tokens: pt,
                },
            })
            .into_response()
        }
        Ok(Err(e)) => {
            // 501 only when the backend explicitly doesn't implement embeddings.
            // Real failures (HTTP errors, JSON parse, llama-server errors) → 500.
            let status = if e.to_string().contains("not supported") {
                StatusCode::NOT_IMPLEMENTED
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
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
