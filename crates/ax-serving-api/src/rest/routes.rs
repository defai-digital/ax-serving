//! Axum route handlers.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use ax_serving_engine::{
    BackendType, ChatMessage, GenerateEvent, GenerateInput, LoadConfig, current_rss_bytes,
};
use axum::Json;
use axum::extract::{Extension, Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{self, Stream};
use serde::Serialize;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::schema::*;
use super::validation::{
    build_generation_params, cache_ttl_err, resolve_grammar, resolve_logprobs, validate_max_tokens,
    validate_response_format, validate_sampling_params,
};
use crate::ServingLayer;
use crate::auth::RequestId;
use crate::cache::{CacheInflightEnter, CacheInflightLeaderGuard, CachePreference};
use crate::project_policy;
use crate::registry::RegistryError;
use crate::scheduler::{SchedulerError, SchedulerPermit};
use tokio::sync::OwnedSemaphorePermit;

/// Map a scheduler error to the correct HTTP status code.
///
/// - [`SchedulerError::QueueFull`] → 429 Too Many Requests (client should retry with back-off)
/// - All other variants → 503 Service Unavailable (server-side overload / shutdown)
fn scheduler_error_status(e: &anyhow::Error) -> StatusCode {
    match e.downcast_ref::<SchedulerError>() {
        Some(SchedulerError::QueueFull { .. }) => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    }
}

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

/// Characters per token for the heuristic fallback estimator (4 chars ≈ 1 token,
/// matching the GPT-3/4 rule-of-thumb for English text).
const CHARS_PER_TOKEN: u64 = 4;

/// Token overhead added per chat message to account for role/separator framing.
/// Matches the +4 tokens per message used by the OpenAI tiktoken cookbook.
const MESSAGE_FRAMING_TOKENS: u64 = 4;

fn estimated_tokens_from_text(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    chars.saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

fn estimate_chat_prompt_tokens(messages: &[InputMessage]) -> u64 {
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

fn estimate_text_prompt_tokens(prompt: &str) -> u64 {
    estimated_tokens_from_text(prompt).max(1)
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
    if req.model.len() > MAX_MODEL_ID_BYTES {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "model id too long"})),
        )
            .into_response();
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
    let effective_max_tokens: Option<u32> = req.max_tokens.or_else(|| {
        let d = layer.default_max_tokens;
        if d > 0 { Some(d) } else { None }
    });
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

    let handle = entry.handle;
    let model_name = req.model.clone();
    let cache_requested = req.cache.unwrap_or(CachePreference::Enable);
    let cache_active =
        cache_requested == CachePreference::Enable && !req.stream && layer.cache.is_some();
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
                                    return (
                                        StatusCode::OK,
                                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                                        hit_json,
                                    )
                                        .into_response();
                                }
                                layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!("cache read error: invalid cached JSON payload");
                            }
                            Ok(None) => {}
                            Err(e) => {
                                layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!("cache read error: {e}");
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
                layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("cache key generation error: {e}");
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
                        return (
                            StatusCode::OK,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            hit_json,
                        )
                            .into_response();
                    }
                    layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("cache read error: invalid cached JSON payload");
                }
                Ok(None) => {}
                Err(e) => {
                    layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("cache read error: {e}");
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
        estimate_chat_prompt_tokens(&req.messages)
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

                            let logprobs_payload = if logprobs {
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
                                        token: text.clone(),
                                        logprob: lp,
                                        bytes: text.as_bytes().to_vec(),
                                        top_logprobs,
                                    }]
                                } else {
                                    Vec::new()
                                };
                                Some(StreamLogprobs { content })
                            } else {
                                None
                            };

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
                                    rx, id, model, created, 1, false, None, None, logprobs, None,
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
                                    finish_reason: Some(match stats.stop_reason.as_str() {
                                        "length" => "length",
                                        "content_filter" => "content_filter",
                                        _ => "stop",
                                    }),
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
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 2, false, None, None, logprobs, None,
                                    metrics,
                                ),
                            ))
                        }
                        // TokenLogprob is handled above; this arm is unreachable.
                        GenerateEvent::TokenLogprob { .. } => unreachable!(),
                    }
                }
            }
        },
    );

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}

/// Inject `X-Ax-Stage-Timing: queue_wait_us=<N>` into any response.
///
/// Called on both the success path and all error paths so callers can always
/// correlate observed latency with time spent waiting in the admission queue,
/// regardless of whether the request eventually succeeded or failed.
#[inline]
fn with_timing(mut resp: Response, queue_wait_us: u64) -> Response {
    if let Ok(val) = HeaderValue::from_str(&format!("queue_wait_us={queue_wait_us}")) {
        resp.headers_mut().insert("x-ax-stage-timing", val);
    }
    resp
}

fn record_generation_stats(
    metrics: &crate::metrics::MetricsStore,
    stats: &ax_serving_engine::GenerationStats,
) {
    metrics.record_generation_stats(stats);
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
        if let Err(e) = cache.set(&key, &response, ttl).await {
            cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("cache write error: {e}");
        } else {
            metrics.record_cache_fill();
        }
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
    if req.model.len() > MAX_MODEL_ID_BYTES {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "model id too long"})),
        )
            .into_response();
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

    let effective_max_tokens: Option<u32> = req.max_tokens.or_else(|| {
        let d = layer.default_max_tokens;
        if d > 0 { Some(d) } else { None }
    });
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
                                    return (
                                        StatusCode::OK,
                                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                                        hit_json,
                                    )
                                        .into_response();
                                }
                                layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!("cache read error: invalid cached JSON payload");
                            }
                            Ok(None) => {}
                            Err(e) => {
                                layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!("cache read error: {e}");
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
                layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("cache key generation error: {e}");
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
                        return (
                            StatusCode::OK,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            hit_json,
                        )
                            .into_response();
                    }
                    layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("cache read error: invalid cached JSON payload");
                }
                Ok(None) => {}
                Err(e) => {
                    layer.cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("cache read error: {e}");
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
        estimate_text_prompt_tokens(&req.prompt)
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

                            let logprobs_payload = if logprobs {
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
                                        token: text.clone(),
                                        logprob: lp,
                                        bytes: text.as_bytes().to_vec(),
                                        top_logprobs,
                                    }]
                                } else {
                                    Vec::new()
                                };
                                Some(StreamLogprobs { content })
                            } else {
                                None
                            };

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
                                    finish_reason: Some(match stats.stop_reason.as_str() {
                                        "length" => "length",
                                        "content_filter" => "content_filter",
                                        _ => "stop",
                                    }),
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
                            Some((
                                Ok(ev),
                                (
                                    rx, id, model, created, 2, false, None, None, logprobs, None,
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
                        // TokenLogprob was resolved above; this arm is unreachable.
                        GenerateEvent::TokenLogprob { .. } => unreachable!(),
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
        if let Err(e) = cache.set(&key, &response, ttl).await {
            cache_metrics.errors.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("cache write error: {e}");
        } else {
            metrics.record_cache_fill();
        }
    }

    with_timing(
        (StatusCode::OK, Json(response)).into_response(),
        queue_wait_us,
    )
}

// ── Other routes ──────────────────────────────────────────────────────────────

/// GET /v1/models
pub async fn list_models(State(layer): State<Arc<ServingLayer>>) -> Json<ModelsResponse> {
    let ids = layer.registry.list_ids();
    let now = unix_now();
    Json(ModelsResponse {
        object: "list",
        data: ids
            .into_iter()
            .map(|id| ModelEntry {
                id,
                object: "model",
                created: now,
                owned_by: "ax-serving",
            })
            .collect(),
    })
}

/// GET /health
pub async fn health(State(layer): State<Arc<ServingLayer>>) -> Json<HealthResponse> {
    let thermal = layer.backend.thermal_state();
    let loaded_models = layer.registry.list_ids();
    let model_available = !loaded_models.is_empty();
    let ready = !matches!(thermal, ax_serving_engine::ThermalState::Critical);
    let reason = match (ready, model_available) {
        (false, false) => Some("thermal_critical_no_models"),
        (false, true) => Some("thermal_critical"),
        (true, false) => Some("no_models_loaded"),
        (true, true) => None,
    };
    let status = if ready && model_available {
        "ok"
    } else {
        "degraded"
    };

    Json(HealthResponse {
        status,
        ready,
        model_available,
        reason,
        thermal: thermal.as_str().to_string(),
        loaded_models: loaded_models.clone(),
        loaded_model_count: loaded_models.len(),
        uptime_secs: layer.metrics.uptime_secs(),
    })
}

/// `GET /v1/admin/startup-report` — authenticated runtime and config summary.
pub async fn admin_startup_report(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    Json(serving_startup_report_value(&layer))
}

/// `GET /v1/admin/status` — authenticated operational summary for the serving runtime.
pub async fn admin_status(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
) -> impl IntoResponse {
    let thermal = layer.backend.thermal_state();
    let loaded_models = layer.registry.list_ids();
    let status = if thermal.as_str() == "Critical" || loaded_models.is_empty() {
        "degraded"
    } else {
        "ok"
    };
    let scheduler = &layer.scheduler.metrics;
    let metrics = &layer.metrics;

    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "service": "serving",
        "status": status,
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "license": layer.license.to_json(),
        "runtime": {
            "rest_addr": layer.config.rest_addr,
            "grpc_socket": layer.config.grpc_socket,
            "cache_enabled": layer.cache.is_some(),
            "split_scheduler": layer.config.split_scheduler,
        },
        "models": {
            "loaded_model_count": loaded_models.len(),
            "loaded_models": loaded_models,
        },
        "scheduler": {
            "queue_depth": scheduler.queue_depth.load(Ordering::Relaxed),
            "inflight_count": scheduler.inflight_count.load(Ordering::Relaxed),
            "rejected_requests": scheduler.rejected_requests.load(Ordering::Relaxed),
            "effective_inflight_limit": layer.scheduler.effective_inflight_limit(),
        },
        "system": {
            "thermal": thermal.as_str(),
            "rss_bytes": current_rss_bytes(),
            "uptime_secs": metrics.uptime_secs(),
        }
    }))
}

/// GET /v1/metrics — scheduler and serving metrics (JSON).
pub async fn metrics(State(layer): State<Arc<ServingLayer>>) -> Json<serde_json::Value> {
    let m = &layer.scheduler.metrics;
    let cfg = layer.scheduler.config();

    // Per-model KV estimates — no last_accessed_ms update.
    let models_meta = layer.registry.loaded_models_with_meta();
    let model_kv: serde_json::Value = models_meta
        .iter()
        .map(|(id, meta)| {
            (
                id.clone(),
                serde_json::json!({ "estimated_kv_bytes": meta.estimated_kv_bytes() }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();

    // Single-snapshot percentiles: compute each histogram's tuple once so that
    // p50 ≤ p95 ≤ p99 is always maintained in the output. Three independent
    // calls to snapshot() can read different shard subsets under contention
    // (try_lock failures), making p50 > p99 possible.
    let (qw_p50, qw_p95, qw_p99) = m.queue_wait_percentiles_us();
    let (e2e_p50, e2e_p95, e2e_p99) = m.e2e_percentiles_us();
    let (ttft_p50, ttft_p95, ttft_p99) = m.ttft_percentiles_us();

    Json(serde_json::json!({
        "scheduler": {
            "queue_depth": m.queue_depth.load(Ordering::Relaxed),
            "inflight_count": m.inflight_count.load(Ordering::Relaxed),
            "total_requests": m.total_requests.load(Ordering::Relaxed),
            "rejected_requests": m.rejected_requests.load(Ordering::Relaxed),
            "queued_requests": m.queued_requests.load(Ordering::Relaxed),
            "avg_queue_wait_us": m.avg_queue_wait_us(),
            "queue_wait_p50_us": qw_p50,
            "queue_wait_p95_us": qw_p95,
            "queue_wait_p99_us": qw_p99,
            "e2e_p50_us": e2e_p50,
            "e2e_p95_us": e2e_p95,
            "e2e_p99_us": e2e_p99,
            "cache_follower_waiting": m.cache_follower_waiting.load(Ordering::Relaxed),
            "prefill_tokens_active": m.prefill_tokens_active.load(Ordering::Relaxed),
            "decode_sequences_active": m.decode_sequences_active.load(Ordering::Relaxed),
            "ttft_p50_us": ttft_p50,
            "ttft_p95_us": ttft_p95,
            "ttft_p99_us": ttft_p99,
            "effective_inflight_limit": layer.scheduler.effective_inflight_limit(),
            "adaptive_target_p99_ms": layer.scheduler.adaptive_target_p99_ms(),
            "split_scheduler_enabled": layer.scheduler.split_enabled,
            "max_inflight": cfg.max_inflight,
            "max_queue": cfg.max_queue,
            "max_wait_ms": cfg.max_wait_ms,
        },
        "uptime_secs": layer.metrics.uptime_secs(),
        "loaded_models": layer.registry.list_ids(),
        "thermal": layer.backend.thermal_state().as_str(),
        "rss_bytes": current_rss_bytes(),
        "models": model_kv,
        "cache": {
            "enabled": layer.cache.is_some(),
            "hits": layer.cache_metrics.hits.load(Ordering::Relaxed),
            "misses": layer.cache_metrics.misses.load(Ordering::Relaxed),
            "writes": layer.cache_metrics.writes.load(Ordering::Relaxed),
            "errors": layer.cache_metrics.errors.load(Ordering::Relaxed),
        },
        "request_classes": {
            "cold_requests_total": layer.metrics.cold_requests_total(),
            "exact_cache_hits_total": layer.metrics.exact_cache_hits_total(),
            "cache_follower_hits_total": layer.metrics.cache_follower_hits_total(),
            "cache_fills_total": layer.metrics.cache_fills_total(),
        }
    }))
}

/// `GET /v1/admin/diagnostics` — authenticated diagnostics bundle.
pub async fn admin_diagnostics(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
) -> impl IntoResponse {
    let health = health(State(Arc::clone(&layer))).await;
    let metrics = metrics(State(Arc::clone(&layer))).await;
    let models = list_models(State(Arc::clone(&layer))).await;
    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "generated_at": unix_now(),
        "startup_report": serving_startup_report_value(&layer),
        "health": health.0,
        "metrics": metrics.0,
        "models": models.0,
        "audit_tail": layer.audit.tail(50),
    }))
}

/// `GET /v1/admin/policy` — authenticated project-policy summary.
pub async fn admin_policy(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    Json(project_policy::summary_json(&layer.config.project_policy))
}

/// `GET /v1/admin/audit` — authenticated recent audit events.
pub async fn admin_audit(
    State(layer): State<Arc<ServingLayer>>,
    Query(query): Query<AuditQuery>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "events": layer.audit.tail(query.limit.clamp(1, 200)),
    }))
}

/// GET /metrics — Prometheus scrape endpoint.
///
/// Emits all serving metrics in Prometheus text format (version 0.0.4).
/// Content-Type: `text/plain; version=0.0.4`
pub async fn prometheus_metrics(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    let m = &layer.scheduler.metrics;
    // Single-snapshot percentile tuples — prevents p50 > p99 under shard contention.
    let (qw_p50, qw_p95, qw_p99) = m.queue_wait_percentiles_us();
    let (e2e_p50, e2e_p95, e2e_p99) = m.e2e_percentiles_us();
    let (ttft_p50, ttft_p95, ttft_p99) = m.ttft_percentiles_us();
    let thermal_val = layer.backend.thermal_state() as u64;
    let models_meta = layer.registry.loaded_models_with_meta();
    let loaded_count = models_meta.len();

    // Pre-allocate enough for all metric lines; avoids re-allocations on the hot path.
    const PROMETHEUS_BUF_CAPACITY: usize = 2048;
    let mut buf = String::with_capacity(PROMETHEUS_BUF_CAPACITY);

    // ── Scheduler ──────────────────────────────────────────────────────────
    buf.push_str("# HELP axs_scheduler_queue_depth Current request queue depth\n");
    buf.push_str("# TYPE axs_scheduler_queue_depth gauge\n");
    buf.push_str(&format!(
        "axs_scheduler_queue_depth {}\n",
        m.queue_depth.load(Ordering::Relaxed)
    ));

    buf.push_str("# HELP axs_scheduler_inflight_count Active inference requests\n");
    buf.push_str("# TYPE axs_scheduler_inflight_count gauge\n");
    buf.push_str(&format!(
        "axs_scheduler_inflight_count {}\n",
        m.inflight_count.load(Ordering::Relaxed)
    ));

    buf.push_str("# HELP axs_scheduler_total_requests_total Total requests received\n");
    buf.push_str("# TYPE axs_scheduler_total_requests_total counter\n");
    buf.push_str(&format!(
        "axs_scheduler_total_requests_total {}\n",
        m.total_requests.load(Ordering::Relaxed)
    ));

    buf.push_str(
        "# HELP axs_scheduler_rejected_requests_total Total requests rejected (queue full)\n",
    );
    buf.push_str("# TYPE axs_scheduler_rejected_requests_total counter\n");
    buf.push_str(&format!(
        "axs_scheduler_rejected_requests_total {}\n",
        m.rejected_requests.load(Ordering::Relaxed)
    ));

    buf.push_str(
        "# HELP axs_scheduler_queued_requests_total Requests that entered the slow-path wait queue\n",
    );
    buf.push_str("# TYPE axs_scheduler_queued_requests_total counter\n");
    buf.push_str(&format!(
        "axs_scheduler_queued_requests_total {}\n",
        m.queued_requests.load(Ordering::Relaxed)
    ));

    buf.push_str(
        "# HELP axs_scheduler_avg_queue_wait_us Average queue wait time in microseconds\n",
    );
    buf.push_str("# TYPE axs_scheduler_avg_queue_wait_us gauge\n");
    buf.push_str(&format!(
        "axs_scheduler_avg_queue_wait_us {}\n",
        m.avg_queue_wait_us()
    ));

    buf.push_str(
        "# HELP axs_scheduler_queue_wait_p50_us Rolling P50 queue wait in microseconds (slow-path only)\n",
    );
    buf.push_str("# TYPE axs_scheduler_queue_wait_p50_us gauge\n");
    buf.push_str(&format!("axs_scheduler_queue_wait_p50_us {qw_p50}\n"));

    buf.push_str(
        "# HELP axs_scheduler_queue_wait_p95_us Rolling P95 queue wait in microseconds (slow-path only)\n",
    );
    buf.push_str("# TYPE axs_scheduler_queue_wait_p95_us gauge\n");
    buf.push_str(&format!("axs_scheduler_queue_wait_p95_us {qw_p95}\n"));

    buf.push_str(
        "# HELP axs_scheduler_queue_wait_p99_us Rolling P99 queue wait in microseconds (slow-path only)\n",
    );
    buf.push_str("# TYPE axs_scheduler_queue_wait_p99_us gauge\n");
    buf.push_str(&format!("axs_scheduler_queue_wait_p99_us {qw_p99}\n"));

    buf.push_str(
        "# HELP axs_scheduler_e2e_p50_us Rolling P50 end-to-end latency in microseconds\n",
    );
    buf.push_str("# TYPE axs_scheduler_e2e_p50_us gauge\n");
    buf.push_str(&format!("axs_scheduler_e2e_p50_us {e2e_p50}\n"));

    buf.push_str(
        "# HELP axs_scheduler_e2e_p95_us Rolling P95 end-to-end latency in microseconds\n",
    );
    buf.push_str("# TYPE axs_scheduler_e2e_p95_us gauge\n");
    buf.push_str(&format!("axs_scheduler_e2e_p95_us {e2e_p95}\n"));

    buf.push_str(
        "# HELP axs_scheduler_e2e_p99_us Rolling P99 end-to-end latency in microseconds\n",
    );
    buf.push_str("# TYPE axs_scheduler_e2e_p99_us gauge\n");
    buf.push_str(&format!("axs_scheduler_e2e_p99_us {e2e_p99}\n"));

    buf.push_str(
        "# HELP axs_cache_follower_waiting Cache followers currently waiting pre-permit (WS3)\n",
    );
    buf.push_str("# TYPE axs_cache_follower_waiting gauge\n");
    buf.push_str(&format!(
        "axs_cache_follower_waiting {}\n",
        m.cache_follower_waiting.load(Ordering::Relaxed)
    ));

    buf.push_str(
        "# HELP axs_scheduler_prefill_tokens_active Estimated prompt tokens currently in prefill\n",
    );
    buf.push_str("# TYPE axs_scheduler_prefill_tokens_active gauge\n");
    buf.push_str(&format!(
        "axs_scheduler_prefill_tokens_active {}\n",
        m.prefill_tokens_active.load(Ordering::Relaxed)
    ));

    buf.push_str(
        "# HELP axs_scheduler_decode_sequences_active Active sequences currently in decode\n",
    );
    buf.push_str("# TYPE axs_scheduler_decode_sequences_active gauge\n");
    buf.push_str(&format!(
        "axs_scheduler_decode_sequences_active {}\n",
        m.decode_sequences_active.load(Ordering::Relaxed)
    ));

    buf.push_str(
        "# HELP axs_ttft_p50_us Rolling P50 time-to-first-token in microseconds (streaming only)\n",
    );
    buf.push_str("# TYPE axs_ttft_p50_us gauge\n");
    buf.push_str(&format!("axs_ttft_p50_us {ttft_p50}\n"));

    buf.push_str(
        "# HELP axs_ttft_p95_us Rolling P95 time-to-first-token in microseconds (streaming only)\n",
    );
    buf.push_str("# TYPE axs_ttft_p95_us gauge\n");
    buf.push_str(&format!("axs_ttft_p95_us {ttft_p95}\n"));

    buf.push_str(
        "# HELP axs_ttft_p99_us Rolling P99 time-to-first-token in microseconds (streaming only)\n",
    );
    buf.push_str("# TYPE axs_ttft_p99_us gauge\n");
    buf.push_str(&format!("axs_ttft_p99_us {ttft_p99}\n"));

    buf.push_str(
        "# HELP axs_adaptive_inflight_limit Effective inflight limit (adaptive or static)\n",
    );
    buf.push_str("# TYPE axs_adaptive_inflight_limit gauge\n");
    buf.push_str(&format!(
        "axs_adaptive_inflight_limit {}\n",
        layer.scheduler.effective_inflight_limit()
    ));

    buf.push_str(
        "# HELP axs_adaptive_target_p99_ms AIMD target P99 latency in milliseconds (0 = disabled)\n",
    );
    buf.push_str("# TYPE axs_adaptive_target_p99_ms gauge\n");
    buf.push_str(&format!(
        "axs_adaptive_target_p99_ms {}\n",
        layer.scheduler.adaptive_target_p99_ms().unwrap_or(0)
    ));

    buf.push_str(
        "# HELP axs_request_class_cold_requests_total Requests that executed inference without a cache result\n",
    );
    buf.push_str("# TYPE axs_request_class_cold_requests_total counter\n");
    buf.push_str(&format!(
        "axs_request_class_cold_requests_total {}\n",
        layer.metrics.cold_requests_total()
    ));

    buf.push_str(
        "# HELP axs_request_class_exact_cache_hits_total Requests served immediately from exact response-cache hits\n",
    );
    buf.push_str("# TYPE axs_request_class_exact_cache_hits_total counter\n");
    buf.push_str(&format!(
        "axs_request_class_exact_cache_hits_total {}\n",
        layer.metrics.exact_cache_hits_total()
    ));

    buf.push_str(
        "# HELP axs_request_class_cache_follower_hits_total Requests served from follower waits after a leader cache fill\n",
    );
    buf.push_str("# TYPE axs_request_class_cache_follower_hits_total counter\n");
    buf.push_str(&format!(
        "axs_request_class_cache_follower_hits_total {}\n",
        layer.metrics.cache_follower_hits_total()
    ));

    buf.push_str(
        "# HELP axs_request_class_cache_fills_total Successful exact response-cache writes after inference\n",
    );
    buf.push_str("# TYPE axs_request_class_cache_fills_total counter\n");
    buf.push_str(&format!(
        "axs_request_class_cache_fills_total {}\n",
        layer.metrics.cache_fills_total()
    ));

    // ── Thermal ────────────────────────────────────────────────────────────
    buf.push_str(
        "# HELP axs_thermal_state Thermal pressure state (0=Nominal 1=Fair 2=Serious 3=Critical)\n",
    );
    buf.push_str("# TYPE axs_thermal_state gauge\n");
    buf.push_str(&format!("axs_thermal_state {thermal_val}\n"));

    // ── Cache ──────────────────────────────────────────────────────────────
    buf.push_str("# HELP axs_cache_hits_total Response cache hits\n");
    buf.push_str("# TYPE axs_cache_hits_total counter\n");
    buf.push_str(&format!(
        "axs_cache_hits_total {}\n",
        layer.cache_metrics.hits.load(Ordering::Relaxed)
    ));

    buf.push_str("# HELP axs_cache_misses_total Response cache misses\n");
    buf.push_str("# TYPE axs_cache_misses_total counter\n");
    buf.push_str(&format!(
        "axs_cache_misses_total {}\n",
        layer.cache_metrics.misses.load(Ordering::Relaxed)
    ));

    buf.push_str("# HELP axs_cache_writes_total Response cache writes\n");
    buf.push_str("# TYPE axs_cache_writes_total counter\n");
    buf.push_str(&format!(
        "axs_cache_writes_total {}\n",
        layer.cache_metrics.writes.load(Ordering::Relaxed)
    ));

    {
        let hits = layer.cache_metrics.hits.load(Ordering::Relaxed);
        let misses = layer.cache_metrics.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let ratio = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        buf.push_str(
            "# HELP axs_cache_hit_ratio Response cache hit ratio since startup (0.0–1.0)\n",
        );
        buf.push_str("# TYPE axs_cache_hit_ratio gauge\n");
        buf.push_str(&format!("axs_cache_hit_ratio {ratio:.4}\n"));
    }

    // ── System ────────────────────────────────────────────────────────────
    buf.push_str("# HELP axs_uptime_seconds Seconds since server start\n");
    buf.push_str("# TYPE axs_uptime_seconds counter\n");
    buf.push_str(&format!(
        "axs_uptime_seconds {}\n",
        layer.metrics.uptime_secs()
    ));

    buf.push_str("# HELP axs_rss_bytes Process resident set size in bytes\n");
    buf.push_str("# TYPE axs_rss_bytes gauge\n");
    buf.push_str(&format!("axs_rss_bytes {}\n", current_rss_bytes()));

    // ── Models ────────────────────────────────────────────────────────────
    buf.push_str("# HELP axs_loaded_models_total Number of currently loaded models\n");
    buf.push_str("# TYPE axs_loaded_models_total gauge\n");
    buf.push_str(&format!("axs_loaded_models_total {loaded_count}\n"));

    buf.push_str("# HELP axs_model_kv_bytes_estimated Estimated KV cache bytes for full context\n");
    buf.push_str("# TYPE axs_model_kv_bytes_estimated gauge\n");
    for (id, meta) in &models_meta {
        buf.push_str(&format!(
            "axs_model_kv_bytes_estimated{{model=\"{id}\"}} {}\n",
            meta.estimated_kv_bytes()
        ));
    }

    // ── SLO alerting gauges ────────────────────────────────────────────────
    // 1 = SLO met (pass), 0 = SLO violated (fail).
    // Thresholds are read from env at scrape time so they can be tuned without
    // restarting the server (only takes effect on the next scrape).
    let slo_e2e_p99_ms = std::env::var("AXS_SLO_E2E_P99_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5_000);
    let slo_queue_p99_ms = std::env::var("AXS_SLO_QUEUE_P99_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1_000);
    let slo_max_error_rate = std::env::var("AXS_SLO_MAX_ERROR_RATE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.05);

    let e2e_p99_us = m.e2e_p99_us();
    let queue_p99_us = m.queue_wait_p99_us();
    let total = m.total_requests.load(Ordering::Relaxed);
    let rejected = m.rejected_requests.load(Ordering::Relaxed);
    let error_rate = if total > 0 {
        rejected as f64 / total as f64
    } else {
        0.0
    };

    // Require at least one completed request before reporting pass.
    // Without this guard all three gauges would be 1 at startup / idle,
    // which is a false positive that masks misconfigured alerting rules.
    let have_data = total > 0;
    let e2e_pass = u8::from(have_data && e2e_p99_us <= slo_e2e_p99_ms * 1_000);
    let queue_pass = u8::from(have_data && queue_p99_us <= slo_queue_p99_ms * 1_000);
    let error_pass = u8::from(have_data && error_rate <= slo_max_error_rate);

    buf.push_str("# HELP axs_slo_e2e_p99_pass 1 if e2e P99 latency is within SLO, 0 otherwise\n");
    buf.push_str("# TYPE axs_slo_e2e_p99_pass gauge\n");
    buf.push_str(&format!("axs_slo_e2e_p99_pass {e2e_pass}\n"));

    buf.push_str("# HELP axs_slo_queue_p99_pass 1 if queue-wait P99 is within SLO, 0 otherwise\n");
    buf.push_str("# TYPE axs_slo_queue_p99_pass gauge\n");
    buf.push_str(&format!("axs_slo_queue_p99_pass {queue_pass}\n"));

    buf.push_str("# HELP axs_slo_error_rate_pass 1 if rejection rate is within SLO, 0 otherwise\n");
    buf.push_str("# TYPE axs_slo_error_rate_pass gauge\n");
    buf.push_str(&format!("axs_slo_error_rate_pass {error_pass}\n"));

    // ── Burn-rate alerting ────────────────────────────────────────────────────
    // Multi-window SLO burn rate.  error_budget = 0.001 (99.9% availability).
    // Fast burn: 1h window > 14.4× → 2% budget in 1h   (Google SRE chapter 5).
    // Slow burn: 6h window >  6.0× → 5% budget in 6h.
    let burn_1h = layer.metrics.burn_1h.lock().unwrap().burn_rate(0.001);
    let burn_6h = layer.metrics.burn_6h.lock().unwrap().burn_rate(0.001);
    let burn_alert = u8::from((burn_1h > 14.4) || (burn_6h > 6.0));

    buf.push_str("# HELP axs_slo_burn_rate_1h SLO error burn rate over 1-hour sliding window\n");
    buf.push_str("# TYPE axs_slo_burn_rate_1h gauge\n");
    buf.push_str(&format!("axs_slo_burn_rate_1h {burn_1h}\n"));

    buf.push_str("# HELP axs_slo_burn_rate_6h SLO error burn rate over 6-hour sliding window\n");
    buf.push_str("# TYPE axs_slo_burn_rate_6h gauge\n");
    buf.push_str(&format!("axs_slo_burn_rate_6h {burn_6h}\n"));

    buf.push_str(
        "# HELP axs_slo_burn_rate_alert 1 if multi-window burn-rate alert is firing, 0 otherwise\n",
    );
    buf.push_str("# TYPE axs_slo_burn_rate_alert gauge\n");
    buf.push_str(&format!("axs_slo_burn_rate_alert {burn_alert}\n"));

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        buf,
    )
}

// ── Model management ──────────────────────────────────────────────────────────

/// POST /v1/models — load a model from a GGUF file.
pub async fn rest_load_model(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let pooling_type = match req.pooling_type.as_deref() {
        Some(raw) => match normalize_pooling_type(raw) {
            Some(v) => Some(v),
            None => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(serde_json::json!({
                        "error": "invalid pooling_type; expected one of: none, mean, cls, last, rank"
                    })),
                )
                    .into_response();
            }
        },
        None => None,
    };

    let path = std::path::PathBuf::from(&req.path);
    let config = LoadConfig {
        context_length: req.context_length.unwrap_or(0),
        backend_type: BackendType::Auto,
        llama_cpp_n_gpu_layers: req.n_gpu_layers,
        mmproj_path: req.mmproj_path.clone(),
        backend_hint: req.backend.clone(),
        enable_embeddings: req.enable_embeddings,
        pooling_type,
    };
    let model_id = req.model_id.clone();
    let layer_for_load = Arc::clone(&layer);

    let result = tokio::task::spawn_blocking(move || {
        layer_for_load
            .registry
            .load(&model_id, &path, config, layer_for_load.backend.as_ref())
    })
    .await;

    match result {
        Ok(Ok(entry)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_load",
                "model",
                Some(entry.id.clone()),
                "ok",
                Some(serde_json::json!({
                    "path": entry.path.display().to_string(),
                    "architecture": entry.metadata.architecture,
                })),
            );
            let (ready, model_available, loaded_model_count) = lifecycle_snapshot(&layer);
            (
                StatusCode::CREATED,
                Json(LoadModelResponse {
                    model_id: entry.id.clone(),
                    state: "loaded",
                    ready,
                    model_available,
                    loaded_model_count,
                    architecture: entry.metadata.architecture.clone(),
                    context_length: entry.metadata.context_length,
                    load_time_ms: entry.metadata.load_time_ms,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_load",
                "model",
                Some(req.model_id.clone()),
                "error",
                Some(serde_json::json!({
                    "path": req.path,
                    "error": e.to_string(),
                })),
            );
            let status = match e.downcast_ref::<RegistryError>() {
                Some(RegistryError::AlreadyLoaded(_)) => StatusCode::CONFLICT,
                Some(
                    RegistryError::FileNotFound(_)
                    | RegistryError::InvalidFormat(_)
                    | RegistryError::InvalidModelId(_),
                ) => StatusCode::UNPROCESSABLE_ENTITY,
                Some(RegistryError::PathNotAllowed(_)) => StatusCode::FORBIDDEN,
                Some(RegistryError::CapacityExceeded(_)) => StatusCode::SERVICE_UNAVAILABLE,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
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

fn is_valid_pooling_type(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "none" | "mean" | "cls" | "last" | "rank"
    )
}

fn normalize_pooling_type(s: &str) -> Option<String> {
    let canonical = s.trim().to_ascii_lowercase();
    if is_valid_pooling_type(&canonical) {
        Some(canonical)
    } else {
        None
    }
}

fn lifecycle_snapshot(layer: &Arc<ServingLayer>) -> (bool, bool, usize) {
    let loaded_model_count = layer.registry.list_ids().len();
    let model_available = loaded_model_count > 0;
    let ready = !matches!(
        layer.backend.thermal_state(),
        ax_serving_engine::ThermalState::Critical
    );
    (ready, model_available, loaded_model_count)
}

/// DELETE /v1/models/:id — unload a loaded model.
pub async fn rest_unload_model(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(model_id): Path<String>,
) -> Response {
    let id_for_response = model_id.clone();
    let layer_for_unload = Arc::clone(&layer);
    let result = tokio::task::spawn_blocking(move || {
        layer_for_unload
            .registry
            .unload(&model_id, layer_for_unload.backend.as_ref())
    })
    .await;

    match result {
        Ok(Ok(())) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_unload",
                "model",
                Some(id_for_response.clone()),
                "ok",
                None,
            );
            let (ready, model_available, loaded_model_count) = lifecycle_snapshot(&layer);
            (
                StatusCode::OK,
                Json(UnloadModelResponse {
                    model_id: id_for_response,
                    state: "unloaded",
                    ready,
                    model_available,
                    loaded_model_count,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_unload",
                "model",
                Some(id_for_response.clone()),
                "error",
                Some(serde_json::json!({ "error": e.to_string() })),
            );
            let status = match e.downcast_ref::<RegistryError>() {
                Some(RegistryError::NotLoaded(_)) => StatusCode::NOT_FOUND,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
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

/// POST /v1/models/:id/reload — atomically reload from the same path and config.
pub async fn rest_reload_model(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(model_id): Path<String>,
) -> Response {
    let layer_for_reload = Arc::clone(&layer);
    let reload_id = model_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        layer_for_reload
            .registry
            .reload(&reload_id, layer_for_reload.backend.as_ref())
    })
    .await;

    match result {
        Ok(Ok(entry)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_reload",
                "model",
                Some(entry.id.clone()),
                "ok",
                Some(serde_json::json!({
                    "path": entry.path.display().to_string(),
                    "architecture": entry.metadata.architecture,
                })),
            );
            let (ready, model_available, loaded_model_count) = lifecycle_snapshot(&layer);
            (
                StatusCode::OK,
                Json(ReloadModelResponse {
                    model_id: entry.id.clone(),
                    state: "loaded",
                    ready,
                    model_available,
                    loaded_model_count,
                    architecture: entry.metadata.architecture.clone(),
                    load_time_ms: entry.metadata.load_time_ms,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            layer.audit.record(
                audit_actor(req_id),
                "model_reload",
                "model",
                Some(model_id.clone()),
                "error",
                Some(serde_json::json!({ "error": e.to_string() })),
            );
            let status = match e.downcast_ref::<RegistryError>() {
                Some(RegistryError::NotLoaded(_)) => StatusCode::NOT_FOUND,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
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

/// Normalized form of a single chat message used exclusively for cache-key
/// construction.  Role is lowercased and content is whitespace-trimmed so
/// that minor client-side formatting differences never cause a false miss.
#[derive(Serialize)]
struct NormalizedMessage {
    role: String,
    content: String,
}

/// Cache key payload (v2).
///
/// # Normalization rules
/// - `requested_model_id` is dropped: the resolved path already uniquely
///   identifies the weights, so model-alias changes no longer bust the cache.
/// - `messages`: role lowercased, content trimmed (leading/trailing whitespace).
/// - Floating-point params serialized with 4-decimal precision to absorb f32
///   representation noise (`0.6999999` and `0.7` both become `"0.7000"`).
#[derive(Serialize)]
struct CacheKeyPayload<'a> {
    version: &'static str,
    resolved_model_path: &'a str,
    resolved_model_arch: &'a str,
    messages: Vec<NormalizedMessage>,
    temperature: String,
    top_p: String,
    min_p: Option<String>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
    repeat_penalty: String,
}

/// Build the raw bytes that are fed into the SHA-256 cache key.
///
/// `effective_max_tokens` should already have the server-side default applied
/// (so requests without `max_tokens` and requests with `max_tokens=default`
/// share the same cache entry).
fn build_cache_key(
    req: &ChatCompletionRequest,
    resolved_model_path: &str,
    resolved_model_arch: &str,
    effective_max_tokens: Option<u32>,
) -> anyhow::Result<Vec<u8>> {
    let messages = req
        .messages
        .iter()
        .map(|m| NormalizedMessage {
            role: m.role.to_lowercase(),
            content: m.content.as_text().trim().to_string(),
        })
        .collect();

    let payload = CacheKeyPayload {
        version: "v2",
        resolved_model_path,
        resolved_model_arch,
        messages,
        temperature: format!("{:.4}", req.temperature),
        top_p: format!("{:.4}", req.top_p),
        min_p: req.min_p.map(|v| format!("{v:.4}")),
        top_k: req.top_k,
        max_tokens: effective_max_tokens,
        seed: req.seed,
        repeat_penalty: format!("{:.4}", req.repeat_penalty),
    };
    serde_json::to_vec(&payload).map_err(anyhow::Error::from)
}

/// Cache key payload for text completions (`POST /v1/completions`).
///
/// Normalisation mirrors `CacheKeyPayload`: prompt is trimmed, floats at 4dp.
#[derive(Serialize)]
struct TextCacheKeyPayload<'a> {
    version: &'static str,
    kind: &'static str,
    resolved_model_path: &'a str,
    resolved_model_arch: &'a str,
    prompt: &'a str,
    temperature: String,
    top_p: String,
    min_p: Option<String>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
    repeat_penalty: String,
}

/// Build the raw bytes for the SHA-256 text-completion cache key.
fn build_text_cache_key(
    req: &CompletionRequest,
    resolved_model_path: &str,
    resolved_model_arch: &str,
    effective_max_tokens: Option<u32>,
) -> anyhow::Result<Vec<u8>> {
    let payload = TextCacheKeyPayload {
        version: "v1",
        kind: "text_completion",
        resolved_model_path,
        resolved_model_arch,
        prompt: req.prompt.trim(),
        temperature: format!("{:.4}", req.temperature),
        top_p: format!("{:.4}", req.top_p),
        min_p: req.min_p.map(|v| format!("{v:.4}")),
        top_k: req.top_k,
        max_tokens: effective_max_tokens,
        seed: req.seed,
        repeat_penalty: format!("{:.4}", req.repeat_penalty),
    };
    serde_json::to_vec(&payload).map_err(anyhow::Error::from)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(serde::Deserialize)]
pub struct AuditQuery {
    #[serde(default = "default_audit_limit")]
    limit: usize,
}

fn default_audit_limit() -> usize {
    50
}

fn audit_actor(req_id: Option<Extension<RequestId>>) -> String {
    req_id
        .map(|id| format!("request:{}", id.0.0))
        .unwrap_or_else(|| "request:unknown".to_string())
}

fn serving_startup_report_value(layer: &Arc<ServingLayer>) -> serde_json::Value {
    let config_validation = layer.config.validate().err().map(|e| e.to_string());
    let allowed_model_dirs = std::env::var("AXS_MODEL_ALLOWED_DIRS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    serde_json::json!({
        "service": "serving",
        "status": if config_validation.is_none() { "ok" } else { "degraded" },
        "config_valid": config_validation.is_none(),
        "config_error": config_validation,
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "license": layer.license.to_json(),
        "runtime": {
            "rest_addr": layer.config.rest_addr,
            "grpc_socket": layer.config.grpc_socket,
            "grpc_host": layer.config.grpc_host,
            "grpc_port": layer.config.grpc_port,
            "cache_enabled": layer.config.cache.enabled,
            "split_scheduler": layer.config.split_scheduler,
            "default_max_tokens": layer.config.default_max_tokens,
            "idle_timeout_secs": layer.config.idle_timeout_secs,
            "thermal_poll_secs": layer.config.thermal_poll_secs,
            "sched_max_inflight": layer.config.sched_max_inflight,
            "sched_max_queue": layer.config.sched_max_queue,
        },
        "scheduler": {
            "effective_inflight_limit": layer.scheduler.effective_inflight_limit(),
            "split_scheduler_enabled": layer.scheduler.split_enabled,
            "scheduler_managed_batching": false,
            "batch_hints_advisory_only": true,
            "max_batch_size_hint": layer.config.sched_max_batch_size,
            "batch_window_ms_hint": layer.config.sched_batch_window_ms,
        },
        "cache": {
            "enabled": layer.cache.is_some(),
            "mode": if layer.cache.is_some() { "exact_response" } else { "disabled" },
            "kv_prefix_cache": false,
        },
        "trust": {
            "allowed_model_dirs": allowed_model_dirs,
        },
        "project_policy": project_policy::summary_json(&layer.config.project_policy),
        "governance": {
            "project_policy_enabled": layer.config.project_policy.enabled,
        }
    })
}

// ── Dashboard + License handlers ──────────────────────────────────────────────

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

    if req.model.len() > MAX_MODEL_ID_BYTES {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "model id too long"})),
        )
            .into_response();
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

    let result = tokio::task::spawn_blocking(move || {
        if let Some(texts) = strings_owned {
            layer
                .backend
                .embed(handle, &EmbedInput::Strings(&texts), &config)
        } else {
            let seqs = tokens_owned.unwrap();
            layer
                .backend
                .embed(handle, &EmbedInput::Tokens(&seqs), &config)
        }
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

/// `GET /dashboard` — embedded monitoring dashboard (no auth required).
pub async fn dashboard() -> impl IntoResponse {
    axum::response::Html(include_str!("../dashboard.html"))
}

/// `GET /v1/license` — current license state (no auth required).
pub async fn get_license(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    Json(layer.license.to_json())
}

/// `POST /v1/license` — activate a license key (no auth required).
///
/// Body: `{"key": "<license-key>"}`
pub async fn set_license(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let Some(key) = body.get("key").and_then(|v| v.as_str()) else {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "missing field: key"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "missing field: key"})),
        )
            .into_response();
    };
    let key = key.trim().to_string();
    if key.is_empty() {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "key must not be empty"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "key must not be empty"})),
        )
            .into_response();
    }
    match layer.license.set_key(key) {
        Ok(()) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "ok",
                None,
            );
            Json(layer.license.to_json()).into_response()
        }
        Err(e) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "error",
                Some(serde_json::json!({"error": e.to_string()})),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_req(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "default".into(),
            messages: vec![InputMessage {
                role: "user".into(),
                content: MessageContent::Text(content.into()),
                name: None,
            }],
            stream: false,
            temperature: 0.0,
            max_tokens: Some(16),
            top_p: 1.0,
            min_p: None,
            top_k: Some(1),
            seed: None,
            repeat_penalty: 1.1,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            grammar: None,
            response_format: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tools: None,
            tool_choice: None,
            cache: Some(CachePreference::Enable),
            cache_ttl: Some("1h".into()),
            logprobs: None,
            top_logprobs: None,
        }
    }

    #[test]
    fn cache_key_changes_for_different_resolved_models() {
        let req = mk_req("same prompt");
        let k1 = build_cache_key(&req, "models/Qwen3-8B-Q4_K_M.gguf", "qwen3", Some(16)).unwrap();
        let k2 = build_cache_key(
            &req,
            "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            "llama",
            Some(16),
        )
        .unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_stable_for_same_inputs() {
        let req = mk_req("same prompt");
        let k1 = build_cache_key(&req, "models/Qwen3-8B-Q4_K_M.gguf", "qwen3", Some(16)).unwrap();
        let k2 = build_cache_key(&req, "models/Qwen3-8B-Q4_K_M.gguf", "qwen3", Some(16)).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_normalizes_message_whitespace() {
        let r1 = mk_req("Hello world");
        let r2 = mk_req("  Hello world  ");
        let k1 = build_cache_key(&r1, "model.gguf", "llama", Some(16)).unwrap();
        let k2 = build_cache_key(&r2, "model.gguf", "llama", Some(16)).unwrap();
        assert_eq!(
            k1, k2,
            "leading/trailing whitespace must not affect cache key"
        );
    }

    #[test]
    fn cache_key_normalizes_role_case() {
        let mut r1 = mk_req("Hello");
        r1.messages[0].role = "User".into();
        let mut r2 = mk_req("Hello");
        r2.messages[0].role = "user".into();
        let k1 = build_cache_key(&r1, "model.gguf", "llama", Some(16)).unwrap();
        let k2 = build_cache_key(&r2, "model.gguf", "llama", Some(16)).unwrap();
        assert_eq!(k1, k2, "role case must not affect cache key");
    }

    #[test]
    fn cache_key_normalizes_float_precision_noise() {
        // f32 can represent 0.7 as ~0.6999999762; both must hash identically.
        let mut r1 = mk_req("Hi");
        r1.temperature = 0.7_f32;
        let mut r2 = mk_req("Hi");
        // Next representable f32 above 0.7 — rounds to "0.7000" at 4dp.
        r2.temperature = 0.700_001_f32;
        let k1 = build_cache_key(&r1, "model.gguf", "llama", Some(16)).unwrap();
        let k2 = build_cache_key(&r2, "model.gguf", "llama", Some(16)).unwrap();
        assert_eq!(
            k1, k2,
            "sub-4-decimal float noise must not affect cache key"
        );
    }

    #[test]
    fn cache_key_same_for_no_max_tokens_and_default() {
        // Verifies that req.max_tokens is NOT used in the cache key — only
        // effective_max_tokens (the server-resolved value) matters.  A request
        // with max_tokens=Some(16) and one with max_tokens=None both produce
        // the same cache entry when effective_max_tokens is the same.
        let req_explicit = mk_req("prompt");
        let req_none = ChatCompletionRequest {
            max_tokens: None,
            ..mk_req("prompt")
        };
        let k_explicit = build_cache_key(&req_explicit, "m.gguf", "llama", Some(16)).unwrap();
        let k_none = build_cache_key(&req_none, "m.gguf", "llama", Some(16)).unwrap();
        assert_eq!(
            k_explicit, k_none,
            "explicit max_tokens matching default must share cache key"
        );
    }

    #[test]
    fn sampling_params_valid_boundaries_accepted() {
        assert!(
            validate_sampling_params(
                0.0,
                1.0,
                None,
                Some(1),
                0.1,
                None,
                None,
                Some(true),
                Some(20),
                Some(2)
            )
            .is_none()
        );
        assert!(
            validate_sampling_params(
                2.0,
                0.01,
                None,
                None,
                10.0,
                Some(-2.0),
                Some(2.0),
                None,
                None,
                Some(0),
            )
            .is_none()
        );
    }

    #[test]
    fn sampling_params_temperature_out_of_range() {
        assert!(
            validate_sampling_params(-0.1, 1.0, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
        assert!(
            validate_sampling_params(2.01, 1.0, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
    }

    #[test]
    fn sampling_params_top_p_out_of_range() {
        assert!(
            validate_sampling_params(1.0, 0.0, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
        assert!(
            validate_sampling_params(1.0, 1.01, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
    }

    #[test]
    fn sampling_params_top_k_zero_rejected() {
        assert!(
            validate_sampling_params(1.0, 1.0, None, Some(0), 1.1, None, None, None, None, None)
                .is_some()
        );
        assert!(
            validate_sampling_params(1.0, 1.0, None, Some(1), 1.1, None, None, None, None, None)
                .is_none()
        );
    }

    #[test]
    fn sampling_params_penalties_out_of_range() {
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                Some(2.01),
                None,
                None,
                None,
                None,
            )
            .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                Some(-2.01),
                None,
                None,
                None,
            )
            .is_some()
        );
    }

    #[test]
    fn sampling_params_top_logprobs_over_limit() {
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(true),
                Some(21),
                None,
            )
            .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(true),
                Some(20),
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn sampling_params_top_logprobs_requires_logprobs() {
        assert!(
            validate_sampling_params(1.0, 1.0, None, None, 1.1, None, None, None, Some(1), None)
                .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(false),
                Some(1),
                None,
            )
            .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(true),
                Some(1),
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn sampling_params_mirostat_invalid() {
        assert!(
            validate_sampling_params(1.0, 1.0, None, None, 1.1, None, None, None, None, Some(3))
                .is_some()
        );
        assert!(
            validate_sampling_params(1.0, 1.0, None, None, 1.1, None, None, None, None, Some(2))
                .is_none()
        );
    }

    #[test]
    fn response_format_validation_rejects_unknown_values() {
        let invalid = ResponseFormat {
            format_type: "xml".into(),
        };
        assert!(validate_response_format(Some(&invalid)).is_some());
        let text = ResponseFormat {
            format_type: "text".into(),
        };
        assert!(validate_response_format(Some(&text)).is_none());
        let json = ResponseFormat {
            format_type: "json_object".into(),
        };
        assert!(validate_response_format(Some(&json)).is_none());
    }

    #[test]
    fn max_tokens_validation_rejects_zero_and_schema_limit() {
        assert!(validate_max_tokens(Some(0)).is_some());
        assert!(validate_max_tokens(Some(MAX_MAX_TOKENS)).is_none());
        assert!(validate_max_tokens(Some(MAX_MAX_TOKENS + 1)).is_some());
    }

    #[test]
    fn pooling_type_validation_accepts_allowed_values() {
        for v in ["none", "mean", "cls", "last", "rank", "MEAN"] {
            assert!(is_valid_pooling_type(v), "expected valid pooling_type: {v}");
        }
    }

    #[test]
    fn pooling_type_validation_rejects_unknown_values() {
        for v in ["", "avg", "median", "foo"] {
            assert!(
                !is_valid_pooling_type(v),
                "expected invalid pooling_type: {v}"
            );
        }
    }

    #[test]
    fn scheduler_error_status_queue_full_is_429() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::QueueFull {
            waiting: 64,
            max: 64,
        }
        .into();
        assert_eq!(scheduler_error_status(&e), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn scheduler_error_status_timeout_is_503() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::Timeout { wait_ms: 250 }.into();
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn scheduler_error_status_throttled_is_503() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::Throttled { cap: 8 }.into();
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn scheduler_error_status_shutting_down_is_503() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::ShuttingDown.into();
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn scheduler_error_status_generic_error_is_503() {
        let e: anyhow::Error = anyhow::anyhow!("some unexpected error");
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }
}
