//! gRPC service implementation for AxServingService.

use std::sync::Arc;
use std::time::Instant;

/// Capacity of the in-process mpsc channels used for streaming inference events.
///
/// Sized to buffer ~512 tokens ahead without backpressure on the generator
/// while keeping memory overhead negligible (~16 KB per active stream).
const INFER_CHANNEL_CAPACITY: usize = 512;

use ax_serving_engine::{
    BackendType, ChatMessage, GenerateEvent, GenerateInput, GenerationParams, LoadConfig,
    ThermalState,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::proto::{self, ax_serving_service_server::AxServingService as AxServingServiceTrait};
use crate::ServingLayer;
use crate::registry::RegistryError;
use crate::rest::schema::{MAX_CONTENT_BYTES, MAX_MAX_TOKENS, MAX_MESSAGES, MAX_MODEL_ID_BYTES};

/// Map a registry `anyhow::Error` to a gRPC [`Status`] using the typed
/// [`RegistryError`] variants where available, falling back to `internal`.
fn registry_error_to_status(e: &anyhow::Error) -> Status {
    let msg = e.to_string();
    match e.downcast_ref::<RegistryError>() {
        Some(RegistryError::AlreadyLoaded(_)) => Status::already_exists(msg),
        Some(RegistryError::NotLoaded(_)) => Status::not_found(msg),
        Some(
            RegistryError::FileNotFound(_)
            | RegistryError::InvalidFormat(_)
            | RegistryError::InvalidModelId(_),
        ) => Status::invalid_argument(msg),
        Some(RegistryError::PathNotAllowed(_)) => Status::permission_denied(msg),
        Some(RegistryError::CapacityExceeded(_)) => Status::resource_exhausted(msg),
        Some(RegistryError::Busy(_)) => Status::failed_precondition(msg),
        _ => Status::internal(msg),
    }
}

fn usize_to_u32_saturating(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

/// gRPC service backed by a [`ServingLayer`].
#[derive(Clone)]
pub struct AxServingService {
    layer: Arc<ServingLayer>,
}

impl AxServingService {
    pub fn new(layer: Arc<ServingLayer>) -> Self {
        Self { layer }
    }
}

#[tonic::async_trait]
impl AxServingServiceTrait for AxServingService {
    // ── LoadModel ─────────────────────────────────────────────────────────────

    async fn load_model(
        &self,
        request: Request<proto::LoadModelRequest>,
    ) -> Result<Response<proto::LoadModelResponse>, Status> {
        let req = request.into_inner();

        if req.model_id.is_empty() {
            return Err(Status::invalid_argument("model_id cannot be empty"));
        }
        if req.model_path.is_empty() {
            return Err(Status::invalid_argument("model_path cannot be empty"));
        }

        let path = std::path::Path::new(&req.model_path);
        let file_size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let backend_hint = infer_backend_hint_from_path(path);

        let config = LoadConfig {
            context_length: req.context_length,
            backend_type: proto_backend_to_engine(req.backend),
            llama_cpp_n_gpu_layers: None,
            mmproj_path: None,
            // Keep runtime serving behavior aligned with REST defaults: route
            // artifact directories via policy, but keep GGUF on explicit
            // compatibility loading so public native defaults do not select it.
            backend_hint: Some(backend_hint.to_string()),
            enable_embeddings: None,
            pooling_type: None,
        };

        let entry = self
            .layer
            .registry
            .load(&req.model_id, path, config, &*self.layer.backend)
            .map_err(|ref e| registry_error_to_status(e))?;

        let meta = &entry.metadata;
        let info = proto::ModelInfo {
            id: entry.id.clone(),
            architecture: meta.architecture.clone(),
            n_layers: meta.n_layers,
            n_heads: meta.n_heads,
            n_kv_heads: meta.n_kv_heads,
            embedding_dim: meta.embedding_dim,
            vocab_size: meta.vocab_size,
            context_length: meta.context_length,
            file_size_bytes,
            // Report the resolved backend, not the client-supplied hint.
            backend: engine_backend_type_to_proto(meta.resolved_backend) as i32,
        };

        Ok(Response::new(proto::LoadModelResponse {
            model_id: entry.id.clone(),
            info: Some(info),
            load_time_ms: meta.load_time_ms,
            peak_rss_bytes: meta.peak_rss_bytes,
        }))
    }

    // ── UnloadModel ───────────────────────────────────────────────────────────

    async fn unload_model(
        &self,
        request: Request<proto::UnloadModelRequest>,
    ) -> Result<Response<proto::UnloadModelResponse>, Status> {
        let req = request.into_inner();

        self.layer
            .registry
            .unload(&req.model_id, &*self.layer.backend)
            .map_err(|ref e| registry_error_to_status(e))?;

        Ok(Response::new(proto::UnloadModelResponse {
            model_id: req.model_id,
        }))
    }

    // ── ListModels ────────────────────────────────────────────────────────────

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> Result<Response<proto::ListModelsResponse>, Status> {
        // Use list_entries() (no touch) rather than list_ids() + get() to
        // avoid resetting last_accessed_ms on every ListModels call, which
        // would prevent idle eviction from ever firing.
        let models = self
            .layer
            .registry
            .list_entries()
            .into_iter()
            .map(|entry| {
                let file_size_bytes = std::fs::metadata(&entry.path).map(|m| m.len()).unwrap_or(0);
                proto::ModelInfo {
                    id: entry.id.clone(),
                    architecture: entry.metadata.architecture.clone(),
                    n_layers: entry.metadata.n_layers,
                    n_heads: entry.metadata.n_heads,
                    n_kv_heads: entry.metadata.n_kv_heads,
                    embedding_dim: entry.metadata.embedding_dim,
                    vocab_size: entry.metadata.vocab_size,
                    context_length: entry.metadata.context_length,
                    file_size_bytes,
                    // Report the resolved backend, not the hardcoded Auto default.
                    backend: engine_backend_type_to_proto(entry.metadata.resolved_backend) as i32,
                }
            })
            .collect();

        Ok(Response::new(proto::ListModelsResponse { models }))
    }

    // ── Infer (server-streaming) ───────────────────────────────────────────────

    type InferStream = ReceiverStream<Result<proto::InferResponse, Status>>;

    async fn infer(
        &self,
        request: Request<proto::InferRequest>,
    ) -> Result<Response<Self::InferStream>, Status> {
        let req = request.into_inner();

        // Validate model and input BEFORE acquiring the scheduler permit so
        // that invalid or missing-model requests never consume a concurrency
        // slot that could have served a valid request.
        if let Some(status) = validate_infer_request(&req) {
            return Err(status);
        }

        let entry = self
            .layer
            .registry
            .get(&req.model_id)
            .ok_or_else(|| Status::not_found(format!("model '{}' not loaded", req.model_id)))?;

        let handle = entry.handle;

        let input = if !req.messages.is_empty() {
            GenerateInput::Chat(
                req.messages
                    .iter()
                    .map(|m| ChatMessage {
                        role: m.role.clone(),
                        content: serde_json::Value::String(m.content.clone()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    })
                    .collect(),
            )
        } else if !req.prompt.is_empty() {
            GenerateInput::Text(req.prompt.clone())
        } else {
            return Err(Status::invalid_argument(
                "either prompt or messages must be non-empty",
            ));
        };

        // All validation passed — now acquire the same per-model and global
        // concurrency permits used by REST inference. Per-model first avoids
        // holding a global slot while waiting behind another request for the
        // same model.
        let pm_permit = self
            .layer
            .per_model_scheduler
            .acquire(&req.model_id, self.layer.scheduler.config().max_wait_ms)
            .await
            .map_err(|e| Status::resource_exhausted(e.to_string()))?;

        let permit = self
            .layer
            .scheduler
            .acquire()
            .await
            .map_err(|e| Status::resource_exhausted(e.to_string()))?;

        let params = infer_generation_params(&req, self.layer.default_max_tokens);

        let (engine_tx, mut engine_rx) = mpsc::channel::<GenerateEvent>(INFER_CHANNEL_CAPACITY);

        self.layer
            .backend
            .generate(handle, input, params, engine_tx)
            .map_err(|e| Status::internal(e.to_string()))?;

        let (out_tx, out_rx) =
            mpsc::channel::<Result<proto::InferResponse, Status>>(INFER_CHANNEL_CAPACITY);

        // Start timing before spawning so total_time_ms covers the full
        // generation window, not just task-scheduled-to-done.
        let start = Instant::now();
        let model_entry = entry;

        tokio::spawn(async move {
            // Hold the loaded model entry so unload/reload sees the handle as
            // busy until the stream has fully finished or the client disconnects.
            let _model_entry = model_entry;
            // Hold both scheduler permits for the full stream lifetime.
            let _pm_permit = pm_permit;
            let _permit = permit;

            while let Some(event) = engine_rx.recv().await {
                let msg = match event {
                    GenerateEvent::Token(text) => Ok(proto::InferResponse {
                        text,
                        token_id: 0,
                        finished: false,
                        finish_reason: proto::FinishReason::Unspecified as i32,
                        metrics: None,
                    }),
                    GenerateEvent::Done(stats) => {
                        let total_ms = start.elapsed().as_millis() as u64;
                        Ok(proto::InferResponse {
                            text: String::new(),
                            token_id: 0,
                            finished: true,
                            finish_reason: grpc_finish_reason(&stats.stop_reason) as i32,
                            metrics: Some(proto::GenerationMetrics {
                                prefill_tokens: usize_to_u32_saturating(stats.prompt_tokens),
                                decode_tokens: usize_to_u32_saturating(stats.completion_tokens),
                                prefill_tok_per_sec: stats.prefill_tok_per_sec as f32,
                                decode_tok_per_sec: stats.decode_tok_per_sec as f32,
                                total_time_ms: total_ms,
                            }),
                        })
                    }
                    GenerateEvent::Error(e) => Err(Status::internal(e)),
                    // Tool calls and logprobs are not in the gRPC proto schema — skip.
                    GenerateEvent::ToolCall { .. } | GenerateEvent::TokenLogprob { .. } => continue,
                };

                let is_err = msg.is_err();
                if out_tx.send(msg).await.is_err() || is_err {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(out_rx)))
    }

    // ── Health ────────────────────────────────────────────────────────────────

    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        let thermal = engine_thermal_to_proto(self.layer.backend.thermal_state());

        Ok(Response::new(proto::HealthResponse {
            status: proto::ServingStatus::Serving as i32,
            model_ids: self.layer.registry.list_ids(),
            uptime_secs: self.layer.metrics.uptime_secs(),
            thermal_state: thermal as i32,
        }))
    }

    // ── GetMetrics ────────────────────────────────────────────────────────────

    async fn get_metrics(
        &self,
        _request: Request<proto::GetMetricsRequest>,
    ) -> Result<Response<proto::GetMetricsResponse>, Status> {
        use std::sync::atomic::Ordering;

        let m = &self.layer.scheduler.metrics;
        let thermal = engine_thermal_to_proto(self.layer.backend.thermal_state());

        Ok(Response::new(proto::GetMetricsResponse {
            system: Some(proto::SystemMetrics {
                rss_bytes: crate::metrics::current_rss_bytes(),
                peak_rss_bytes: 0,
                thermal_state: thermal as i32,
                inflight_count: m.inflight_count.load(Ordering::Relaxed),
                total_requests: m.total_requests.load(Ordering::Relaxed),
                rejected_requests: m.rejected_requests.load(Ordering::Relaxed),
                avg_queue_wait_us: m.avg_queue_wait_us(),
            }),
            models: Vec::new(),
        }))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn validate_infer_request(req: &proto::InferRequest) -> Option<Status> {
    if let Some(status) = validate_grpc_model_id(&req.model_id) {
        return Some(status);
    }

    if req.max_tokens > MAX_MAX_TOKENS {
        return Some(Status::invalid_argument(format!(
            "max_tokens exceeds limit ({MAX_MAX_TOKENS})"
        )));
    }

    if let Some(sampling) = req.sampling.as_ref() {
        if !sampling.temperature.is_finite() || !(0.0..=2.0).contains(&sampling.temperature) {
            return Some(Status::invalid_argument("temperature must be in [0, 2]"));
        }
        if !sampling.top_p.is_finite() || !(0.0..=1.0).contains(&sampling.top_p) {
            return Some(Status::invalid_argument("top_p must be in [0, 1]"));
        }
        if !sampling.repeat_penalty.is_finite()
            || sampling.repeat_penalty < 0.0
            || sampling.repeat_penalty > 10.0
        {
            return Some(Status::invalid_argument(
                "repeat_penalty must be 0 or in (0, 10]",
            ));
        }
    }

    if !req.messages.is_empty() {
        if req.messages.len() > MAX_MESSAGES {
            return Some(Status::invalid_argument(format!(
                "too many messages (max {MAX_MESSAGES})"
            )));
        }
        for (index, message) in req.messages.iter().enumerate() {
            if message.content.len() > MAX_CONTENT_BYTES {
                return Some(Status::invalid_argument(format!(
                    "message content at index {index} exceeds {MAX_CONTENT_BYTES} bytes"
                )));
            }
        }
        return None;
    }

    if req.prompt.is_empty() {
        return Some(Status::invalid_argument(
            "either prompt or messages must be non-empty",
        ));
    }

    if req.prompt.len() > MAX_CONTENT_BYTES {
        return Some(Status::invalid_argument(format!(
            "prompt exceeds {MAX_CONTENT_BYTES} bytes"
        )));
    }

    None
}

fn infer_generation_params(req: &proto::InferRequest, default_max_tokens: u32) -> GenerationParams {
    let sampling = req.sampling.as_ref();
    GenerationParams {
        stream: true,
        temperature: sampling
            .map(|params| params.temperature as f64)
            .filter(|temperature| *temperature > 0.0),
        top_p: sampling.and_then(|params| {
            if params.top_p > 0.0 {
                Some(params.top_p as f64)
            } else {
                None
            }
        }),
        top_k: sampling.and_then(|params| {
            if params.top_k > 0 {
                Some(params.top_k as usize)
            } else {
                None
            }
        }),
        max_tokens: {
            // proto3 uses 0 as the default (field omitted by client). Apply
            // the same server-side cap as the REST path: fall back to
            // `default_max_tokens` when the client sends 0; a configured
            // default of 0 means "no cap" (pass None to the backend).
            if req.max_tokens > 0 {
                Some(req.max_tokens as usize)
            } else if default_max_tokens > 0 {
                Some(default_max_tokens as usize)
            } else {
                None
            }
        },
        stop_seqs: Vec::new(),
        seed: sampling.and_then(|params| {
            if params.seed > 0 {
                Some(params.seed)
            } else {
                None
            }
        }),
        repeat_penalty: sampling.and_then(|params| {
            if params.repeat_penalty > 0.0 {
                Some(params.repeat_penalty as f64)
            } else {
                None
            }
        }),
        ..Default::default()
    }
}

fn validate_grpc_model_id(model_id: &str) -> Option<Status> {
    let trimmed = model_id.trim();
    if trimmed.is_empty() {
        return Some(Status::invalid_argument("model_id must not be empty"));
    }
    if model_id != trimmed {
        return Some(Status::invalid_argument(
            "model_id contains unsupported whitespace",
        ));
    }
    if model_id.len() > MAX_MODEL_ID_BYTES {
        return Some(Status::invalid_argument(format!(
            "model_id exceeds max length of {MAX_MODEL_ID_BYTES}"
        )));
    }
    if !model_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Some(Status::invalid_argument(
            "model_id must be alphanumeric with '-', '_', or '.'",
        ));
    }

    None
}

fn proto_backend_to_engine(backend: i32) -> BackendType {
    match backend {
        x if x == proto::BackendType::Cpu as i32 => BackendType::Cpu,
        x if x == proto::BackendType::Metal as i32 => BackendType::Metal,
        _ => BackendType::Auto,
    }
}

fn infer_backend_hint_from_path(path: &std::path::Path) -> &'static str {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        "llama_cpp"
    } else {
        "auto"
    }
}

fn engine_backend_type_to_proto(backend: BackendType) -> proto::BackendType {
    match backend {
        BackendType::Metal => proto::BackendType::Metal,
        BackendType::Cpu => proto::BackendType::Cpu,
        BackendType::Auto => proto::BackendType::Auto,
    }
}

fn engine_thermal_to_proto(state: ThermalState) -> proto::ThermalState {
    match state {
        ThermalState::Nominal => proto::ThermalState::Nominal,
        ThermalState::Fair => proto::ThermalState::Fair,
        ThermalState::Serious => proto::ThermalState::Serious,
        ThermalState::Critical => proto::ThermalState::Critical,
    }
}

fn grpc_finish_reason(reason: &str) -> proto::FinishReason {
    match reason {
        "length" => proto::FinishReason::MaxTokens,
        // The engine's `"stop"` reason currently covers both natural EOS and
        // backend stop-sequence termination. Preserve the legacy EOS mapping
        // unless the backend can expose a distinct stop-sequence reason.
        "stop_sequence" => proto::FinishReason::StopSequence,
        "stop" | "" => proto::FinishReason::Eos,
        _ => proto::FinishReason::Unspecified,
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use ax_serving_engine::{
        GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig, ModelHandle,
        ModelMetadata,
    };
    use tonic::Request;

    use super::*;
    use crate::config::ServeConfig;
    use crate::registry::RegistryError;

    fn valid_infer_request() -> proto::InferRequest {
        proto::InferRequest {
            model_id: "test-model".to_string(),
            prompt: "hello".to_string(),
            messages: Vec::new(),
            sampling: None,
            max_tokens: 128,
        }
    }

    struct HoldingBackend {
        sender: Mutex<Option<mpsc::Sender<GenerateEvent>>>,
    }

    impl HoldingBackend {
        fn new() -> Self {
            Self {
                sender: Mutex::new(None),
            }
        }

        fn release_generation(&self) {
            let _ = self.sender.lock().unwrap().take();
        }
    }

    impl InferenceBackend for HoldingBackend {
        fn load_model(
            &self,
            _path: &Path,
            _config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            Ok((
                ModelHandle(1),
                ModelMetadata {
                    architecture: "holding".into(),
                    n_layers: 0,
                    n_heads: 0,
                    n_kv_heads: 0,
                    embedding_dim: 0,
                    vocab_size: 0,
                    context_length: 2048,
                    load_time_ms: 1,
                    peak_rss_bytes: 0,
                    resolved_backend: BackendType::Auto,
                },
            ))
        }

        fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _handle: ModelHandle,
            _input: GenerateInput,
            _params: GenerationParams,
            tx: mpsc::Sender<GenerateEvent>,
        ) -> anyhow::Result<()> {
            *self.sender.lock().unwrap() = Some(tx);
            Ok(())
        }

        fn tokenize(
            &self,
            _handle: ModelHandle,
            _text: &str,
            _add_bos: bool,
        ) -> anyhow::Result<Vec<u32>> {
            Ok(Vec::new())
        }

        fn decode_tokens(&self, _handle: ModelHandle, _tokens: &[u32]) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn eos_tokens(&self, _handle: ModelHandle) -> anyhow::Result<Vec<u32>> {
            Ok(vec![2])
        }

        fn thermal_state(&self) -> ThermalState {
            ThermalState::Nominal
        }

        fn recommended_concurrency(&self) -> usize {
            8
        }
    }

    #[test]
    fn registry_error_maps_to_grpc_status() {
        let cases: &[(&dyn Fn() -> anyhow::Error, tonic::Code)] = &[
            (
                &|| RegistryError::AlreadyLoaded("m".into()).into(),
                tonic::Code::AlreadyExists,
            ),
            (
                &|| RegistryError::NotLoaded("m".into()).into(),
                tonic::Code::NotFound,
            ),
            (
                &|| RegistryError::FileNotFound("p".into()).into(),
                tonic::Code::InvalidArgument,
            ),
            (
                &|| RegistryError::InvalidFormat("p".into()).into(),
                tonic::Code::InvalidArgument,
            ),
            (
                &|| RegistryError::InvalidModelId("bad".into()).into(),
                tonic::Code::InvalidArgument,
            ),
            (
                &|| RegistryError::PathNotAllowed("p".into()).into(),
                tonic::Code::PermissionDenied,
            ),
            (
                &|| RegistryError::CapacityExceeded(16).into(),
                tonic::Code::ResourceExhausted,
            ),
            (
                &|| RegistryError::Busy("m".into()).into(),
                tonic::Code::FailedPrecondition,
            ),
            (
                &|| anyhow::anyhow!("generic backend error"),
                tonic::Code::Internal,
            ),
        ];

        for (make_err, expected_code) in cases {
            let status = registry_error_to_status(&make_err());
            assert_eq!(
                status.code(),
                *expected_code,
                "wrong gRPC code for error: {}",
                make_err()
            );
        }
    }

    #[test]
    fn infer_validation_accepts_valid_prompt_request() {
        assert!(validate_infer_request(&valid_infer_request()).is_none());
    }

    #[test]
    fn grpc_usage_conversion_saturates_u32_fields() {
        assert_eq!(usize_to_u32_saturating(u32::MAX as usize + 1), u32::MAX);
    }

    #[test]
    fn grpc_finish_reason_maps_backend_stop_reasons() {
        assert_eq!(grpc_finish_reason(""), proto::FinishReason::Eos);
        assert_eq!(grpc_finish_reason("stop"), proto::FinishReason::Eos);
        assert_eq!(grpc_finish_reason("length"), proto::FinishReason::MaxTokens);
        assert_eq!(
            grpc_finish_reason("stop_sequence"),
            proto::FinishReason::StopSequence
        );
        assert_eq!(
            grpc_finish_reason("content_filter"),
            proto::FinishReason::Unspecified
        );
        assert_eq!(
            grpc_finish_reason("tool_calls"),
            proto::FinishReason::Unspecified
        );
    }

    #[test]
    fn infer_validation_rejects_empty_model_id() {
        let mut req = valid_infer_request();
        req.model_id.clear();

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("model_id"));
    }

    #[test]
    fn infer_validation_rejects_invalid_model_id() {
        let mut req = valid_infer_request();
        req.model_id = "bad model".to_string();

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("model_id"));
    }

    #[test]
    fn infer_validation_rejects_oversized_prompt() {
        let mut req = valid_infer_request();
        req.prompt = "a".repeat(MAX_CONTENT_BYTES + 1);

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("prompt exceeds"));
    }

    #[test]
    fn infer_validation_rejects_too_many_messages() {
        let mut req = valid_infer_request();
        req.prompt.clear();
        req.messages = (0..=MAX_MESSAGES)
            .map(|_| proto::ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            })
            .collect();

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("too many messages"));
    }

    #[test]
    fn infer_validation_rejects_oversized_message() {
        let mut req = valid_infer_request();
        req.prompt.clear();
        req.messages = vec![proto::ChatMessage {
            role: "user".to_string(),
            content: "a".repeat(MAX_CONTENT_BYTES + 1),
        }];

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("message content"));
    }

    #[test]
    fn infer_validation_rejects_max_tokens_over_limit() {
        let mut req = valid_infer_request();
        req.max_tokens = MAX_MAX_TOKENS + 1;

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("max_tokens exceeds"));
    }

    #[test]
    fn infer_validation_rejects_invalid_sampling_ranges() {
        let mut req = valid_infer_request();
        req.sampling = Some(proto::SamplingParams {
            temperature: 2.5,
            top_k: 0,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: 0,
        });

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("temperature"));

        req.sampling = Some(proto::SamplingParams {
            temperature: 0.7,
            top_k: 0,
            top_p: 1.5,
            repeat_penalty: 1.1,
            seed: 0,
        });

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("top_p"));

        req.sampling = Some(proto::SamplingParams {
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            repeat_penalty: 11.0,
            seed: 0,
        });

        let status = validate_infer_request(&req).unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("repeat_penalty"));
    }

    #[test]
    fn infer_generation_params_preserve_sampling_seed_and_repeat_penalty() {
        let mut req = valid_infer_request();
        req.max_tokens = 0;
        req.sampling = Some(proto::SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.15,
            seed: 12345,
        });

        let params = infer_generation_params(&req, 2048);

        assert!(params.stream);
        assert!(
            (params.temperature.unwrap() - 0.7).abs() < 0.000001,
            "temperature should be preserved"
        );
        assert_eq!(params.top_k, Some(40));
        assert!(
            (params.top_p.unwrap() - 0.95).abs() < 0.000001,
            "top_p should be preserved"
        );
        assert_eq!(params.max_tokens, Some(2048));
        assert_eq!(params.seed, Some(12345));
        assert!(
            (params.repeat_penalty.unwrap() - 1.15).abs() < 0.000001,
            "repeat_penalty should be preserved"
        );
    }

    #[test]
    fn infer_generation_params_treat_proto_zero_sampling_values_as_omitted() {
        let mut req = valid_infer_request();
        req.max_tokens = 0;
        req.sampling = Some(proto::SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            repeat_penalty: 0.0,
            seed: 0,
        });

        let params = infer_generation_params(&req, 0);

        assert_eq!(params.temperature, None);
        assert_eq!(params.top_k, None);
        assert_eq!(params.top_p, None);
        assert_eq!(params.max_tokens, None);
        assert_eq!(params.seed, None);
        assert_eq!(params.repeat_penalty, None);
    }

    #[tokio::test]
    async fn grpc_infer_enforces_per_model_limit() {
        let backend = Arc::new(HoldingBackend::new());
        let mut config = ServeConfig {
            sched_per_model_max_inflight: 1,
            sched_max_inflight: 8,
            sched_max_wait_ms: 20,
            ..ServeConfig::default()
        };
        config.cache.enabled = false;
        let layer = Arc::new(ServingLayer::new(
            Arc::clone(&backend) as Arc<dyn InferenceBackend>,
            config,
        ));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"dummy").unwrap();
        layer
            .registry
            .load(
                "test-model",
                &path,
                LoadConfig {
                    context_length: 0,
                    backend_type: BackendType::Auto,
                    llama_cpp_n_gpu_layers: None,
                    mmproj_path: None,
                    backend_hint: Some("auto".into()),
                    enable_embeddings: None,
                    pooling_type: None,
                },
                backend.as_ref(),
            )
            .unwrap();

        let service = AxServingService::new(Arc::clone(&layer));
        let first = service
            .infer(Request::new(valid_infer_request()))
            .await
            .expect("first infer should acquire the per-model slot");

        let unload_err = layer
            .registry
            .unload("test-model", backend.as_ref())
            .expect_err("active stream should keep the model handle busy");
        assert!(
            unload_err
                .downcast_ref::<RegistryError>()
                .is_some_and(|e| matches!(e, RegistryError::Busy(id) if id == "test-model")),
            "unexpected unload error: {unload_err}"
        );

        let second = service.infer(Request::new(valid_infer_request())).await;
        assert!(second.is_err(), "second same-model infer should be capped");
        let status = second.unwrap_err();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
        assert!(
            status.message().contains("per-model slot timeout"),
            "unexpected status message: {}",
            status.message()
        );

        drop(first);
        backend.release_generation();
    }

    #[test]
    fn proto_backend_to_engine_all_variants() {
        assert_eq!(
            proto_backend_to_engine(proto::BackendType::Cpu as i32),
            BackendType::Cpu
        );
        assert_eq!(
            proto_backend_to_engine(proto::BackendType::Metal as i32),
            BackendType::Metal
        );
        assert_eq!(
            proto_backend_to_engine(proto::BackendType::Auto as i32),
            BackendType::Auto
        );
        // Unknown value must fall back to Auto.
        assert_eq!(proto_backend_to_engine(999), BackendType::Auto);
    }

    #[test]
    fn grpc_load_infers_llama_cpp_for_gguf_paths() {
        assert_eq!(
            infer_backend_hint_from_path(Path::new("/models/model.gguf")),
            "llama_cpp"
        );
        assert_eq!(
            infer_backend_hint_from_path(Path::new("/models/model.GGUF")),
            "llama_cpp"
        );
        assert_eq!(
            infer_backend_hint_from_path(Path::new("/models/ax-artifacts")),
            "auto"
        );
    }

    #[test]
    fn engine_thermal_to_proto_all_variants() {
        assert!(matches!(
            engine_thermal_to_proto(ThermalState::Nominal),
            proto::ThermalState::Nominal
        ));
        assert!(matches!(
            engine_thermal_to_proto(ThermalState::Fair),
            proto::ThermalState::Fair
        ));
        assert!(matches!(
            engine_thermal_to_proto(ThermalState::Serious),
            proto::ThermalState::Serious
        ));
        assert!(matches!(
            engine_thermal_to_proto(ThermalState::Critical),
            proto::ThermalState::Critical
        ));
    }
}
