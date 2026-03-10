//! gRPC service implementation for AxServingService.

use std::sync::Arc;
use std::time::Instant;

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
        _ => Status::internal(msg),
    }
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

        let config = LoadConfig {
            context_length: req.context_length,
            backend_type: proto_backend_to_engine(req.backend),
            llama_cpp_n_gpu_layers: None,
            mmproj_path: None,
            backend_hint: None,
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
            backend: req.backend,
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
                    backend: proto::BackendType::Auto as i32,
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

        let permit = self
            .layer
            .scheduler
            .acquire()
            .await
            .map_err(|e| Status::resource_exhausted(e.to_string()))?;

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

        let s = req.sampling.as_ref();
        let params = GenerationParams {
            stream: true,
            temperature: s.map(|p| p.temperature as f64).filter(|&t| t > 0.0),
            top_p: s.and_then(|p| {
                if p.top_p > 0.0 {
                    Some(p.top_p as f64)
                } else {
                    None
                }
            }),
            top_k: s.and_then(|p| {
                if p.top_k > 0 {
                    Some(p.top_k as usize)
                } else {
                    None
                }
            }),
            max_tokens: {
                // proto3 uses 0 as the default (field omitted by client).
                // Apply the same server-side cap as the REST path: fall back to
                // `default_max_tokens` when the client sends 0; a configured
                // default of 0 means "no cap" (pass None to the backend).
                let d = self.layer.default_max_tokens;
                if req.max_tokens > 0 {
                    Some(req.max_tokens as usize)
                } else if d > 0 {
                    Some(d as usize)
                } else {
                    None
                }
            },
            stop_seqs: Vec::new(),
            seed: None,
            repeat_penalty: None,
            ..Default::default()
        };

        let (engine_tx, mut engine_rx) = mpsc::channel::<GenerateEvent>(512);

        self.layer
            .backend
            .generate(handle, input, params, engine_tx)
            .map_err(|e| Status::internal(e.to_string()))?;

        let (out_tx, out_rx) = mpsc::channel::<Result<proto::InferResponse, Status>>(512);

        // Start timing before spawning so total_time_ms covers the full
        // generation window, not just task-scheduled-to-done.
        let start = Instant::now();

        tokio::spawn(async move {
            // Hold the scheduler permit for the full stream lifetime.
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
                            finish_reason: proto::FinishReason::Eos as i32,
                            metrics: Some(proto::GenerationMetrics {
                                prefill_tokens: stats.prompt_tokens as u32,
                                decode_tokens: stats.completion_tokens as u32,
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

fn proto_backend_to_engine(backend: i32) -> BackendType {
    match backend {
        x if x == proto::BackendType::Cpu as i32 => BackendType::Cpu,
        x if x == proto::BackendType::Metal as i32 => BackendType::Metal,
        _ => BackendType::Auto,
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

#[cfg(test)]
mod tests {
    use crate::registry::RegistryError;

    use super::*;

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
