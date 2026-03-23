use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use ax_serving_engine::{
    GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, ModelHandle,
};
use tokio::sync::mpsc;

#[derive(Debug, Default)]
pub struct GenerationBatchMetrics {
    pub executed_batches: AtomicU64,
    pub executed_requests: AtomicU64,
    pub failed_batches: AtomicU64,
    pub largest_batch_requests: AtomicUsize,
}

#[derive(Clone)]
pub struct GenerationBatcher {
    backend: Arc<dyn InferenceBackend>,
    max_batch_size: usize,
    batch_window: Duration,
    metrics: Arc<GenerationBatchMetrics>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GenerationRequestClass {
    ChatCompletions,
    Completions,
}

impl GenerationRequestClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat_completions",
            Self::Completions => "completions",
        }
    }
}

pub struct GenerationBatchRequest {
    pub handle: ModelHandle,
    pub request_class: GenerationRequestClass,
    pub input: GenerateInput,
    pub params: GenerationParams,
}

impl GenerationBatcher {
    pub fn new(
        backend: Arc<dyn InferenceBackend>,
        max_batch_size: usize,
        batch_window_ms: u64,
    ) -> Self {
        Self {
            backend,
            max_batch_size: max_batch_size.max(1),
            batch_window: Duration::from_millis(batch_window_ms),
            metrics: Arc::new(GenerationBatchMetrics::default()),
        }
    }

    pub fn enabled(&self) -> bool {
        let _configured_for_batching = self.max_batch_size > 1 && !self.batch_window.is_zero();
        false
    }

    pub fn metrics(&self) -> &Arc<GenerationBatchMetrics> {
        &self.metrics
    }

    pub async fn submit(
        &self,
        request: GenerationBatchRequest,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        let (tx, rx) = mpsc::channel::<GenerateEvent>(512);
        self.metrics
            .executed_requests
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .largest_batch_requests
            .fetch_max(1, Ordering::Relaxed);

        if let Err(error) = self.execute_direct(request, tx) {
            self.metrics.failed_batches.fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }

        Ok(rx)
    }

    fn execute_direct(
        &self,
        request: GenerationBatchRequest,
        tx: mpsc::Sender<GenerateEvent>,
    ) -> Result<()> {
        self.backend
            .generate(request.handle, request.input, request.params, tx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use ax_serving_engine::{
        EmbedConfig, EmbedInput, EmbedResult, LoadConfig, ModelMetadata, ThermalState,
    };

    // ── Test double ────────────────────────────────────────────────────────────

    struct SpyBackend {
        generate_calls: Arc<AtomicUsize>,
        fail_generate: AtomicBool,
    }

    impl SpyBackend {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                generate_calls: Arc::new(AtomicUsize::new(0)),
                fail_generate: AtomicBool::new(false),
            })
        }
    }

    impl InferenceBackend for SpyBackend {
        fn load_model(
            &self,
            _: &Path,
            _: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            unimplemented!()
        }

        fn unload_model(&self, _: ModelHandle) -> anyhow::Result<()> {
            unimplemented!()
        }

        fn generate(
            &self,
            _: ModelHandle,
            _: GenerateInput,
            _: GenerationParams,
            tx: tokio::sync::mpsc::Sender<GenerateEvent>,
        ) -> anyhow::Result<()> {
            if self.fail_generate.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("deliberate generate failure"));
            }
            self.generate_calls.fetch_add(1, Ordering::Relaxed);
            let _ = tx.try_send(GenerateEvent::Done(Default::default()));
            Ok(())
        }

        fn tokenize(&self, _: ModelHandle, _: &str, _: bool) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn decode_tokens(&self, _: ModelHandle, _: &[u32]) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn eos_tokens(&self, _: ModelHandle) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn thermal_state(&self) -> ThermalState {
            ThermalState::Nominal
        }

        fn recommended_concurrency(&self) -> usize {
            4
        }

        fn embed(
            &self,
            _: ModelHandle,
            _: &EmbedInput<'_>,
            _: &EmbedConfig,
        ) -> anyhow::Result<EmbedResult> {
            Err(anyhow::anyhow!("embed not supported"))
        }
    }

    fn make_request() -> GenerationBatchRequest {
        GenerationBatchRequest {
            handle: ModelHandle(1),
            request_class: GenerationRequestClass::ChatCompletions,
            input: GenerateInput::Text("hello".to_string()),
            params: GenerationParams::default(),
        }
    }

    #[test]
    fn generation_batching_is_disabled_until_backend_support_exists() {
        let batcher = GenerationBatcher::new(SpyBackend::new(), 8, 10);
        assert!(!batcher.enabled());
    }

    #[tokio::test]
    async fn submit_routes_directly_to_backend_generate() {
        let spy = SpyBackend::new();
        let generate_calls = Arc::clone(&spy.generate_calls);
        let batcher = GenerationBatcher::new(spy, 8, 10);

        let mut rx = batcher.submit(make_request()).await.unwrap();

        assert_eq!(
            generate_calls.load(Ordering::Relaxed),
            1,
            "generation batcher should fall back to direct generate()"
        );
        assert!(matches!(rx.recv().await, Some(GenerateEvent::Done(_))));
        assert_eq!(
            batcher.metrics().executed_requests.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            batcher
                .metrics()
                .largest_batch_requests
                .load(Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn direct_failure_increments_failed_metrics() {
        let spy = SpyBackend::new();
        spy.fail_generate.store(true, Ordering::Relaxed);
        let batcher = GenerationBatcher::new(spy, 8, 10);

        let error = batcher.submit(make_request()).await.unwrap_err();

        assert!(error.to_string().contains("deliberate generate failure"));
        assert_eq!(batcher.metrics().failed_batches.load(Ordering::Relaxed), 1);
    }
}
