use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use ax_serving_engine::{EmbedConfig, EmbedInput, InferenceBackend, ModelHandle};
use tokio::sync::{Mutex, oneshot};
use tracing::warn;

use crate::scheduler::{PerModelScheduler, Scheduler};

#[derive(Debug, Default)]
pub struct EmbeddingBatchMetrics {
    pub executed_batches: AtomicU64,
    pub executed_requests: AtomicU64,
    pub executed_inputs: AtomicU64,
    pub failed_batches: AtomicU64,
    pub largest_batch_inputs: AtomicUsize,
}

#[derive(Clone)]
pub struct EmbeddingBatcher {
    inner: Arc<Mutex<HashMap<BatchKey, PendingBatchQueue>>>,
    backend: Arc<dyn InferenceBackend>,
    scheduler: Arc<Scheduler>,
    per_model_scheduler: Arc<PerModelScheduler>,
    max_batch_size: usize,
    batch_window: Duration,
    metrics: Arc<EmbeddingBatchMetrics>,
}

#[derive(Clone, Debug)]
pub struct EmbeddingBatchResult {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingBatchFailureKind {
    NotImplemented,
    ServiceUnavailable,
    Internal,
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct EmbeddingBatchFailure {
    pub kind: EmbeddingBatchFailureKind,
    pub message: String,
}

impl EmbeddingBatchFailure {
    fn from_error(error: anyhow::Error) -> Self {
        let message = error.to_string();
        let kind = if message.contains("not supported") {
            EmbeddingBatchFailureKind::NotImplemented
        } else if error
            .downcast_ref::<crate::scheduler::SchedulerError>()
            .is_some()
            || message.contains("per-model slot timeout")
            || message.contains("per-model semaphore closed")
        {
            EmbeddingBatchFailureKind::ServiceUnavailable
        } else {
            EmbeddingBatchFailureKind::Internal
        };
        Self { kind, message }
    }
}

#[derive(Clone, Debug)]
pub enum EmbeddingBatchRequestInput {
    Strings(Vec<String>),
    Tokens(Vec<Vec<u32>>),
}

#[derive(Clone, Debug)]
pub struct EmbeddingBatchRequest {
    pub handle: ModelHandle,
    pub model: String,
    pub config: EmbedConfig,
    pub input: EmbeddingBatchRequestInput,
}

impl EmbeddingBatchRequest {
    fn input_kind(&self) -> InputKind {
        match self.input {
            EmbeddingBatchRequestInput::Strings(_) => InputKind::Strings,
            EmbeddingBatchRequestInput::Tokens(_) => InputKind::Tokens,
        }
    }

    fn input_len(&self) -> usize {
        match &self.input {
            EmbeddingBatchRequestInput::Strings(texts) => texts.len(),
            EmbeddingBatchRequestInput::Tokens(seqs) => seqs.len(),
        }
    }

    fn usage_weight(&self) -> u32 {
        match &self.input {
            EmbeddingBatchRequestInput::Strings(texts) => texts
                .iter()
                .map(|text| text.chars().count() as u32)
                .sum::<u32>()
                .max(1),
            EmbeddingBatchRequestInput::Tokens(seqs) => {
                seqs.iter().map(|seq| seq.len() as u32).sum::<u32>().max(1)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum InputKind {
    Strings,
    Tokens,
}

#[derive(Clone, Debug, Eq)]
struct BatchKey {
    handle: ModelHandle,
    model: String,
    normalize: bool,
    truncate: bool,
    input_kind: InputKind,
}

impl PartialEq for BatchKey {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
            && self.model == other.model
            && self.normalize == other.normalize
            && self.truncate == other.truncate
            && self.input_kind == other.input_kind
    }
}

impl Hash for BatchKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.model.hash(state);
        self.normalize.hash(state);
        self.truncate.hash(state);
        self.input_kind.hash(state);
    }
}

impl BatchKey {
    fn from_request(request: &EmbeddingBatchRequest) -> Self {
        Self {
            handle: request.handle,
            model: request.model.clone(),
            normalize: request.config.normalize,
            truncate: request.config.truncate,
            input_kind: request.input_kind(),
        }
    }
}

struct PendingBatchQueue {
    next_id: u64,
    batches: VecDeque<PendingBatch>,
}

impl Default for PendingBatchQueue {
    fn default() -> Self {
        Self {
            next_id: 1,
            batches: VecDeque::new(),
        }
    }
}

struct PendingBatch {
    id: u64,
    total_inputs: usize,
    items: Vec<PendingEmbeddingRequest>,
}

struct PendingEmbeddingRequest {
    request: EmbeddingBatchRequest,
    tx: oneshot::Sender<std::result::Result<EmbeddingBatchResult, EmbeddingBatchFailure>>,
}

impl EmbeddingBatcher {
    pub fn new(
        backend: Arc<dyn InferenceBackend>,
        scheduler: Arc<Scheduler>,
        per_model_scheduler: Arc<PerModelScheduler>,
        max_batch_size: usize,
        batch_window_ms: u64,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            backend,
            scheduler,
            per_model_scheduler,
            max_batch_size: max_batch_size.max(1),
            batch_window: Duration::from_millis(batch_window_ms),
            metrics: Arc::new(EmbeddingBatchMetrics::default()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.max_batch_size > 1 && !self.batch_window.is_zero()
    }

    pub fn metrics(&self) -> &Arc<EmbeddingBatchMetrics> {
        &self.metrics
    }

    pub async fn submit(
        &self,
        request: EmbeddingBatchRequest,
    ) -> std::result::Result<EmbeddingBatchResult, EmbeddingBatchFailure> {
        if !self.enabled() || request.input_len() >= self.max_batch_size {
            return self
                .execute_direct(request)
                .await
                .map_err(EmbeddingBatchFailure::from_error);
        }

        let key = BatchKey::from_request(&request);
        let (tx, rx) = oneshot::channel();
        let mut flush_now = None;
        let mut spawn_timer_for = None;

        {
            let mut inner = self.inner.lock().await;
            let queue = inner.entry(key.clone()).or_default();

            let fits_existing = queue
                .batches
                .back()
                .map(|batch| batch.total_inputs + request.input_len() <= self.max_batch_size)
                .unwrap_or(false);

            if !fits_existing {
                let batch_id = queue.next_id;
                queue.next_id += 1;
                queue.batches.push_back(PendingBatch {
                    id: batch_id,
                    total_inputs: 0,
                    items: Vec::new(),
                });
                spawn_timer_for = Some(batch_id);
            }

            let Some(batch) = queue.batches.back_mut() else {
                return Err(EmbeddingBatchFailure {
                    kind: EmbeddingBatchFailureKind::Internal,
                    message: "embedding batch queue is empty".into(),
                });
            };
            batch.total_inputs += request.input_len();
            batch.items.push(PendingEmbeddingRequest { request, tx });

            if batch.total_inputs >= self.max_batch_size {
                flush_now = Some(batch.id);
            }
        }

        if let Some(batch_id) = spawn_timer_for {
            // Skip the deferred timer if the batch already reached max_batch_size
            // and will be flushed immediately — avoids spawning a redundant task
            // that sleeps for batch_window then discovers the batch is already gone.
            if flush_now.is_none() {
                let this = self.clone();
                let key_for_task = key.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(this.batch_window).await;
                    this.flush_batch(key_for_task, batch_id).await;
                });
            }
        }

        if let Some(batch_id) = flush_now {
            let this = self.clone();
            tokio::spawn(async move {
                this.flush_batch(key, batch_id).await;
            });
        }

        rx.await.map_err(|_| EmbeddingBatchFailure {
            kind: EmbeddingBatchFailureKind::Internal,
            message: "embedding batch coordinator dropped".into(),
        })?
    }

    async fn execute_direct(&self, request: EmbeddingBatchRequest) -> Result<EmbeddingBatchResult> {
        let model = request.model.clone();
        let handle = request.handle;
        let config = request.config.clone();

        let pm_permit = self
            .per_model_scheduler
            .acquire(&model, self.scheduler.config().max_wait_ms)
            .await?;
        let permit = self.scheduler.acquire().await?;

        let backend = Arc::clone(&self.backend);
        let input = request.input;
        let join_result = tokio::task::spawn_blocking(move || match input {
            EmbeddingBatchRequestInput::Strings(texts) => {
                backend.embed(handle, &EmbedInput::Strings(&texts), &config)
            }
            EmbeddingBatchRequestInput::Tokens(seqs) => {
                backend.embed(handle, &EmbedInput::Tokens(&seqs), &config)
            }
        })
        .await;

        drop(permit);
        drop(pm_permit);

        let result = join_result.context("direct embedding task panicked")?;
        let result = result?;
        Ok(EmbeddingBatchResult {
            embeddings: result.embeddings,
            prompt_tokens: result.prompt_tokens,
        })
    }

    async fn flush_batch(&self, key: BatchKey, batch_id: u64) {
        let batch = {
            let mut inner = self.inner.lock().await;
            let Some(queue) = inner.get_mut(&key) else {
                return;
            };
            let Some(index) = queue.batches.iter().position(|batch| batch.id == batch_id) else {
                return;
            };
            let batch = queue.batches.remove(index).unwrap_or_else(|| {
                warn!("batch was removed before flush; dropping coordinator request");
                PendingBatch {
                    id: batch_id,
                    total_inputs: 0,
                    items: Vec::new(),
                }
            });
            if queue.batches.is_empty() {
                inner.remove(&key);
            }
            batch
        };

        let PendingBatch {
            id: _,
            total_inputs: _,
            items,
        } = batch;
        let result = self.execute_batch_backend(&items).await;
        match result {
            Ok(output) => {
                for ((item, embeddings), prompt_tokens) in items
                    .into_iter()
                    .zip(output.per_request_embeddings.into_iter())
                    .zip(output.per_request_prompt_tokens.into_iter())
                {
                    let _ = item.tx.send(Ok(EmbeddingBatchResult {
                        embeddings,
                        prompt_tokens,
                    }));
                }
            }
            Err(error) => {
                self.metrics.failed_batches.fetch_add(1, Ordering::Relaxed);
                let failure = EmbeddingBatchFailure::from_error(error);
                for item in items {
                    let _ = item.tx.send(Err(failure.clone()));
                }
            }
        }
    }

    async fn execute_batch_backend(
        &self,
        items: &[PendingEmbeddingRequest],
    ) -> Result<BatchExecutionOutput> {
        let first = items
            .first()
            .context("embedding batch execution requires at least one request")?;
        let model = first.request.model.clone();
        let handle = first.request.handle;
        let config = first.request.config.clone();

        let pm_permit = self
            .per_model_scheduler
            .acquire(&model, self.scheduler.config().max_wait_ms)
            .await?;
        let permit = self.scheduler.acquire().await?;

        let total_requests = items.len();
        let total_inputs = items
            .iter()
            .map(|item| item.request.input_len())
            .sum::<usize>();
        self.metrics
            .executed_batches
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .executed_requests
            .fetch_add(total_requests as u64, Ordering::Relaxed);
        self.metrics
            .executed_inputs
            .fetch_add(total_inputs as u64, Ordering::Relaxed);
        self.metrics
            .largest_batch_inputs
            .fetch_max(total_inputs, Ordering::Relaxed);

        let input_kind = first.request.input_kind();
        let backend = Arc::clone(&self.backend);
        let join_result = match input_kind {
            InputKind::Strings => {
                let inputs = items
                    .iter()
                    .flat_map(|item| match &item.request.input {
                        EmbeddingBatchRequestInput::Strings(texts) => texts.clone(),
                        EmbeddingBatchRequestInput::Tokens(_) => Vec::new(),
                    })
                    .collect::<Vec<_>>();
                tokio::task::spawn_blocking(move || {
                    backend.embed(handle, &EmbedInput::Strings(&inputs), &config)
                })
                .await
            }
            InputKind::Tokens => {
                let inputs = items
                    .iter()
                    .flat_map(|item| match &item.request.input {
                        EmbeddingBatchRequestInput::Tokens(tokens) => tokens.clone(),
                        EmbeddingBatchRequestInput::Strings(_) => Vec::new(),
                    })
                    .collect::<Vec<_>>();
                tokio::task::spawn_blocking(move || {
                    backend.embed(handle, &EmbedInput::Tokens(&inputs), &config)
                })
                .await
            }
        };

        drop(permit);
        drop(pm_permit);

        let result = match input_kind {
            InputKind::Strings => join_result.context("embedding string batch task panicked")?,
            InputKind::Tokens => join_result.context("embedding token batch task panicked")?,
        };
        let result = result?;
        if result.embeddings.len() != total_inputs {
            return Err(anyhow!(
                "embedding batch result count mismatch: expected {total_inputs}, got {}",
                result.embeddings.len()
            ));
        }

        let usage_weights = items
            .iter()
            .map(|item| item.request.usage_weight())
            .collect::<Vec<_>>();
        let prompt_tokens = split_prompt_tokens(result.prompt_tokens, &usage_weights);
        let mut embeddings = result.embeddings.into_iter();
        let mut per_request_embeddings = Vec::with_capacity(total_requests);

        for item in items {
            let input_len = item.request.input_len();
            let mut request_embeddings = Vec::with_capacity(input_len);
            for _ in 0..input_len {
                request_embeddings.push(
                    embeddings
                        .next()
                        .context("embedding batch split ran out of vectors early")?,
                );
            }
            per_request_embeddings.push(request_embeddings);
        }

        Ok(BatchExecutionOutput {
            per_request_embeddings,
            per_request_prompt_tokens: prompt_tokens,
        })
    }
}

struct BatchExecutionOutput {
    per_request_embeddings: Vec<Vec<Vec<f32>>>,
    per_request_prompt_tokens: Vec<u32>,
}

fn split_prompt_tokens(total: u32, weights: &[u32]) -> Vec<u32> {
    if weights.is_empty() {
        return Vec::new();
    }
    if weights.len() == 1 {
        return vec![total];
    }

    let weight_sum = weights.iter().copied().sum::<u32>().max(1) as f64;
    let mut allocated = Vec::with_capacity(weights.len());
    let mut used = 0u32;
    let mut remainders = Vec::with_capacity(weights.len());

    for (index, weight) in weights.iter().copied().enumerate() {
        let exact = (total as f64) * (weight as f64 / weight_sum);
        let floor = exact.floor() as u32;
        allocated.push(floor);
        used = used.saturating_add(floor);
        remainders.push((index, exact - floor as f64));
    }

    remainders.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut remaining = total.saturating_sub(used);
    for (index, _) in remainders {
        if remaining == 0 {
            break;
        }
        allocated[index] = allocated[index].saturating_add(1);
        remaining -= 1;
    }
    allocated
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ax_serving_engine::{
        EmbedConfig, EmbedInput, EmbedResult, GenerateEvent, GenerateInput, GenerationParams,
        InferenceBackend, LoadConfig, ModelHandle, ModelMetadata, ThermalMonitor, ThermalState,
    };

    use super::{
        BatchKey, EmbeddingBatchRequest, EmbeddingBatchRequestInput, EmbeddingBatcher, InputKind,
        split_prompt_tokens,
    };
    use crate::scheduler::{PerModelScheduler, Scheduler, SchedulerConfig};

    // ── Test double ─────────────────────────────────────────────────────────────

    struct EchoEmbedBackend {
        embed_calls: Arc<AtomicUsize>,
        /// Dimension of each returned embedding vector.
        dim: usize,
    }

    impl EchoEmbedBackend {
        fn new(dim: usize) -> Arc<Self> {
            Arc::new(Self {
                embed_calls: Arc::new(AtomicUsize::new(0)),
                dim,
            })
        }
    }

    struct PanicThenOkEmbedBackend {
        calls: Arc<AtomicUsize>,
    }

    impl PanicThenOkEmbedBackend {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: Arc::new(AtomicUsize::new(0)),
            })
        }
    }

    impl InferenceBackend for PanicThenOkEmbedBackend {
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
            _: tokio::sync::mpsc::Sender<GenerateEvent>,
        ) -> anyhow::Result<()> {
            unimplemented!()
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
            1
        }

        fn embed(
            &self,
            _: ModelHandle,
            inputs: &EmbedInput<'_>,
            _: &EmbedConfig,
        ) -> anyhow::Result<EmbedResult> {
            let call_index = self.calls.fetch_add(1, Ordering::Relaxed);
            if call_index == 0 {
                panic!("deliberate embed panic");
            }

            let count = match inputs {
                EmbedInput::Strings(texts) => texts.len(),
                EmbedInput::Tokens(seqs) => seqs.len(),
            };
            Ok(EmbedResult {
                embeddings: vec![vec![1.0f32, 2.0f32]; count],
                prompt_tokens: count as u32,
            })
        }
    }

    impl InferenceBackend for EchoEmbedBackend {
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
            _: tokio::sync::mpsc::Sender<GenerateEvent>,
        ) -> anyhow::Result<()> {
            unimplemented!()
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
            _handle: ModelHandle,
            inputs: &EmbedInput<'_>,
            _config: &EmbedConfig,
        ) -> anyhow::Result<EmbedResult> {
            self.embed_calls.fetch_add(1, Ordering::Relaxed);
            let count = match inputs {
                EmbedInput::Strings(texts) => texts.len(),
                EmbedInput::Tokens(seqs) => seqs.len(),
            };
            let embeddings = (0..count).map(|_| vec![0.0f32; self.dim]).collect();
            Ok(EmbedResult {
                embeddings,
                prompt_tokens: count as u32 * 4,
            })
        }
    }

    fn make_scheduler() -> Arc<Scheduler> {
        Arc::new(Scheduler::new(
            SchedulerConfig {
                max_inflight: 8,
                max_queue: 64,
                max_wait_ms: 1_000,
                overload_policy: crate::scheduler::OverloadPolicy::Queue,
            },
            Arc::new(ThermalMonitor::new()),
        ))
    }

    fn make_scheduler_with_limits(max_inflight: usize, max_wait_ms: u64) -> Arc<Scheduler> {
        Arc::new(Scheduler::new(
            SchedulerConfig {
                max_inflight,
                max_queue: 8,
                max_wait_ms,
                overload_policy: crate::scheduler::OverloadPolicy::Queue,
            },
            Arc::new(ThermalMonitor::new()),
        ))
    }

    fn make_request(texts: Vec<&str>) -> EmbeddingBatchRequest {
        EmbeddingBatchRequest {
            handle: ModelHandle(1),
            model: "embed-model".into(),
            config: EmbedConfig::default(),
            input: EmbeddingBatchRequestInput::Strings(
                texts.into_iter().map(|s| s.to_string()).collect(),
            ),
        }
    }

    // ── enabled() ─────────────────────────────────────────────────────────────

    #[test]
    fn enabled_false_when_max_batch_size_is_one() {
        let backend = EchoEmbedBackend::new(3);
        let sched = make_scheduler();
        let pm = Arc::new(PerModelScheduler::new(4));
        let batcher = EmbeddingBatcher::new(backend, sched, pm, 1, 50);
        assert!(!batcher.enabled());
    }

    #[test]
    fn enabled_false_when_batch_window_is_zero() {
        let backend = EchoEmbedBackend::new(3);
        let sched = make_scheduler();
        let pm = Arc::new(PerModelScheduler::new(4));
        let batcher = EmbeddingBatcher::new(backend, sched, pm, 8, 0);
        assert!(!batcher.enabled());
    }

    #[test]
    fn enabled_true_when_conditions_met() {
        let backend = EchoEmbedBackend::new(3);
        let sched = make_scheduler();
        let pm = Arc::new(PerModelScheduler::new(4));
        let batcher = EmbeddingBatcher::new(backend, sched, pm, 8, 50);
        assert!(batcher.enabled());
    }

    // ── direct execution when disabled ────────────────────────────────────────

    #[tokio::test]
    async fn disabled_batcher_executes_directly() {
        let backend = EchoEmbedBackend::new(4);
        let embed_calls = Arc::clone(&backend.embed_calls);
        let sched = make_scheduler();
        let pm = Arc::new(PerModelScheduler::new(4));
        // max_batch_size=1 disables batching.
        let batcher = EmbeddingBatcher::new(backend, sched, pm, 1, 50);

        let result = batcher.submit(make_request(vec!["hello", "world"])).await;
        assert!(result.is_ok(), "direct embed should succeed: {result:?}");
        let r = result.unwrap();
        assert_eq!(r.embeddings.len(), 2);
        assert_eq!(r.embeddings[0].len(), 4);
        assert_eq!(embed_calls.load(Ordering::Relaxed), 1);
    }

    // ── large request bypasses batching ───────────────────────────────────────

    #[tokio::test]
    async fn large_request_bypasses_batching_queue() {
        let backend = EchoEmbedBackend::new(2);
        let embed_calls = Arc::clone(&backend.embed_calls);
        let sched = make_scheduler();
        let pm = Arc::new(PerModelScheduler::new(4));
        // max_batch_size=4, batch_window=10ms — enabled.
        let batcher = EmbeddingBatcher::new(backend, sched, pm, 4, 10);
        assert!(batcher.enabled());

        // A request with 5 inputs (≥ max_batch_size=4) should bypass the queue.
        let req = make_request(vec!["a", "b", "c", "d", "e"]);
        let result = batcher.submit(req).await;
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.embeddings.len(), 5);
        // Backend should have been called once directly.
        assert_eq!(embed_calls.load(Ordering::Relaxed), 1);
    }

    // ── metrics track executed requests ───────────────────────────────────────

    #[tokio::test]
    async fn metrics_track_executed_requests_on_direct_path() {
        let backend = EchoEmbedBackend::new(2);
        let sched = make_scheduler();
        let pm = Arc::new(PerModelScheduler::new(4));
        let batcher = EmbeddingBatcher::new(backend, sched, pm, 1, 0); // disabled

        batcher.submit(make_request(vec!["a", "b"])).await.unwrap();
        batcher.submit(make_request(vec!["c"])).await.unwrap();

        // Direct path does not bump executed_batches/executed_requests (those are
        // only bumped in execute_batch_backend for the batching path).
        let m = batcher.metrics();
        assert_eq!(m.executed_batches.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn panic_in_direct_embed_releases_scheduler_permits() {
        let backend = PanicThenOkEmbedBackend::new();
        let sched = make_scheduler_with_limits(1, 50);
        let pm = Arc::new(PerModelScheduler::new(1));
        let batcher = EmbeddingBatcher::new(backend, sched, pm, 1, 0); // disabled

        let first = batcher.submit(make_request(vec!["boom"])).await;
        assert!(
            first.is_err(),
            "first request should surface the task panic"
        );

        let second = batcher.submit(make_request(vec!["ok"])).await;
        assert!(
            second.is_ok(),
            "scheduler permits leaked after panic: {second:?}"
        );
        assert_eq!(second.unwrap().embeddings.len(), 1);
    }

    // ── BatchKey equality / hash ──────────────────────────────────────────────

    #[test]
    fn batch_key_same_fields_are_equal() {
        let key1 = BatchKey {
            handle: ModelHandle(1),
            model: "m".into(),
            normalize: true,
            truncate: false,
            input_kind: InputKind::Strings,
        };
        let key2 = key1.clone();
        assert_eq!(key1, key2);
    }

    #[test]
    fn batch_key_different_normalize_are_not_equal() {
        let key1 = BatchKey {
            handle: ModelHandle(1),
            model: "m".into(),
            normalize: true,
            truncate: false,
            input_kind: InputKind::Strings,
        };
        let key2 = BatchKey {
            normalize: false,
            ..key1.clone()
        };
        assert_ne!(key1, key2);
    }

    #[test]
    fn batch_key_different_input_kind_are_not_equal() {
        let key1 = BatchKey {
            handle: ModelHandle(1),
            model: "m".into(),
            normalize: false,
            truncate: false,
            input_kind: InputKind::Strings,
        };
        let key2 = BatchKey {
            input_kind: InputKind::Tokens,
            ..key1.clone()
        };
        assert_ne!(key1, key2);
    }

    // ── split_prompt_tokens ───────────────────────────────────────────────────

    #[test]
    fn split_prompt_tokens_preserves_total() {
        let parts = split_prompt_tokens(11, &[5, 3, 2]);
        assert_eq!(parts.iter().sum::<u32>(), 11);
    }

    #[test]
    fn split_prompt_tokens_single_request_keeps_exact_total() {
        assert_eq!(split_prompt_tokens(9, &[2]), vec![9]);
    }

    #[test]
    fn split_prompt_tokens_empty_weights_returns_empty() {
        assert!(split_prompt_tokens(100, &[]).is_empty());
    }

    #[test]
    fn split_prompt_tokens_zero_total_returns_all_zeros() {
        let parts = split_prompt_tokens(0, &[3, 5, 2]);
        assert_eq!(parts, vec![0, 0, 0]);
    }

    #[test]
    fn split_prompt_tokens_equal_weights_distributes_evenly() {
        let parts = split_prompt_tokens(9, &[1, 1, 1]);
        assert_eq!(parts.iter().sum::<u32>(), 9);
        // Each should get ~3.
        for p in &parts {
            assert!(*p >= 2 && *p <= 4, "uneven split: {parts:?}");
        }
    }
}
