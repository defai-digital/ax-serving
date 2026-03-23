//! ax-serving-py: PyO3 Python bindings for ax-serving.
//!
//! Exposes a minimal Python API for loading models and running inference
//! against the native/llama.cpp backend chain via `InferenceBackend`.
//!
//! # Usage (Python)
//!
//! ```python
//! import ax_serving
//!
//! model = ax_serving.AxModel.load("/path/to/model.gguf")
//!
//! # Text completion
//! reply = model.generate("The capital of France is", max_tokens=32)
//!
//! # Chat
//! reply = model.chat([("user", "Hello!")], max_tokens=128, temperature=0.7)
//!
//! # Tokenize / decode
//! ids = model.tokenize("Hello world")
//! text = model.decode(ids)
//! ```

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-serving-py only supports aarch64-apple-darwin (Apple Silicon M3+)");

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use ax_serving_engine::{
    ChatMessage, GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig,
    ModelHandle, RouterBackend,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::mpsc;

// ── AxModel ───────────────────────────────────────────────────────────────────

/// A loaded inference model.
///
/// Created via `AxModel.load(path)`. Internally wraps the `RouterBackend`
/// (which auto-selects the native backend or llama.cpp based on model architecture).
#[pyclass]
pub struct AxModel {
    backend: Arc<dyn InferenceBackend>,
    handle: ModelHandle,
    /// Small current-thread runtime used only to drain the mpsc channel
    /// returned by `generate()`. Separate from the backend's own runtime to
    /// avoid any nested `block_on` calls.
    runtime: tokio::runtime::Runtime,
    /// Guard `Runtime::block_on` — current-thread runtimes must not be entered
    /// concurrently from multiple Python threads.
    runtime_lock: Mutex<()>,
}

#[pymethods]
impl AxModel {
    /// Load a GGUF model from `path`. Raises `RuntimeError` on failure.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let backend: Arc<dyn InferenceBackend> = Arc::new(RouterBackend::from_env());
        let (handle, _meta) = backend
            .load_model(&PathBuf::from(path), LoadConfig::default())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            backend,
            handle,
            runtime,
            runtime_lock: Mutex::new(()),
        })
    }

    /// Generate text from a raw prompt string.
    ///
    /// Releases the GIL during inference so other Python threads can run.
    /// Returns the full generated text as a single string.
    #[pyo3(signature = (prompt, max_tokens=None, temperature=None))]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
    ) -> PyResult<String> {
        let (tx, rx) = mpsc::channel::<GenerateEvent>(512);
        let params = GenerationParams {
            temperature,
            max_tokens,
            ..Default::default()
        };

        self.backend
            .generate(self.handle, GenerateInput::Text(prompt), params, tx)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.collect_events(py, rx)
    }

    /// Generate text from structured chat messages.
    ///
    /// `messages` is a list of `(role, content)` tuples, e.g.
    /// `[("system", "You are helpful."), ("user", "Hello!")]`.
    ///
    /// Releases the GIL during inference.
    #[pyo3(signature = (messages, max_tokens=None, temperature=None))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<(String, String)>,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
    ) -> PyResult<String> {
        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| ChatMessage {
                role,
                content: serde_json::Value::String(content),
            })
            .collect();

        let (tx, rx) = mpsc::channel::<GenerateEvent>(512);
        let params = GenerationParams {
            temperature,
            max_tokens,
            ..Default::default()
        };

        self.backend
            .generate(self.handle, GenerateInput::Chat(chat_messages), params, tx)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.collect_events(py, rx)
    }

    /// Encode `text` into token IDs.
    ///
    /// `add_bos` controls whether a beginning-of-sequence token is prepended.
    #[pyo3(signature = (text, add_bos=false))]
    fn tokenize(&self, text: &str, add_bos: bool) -> PyResult<Vec<u32>> {
        self.backend
            .tokenize(self.handle, text, add_bos)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Decode a list of token IDs back to a UTF-8 string.
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.backend
            .decode_tokens(self.handle, &tokens)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

impl AxModel {
    /// Drain the generate event channel into a complete string.
    ///
    /// Releases the Python GIL for the duration of the blocking wait so that
    /// other Python threads (e.g. a GUI or another model) can run.
    fn collect_events(
        &self,
        py: Python<'_>,
        mut rx: mpsc::Receiver<GenerateEvent>,
    ) -> PyResult<String> {
        let runtime_guard = self
            .runtime_lock
            .lock()
            .map_err(|_| PyRuntimeError::new_err("runtime lock poisoned"))?;
        let runtime = &self.runtime;

        let result: Result<String, String> = py.allow_threads(move || {
            runtime.block_on(async move {
                let mut output = String::new();
                while let Some(event) = rx.recv().await {
                    match event {
                        GenerateEvent::Token(text) => output.push_str(&text),
                        GenerateEvent::TokenLogprob { .. } | GenerateEvent::ToolCall { .. } => {}
                        GenerateEvent::Done(_) => break,
                        GenerateEvent::Error(e) => return Err(e),
                    }
                }
                Ok(output)
            })
        });
        drop(runtime_guard);

        result.map_err(PyRuntimeError::new_err)
    }
}

impl Drop for AxModel {
    fn drop(&mut self) {
        if let Err(e) = self.backend.unload_model(self.handle) {
            tracing::warn!("ax-serving-py: unload_model on drop failed: {e}");
        }
    }
}

// ── Module ────────────────────────────────────────────────────────────────────

/// ax_serving — Python bindings for the ax-serving inference platform.
///
/// Exports: `AxModel`
#[pymodule]
fn ax_serving(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AxModel>()?;
    Ok(())
}
