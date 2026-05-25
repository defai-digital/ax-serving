//! llama_tokenize / llama_token_to_piece C API functions.

use std::str;

use crate::types::LlamaModel;

pub type LlamaToken = i32;

/// Tokenize a text string into token IDs.
///
/// Returns the number of tokens written, or a negative value on error.
///
/// # Safety
/// - `text` must point to at least `text_len` bytes of valid UTF-8.
/// - `tokens` must point to a buffer of at least `n_tokens_max` `i32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_tokenize(
    model: *const LlamaModel,
    text: *const libc::c_char,
    text_len: i32,
    tokens: *mut LlamaToken,
    n_tokens_max: i32,
    add_bos: bool,
    _special: bool,
) -> i32 {
    if model.is_null() || text.is_null() || tokens.is_null() {
        return -1;
    }
    if n_tokens_max <= 0 {
        return -1;
    }
    if text_len < 0 {
        return -1;
    }

    let text_bytes = unsafe { std::slice::from_raw_parts(text as *const u8, text_len as usize) };
    let text_str = match str::from_utf8(text_bytes) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let m = unsafe { &*model };
    let result = m.inner.backend.tokenize(m.inner.handle, text_str, add_bos);

    match result {
        Ok(ids) => {
            // llama.h contract: return -(n_needed) when buffer is too small.
            // Callers use the negative value to learn the required size and retry.
            if ids.len() > n_tokens_max as usize {
                return -(ids.len().min(i32::MAX as usize) as i32);
            }
            for (i, id) in ids.iter().enumerate() {
                unsafe {
                    *tokens.add(i) = *id as LlamaToken;
                }
            }
            ids.len() as i32
        }
        Err(e) => {
            tracing::error!("llama_tokenize: {e}");
            -1
        }
    }
}

/// Decode a single token ID to its UTF-8 string piece.
///
/// Returns the number of bytes written (may be 0 for special tokens).
///
/// # Safety
/// - `buf` must point to a writable buffer of at least `length` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_token_to_piece(
    model: *const LlamaModel,
    token: LlamaToken,
    buf: *mut libc::c_char,
    length: i32,
) -> i32 {
    if model.is_null() || buf.is_null() || length <= 0 {
        return -1;
    }

    let m = unsafe { &*model };
    match m
        .inner
        .backend
        .decode_tokens(m.inner.handle, &[token as u32])
    {
        Ok(piece) => {
            let bytes = piece.as_bytes();
            let n = bytes.len();
            // llama.h contract: return -(n_needed) when buffer too small.
            // The buffer is byte-counted output, not a C string; exact-size
            // buffers are valid and no trailing nul is written.
            if n > length as usize {
                return -(n.min(i32::MAX as usize) as i32);
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
            }
            n as i32
        }
        Err(e) => {
            tracing::error!("llama_token_to_piece: {e}");
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use ax_serving_engine::{
        GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig, ModelHandle,
        ModelMetadata, ThermalState,
    };

    use super::*;
    use crate::types::LlamaModel;

    struct RecordingBackend {
        observed: Arc<Mutex<Option<String>>>,
        decoded: String,
    }

    impl InferenceBackend for RecordingBackend {
        fn load_model(
            &self,
            _path: &Path,
            _config: LoadConfig,
        ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
            anyhow::bail!("not used in tests")
        }

        fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _handle: ModelHandle,
            _input: GenerateInput,
            _params: GenerationParams,
            _tx: tokio::sync::mpsc::Sender<GenerateEvent>,
        ) -> anyhow::Result<()> {
            anyhow::bail!("not used in tests")
        }

        fn tokenize(
            &self,
            _handle: ModelHandle,
            text: &str,
            _add_bos: bool,
        ) -> anyhow::Result<Vec<u32>> {
            *self.observed.lock().unwrap() = Some(text.to_string());
            Ok(vec![1, 2])
        }

        fn decode_tokens(&self, _handle: ModelHandle, _tokens: &[u32]) -> anyhow::Result<String> {
            Ok(self.decoded.clone())
        }

        fn eos_tokens(&self, _handle: ModelHandle) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn thermal_state(&self) -> ThermalState {
            ThermalState::Nominal
        }

        fn recommended_concurrency(&self) -> usize {
            1
        }
    }

    #[test]
    fn llama_tokenize_respects_text_len_without_requiring_nul_termination() {
        let observed = Arc::new(Mutex::new(None));
        let backend = Arc::new(RecordingBackend {
            observed: Arc::clone(&observed),
            decoded: String::new(),
        });
        let model = LlamaModel::new(ModelHandle(1), backend, 8, 16, -1, -1, -1);
        let bytes = b"helloTRAILING\0";
        let mut out = [0_i32; 4];

        let n = unsafe {
            llama_tokenize(
                &model,
                bytes.as_ptr() as *const libc::c_char,
                5,
                out.as_mut_ptr(),
                out.len() as i32,
                false,
                false,
            )
        };

        assert_eq!(n, 2);
        assert_eq!(&out[..2], &[1, 2]);
        assert_eq!(observed.lock().unwrap().as_deref(), Some("hello"));
    }

    #[test]
    fn llama_tokenize_rejects_negative_text_len() {
        let observed = Arc::new(Mutex::new(None));
        let backend = Arc::new(RecordingBackend {
            observed: Arc::clone(&observed),
            decoded: String::new(),
        });
        let model = LlamaModel::new(ModelHandle(1), backend, 8, 16, -1, -1, -1);
        let bytes = b"hello";
        let mut out = [0_i32; 4];

        let n = unsafe {
            llama_tokenize(
                &model,
                bytes.as_ptr() as *const libc::c_char,
                -1,
                out.as_mut_ptr(),
                out.len() as i32,
                false,
                false,
            )
        };

        assert_eq!(n, -1);
        assert!(observed.lock().unwrap().is_none());
    }

    #[test]
    fn llama_token_to_piece_accepts_exact_size_buffer() {
        let backend = Arc::new(RecordingBackend {
            observed: Arc::new(Mutex::new(None)),
            decoded: "hello".to_string(),
        });
        let model = LlamaModel::new(ModelHandle(1), backend, 8, 16, -1, -1, -1);
        let mut out = [0_i8; 5];

        let n = unsafe { llama_token_to_piece(&model, 1, out.as_mut_ptr(), out.len() as i32) };

        assert_eq!(n, 5);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len()) },
            b"hello"
        );
    }

    #[test]
    fn llama_token_to_piece_reports_required_size_when_buffer_too_small() {
        let backend = Arc::new(RecordingBackend {
            observed: Arc::new(Mutex::new(None)),
            decoded: "hello".to_string(),
        });
        let model = LlamaModel::new(ModelHandle(1), backend, 8, 16, -1, -1, -1);
        let mut out = [0_i8; 4];

        let n = unsafe { llama_token_to_piece(&model, 1, out.as_mut_ptr(), out.len() as i32) };

        assert_eq!(n, -5);
        assert_eq!(out, [0_i8; 4]);
    }
}
