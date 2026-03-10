//! llama_tokenize / llama_token_to_piece C API functions.

use std::ffi::CStr;

use crate::types::LlamaModel;

pub type LlamaToken = i32;

/// Tokenize a text string into token IDs.
///
/// Returns the number of tokens written, or a negative value on error.
///
/// # Safety
/// - `text` must be a valid null-terminated UTF-8 C string.
/// - `tokens` must point to a buffer of at least `n_tokens_max` `i32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_tokenize(
    model: *const LlamaModel,
    text: *const libc::c_char,
    _text_len: i32,
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

    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let m = unsafe { &*model };
    let result = m.backend.tokenize(m.handle, text_str, add_bos);

    match result {
        Ok(ids) => {
            // llama.h contract: return -(n_needed) when buffer is too small.
            // Callers use the negative value to learn the required size and retry.
            if ids.len() > n_tokens_max as usize {
                return -(ids.len() as i32);
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
    match m.backend.decode_tokens(m.handle, &[token as u32]) {
        Ok(piece) => {
            let bytes = piece.as_bytes();
            let n = bytes.len();
            // llama.h contract: return -(n_needed) when buffer too small.
            // `n + 1` is needed (n bytes + null terminator); length must be > n.
            if n >= length as usize {
                return -(n as i32);
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
                *buf.add(n) = 0; // null terminator
            }
            n as i32
        }
        Err(e) => {
            tracing::error!("llama_token_to_piece: {e}");
            -1
        }
    }
}
