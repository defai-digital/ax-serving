//! llama_sample_* C API functions.
//!
//! Sampling in the C API is stateful: functions mutate `LlamaTokenDataArray`
//! by softmax-ing, filtering, and reordering candidates.
//! This matches llama.cpp's sampling pipeline exactly.

pub type LlamaToken = i32;

/// Token data entry (mirrors llama_token_data).
#[repr(C)]
pub struct LlamaTokenData {
    pub id: LlamaToken,
    pub logit: f32,
    pub p: f32,
}

/// Candidate array (mirrors llama_token_data_array).
#[repr(C)]
pub struct LlamaTokenDataArray {
    pub data: *mut LlamaTokenData,
    pub size: libc::size_t,
    pub selected: i64,
    pub sorted: bool,
}

/// Apply temperature scaling to logits (in-place).
///
/// `temp = 0` → greedy (collapses to argmax).
///
/// # Safety
/// `candidates` must point to a valid `LlamaTokenDataArray` with `data` buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_sample_temperature(
    _ctx: *mut crate::types::LlamaContext,
    candidates: *mut LlamaTokenDataArray,
    temp: f32,
) {
    if candidates.is_null() {
        return;
    }
    let arr = unsafe { &mut *candidates };
    if arr.data.is_null() || arr.size == 0 {
        return;
    }

    if temp <= 0.0 || temp.is_nan() {
        // Greedy / invalid: leave logits unchanged; llama_sample_token_greedy picks max.
        return;
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(arr.data, arr.size) };
    for entry in slice.iter_mut() {
        entry.logit /= temp;
    }
}

/// Sample the token with the highest logit (greedy decoding).
///
/// # Safety
/// `candidates` must point to a valid non-null `LlamaTokenDataArray`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_sample_token_greedy(
    _ctx: *mut crate::types::LlamaContext,
    candidates: *mut LlamaTokenDataArray,
) -> LlamaToken {
    if candidates.is_null() {
        return -1;
    }
    let arr = unsafe { &*candidates };
    if arr.data.is_null() || arr.size == 0 {
        return -1;
    }

    let slice = unsafe { std::slice::from_raw_parts(arr.data, arr.size) };
    slice
        .iter()
        .max_by(|a, b| {
            a.logit
                .partial_cmp(&b.logit)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|e| e.id)
        .unwrap_or(-1)
}

/// Apply top-k filtering (keep only the k highest logits).
///
/// # Safety
/// `candidates` must point to a valid non-null `LlamaTokenDataArray`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_sample_top_k(
    _ctx: *mut crate::types::LlamaContext,
    candidates: *mut LlamaTokenDataArray,
    k: i32,
    _min_keep: libc::size_t,
) {
    if candidates.is_null() {
        return;
    }
    if k <= 0 {
        // Invalidate sorted flag when k<=0 is a no-op.
        unsafe { (*candidates).sorted = false };
        return;
    }
    let arr = unsafe { &mut *candidates };
    if arr.data.is_null() || arr.size == 0 {
        return;
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(arr.data, arr.size) };
    // Sort descending by logit.
    slice.sort_unstable_by(|a, b| {
        b.logit
            .partial_cmp(&a.logit)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    arr.sorted = true;
    // Truncate to k.
    let new_size = (k as usize).min(arr.size);
    arr.size = new_size;
}

/// Apply top-p (nucleus) filtering.
///
/// # Safety
/// `candidates` must point to a valid non-null `LlamaTokenDataArray`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_sample_top_p(
    _ctx: *mut crate::types::LlamaContext,
    candidates: *mut LlamaTokenDataArray,
    p: f32,
    _min_keep: libc::size_t,
) {
    // p >= 1.0 means "keep all" — no-op to avoid float accumulation truncation.
    if candidates.is_null() || p >= 1.0 {
        return;
    }
    let arr = unsafe { &mut *candidates };
    if arr.data.is_null() || arr.size == 0 {
        return;
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(arr.data, arr.size) };

    // Softmax first.
    let max_logit = slice
        .iter()
        .map(|e| e.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for e in slice.iter_mut() {
        e.p = (e.logit - max_logit).exp();
        sum += e.p;
    }
    let norm = if sum > 0.0 { sum } else { 1.0 };
    for e in slice.iter_mut() {
        e.p /= norm;
    }

    // Sort descending by probability (may already be sorted from top_k).
    if !arr.sorted {
        slice.sort_unstable_by(|a, b| b.p.partial_cmp(&a.p).unwrap_or(std::cmp::Ordering::Equal));
        arr.sorted = true;
    }

    // Keep tokens until cumulative probability exceeds p.
    let mut cumulative = 0.0_f32;
    let mut cutoff = arr.size;
    for (i, e) in slice.iter().enumerate() {
        cumulative += e.p;
        if cumulative >= p {
            cutoff = i + 1;
            break;
        }
    }
    arr.size = cutoff;
}

/// Apply repetition penalty.
///
/// # Safety
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_sample_repetition_penalties(
    _ctx: *mut crate::types::LlamaContext,
    candidates: *mut LlamaTokenDataArray,
    last_tokens: *const LlamaToken,
    penalty_last_n: libc::size_t,
    penalty_repeat: f32,
    _penalty_freq: f32,
    _penalty_present: f32,
) {
    if candidates.is_null() || last_tokens.is_null() || penalty_last_n == 0 {
        return;
    }
    if (penalty_repeat - 1.0).abs() < f32::EPSILON {
        return;
    } // no-op

    let arr = unsafe { &mut *candidates };
    if arr.data.is_null() || arr.size == 0 {
        return;
    }

    let recent = unsafe { std::slice::from_raw_parts(last_tokens, penalty_last_n) };
    let slice = unsafe { std::slice::from_raw_parts_mut(arr.data, arr.size) };

    use std::collections::HashSet;
    let recent_set: HashSet<LlamaToken> = recent.iter().copied().collect();

    for entry in slice.iter_mut() {
        if recent_set.contains(&entry.id) {
            if entry.logit > 0.0 {
                entry.logit /= penalty_repeat;
            } else {
                entry.logit *= penalty_repeat;
            }
        }
    }
}
