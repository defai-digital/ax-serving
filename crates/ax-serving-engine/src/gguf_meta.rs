//! Minimal GGUF v2/v3 metadata reader.
//!
//! Reads only the KV metadata section — does **not** parse tensor data.
//! Used for:
//! - Architecture detection before routing (`general.architecture`)
//! - Backfilling `ModelMetadata` fields without requiring a full model load
//!
//! # Format
//!
//! ```text
//! magic:      [u8; 4] = b"GGUF"
//! version:    u32 LE  (2 or 3)
//! n_tensors:  u64 LE
//! n_kv:       u64 LE
//! kv_pairs:   [key: string, val_type: u32, value: ...] × n_kv
//! ```
//!
//! All multi-byte integers are little-endian.  Strings are length-prefixed:
//! `u64` byte count followed by UTF-8 bytes (no NUL terminator).

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// ── Public types ──────────────────────────────────────────────────────────────

/// Architecture and shape metadata extracted from a GGUF file header.
#[derive(Debug, Default, Clone)]
pub struct GgufMeta {
    /// Value of `general.architecture` (e.g. `"llama"`, `"qwen2"`, `"gemma3"`).
    /// Empty string if the key was not found.
    pub architecture: String,
    /// `{arch}.context_length`
    pub context_length: u32,
    /// `{arch}.block_count` (number of transformer layers)
    pub block_count: u32,
    /// `{arch}.attention.head_count`
    pub head_count: u32,
    /// `{arch}.attention.head_count_kv`
    pub head_count_kv: u32,
    /// `{arch}.embedding_length`
    pub embedding_length: u32,
    /// `{arch}.vocab_size` or length of `tokenizer.ggml.tokens` array
    pub vocab_size: u32,
    /// `{arch}.pooling_type` — non-zero for embedding models (e.g. `3` for Qwen3-Embedding).
    /// Used to auto-detect that `--embedding` must be passed to llama-server.
    pub pooling_type: u32,
}

/// Hard errors from the GGUF metadata reader.
#[derive(Debug)]
pub enum GgufMetaError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for GgufMetaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error reading GGUF header: {e}"),
            Self::InvalidMagic(m) => write!(f, "invalid GGUF magic: {:?} (expected b\"GGUF\")", m),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version {v} (expected 2 or 3)")
            }
        }
    }
}

impl std::error::Error for GgufMetaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Read GGUF metadata from a file path.
///
/// Returns `Err` only for hard failures (bad magic, unsupported version,
/// I/O error on the header itself).  Missing optional fields default to
/// `0` / empty string.
pub fn read_gguf_meta(path: &Path) -> Result<GgufMeta, GgufMetaError> {
    let f = std::fs::File::open(path).map_err(GgufMetaError::Io)?;
    parse_gguf_meta(std::io::BufReader::new(f))
}

// ── Core parser ───────────────────────────────────────────────────────────────

fn parse_gguf_meta<R: Read + Seek>(mut r: R) -> Result<GgufMeta, GgufMetaError> {
    // 1. Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(GgufMetaError::Io)?;
    if &magic != b"GGUF" {
        return Err(GgufMetaError::InvalidMagic(magic));
    }

    // 2. Version (u32 LE) — accept 2 or 3
    let version = read_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        return Err(GgufMetaError::UnsupportedVersion(version));
    }

    // 3. n_tensors (u64), n_kv (u64) — layout identical for v2 and v3
    let _n_tensors = read_u64(&mut r)?;
    let n_kv = read_u64(&mut r)?;

    // 4. Collect KV pairs into typed buckets for the keys we care about.
    let mut strings: HashMap<String, String> = HashMap::new();
    let mut u32s: HashMap<String, u32> = HashMap::new();
    let mut u64s: HashMap<String, u64> = HashMap::new();
    let mut array_lens: HashMap<String, u64> = HashMap::new();

    // Cap iteration to prevent runaway on malformed files.
    let limit = n_kv.min(2048);

    for _ in 0..limit {
        let key = match read_str(&mut r) {
            Ok(s) => s,
            Err(_) => break,
        };
        let val_type = match read_u32(&mut r) {
            Ok(v) => v,
            Err(_) => break,
        };

        match val_type {
            // UINT32
            4 => match read_u32(&mut r) {
                Ok(v) => {
                    u32s.insert(key, v);
                }
                Err(_) => break,
            },
            // STRING
            8 => match read_str(&mut r) {
                Ok(s) => {
                    strings.insert(key, s);
                }
                Err(_) => break,
            },
            // ARRAY — store only the element count, skip elements
            9 => {
                let elem_type = match read_u32(&mut r) {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let n = match read_u64(&mut r) {
                    Ok(v) => v,
                    Err(_) => break,
                };
                array_lens.insert(key, n);
                if skip_array_elements(&mut r, elem_type, n).is_err() {
                    break;
                }
            }
            // UINT64
            10 => match read_u64(&mut r) {
                Ok(v) => {
                    u64s.insert(key, v);
                }
                Err(_) => break,
            },
            // All other value types: skip
            _ => {
                if skip_value(&mut r, val_type).is_err() {
                    break;
                }
            }
        }
    }

    // 5. Extract metadata from the collected maps.
    let architecture = strings.remove("general.architecture").unwrap_or_default();
    let arch = architecture.clone();

    let vocab_size = u32s
        .get(&format!("{arch}.vocab_size"))
        .copied()
        // Use try_from instead of `as u32` to avoid silent truncation if a
        // future model stores a vocab_size that exceeds u32::MAX.
        .or_else(|| {
            u64s.get(&format!("{arch}.vocab_size"))
                .and_then(|&v| u32::try_from(v).ok())
        })
        .or_else(|| {
            array_lens
                .get("tokenizer.ggml.tokens")
                .and_then(|&v| u32::try_from(v).ok())
        })
        .unwrap_or(0);

    Ok(GgufMeta {
        context_length: u32s
            .get(&format!("{arch}.context_length"))
            .copied()
            .unwrap_or(0),
        block_count: u32s
            .get(&format!("{arch}.block_count"))
            .copied()
            .unwrap_or(0),
        head_count: u32s
            .get(&format!("{arch}.attention.head_count"))
            .copied()
            .unwrap_or(0),
        head_count_kv: u32s
            .get(&format!("{arch}.attention.head_count_kv"))
            .copied()
            .unwrap_or(0),
        embedding_length: u32s
            .get(&format!("{arch}.embedding_length"))
            .copied()
            .unwrap_or(0),
        vocab_size,
        pooling_type: u32s
            .get(&format!("{arch}.pooling_type"))
            .copied()
            .unwrap_or(0),
        architecture,
    })
}

// ── Low-level read helpers ────────────────────────────────────────────────────

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufMetaError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(GgufMetaError::Io)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufMetaError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(GgufMetaError::Io)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a GGUF length-prefixed string (u64 len + utf-8 bytes).
/// Rejects strings > 65 535 bytes to bound memory use.
fn read_str<R: Read>(r: &mut R) -> Result<String, GgufMetaError> {
    let len = read_u64(r)? as usize;
    if len > 0xFFFF {
        return Err(io_err("GGUF string too long (> 65535 bytes)"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(GgufMetaError::Io)?;
    String::from_utf8(buf).map_err(|e| io_err(e.to_string()))
}

/// Skip a single GGUF value of the given `val_type`.
fn skip_value<R: Read + Seek>(r: &mut R, val_type: u32) -> Result<(), GgufMetaError> {
    match val_type {
        0 | 1 | 7 => seek_fwd(r, 1)?, // UINT8, INT8, BOOL
        2 | 3 => seek_fwd(r, 2)?,     // UINT16, INT16
        4..=6 => seek_fwd(r, 4)?,     // UINT32, INT32, FLOAT32
        8 => {
            let n = read_u64(r)?;
            if n > i64::MAX as u64 {
                return Err(io_err("GGUF string value length too large to skip"));
            }
            seek_fwd(r, n as i64)?;
        } // STRING
        9 => {
            let elem_type = read_u32(r)?;
            let n = read_u64(r)?;
            skip_array_elements(r, elem_type, n)?;
        }
        10..=12 => seek_fwd(r, 8)?, // UINT64, INT64, FLOAT64
        _ => return Err(io_err(format!("unknown GGUF value type: {val_type}"))),
    }
    Ok(())
}

/// Skip `n` array elements of `elem_type`.
fn skip_array_elements<R: Read + Seek>(
    r: &mut R,
    elem_type: u32,
    n: u64,
) -> Result<(), GgufMetaError> {
    // Guard against adversarial GGUF files with unrealistic element counts.
    // 2 000 000 covers the largest realistic vocabulary (~150 k) with headroom.
    // Values beyond this indicate a malformed or malicious file.
    const MAX_ELEMENTS: u64 = 2_000_000;
    if n > MAX_ELEMENTS {
        return Err(io_err(format!(
            "GGUF array element count {n} exceeds limit {MAX_ELEMENTS}"
        )));
    }

    // After the cap check, n fits in i64 and n * 8 cannot overflow i64.
    let n = n as i64;
    match elem_type {
        0 | 1 | 7 => seek_fwd(r, n)?,
        2 | 3 => seek_fwd(r, n * 2)?,
        4..=6 => seek_fwd(r, n * 4)?,
        10..=12 => seek_fwd(r, n * 8)?,
        8 => {
            // STRING elements: iterate — each has its own u64 length prefix.
            for _ in 0..n {
                let len = read_u64(r)?;
                if len > i64::MAX as u64 {
                    return Err(io_err("GGUF string element length too large to skip"));
                }
                seek_fwd(r, len as i64)?;
            }
        }
        _ => {
            return Err(io_err(format!(
                "unknown GGUF array element type: {elem_type}"
            )));
        }
    }
    Ok(())
}

fn seek_fwd<R: Seek>(r: &mut R, n: i64) -> Result<(), GgufMetaError> {
    if n > 0 {
        r.seek(SeekFrom::Current(n)).map_err(GgufMetaError::Io)?;
    }
    Ok(())
}

fn io_err(msg: impl Into<String>) -> GgufMetaError {
    GgufMetaError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ── Test GGUF builder ─────────────────────────────────────────────────────

    fn build_test_gguf(arch: &str, u32_kvs: &[(&str, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        let n_kv = 1 + u32_kvs.len() as u64;
        buf.extend_from_slice(&n_kv.to_le_bytes());

        push_string_kv(&mut buf, "general.architecture", arch);
        for (key, val) in u32_kvs {
            push_u32_kv(&mut buf, key, *val);
        }
        buf
    }

    fn push_str_raw(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn push_string_kv(buf: &mut Vec<u8>, key: &str, val: &str) {
        push_str_raw(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        push_str_raw(buf, val);
    }

    fn push_u32_kv(buf: &mut Vec<u8>, key: &str, val: u32) {
        push_str_raw(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&val.to_le_bytes());
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn reads_llama_architecture() {
        let data = build_test_gguf(
            "llama",
            &[
                ("llama.block_count", 32),
                ("llama.embedding_length", 4096),
                ("llama.attention.head_count", 32),
                ("llama.attention.head_count_kv", 8),
                ("llama.context_length", 4096),
                ("llama.vocab_size", 32000),
            ],
        );
        let meta = parse_gguf_meta(Cursor::new(data)).unwrap();
        assert_eq!(meta.architecture, "llama");
        assert_eq!(meta.block_count, 32);
        assert_eq!(meta.embedding_length, 4096);
        assert_eq!(meta.head_count, 32);
        assert_eq!(meta.head_count_kv, 8);
        assert_eq!(meta.context_length, 4096);
        assert_eq!(meta.vocab_size, 32000);
    }

    #[test]
    fn reads_qwen3_architecture() {
        let data = build_test_gguf(
            "qwen3",
            &[
                ("qwen3.block_count", 36),
                ("qwen3.embedding_length", 4096),
                ("qwen3.attention.head_count", 32),
                ("qwen3.attention.head_count_kv", 8),
                ("qwen3.context_length", 32768),
                ("qwen3.vocab_size", 151936),
            ],
        );
        let meta = parse_gguf_meta(Cursor::new(data)).unwrap();
        assert_eq!(meta.architecture, "qwen3");
        assert_eq!(meta.vocab_size, 151936);
        assert_eq!(meta.context_length, 32768);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = build_test_gguf("llama", &[]);
        data[0] = b'X'; // corrupt magic
        assert!(matches!(
            parse_gguf_meta(Cursor::new(data)),
            Err(GgufMetaError::InvalidMagic(_))
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut data = build_test_gguf("llama", &[]);
        // Overwrite version field (bytes 4-7) with version 1.
        data[4] = 1;
        data[5] = 0;
        data[6] = 0;
        data[7] = 0;
        assert!(matches!(
            parse_gguf_meta(Cursor::new(data)),
            Err(GgufMetaError::UnsupportedVersion(1))
        ));
    }

    #[test]
    fn skips_unknown_keys_gracefully() {
        // Add a spurious UINT32 key before the architecture KV.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 kv pairs

        // An unexpected UINT32 key first.
        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        push_str(&mut buf, "general.quantization_version");
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&2u32.to_le_bytes());

        push_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        push_str(&mut buf, "gemma3");

        push_str(&mut buf, "gemma3.block_count");
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&46u32.to_le_bytes());

        let meta = parse_gguf_meta(Cursor::new(buf)).unwrap();
        assert_eq!(meta.architecture, "gemma3");
        assert_eq!(meta.block_count, 46);
    }

    #[test]
    fn handles_array_kv() {
        // Simulate a UINT32 array (e.g. rope_freq overrides) before architecture.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 kv

        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        // ARRAY of 3 UINT32s
        push_str(&mut buf, "some.array");
        buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY type
        buf.extend_from_slice(&4u32.to_le_bytes()); // elem type UINT32
        buf.extend_from_slice(&3u64.to_le_bytes()); // n=3
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());

        push_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut buf, "mistral");

        let meta = parse_gguf_meta(Cursor::new(buf)).unwrap();
        assert_eq!(meta.architecture, "mistral");
    }

    #[test]
    fn rejects_oversized_array_element_count() {
        // A GGUF with an array whose element count exceeds MAX_ELEMENTS (2_000_000).
        // The parser must return Err rather than integer-overflowing or looping.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1

        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        push_str(&mut buf, "some.huge.array");
        buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY type
        buf.extend_from_slice(&4u32.to_le_bytes()); // elem type UINT32
        buf.extend_from_slice(&3_000_000u64.to_le_bytes()); // n = 3M > MAX_ELEMENTS
        // No actual data — parser should reject before reading elements.

        // parse_gguf_meta breaks on the first KV error and returns partial results
        // (not Err) — architecture will be empty, metadata will be zeroed.
        let meta = parse_gguf_meta(Cursor::new(buf)).unwrap();
        assert_eq!(meta.architecture, "");
    }

    #[test]
    fn missing_optional_fields_are_zero() {
        let data = build_test_gguf("llama", &[]); // only architecture, no shape fields
        let meta = parse_gguf_meta(Cursor::new(data)).unwrap();
        assert_eq!(meta.architecture, "llama");
        assert_eq!(meta.block_count, 0);
        assert_eq!(meta.vocab_size, 0);
    }
}
