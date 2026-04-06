# BUG-062: Float Precision Loss in Memory Budget Check

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/memory.rs:13`
**Status:** ❌ FALSE POSITIVE

## Description

```rust
(model_bytes as f64 * 1.1) as u64
```

Converts to f64 for a 10% headroom calculation. f64 has 53 bits of mantissa; u64 has 64 bits. For very large models (unlikely today but possible with future multi-TB models on shared storage), f64 loses integer precision, potentially underestimating the requirement. Also, the cast to u64 truncates (not rounds), so the headroom is slightly less than 10%.

## Fix

Use integer arithmetic: `model_bytes + model_bytes / 10`.
