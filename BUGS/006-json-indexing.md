# BUG-006: JSON Array Indexing Without Bounds Check

**Severity:** Low
**File:** `crates/ax-serving-engine/src/llamacpp.rs`
**Lines:** 1401, 1435, 1461, 1495, 1520, 1546
**Status:** ❌ FALSE POSITIVE — No fix needed

## Code

```rust
val["choices"][0]["logprobs"]["content"]  // Line 1401
val["choices"][0]["text"].as_str()        // Line 1435
val["choices"][0]["finish_reason"]        // Line 1461
val["choices"][0]["message"]["content"]   // Line 1495
val["choices"][0]["message"]["tool_calls"] // Line 1520
val["choices"][0]["finish_reason"]        // Line 1546
```

## Analysis

In `serde_json`, indexing a `Value::Array` with an out-of-bounds integer returns `Value::Null` — it does **not** panic. All downstream access chains terminate in null-safe operations:

- `parse_nonstream_logprobs`: `.as_array()` on `Null` returns `None`; `.map(...)` on `None` returns `None`/empty Vec — no panic
- Text/content reads: `.as_str()` on `Null` returns `None`; `.or_else(...).unwrap_or("")` returns `""` — no panic
- `finish_reason`: `.as_str().unwrap_or("stop")` returns `"stop"` — no panic
- `tool_calls`: `.as_array()` on `Null` returns `None`; `if let Some(...)` does not enter the block — no panic

## Verdict

**False Positive** — `serde_json` OOB indexing is defined to return `Null`. Every accessor in these chains handles `Null` gracefully. No bounds check is needed.
