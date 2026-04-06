# BUG-015: Unbounded Startup Wait in Thor Agent `wait_for_sglang`

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/sglang.rs`
**Lines:** 15–25
**Status:** ✅ FIXED (2026-03-28)

## Description

The function loops forever with no timeout or maximum retry count when waiting for the SGLang runtime to become healthy.

```rust
pub async fn wait_for_sglang(client: &reqwest::Client, base_url: &str) -> Result<()> {
    let url = format!("{base_url}/health");
    loop {
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            // ... log warning, sleep 1s, retry
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}
```

## Impact

If the SGLang runtime never starts (misconfiguration, binary not installed, port conflict), the Thor agent hangs indefinitely on startup with no timeout or diagnostic guidance. Blocks the entire worker initialization.

## Fix

Add a configurable timeout:

```rust
pub async fn wait_for_sglang(client: &reqwest::Client, base_url: &str, timeout: Duration) -> Result<()> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("SGLang runtime at {base_url} did not become healthy within {:?}", timeout);
        }
        // ... existing retry logic
    }
}
```

## Fix Applied

Added a configurable timeout (`AXS_THOR_STARTUP_TIMEOUT_SECS`, default 120s). The loop now computes a deadline before entering and checks it after each failed probe. Exceeding the deadline returns `anyhow::bail!` with a clear diagnostic message naming the URL and timeout.
