// DEFERRED (Phase 2): Token-level KV prefix cache.
//
// The ax-engine design used block-level KV export/import for cross-request
// prefix reuse (key = (model_fingerprint, token_hash), LRU eviction).
// This is blocked on a public KV block export/import API in the active native
// backend path. As of 2026-03-23, ax-serving does not expose a stable backend-
// agnostic API for exporting or importing KV blocks across requests.
//
// Current alternative: `ResponseCache` (below) provides exact-match caching
// at the full-response level via Redis/Valkey. This covers the most common
// use case (repeated identical prompts) with no backend-specific dependency.
//
// When a native backend exposes KV block export:
//   1. Add `KvPrefixCache` struct with block allocator (port from ax-engine).
//   2. Wire into `ServingLayer::generate()` pre/post hooks.
//   3. Expose block stats in `/metrics` and `/v1/metrics`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{Context, Result};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex as AsyncMutex, broadcast};
use tracing::warn;

use crate::config::CacheConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CachePreference {
    Enable,
    Disable,
}

#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub writes: AtomicU64,
    pub errors: AtomicU64,
}

pub struct ResponseCache {
    client: redis::Client,
    conn: AsyncMutex<Option<redis::aio::MultiplexedConnection>>,
    key_prefix: String,
    default_ttl: Duration,
    max_ttl: Duration,
    metrics: Arc<CacheMetrics>,
}

#[derive(Default)]
pub struct CacheInflight {
    inner: StdMutex<HashMap<String, broadcast::Sender<()>>>,
}

pub enum CacheInflightEnter {
    Leader(CacheInflightLeaderGuard),
    Follower(broadcast::Receiver<()>),
}

pub struct CacheInflightLeaderGuard {
    key: String,
    tx: broadcast::Sender<()>,
    inflight: Arc<CacheInflight>,
}

impl Drop for CacheInflightLeaderGuard {
    fn drop(&mut self) {
        // Always remove the key, even when the mutex is poisoned.  If we skip
        // the remove on poison, the key stays in the map permanently: every
        // subsequent caller for the same cache key becomes a follower
        // subscribing to the now-dropped sender and receives RecvError::Closed.
        match self.inflight.inner.lock() {
            Ok(mut g) => {
                g.remove(&self.key);
            }
            Err(e) => {
                e.into_inner().remove(&self.key);
            }
        }
        let _ = self.tx.send(());
    }
}

impl CacheInflight {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter(self: &Arc<Self>, key: &str) -> CacheInflightEnter {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(err) => {
                warn!("cache inflight mutex poisoned; recovering from poisoned lock");
                err.into_inner()
            }
        };
        if let Some(tx) = guard.get(key) {
            return CacheInflightEnter::Follower(tx.subscribe());
        }
        let (tx, _rx) = broadcast::channel(16);
        guard.insert(key.to_string(), tx.clone());
        CacheInflightEnter::Leader(CacheInflightLeaderGuard {
            key: key.to_string(),
            tx,
            inflight: Arc::clone(self),
        })
    }
}

impl ResponseCache {
    pub fn new(cfg: &CacheConfig) -> Result<Self> {
        let client =
            redis::Client::open(cfg.url.as_str()).with_context(|| "opening valkey client")?;
        let default_ttl = parse_ttl(&cfg.default_ttl)?;
        let max_ttl = parse_ttl(&cfg.max_ttl)?;
        Ok(Self {
            client,
            conn: AsyncMutex::new(None),
            key_prefix: cfg.key_prefix.clone(),
            default_ttl,
            max_ttl,
            metrics: Arc::new(CacheMetrics::default()),
        })
    }

    pub fn metrics(&self) -> Arc<CacheMetrics> {
        Arc::clone(&self.metrics)
    }

    pub fn ttl_for_request(&self, requested: Option<&str>) -> Result<Duration> {
        let mut ttl = requested
            .map(parse_ttl)
            .transpose()?
            .unwrap_or(self.default_ttl);
        if ttl > self.max_ttl {
            ttl = self.max_ttl;
        }
        Ok(ttl)
    }

    pub fn make_key(&self, payload: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(payload);
        let digest = hex::encode(h.finalize());
        format!("{}:{digest}", self.key_prefix)
    }

    async fn connection(&self) -> Result<redis::aio::MultiplexedConnection> {
        let mut guard = self.conn.lock().await;
        if let Some(conn) = guard.as_ref() {
            return Ok(conn.clone());
        }
        let conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .with_context(|| "connecting to valkey")?;
        *guard = Some(conn.clone());
        Ok(conn)
    }

    async fn reset_connection(&self) {
        let mut guard = self.conn.lock().await;
        *guard = None;
    }

    pub async fn get(&self, key: &str) -> Result<Option<String>> {
        let mut conn = self.connection().await?;
        let raw: Option<String> = match conn.get(key).await {
            Ok(v) => v,
            Err(_) => {
                // Retry once with a fresh connection; stale sockets can occur
                // after server restarts/network blips.
                self.reset_connection().await;
                let mut conn = self.connection().await?;
                conn.get(key)
                    .await
                    .with_context(|| "reading cached response")?
            }
        };
        let Some(raw) = raw else {
            self.metrics.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        self.metrics.hits.fetch_add(1, Ordering::Relaxed);
        Ok(Some(raw))
    }

    pub async fn set<T: Serialize>(&self, key: &str, value: &T, ttl: Duration) -> Result<()> {
        let mut conn = self.connection().await?;
        let raw = serde_json::to_string(value).with_context(|| "serializing response for cache")?;
        if conn
            .set_ex::<_, _, ()>(key, raw.clone(), ttl.as_secs())
            .await
            .is_err()
        {
            self.reset_connection().await;
            let mut conn = self.connection().await?;
            conn.set_ex::<_, _, ()>(key, raw, ttl.as_secs())
                .await
                .with_context(|| "writing cached response")?;
        }
        self.metrics.writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

const SECS_PER_MIN: u64 = 60;
const SECS_PER_HOUR: u64 = 60 * SECS_PER_MIN;
const SECS_PER_DAY: u64 = 24 * SECS_PER_HOUR;
const SECS_PER_WEEK: u64 = 7 * SECS_PER_DAY;
const SECS_PER_MONTH: u64 = 30 * SECS_PER_DAY;

/// Hard cap: any TTL above this is a configuration error, not a valid cache lifetime.
const MAX_TTL_SECS: u64 = 90 * SECS_PER_DAY;

pub fn parse_ttl(input: &str) -> Result<Duration> {
    let s = input.trim().to_ascii_lowercase();
    let parse = |n: &str| -> Result<u64> { n.parse::<u64>().map_err(anyhow::Error::from) };
    let checked_mul = |a: u64, b: u64| -> Result<u64> {
        a.checked_mul(b)
            .ok_or_else(|| anyhow::anyhow!("TTL value overflows u64"))
    };

    let secs = if let Some(n) = s.strip_suffix("months") {
        checked_mul(parse(n.trim())?, SECS_PER_MONTH)?
    } else if let Some(n) = s.strip_suffix("month") {
        checked_mul(parse(n.trim())?, SECS_PER_MONTH)?
    } else if let Some(n) = s.strip_suffix("mo") {
        checked_mul(parse(n.trim())?, SECS_PER_MONTH)?
    } else if let Some(n) = s.strip_suffix('w') {
        checked_mul(parse(n)?, SECS_PER_WEEK)?
    } else if let Some(n) = s.strip_suffix('d') {
        checked_mul(parse(n)?, SECS_PER_DAY)?
    } else if let Some(n) = s.strip_suffix('h') {
        checked_mul(parse(n)?, SECS_PER_HOUR)?
    } else if let Some(n) = s.strip_suffix('m') {
        checked_mul(parse(n)?, SECS_PER_MIN)?
    } else if let Some(n) = s.strip_suffix('s') {
        parse(n)?
    } else {
        parse(&s)?
    };

    if secs == 0 {
        anyhow::bail!("ttl must be > 0");
    }
    if secs > MAX_TTL_SECS {
        anyhow::bail!("ttl exceeds maximum of 90 days");
    }
    Ok(Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ttl_variants() {
        assert_eq!(parse_ttl("1h").unwrap(), Duration::from_secs(3600));
        assert_eq!(parse_ttl("4h").unwrap(), Duration::from_secs(4 * 3600));
        assert_eq!(parse_ttl("24h").unwrap(), Duration::from_secs(24 * 3600));
        assert_eq!(parse_ttl("1w").unwrap(), Duration::from_secs(7 * 24 * 3600));
        assert_eq!(
            parse_ttl("1month").unwrap(),
            Duration::from_secs(30 * 24 * 3600)
        );
        // Regression: "months" suffix was incorrectly matched as "month" + "s",
        // causing parse("2s") to fail. "months" must be checked before "month".
        assert_eq!(
            parse_ttl("2months").unwrap(),
            Duration::from_secs(2 * 30 * 24 * 3600)
        );
        assert_eq!(
            parse_ttl("1months").unwrap(),
            Duration::from_secs(30 * 24 * 3600)
        );
        assert_eq!(
            parse_ttl("30d").unwrap(),
            Duration::from_secs(30 * 24 * 3600)
        );
    }

    #[test]
    fn parse_ttl_rejects_invalid() {
        assert!(parse_ttl("0").is_err());
        assert!(parse_ttl("0h").is_err());
        assert!(parse_ttl("abc").is_err());
    }

    #[test]
    fn parse_ttl_rejects_overflow() {
        // Hard cap is 90 days; all values below vastly exceed it.
        assert!(parse_ttl("20000000months").is_err());
        assert!(parse_ttl("1000000000w").is_err());
        assert!(parse_ttl("100000000000d").is_err());
        // Values over 90 days must also be rejected.
        assert!(parse_ttl("91d").is_err());
        assert!(parse_ttl("4mo").is_err()); // 4 months = 120d, over cap
    }

    #[test]
    fn ttl_for_request_caps_to_max() {
        let cfg = CacheConfig {
            enabled: true,
            url: "redis://127.0.0.1:6379".into(),
            key_prefix: "axs:test".into(),
            default_ttl: "1h".into(),
            max_ttl: "30d".into(),
        };
        let cache = ResponseCache::new(&cfg).unwrap();
        // 60d is under the 90-day hard cap but over the 30-day config cap;
        // ttl_for_request must clamp it to max_ttl = 30d.
        assert_eq!(
            cache.ttl_for_request(Some("60d")).unwrap(),
            Duration::from_secs(30 * 24 * 3600)
        );
        assert_eq!(
            cache.ttl_for_request(None).unwrap(),
            Duration::from_secs(3600)
        );
    }

    #[tokio::test]
    async fn cache_inflight_single_leader() {
        let inflight = Arc::new(CacheInflight::new());
        let k = "k1";
        let leader = match inflight.enter(k) {
            CacheInflightEnter::Leader(g) => g,
            CacheInflightEnter::Follower(_) => panic!("expected leader"),
        };
        let mut follower = match inflight.enter(k) {
            CacheInflightEnter::Follower(rx) => rx,
            CacheInflightEnter::Leader(_) => panic!("expected follower"),
        };

        let wait = tokio::spawn(async move {
            let _ = follower.recv().await;
            true
        });
        drop(leader);
        assert!(wait.await.unwrap());
    }
}
