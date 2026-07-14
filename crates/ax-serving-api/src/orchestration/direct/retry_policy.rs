//! Safe-retry predicates for direct dispatch.
//!
//! # Commitment rules (must not be weakened)
//!
//! - At most **one** safe retry is allowed (`max_dispatch_attempts` clamped to 1..=2).
//! - A second attempt is allowed only for a **proven connection failure** or a
//!   **trusted typed pre-admission rejection** (`x-ax-admission-state: not-admitted`).
//! - Arbitrary runtime 5xx responses are **never** retried: the runtime may already
//!   have admitted the request.
//! - Once response headers / first client byte have been committed on a streaming
//!   path, no retry is possible (retry decisions are made before `build_response`).

/// Trusted typed pre-admission rejection from the agent runtime.
///
/// Only this header-marked rejection (on a non-success status) is eligible for a
/// safe second attempt. Generic 4xx/5xx without the admission-state header are not.
pub(super) fn is_typed_not_admitted(response: &reqwest::Response) -> bool {
    !response.status().is_success()
        && response
            .headers()
            .get(ax_serving_protocol::ADMISSION_STATE_HEADER)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.eq_ignore_ascii_case("not-admitted"))
}

/// Connect-level failure that did not time out — eligible for one safe retry.
pub(super) fn is_retryable_connect_failure(error: &reqwest::Error) -> bool {
    error.is_connect() && !error.is_timeout()
}
