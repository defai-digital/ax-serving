//! Time-related utility functions.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Returns current Unix timestamp in milliseconds.
pub fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Returns elapsed milliseconds since the given instant.
pub fn elapsed_ms(instant: Instant) -> u64 {
    instant.elapsed().as_millis() as u64
}

/// Returns elapsed microseconds since the given instant.
pub fn elapsed_us(instant: Instant) -> u64 {
    instant.elapsed().as_micros() as u64
}
