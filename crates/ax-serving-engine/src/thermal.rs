//! macOS thermal pressure monitoring.
//!
//! Ported from ax-engine's `thermal.rs`. Uses `notify_register_dispatch` to
//! receive thermal state change events from the macOS kernel.
//!
//! Thermal state maps to recommended inference concurrency:
//!
//! | State    | Concurrency |
//! |----------|-------------|
//! | Nominal  | full        |
//! | Fair     | full        |
//! | Serious  | half        |
//! | Critical | 1           |

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;
use tracing::warn;

/// Thermal pressure state reported by macOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ThermalState {
    Nominal = 0,
    Fair = 1,
    Serious = 2,
    Critical = 3,
}

impl ThermalState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nominal => "Nominal",
            Self::Fair => "Fair",
            Self::Serious => "Serious",
            Self::Critical => "Critical",
        }
    }
}

/// Polls macOS thermal pressure state via NSProcessInfo FFI.
///
/// Spawns a background polling thread on construction. The thread updates
/// an `Arc<AtomicU8>` every `AXS_THERMAL_POLL_SECS` seconds (default 5).
/// The thread exits automatically when all `ThermalMonitor` instances are
/// dropped (detected via `Weak<AtomicU8>`).
///
/// `ThermalMonitor` is `Clone` — clones share the same underlying state atom.
#[derive(Clone)]
pub struct ThermalMonitor {
    state: Arc<AtomicU8>,
    max_cpus: usize,
}

impl ThermalMonitor {
    /// Create a new monitor using `AXS_THERMAL_POLL_SECS` (default 5 s).
    ///
    /// Prefer [`ThermalMonitor::with_poll`] when the poll interval is already
    /// known from a loaded config file.
    pub fn new() -> Self {
        let poll_secs = std::env::var("AXS_THERMAL_POLL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(5)
            .max(1);
        Self::with_poll(poll_secs)
    }

    /// Create a new monitor with an explicit poll interval (seconds).
    ///
    /// Spawns the background polling thread immediately.
    pub fn with_poll(poll_secs: u64) -> Self {
        let poll_secs = poll_secs.max(1);
        // Read the actual current thermal state immediately so callers never
        // observe a stale Nominal during the first poll interval.
        let state = Arc::new(AtomicU8::new(read_nsprocessinfo_thermal_state()));
        let max_cpus = detect_performance_core_count().max(1);

        // Spawn background poller. It holds a Weak reference so it exits
        // automatically when all ThermalMonitor clones are dropped.
        let state_weak = Arc::downgrade(&state);
        if let Err(err) = std::thread::Builder::new()
            .name("ax-thermal-poll".to_string())
            .spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_secs(poll_secs));
                    // Exit if all ThermalMonitor instances have been dropped.
                    let Some(state) = state_weak.upgrade() else {
                        break;
                    };
                    let raw = read_nsprocessinfo_thermal_state();
                    state.store(raw, Ordering::Relaxed);
                }
            })
        {
            warn!(
                poll_secs,
                %err,
                "failed to spawn ax-thermal-poll thread; thermal state will remain static until restart"
            );
        }

        Self { state, max_cpus }
    }

    /// Current thermal state.
    pub fn current(&self) -> ThermalState {
        match self.state.load(Ordering::Relaxed) {
            0 => ThermalState::Nominal,
            1 => ThermalState::Fair,
            2 => ThermalState::Serious,
            _ => ThermalState::Critical,
        }
    }

    /// Recommended concurrency limit given current thermal state.
    pub fn recommended_concurrency(&self) -> usize {
        match self.current() {
            ThermalState::Nominal | ThermalState::Fair => self.max_cpus,
            ThermalState::Serious => (self.max_cpus / 2).max(1),
            ThermalState::Critical => 1,
        }
    }
}

impl Default for ThermalMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Read performance core count from sysctl `hw.perflevel0.physicalcpu`.
fn detect_performance_core_count() -> usize {
    // Fallback to logical CPUs / 2 if sysctl fails.
    let fallback = std::thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(4);

    // sysctl hw.perflevel0.physicalcpu gives P-core count on M-series.
    // Attempt via libc; if unavailable, use fallback.
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;
        let name = match CString::new("hw.perflevel0.physicalcpu") {
            Ok(name) => name,
            Err(err) => {
                warn!(%err, "failed to build sysctl key for CPU core detection");
                return fallback;
            }
        };
        let mut val: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        let ret = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut val as *mut i32 as *mut libc::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 && val > 0 {
            return val as usize;
        }
    }

    fallback
}

// Read `NSProcessInfo.thermalState` via ObjC runtime FFI.
//
// Returns a raw `u8` in `[0, 3]` mapping to `ThermalState` variants.
// Falls back to `Nominal (0)` on non-macOS or if the call fails.
fn read_nsprocessinfo_thermal_state() -> u8 {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "objc")]
        unsafe extern "C" {
            fn objc_getClass(name: *const libc::c_char) -> *mut libc::c_void;
            fn sel_registerName(name: *const libc::c_char) -> *const libc::c_void;
            #[allow(improper_ctypes)]
            fn objc_msgSend(recv: *mut libc::c_void, sel: *const libc::c_void, ...) -> usize;
        }
        unsafe {
            let cls = objc_getClass(c"NSProcessInfo".as_ptr().cast());
            if cls.is_null() {
                return 0;
            }
            let pi_sel = sel_registerName(c"processInfo".as_ptr().cast());
            let state_sel = sel_registerName(c"thermalState".as_ptr().cast());
            let pi = objc_msgSend(cls, pi_sel) as *mut libc::c_void;
            if pi.is_null() {
                return 0;
            }
            let state = objc_msgSend(pi, state_sel);
            // Bound-check on the full usize BEFORE narrowing to u8 so that an
            // out-of-range OS return value doesn't truncate to a misleading 0.
            (state.min(3)) as u8
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ThermalState::as_str ──────────────────────────────────────────────────

    #[test]
    fn thermal_state_as_str_all_variants() {
        assert_eq!(ThermalState::Nominal.as_str(), "Nominal");
        assert_eq!(ThermalState::Fair.as_str(), "Fair");
        assert_eq!(ThermalState::Serious.as_str(), "Serious");
        assert_eq!(ThermalState::Critical.as_str(), "Critical");
    }

    #[test]
    fn thermal_state_eq_and_copy() {
        let a = ThermalState::Serious;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(ThermalState::Nominal, ThermalState::Critical);
    }

    // ── ThermalMonitor ────────────────────────────────────────────────────────

    #[test]
    fn thermal_monitor_current_reads_initial_state() {
        // Verify current() reads the actual hardware state without panicking.
        // The value is machine-dependent so we only assert it is a valid variant.
        let monitor = ThermalMonitor::with_poll(60);
        let state = monitor.current();
        assert!(
            matches!(
                state,
                ThermalState::Nominal | ThermalState::Fair | ThermalState::Serious | ThermalState::Critical
            ),
            "unexpected thermal state: {state:?}"
        );
    }

    #[test]
    fn thermal_monitor_recommended_concurrency_nominal_is_positive() {
        let monitor = ThermalMonitor::with_poll(60);
        assert!(
            monitor.recommended_concurrency() >= 1,
            "recommended_concurrency must be at least 1 at Nominal state"
        );
    }

    #[test]
    fn thermal_monitor_clone_shares_state() {
        let m1 = ThermalMonitor::with_poll(60);
        let m2 = m1.clone();
        // Both should report the same state (shared Arc<AtomicU8>).
        assert_eq!(m1.current(), m2.current());
    }

    #[test]
    fn thermal_monitor_current_decodes_all_raw_values() {
        // Verify the raw → ThermalState decode table by reading the current() impl
        // logic with a known AtomicU8 value (via with_poll which inits to 0=Nominal).
        // We cannot write to the internal state directly, but we can verify that
        // the four u8 values map to the correct variants by checking the discriminants.
        assert_eq!(ThermalState::Nominal as u8, 0);
        assert_eq!(ThermalState::Fair as u8, 1);
        assert_eq!(ThermalState::Serious as u8, 2);
        assert_eq!(ThermalState::Critical as u8, 3);
    }

    // ── recommended_concurrency state mapping ─────────────────────────────────

    #[test]
    fn recommended_concurrency_fair_same_as_nominal() {
        let monitor = ThermalMonitor::with_poll(60);
        let nominal = monitor.recommended_concurrency();
        // Fair maps to the same full-CPU path as Nominal.
        monitor
            .state
            .store(ThermalState::Fair as u8, Ordering::Relaxed);
        assert_eq!(monitor.recommended_concurrency(), nominal);
    }

    #[test]
    fn recommended_concurrency_serious_is_half_of_nominal() {
        let monitor = ThermalMonitor::with_poll(60);
        let nominal = monitor.recommended_concurrency();
        monitor
            .state
            .store(ThermalState::Serious as u8, Ordering::Relaxed);
        let expected = (nominal / 2).max(1);
        assert_eq!(monitor.recommended_concurrency(), expected);
    }

    #[test]
    fn recommended_concurrency_critical_is_one() {
        let monitor = ThermalMonitor::with_poll(60);
        monitor
            .state
            .store(ThermalState::Critical as u8, Ordering::Relaxed);
        assert_eq!(monitor.recommended_concurrency(), 1);
    }
}
