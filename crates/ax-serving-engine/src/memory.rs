//! Memory budget check before model loading.
//!
//! Queries macOS for available physical memory via `host_statistics64`
//! and validates that the model can fit before attempting to load.

const GIB: u64 = 1024 * 1024 * 1024;

fn required_memory_bytes(model_bytes: u64) -> u64 {
    let (num, denom) = if model_bytes >= 16 * GIB {
        (14u64, 10u64)
    } else if model_bytes >= 4 * GIB {
        (13u64, 10u64)
    } else {
        (12u64, 10u64)
    };
    model_bytes.saturating_mul(num) / denom
}

/// Check whether enough memory is available to load a model of the given size.
///
/// `model_bytes` is the raw file size of the GGUF. Actual resident size will
/// be similar (mmap) but GPU buffers add overhead, especially on larger Metal
/// models, so the headroom scales with model size.
pub fn check_memory_budget(model_bytes: u64) -> anyhow::Result<()> {
    let available = available_bytes();
    let required = required_memory_bytes(model_bytes);

    if available < required {
        anyhow::bail!(
            "insufficient memory: model needs ~{:.1}GB, only {:.1}GB available",
            required as f64 / 1e9,
            available as f64 / 1e9,
        );
    }
    Ok(())
}

/// Query available physical memory from the macOS kernel.
fn available_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;

        // host_statistics64 with HOST_VM_INFO64 flavor.
        // Returns vm_statistics64_data_t with free_count and inactive_count.
        // Each page is typically 16KB on Apple Silicon.
        unsafe extern "C" {
            fn mach_port_deallocate(task: u32, name: u32) -> i32;
        }
        #[allow(deprecated)]
        let host = unsafe { libc::mach_host_self() };
        let mut stats = MaybeUninit::<libc::vm_statistics64_data_t>::uninit();
        let mut count = (std::mem::size_of::<libc::vm_statistics64_data_t>()
            / std::mem::size_of::<libc::integer_t>()) as u32;

        let ret = unsafe {
            libc::host_statistics64(
                host,
                libc::HOST_VM_INFO64,
                stats.as_mut_ptr() as libc::host_info64_t,
                &mut count,
            )
        };

        // Deallocate the Mach port to avoid leaking a send-right per call.
        #[allow(deprecated)]
        let task = unsafe { libc::mach_task_self() };
        let _ = unsafe { mach_port_deallocate(task, host) };

        if ret == libc::KERN_SUCCESS {
            let stats = unsafe { stats.assume_init() };
            let raw_page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) };
            let page_size = if raw_page_size > 0 {
                raw_page_size as u64
            } else {
                4096
            };
            let free = stats.free_count as u64 + stats.inactive_count as u64;
            return free * page_size;
        }
    }

    // Fallback: assume 8GB available (conservative).
    8 * 1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::{GIB, required_memory_bytes};

    #[test]
    fn required_memory_budget_scales_with_model_size() {
        assert_eq!(required_memory_bytes(2 * GIB), 2 * GIB * 12 / 10);
        assert_eq!(required_memory_bytes(8 * GIB), 8 * GIB * 13 / 10);
        assert_eq!(required_memory_bytes(32 * GIB), 32 * GIB * 14 / 10);
    }
}
