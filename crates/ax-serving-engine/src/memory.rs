//! Memory budget check before model loading.
//!
//! Queries macOS for available physical memory via `host_statistics64`
//! and validates that the model can fit before attempting to load.

/// Check whether enough memory is available to load a model of the given size.
///
/// `model_bytes` is the raw file size of the GGUF. Actual resident size will
/// be similar (mmap) but GPU buffers add overhead; we gate on 1.1× to leave
/// headroom.
pub fn check_memory_budget(model_bytes: u64) -> anyhow::Result<()> {
    let available = available_bytes();
    let required = (model_bytes as f64 * 1.1) as u64;

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
