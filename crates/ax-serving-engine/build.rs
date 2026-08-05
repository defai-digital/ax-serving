// Build script for ax-serving-engine.
//
// When the `libllama` feature is enabled, this script:
//   1. Locates the system `llama.h` (homebrew, conda, or `$AXS_LLAMA_INCLUDE_DIR`).
//   2. Generates Rust FFI bindings via `bindgen`.
//   3. Emits `cargo:rustc-link-lib=llama` + search paths so Rust can link against
//      the system `libllama.dylib` at build time.
//
// Without the feature the script is a no-op; `libllama.rs` is excluded from
// compilation via `#[cfg(feature = "libllama")]` in `lib.rs`.
//
// Quick setup:
//   brew install llama.cpp
//   cargo build --features ax-serving-engine/libllama
//
// Override search paths via env vars:
//   `AXS_LLAMA_INCLUDE_DIR` — directory containing `llama.h`
//   `AXS_LLAMA_LIB_DIR`     — directory containing `libllama.dylib`

fn main() {
    println!("cargo:rustc-check-cfg=cfg(llama_sampler_penalties_has_n_vocab)");
    println!("cargo:rustc-check-cfg=cfg(llama_sampler_penalties_has_last_n)");

    // Only run when the `libllama` feature is explicitly enabled.
    if std::env::var("CARGO_FEATURE_LIBLLAMA").is_err() {
        return;
    }

    // ── Locate llama.h ────────────────────────────────────────────────────────

    let search_include = [
        std::env::var("AXS_LLAMA_INCLUDE_DIR").ok(),
        Some("/opt/homebrew/include".to_string()),
        Some("/usr/local/include".to_string()),
        Some("/opt/local/include".to_string()),
        Some("/usr/include".to_string()),
    ];

    let header = search_include
        .iter()
        .filter_map(|o| o.as_deref())
        .map(|dir| std::path::Path::new(dir).join("llama.h"))
        .find(|h| h.exists())
        .unwrap_or_else(|| {
            panic!(
                "\n\
                 ax-serving-engine: libllama feature enabled but llama.h not found.\n\
                 Install llama.cpp (e.g. `brew install llama.cpp`) or set\n\
                 AXS_LLAMA_INCLUDE_DIR=/path/to/include.\n"
            )
        });

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=AXS_LLAMA_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=AXS_LLAMA_LIB_DIR");

    // ── Emit link directives ──────────────────────────────────────────────────

    // Add common library search paths (homebrew on Apple Silicon, then x86).
    println!("cargo:rustc-link-search=/opt/homebrew/lib");
    println!("cargo:rustc-link-search=/usr/local/lib");
    if let Ok(dir) = std::env::var("AXS_LLAMA_LIB_DIR") {
        println!("cargo:rustc-link-search={dir}");
    }
    // Dynamic link against libllama.
    println!("cargo:rustc-link-lib=dylib=llama");

    // ── Generate FFI bindings via bindgen ─────────────────────────────────────

    let bindings = bindgen::Builder::default()
        .header(header.to_str().expect("non-UTF-8 header path"))
        // Only generate items prefixed with `llama_` or `ggml_type`
        // (ggml_type is needed for KV-cache quantization params).
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*|ggml_type")
        .allowlist_var("LLAMA_.*|GGML_.*")
        // Treat llama_model, llama_context, llama_sampler as opaque pointers
        // so we never accidentally dereference them from Rust.
        .opaque_type("llama_model")
        .opaque_type("llama_context")
        .opaque_type("llama_sampler")
        .opaque_type("llama_vocab")
        // Suppress warnings for non-standard C patterns in the header.
        .clang_arg("-Wno-language-extension-token")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate libllama bindings");

    let out_dir =
        std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo"));
    let bindings_rs = bindings
        .to_string()
        // Rust 2024 requires `unsafe extern` blocks.
        .replace("extern \"C\" {", "unsafe extern \"C\" {");

    // llama.cpp has shipped three penalty-sampler signatures:
    //   (penalty_last_n, repeat, frequency, presence)
    //   (n_vocab, penalty_last_n, repeat, frequency, presence)
    //   (n_vocab, repeat, frequency, presence)
    // Detect the exact generated binding instead of coupling AX Serving to a
    // particular Homebrew bottle or silently swapping two integer arguments.
    let (_, penalties_tail) = bindings_rs
        .split_once("pub fn llama_sampler_init_penalties(")
        .expect("llama_sampler_init_penalties missing from generated bindings");
    let (penalties_signature, _) = penalties_tail
        .split_once(") ->")
        .expect("could not parse llama_sampler_init_penalties binding");
    let has_n_vocab = penalties_signature.contains("n_vocab:");
    let has_last_n = penalties_signature.contains("penalty_last_n:");
    assert!(
        has_n_vocab || has_last_n,
        "unsupported llama_sampler_init_penalties binding"
    );
    if has_n_vocab {
        println!("cargo:rustc-cfg=llama_sampler_penalties_has_n_vocab");
    }
    if has_last_n {
        println!("cargo:rustc-cfg=llama_sampler_penalties_has_last_n");
    }

    std::fs::write(out_dir.join("libllama_bindings.rs"), bindings_rs)
        .expect("failed to write libllama_bindings.rs");
}
