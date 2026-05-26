use std::path::Path;

use ax_serving_engine::LoadConfig;

pub fn for_model_path(path: &Path) -> LoadConfig {
    LoadConfig {
        backend_hint: path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            .then(|| "llama_cpp".to_string()),
        ..LoadConfig::default()
    }
}
