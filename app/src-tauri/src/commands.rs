//! Tauri command handlers. Frontend talks to these via `invoke`.
//!
//! Today every handler is a thin stub — they exist so the IPC surface
//! is visible (command names, argument shapes, return types). Real
//! implementations land once the visual layer settles.

use crate::{BuildRequest, CorpusMeta, KwicRequest, KwicResult};

#[tauri::command]
pub fn list_corpora() -> Vec<CorpusMeta> {
    // TODO: read from `<data_dir>/corpust/corpora/` registry
    Vec::new()
}

#[tauri::command]
pub fn open_corpus(index_path: String) -> Result<CorpusMeta, String> {
    // TODO: corpust_index::CorpusIndex::open + derive CorpusMeta
    let _ = index_path;
    Err("not yet implemented".to_string())
}

#[tauri::command]
pub fn run_kwic(req: KwicRequest) -> Result<KwicResult, String> {
    // TODO: resolve corpus handle, call corpust_query::kwic, time it,
    // shape hits into KwicHit records.
    let _ = req;
    Err("not yet implemented".to_string())
}

#[tauri::command]
pub fn build_index(req: BuildRequest) -> Result<String, String> {
    // TODO: kick off an async build task, stream progress via the
    // `build:progress` event, return a task id.
    let _ = req;
    Err("not yet implemented".to_string())
}
