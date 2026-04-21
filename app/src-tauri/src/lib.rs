//! Tauri backend for the corpust desktop app.
//!
//! Thin command layer over `corpust-query` + `corpust-index`. The React
//! frontend calls these via `@tauri-apps/api::invoke`. Everything below
//! is intentionally stub-heavy right now — the visual shape of the UI
//! comes first; command bodies get fleshed out once the layout is
//! locked in.

use serde::{Deserialize, Serialize};

mod commands;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::list_corpora,
            commands::open_corpus,
            commands::run_kwic,
            commands::build_index,
        ])
        .run(tauri::generate_context!())
        .expect("error while running corpust");
}

// ---------------------------------------------------------------------------
// Shared DTOs (mirror TS types in app/src/types.ts)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusMeta {
    pub id: String,
    pub name: String,
    pub index_path: String,
    pub source_path: String,
    pub annotated: bool,
    pub doc_count: u64,
    pub token_count: u64,
    pub built_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tagger_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryLayer {
    Word,
    Lemma,
    Pos,
}

impl From<QueryLayer> for corpust_index::QueryLayer {
    fn from(l: QueryLayer) -> Self {
        match l {
            QueryLayer::Word => corpust_index::QueryLayer::Word,
            QueryLayer::Lemma => corpust_index::QueryLayer::Lemma,
            QueryLayer::Pos => corpust_index::QueryLayer::Pos,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KwicRequest {
    pub corpus_id: String,
    pub term: String,
    pub layer: QueryLayer,
    pub context: usize,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KwicHit {
    pub doc_id: u64,
    pub path: String,
    pub left: String,
    pub hit: String,
    pub right: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KwicResult {
    pub hits: Vec<KwicHit>,
    pub elapsed_ms: f64,
    pub truncated: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildRequest {
    pub source_path: String,
    pub out_path: String,
    pub annotate: bool,
}
