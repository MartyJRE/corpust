//! Tauri backend for the corpust desktop app.
//!
//! Thin command layer over `corpust-query` + `corpust-index`. The React
//! frontend calls these via `@tauri-apps/api::invoke`. Everything below
//! is intentionally stub-heavy right now — the visual shape of the UI
//! comes first; command bodies get fleshed out once the layout is
//! locked in.

use corpust_index::CorpusIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

mod commands;

/// Process-local registry of opened corpora, keyed by the
/// `corpusId` string we hand back to the frontend.
pub struct AppState {
    pub corpora: Mutex<HashMap<String, OpenedCorpus>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            corpora: Mutex::new(HashMap::new()),
        }
    }
}

/// One corpus loaded into the current process — the Tantivy handle
/// plus the metadata we've serialized to the frontend.
pub struct OpenedCorpus {
    pub index: CorpusIndex,
    pub meta: CorpusMeta,
}

pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::list_corpora,
            commands::open_corpus,
            commands::run_kwic,
            commands::run_collocates,
            commands::build_index,
        ])
        .run(tauri::generate_context!())
        .expect("error while running corpust");
}

// ---------------------------------------------------------------------------
// Shared DTOs (mirror TS types in app/src/types.ts)
// ---------------------------------------------------------------------------

// Re-export the persisted-corpus DTOs from `corpust-io` so the CLI
// and the Tauri side share one definition.
pub use corpust_io::metadata::{CorpusMeta, CorpusMetaEnvelope};

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
pub struct CollocatesRequest {
    pub corpus_id: String,
    pub term: String,
    pub layer: QueryLayer,
    /// Number of tokens to consider on the left of the node.
    /// 0 = ignore the left context entirely.
    pub left_window: usize,
    /// Number of tokens to consider on the right of the node.
    pub right_window: usize,
    /// Max number of collocate candidates to return.
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Collocate {
    pub word: String,
    pub pos: String,
    pub left_count: u32,
    pub right_count: u32,
    pub total: u32,
    pub log_dice: f64,
    pub mi: f64,
    pub z: f64,
    pub dist: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollocatesResult {
    pub collocates: Vec<Collocate>,
    pub elapsed_ms: f64,
    pub node_hits: u32,
    pub window_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildRequest {
    pub source_path: String,
    pub annotate: bool,
    /// Display name for the resulting corpus. Optional — we fall
    /// back to the source directory's basename.
    #[serde(default)]
    pub name: Option<String>,
    /// Which annotator implementation to use when `annotate=true`.
    /// Defaults to the pure-Rust in-process tagger; the subprocess
    /// path is kept as an option so users can A/B correctness and
    /// speed.
    #[serde(default)]
    pub tagger: TaggerKind,
}

// `CorpusMetaEnvelope` is re-exported from `corpust_io::metadata`
// above. No local definition.

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TaggerKind {
    /// Pure-Rust in-process TreeTagger port. Fast (~2.5× end-to-end
    /// speedup over subprocess) but currently ~92% POS accuracy.
    #[default]
    Rust,
    /// Bundled `tree-tagger` binary; one subprocess per document.
    /// Accurate (LancsBox parity) but slow.
    Subprocess,
}
