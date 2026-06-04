//! Tauri backend for the corpust desktop app.
//!
//! Thin command layer over `corpust-query` + `corpust-index`. The React
//! frontend calls these via `@tauri-apps/api::invoke`; the command
//! bodies live in `commands.rs` and drive the real index / query crates.

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

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
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
            commands::list_documents,
            commands::run_frequencies,
            commands::run_term_distribution,
            commands::run_collocate_distance,
            commands::expand_context,
            commands::write_text_file,
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

/// Document-metadata filter shared by every query command. Empty
/// dimensions (the default) impose no constraint. Mirrors the TS
/// `DocFilter` and converts into [`corpust_index::DocFilter`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocFilterDto {
    #[serde(default)]
    pub year_min: Option<u32>,
    #[serde(default)]
    pub year_max: Option<u32>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

impl From<DocFilterDto> for corpust_index::DocFilter {
    fn from(f: DocFilterDto) -> Self {
        corpust_index::DocFilter {
            year_min: f.year_min,
            year_max: f.year_max,
            author: f.author,
            path: f.path,
        }
    }
}

impl DocFilterDto {
    /// Convert an optional DTO into a resolved filter (empty when absent).
    pub fn resolve(opt: Option<DocFilterDto>) -> corpust_index::DocFilter {
        opt.map(Into::into).unwrap_or_default()
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
    /// Hits to skip before this page (concordance pagination).
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub filter: Option<DocFilterDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KwicHit {
    pub doc_id: u64,
    pub path: String,
    /// Token position of the hit in its document; the frontend feeds it
    /// back to `expand_context` to widen the concordance window.
    pub hit_position: usize,
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
    /// Total matches across the whole corpus (paging denominator).
    pub total: u64,
    /// Index of the first hit in this page (0-based), for "N–M of total".
    pub offset: u64,
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
    #[serde(default)]
    pub filter: Option<DocFilterDto>,
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
    /// Node-term occurrences the scan covered. Equals the node's true
    /// corpus frequency unless `truncated` is set.
    pub node_hits: u32,
    pub window_tokens: u32,
    /// True when the safety ceiling cut the full-corpus scan short, so
    /// the scores reflect a (large) sample rather than every occurrence.
    pub truncated: bool,
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
    /// speedup over subprocess) at ~99% POS accuracy (98.8–99.8% across
    /// the Gutenberg / UNSC samples).
    #[default]
    Rust,
    /// Bundled `tree-tagger` binary; one subprocess per document.
    /// Accurate (LancsBox parity) but slow.
    Subprocess,
}

// ---- Document list (CorpusDetail) ----

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentInfo {
    pub doc_id: u64,
    pub path: String,
    pub token_count: usize,
    /// Title / author / year extracted from the document body at index
    /// time. `None` when the extractor couldn't confidently find them;
    /// serialises to `null` so the frontend renders a muted fallback.
    pub title: Option<String>,
    pub author: Option<String>,
    pub year: Option<u32>,
}

// ---- Frequency table (FrequencyView word/POS tables) ----

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrequenciesRequest {
    pub corpus_id: String,
    pub layer: QueryLayer,
    pub limit: usize,
    #[serde(default)]
    pub filter: Option<DocFilterDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FreqRow {
    pub term: String,
    pub count: u64,
    pub pct: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FrequenciesResult {
    pub rows: Vec<FreqRow>,
    pub total_tokens: u64,
    pub elapsed_ms: f64,
}

// ---- Term distribution (FrequencyView per-doc table + dispersion) ----

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TermDistRequest {
    pub corpus_id: String,
    pub term: String,
    pub layer: QueryLayer,
    pub buckets: usize,
    #[serde(default)]
    pub filter: Option<DocFilterDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DocTermCount {
    pub doc_id: u64,
    pub path: String,
    pub hits: u64,
    pub token_count: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TermDistResult {
    pub doc_counts: Vec<DocTermCount>,
    pub dispersion: Vec<u32>,
    pub total_hits: u64,
    pub elapsed_ms: f64,
}

// ---- Collocation by distance (positional profile) ----

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollocateDistanceRequest {
    pub corpus_id: String,
    pub term: String,
    pub layer: QueryLayer,
    pub left_window: usize,
    pub right_window: usize,
    /// How many of the busiest collocates to return rows for.
    pub limit: usize,
    #[serde(default)]
    pub filter: Option<DocFilterDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DistanceRow {
    pub word: String,
    pub total: u32,
    /// One count per offset in `offsets` (same length/order).
    pub counts: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollocateDistanceResult {
    /// Signed offsets the columns represent, e.g. `[-3,-2,-1,1,2,3]`.
    pub offsets: Vec<i32>,
    pub rows: Vec<DistanceRow>,
    pub node_hits: u32,
    pub truncated: bool,
    pub elapsed_ms: f64,
}

// ---- Context expansion (ContextDrawer) ----

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpandRequest {
    pub corpus_id: String,
    pub doc_id: u64,
    pub position: usize,
    pub context: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpandedContext {
    pub doc_id: u64,
    pub path: String,
    pub before: String,
    pub hit: String,
    pub after: String,
    pub token_count: usize,
}
