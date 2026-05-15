//! Versioned on-disk sidecar describing a built corpus.
//!
//! Each corpus directory under `<data_root>/corpora/<slug>/` has a
//! `metadata.json` next to the tantivy index. The Tauri UI's
//! `list_corpora` command reads this file to populate the corpus list,
//! and `corpust index` writes it after a successful build.
//!
//! The envelope carries a `schema_version` so future field renames /
//! removals can be migrated rather than silently breaking older
//! indexes.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Display + provenance info for one built corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusMeta {
    pub id: String,
    pub name: String,
    /// Coarse classification — "literary", "legal", "news", "mixed".
    /// We don't classify yet, so always "mixed".
    pub kind: String,
    pub index_path: String,
    pub source_path: String,
    pub annotated: bool,
    pub doc_count: u64,
    pub token_count: u64,
    /// Unique-type count — `0` until a counting pass lands.
    pub types: u64,
    pub avg_doc_len: u64,
    pub built_at: String,
    pub build_ms: u64,
    pub languages: Vec<String>,
    pub tokeniser: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotator: Option<String>,
    pub size_on_disk: u64,
    /// Backend-only identifier for the tagger used at build time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tagger_id: Option<String>,
}

impl CorpusMeta {
    /// Empty-but-valid metadata with defaults. Builders fill the
    /// stats fields after the index commits.
    pub fn stub(id: String, name: String, index_path: String) -> Self {
        Self {
            id,
            name,
            kind: "mixed".to_owned(),
            index_path: index_path.clone(),
            source_path: index_path,
            annotated: false,
            doc_count: 0,
            token_count: 0,
            types: 0,
            avg_doc_len: 0,
            built_at: String::new(),
            build_ms: 0,
            languages: vec!["en".to_owned()],
            tokeniser: "corpust".to_owned(),
            annotator: None,
            size_on_disk: 0,
            tagger_id: None,
        }
    }
}

/// Versioned wrapper around [`CorpusMeta`]. The on-disk JSON has the
/// shape `{ "schemaVersion": 1, "corpus": { ... } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusMetaEnvelope {
    pub schema_version: u32,
    pub corpus: CorpusMeta,
}

impl CorpusMetaEnvelope {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn wrap(corpus: CorpusMeta) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            corpus,
        }
    }
}

/// Serialize the envelope as pretty JSON and write to `path`. Used by
/// both the Tauri build command and the CLI's `index` subcommand so
/// CLI-built corpora show up in the UI's `list_corpora` output.
pub fn write_metadata_file(path: &Path, meta: &CorpusMeta) -> Result<()> {
    let envelope = CorpusMetaEnvelope::wrap(meta.clone());
    let json = serde_json::to_vec_pretty(&envelope)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Recursive on-disk byte total for a directory. Returns `None` on any
/// I/O error — callers should treat this as a display-only stat that
/// shouldn't fail the build.
pub fn dir_size(path: &Path) -> Option<u64> {
    fn walk(path: &Path, total: &mut u64) -> std::io::Result<()> {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let md = entry.metadata()?;
            if md.is_dir() {
                walk(&entry.path(), total)?;
            } else {
                *total += md.len();
            }
        }
        Ok(())
    }
    let mut total = 0u64;
    walk(path, &mut total).ok()?;
    Some(total)
}

/// `built_at` timestamp string. We don't depend on a heavyweight time
/// crate; the UI only displays this verbatim. Format is
/// `unix:<seconds>` so it sorts lexicographically.
pub fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}
