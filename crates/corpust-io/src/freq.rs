//! Precomputed corpus-wide frequency tables, persisted next to the
//! index as `frequencies.json`.
//!
//! `CorpusIndex::frequencies` does a full term-dictionary scan with one
//! postings read per term. That's fine at the current scale but grows
//! with the corpus and is recomputed every time the FrequencyView opens.
//! To keep large corpora snappy, the build step precomputes the per-layer
//! top-N (plus the field's grand total) once and writes it here; the
//! query path serves from this sidecar and falls back to the live scan
//! only when the file is absent (e.g. a corpus built before this landed).
//!
//! The envelope carries a `schema_version` mirroring the metadata
//! sidecar, so the format can be migrated rather than silently
//! misread.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// How many top terms the build step precomputes per layer. Chosen to
/// cover the FrequencyView's request clamp (the Tauri command caps
/// `limit` at 1000), so the sidecar always satisfies a UI request
/// without falling back to a live scan.
pub const PRECOMPUTE_LIMIT: usize = 1000;

/// One layer's precomputed table: the top terms (descending by count)
/// and the field's grand-total occurrence count — the denominator the
/// UI uses to turn counts into percentages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerFreq {
    pub rows: Vec<(String, u64)>,
    pub total: u64,
}

impl LayerFreq {
    /// Build from the `(rows, total)` tuple `CorpusIndex::frequencies`
    /// returns.
    pub fn from_table(table: (Vec<(String, u64)>, u64)) -> Self {
        let (rows, total) = table;
        Self { rows, total }
    }
}

/// Precomputed frequency tables for every query layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FreqTables {
    /// The `limit` the tables were computed with — i.e. each layer holds
    /// at most this many rows. A reader can only serve requests for up to
    /// this many terms; larger requests must fall back to a live scan.
    pub limit: usize,
    pub word: LayerFreq,
    pub lemma: LayerFreq,
    pub pos: LayerFreq,
}

/// Versioned wrapper around [`FreqTables`]. On-disk shape is
/// `{ "schemaVersion": 1, "tables": { ... } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FreqTablesEnvelope {
    pub schema_version: u32,
    pub tables: FreqTables,
}

impl FreqTablesEnvelope {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn wrap(tables: FreqTables) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            tables,
        }
    }
}

/// Serialize the envelope as pretty JSON and write to `path`. Called by
/// both the CLI `index` subcommand and the Tauri build command after the
/// index commits.
pub fn write_freq_file(path: &Path, tables: &FreqTables) -> Result<()> {
    let envelope = FreqTablesEnvelope::wrap(tables.clone());
    let json = serde_json::to_vec_pretty(&envelope)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Read and unwrap the frequency sidecar at `path`. Returns the inner
/// [`FreqTables`]; the caller decides whether a missing file or a
/// too-small `limit` warrants a live-scan fallback.
pub fn read_freq_file(path: &Path) -> Result<FreqTables> {
    let bytes = std::fs::read(path)?;
    let envelope: FreqTablesEnvelope = serde_json::from_slice(&bytes)?;
    Ok(envelope.tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frequencies.json");
        let tables = FreqTables {
            limit: PRECOMPUTE_LIMIT,
            word: LayerFreq {
                rows: vec![("the".into(), 42), ("and".into(), 17)],
                total: 100,
            },
            lemma: LayerFreq::default(),
            pos: LayerFreq {
                rows: vec![("NN".into(), 30)],
                total: 100,
            },
        };

        write_freq_file(&path, &tables).unwrap();
        let back = read_freq_file(&path).unwrap();

        assert_eq!(back.limit, PRECOMPUTE_LIMIT);
        assert_eq!(back.word.rows, vec![("the".into(), 42), ("and".into(), 17)]);
        assert_eq!(back.word.total, 100);
        assert!(back.lemma.rows.is_empty());
        assert_eq!(back.pos.rows, vec![("NN".into(), 30)]);

        // camelCase + version contract.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"schemaVersion\""));
    }

    #[test]
    fn read_missing_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.json");
        assert!(read_freq_file(&missing).is_err());
    }

    #[test]
    fn from_table_splits_tuple() {
        let lf = LayerFreq::from_table((vec![("x".into(), 5)], 5));
        assert_eq!(lf.rows, vec![("x".into(), 5)]);
        assert_eq!(lf.total, 5);
    }
}
