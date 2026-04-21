//! Positional inverted index, backed by Tantivy.
//!
//! Phase 0 stores documents with a single `body` text field indexed with
//! positions, plus stored copies of the doc id / path / body for retrieval.
//! KWIC extraction re-tokenizes the stored body — cheap at small scale, will
//! be replaced with direct positional reads once we outgrow it.

use anyhow::{Context, Result};
use corpust_core::{DocId, Document};
use std::path::{Path, PathBuf};
use tantivy::{
    Index, IndexReader, ReloadPolicy, TantivyDocument,
    collector::TopDocs,
    doc,
    query::QueryParser,
    schema::{Field, IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions, Value},
};
use unicode_segmentation::UnicodeSegmentation;

/// Default KWIC context size, in tokens per side.
pub const DEFAULT_CONTEXT: usize = 7;

/// Default cap on returned KWIC hits.
pub const DEFAULT_LIMIT: usize = 50;

pub struct CorpusIndex {
    index: Index,
    reader: IndexReader,
    fields: Fields,
}

#[derive(Clone, Copy)]
struct Fields {
    doc_id: Field,
    path: Field,
    body: Field,
}

/// One concordance line.
#[derive(Debug, Clone)]
pub struct KwicHit {
    pub doc_id: DocId,
    pub path: PathBuf,
    pub left: String,
    pub hit: String,
    pub right: String,
}

impl CorpusIndex {
    /// Create a new index on disk, overwriting any index already present.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("clearing {}", path.display()))?;
        }
        std::fs::create_dir_all(path)
            .with_context(|| format!("creating {}", path.display()))?;

        let (schema, fields) = build_schema();
        let index = Index::create_in_dir(path, schema)?;
        Self::from_index(index, fields)
    }

    /// Open an existing index on disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let index = Index::open_in_dir(path)?;
        let schema = index.schema();
        let fields = Fields {
            doc_id: schema.get_field("doc_id")?,
            path: schema.get_field("path")?,
            body: schema.get_field("body")?,
        };
        Self::from_index(index, fields)
    }

    fn from_index(index: Index, fields: Fields) -> Result<Self> {
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        Ok(Self {
            index,
            reader,
            fields,
        })
    }

    /// Index a batch of documents. Commits once at the end.
    pub fn add_documents(&self, documents: impl IntoIterator<Item = Document>) -> Result<()> {
        let mut writer = self.index.writer(50_000_000)?;
        for document in documents {
            writer.add_document(doc!(
                self.fields.doc_id => document.id,
                self.fields.path => document.path.display().to_string(),
                self.fields.body => document.text,
            ))?;
        }
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Run a KWIC (key word in context) query for a single term.
    ///
    /// `context` is the number of surrounding tokens to include on each side.
    pub fn kwic(&self, term: &str, context: usize, limit: usize) -> Result<Vec<KwicHit>> {
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.fields.body]);
        let query = parser.parse_query(term)?;
        // Pull a generous set of candidate docs; we still cap total hits below.
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit.max(1) * 4))?;

        let needle = term.to_lowercase();
        let mut hits = Vec::with_capacity(limit);

        for (_score, addr) in top_docs {
            if hits.len() >= limit {
                break;
            }
            let retrieved: TantivyDocument = searcher.doc(addr)?;
            let body = retrieved
                .get_first(self.fields.body)
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let path = retrieved
                .get_first(self.fields.path)
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let doc_id = retrieved
                .get_first(self.fields.doc_id)
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            let tokens: Vec<&str> = body.unicode_words().collect();
            for (i, word) in tokens.iter().enumerate() {
                if word.to_lowercase() == needle {
                    let left_start = i.saturating_sub(context);
                    let right_end = (i + 1 + context).min(tokens.len());
                    hits.push(KwicHit {
                        doc_id,
                        path: PathBuf::from(path),
                        left: tokens[left_start..i].join(" "),
                        hit: word.to_string(),
                        right: tokens[i + 1..right_end].join(" "),
                    });
                    if hits.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(hits)
    }
}

fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let doc_id = builder.add_u64_field("doc_id", STORED);
    let path = builder.add_text_field("path", STORED);

    let body_indexing = TextFieldIndexing::default()
        .set_tokenizer("default")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let body_options = TextOptions::default()
        .set_indexing_options(body_indexing)
        .set_stored();
    let body = builder.add_text_field("body", body_options);

    (
        builder.build(),
        Fields {
            doc_id,
            path,
            body,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_kwic() {
        let tmp = std::env::temp_dir().join(format!("corpust-idx-{}", rand_suffix()));
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents([Document {
            id: 0,
            path: PathBuf::from("a.txt"),
            text: "the quick brown fox jumps over the lazy dog".to_string(),
        }])
        .unwrap();

        let hits = idx.kwic("the", 2, 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].hit.to_lowercase(), "the");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}
