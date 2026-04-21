//! Positional inverted index, backed by Tantivy.
//!
//! A custom Tokenizer (Unicode-word-segmentation + lowercasing) drives both
//! indexing and — implicitly — position numbering, so that display-side
//! retokenization (also via `unicode-segmentation`) aligns with the token
//! positions Tantivy stores in its posting lists. That alignment lets
//! [`CorpusIndex::kwic`] skip the term directly to its hits via positional
//! reads rather than scanning every token in every matched document.

use anyhow::{Context, Result};
use corpust_core::{DocId, Document};
use std::path::{Path, PathBuf};
use tantivy::{
    DocAddress, DocSet, Index, IndexReader, ReloadPolicy, TERMINATED, TantivyDocument, Term, doc,
    postings::Postings,
    schema::{Field, IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions, Value},
    tokenizer::{LowerCaser, TextAnalyzer, Token, TokenStream, Tokenizer},
};
use unicode_segmentation::UnicodeSegmentation;

/// Default KWIC context size, in tokens per side.
pub const DEFAULT_CONTEXT: usize = 7;

/// Default cap on returned KWIC hits.
pub const DEFAULT_LIMIT: usize = 50;

const TOKENIZER_NAME: &str = "corpust";

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
        register_tokenizer(&index);
        Self::from_index(index, fields)
    }

    /// Open an existing index on disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let index = Index::open_in_dir(path)?;
        register_tokenizer(&index);
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
    /// Case-insensitive. `context` is the number of surrounding tokens to
    /// include on each side.
    pub fn kwic(&self, term: &str, context: usize, limit: usize) -> Result<Vec<KwicHit>> {
        let searcher = self.reader.searcher();
        let lowered = term.to_lowercase();
        let term_obj = Term::from_field_text(self.fields.body, &lowered);

        let mut hits = Vec::with_capacity(limit);
        let mut positions_buf: Vec<u32> = Vec::new();

        'segments: for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            if hits.len() >= limit {
                break;
            }
            let inv_idx = seg_reader.inverted_index(self.fields.body)?;
            let Some(mut postings) = inv_idx
                .read_postings(&term_obj, IndexRecordOption::WithFreqsAndPositions)?
            else {
                continue;
            };

            loop {
                let doc = postings.doc();
                if doc == TERMINATED {
                    continue 'segments;
                }
                if hits.len() >= limit {
                    break 'segments;
                }

                let doc_addr = DocAddress::new(seg_ord as u32, doc);
                let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
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

                positions_buf.clear();
                postings.positions(&mut positions_buf);

                // Display-side tokenization: must align with the indexer.
                let tokens: Vec<&str> = body.unicode_words().collect();

                for &pos in &positions_buf {
                    if hits.len() >= limit {
                        break;
                    }
                    let i = pos as usize;
                    if i >= tokens.len() {
                        continue;
                    }
                    let left_start = i.saturating_sub(context);
                    let right_end = (i + 1 + context).min(tokens.len());
                    hits.push(KwicHit {
                        doc_id,
                        path: PathBuf::from(path),
                        left: tokens[left_start..i].join(" "),
                        hit: tokens[i].to_string(),
                        right: tokens[i + 1..right_end].join(" "),
                    });
                }

                postings.advance();
            }
        }

        Ok(hits)
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct UnicodeWordTokenizer;

struct UnicodeWordStream<'a> {
    iter: std::vec::IntoIter<(usize, &'a str)>,
    token: Token,
}

impl Tokenizer for UnicodeWordTokenizer {
    type TokenStream<'a> = UnicodeWordStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let words: Vec<(usize, &str)> = text.unicode_word_indices().collect();
        UnicodeWordStream {
            iter: words.into_iter(),
            token: Token {
                position: usize::MAX,
                ..Token::default()
            },
        }
    }
}

impl<'a> TokenStream for UnicodeWordStream<'a> {
    fn advance(&mut self) -> bool {
        match self.iter.next() {
            Some((byte_start, word)) => {
                self.token.position = self.token.position.wrapping_add(1);
                self.token.offset_from = byte_start;
                self.token.offset_to = byte_start + word.len();
                self.token.text.clear();
                self.token.text.push_str(word);
                true
            }
            None => false,
        }
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

fn register_tokenizer(index: &Index) {
    let analyzer = TextAnalyzer::builder(UnicodeWordTokenizer)
        .filter(LowerCaser)
        .build();
    index.tokenizers().register(TOKENIZER_NAME, analyzer);
}

fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let doc_id = builder.add_u64_field("doc_id", STORED);
    let path = builder.add_text_field("path", STORED);

    let body_indexing = TextFieldIndexing::default()
        .set_tokenizer(TOKENIZER_NAME)
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
        assert_eq!(hits[0].hit, "the");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn kwic_preserves_case_on_display() {
        let tmp = std::env::temp_dir().join(format!("corpust-idx-{}", rand_suffix()));
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents([Document {
            id: 0,
            path: PathBuf::from("a.txt"),
            text: "The quick brown fox jumps over THE lazy dog".to_string(),
        }])
        .unwrap();

        let hits = idx.kwic("the", 1, 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|h| h.hit == "The"));
        assert!(hits.iter().any(|h| h.hit == "THE"));
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
