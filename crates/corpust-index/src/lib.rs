//! Positional inverted index, backed by Tantivy.
//!
//! Three aligned text fields per document:
//!
//! - `body`        — word forms (always populated)
//! - `body_lemma`  — lemmas (populated only when an [`Annotator`] is used)
//! - `body_pos`    — POS tags (populated only when an [`Annotator`] is used)
//!
//! All three share the same token positions: when an annotator is present,
//! it drives tokenization across every layer via Tantivy's
//! `PreTokenizedString`, so a position `p` refers to the same token in
//! every field. When no annotator is passed to [`CorpusIndex::add_documents`],
//! we fall back to the registered "corpust" tokenizer for `body` and leave
//! `body_lemma` / `body_pos` empty for that document.
//!
//! A stored `token_offsets` sidecar carries per-token byte offsets so KWIC
//! context extraction is O(context) regardless of document length.

use anyhow::{Context, Result};
use corpust_annotate::Annotator;
use corpust_core::{DocId, Document};
use std::path::{Path, PathBuf};
use tantivy::{
    DocAddress, DocSet, Index, IndexReader, ReloadPolicy, TERMINATED, TantivyDocument, Term, doc,
    postings::Postings,
    schema::{
        BytesOptions, Field, IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions,
        Value,
    },
    tokenizer::{LowerCaser, PreTokenizedString, TextAnalyzer, Token, TokenStream, Tokenizer},
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
    body_lemma: Field,
    body_pos: Field,
    token_offsets: Field,
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
            body_lemma: schema.get_field("body_lemma")?,
            body_pos: schema.get_field("body_pos")?,
            token_offsets: schema.get_field("token_offsets")?,
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
    ///
    /// If `annotator` is `Some`, the annotator's tokenization drives every
    /// layer (body / body_lemma / body_pos) and its byte offsets populate
    /// `token_offsets`. If `None`, the registered "corpust" tokenizer handles
    /// `body`, lemma / pos fields are left empty for each document, and
    /// `token_offsets` is derived from `unicode_word_indices`.
    pub fn add_documents(
        &self,
        documents: impl IntoIterator<Item = Document>,
        annotator: Option<&dyn Annotator>,
    ) -> Result<()> {
        let mut writer = self.index.writer(50_000_000)?;
        for document in documents {
            match annotator {
                Some(a) => self.add_annotated(&mut writer, &document, a)?,
                None => self.add_unannotated(&mut writer, &document)?,
            }
        }
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    fn add_unannotated(
        &self,
        writer: &mut tantivy::IndexWriter,
        document: &Document,
    ) -> Result<()> {
        let offsets: Vec<u32> = document
            .text
            .unicode_word_indices()
            .map(|(start, _)| start as u32)
            .collect();
        let offsets_bytes = offsets_to_bytes(&offsets);

        writer.add_document(doc!(
            self.fields.doc_id => document.id,
            self.fields.path => document.path.display().to_string(),
            self.fields.body => document.text.clone(),
            self.fields.token_offsets => offsets_bytes,
        ))?;
        Ok(())
    }

    fn add_annotated(
        &self,
        writer: &mut tantivy::IndexWriter,
        document: &Document,
        annotator: &dyn Annotator,
    ) -> Result<()> {
        let annotated = annotator.annotate(&document.text)?;

        let mut body_tokens = Vec::with_capacity(annotated.len());
        let mut lemma_tokens = Vec::with_capacity(annotated.len());
        let mut pos_tokens = Vec::with_capacity(annotated.len());
        let mut offsets: Vec<u32> = Vec::with_capacity(annotated.len());

        for t in &annotated {
            offsets.push(t.byte_start as u32);
            body_tokens.push(Token {
                offset_from: t.byte_start,
                offset_to: t.byte_end,
                position: t.position as usize,
                text: t.word.to_lowercase(),
                position_length: 1,
            });
            lemma_tokens.push(Token {
                offset_from: t.byte_start,
                offset_to: t.byte_end,
                position: t.position as usize,
                text: t
                    .lemma
                    .as_deref()
                    .map(str::to_lowercase)
                    .unwrap_or_default(),
                position_length: 1,
            });
            pos_tokens.push(Token {
                offset_from: t.byte_start,
                offset_to: t.byte_end,
                position: t.position as usize,
                // POS tags keep original case — conventionally uppercase.
                text: t.pos.as_deref().unwrap_or("").to_string(),
                position_length: 1,
            });
        }

        let body_pre = PreTokenizedString {
            text: document.text.clone(),
            tokens: body_tokens,
        };
        let lemma_pre = PreTokenizedString {
            text: String::new(),
            tokens: lemma_tokens,
        };
        let pos_pre = PreTokenizedString {
            text: String::new(),
            tokens: pos_tokens,
        };
        let offsets_bytes = offsets_to_bytes(&offsets);

        writer.add_document(doc!(
            self.fields.doc_id => document.id,
            self.fields.path => document.path.display().to_string(),
            self.fields.body => body_pre,
            self.fields.body_lemma => lemma_pre,
            self.fields.body_pos => pos_pre,
            self.fields.token_offsets => offsets_bytes,
        ))?;
        Ok(())
    }

    /// Run a KWIC (key word in context) query for a single term on the
    /// `body` (word form) layer. Case-insensitive.
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
                let offsets_bytes = retrieved
                    .get_first(self.fields.token_offsets)
                    .and_then(|v| v.as_bytes())
                    .unwrap_or_default();
                let offsets = bytes_to_offsets(offsets_bytes);

                positions_buf.clear();
                postings.positions(&mut positions_buf);

                for &pos in &positions_buf {
                    if hits.len() >= limit {
                        break;
                    }
                    let p = pos as usize;
                    if p >= offsets.len() {
                        continue;
                    }

                    let window_start = p.saturating_sub(context);
                    let window_end = (p + context + 1).min(offsets.len());
                    let byte_start = offsets[window_start] as usize;
                    let byte_end = if window_end < offsets.len() {
                        offsets[window_end] as usize
                    } else {
                        body.len()
                    };

                    let window_text = &body[byte_start..byte_end];
                    let window_tokens: Vec<&str> = window_text.unicode_words().collect();
                    let hit_idx = p - window_start;
                    if hit_idx >= window_tokens.len() {
                        continue;
                    }

                    hits.push(KwicHit {
                        doc_id,
                        path: PathBuf::from(path),
                        left: window_tokens[..hit_idx].join(" "),
                        hit: window_tokens[hit_idx].to_string(),
                        right: window_tokens[hit_idx + 1..].join(" "),
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

fn offsets_to_bytes(offsets: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(offsets.len() * 4);
    for &o in offsets {
        buf.extend_from_slice(&o.to_le_bytes());
    }
    buf
}

fn bytes_to_offsets(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let doc_id = builder.add_u64_field("doc_id", STORED);
    let path = builder.add_text_field("path", STORED);

    let indexing = TextFieldIndexing::default()
        .set_tokenizer(TOKENIZER_NAME)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);

    let body_options = TextOptions::default()
        .set_indexing_options(indexing.clone())
        .set_stored();
    let body = builder.add_text_field("body", body_options);

    let lemma_options = TextOptions::default().set_indexing_options(indexing.clone());
    let body_lemma = builder.add_text_field("body_lemma", lemma_options);

    let pos_options = TextOptions::default().set_indexing_options(indexing);
    let body_pos = builder.add_text_field("body_pos", pos_options);

    let token_offsets =
        builder.add_bytes_field("token_offsets", BytesOptions::default().set_stored());

    (
        builder.build(),
        Fields {
            doc_id,
            path,
            body,
            body_lemma,
            body_pos,
            token_offsets,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use corpust_annotate::WordOnlyAnnotator;

    #[test]
    fn round_trip_kwic() {
        let tmp = tempdir();
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents(
            [Document {
                id: 0,
                path: PathBuf::from("a.txt"),
                text: "the quick brown fox jumps over the lazy dog".to_string(),
            }],
            None,
        )
        .unwrap();

        let hits = idx.kwic("the", 2, 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].hit, "the");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn kwic_preserves_case_on_display() {
        let tmp = tempdir();
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents(
            [Document {
                id: 0,
                path: PathBuf::from("a.txt"),
                text: "The quick brown fox jumps over THE lazy dog".to_string(),
            }],
            None,
        )
        .unwrap();

        let hits = idx.kwic("the", 1, 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|h| h.hit == "The"));
        assert!(hits.iter().any(|h| h.hit == "THE"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn kwic_window_bounds_are_exact() {
        let tmp = tempdir();
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents(
            [Document {
                id: 0,
                path: PathBuf::from("a.txt"),
                text: "alpha beta gamma delta target epsilon zeta eta theta iota".to_string(),
            }],
            None,
        )
        .unwrap();

        let hits = idx.kwic("target", 2, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let h = &hits[0];
        assert_eq!(h.left, "gamma delta");
        assert_eq!(h.hit, "target");
        assert_eq!(h.right, "epsilon zeta");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn kwic_window_clamps_at_doc_edges() {
        let tmp = tempdir();
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents(
            [Document {
                id: 0,
                path: PathBuf::from("a.txt"),
                text: "target one two three".to_string(),
            }],
            None,
        )
        .unwrap();

        let hits = idx.kwic("target", 10, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].left, "");
        assert_eq!(hits[0].right, "one two three");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn annotated_path_indexes_successfully() {
        // WordOnlyAnnotator doesn't produce lemma/pos, but exercising it
        // proves the PreTokenizedString plumbing is wired up correctly.
        let tmp = tempdir();
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents(
            [Document {
                id: 0,
                path: PathBuf::from("a.txt"),
                text: "the quick brown fox".to_string(),
            }],
            Some(&WordOnlyAnnotator),
        )
        .unwrap();

        let hits = idx.kwic("quick", 1, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hit, "quick");
        assert_eq!(hits[0].left, "the");
        assert_eq!(hits[0].right, "brown");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn tempdir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("corpust-idx-{}", rand_suffix()))
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}
