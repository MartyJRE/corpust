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

pub mod assoc;

use anyhow::{Context, Result};
use corpust_annotate::{AnnotatedToken, Annotator};
use corpust_core::{DocId, Document};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tantivy::{
    DocAddress, DocSet, Index, IndexReader, ReloadPolicy, TERMINATED, TantivyDocument, Term, doc,
    postings::Postings,
    schema::{
        BytesOptions, FAST, Field, IndexRecordOption, STORED, Schema, TextFieldIndexing,
        TextOptions, Value,
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
    /// Query-result caches. The index is immutable once opened (the app
    /// re-opens a fresh `CorpusIndex` after a rebuild), so memoizing these
    /// O(corpus) reads is sound and turns repeat layer-toggles / revisits
    /// from seconds into nothing. Keyed by their full input so different
    /// layers / terms / limits don't collide.
    doc_cache: OnceLock<Vec<DocumentInfo>>,
    freq_cache: Mutex<HashMap<(QueryLayer, usize), FreqTable>>,
    dist_cache: Mutex<HashMap<(String, QueryLayer, usize), TermDistribution>>,
    /// Cache of full-corpus collocate scans, keyed by
    /// `(term, layer, left, right)`. The scan is the expensive part of a
    /// collocation query (a positional pass over every node occurrence);
    /// marginals + scores are recomputed cheaply on top.
    colloc_cache: Mutex<HashMap<(String, QueryLayer, usize, usize), CollocateScan>>,
}

/// Top-N `(term, count)` rows plus the field's grand-total token count —
/// the return shape of [`CorpusIndex::frequencies`].
pub type FreqTable = (Vec<(String, u64)>, u64);

/// The [`FreqTable`] for each [`QueryLayer`], produced in one shot by
/// [`CorpusIndex::all_frequencies`]. Build paths persist this so the
/// FrequencyView doesn't re-scan the term dictionary on every open.
#[derive(Debug, Clone)]
pub struct AllFrequencies {
    pub word: FreqTable,
    pub lemma: FreqTable,
    pub pos: FreqTable,
}

#[derive(Clone, Copy)]
struct Fields {
    doc_id: Field,
    path: Field,
    body: Field,
    body_lemma: Field,
    body_pos: Field,
    token_offsets: Field,
    /// Optional so indexes built before these fields landed still open
    /// cleanly (their schema lacks them). `token_count` is a FAST u64;
    /// the metadata fields are stored text/u64.
    token_count: Option<Field>,
    title: Option<Field>,
    author: Option<Field>,
    year: Option<Field>,
}

/// Which annotation layer a query targets.
///
/// `Word` queries the surface-form inverted index (case-insensitive).
/// `Lemma` queries the lemma layer (case-insensitive) — only populated
/// for documents indexed with an [`Annotator`] that emits lemmas.
/// `Pos` queries the POS-tag layer (case-sensitive — conventionally
/// uppercase tagsets like Penn Treebank).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryLayer {
    Word,
    Lemma,
    Pos,
}

/// One concordance line.
#[derive(Debug, Clone)]
pub struct KwicHit {
    pub doc_id: DocId,
    pub path: PathBuf,
    /// Token position of the hit within its document. Lets callers
    /// re-expand a wider context later via [`CorpusIndex::context_at`].
    pub hit_position: usize,
    pub left: String,
    pub hit: String,
    pub right: String,
}

/// A page of concordance lines plus the total match count, so callers can
/// paginate ("showing 1–200 of 12,431").
#[derive(Debug, Clone)]
pub struct KwicPage {
    pub hits: Vec<KwicHit>,
    /// Total occurrences of the term on the layer (the paging denominator).
    pub total: usize,
}

/// One document's summary, for the corpus document list.
#[derive(Debug, Clone)]
pub struct DocumentInfo {
    pub doc_id: DocId,
    pub path: PathBuf,
    pub token_count: usize,
    /// Title parsed from the document body at index time. `None` when no
    /// title could be confidently extracted.
    pub title: Option<String>,
    /// Author parsed from the document body at index time.
    pub author: Option<String>,
    /// Publication / release year (1500–2100) parsed from the body.
    pub year: Option<u32>,
}

/// Per-document metadata extracted from a document body at index time.
/// Every field is best-effort: a field is `None` rather than guessed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub year: Option<u32>,
}

/// Per-document occurrence count of a term.
#[derive(Debug, Clone)]
pub struct DocTermCount {
    pub doc_id: DocId,
    pub path: PathBuf,
    pub hits: u64,
    pub token_count: u64,
}

/// Raw windowed co-occurrence counts for a collocation query, produced
/// by a single positional pass over every occurrence of the node term.
/// Association scores ([`assoc`]) are computed on top of this plus the
/// corpus marginals — kept separate so the expensive scan can be cached
/// independently of the (cheap) scoring.
#[derive(Debug, Clone)]
pub struct CollocateScan {
    /// Number of node occurrences scanned. Equals the node's true corpus
    /// frequency unless `truncated` is set.
    pub node_freq: u64,
    /// Per-collocate `(left_count, right_count)` within the window.
    pub counts: HashMap<String, (u32, u32)>,
    /// Total collocate slots actually counted (sum of left+right hits) —
    /// the UI's "window tokens" figure.
    pub window_tokens: u64,
    /// True when the safety ceiling cut the scan short; scores then
    /// reflect a (large) sample of the node's occurrences, not all of
    /// them.
    pub truncated: bool,
}

/// Per-collocate counts bucketed by signed distance from the node, for
/// the collocation-by-distance view. `slots` runs `[-left … -1, +1 … +right]`
/// (the node position itself is excluded).
#[derive(Debug, Clone)]
pub struct DistanceProfile {
    pub node_freq: u64,
    /// The signed offsets the columns correspond to, e.g. `[-3,-2,-1,1,2,3]`.
    pub offsets: Vec<i32>,
    /// Per collocate word: a count for each offset in `offsets` (same length).
    pub rows: HashMap<String, Vec<u32>>,
    pub truncated: bool,
}

/// Corpus-wide distribution of a term: per-document hit counts plus a
/// dispersion histogram over the whole corpus.
#[derive(Debug, Clone)]
pub struct TermDistribution {
    /// Documents that contain the term, sorted by descending hit count.
    pub doc_counts: Vec<DocTermCount>,
    /// Per-bucket occurrence counts over a global position axis
    /// (documents concatenated in `doc_id` order). Length == requested
    /// bucket count.
    pub dispersion: Vec<u32>,
    pub total_hits: u64,
}

impl CorpusIndex {
    /// Create a new index on disk, overwriting any index already present.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("clearing {}", path.display()))?;
        }
        std::fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;

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
            token_count: schema.get_field("token_count").ok(),
            title: schema.get_field("title").ok(),
            author: schema.get_field("author").ok(),
            year: schema.get_field("year").ok(),
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
            doc_cache: OnceLock::new(),
            freq_cache: Mutex::new(HashMap::new()),
            dist_cache: Mutex::new(HashMap::new()),
            colloc_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Index a batch of documents. Commits once at the end.
    ///
    /// When `annotator` is `Some`, annotation runs across documents in
    /// parallel via rayon (each worker calls `annotator.annotate()`
    /// concurrently; annotators are expected to be stateless or handle
    /// concurrency internally). Writes to the Tantivy index stay
    /// sequential — Tantivy's IndexWriter is single-threaded.
    ///
    /// When `annotator` is `None`, the registered "corpust" tokenizer
    /// handles `body`; lemma / POS fields are left empty for each
    /// document.
    pub fn add_documents(
        &self,
        documents: impl IntoIterator<Item = Document>,
        annotator: Option<&(dyn Annotator + Sync)>,
    ) -> Result<()> {
        self.add_documents_with_progress(documents, annotator, |_| {})
    }

    /// Like [`Self::add_documents`] but reports each document written
    /// to the callback. `on_progress(n)` is invoked with the
    /// cumulative doc count after each write (1-based), so callers can
    /// drive a progress bar or emit UI events.
    ///
    /// The callback runs on the main thread (between Tantivy writes),
    /// never concurrently — safe to capture a non-`Sync` event bus.
    pub fn add_documents_with_progress<F>(
        &self,
        documents: impl IntoIterator<Item = Document>,
        annotator: Option<&(dyn Annotator + Sync)>,
        on_progress: F,
    ) -> Result<()>
    where
        F: FnMut(usize),
    {
        let mut writer = self.index.writer(50_000_000)?;
        let mut on_progress = on_progress;
        let mut done = 0usize;

        match annotator {
            None => {
                for document in documents {
                    self.add_unannotated(&mut writer, &document)?;
                    done += 1;
                    on_progress(done);
                }
            }
            Some(a) => {
                // Process in chunks so peak memory from buffered
                // annotation output stays bounded on big corpora.
                const CHUNK_SIZE: usize = 16;
                let docs: Vec<Document> = documents.into_iter().collect();
                for chunk in docs.chunks(CHUNK_SIZE) {
                    let annotated: Vec<(usize, Vec<AnnotatedToken<'_>>)> = chunk
                        .par_iter()
                        .enumerate()
                        .map(|(i, doc)| -> Result<_> {
                            let tokens = a.annotate(&doc.text)?;
                            Ok((i, tokens))
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Preserve input order when writing.
                    let mut ordered = annotated;
                    ordered.sort_by_key(|(i, _)| *i);
                    for (i, tokens) in ordered {
                        self.add_annotated(&mut writer, &chunk[i], &tokens)?;
                        done += 1;
                        on_progress(done);
                    }
                }
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

        let mut tantivy_doc = doc!(
            self.fields.doc_id => document.id,
            self.fields.path => document.path.display().to_string(),
            self.fields.body => document.text.clone(),
            self.fields.token_offsets => offsets_bytes,
        );
        if let Some(f) = self.fields.token_count {
            tantivy_doc.add_u64(f, offsets.len() as u64);
        }
        self.add_metadata_fields(&mut tantivy_doc, &document.text);
        writer.add_document(tantivy_doc)?;
        Ok(())
    }

    /// Extract title/author/year from the body and append them as stored
    /// fields, skipping any field we couldn't confidently extract (and any
    /// schema field absent on a pre-metadata index).
    fn add_metadata_fields(&self, tantivy_doc: &mut TantivyDocument, body: &str) {
        let meta = extract_metadata(body);
        if let (Some(field), Some(title)) = (self.fields.title, meta.title) {
            tantivy_doc.add_text(field, title);
        }
        if let (Some(field), Some(author)) = (self.fields.author, meta.author) {
            tantivy_doc.add_text(field, author);
        }
        if let (Some(field), Some(year)) = (self.fields.year, meta.year) {
            tantivy_doc.add_u64(field, year as u64);
        }
    }

    fn add_annotated(
        &self,
        writer: &mut tantivy::IndexWriter,
        document: &Document,
        annotated: &[AnnotatedToken<'_>],
    ) -> Result<()> {
        let mut body_tokens = Vec::with_capacity(annotated.len());
        let mut lemma_tokens = Vec::with_capacity(annotated.len());
        let mut pos_tokens = Vec::with_capacity(annotated.len());
        let mut offsets: Vec<u32> = Vec::with_capacity(annotated.len());

        for t in annotated {
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

        let mut tantivy_doc = doc!(
            self.fields.doc_id => document.id,
            self.fields.path => document.path.display().to_string(),
            self.fields.body => body_pre,
            self.fields.body_lemma => lemma_pre,
            self.fields.body_pos => pos_pre,
            self.fields.token_offsets => offsets_bytes,
        );
        if let Some(f) = self.fields.token_count {
            tantivy_doc.add_u64(f, offsets.len() as u64);
        }
        self.add_metadata_fields(&mut tantivy_doc, &document.text);
        writer.add_document(tantivy_doc)?;
        Ok(())
    }

    /// Run a KWIC (key word in context) query for a single term on the
    /// requested [`QueryLayer`]. Context extraction always uses the
    /// stored original text in `body` — regardless of which layer the
    /// hit was located on, the reported concordance line shows the
    /// source-faithful surface form.
    pub fn kwic(
        &self,
        term: &str,
        layer: QueryLayer,
        context: usize,
        limit: usize,
        offset: usize,
    ) -> Result<KwicPage> {
        let searcher = self.reader.searcher();
        let (query_field, lookup_term) = match layer {
            QueryLayer::Word => (self.fields.body, term.to_lowercase()),
            QueryLayer::Lemma => (self.fields.body_lemma, term.to_lowercase()),
            QueryLayer::Pos => (self.fields.body_pos, term.to_string()),
        };
        let term_obj = Term::from_field_text(query_field, &lookup_term);

        // Hits are globally ordered by (segment, doc, position). We walk
        // every posting to count the grand total (cheap — `term_freq` per
        // doc, no positions needed) but only fetch the document + build
        // windows for the docs whose hit range intersects the requested
        // page `[offset, offset + limit)`.
        let page_end = offset.saturating_add(limit);
        let mut hits = Vec::with_capacity(limit.min(512));
        let mut total: usize = 0;
        let mut positions_buf: Vec<u32> = Vec::new();

        'segments: for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            let inv_idx = seg_reader.inverted_index(query_field)?;
            let Some(mut postings) =
                inv_idx.read_postings(&term_obj, IndexRecordOption::WithFreqsAndPositions)?
            else {
                continue;
            };

            loop {
                let doc = postings.doc();
                if doc == TERMINATED {
                    continue 'segments;
                }
                let tf = postings.term_freq() as usize;
                let doc_start = total;
                total += tf;

                // Build hits only for docs overlapping the page window.
                if doc_start < page_end && total > offset {
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
                    let offsets = bytes_to_offsets(
                        retrieved
                            .get_first(self.fields.token_offsets)
                            .and_then(|v| v.as_bytes())
                            .unwrap_or_default(),
                    );

                    positions_buf.clear();
                    postings.positions(&mut positions_buf);
                    for (k, &pos) in positions_buf.iter().enumerate() {
                        let gi = doc_start + k; // global hit index
                        if gi < offset {
                            continue;
                        }
                        if gi >= page_end {
                            break;
                        }
                        let p = pos as usize;
                        if let Some((left, hit, right)) = window(body, &offsets, p, context) {
                            hits.push(KwicHit {
                                doc_id,
                                path: PathBuf::from(path),
                                hit_position: p,
                                left,
                                hit,
                                right,
                            });
                        }
                    }
                }

                postings.advance();
            }
        }

        Ok(KwicPage { hits, total })
    }

    /// Re-extract the concordance window around a known hit position in a
    /// specific document — used to expand the context shown for a single
    /// KWIC line after the fact, without re-running the whole query.
    ///
    /// Returns `(left, hit, right, token_count)` or `None` if the document
    /// or position can't be located. Locating the document is a linear
    /// scan over stored doc ids; document counts are modest so this is
    /// cheap enough for an on-click expansion.
    pub fn context_at(
        &self,
        doc_id: DocId,
        position: usize,
        context: usize,
    ) -> Result<Option<(String, String, String, usize)>> {
        let searcher = self.reader.searcher();
        for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            for local in seg_reader.doc_ids_alive() {
                let doc_addr = DocAddress::new(seg_ord as u32, local);
                let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
                let stored_id = retrieved
                    .get_first(self.fields.doc_id)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX);
                if stored_id != doc_id {
                    continue;
                }
                let body = retrieved
                    .get_first(self.fields.body)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let offsets = retrieved
                    .get_first(self.fields.token_offsets)
                    .and_then(|v| v.as_bytes())
                    .map(bytes_to_offsets)
                    .unwrap_or_default();
                let token_count = offsets.len();
                return Ok(window(body, &offsets, position, context)
                    .map(|(l, h, r)| (l, h, r, token_count)));
            }
        }
        Ok(None)
    }

    /// Enumerate every (alive) document with its path and token count.
    /// Sorted by `doc_id`.
    pub fn list_documents(&self) -> Result<Vec<DocumentInfo>> {
        if let Some(cached) = self.doc_cache.get() {
            return Ok(cached.clone());
        }
        let searcher = self.reader.searcher();
        let mut out = Vec::new();
        for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            for local in seg_reader.doc_ids_alive() {
                let doc_addr = DocAddress::new(seg_ord as u32, local);
                let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
                let doc_id = retrieved
                    .get_first(self.fields.doc_id)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let path = retrieved
                    .get_first(self.fields.path)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let token_count = retrieved
                    .get_first(self.fields.token_offsets)
                    .and_then(|v| v.as_bytes())
                    .map(|b| bytes_to_offsets(b).len())
                    .unwrap_or(0);
                let title = self.fields.title.and_then(|f| {
                    retrieved
                        .get_first(f)
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                });
                let author = self.fields.author.and_then(|f| {
                    retrieved
                        .get_first(f)
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                });
                let year = self
                    .fields
                    .year
                    .and_then(|f| retrieved.get_first(f).and_then(|v| v.as_u64()))
                    .map(|y| y as u32);
                out.push(DocumentInfo {
                    doc_id,
                    path: PathBuf::from(path),
                    token_count,
                    title,
                    author,
                    year,
                });
            }
        }
        out.sort_by_key(|d| d.doc_id);
        // Cache the document list — it never changes for an open index and
        // is hit on every `term_distribution` call (global position axis).
        let _ = self.doc_cache.set(out.clone());
        Ok(out)
    }

    /// Corpus-wide top-`limit` term frequencies on `layer`. Returns the
    /// `(term, count)` rows sorted by descending count, plus the grand
    /// total of (non-empty) token occurrences in the field — the
    /// denominator callers use to turn counts into percentages.
    ///
    /// NOTE: this is a full term-dictionary scan with one postings read
    /// per term. Fine at the current scale; for billion-word corpora the
    /// table should be precomputed at build time instead.
    pub fn frequencies(&self, layer: QueryLayer, limit: usize) -> Result<FreqTable> {
        if let Some(cached) = self
            .freq_cache
            .lock()
            .expect("freq_cache poisoned")
            .get(&(layer, limit))
        {
            return Ok(cached.clone());
        }
        let searcher = self.reader.searcher();
        let field = self.layer_field(layer);
        let mut totals: HashMap<String, u64> = HashMap::new();
        let mut grand_total: u64 = 0;
        for seg_reader in searcher.segment_readers() {
            let inv = seg_reader.inverted_index(field)?;
            let term_dict = inv.terms();
            let mut stream = term_dict.stream()?;
            while stream.advance() {
                let key = match std::str::from_utf8(stream.key()) {
                    Ok(k) if !k.is_empty() => k.to_string(),
                    _ => continue,
                };
                let term_info = stream.value().clone();
                let mut postings =
                    inv.read_postings_from_terminfo(&term_info, IndexRecordOption::WithFreqs)?;
                let mut sum: u64 = 0;
                loop {
                    let doc = postings.doc();
                    if doc == TERMINATED {
                        break;
                    }
                    sum += postings.term_freq() as u64;
                    postings.advance();
                }
                grand_total += sum;
                *totals.entry(key).or_default() += sum;
            }
        }
        let mut rows: Vec<(String, u64)> = totals.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        rows.truncate(limit);
        let result = (rows, grand_total);
        self.freq_cache
            .lock()
            .expect("freq_cache poisoned")
            .insert((layer, limit), result.clone());
        Ok(result)
    }

    /// Compute the top-`limit` [`FreqTable`] for all three layers in one
    /// call. Build paths use this to precompute and persist the tables
    /// (see the NOTE on [`Self::frequencies`]); `Lemma` / `Pos` come back
    /// empty for corpora indexed without annotation.
    pub fn all_frequencies(&self, limit: usize) -> Result<AllFrequencies> {
        Ok(AllFrequencies {
            word: self.frequencies(QueryLayer::Word, limit)?,
            lemma: self.frequencies(QueryLayer::Lemma, limit)?,
            pos: self.frequencies(QueryLayer::Pos, limit)?,
        })
    }

    /// Total token count on `layer` across the whole corpus — the `N`
    /// denominator for association measures. O(segments): Tantivy tracks
    /// this per inverted index, so no scan is needed.
    pub fn layer_total_tokens(&self, layer: QueryLayer) -> Result<u64> {
        let searcher = self.reader.searcher();
        let field = self.layer_field(layer);
        let mut total = 0u64;
        for seg_reader in searcher.segment_readers() {
            total += seg_reader.inverted_index(field)?.total_num_tokens();
        }
        Ok(total)
    }

    /// Corpus-wide frequency of each given term on `layer`. Used to fetch
    /// the `f(node)` / `f(collocate)` marginals for scoring. Terms absent
    /// from the index map to `0`.
    pub fn term_totals(&self, terms: &[String], layer: QueryLayer) -> Result<HashMap<String, u64>> {
        let searcher = self.reader.searcher();
        let field = self.layer_field(layer);
        let mut out: HashMap<String, u64> = HashMap::with_capacity(terms.len());
        for term in terms {
            let term_obj = Term::from_field_text(field, term);
            let mut sum = 0u64;
            for seg_reader in searcher.segment_readers() {
                let inv = seg_reader.inverted_index(field)?;
                if let Some(mut postings) =
                    inv.read_postings(&term_obj, IndexRecordOption::WithFreqs)?
                {
                    loop {
                        let doc = postings.doc();
                        if doc == TERMINATED {
                            break;
                        }
                        sum += postings.term_freq() as u64;
                        postings.advance();
                    }
                }
            }
            out.insert(term.clone(), sum);
        }
        Ok(out)
    }

    /// Count surface-form collocates in a `[left, right]` token window
    /// around **every** occurrence of `term` on `layer`, in a single
    /// positional pass (one stored-document fetch per document, not per
    /// hit). Collocate candidates are normalised the same way as
    /// [`tokenize_for_collocates`]: lowercased, edge-punctuation trimmed,
    /// single-character and pure-digit tokens dropped.
    ///
    /// `max_nodes` is a safety ceiling: if more than that many node
    /// occurrences are seen the scan stops early and the result is marked
    /// `truncated`. Pass a large value (or `usize::MAX`) for a full scan.
    ///
    /// Results are cached on `(term, layer, left, right)` since the scan
    /// is the costly part of a collocation query.
    pub fn collocate_counts(
        &self,
        term: &str,
        layer: QueryLayer,
        left: usize,
        right: usize,
        max_nodes: usize,
    ) -> Result<CollocateScan> {
        let (query_field, lookup_term) = match layer {
            QueryLayer::Word => (self.fields.body, term.to_lowercase()),
            QueryLayer::Lemma => (self.fields.body_lemma, term.to_lowercase()),
            QueryLayer::Pos => (self.fields.body_pos, term.to_string()),
        };
        let cache_key = (lookup_term.clone(), layer, left, right);
        if let Some(cached) = self
            .colloc_cache
            .lock()
            .expect("colloc_cache poisoned")
            .get(&cache_key)
        {
            return Ok(cached.clone());
        }

        let searcher = self.reader.searcher();
        let term_obj = Term::from_field_text(query_field, &lookup_term);

        let mut counts: HashMap<String, (u32, u32)> = HashMap::new();
        let mut node_freq: u64 = 0;
        let mut window_tokens: u64 = 0;
        let mut truncated = false;
        let mut positions_buf: Vec<u32> = Vec::new();

        'segments: for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            let inv_idx = seg_reader.inverted_index(query_field)?;
            let Some(mut postings) =
                inv_idx.read_postings(&term_obj, IndexRecordOption::WithFreqsAndPositions)?
            else {
                continue;
            };

            loop {
                let doc = postings.doc();
                if doc == TERMINATED {
                    continue 'segments;
                }

                let doc_addr = DocAddress::new(seg_ord as u32, doc);
                let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
                let body = retrieved
                    .get_first(self.fields.body)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let offsets_bytes = retrieved
                    .get_first(self.fields.token_offsets)
                    .and_then(|v| v.as_bytes())
                    .unwrap_or_default();
                let offsets = bytes_to_offsets(offsets_bytes);

                positions_buf.clear();
                postings.positions(&mut positions_buf);

                for &pos in &positions_buf {
                    if node_freq as usize >= max_nodes {
                        truncated = true;
                        break 'segments;
                    }
                    node_freq += 1;
                    let p = pos as usize;
                    for w in window_side(body, &offsets, p, left, Side::Left) {
                        let e = counts.entry(w).or_default();
                        e.0 += 1;
                        window_tokens += 1;
                    }
                    for w in window_side(body, &offsets, p, right, Side::Right) {
                        let e = counts.entry(w).or_default();
                        e.1 += 1;
                        window_tokens += 1;
                    }
                }

                postings.advance();
            }
        }

        let scan = CollocateScan {
            node_freq,
            counts,
            window_tokens,
            truncated,
        };
        self.colloc_cache
            .lock()
            .expect("colloc_cache poisoned")
            .insert(cache_key, scan.clone());
        Ok(scan)
    }

    /// Like [`Self::collocate_counts`] but keeps each collocate's count
    /// bucketed by **signed distance** from the node (`-left … -1, +1 …
    /// +right`), powering the collocation-by-distance view. One positional
    /// pass; tokens are sliced by their stored offsets so a distance maps
    /// to exactly one token.
    pub fn collocate_by_distance(
        &self,
        term: &str,
        layer: QueryLayer,
        left: usize,
        right: usize,
        max_nodes: usize,
    ) -> Result<DistanceProfile> {
        let (query_field, lookup_term) = match layer {
            QueryLayer::Word => (self.fields.body, term.to_lowercase()),
            QueryLayer::Lemma => (self.fields.body_lemma, term.to_lowercase()),
            QueryLayer::Pos => (self.fields.body_pos, term.to_string()),
        };
        let offsets_axis: Vec<i32> = (1..=left as i32)
            .rev()
            .map(|d| -d)
            .chain(1..=right as i32)
            .collect();
        let slot_of = |d: i32| -> usize {
            if d < 0 {
                (d + left as i32) as usize
            } else {
                left + (d as usize - 1)
            }
        };

        let searcher = self.reader.searcher();
        let term_obj = Term::from_field_text(query_field, &lookup_term);
        let mut rows: HashMap<String, Vec<u32>> = HashMap::new();
        let slots = left + right;
        let mut node_freq: u64 = 0;
        let mut truncated = false;
        let mut positions_buf: Vec<u32> = Vec::new();

        'segments: for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            let inv_idx = seg_reader.inverted_index(query_field)?;
            let Some(mut postings) =
                inv_idx.read_postings(&term_obj, IndexRecordOption::WithFreqsAndPositions)?
            else {
                continue;
            };
            loop {
                let doc = postings.doc();
                if doc == TERMINATED {
                    continue 'segments;
                }
                let doc_addr = DocAddress::new(seg_ord as u32, doc);
                let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
                let body = retrieved
                    .get_first(self.fields.body)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let offsets = bytes_to_offsets(
                    retrieved
                        .get_first(self.fields.token_offsets)
                        .and_then(|v| v.as_bytes())
                        .unwrap_or_default(),
                );
                let n = offsets.len();
                let start_of = |i: usize| -> usize {
                    if i < n {
                        offsets[i] as usize
                    } else {
                        body.len()
                    }
                };

                positions_buf.clear();
                postings.positions(&mut positions_buf);
                for &pos in &positions_buf {
                    if node_freq as usize >= max_nodes {
                        truncated = true;
                        break 'segments;
                    }
                    node_freq += 1;
                    let p = pos as usize;
                    if p >= n {
                        continue;
                    }
                    let lo = p.saturating_sub(left);
                    let hi = (p + right + 1).min(n);
                    for i in lo..hi {
                        if i == p {
                            continue;
                        }
                        if let Some(w) = normalize_token(&body[start_of(i)..start_of(i + 1)]) {
                            let slot = slot_of(i as i32 - p as i32);
                            rows.entry(w).or_insert_with(|| vec![0u32; slots])[slot] += 1;
                        }
                    }
                }
                postings.advance();
            }
        }

        Ok(DistanceProfile {
            node_freq,
            offsets: offsets_axis,
            rows,
            truncated,
        })
    }

    /// Per-document hit counts and a corpus-wide dispersion histogram for
    /// a single term on `layer`, in one pass over the term's postings.
    ///
    /// The dispersion axis treats the corpus as one long stream of tokens
    /// (documents concatenated in `doc_id` order); each occurrence falls
    /// into one of `buckets` equal-width buckets along that axis.
    pub fn term_distribution(
        &self,
        term: &str,
        layer: QueryLayer,
        buckets: usize,
    ) -> Result<TermDistribution> {
        let buckets = buckets.max(1);
        let cache_key = (term.to_string(), layer, buckets);
        if let Some(cached) = self
            .dist_cache
            .lock()
            .expect("dist_cache poisoned")
            .get(&cache_key)
        {
            return Ok(cached.clone());
        }
        let searcher = self.reader.searcher();
        let field = self.layer_field(layer);
        let lookup = match layer {
            QueryLayer::Word | QueryLayer::Lemma => term.to_lowercase(),
            QueryLayer::Pos => term.to_string(),
        };
        let term_obj = Term::from_field_text(field, &lookup);

        // Global position axis: documents in doc_id order, with cumulative
        // token offsets. Reuse list_documents for ordering + token counts.
        let docs = self.list_documents()?;
        let mut global_start: HashMap<DocId, u64> = HashMap::new();
        let mut cursor: u64 = 0;
        for d in &docs {
            global_start.insert(d.doc_id, cursor);
            cursor += d.token_count as u64;
        }
        let total_tokens = cursor.max(1);

        let mut dispersion = vec![0u32; buckets];
        let mut doc_hits: HashMap<DocId, u64> = HashMap::new();
        let mut total_hits: u64 = 0;
        let mut positions_buf: Vec<u32> = Vec::new();

        for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            let inv = seg_reader.inverted_index(field)?;
            let Some(mut postings) =
                inv.read_postings(&term_obj, IndexRecordOption::WithFreqsAndPositions)?
            else {
                continue;
            };
            // Read the stored doc id from the FAST column when present —
            // a columnar lookup, vs fetching the whole stored document
            // (incl. the ~MB body) per matched doc. Falls back to the
            // stored field on indexes built before doc_id was FAST.
            let doc_id_col = seg_reader.fast_fields().u64("doc_id").ok();
            loop {
                let doc = postings.doc();
                if doc == TERMINATED {
                    break;
                }
                let stored_id = match &doc_id_col {
                    Some(col) => col.first(doc).unwrap_or(0),
                    None => {
                        let doc_addr = DocAddress::new(seg_ord as u32, doc);
                        let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
                        retrieved
                            .get_first(self.fields.doc_id)
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0)
                    }
                };
                let freq = postings.term_freq() as u64;
                *doc_hits.entry(stored_id).or_default() += freq;
                total_hits += freq;

                let base = *global_start.get(&stored_id).unwrap_or(&0);
                positions_buf.clear();
                postings.positions(&mut positions_buf);
                for &p in &positions_buf {
                    let global_pos = base + p as u64;
                    let b = ((global_pos * buckets as u64) / total_tokens) as usize;
                    dispersion[b.min(buckets - 1)] += 1;
                }
                postings.advance();
            }
        }

        let mut doc_counts: Vec<DocTermCount> = docs
            .iter()
            .filter_map(|d| {
                let hits = *doc_hits.get(&d.doc_id).unwrap_or(&0);
                (hits > 0).then(|| DocTermCount {
                    doc_id: d.doc_id,
                    path: d.path.clone(),
                    hits,
                    token_count: d.token_count as u64,
                })
            })
            .collect();
        doc_counts.sort_by(|a, b| b.hits.cmp(&a.hits).then_with(|| a.doc_id.cmp(&b.doc_id)));

        let result = TermDistribution {
            doc_counts,
            dispersion,
            total_hits,
        };
        self.dist_cache
            .lock()
            .expect("dist_cache poisoned")
            .insert(cache_key, result.clone());
        Ok(result)
    }

    fn layer_field(&self, layer: QueryLayer) -> Field {
        match layer {
            QueryLayer::Word => self.fields.body,
            QueryLayer::Lemma => self.fields.body_lemma,
            QueryLayer::Pos => self.fields.body_pos,
        }
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

/// Extract the concordance window around token position `p`: `context`
/// tokens of either side, clamped at document edges. Context text is read
/// from the stored original `body` so the surface form is source-faithful
/// regardless of which layer the hit was located on. Returns
/// `(left, hit, right)`, or `None` if `p` is out of range.
fn window(
    body: &str,
    offsets: &[u32],
    p: usize,
    context: usize,
) -> Option<(String, String, String)> {
    let n = offsets.len();
    if p >= n {
        return None;
    }
    let window_start = p.saturating_sub(context);
    let window_end = (p + context + 1).min(n); // exclusive token index

    // Slice each token directly by its stored byte offset. `offsets[i]` is
    // the start of token `i`; the start of token `i+1` (or end of body) is
    // its end. This keeps the displayed hit aligned with the indexed
    // position `p` even when the indexing tokenizer (an annotator's
    // tokenisation) splits the text differently from a naive
    // re-tokenisation — re-tokenising here silently mis-aligned the hit on
    // annotated corpora.
    let start_of = |i: usize| -> usize {
        if i < n {
            offsets[i] as usize
        } else {
            body.len()
        }
    };
    let left = body[start_of(window_start)..start_of(p)].trim();
    let hit = body[start_of(p)..start_of(p + 1)].trim();
    let right = body[start_of(p + 1)..start_of(window_end)].trim();

    Some((left.to_owned(), hit.to_owned(), right.to_owned()))
}

/// Which side of the node a window extends to.
enum Side {
    Left,
    Right,
}

/// Extract and normalise the collocate candidates within `count` token
/// positions on one `side` of node position `p`. Reads surface text from
/// `body` via `offsets`, then applies [`tokenize_for_collocates`].
///
/// Returns an empty vec when `p` is out of range or the side is empty
/// (node at a document edge, or `count == 0`).
fn window_side(body: &str, offsets: &[u32], p: usize, count: usize, side: Side) -> Vec<String> {
    if count == 0 || p >= offsets.len() {
        return Vec::new();
    }
    let (start, end) = match side {
        // positions [p-count, p)
        Side::Left => (p.saturating_sub(count), p),
        // positions (p, p+count]
        Side::Right => (p + 1, (p + 1 + count).min(offsets.len())),
    };
    if start >= end {
        return Vec::new();
    }
    let byte_start = offsets[start] as usize;
    let byte_end = if end < offsets.len() {
        offsets[end] as usize
    } else {
        body.len()
    };
    tokenize_for_collocates(&body[byte_start..byte_end]).collect()
}

/// Normalise a single token slice for collocate counting: lowercase,
/// strip edge punctuation, drop empty/single-character and pure-digit
/// tokens. Returns `None` for tokens that aren't collocate candidates.
fn normalize_token(raw: &str) -> Option<String> {
    let trimmed: String = raw
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if trimmed.chars().count() < 2 {
        return None;
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(trimmed)
}

/// Split a window substring into collocate candidates:
/// whitespace/punctuation-delimited, lowercased, edge-punctuation
/// trimmed, dropping empty/single-character and pure-digit tokens. Kept
/// in sync with the equivalent normalisation the Tauri command applied
/// before collocation moved into the index.
fn tokenize_for_collocates(s: &str) -> impl Iterator<Item = String> + '_ {
    s.unicode_words().filter_map(|tok| {
        let trimmed: String = tok
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if trimmed.chars().count() < 2 {
            return None;
        }
        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        Some(trimmed)
    })
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

// ---------------------------------------------------------------------------
// Metadata extraction
// ---------------------------------------------------------------------------

/// How many leading lines we scan when hunting for bare-style title/author.
const META_SCAN_LINES: usize = 40;

/// Extract best-effort title / author / year from a document body.
///
/// Handles two shapes seen in real corpora:
///
/// 1. **Project Gutenberg boilerplate headers** — explicit `Title:`,
///    `Author:`, and `Release Date:` / `Copyright` label lines.
/// 2. **Bare leading style** — the first meaningful non-empty line is the
///    title and a line of the form `by <name>` (case-insensitive) within the
///    first [`META_SCAN_LINES`] lines is the author.
///
/// Any field that can't be found confidently is left `None` — never guessed.
pub fn extract_metadata(body: &str) -> DocMetadata {
    let labelled = extract_labelled(body);
    // Year search spans the whole header region regardless of shape.
    let year = labelled.year.or_else(|| extract_year(body));

    // If the labelled header gave us a title, trust it wholesale (it's the
    // canonical PG block). Otherwise fall back to the bare leading style.
    if labelled.title.is_some() || labelled.author.is_some() {
        return DocMetadata {
            title: labelled.title,
            author: labelled.author,
            year,
        };
    }

    let bare = extract_bare(body);
    DocMetadata {
        title: bare.title,
        author: bare.author,
        year,
    }
}

/// Pull `Title:` / `Author:` / `Release Date:` style labelled fields from a
/// Project Gutenberg header block. Returns empty when none are present.
fn extract_labelled(body: &str) -> DocMetadata {
    let mut meta = DocMetadata::default();
    for line in body.lines().take(60) {
        let trimmed = line.trim();
        if let Some(rest) = strip_label(trimmed, "title") {
            if meta.title.is_none() && !rest.is_empty() {
                meta.title = Some(rest.to_string());
            }
        } else if let Some(rest) = strip_label(trimmed, "author") {
            if meta.author.is_none() && !rest.is_empty() {
                meta.author = Some(rest.to_string());
            }
        } else if let Some(rest) = strip_label(trimmed, "release date")
            && meta.year.is_none()
        {
            meta.year = year_in(rest);
        }
    }
    meta
}

/// Case-insensitively match `"<label>:"` at the start of `line`, returning the
/// trimmed remainder if it matches.
fn strip_label<'a>(line: &'a str, label: &str) -> Option<&'a str> {
    let bytes = label.len();
    // `label` is ASCII, so a byte-prefix slice is char-safe only when the
    // line is at least that long; guard with `get` to avoid splitting a
    // multibyte char that happens to start within the prefix window.
    let prefix = line.get(..bytes)?;
    if prefix.eq_ignore_ascii_case(label) && line.as_bytes()[bytes] == b':' {
        return Some(line[bytes + 1..].trim());
    }
    None
}

/// Bare leading style: first meaningful line is the title; the first `by …`
/// line within the scan window is the author.
fn extract_bare(body: &str) -> DocMetadata {
    let mut meta = DocMetadata::default();
    let mut seen = 0usize;
    // Set when the previous meaningful line was a lone "by" — the next
    // meaningful line is then the author name.
    let mut author_on_next = false;
    for raw in body.lines() {
        if seen >= META_SCAN_LINES {
            break;
        }
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if is_meta_noise(line) {
            continue;
        }
        seen += 1;

        // Author was deferred from a lone "by" on the previous line.
        if author_on_next {
            author_on_next = false;
            if meta.author.is_none() && is_plausible_name(line) {
                meta.author = Some(normalise_author(line));
                continue;
            }
        }

        // A lone "by" — the author is on the following meaningful line.
        if line.eq_ignore_ascii_case("by") {
            author_on_next = true;
            continue;
        }

        // `by <author>` (the leading "by" is not part of the name).
        if let Some(name) = strip_by_prefix(line) {
            if meta.author.is_none() && is_plausible_name(name) {
                meta.author = Some(normalise_author(name));
            }
            continue;
        }

        // First meaningful content line that isn't a `by` line is the title.
        if meta.title.is_none() {
            meta.title = Some(normalise_title(line));
        }
    }
    meta
}

/// Strip a leading case-insensitive `by ` and return the remainder, or `None`
/// if the line doesn't start with `by`.
fn strip_by_prefix(line: &str) -> Option<&str> {
    let lower = line.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("by ") {
        let offset = line.len() - rest.len();
        Some(line[offset..].trim())
    } else {
        None
    }
}

/// Lines that are clearly not title/author material (proofreading credits,
/// illustration captions, decorative rules).
fn is_meta_noise(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.starts_with("produced")
        || lower.starts_with("[illustration")
        || lower.starts_with("illustration")
        || lower.starts_with("transcriber")
        || lower.starts_with("contents")
        // Distributed-proofreading credit blocks, often wrapped across lines.
        || lower.contains("proofread")
        || lower.contains("pgdp.net")
        || lower.contains("distributed proof")
        // Decorative / structural lines: no alphabetic character at all.
        || !line.chars().any(|c| c.is_alphabetic())
}

/// A plausible author name: at least one alphabetic run, not absurdly long,
/// and not an obvious sentence (heuristic guard against false positives like
/// "by the time he arrived…").
fn is_plausible_name(name: &str) -> bool {
    let trimmed = name.trim_end_matches([',', '.']).trim();
    !trimmed.is_empty() && trimmed.len() <= 80 && trimmed.split_whitespace().count() <= 8
}

/// Trim trailing punctuation a title commonly ends with (`;`, trailing comma).
fn normalise_title(title: &str) -> String {
    title.trim().trim_end_matches([';', ',']).trim().to_string()
}

/// Trim a trailing comma/period from an author line.
fn normalise_author(author: &str) -> String {
    author
        .trim()
        .trim_end_matches([',', '.'])
        .trim()
        .to_string()
}

/// Find a 4-digit year in [1500, 2100] anywhere in a release/copyright line.
fn year_in(text: &str) -> Option<u32> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if bytes[i].is_ascii_digit() {
            // Collect the maximal digit run starting here.
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // Exactly-4-digit runs only, so we don't pick years out of
            // longer identifiers like ebook numbers.
            if i - start == 4
                && let Ok(y) = text[start..i].parse::<u32>()
                && (1500..=2100).contains(&y)
            {
                return Some(y);
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Search the header region for a year on a release/copyright line.
fn extract_year(body: &str) -> Option<u32> {
    for line in body.lines().take(60) {
        let lower = line.to_ascii_lowercase();
        let is_date_line = lower.contains("release date")
            || lower.contains("copyright")
            || lower.contains("published");
        if is_date_line && let Some(y) = year_in(line) {
            return Some(y);
        }
    }
    None
}

fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    // doc_id is FAST so term_distribution can map a matched segment-doc to
    // its stored id via a columnar read, instead of fetching the whole
    // stored document (incl. the ~MB body) per match.
    let doc_id = builder.add_u64_field("doc_id", STORED | FAST);
    let path = builder.add_text_field("path", STORED);
    // Per-document token count, FAST so the distribution axis is built
    // columnar without parsing every doc's token_offsets sidecar.
    let token_count = builder.add_u64_field("token_count", STORED | FAST);

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

    // Per-document metadata extracted at index time. Stored only — not
    // indexed for search (the document list reads them back by address).
    let title = builder.add_text_field("title", STORED);
    let author = builder.add_text_field("author", STORED);
    let year = builder.add_u64_field("year", STORED);

    (
        builder.build(),
        Fields {
            doc_id,
            path,
            body,
            body_lemma,
            body_pos,
            token_offsets,
            token_count: Some(token_count),
            title: Some(title),
            author: Some(author),
            year: Some(year),
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

        let hits = idx.kwic("the", QueryLayer::Word, 2, 10, 0).unwrap().hits;
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

        let hits = idx.kwic("the", QueryLayer::Word, 1, 10, 0).unwrap().hits;
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

        let hits = idx.kwic("target", QueryLayer::Word, 2, 10, 0).unwrap().hits;
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

        let hits = idx
            .kwic("target", QueryLayer::Word, 10, 10, 0)
            .unwrap()
            .hits;
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

        let hits = idx.kwic("quick", QueryLayer::Word, 1, 10, 0).unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hit, "quick");
        assert_eq!(hits[0].left, "the");
        assert_eq!(hits[0].right, "brown");

        // WordOnly emits empty lemma / pos tokens — queries on those
        // layers find nothing.
        let no_lemma = idx.kwic("quick", QueryLayer::Lemma, 1, 10, 0).unwrap().hits;
        assert!(no_lemma.is_empty());
        let no_pos = idx.kwic("NN", QueryLayer::Pos, 1, 10, 0).unwrap().hits;
        assert!(no_pos.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn two_doc_index() -> (std::path::PathBuf, CorpusIndex) {
        let tmp = tempdir();
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents(
            [
                Document {
                    id: 0,
                    path: PathBuf::from("a.txt"),
                    text: "the cat sat on the mat".to_string(),
                },
                Document {
                    id: 1,
                    path: PathBuf::from("b.txt"),
                    text: "the dog and the cat".to_string(),
                },
            ],
            None,
        )
        .unwrap();
        (tmp, idx)
    }

    #[test]
    fn list_documents_reports_paths_and_token_counts() {
        let (tmp, idx) = two_doc_index();
        let docs = idx.list_documents().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].doc_id, 0);
        assert_eq!(docs[0].path, PathBuf::from("a.txt"));
        assert_eq!(docs[0].token_count, 6); // the cat sat on the mat
        assert_eq!(docs[1].doc_id, 1);
        assert_eq!(docs[1].token_count, 5); // the dog and the cat
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn collocate_counts_tallies_window_sides() {
        // doc0: the(0) cat(1) sat(2) on(3) the(4) mat(5)
        // doc1: the(0) dog(1) and(2) the(3) cat(4)
        let (tmp, idx) = two_doc_index();

        let scan = idx
            .collocate_counts("the", QueryLayer::Word, 1, 1, usize::MAX)
            .unwrap();

        assert_eq!(scan.node_freq, 4, "the occurs 4×");
        assert!(!scan.truncated);
        // cat sits immediately right of "the" twice (doc0 p0, doc1 p3).
        assert_eq!(scan.counts.get("cat"), Some(&(0, 2)));
        // on/and precede a "the"; mat/dog follow one.
        assert_eq!(scan.counts.get("on"), Some(&(1, 0)));
        assert_eq!(scan.counts.get("and"), Some(&(1, 0)));
        assert_eq!(scan.counts.get("mat"), Some(&(0, 1)));
        assert_eq!(scan.counts.get("dog"), Some(&(0, 1)));
        // 2(cat) + 1(on) + 1(and) + 1(mat) + 1(dog) collocate slots filled.
        assert_eq!(scan.window_tokens, 6);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn collocate_by_distance_buckets_offsets() {
        // doc0: the(0) cat(1) sat(2) on(3) the(4) mat(5)
        // doc1: the(0) dog(1) and(2) the(3) cat(4)
        let (tmp, idx) = two_doc_index();
        let prof = idx
            .collocate_by_distance("the", QueryLayer::Word, 1, 1, usize::MAX)
            .unwrap();
        assert_eq!(prof.node_freq, 4);
        assert_eq!(prof.offsets, vec![-1, 1]);
        // cat: never immediately left of "the", twice immediately right.
        assert_eq!(prof.rows.get("cat"), Some(&vec![0, 2]));
        // on / and precede a "the"; mat / dog follow one.
        assert_eq!(prof.rows.get("on"), Some(&vec![1, 0]));
        assert_eq!(prof.rows.get("and"), Some(&vec![1, 0]));
        assert_eq!(prof.rows.get("mat"), Some(&vec![0, 1]));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn collocate_marginals_and_truncation() {
        let (tmp, idx) = two_doc_index();

        // N = every surface token across both docs (6 + 5).
        assert_eq!(idx.layer_total_tokens(QueryLayer::Word).unwrap(), 11);

        let totals = idx
            .term_totals(&["the".to_string(), "cat".to_string()], QueryLayer::Word)
            .unwrap();
        assert_eq!(totals.get("the"), Some(&4));
        assert_eq!(totals.get("cat"), Some(&2));

        // Safety ceiling stops the scan early and flags it.
        let scan = idx
            .collocate_counts("the", QueryLayer::Word, 1, 1, 2)
            .unwrap();
        assert_eq!(scan.node_freq, 2);
        assert!(scan.truncated);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn list_documents_round_trips_metadata() {
        let tmp = tempdir();
        let idx = CorpusIndex::create(&tmp).unwrap();
        idx.add_documents(
            [Document {
                id: 0,
                path: PathBuf::from("frank.txt"),
                text: "Frankenstein\n\nby Mary Shelley\n\nRelease Date: 1818\n\nLetter 1"
                    .to_string(),
            }],
            None,
        )
        .unwrap();
        let docs = idx.list_documents().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title.as_deref(), Some("Frankenstein"));
        assert_eq!(docs[0].author.as_deref(), Some("Mary Shelley"));
        assert_eq!(docs[0].year, Some(1818));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn frequencies_ranks_terms_by_total_count() {
        let (tmp, idx) = two_doc_index();
        let (rows, total) = idx.frequencies(QueryLayer::Word, 10).unwrap();
        // 11 tokens across both docs; "the" appears 4 times.
        assert_eq!(total, 11);
        assert_eq!(rows[0], ("the".to_string(), 4));
        // "cat" appears twice; everything else once.
        let cat = rows.iter().find(|(t, _)| t == "cat").unwrap();
        assert_eq!(cat.1, 2);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn context_at_reexpands_window() {
        let (tmp, idx) = two_doc_index();
        // Locate a hit first, then re-expand it wider.
        let hits = idx.kwic("sat", QueryLayer::Word, 1, 10, 0).unwrap().hits;
        assert_eq!(hits.len(), 1);
        let h = &hits[0];
        assert_eq!(h.left, "cat");
        assert_eq!(h.right, "on");

        let (left, hit, right, tokens) = idx
            .context_at(h.doc_id, h.hit_position, 10)
            .unwrap()
            .unwrap();
        assert_eq!(hit, "sat");
        assert_eq!(left, "the cat");
        assert_eq!(right, "on the mat");
        assert_eq!(tokens, 6);

        // Unknown doc id yields None.
        assert!(idx.context_at(999, 0, 3).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn term_distribution_counts_per_doc_and_buckets() {
        let (tmp, idx) = two_doc_index();
        let dist = idx.term_distribution("the", QueryLayer::Word, 4).unwrap();
        assert_eq!(dist.total_hits, 4);
        // Both docs contain "the".
        assert_eq!(dist.doc_counts.len(), 2);
        let total_in_docs: u64 = dist.doc_counts.iter().map(|d| d.hits).sum();
        assert_eq!(total_in_docs, 4);
        // Bucket counts sum to the total number of occurrences.
        assert_eq!(dist.dispersion.iter().map(|&c| c as u64).sum::<u64>(), 4);
        assert_eq!(dist.dispersion.len(), 4);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ---- metadata extraction -------------------------------------------

    #[test]
    fn extract_pg_boilerplate_header() {
        let body = "\
The Project Gutenberg eBook of Frankenstein

Title: Frankenstein; Or, The Modern Prometheus

Author: Mary Wollstonecraft Shelley

Release Date: October 31, 1993 [eBook #84]

Language: English

*** START OF THE PROJECT GUTENBERG EBOOK ***

Letter 1 ...";
        let m = extract_metadata(body);
        assert_eq!(
            m.title.as_deref(),
            Some("Frankenstein; Or, The Modern Prometheus")
        );
        assert_eq!(m.author.as_deref(), Some("Mary Wollstonecraft Shelley"));
        assert_eq!(m.year, Some(1993));
    }

    #[test]
    fn extract_pg_release_date_bracket_year() {
        let body = "Title: Moby Dick\nAuthor: Herman Melville\nRelease Date: December, 2008 [EBook #2701]\n";
        let m = extract_metadata(body);
        assert_eq!(m.title.as_deref(), Some("Moby Dick"));
        assert_eq!(m.author.as_deref(), Some("Herman Melville"));
        assert_eq!(m.year, Some(2008));
    }

    #[test]
    fn extract_bare_leading_style() {
        // Boilerplate-stripped shape: title on line 1, `by <author>` below.
        let body = "\
Frankenstein;

or, the Modern Prometheus

by Mary Wollstonecraft (Godwin) Shelley


 CONTENTS
 Letter 1";
        let m = extract_metadata(body);
        assert_eq!(m.title.as_deref(), Some("Frankenstein"));
        assert_eq!(
            m.author.as_deref(),
            Some("Mary Wollstonecraft (Godwin) Shelley")
        );
        assert_eq!(m.year, None);
    }

    #[test]
    fn extract_bare_capital_by() {
        let body = "MOBY-DICK;\n\nor, THE WHALE.\n\nBy Herman Melville\n\nCONTENTS\n";
        let m = extract_metadata(body);
        assert_eq!(m.title.as_deref(), Some("MOBY-DICK"));
        assert_eq!(m.author.as_deref(), Some("Herman Melville"));
    }

    #[test]
    fn extract_skips_illustration_and_decorative_noise() {
        let body = "\
[Illustration:
                             GEORGE ALLEN
                        ]

                                PRIDE.
                                  by
                             Jane Austen,
";
        let m = extract_metadata(body);
        // First non-noise line wins as title; decorative/illustration lines
        // are skipped. (Indented all-caps publisher lines do count as text,
        // so this documents the heuristic's real behaviour.)
        assert!(m.title.is_some());
        assert_eq!(m.author.as_deref(), Some("Jane Austen"));
    }

    #[test]
    fn extract_returns_none_when_nothing_found() {
        // Pure prose with no leading title/author cues and a sentence-y `by`.
        let body = "The fox ran. It was chased by the dog through the woods.";
        let m = extract_metadata(body);
        // First line becomes the title (defined heuristic), but no author
        // (the `by` is mid-sentence, not line-leading) and no year.
        assert_eq!(m.author, None);
        assert_eq!(m.year, None);
    }

    #[test]
    fn extract_year_only_from_4_digit_runs() {
        // 5-digit ebook ids must not be mistaken for a year.
        assert_eq!(year_in("Release Date: [eBook #58169]"), None);
        assert_eq!(year_in("Release Date: 1851 [eBook #2701]"), Some(1851));
        assert_eq!(year_in("Copyright 1999 by someone"), Some(1999));
        // Out-of-range years are rejected.
        assert_eq!(year_in("circa 1200 BC"), None);
    }

    #[test]
    fn extract_from_real_gutenberg_files_if_present() {
        // Best-effort assertions on real downloaded fixtures. The files are
        // not committed, so skip silently when absent.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/gutenberg");
        let check = |id: &str, want_author: &str| {
            let path = dir.join(format!("{id}.txt"));
            let Ok(body) = std::fs::read_to_string(&path) else {
                return;
            };
            let m = extract_metadata(&body);
            assert!(m.title.is_some(), "expected a title for {id}.txt, got none");
            assert_eq!(
                m.author.as_deref(),
                Some(want_author),
                "author mismatch for {id}.txt"
            );
        };
        check("84", "Mary Wollstonecraft (Godwin) Shelley");
        check("2701", "Herman Melville");
        check("2554", "Fyodor Dostoevsky");
    }

    #[test]
    #[ignore = "diagnostic: prints extraction over all downloaded fixtures"]
    fn dump_real_extractions() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/gutenberg");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        files.sort();
        for f in files {
            let body = std::fs::read_to_string(&f).unwrap();
            let m = extract_metadata(&body);
            println!(
                "{:>12}  title={:?}  author={:?}  year={:?}",
                f.file_name().unwrap().to_string_lossy(),
                m.title,
                m.author,
                m.year
            );
        }
    }

    fn tempdir() -> std::path::PathBuf {
        // Atomic counter so tests running in parallel never collide
        // on the same nanosecond. Process id keeps it unique across
        // concurrent `cargo test` invocations too.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("corpust-idx-{pid}-{nanos}-{seq}"))
    }
}
