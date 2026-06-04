//! CQL (corpus query language) matching.
//!
//! A CQL query is a sequence of token patterns; each token pattern is a set
//! of attribute constraints (`word` / `lemma` / `pos`) that must all hold at
//! the same token position. Values can be exact or regex. A multi-token
//! query matches a run of consecutive positions (token *i+1* at *p+1*).
//!
//! Because `body` / `body_lemma` / `body_pos` are indexed over one shared
//! tokenisation, a position `p` denotes the same token on every layer — so
//! a multi-attribute constraint is just an intersection of per-layer
//! position sets at `p`. [`CorpusIndex::cql_scan`] is the shared primitive;
//! the KWIC / collocation / distribution surfaces all consume it.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use regex::Regex;
use tantivy::postings::Postings;
use tantivy::schema::{IndexRecordOption, Value};
use tantivy::{DocAddress, DocSet, TERMINATED, TantivyDocument, Term};

use crate::{
    CollocateScan, CorpusIndex, DistanceProfile, DocFilter, DocId, DocTermCount, KwicHit, KwicPage,
    QueryLayer, Side, TermDistribution, bytes_to_offsets, normalize_token, window_side,
    window_span,
};

/// Safety ceiling on matched spans for a single CQL scan — mirrors the
/// collocation node ceiling. Broad patterns (`[pos="N.*"]`) stay bounded;
/// callers surface the cut via a `truncated` flag.
const MAX_SPANS: u64 = 1_000_000;

/// One token's constraint on a single annotation layer.
#[derive(Debug, Clone)]
pub struct AttrConstraint {
    pub layer: QueryLayer,
    pub matcher: Matcher,
}

/// How an attribute value is matched against the indexed terms.
#[derive(Debug, Clone)]
pub enum Matcher {
    /// Exact term (already cased for its layer: lowercased for word/lemma).
    Exact(String),
    /// Full-token-anchored regex (case-insensitive for word/lemma).
    Regex(Regex),
}

/// One token position: every constraint must hold at the same position.
#[derive(Debug, Clone)]
pub struct TokenPattern {
    pub constraints: Vec<AttrConstraint>,
}

/// A sequence of token patterns matched over consecutive positions.
#[derive(Debug, Clone)]
pub struct CqlQuery {
    pub tokens: Vec<TokenPattern>,
}

impl CqlQuery {
    pub fn span_len(&self) -> usize {
        self.tokens.len()
    }
}

/// Intersect several ascending-sorted position vectors into one.
fn intersect_sorted(sets: &[&Vec<u32>]) -> Vec<u32> {
    let Some((first, rest)) = sets.split_first() else {
        return Vec::new();
    };
    let mut out: Vec<u32> = (*first).clone();
    for s in rest {
        let set: HashSet<u32> = s.iter().copied().collect();
        out.retain(|p| set.contains(p));
    }
    out.sort_unstable();
    out.dedup();
    out
}

impl CorpusIndex {
    /// Drive `f` over every matching document, passing the matched span
    /// start positions (token indices, ascending). Returns
    /// `(total_spans, truncated)`. Stops once `MAX_SPANS` is reached.
    ///
    /// `f` receives `(doc_id, path, body, offsets, starts)`.
    fn cql_scan(
        &self,
        query: &CqlQuery,
        allowed: Option<&HashSet<DocId>>,
        mut f: impl FnMut(DocId, &str, &str, &[u32], &[usize]),
    ) -> Result<(u64, bool)> {
        let span_len = query.span_len();
        if span_len == 0 {
            return Ok((0, false));
        }
        let searcher = self.reader.searcher();
        let mut total: u64 = 0;
        let mut truncated = false;

        'segments: for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
            // Resolve each (token, constraint) into a slot with positional
            // postings for every matching term in this segment. `slot_accs`
            // holds each slot's matched positions in the current document.
            let mut slot_accs: Vec<Vec<u32>> = Vec::new();
            let mut token_slots: Vec<Vec<usize>> = vec![Vec::new(); span_len];
            // (slot index, postings cursor)
            let mut cursors: Vec<(usize, tantivy::postings::SegmentPostings)> = Vec::new();

            for (ti, token) in query.tokens.iter().enumerate() {
                for constraint in &token.constraints {
                    let field = self.layer_field(constraint.layer);
                    let inv = seg_reader.inverted_index(field)?;
                    // Resolve the constraint to matching term texts in this
                    // segment's dictionary: Exact → the term itself (no
                    // scan); Regex → stream the dictionary, keep full matches.
                    let terms: Vec<String> = match &constraint.matcher {
                        Matcher::Exact(t) => vec![t.clone()],
                        Matcher::Regex(re) => {
                            let mut out = Vec::new();
                            let mut stream = inv.terms().stream()?;
                            while stream.advance() {
                                if let Ok(key) = std::str::from_utf8(stream.key())
                                    && !key.is_empty()
                                    && re.is_match(key)
                                {
                                    out.push(key.to_string());
                                }
                            }
                            out
                        }
                    };
                    let slot_id = slot_accs.len();
                    let mut any = false;
                    for term in &terms {
                        let term_obj = Term::from_field_text(field, term);
                        if let Some(postings) =
                            inv.read_postings(&term_obj, IndexRecordOption::WithFreqsAndPositions)?
                        {
                            cursors.push((slot_id, postings));
                            any = true;
                        }
                    }
                    // A constraint with no present terms can never hold →
                    // its token never matches → no hits in this segment.
                    if !any {
                        continue 'segments;
                    }
                    slot_accs.push(Vec::new());
                    token_slots[ti].push(slot_id);
                }
            }
            if cursors.is_empty() {
                continue 'segments;
            }

            let doc_id_col = seg_reader.fast_fields().u64("doc_id").ok();
            let mut posbuf: Vec<u32> = Vec::new();

            loop {
                // Smallest doc id any cursor is currently positioned at.
                let mut min_doc = TERMINATED;
                for (_, c) in &cursors {
                    let d = c.doc();
                    if d != TERMINATED && d < min_doc {
                        min_doc = d;
                    }
                }
                if min_doc == TERMINATED {
                    break;
                }

                let stored_id = match &doc_id_col {
                    Some(col) => col.first(min_doc).unwrap_or(0),
                    None => self.stored_doc_id(&searcher, seg_ord, min_doc)?,
                };
                let keep = allowed.is_none_or(|set| set.contains(&stored_id));

                if !keep {
                    for (_, c) in &mut cursors {
                        if c.doc() == min_doc {
                            c.advance();
                        }
                    }
                    continue;
                }

                for a in &mut slot_accs {
                    a.clear();
                }
                for (slot, c) in &mut cursors {
                    if c.doc() == min_doc {
                        posbuf.clear();
                        c.positions(&mut posbuf);
                        slot_accs[*slot].extend_from_slice(&posbuf);
                        c.advance();
                    }
                }

                // Per-token positions = intersection of its constraint slots.
                let token_pos: Vec<Vec<u32>> = (0..span_len)
                    .map(|ti| {
                        let refs: Vec<&Vec<u32>> =
                            token_slots[ti].iter().map(|&s| &slot_accs[s]).collect();
                        if refs.len() == 1 {
                            let mut v = refs[0].clone();
                            v.sort_unstable();
                            v.dedup();
                            v
                        } else {
                            intersect_sorted(&refs)
                        }
                    })
                    .collect();

                // Sequence match: p in token_pos[0] with p+i in token_pos[i].
                let starts: Vec<usize> = if span_len == 1 {
                    token_pos[0].iter().map(|&p| p as usize).collect()
                } else {
                    let later: Vec<HashSet<u32>> = token_pos[1..]
                        .iter()
                        .map(|v| v.iter().copied().collect())
                        .collect();
                    token_pos[0]
                        .iter()
                        .filter(|&&p0| {
                            (1..span_len).all(|i| later[i - 1].contains(&(p0 + i as u32)))
                        })
                        .map(|&p| p as usize)
                        .collect()
                };

                if !starts.is_empty() {
                    let doc_addr = DocAddress::new(seg_ord as u32, min_doc);
                    let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
                    let body = retrieved
                        .get_first(self.fields.body)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let path = retrieved
                        .get_first(self.fields.path)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let offsets = bytes_to_offsets(
                        retrieved
                            .get_first(self.fields.token_offsets)
                            .and_then(|v| v.as_bytes())
                            .unwrap_or_default(),
                    );
                    f(stored_id, path, body, &offsets, &starts);
                    total += starts.len() as u64;
                    if total >= MAX_SPANS {
                        truncated = true;
                        break 'segments;
                    }
                }
            }
        }

        Ok((total, truncated))
    }

    /// [`Self::kwic`] for a parsed CQL query. Hits are spans of
    /// `query.span_len()` tokens; `total` counts all spans (subcorpus +
    /// ceiling aware).
    pub fn cql_kwic(
        &self,
        query: &CqlQuery,
        context: usize,
        limit: usize,
        offset: usize,
        filter: Option<&DocFilter>,
    ) -> Result<KwicPage> {
        let allowed = self.resolve_filter(filter)?;
        let span_len = query.span_len();
        let page_end = offset.saturating_add(limit);
        let mut hits: Vec<KwicHit> = Vec::with_capacity(limit.min(512));
        let mut idx: usize = 0;
        let (total, _truncated) = self.cql_scan(
            query,
            allowed.as_ref(),
            |doc_id, path, body, offsets, starts| {
                for &p in starts {
                    let gi = idx;
                    idx += 1;
                    if gi < offset || gi >= page_end {
                        continue;
                    }
                    if let Some((left, hit, right)) =
                        window_span(body, offsets, p, span_len, context)
                    {
                        hits.push(KwicHit {
                            doc_id,
                            path: std::path::PathBuf::from(path),
                            hit_position: p,
                            left,
                            hit,
                            right,
                        });
                    }
                }
            },
        )?;
        Ok(KwicPage {
            hits,
            total: total as usize,
        })
    }

    /// [`Self::collocate_counts`] for a CQL query. The matched span is the
    /// node: collocates are counted left of the span start and right of the
    /// span end.
    pub fn cql_collocate_counts(
        &self,
        query: &CqlQuery,
        left: usize,
        right: usize,
        filter: Option<&DocFilter>,
    ) -> Result<CollocateScan> {
        let allowed = self.resolve_filter(filter)?;
        let span_len = query.span_len();
        let mut counts: HashMap<String, (u32, u32)> = HashMap::new();
        let mut window_tokens: u64 = 0;
        let (node_freq, truncated) = self.cql_scan(
            query,
            allowed.as_ref(),
            |_doc, _path, body, offsets, starts| {
                for &p in starts {
                    let last = p + span_len - 1;
                    for w in window_side(body, offsets, p, left, Side::Left) {
                        counts.entry(w).or_default().0 += 1;
                        window_tokens += 1;
                    }
                    for w in window_side(body, offsets, last, right, Side::Right) {
                        counts.entry(w).or_default().1 += 1;
                        window_tokens += 1;
                    }
                }
            },
        )?;
        Ok(CollocateScan {
            node_freq,
            counts,
            window_tokens,
            truncated,
        })
    }

    /// [`Self::collocate_by_distance`] for a CQL query. Distances are
    /// measured from the span: `-k` is the token `k` before the span start,
    /// `+k` the token `k` after the span end.
    pub fn cql_collocate_by_distance(
        &self,
        query: &CqlQuery,
        left: usize,
        right: usize,
        filter: Option<&DocFilter>,
    ) -> Result<DistanceProfile> {
        let allowed = self.resolve_filter(filter)?;
        let span_len = query.span_len();
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
        let slots = left + right;
        let mut rows: HashMap<String, Vec<u32>> = HashMap::new();
        let (node_freq, truncated) = self.cql_scan(
            query,
            allowed.as_ref(),
            |_doc, _path, body, offsets, starts| {
                let n = offsets.len();
                let start_of = |i: usize| -> usize {
                    if i < n {
                        offsets[i] as usize
                    } else {
                        body.len()
                    }
                };
                for &p in starts {
                    let last = p + span_len - 1;
                    // left of the span start
                    for k in 1..=left {
                        if p < k {
                            break;
                        }
                        let i = p - k;
                        if let Some(w) = normalize_token(&body[start_of(i)..start_of(i + 1)]) {
                            rows.entry(w).or_insert_with(|| vec![0u32; slots])
                                [slot_of(-(k as i32))] += 1;
                        }
                    }
                    // right of the span end
                    for k in 1..=right {
                        let i = last + k;
                        if i >= n {
                            break;
                        }
                        if let Some(w) = normalize_token(&body[start_of(i)..start_of(i + 1)]) {
                            rows.entry(w).or_insert_with(|| vec![0u32; slots])
                                [slot_of(k as i32)] += 1;
                        }
                    }
                }
            },
        )?;
        Ok(DistanceProfile {
            node_freq,
            offsets: offsets_axis,
            rows,
            truncated,
        })
    }

    /// [`Self::term_distribution`] for a CQL query: per-document span counts
    /// plus a dispersion histogram over the global position axis. The axis
    /// stays whole-corpus; only matching documents contribute.
    pub fn cql_term_distribution(
        &self,
        query: &CqlQuery,
        buckets: usize,
        filter: Option<&DocFilter>,
    ) -> Result<TermDistribution> {
        let allowed = self.resolve_filter(filter)?;
        let buckets = buckets.max(1);
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
        self.cql_scan(
            query,
            allowed.as_ref(),
            |doc_id, _path, _body, _offsets, starts| {
                *doc_hits.entry(doc_id).or_default() += starts.len() as u64;
                total_hits += starts.len() as u64;
                let base = *global_start.get(&doc_id).unwrap_or(&0);
                for &p in starts {
                    let global_pos = base + p as u64;
                    let b = ((global_pos * buckets as u64) / total_tokens) as usize;
                    dispersion[b.min(buckets - 1)] += 1;
                }
            },
        )?;

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

        Ok(TermDistribution {
            doc_counts,
            dispersion,
            total_hits,
        })
    }
}
