//! Pure-Rust TreeTagger.
//!
//! Reads Helmut Schmid's `.par` parameter files directly and performs
//! POS / lemma tagging in-process — no subprocess, no Perl, no bundled
//! platform binary. The published algorithm (Schmid 1994/1995) combined
//! with a reverse-engineered format reader gives us bit-for-bit parity
//! with `tree-tagger -token -lemma` while eliminating the per-document
//! model-reload tax and the IPC plumbing around it.
//!
//! The crate is layered so each archaeology step is testable in isolation:
//!
//! ```text
//! par::load ──► Model { header, lexicon, suffix, prefix, dtree }
//!                   │
//!                   ▼
//! Tagger ──► tokenize ──► candidate lookup ──► viterbi ──► AnnotatedToken
//! ```
//!
//! At the time of writing only the header reader is implemented. The
//! remaining sections land one by one, each gated by a differential
//! test against the reference `tree-tagger` binary.

pub mod par;
pub mod testkit;
pub mod viterbi;

/// Re-export the Perl-compatible tokenizer so callers that depend on
/// `corpust-tagger` don't also need to pull in `corpust-tokenize`.
pub use corpust_tokenize::treetagger as tokenize;

use anyhow::Result;
use corpust_annotate::{AnnotatedToken, Annotator};
use corpust_core::Position;
use std::borrow::Cow;
use std::path::Path;

/// In-process TreeTagger.
///
/// Constructed from a `.par` file; the loaded [`Model`] is immutable and
/// cheap to share across rayon workers via a normal `&Tagger` borrow.
///
/// **Current mode: dtree-driven bigram Viterbi.** Tokens are
/// tokenized via `corpust_tokenize::treetagger::Tokenizer` and the
/// per-token candidate list comes from either the lexicon (known
/// words) or the suffix trie's full distribution + capitalization
/// boost (unknown words). `viterbi::tag_sequence` then picks the
/// best path under  `argmax_t P(t | w) × P(t | ctx)`  with the
/// dtree-confidence pruning trick (see viterbi module docs).
/// Models without a usable dtree degrade to per-token lexicon
/// argmax with no context.
pub struct Tagger {
    model: par::Model,
    /// Cached forest + entry-root for the dtree. `None` when the
    /// model has no dtree or reconstruction failed; in either case
    /// tagging falls back to lexicon-only argmax.
    dtree: Option<par::dtree::Traversal>,
    tokenizer: tokenize::Tokenizer,
    language: &'static str,
    id: String,
}

impl Tagger {
    /// Load a `.par` file and wrap it behind the [`Annotator`] trait.
    ///
    /// `abbreviations` is the list of multi-token strings the tokenizer
    /// should keep together (e.g. `"Mr."`, `"U.S.A."`). Pass
    /// `std::iter::empty()` for a bare tokenizer — that's fine for
    /// simple text and for toy-model testing, but degrades real-world
    /// TreeTagger parity on proper nouns and acronyms.
    pub fn load(
        path: impl AsRef<Path>,
        language: &'static str,
        abbreviations: impl IntoIterator<Item = String>,
    ) -> Result<Self> {
        let model = par::load(path.as_ref())?;
        let dtree = model.dtree.as_ref().and_then(|dt| dt.traversal().ok());
        Ok(Self {
            model,
            dtree,
            tokenizer: tokenize::Tokenizer::new(abbreviations),
            language,
            id: format!("treetagger-rs-{language}"),
        })
    }

    /// Access the loaded parameter model — mostly for tests and tooling
    /// that wants to introspect the reverse-engineered structure.
    pub fn model(&self) -> &par::Model {
        &self.model
    }

    /// Build the candidate list for one token: lexicon entries when
    /// known, otherwise the unknown-word distribution from the
    /// suffix trie (with prefix trie / dtree default as fallbacks
    /// and a capitalized → NP boost). Lemma is pre-resolved.
    fn candidates_for(&self, word: &str) -> Vec<viterbi::Cand> {
        if let Some(entry) = self.model.lexicon.lookup(word) {
            if !entry.candidates.is_empty() {
                return entry
                    .candidates
                    .iter()
                    .map(|c| viterbi::Cand {
                        tag_id: c.tag_id,
                        lex_prob: c.prob as f64,
                        lemma: self.model.lexicon.lemma(c.lemma_index).map(str::to_owned),
                    })
                    .collect();
            }
        }
        self.unknown_word_candidates(word)
    }

    /// Distribution of plausible tags for a word missing from the
    /// lexicon. Pulls the suffix-trie distribution (or the prefix
    /// trie if suffix has no match), then for "Capitalized"
    /// (mixed-case, first-letter-upper) words ensures NP appears
    /// with non-trivial weight — TreeTagger's default-to-proper-
    /// noun convention for unknown capitalized tokens.
    fn unknown_word_candidates(&self, word: &str) -> Vec<viterbi::Cand> {
        let chars: Vec<char> = word.chars().collect();
        let first_upper = chars.first().copied().map(|c| c.is_uppercase()).unwrap_or(false);
        let rest_lower = chars.iter().skip(1).all(|c| !c.is_uppercase());
        // Mixed-case "Capitalized" words (first letter upper, rest
        // lower) get an NP boost — TreeTagger treats those as
        // candidates for proper noun. ALL-CAPS words tend to be
        // headings ("BOOK") tagged as NN/JJ rather than NP, so we
        // exclude them from the boost.
        let np_candidate = first_upper && rest_lower;

        let trie_dist = self.model.tries.as_ref().and_then(|tries| {
            tries
                .suffix
                .lookup(word.chars().rev())
                .or_else(|| tries.prefix.lookup(word.chars()))
        });

        let mut cands: Vec<viterbi::Cand> = match trie_dist {
            Some(d) => d
                .probs
                .iter()
                .map(|tp| viterbi::Cand {
                    tag_id: tp.tag_id as u32,
                    lex_prob: tp.prob as f64,
                    lemma: None,
                })
                .collect(),
            None => Vec::new(),
        };

        // Capitalization heuristic: for "Mixed-case" words,
        // boost NP to a meaningful share so context-aware Viterbi
        // can still pick it. Without this, suffix-trie alone tags
        // "Accolon" as VVN (the -lon ending) regardless of context.
        if np_candidate {
            if let Some(np) = self.tag_id_by_name("NP") {
                let np_id = u32::from(np);
                let np_boost = 0.7_f64;
                let scale = (1.0 - np_boost).max(0.0);
                for c in cands.iter_mut() {
                    c.lex_prob *= scale;
                }
                if let Some(c) = cands.iter_mut().find(|c| c.tag_id == np_id) {
                    c.lex_prob += np_boost;
                } else {
                    cands.push(viterbi::Cand {
                        tag_id: np_id,
                        lex_prob: np_boost,
                        lemma: None,
                    });
                }
            }
        }

        if cands.is_empty() {
            // Last resort: dtree Default leaf or SENT as a sentinel.
            let fallback = self
                .model
                .dtree
                .as_ref()
                .and_then(|dt| {
                    dt.default()
                        .distribution
                        .probs
                        .iter()
                        .max_by(|a, b| {
                            a.prob
                                .partial_cmp(&b.prob)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|tp| tp.tag_id)
                })
                .unwrap_or(self.model.header.sent_tag_index as u32);
            cands.push(viterbi::Cand {
                tag_id: fallback,
                lex_prob: 1.0,
                lemma: None,
            });
        }
        cands
    }

    fn tag_id_by_name(&self, name: &str) -> Option<u8> {
        self.model.header.tag_id(name).and_then(|v| u8::try_from(v).ok())
    }
}

impl Annotator for Tagger {
    fn annotate<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        let tokens = self.tokenizer.tokenize(text);
        let cands: Vec<Vec<viterbi::Cand>> =
            tokens.iter().map(|t| self.candidates_for(t)).collect();
        let tagged: Vec<viterbi::Tagged> = match self.dtree.as_ref() {
            Some(traversal) => viterbi::tag_sequence(&cands, traversal, &self.model.header),
            None => cands
                .iter()
                .map(|cs| {
                    let best = cs.iter().max_by(|a, b| {
                        a.lex_prob
                            .partial_cmp(&b.lex_prob)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    match best {
                        Some(c) => viterbi::Tagged {
                            pos: self.model.header.tag(c.tag_id).map(str::to_owned),
                            lemma: c.lemma.clone(),
                        },
                        None => viterbi::Tagged {
                            pos: None,
                            lemma: None,
                        },
                    }
                })
                .collect(),
        };
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(tokens.len());
        for (position, (tok, t)) in tokens.into_iter().zip(tagged).enumerate() {
            let (start, end) = match text[cursor..].find(tok.as_str()) {
                Some(off) => {
                    let s = cursor + off;
                    let e = (s + tok.len()).min(text.len());
                    (s, e)
                }
                None => (cursor, cursor),
            };
            out.push(AnnotatedToken {
                word: Cow::Owned(tok),
                lemma: t.lemma.map(Cow::Owned),
                pos: t.pos.map(Cow::Owned),
                byte_start: start,
                byte_end: end,
                position: position as Position,
            });
            cursor = end;
        }
        Ok(out)
    }

    fn supported_languages(&self) -> &[&'static str] {
        std::slice::from_ref(&self.language)
    }

    fn id(&self) -> &str {
        &self.id
    }
}

// Keep the trait object usable inside the indexer's
// `Option<&(dyn Annotator + Sync)>` signature.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tagger>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn bundle_path() -> Option<PathBuf> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("resources/treetagger");
        p.exists().then_some(p)
    }

    fn english_abbreviations() -> Vec<String> {
        let p = bundle_path().unwrap().join("lib/english-abbreviations");
        if !p.exists() {
            return Vec::new();
        }
        std::fs::read_to_string(&p)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| {
                let t = l.trim();
                (!t.is_empty() && !t.starts_with('#')).then(|| t.to_owned())
            })
            .collect()
    }

    /// Baseline mode sanity check: the Rust Tagger produces non-empty
    /// (tag, lemma) pairs for a short English sentence.
    #[test]
    fn baseline_produces_tagged_stream() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let tagger = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let tokens = tagger.annotate("The quick brown fox.").unwrap();
        assert_eq!(tokens.len(), 5, "expected 5 tokens (The quick brown fox .)");
        for t in &tokens {
            assert!(t.pos.is_some(), "{}: should have a POS tag", t.word);
            assert!(t.lemma.is_some(), "{}: should have a lemma", t.word);
        }
    }

    /// Speed bench — pure-Rust `Tagger` vs the subprocess Oracle on
    /// the same Gutenberg sample. Loads both once, warms up, then
    /// measures per-call wall clock over N iterations on the same
    /// text. `#[ignore]` because it runs the subprocess N+1 times
    /// and is expensive.
    ///
    /// Run with `cargo test -p corpust-tagger --release --lib
    /// tests::speed_bench -- --nocapture --ignored`.
    #[test]
    #[ignore]
    fn speed_bench() {
        use std::time::Instant;
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap()
            .chars().take(10_000).collect();
        let token_estimate = sample.split_whitespace().count();

        let t0 = Instant::now();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let oracle_load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let subject_load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        eprintln!("Load time:");
        eprintln!("  oracle  (subprocess, no persistent state): {oracle_load_ms:7.2} ms");
        eprintln!("  subject (in-process, loads .par):          {subject_load_ms:7.2} ms");

        // Warm up (JIT caches, disk caches).
        let _ = oracle.annotate(&sample).unwrap();
        let _ = subject.annotate(&sample).unwrap();

        let iterations = 5;
        let t0 = Instant::now();
        let mut tokens_o = 0;
        for _ in 0..iterations {
            tokens_o = oracle.annotate(&sample).unwrap().len();
        }
        let oracle_total = t0.elapsed();

        let t0 = Instant::now();
        let mut tokens_s = 0;
        for _ in 0..iterations {
            tokens_s = subject.annotate(&sample).unwrap().len();
        }
        let subject_total = t0.elapsed();

        let o_per = oracle_total.as_secs_f64() * 1000.0 / iterations as f64;
        let s_per = subject_total.as_secs_f64() * 1000.0 / iterations as f64;
        let speedup = o_per / s_per;

        eprintln!();
        eprintln!("Per-call .annotate() on {} tokens (~{} whitespace words):", tokens_o, token_estimate);
        eprintln!("  oracle:   {o_per:8.2} ms/call  ({:.2} tokens/ms)", tokens_o as f64 / o_per);
        eprintln!("  subject:  {s_per:8.2} ms/call  ({:.2} tokens/ms)", tokens_s as f64 / s_per);
        eprintln!("  speedup:  {speedup:.1}× (pure-Rust vs spawn-per-call subprocess)");
    }

    /// Sample 20 remaining unknown-word POS errors so we can see the
    /// kind of word where the suffix-trie guess still disagrees with
    /// the oracle.
    #[test]
    #[ignore]
    fn unknown_error_clustering() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap().chars().take(10_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let report = testkit::diff(&oracle, &subject, &sample).unwrap();

        let mut errors: Vec<_> = report.mismatches.iter()
            .filter(|m| m.kind == testkit::MismatchKind::Pos
                && subject.model.lexicon.lookup(&m.subject_word).is_none())
            .collect();
        errors.sort_by(|a, b| a.subject_word.cmp(&b.subject_word));
        eprintln!("Unknown-word POS errors (showing first 25 of {}):", errors.len());
        for m in errors.iter().take(25) {
            let first_char = m.subject_word.chars().next().unwrap_or('?');
            let cap = if first_char.is_uppercase() { "CAP" } else { "   " };
            eprintln!("  {} {:<15} oracle={:<5} subject={}",
                cap, m.subject_word,
                m.oracle_pos.as_deref().unwrap_or("-"),
                m.subject_pos.as_deref().unwrap_or("-"));
        }
        let cap_errors = errors.iter().filter(|m| m.subject_word.chars().next().unwrap_or('?').is_uppercase()).count();
        eprintln!("\nOf {} unknown-word POS errors: {} are capitalized, {} are not",
            errors.len(), cap_errors, errors.len() - cap_errors);
    }

    /// Dump the top-20 words responsible for ambiguous-known-word POS
    /// errors on the gutenberg sample. Lets us see whether the
    /// remaining gap is concentrated on a few high-frequency words
    /// or spread thin.
    #[test]
    #[ignore]
    fn ambiguous_error_clustering() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap().chars().take(10_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let report = testkit::diff(&oracle, &subject, &sample).unwrap();

        use std::collections::HashMap;
        let mut by_word: HashMap<(String, String, String), usize> = HashMap::new();
        for m in &report.mismatches {
            if m.kind != testkit::MismatchKind::Pos { continue }
            let n_cand = subject.model.lexicon.lookup(&m.subject_word)
                .map(|e| e.candidates.len()).unwrap_or(0);
            if n_cand <= 1 { continue }
            let key = (
                m.subject_word.clone(),
                m.oracle_pos.clone().unwrap_or_default(),
                m.subject_pos.clone().unwrap_or_default(),
            );
            *by_word.entry(key).or_insert(0) += 1;
        }
        let mut pairs: Vec<_> = by_word.into_iter().collect();
        pairs.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        eprintln!("Top ambiguous-known mismatches (word, oracle_pos, subject_pos, count):");
        for ((w, op, sp), c) in pairs.iter().take(20) {
            eprintln!("  {c:>3}× {w:<15} oracle={op:<6} subject={sp}");
        }
        let total: usize = pairs.iter().map(|(_, c)| c).sum();
        eprintln!("Total ambiguous-known POS errors: {total} across {} distinct (word, pos-diff) tuples", pairs.len());
    }

    /// Same #[ignored] baseline but broken down by error source
    /// (unknown vs ambiguous-known) so we can see where the remaining
    /// POS errors live. Informs where to invest next: trie-based
    /// unknown-word guessing vs dtree-based context disambiguation.
    #[test]
    #[ignore]
    fn baseline_error_breakdown_on_gutenberg_sample() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap().chars().take(10_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let report = testkit::diff(&oracle, &subject, &sample).unwrap();

        let mut unknown_pos_err = 0;
        let mut ambig_known_pos_err = 0;
        let mut single_known_pos_err = 0;
        for m in &report.mismatches {
            if m.kind != testkit::MismatchKind::Pos { continue }
            match subject.model.lexicon.lookup(&m.subject_word) {
                None => unknown_pos_err += 1,
                Some(entry) => {
                    if entry.candidates.len() <= 1 {
                        single_known_pos_err += 1;
                    } else {
                        ambig_known_pos_err += 1;
                    }
                }
            }
        }
        eprintln!(
            "POS-error breakdown on {}-token sample (total POS err = {}):\n\
             \t{} unknown-word errors (fixable by trie prob linkage)\n\
             \t{} ambiguous known-word errors (fixable by dtree Viterbi)\n\
             \t{} single-candidate known errors (impossible without context or bug)",
            report.oracle_tokens, report.pos_errors(),
            unknown_pos_err, ambig_known_pos_err, single_known_pos_err,
        );
    }

    /// Larger-corpus accuracy snapshot — ignored by default because it
    /// runs the subprocess Oracle over a ~10 KB Gutenberg sample. Run
    /// with `cargo test -p corpust-tagger --lib -- --nocapture --ignored`
    /// to print the number; useful when validating a proposed Viterbi
    /// change against the current lexicon-first baseline.
    #[test]
    #[ignore]
    fn baseline_vs_oracle_on_gutenberg_sample() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() {
            return;
        }
        let full = std::fs::read_to_string(&text_path).unwrap();
        let sample: String = full.chars().take(10_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let report = testkit::diff(&oracle, &subject, &sample).unwrap();
        eprintln!(
            "gutenberg sample: {} oracle / {} subject tokens, {} exact, \
             {} word-err, {} POS-err, {} lemma-err, pos_acc={:.4}",
            report.oracle_tokens,
            report.subject_tokens,
            report.matches,
            report.word_errors(),
            report.pos_errors(),
            report.lemma_errors(),
            report.pos_accuracy()
        );
    }

    /// Probe specific lexicon entries to debug Viterbi mistakes.
    #[test]
    #[ignore]
    fn lexicon_probe_words() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let model = par::load(&par).unwrap();
        for word in ["King", "king", "How", "how", "I", "Table", "table"] {
            match model.lexicon.lookup(word) {
                Some(entry) => {
                    eprintln!("=== {word:<10} ({} candidates) ===", entry.candidates.len());
                    for c in &entry.candidates {
                        let name = model.header.tag(c.tag_id).unwrap_or("?");
                        let lemma = model.lexicon.lemma(c.lemma_index).unwrap_or("?");
                        eprintln!(
                            "  {:>2} {:>5}  P = {:.4}  lemma={lemma:?}",
                            c.tag_id, name, c.prob
                        );
                    }
                }
                None => eprintln!("=== {word:<10} NOT IN LEXICON ==="),
            }
        }
    }

    /// Diff the pure-Rust lexicon-first tagger against the subprocess
    /// oracle on a short English sample. We don't expect parity yet —
    /// but the POS-accuracy number is the baseline we have to beat
    /// once Viterbi + decision-tree traversal land.
    #[test]
    fn baseline_vs_oracle_on_english_sample() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();

        let sample = "The quick brown fox jumps over the lazy dog. \
                      She sells seashells by the seashore. \
                      A rose by any other name would smell as sweet.";

        let report = testkit::diff(&oracle, &subject, sample).unwrap();
        let total = report.matches + report.mismatches.len();
        let pos_acc = report.pos_accuracy();
        eprintln!(
            "lexicon-first vs oracle on {} aligned tokens: \
             {} exact, {} word-mismatches, {} POS, {} lemma; pos_accuracy={:.3}",
            total,
            report.matches,
            report.word_errors(),
            report.pos_errors(),
            report.lemma_errors(),
            pos_acc
        );
        // Very loose correctness floor: we should at least produce
        // tokens at comparable count and get *some* matches. The
        // precise accuracy is informational — no hard floor until
        // the Viterbi path lands.
        assert_eq!(report.oracle_tokens, report.subject_tokens,
            "token counts should match — if they diverge, tokenizer parity is broken");
        assert!(report.matches > 0, "at least some tokens should match exactly");
    }
}
