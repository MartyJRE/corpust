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
    /// Viterbi's relative-pruning threshold. Candidates whose
    /// `lex_prob` is below `threshold × max_lex_prob` for a given
    /// token are dropped before the dtree weighs in. `0.0` disables
    /// pruning entirely; higher values trust the lexicon more.
    pruning_threshold: f64,
    /// Override for the per-tag marginal `P(t)` used in Viterbi's
    /// Bayes-correction step. `None` falls back to
    /// `normalize_prior(tries.tag_prelude)`. Mostly for diagnostic
    /// sweeps; not normally exposed.
    tag_prior_override: Option<Vec<f64>>,
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
            pruning_threshold: 0.001,
            tag_prior_override: None,
        })
    }

    /// Access the loaded parameter model — mostly for tests and tooling
    /// that wants to introspect the reverse-engineered structure.
    pub fn model(&self) -> &par::Model {
        &self.model
    }

    /// Override the bigram-vs-prior mix weight used by the dtree
    /// ensemble in [`par::dtree::Traversal::predict_combined`]. Useful
    /// for sweeping λ during accuracy tuning. No-op when the model
    /// has no dtree.
    pub fn set_lambda_bigram(&mut self, lambda: f64) {
        if let Some(tr) = self.dtree.as_mut() {
            tr.lambda_bigram = lambda;
        }
    }

    /// Override the Viterbi pruning threshold; `0.0` disables
    /// pruning entirely, `1.0` keeps only the top-lex candidate(s).
    pub fn set_pruning_threshold(&mut self, threshold: f64) {
        self.pruning_threshold = threshold;
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
    /// lexicon. Tries (in order):
    ///
    /// 1. **Numeric pattern** — any digit anywhere → CD.
    /// 2. **Roman numeral pattern** — `^[IVXLCDM]+$` → NP.
    /// 3. **All-caps lowercase fallback** — for an all-uppercase
    ///    word with no original-case lex hit, look up the lowercase
    ///    form ("BOOK" → "book") to handle headings.
    /// 4. **Suffix trie distribution** + capitalized → NP boost.
    /// 5. Last resort: peak of the dtree Default leaf.
    fn unknown_word_candidates(&self, word: &str) -> Vec<viterbi::Cand> {
        let chars: Vec<char> = word.chars().collect();
        let first_upper = chars.first().copied().map(|c| c.is_uppercase()).unwrap_or(false);
        let all_upper = !chars.is_empty()
            && chars.iter().all(|c| !c.is_alphabetic() || c.is_uppercase());
        // Boost NP for any first-uppercase word that survives the
        // numeric / Roman / lowercase-lexicon fallbacks. Mixed-case
        // ("Pendragon") and all-caps ("WILLIAM") tokens both
        // qualify here since their lowercase forms didn't match
        // the lexicon.
        let np_candidate = first_upper;

        // 1. Digits anywhere → cardinal number.
        if chars.iter().any(|c| c.is_ascii_digit()) {
            if let Some(cd) = self.tag_id_by_name("CD") {
                return vec![viterbi::Cand {
                    tag_id: u32::from(cd),
                    lex_prob: 1.0,
                    lemma: None,
                }];
            }
        }

        // 2. Roman numerals → NP. Only triggers when every
        // character is in the Roman set; the lexicon catches "I"
        // (the pronoun) before we get here, so single-letter false
        // positives like "V" alone are vanishingly rare.
        if all_upper
            && chars.iter().all(|c| matches!(*c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
        {
            if let Some(np) = self.tag_id_by_name("NP") {
                return vec![viterbi::Cand {
                    tag_id: u32::from(np),
                    lex_prob: 1.0,
                    lemma: None,
                }];
            }
        }

        // 3. Capitalized lowercase fallback. For all-caps headings
        // ("BOOK" → "book") AND mixed-case sentence-internal
        // capitalizations ("Hundred" → "hundred", "Beast" →
        // "beast"), try the lexicon with the lowercased form. Words
        // we'd otherwise tag NP via the heuristic become plain
        // NN/CD/JJ when their lowercase is a known common-class
        // word. Skip the fallback when no letter is uppercase since
        // that means the original lookup already exhausted lexicon.
        if first_upper {
            let lc: String = word.chars().flat_map(|c| c.to_lowercase()).collect();
            if lc != word {
                if let Some(entry) = self.model.lexicon.lookup(&lc) {
                    if !entry.candidates.is_empty() {
                        return entry
                            .candidates
                            .iter()
                            .map(|c| viterbi::Cand {
                                tag_id: c.tag_id,
                                lex_prob: c.prob as f64,
                                lemma: self
                                    .model
                                    .lexicon
                                    .lemma(c.lemma_index)
                                    .map(str::to_owned),
                            })
                            .collect();
                    }
                }
            }
        }

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

/// Normalize per-tag values from the tries slab into a proper
/// probability distribution. Kept around as a building block for
/// the `compare_tag_prior_sources` diagnostic. Not used in the
/// default Viterbi path — `tag_prelude`'s values disagree with
/// real-text marginal frequencies, see `Annotator::annotate`.
#[allow(dead_code)]
fn normalize_prior(prelude: &[f64]) -> Vec<f64> {
    let total: f64 = prelude.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    prelude.iter().map(|v| v / total).collect()
}

impl Annotator for Tagger {
    fn annotate<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        let tokens = self.tokenizer.tokenize(text);
        let cands: Vec<Vec<viterbi::Cand>> =
            tokens.iter().map(|t| self.candidates_for(t)).collect();
        let tagged: Vec<viterbi::Tagged> = match self.dtree.as_ref() {
            Some(traversal) => {
                // Default prior: the dtree's averaged-leaf marginal,
                // which is the per-tag frequency the tree itself was
                // trained against. The `tries.tag_prelude` block in
                // the .par file looks like training counts at first
                // glance but emits values that disagree wildly with
                // any English-text distribution (`#`=6.8 %, `,`=0.02 %
                // on `english.par`); using it as `P(t)` shifts
                // Viterbi's argmax in the wrong direction. The dtree
                // marginal `(#=0.015 %, ,=5.2 %)` matches what we'd
                // expect from training corpora.
                let tag_prior: Vec<f64> = if let Some(ov) = &self.tag_prior_override {
                    ov.clone()
                } else {
                    traversal.marginal.probs.iter().map(|tp| tp.prob).collect()
                };
                viterbi::tag_sequence_with(
                    &cands,
                    traversal,
                    &self.model.header,
                    &tag_prior,
                    self.pruning_threshold,
                )
            }
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

    /// Sweep the dtree bigram-vs-prior mix weight `λ` on the gutenberg
    /// sample and print pos_acc for each value. Ignored by default;
    /// run with `cargo test -p corpust-tagger --lib
    /// sweep_lambda_bigram_on_gutenberg_sample --
    /// --nocapture --ignored`.
    #[test]
    #[ignore]
    fn sweep_lambda_bigram_on_gutenberg_sample() {
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
        let sample: String = full.chars().take(50_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let mut subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let lambdas = [0.00, 0.30, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00];
        for &lambda in &lambdas {
            subject.set_lambda_bigram(lambda);
            let report = testkit::diff(&oracle, &subject, &sample).unwrap();
            eprintln!(
                "λ_bigram={lambda:.2}  pos_acc={:.4}  POS-err={}",
                report.pos_accuracy(),
                report.pos_errors()
            );
        }
    }

    /// Walk the tree like `traverse_tree`, but record every
    /// predicate along the path: which back-position was tested,
    /// what tag value was checked, what the observed context tag
    /// actually was, and whether the predicate evaluated true.
    /// Returns a human-readable summary ending with the leaf's
    /// argmax. Kept around as a diagnostic helper; not currently
    /// called by any test.
    #[allow(dead_code)]
    fn trace_traversal(
        forest: &par::dtree::TreeForest,
        root_idx: usize,
        context: &[u32],
        model: &par::Model,
    ) -> String {
        let tag_name = |t: u32| {
            model
                .header
                .tags
                .get(t as usize)
                .map(String::as_str)
                .unwrap_or("?")
                .to_string()
        };
        let mut idx = root_idx;
        let mut steps: Vec<String> = Vec::new();
        loop {
            match &forest.nodes[idx] {
                par::dtree::TreeNode::Leaf { distribution, .. } => {
                    let argmax = distribution
                        .probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.prob.partial_cmp(&b.prob).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0);
                    steps.push(format!(
                        "→ leaf[w={},argmax={}]",
                        distribution.weight,
                        tag_name(argmax)
                    ));
                    return steps.join(" ");
                }
                par::dtree::TreeNode::Internal { predicate, yes, no, .. } => {
                    let back = predicate.back_pos_i as usize;
                    let observed = context.get(back).copied();
                    let hit = observed == Some(predicate.test_tag_id);
                    steps.push(format!(
                        "ctx[{}]={} ==? {} → {}",
                        back,
                        observed.map(tag_name).unwrap_or_else(|| "OOB".to_string()),
                        tag_name(predicate.test_tag_id),
                        if hit { "Y" } else { "N" }
                    ));
                    idx = if hit { *yes } else { *no };
                }
            }
        }
    }

    /// Bit-identical parity check between our parsed bigram tree
    /// and the oracle binary's per-context distribution dump (from
    /// `tree-tagger -print-prob-tree english.par`). For every
    /// `(t_-1, t_-2)` context the oracle emits, our `traverse_tree`
    /// at the inference root must produce the same argmax and
    /// near-identical probabilities.
    ///
    /// This nails down the end-to-end dtree layout: section offset
    /// (0xd231a3 — discovered via lldb tracing of `read_subtree`,
    /// see #13), 4-byte `Context_Size` header, internal predicate
    /// direction (`context[back_pos_i] == test_tag_id` with `0=t_-2`,
    /// `1=t_-1`), yes-first child order, and 12-byte distribution
    /// entries summing to ~1.0.
    ///
    /// Ignored by default — needs `/tmp/print-prob-tree.txt`:
    ///
    ///     tree-tagger -print-prob-tree english.par > /tmp/print-prob-tree.txt
    #[test]
    #[ignore]
    fn diff_bigram_tree_vs_oracle() {
        use std::collections::HashMap;
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let path = "/tmp/print-prob-tree.txt";
        if !Path::new(path).exists() {
            eprintln!("missing {path} — generate with `tree-tagger -print-prob-tree english.par`");
            return;
        }
        let raw = std::fs::read_to_string(path).unwrap();

        let model = par::load(&par).unwrap();
        let n_tags = model.header.tags.len();
        let tag_to_id: HashMap<String, u32> = model
            .header
            .tags
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u32))
            .collect();

        // Parse oracle dump (mirrors binary_prob_tree_upper_bound).
        let mut oracle_table: HashMap<(u32, u32), Vec<f64>> = HashMap::new();
        let mut cur_t1: Option<u32> = None;
        let mut cur_t2: Option<u32> = None;
        let mut cur_probs: Vec<f64> = vec![0.0; n_tags];
        let mut cur_idx = 0usize;
        let flush =
            |t1: Option<u32>, t2: Option<u32>, probs: &[f64], tbl: &mut HashMap<(u32, u32), Vec<f64>>| {
                if let (Some(t1), Some(t2)) = (t1, t2) {
                    tbl.insert((t1, t2), probs.to_vec());
                }
            };
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("tag[-1] = ") {
                flush(cur_t1, cur_t2, &cur_probs, &mut oracle_table);
                cur_t1 = tag_to_id.get(rest.trim()).copied();
                cur_t2 = None;
            } else if let Some(rest) = line.strip_prefix("\ttag[-2] = ") {
                flush(cur_t1, cur_t2, &cur_probs, &mut oracle_table);
                cur_t2 = tag_to_id.get(rest.trim()).copied();
                cur_probs = vec![0.0; n_tags];
                cur_idx = 0;
            } else if line.starts_with("\t\t") && cur_t1.is_some() && cur_t2.is_some() {
                let trimmed = line.trim();
                if let Some((_tag, prob_str)) = trimmed.rsplit_once(' ') {
                    let p: f64 = prob_str.parse().unwrap_or(0.0);
                    if cur_idx < n_tags {
                        cur_probs[cur_idx] = p;
                        cur_idx += 1;
                    }
                }
            }
        }
        flush(cur_t1, cur_t2, &cur_probs, &mut oracle_table);

        // Walk our reconstructed tree's inference root.
        let dt = model.dtree.as_ref().expect("model has no dtree");
        let traversal = dt.traversal().unwrap();
        let inference_root = traversal.root;

        let mut argmax_disagree = 0usize;
        let mut max_abs = 0.0f64;
        let mut sum_kl = 0.0f64;
        for (&(t1, t2), oracle_probs) in &oracle_table {
            let ctx = [t2, t1];
            let ours = par::dtree::traverse_tree(&traversal.forest, inference_root, &ctx);
            let mut abs_per_ctx: f64 = 0.0;
            let mut kl = 0.0f64;
            for k in 0..n_tags {
                let p_ours = ours.probs[k].prob;
                let p_oracle = oracle_probs[k];
                let d = (p_ours - p_oracle).abs();
                if d > abs_per_ctx {
                    abs_per_ctx = d;
                }
                if p_ours > 1e-12 && p_oracle > 1e-12 {
                    kl += p_oracle * (p_oracle.ln() - p_ours.ln());
                }
            }
            max_abs = max_abs.max(abs_per_ctx);
            sum_kl += kl.max(0.0);

            let argmax = |probs: &[f64]| {
                probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            let our_probs: Vec<f64> = ours.probs.iter().map(|tp| tp.prob).collect();
            if argmax(&our_probs) != argmax(oracle_probs) {
                argmax_disagree += 1;
            }
        }
        eprintln!(
            "compared {} contexts: argmax disagrees on {}, max abs-diff={:.6}, mean KL={:.6}",
            oracle_table.len(),
            argmax_disagree,
            max_abs,
            sum_kl / oracle_table.len() as f64,
        );
        assert_eq!(
            argmax_disagree, 0,
            "expected bit-identical parity with oracle's per-context distributions; \
             a drift here means the dtree byte layout or traversal has changed"
        );
        assert!(
            max_abs < 1e-5,
            "per-context probability disagreement exceeds 1e-5 (max_abs={max_abs:e})"
        );
        let _ = tag_to_id; // keep symbol live for future diagnostics
    }

    /// Compare per-token tagging output of our pipeline against the
    /// oracle-override pipeline. Find the tokens our pipeline gets
    /// wrong but the override gets right — those are the errors
    /// caused specifically by our tree disagreeing with oracle on
    /// the (t_-1, t_-2) → distribution mapping. Ignored.
    #[test]
    #[ignore]
    fn errors_we_lose_to_oracle_override() {
        use std::collections::HashMap;
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let dump_path = "/tmp/print-prob-tree.txt";
        if !Path::new(dump_path).exists() {
            return;
        }
        let raw = std::fs::read_to_string(dump_path).unwrap();
        let model = par::load(&par).unwrap();
        let n_tags = model.header.tags.len();
        let tag_to_id: HashMap<String, u32> = model
            .header
            .tags
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u32))
            .collect();
        let mut table: HashMap<(u32, u32), par::dtree::Distribution> = HashMap::new();
        let mut cur_t1: Option<u32> = None;
        let mut cur_t2: Option<u32> = None;
        let mut cur_probs: Vec<f64> = vec![0.0; n_tags];
        let mut cur_idx = 0usize;
        let flush =
            |t1: Option<u32>, t2: Option<u32>, probs: &[f64], tbl: &mut HashMap<(u32, u32), par::dtree::Distribution>| {
                if let (Some(t1), Some(t2)) = (t1, t2) {
                    tbl.insert(
                        (t1, t2),
                        par::dtree::Distribution {
                            weight: 0,
                            probs: probs
                                .iter()
                                .enumerate()
                                .map(|(i, p)| par::dtree::TagProb {
                                    tag_id: i as u32,
                                    prob: *p,
                                })
                                .collect(),
                        },
                    );
                }
            };
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("tag[-1] = ") {
                flush(cur_t1, cur_t2, &cur_probs, &mut table);
                cur_t1 = tag_to_id.get(rest.trim()).copied();
                cur_t2 = None;
            } else if let Some(rest) = line.strip_prefix("\ttag[-2] = ") {
                flush(cur_t1, cur_t2, &cur_probs, &mut table);
                cur_t2 = tag_to_id.get(rest.trim()).copied();
                cur_probs = vec![0.0; n_tags];
                cur_idx = 0;
            } else if line.starts_with("\t\t") && cur_t1.is_some() && cur_t2.is_some() {
                let trimmed = line.trim();
                if let Some((_tag, prob_str)) = trimmed.rsplit_once(' ') {
                    let p: f64 = prob_str.parse().unwrap_or(0.0);
                    if cur_idx < n_tags {
                        cur_probs[cur_idx] = p;
                        cur_idx += 1;
                    }
                }
            }
        }
        flush(cur_t1, cur_t2, &cur_probs, &mut table);

        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap()
            .chars().take(10_000).collect();

        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let ours = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let mut with_table = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        if let Some(t) = with_table.dtree.as_mut() {
            t.override_table = Some(table);
        }

        let r_ours = testkit::diff(&oracle, &ours, &sample).unwrap();
        let r_table = testkit::diff(&oracle, &with_table, &sample).unwrap();
        eprintln!("ours: {} POS-err / 2032 ({:.4})", r_ours.pos_errors(), r_ours.pos_accuracy());
        eprintln!("table: {} POS-err / 2032 ({:.4})", r_table.pos_errors(), r_table.pos_accuracy());

        // The mismatches list (token, our_pos, oracle_pos, prev_ctx)
        // tells us exactly which contexts our tree gets wrong.
        // Find tokens where ours is wrong but table is right.
        let mut errs_ours: std::collections::HashSet<usize> = Default::default();
        for m in &r_ours.mismatches {
            errs_ours.insert(m.position as usize);
        }
        let mut errs_table: std::collections::HashSet<usize> = Default::default();
        for m in &r_table.mismatches {
            errs_table.insert(m.position as usize);
        }
        let we_lose: Vec<_> = errs_ours.difference(&errs_table).collect();
        eprintln!("tokens we lose specifically vs override: {}", we_lose.len());
        for &&pos in we_lose.iter().take(20) {
            let m = r_ours.mismatches.iter().find(|m| m.position as usize == pos);
            if let Some(m) = m {
                eprintln!("  pos {}: word={:?} ours={:?} oracle={:?}", pos, m.oracle_word, m.subject_pos, m.oracle_pos);
            }
        }
    }

    /// Compare different `tag_prior` sources in Viterbi end-to-end.
    /// Ignored.
    #[test]
    #[ignore]
    fn compare_tag_prior_sources() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap()
            .chars().take(10_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();

        // Source 1: default (normalized tag_prelude).
        let tagger = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let r = testkit::diff(&oracle, &tagger, &sample).unwrap();
        eprintln!("tag_prelude (default): POS-err={} pos_acc={:.4}", r.pos_errors(), r.pos_accuracy());

        // Source 2: no prior at all.
        let mut tagger2 = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        tagger2.tag_prior_override = Some(Vec::new());
        let r = testkit::diff(&oracle, &tagger2, &sample).unwrap();
        eprintln!("no prior:              POS-err={} pos_acc={:.4}", r.pos_errors(), r.pos_accuracy());

        // Source 3: dtree-leaves marginal.
        let mut tagger3 = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let marginal: Vec<f64> = tagger3.dtree.as_ref().unwrap()
            .marginal.probs.iter().map(|tp| tp.prob).collect();
        tagger3.tag_prior_override = Some(marginal);
        let r = testkit::diff(&oracle, &tagger3, &sample).unwrap();
        eprintln!("dtree marginal:        POS-err={} pos_acc={:.4}", r.pos_errors(), r.pos_accuracy());
    }

    /// Sweep Viterbi's relative-pruning threshold over the 10 KB
    /// Gutenberg sample, printing pos_acc for each value. Ignored.
    #[test]
    #[ignore]
    fn sweep_pruning_threshold_on_gutenberg() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let full = std::fs::read_to_string(&text_path).unwrap();
        let sample: String = full.chars().take(50_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let mut subject = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let thresholds = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.75];
        for &t in &thresholds {
            subject.set_pruning_threshold(t);
            let r = testkit::diff(&oracle, &subject, &sample).unwrap();
            eprintln!(
                "prune={t:.3}  POS-err={}  pos_acc={:.4}",
                r.pos_errors(),
                r.pos_accuracy()
            );
        }
    }

    /// Dump every per-token disagreement against the oracle on the
    /// 10 KB Gutenberg sample, with enough context to categorize.
    /// Per error: word, our tag, oracle tag, whether the word is in
    /// our lexicon, our top lexical candidates, and the few preceding
    /// tokens. Categorizes by (in-lexicon? / our-pick-was-a-candidate?)
    /// so the dominant failure mode pops out.
    ///
    /// Ignored by default. Useful for end-to-end parity archaeology
    /// once the dtree side is bit-identical (see `diff_bigram_tree_vs_oracle`).
    #[test]
    #[ignore]
    fn categorize_residual_errors_on_gutenberg() {
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap()
            .chars().take(10_000).collect();

        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let ours = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        let report = testkit::diff(&oracle, &ours, &sample).unwrap();
        eprintln!(
            "ours: {} POS-err / {} ({:.4})",
            report.pos_errors(),
            report.oracle_tokens,
            report.pos_accuracy()
        );

        // For each POS mismatch, look up the word in our lexicon and
        // assemble candidate info.
        let mut by_category: std::collections::HashMap<&str, Vec<&testkit::Mismatch>> =
            Default::default();
        for m in &report.mismatches {
            if m.kind != testkit::MismatchKind::Pos {
                continue;
            }
            let word = &m.oracle_word;
            let in_lex = ours.model().lexicon.lookup(word).is_some();
            let our_pos = m.subject_pos.as_deref().unwrap_or("?");
            let oracle_pos = m.oracle_pos.as_deref().unwrap_or("?");
            let category = if !in_lex {
                "unknown-word"
            } else {
                let entry = ours.model().lexicon.lookup(word).unwrap();
                let cand_tags: Vec<&str> = entry
                    .candidates
                    .iter()
                    .filter_map(|c| ours.model().header.tag(c.tag_id))
                    .collect();
                let our_in_cands = cand_tags.iter().any(|t| *t == our_pos);
                let oracle_in_cands = cand_tags.iter().any(|t| *t == oracle_pos);
                match (our_in_cands, oracle_in_cands) {
                    (true, true) => "both-in-lex",
                    (true, false) => "we-picked-from-lex_oracle-didnt",
                    (false, true) => "we-picked-outside-lex",
                    (false, false) => "neither-in-lex",
                }
            };
            by_category.entry(category).or_default().push(m);
        }

        eprintln!("\nresidual POS errors by category:");
        let mut cats: Vec<_> = by_category.iter().collect();
        cats.sort_by_key(|(_, ms)| std::cmp::Reverse(ms.len()));
        for (cat, ms) in &cats {
            eprintln!("  {cat}: {} errors", ms.len());
        }

        eprintln!("\nfirst 15 per category:");
        for (cat, ms) in &cats {
            eprintln!("--- {cat} ---");
            for m in ms.iter().take(15) {
                let word = &m.oracle_word;
                let lex_summary = match ours.model().lexicon.lookup(word) {
                    Some(entry) => {
                        let mut cands: Vec<(String, f32)> = entry
                            .candidates
                            .iter()
                            .map(|c| {
                                (
                                    ours.model()
                                        .header
                                        .tag(c.tag_id)
                                        .map(|s| s.to_string())
                                        .unwrap_or_default(),
                                    c.prob,
                                )
                            })
                            .collect();
                        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        cands
                            .iter()
                            .take(4)
                            .map(|(t, p)| format!("{t}={p:.3}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    }
                    None => "(not in lex)".to_string(),
                };
                eprintln!(
                    "  pos={:>4} {:<18} ours={:>4} oracle={:>4}  lex=[{}]",
                    m.position,
                    format!("\"{}\"", word),
                    m.subject_pos.as_deref().unwrap_or("?"),
                    m.oracle_pos.as_deref().unwrap_or("?"),
                    lex_summary
                );
            }
        }
    }

    /// Run our Viterbi pipeline using `tree-tagger -print-prob-tree
    /// english.par` output as the dtree's prediction lookup.
    /// Bypasses our reverse-engineered tree entirely. Two outcomes:
    ///
    /// - **Accuracy jumps to ~oracle**: the formula is right, our
    ///   tree parsing is wrong (or incomplete).
    /// - **Accuracy stays similar**: the formula itself differs.
    ///
    /// Requires `/tmp/print-prob-tree.txt`:
    /// `tree-tagger -print-prob-tree english.par > /tmp/print-prob-tree.txt`.
    #[test]
    #[ignore]
    fn binary_prob_tree_upper_bound() {
        use std::collections::HashMap;
        let Some(bundle) = bundle_path() else { return };
        let par = bundle.join("lib/english.par");
        let path = "/tmp/print-prob-tree.txt";
        if !Path::new(path).exists() {
            eprintln!("missing {path}");
            return;
        }
        let raw = std::fs::read_to_string(path).unwrap();

        // Parse print-prob-tree.txt into a (tag_-1, tag_-2) → 58-tag
        // distribution map. Lines look like:
        //   "tag[-1] = NP"
        //   "\ttag[-2] = NP"
        //   "\t\t    # 0.000000"
        //   "\t\t    $ 0.000295"
        //   ...
        let m = par::load(&par).unwrap();
        let n_tags = m.header.tags.len();
        let tag_to_id: HashMap<String, u32> = m
            .header
            .tags
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u32))
            .collect();

        let mut table: HashMap<(u32, u32), par::dtree::Distribution> = HashMap::new();
        let mut cur_t1: Option<u32> = None;
        let mut cur_t2: Option<u32> = None;
        let mut cur_probs: Vec<f64> = vec![0.0; n_tags];
        let mut cur_idx = 0usize;

        let flush = |t1: Option<u32>,
                     t2: Option<u32>,
                     probs: &[f64],
                     table: &mut HashMap<(u32, u32), par::dtree::Distribution>| {
            if let (Some(t1), Some(t2)) = (t1, t2) {
                let dist_probs = probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| par::dtree::TagProb {
                        tag_id: i as u32,
                        prob: *p,
                    })
                    .collect();
                // If context is [..., t_{-2}, t_{-1}], then predict uses context.last() as t1 and context[len-2] as t2.
                // Binary output:
                // tag[-1] = A
                //   tag[-2] = B
                // So the key should be (A, B).
                table.insert(
                    (t1, t2),
                    par::dtree::Distribution {
                        weight: 0,
                        probs: dist_probs,
                    },
                );
            }
        };

        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("tag[-1] = ") {
                flush(cur_t1, cur_t2, &cur_probs, &mut table);
                cur_t1 = tag_to_id.get(rest.trim()).copied();
                cur_t2 = None;
            } else if let Some(rest) = line.strip_prefix("\ttag[-2] = ") {
                flush(cur_t1, cur_t2, &cur_probs, &mut table);
                cur_t2 = tag_to_id.get(rest.trim()).copied();
                cur_probs = vec![0.0; n_tags];
                cur_idx = 0;
            } else if line.starts_with("\t\t") && cur_t1.is_some() && cur_t2.is_some() {
                let trimmed = line.trim();
                if let Some((_tag, prob_str)) = trimmed.rsplit_once(' ') {
                    let p: f64 = prob_str.parse().unwrap_or(0.0);
                    if cur_idx < n_tags {
                        cur_probs[cur_idx] = p;
                        cur_idx += 1;
                    }
                }
            }
        }
        flush(cur_t1, cur_t2, &cur_probs, &mut table);

        eprintln!("parsed {} (tag_-1, tag_-2) → distribution entries", table.len());

        // Build a tagger with the override table installed.
        let mut tagger = Tagger::load(&par, "english", english_abbreviations()).unwrap();
        if let Some(t) = tagger.dtree.as_mut() {
            t.override_table = Some(table);
        } else {
            eprintln!("model has no dtree traversal — skipping");
            return;
        }

        // Run the diff against the oracle.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap();
        let text_path = repo.join("testdata/gutenberg/1251.txt");
        if !text_path.exists() { return }
        let sample: String = std::fs::read_to_string(&text_path).unwrap()
            .chars().take(10_000).collect();
        let oracle = testkit::Oracle::from_bundle(&bundle, "english").unwrap();
        let report = testkit::diff(&oracle, &tagger, &sample).unwrap();
        eprintln!(
            "binary-prob-tree experiment: {}/{} exact, {} POS-err, pos_acc={:.4}",
            report.matches,
            report.oracle_tokens,
            report.pos_errors(),
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
        for word in [
            "King", "king", "How", "how", "I", "Table", "table",
            "saved", "made", "slew", "have", "that",
        ] {
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
