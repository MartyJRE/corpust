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
/// **Current mode of operation: lexicon-first baseline.** Tokens are
/// tokenized via `corpust_tokenize::treetagger::Tokenizer` and tagged by
/// picking the maximum-probability candidate from the `.par` lexicon.
/// For unknown words we fall back to the peak tag of the decision-tree
/// Default distribution. Context-based disambiguation via Viterbi over
/// the decision tree is **not wired yet** — sub-task 2 of
/// `pure-rust-treetagger.md` is still working out what the 3 u32 fields
/// of each Internal record actually encode. This is a known-degraded
/// correctness baseline that matches the subprocess oracle on words
/// with a single dominant candidate and diverges on ambiguous words.
pub struct Tagger {
    model: par::Model,
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
        Ok(Self {
            model,
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

    /// Choose `(pos, lemma)` for a single token in isolation. Pure
    /// lexicon lookup, with suffix-trie guessing for unknowns and a
    /// final Default-distribution fallback. No context from
    /// surrounding tokens yet.
    fn tag_token(&self, word: &str) -> (Option<String>, Option<String>) {
        if let Some(entry) = self.model.lexicon.lookup(word) {
            if let Some(best) = entry
                .candidates
                .iter()
                .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal))
            {
                let pos = self.model.header.tag(best.tag_id).map(str::to_owned);
                let lemma = self.model.lexicon.lemma(best.lemma_index).map(str::to_owned);
                return (pos, lemma);
            }
        }
        // Unknown word: walk the suffix trie (capitalized words also
        // try the prefix trie) for P(tag | affix), then fall back to
        // the dtree Default. Lemma stays `None` — matches the
        // subprocess oracle's `<unknown>` → None convention, so we
        // don't falsely disagree on lemmas.
        let peak_tag_id = self.guess_unknown_tag(word);
        let pos = peak_tag_id.and_then(|t| self.model.header.tag(t.into()).map(str::to_owned));
        (pos, None)
    }

    /// Guess the best POS tag ID for a word missing from the lexicon.
    /// Tries the suffix trie first (words we've seen with this
    /// ending), then the prefix trie for capitalized words, then the
    /// dtree Default as an unconditional prior.
    fn guess_unknown_tag(&self, word: &str) -> Option<u8> {
        let tries = self.model.tries.as_ref()?;
        // Suffix trie: iterate chars last-to-first.
        if let Some(dist) = tries.suffix.lookup(word.chars().rev()) {
            if let Some(tp) = dist.peak() {
                return Some(tp.tag_id);
            }
        }
        // Capitalized? Try prefix trie.
        let first = word.chars().next();
        if matches!(first, Some(c) if c.is_uppercase()) {
            if let Some(dist) = tries.prefix.lookup(word.chars()) {
                if let Some(tp) = dist.peak() {
                    return Some(tp.tag_id);
                }
            }
        }
        // Final fallback: peak of the dtree Default distribution.
        self.model.dtree.as_ref().and_then(|dt| {
            dt.default()
                .distribution
                .probs
                .iter()
                .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal))
                .map(|tp| tp.tag_id as u8)
        })
    }
}

impl Annotator for Tagger {
    fn annotate<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        let tokens = self.tokenizer.tokenize(text);
        // Align each produced token back to its source span by
        // forward-searching. Same strategy as
        // `corpust_annotate::treetagger::align_to_source` — kept
        // in-crate here since the shared extraction hasn't landed.
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(tokens.len());
        for (position, tok) in tokens.into_iter().enumerate() {
            let (pos, lemma) = self.tag_token(&tok);
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
                lemma: lemma.map(Cow::Owned),
                pos: pos.map(Cow::Owned),
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
