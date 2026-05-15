//! Differential-test scaffolding: pit any [`Annotator`] against the
//! subprocess `tree-tagger` binary as a ground truth.
//!
//! The goal is to give any tagger-under-development a one-call
//! "does it match LancsBox-equivalent output?" check. The oracle here
//! is just [`corpust_annotate::treetagger::TreeTagger`] behind a
//! named type, kept in `corpust-tagger` because
//! (a) every piece of the pure-Rust tagger lives in this crate, and
//! (b) we want the diff helpers available to the crate's own
//! integration tests without an extra dev-dep dance.
//!
//! Usage sketch:
//!
//! ```ignore
//! use corpust_tagger::testkit::{Oracle, diff};
//! let oracle = Oracle::from_bundle("./resources/treetagger", "english")?;
//! let report = diff(&oracle, &my_tagger, corpus_text)?;
//! assert!(report.is_exact(), "{} mismatches: {:#?}", report.mismatches.len(), &report.mismatches[..5]);
//! ```

use anyhow::{Context, Result};
use corpust_annotate::treetagger::TreeTagger;
use corpust_annotate::{AnnotatedToken, Annotator};
use std::path::Path;

/// Ground-truth tagger backed by the bundled `tree-tagger` subprocess.
///
/// Output is the canonical reference every other implementation should
/// match byte-for-byte on `(word, pos, lemma)` per position.
pub struct Oracle {
    inner: TreeTagger,
}

impl Oracle {
    /// Load from the repo's bundled TreeTagger layout.
    pub fn from_bundle(bundle_root: impl AsRef<Path>, language: &'static str) -> Result<Self> {
        Ok(Self {
            inner: TreeTagger::from_bundle(bundle_root.as_ref(), language)?,
        })
    }

    /// Run the subprocess tagger on `text`, returning the annotated
    /// token stream exactly as `TreeTagger` produces it.
    pub fn tag<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        self.inner.annotate(text)
    }
}

impl Annotator for Oracle {
    fn annotate<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        self.inner.annotate(text)
    }
    fn supported_languages(&self) -> &[&'static str] {
        self.inner.supported_languages()
    }
    fn id(&self) -> &str {
        self.inner.id()
    }
}

/// A single `(word, pos, lemma)` disagreement between oracle and
/// subject annotators at the same token position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    pub position: usize,
    pub oracle_word: String,
    pub oracle_pos: Option<String>,
    pub oracle_lemma: Option<String>,
    pub subject_word: String,
    pub subject_pos: Option<String>,
    pub subject_lemma: Option<String>,
    pub kind: MismatchKind,
}

/// Which layer the mismatch lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchKind {
    /// Different word strings at the same position. Usually means the
    /// two taggers tokenized differently, so all downstream positions
    /// are misaligned too.
    Word,
    /// Same word, different POS tag.
    Pos,
    /// Same word, different lemma.
    Lemma,
    /// Same word + POS + lemma (appears in [`DiffReport`] only when
    /// `include_matches == true`).
    None,
}

/// Outcome of running [`diff`] on two [`Annotator`]s.
#[derive(Debug, Clone, Default)]
pub struct DiffReport {
    pub oracle_tokens: usize,
    pub subject_tokens: usize,
    pub matches: usize,
    pub mismatches: Vec<Mismatch>,
}

impl DiffReport {
    /// `true` when everything matches at every position.
    pub fn is_exact(&self) -> bool {
        self.mismatches.is_empty() && self.oracle_tokens == self.subject_tokens
    }

    /// Number of POS-only disagreements.
    pub fn pos_errors(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.kind == MismatchKind::Pos)
            .count()
    }

    /// Number of lemma-only disagreements.
    pub fn lemma_errors(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.kind == MismatchKind::Lemma)
            .count()
    }

    /// Number of tokenization/alignment disagreements.
    pub fn word_errors(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.kind == MismatchKind::Word)
            .count()
    }

    /// POS accuracy across the aligned positions. `1.0` when every
    /// `(word, pos)` pair matches. Returns `NaN` for empty inputs.
    pub fn pos_accuracy(&self) -> f64 {
        let aligned = self.matches + self.pos_errors() + self.lemma_errors();
        if aligned == 0 {
            f64::NAN
        } else {
            (self.matches + self.lemma_errors()) as f64 / aligned as f64
        }
    }
}

/// Run both annotators on `text`, compare token-by-token, collect
/// disagreements.
///
/// Token alignment is strictly positional: index 0 of the oracle is
/// compared against index 0 of the subject, and so on. A word-level
/// disagreement at any position doesn't trigger a realignment — it
/// gets recorded as a [`MismatchKind::Word`] and comparison continues.
/// This favors simplicity over robustness to mid-stream drift; real
/// use cases should restart comparison after a Word mismatch rather
/// than treat downstream Pos/Lemma mismatches as independent signals.
pub fn diff(oracle: &dyn Annotator, subject: &dyn Annotator, text: &str) -> Result<DiffReport> {
    let o = oracle.annotate(text).context("oracle tagging failed")?;
    let s = subject.annotate(text).context("subject tagging failed")?;

    let mut report = DiffReport {
        oracle_tokens: o.len(),
        subject_tokens: s.len(),
        matches: 0,
        mismatches: Vec::new(),
    };
    for (i, (oo, ss)) in o.iter().zip(s.iter()).enumerate() {
        let ow = oo.word.as_ref();
        let sw = ss.word.as_ref();
        let op = oo.pos.as_deref().map(str::to_owned);
        let ol = oo.lemma.as_deref().map(str::to_owned);
        let sp = ss.pos.as_deref().map(str::to_owned);
        let sl = ss.lemma.as_deref().map(str::to_owned);

        let kind = if ow != sw {
            MismatchKind::Word
        } else if op != sp {
            MismatchKind::Pos
        } else if ol != sl {
            MismatchKind::Lemma
        } else {
            MismatchKind::None
        };

        if matches!(kind, MismatchKind::None) {
            report.matches += 1;
        } else {
            report.mismatches.push(Mismatch {
                position: i,
                oracle_word: ow.to_owned(),
                oracle_pos: op,
                oracle_lemma: ol,
                subject_word: sw.to_owned(),
                subject_pos: sp,
                subject_lemma: sl,
                kind,
            });
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use corpust_annotate::{AnnotatedToken, Annotator, WordOnlyAnnotator};
    use std::borrow::Cow;
    use std::path::{Path, PathBuf};

    fn bundle_path() -> Option<PathBuf> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("resources/treetagger");
        p.exists().then_some(p)
    }

    /// Oracle compared against itself should report zero mismatches.
    #[test]
    fn oracle_vs_itself_is_exact() {
        let Some(bundle) = bundle_path() else {
            return;
        };
        let o1 = Oracle::from_bundle(&bundle, "english").unwrap();
        let o2 = Oracle::from_bundle(&bundle, "english").unwrap();
        let report = diff(&o1, &o2, "The quick brown fox jumps over the lazy dog.").unwrap();
        assert!(
            report.is_exact(),
            "oracle disagreed with itself: {report:#?}"
        );
        assert!(report.matches > 0);
    }

    /// Oracle vs a deliberately-wrong stub should produce mismatches we
    /// can count.
    #[test]
    fn oracle_vs_word_only_shows_pos_gap() {
        let Some(bundle) = bundle_path() else {
            return;
        };
        let oracle = Oracle::from_bundle(&bundle, "english").unwrap();
        let report = diff(&oracle, &WordOnlyAnnotator, "Dogs bark loudly.").unwrap();
        // WordOnlyAnnotator tokenizes differently (splits only on word
        // boundaries — no punctuation output) so we expect at least a
        // word-level misalignment.
        assert!(!report.is_exact());
        // The oracle always emits a POS; WordOnlyAnnotator never does —
        // so across aligned positions we expect 100% POS-error on
        // whatever overlap exists.
        assert!(
            report
                .mismatches
                .iter()
                .any(|m| m.kind != MismatchKind::None)
        );
    }

    /// Synthetic annotator that mirrors the oracle exactly — checks
    /// that `diff` reports `is_exact()` without needing the bundle.
    #[test]
    fn synthetic_mirror_is_exact() {
        struct Fixed(Vec<(String, String, String)>);
        impl Annotator for Fixed {
            fn annotate<'a>(&self, _text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
                Ok(self
                    .0
                    .iter()
                    .enumerate()
                    .map(|(i, (w, p, l))| AnnotatedToken {
                        word: Cow::Owned(w.clone()),
                        pos: Some(Cow::Owned(p.clone())),
                        lemma: Some(Cow::Owned(l.clone())),
                        byte_start: 0,
                        byte_end: 0,
                        position: i as u32,
                    })
                    .collect())
            }
            fn supported_languages(&self) -> &[&'static str] {
                &["*"]
            }
            fn id(&self) -> &str {
                "fixed"
            }
        }
        let pair = Fixed(vec![
            ("cat".into(), "NN".into(), "cat".into()),
            (".".into(), "SENT".into(), ".".into()),
        ]);
        let pair_copy = Fixed(vec![
            ("cat".into(), "NN".into(), "cat".into()),
            (".".into(), "SENT".into(), ".".into()),
        ]);
        let r = diff(&pair, &pair_copy, "irrelevant").unwrap();
        assert!(r.is_exact());
        assert_eq!(r.matches, 2);
        assert_eq!(r.mismatches.len(), 0);
    }

    #[test]
    fn records_pos_and_lemma_mismatches_separately() {
        struct Fixed(Vec<(String, String, String)>);
        impl Annotator for Fixed {
            fn annotate<'a>(&self, _text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
                Ok(self
                    .0
                    .iter()
                    .enumerate()
                    .map(|(i, (w, p, l))| AnnotatedToken {
                        word: Cow::Owned(w.clone()),
                        pos: Some(Cow::Owned(p.clone())),
                        lemma: Some(Cow::Owned(l.clone())),
                        byte_start: 0,
                        byte_end: 0,
                        position: i as u32,
                    })
                    .collect())
            }
            fn supported_languages(&self) -> &[&'static str] {
                &["*"]
            }
            fn id(&self) -> &str {
                "fixed"
            }
        }
        let oracle = Fixed(vec![
            ("run".into(), "VV".into(), "run".into()),
            ("went".into(), "VVD".into(), "go".into()),
        ]);
        let subj = Fixed(vec![
            ("run".into(), "NN".into(), "run".into()),    // pos differs
            ("went".into(), "VVD".into(), "went".into()), // lemma differs
        ]);
        let r = diff(&oracle, &subj, "irrelevant").unwrap();
        assert_eq!(r.pos_errors(), 1);
        assert_eq!(r.lemma_errors(), 1);
        assert_eq!(r.word_errors(), 0);
        assert!((r.pos_accuracy() - 0.5).abs() < 1e-9);
    }
}
