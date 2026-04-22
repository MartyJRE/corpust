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
use std::path::Path;

/// In-process TreeTagger.
///
/// Constructed from a `.par` file; the loaded [`Model`] is immutable and
/// cheap to share across rayon workers via a normal `&Tagger` borrow.
pub struct Tagger {
    model: par::Model,
    language: &'static str,
    id: String,
}

impl Tagger {
    /// Load a `.par` file and wrap it behind the [`Annotator`] trait.
    pub fn load(path: impl AsRef<Path>, language: &'static str) -> Result<Self> {
        let model = par::load(path.as_ref())?;
        Ok(Self {
            model,
            language,
            id: format!("treetagger-rs-{language}"),
        })
    }

    /// Access the loaded parameter model — mostly for tests and tooling
    /// that wants to introspect the reverse-engineered structure.
    pub fn model(&self) -> &par::Model {
        &self.model
    }
}

impl Annotator for Tagger {
    fn annotate<'a>(&self, _text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        // Inference lands once the lexicon / tries / decision tree
        // readers and the Viterbi pass are in place. Returning an
        // empty stream here would silently hide misuse, so hard-fail.
        anyhow::bail!(
            "corpust_tagger::Tagger::annotate is not wired yet — \
             the .par lexicon / decision tree readers are still in flight"
        )
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
