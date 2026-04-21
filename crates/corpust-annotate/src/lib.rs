//! POS / lemma annotation pipeline.
//!
//! A thin trait-based abstraction so different taggers (TreeTagger,
//! UDPipe, pure-Rust heuristics, …) can be swapped without touching the
//! rest of the codebase.
//!
//! Phase 0 of this crate ships only the trait and a trivial word-only
//! fallback. The TreeTagger adapter lives in its own module once
//! added — it depends on an external subprocess and opt-in resources.

use anyhow::Result;
use corpust_core::Position;
use std::borrow::Cow;

/// Maps raw text to an ordered stream of annotated tokens.
///
/// Implementations own their tokenization — they're free to split
/// contractions, merge clitics, or honor any convention the underlying
/// tagger prescribes. The caller treats positions as authoritative:
/// `token.position` is the 0-based index in the produced stream, and
/// every downstream index layer aligns to it.
pub trait Annotator: Send + Sync {
    /// Tokenize + annotate `text`. Tokens are returned in document
    /// order; `position` fields start at 0 and increase by 1.
    fn annotate<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>>;

    /// ISO 639-1 codes the annotator can handle, or `["*"]` for
    /// language-agnostic taggers.
    fn supported_languages(&self) -> &[&'static str];

    /// Stable identifier used for provenance and version comparison
    /// (e.g. `"treetagger-en-3.2"`, `"word-only"`).
    fn id(&self) -> &str;
}

/// A single token with optional linguistic annotation layers.
///
/// `byte_start` / `byte_end` are offsets into the original source text
/// — the same text that gets stored unmodified in the index so KWIC
/// context rendering stays faithful to the input.
#[derive(Debug, Clone)]
pub struct AnnotatedToken<'a> {
    pub word: Cow<'a, str>,
    pub lemma: Option<Cow<'a, str>>,
    pub pos: Option<Cow<'a, str>>,
    pub byte_start: usize,
    pub byte_end: usize,
    pub position: Position,
}

// ---------------------------------------------------------------------------
// WordOnlyAnnotator
// ---------------------------------------------------------------------------

/// Trivial annotator: Unicode word segmentation, no lemma or POS output.
/// Useful for tests and as a fallback when no real tagger is configured.
pub struct WordOnlyAnnotator;

impl Annotator for WordOnlyAnnotator {
    fn annotate<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        use unicode_segmentation::UnicodeSegmentation;
        Ok(text
            .unicode_word_indices()
            .enumerate()
            .map(|(pos, (start, word))| AnnotatedToken {
                word: Cow::Borrowed(word),
                lemma: None,
                pos: None,
                byte_start: start,
                byte_end: start + word.len(),
                position: pos as Position,
            })
            .collect())
    }

    fn supported_languages(&self) -> &[&'static str] {
        &["*"]
    }

    fn id(&self) -> &str {
        "word-only"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_only_splits_on_word_boundaries_with_offsets() {
        let tokens = WordOnlyAnnotator.annotate("Hello, world!").unwrap();
        assert_eq!(tokens.len(), 2);

        assert_eq!(tokens[0].word, "Hello");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].byte_start, 0);
        assert_eq!(tokens[0].byte_end, 5);
        assert!(tokens[0].lemma.is_none());
        assert!(tokens[0].pos.is_none());

        assert_eq!(tokens[1].word, "world");
        assert_eq!(tokens[1].position, 1);
        assert_eq!(tokens[1].byte_start, 7);
        assert_eq!(tokens[1].byte_end, 12);
    }

    #[test]
    fn word_only_handles_unicode() {
        let tokens = WordOnlyAnnotator.annotate("Pražák píše").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].word, "Pražák");
        assert_eq!(tokens[1].word, "píše");
    }
}
