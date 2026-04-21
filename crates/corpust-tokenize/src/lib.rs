//! Tokenizers.
//!
//! Phase 0 ships a single Unicode-word tokenizer. Real corpus-linguistics
//! tokenization (language-aware, punctuation-aware, clitic splitting, etc.)
//! will arrive as extra implementations behind the [`Tokenizer`] trait.

use corpust_core::{Position, Token};
use unicode_segmentation::UnicodeSegmentation;

pub trait Tokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Box<dyn Iterator<Item = Token<'a>> + 'a>;
}

/// Splits on Unicode word boundaries (`unicode-segmentation::unicode_words`).
///
/// Punctuation and whitespace are dropped; words retain their original case.
pub struct UnicodeWordTokenizer;

impl Tokenizer for UnicodeWordTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Box<dyn Iterator<Item = Token<'a>> + 'a> {
        Box::new(
            text.unicode_word_indices()
                .enumerate()
                .map(|(i, (byte_start, word))| Token {
                    text: word,
                    position: i as Position,
                    byte_start,
                    byte_end: byte_start + word.len(),
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_whitespace_and_punctuation() {
        let tokens: Vec<_> = UnicodeWordTokenizer.tokenize("Hello, world!").collect();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[1].position, 1);
    }

    #[test]
    fn handles_unicode() {
        let tokens: Vec<_> = UnicodeWordTokenizer.tokenize("Pražák píše").collect();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Pražák");
        assert_eq!(tokens[1].text, "píše");
    }
}
