//! Corpust core types.
//!
//! Deliberately tiny and dependency-free — every other crate in the workspace
//! pulls these types, so changes here ripple. Keep it pure.

use std::path::PathBuf;

/// Unique identifier for a document inside a corpus.
pub type DocId = u64;

/// Token position within a document (0-based, in tokens — not bytes).
pub type Position = u32;

/// A document as ingested from disk, before indexing.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: DocId,
    pub path: PathBuf,
    pub text: String,
}

/// A single token produced by a [`Tokenizer`][^1].
///
/// [^1]: defined in the `corpust-tokenize` crate.
#[derive(Debug, Clone, Copy)]
pub struct Token<'a> {
    pub text: &'a str,
    pub position: Position,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_is_constructible() {
        let doc = Document {
            id: 0,
            path: PathBuf::from("a.txt"),
            text: "hello".to_string(),
        };
        assert_eq!(doc.text, "hello");
    }
}
