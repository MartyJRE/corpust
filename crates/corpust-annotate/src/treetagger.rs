//! TreeTagger subprocess adapter.
//!
//! Runs an external `tree-tagger-*` invocation (the shell wrapper that
//! chains `tokenize.pl` → `tree-tagger` for a given language), pipes raw
//! text to its stdin, reads tab-separated `word\tPOS\tlemma` output from
//! stdout, and realigns the output tokens back to byte spans in the
//! original source.
//!
//! v1 trade-offs deliberately taken:
//!
//! - **Spawn per document.** Correctness-first. The model reload
//!   (~50 MB English parameter file) is paid on every call, which is
//!   noticeable on big corpora. A long-running pooled subprocess
//!   landed in a follow-up.
//! - **Trust TreeTagger's tokenization.** The alignment forward-scan
//!   assumes output token strings appear verbatim in the source. True
//!   for TreeTagger's tokenizer on straight text; contractions split
//!   cleanly (`don't` → `do` + `n't` both literally present in the
//!   source substring). Misalignment falls back to a zero-length span
//!   at the cursor, which keeps downstream positional bookkeeping
//!   intact at the cost of a slightly wrong byte window on that token.

use crate::{AnnotatedToken, Annotator};
use anyhow::{Context, Result, bail};
use corpust_core::Position;
use std::borrow::Cow;
use std::io::Write;
use std::process::{Command, Stdio};

/// External TreeTagger invocation.
pub struct TreeTagger {
    command: String,
    args: Vec<String>,
    language: &'static str,
    id: String,
}

impl TreeTagger {
    /// Configure an adapter that spawns `command` with `args` as its
    /// TreeTagger pipeline. `language` is an ISO 639-1 code recorded in
    /// the annotator's id for provenance.
    ///
    /// Typical invocations:
    ///
    /// - `TreeTagger::new("tree-tagger-english", &[], "en")`
    ///   — uses the bundled shell wrapper.
    /// - `TreeTagger::new("tree-tagger", &["-token", "-lemma", "english.par"], "en")`
    ///   — direct invocation, no wrapper.
    pub fn new(command: impl Into<String>, args: &[&str], language: &'static str) -> Self {
        let command = command.into();
        let id = format!("treetagger-{}", language);
        Self {
            command,
            args: args.iter().map(|s| s.to_string()).collect(),
            language,
            id,
        }
    }

    fn run(&self, text: &str) -> Result<Vec<RawTag>> {
        let mut child = Command::new(&self.command)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning `{}`", self.command))?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .context("TreeTagger stdin unexpectedly closed")?;
            stdin
                .write_all(text.as_bytes())
                .context("writing text to TreeTagger")?;
        }
        // Drop stdin to signal EOF.
        drop(child.stdin.take());

        let output = child
            .wait_with_output()
            .context("waiting for TreeTagger")?;

        if !output.status.success() {
            bail!(
                "TreeTagger exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let stdout = String::from_utf8(output.stdout)
            .context("TreeTagger output was not valid UTF-8")?;
        Ok(parse_tsv(&stdout))
    }
}

impl Annotator for TreeTagger {
    fn annotate<'a>(&self, text: &'a str) -> Result<Vec<AnnotatedToken<'a>>> {
        let tags = self.run(text)?;
        Ok(align_to_source(tags, text))
    }

    fn supported_languages(&self) -> &[&'static str] {
        std::slice::from_ref(&self.language)
    }

    fn id(&self) -> &str {
        &self.id
    }
}

#[derive(Debug, Clone)]
struct RawTag {
    word: String,
    pos: Option<String>,
    lemma: Option<String>,
}

fn parse_tsv(output: &str) -> Vec<RawTag> {
    output
        .lines()
        .filter_map(|line| {
            if line.trim().is_empty() {
                return None;
            }
            let mut parts = line.split('\t');
            let word = parts.next()?.to_string();
            let pos = parts.next().map(str::to_string).filter(|s| !s.is_empty());
            let lemma = parts
                .next()
                .map(str::to_string)
                .filter(|s| !s.is_empty() && s != "<unknown>");
            Some(RawTag { word, pos, lemma })
        })
        .collect()
}

fn align_to_source(tags: Vec<RawTag>, text: &str) -> Vec<AnnotatedToken<'_>> {
    let mut cursor = 0;
    let mut aligned = Vec::with_capacity(tags.len());
    for (position, tag) in tags.into_iter().enumerate() {
        let (start, end) = match text[cursor..].find(&tag.word) {
            Some(offset) => {
                let start = cursor + offset;
                let end = (start + tag.word.len()).min(text.len());
                (start, end)
            }
            None => {
                // Fall back to a zero-length span at the cursor. The
                // token keeps its inverted-index entry (useful for
                // lemma/pos queries), but its byte window is a point
                // rather than a span. Acceptable degradation; indexing
                // continues.
                (cursor, cursor)
            }
        };
        aligned.push(AnnotatedToken {
            word: Cow::Owned(tag.word),
            lemma: tag.lemma.map(Cow::Owned),
            pos: tag.pos.map(Cow::Owned),
            byte_start: start,
            byte_end: end,
            position: position as Position,
        });
        cursor = end;
    }
    aligned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tsv_handles_canonical_output() {
        let raw = "The\tDT\tthe\nquick\tJJ\tquick\nbrown\tJJ\tbrown\nfox\tNN\tfox\n.\tSENT\t.\n";
        let tags = parse_tsv(raw);
        assert_eq!(tags.len(), 5);
        assert_eq!(tags[0].word, "The");
        assert_eq!(tags[0].pos.as_deref(), Some("DT"));
        assert_eq!(tags[0].lemma.as_deref(), Some("the"));
        assert_eq!(tags[3].word, "fox");
        assert_eq!(tags[3].pos.as_deref(), Some("NN"));
    }

    #[test]
    fn parse_tsv_strips_unknown_lemma_sentinel() {
        let raw = "Xyzzy\tNNP\t<unknown>\n";
        let tags = parse_tsv(raw);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].word, "Xyzzy");
        assert!(tags[0].lemma.is_none());
    }

    #[test]
    fn align_to_source_straightforward() {
        let text = "The quick brown fox.";
        let tags = vec![
            RawTag {
                word: "The".into(),
                pos: Some("DT".into()),
                lemma: Some("the".into()),
            },
            RawTag {
                word: "quick".into(),
                pos: Some("JJ".into()),
                lemma: Some("quick".into()),
            },
            RawTag {
                word: "brown".into(),
                pos: Some("JJ".into()),
                lemma: Some("brown".into()),
            },
            RawTag {
                word: "fox".into(),
                pos: Some("NN".into()),
                lemma: Some("fox".into()),
            },
            RawTag {
                word: ".".into(),
                pos: Some("SENT".into()),
                lemma: Some(".".into()),
            },
        ];

        let aligned = align_to_source(tags, text);
        assert_eq!(aligned.len(), 5);

        assert_eq!(aligned[0].byte_start, 0);
        assert_eq!(aligned[0].byte_end, 3);
        assert_eq!(&text[aligned[0].byte_start..aligned[0].byte_end], "The");

        assert_eq!(aligned[1].byte_start, 4);
        assert_eq!(aligned[1].byte_end, 9);

        assert_eq!(aligned[3].byte_start, 16);
        assert_eq!(aligned[3].byte_end, 19);

        assert_eq!(aligned[4].byte_start, 19);
        assert_eq!(aligned[4].byte_end, 20);
    }

    #[test]
    fn align_to_source_handles_contraction_split() {
        // TreeTagger splits "don't" into "do" + "n't" — both halves
        // appear literally in the source, so the forward scan lines
        // them up contiguously.
        let text = "I don't know.";
        let tags = vec![
            RawTag {
                word: "I".into(),
                pos: Some("PP".into()),
                lemma: Some("I".into()),
            },
            RawTag {
                word: "do".into(),
                pos: Some("VVP".into()),
                lemma: Some("do".into()),
            },
            RawTag {
                word: "n't".into(),
                pos: Some("RB".into()),
                lemma: Some("not".into()),
            },
            RawTag {
                word: "know".into(),
                pos: Some("VV".into()),
                lemma: Some("know".into()),
            },
            RawTag {
                word: ".".into(),
                pos: Some("SENT".into()),
                lemma: Some(".".into()),
            },
        ];

        let aligned = align_to_source(tags, text);
        assert_eq!(aligned.len(), 5);

        // "do" at [2,4], "n't" at [4,7] — contiguous, no gap.
        assert_eq!(aligned[1].word, "do");
        assert_eq!(aligned[1].byte_start, 2);
        assert_eq!(aligned[1].byte_end, 4);

        assert_eq!(aligned[2].word, "n't");
        assert_eq!(aligned[2].byte_start, 4);
        assert_eq!(aligned[2].byte_end, 7);

        assert_eq!(aligned[2].lemma.as_deref(), Some("not"));
    }

    #[test]
    fn align_to_source_missing_token_keeps_stream_intact() {
        // Pathological: TreeTagger emitted a token that doesn't appear
        // in the source. We don't crash — we emit a zero-length span
        // and continue; positions stay intact.
        let text = "hello world";
        let tags = vec![
            RawTag {
                word: "hello".into(),
                pos: None,
                lemma: None,
            },
            RawTag {
                word: "WIDGET".into(), // not in source
                pos: None,
                lemma: None,
            },
            RawTag {
                word: "world".into(),
                pos: None,
                lemma: None,
            },
        ];

        let aligned = align_to_source(tags, text);
        assert_eq!(aligned.len(), 3);
        assert_eq!(aligned[0].byte_end, 5);
        assert_eq!(aligned[1].byte_start, aligned[1].byte_end); // zero span
        assert_eq!(aligned[2].word, "world");
        assert_eq!(aligned[2].byte_start, 6);
    }
}
