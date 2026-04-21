//! TreeTagger subprocess adapter.
//!
//! Runs a two-stage pipeline per document:
//!
//! ```text
//! raw text ──► perl utf8-tokenize.perl -e -a english-abbreviations
//!           ──► tree-tagger -token -lemma -sgml english.par
//!           ──► word\tPOS\tlemma TSV
//! ```
//!
//! Tokenization happens in Perl (TreeTagger's own `utf8-tokenize.perl`)
//! so contractions and clitics split the way LancsBox expects. The binary
//! is platform-specific; the Perl script and language model are shared.
//! Output tokens are realigned back to byte spans in the source via a
//! forward scan so downstream positional bookkeeping stays intact.
//!
//! v1 trade-offs:
//!
//! - **Spawn per document** — both tokenizer and tagger subprocesses are
//!   created fresh for each call. The ~14 MB model reload dominates for
//!   large corpora; pooling lands in a follow-up.
//! - **Perl required** — preinstalled on macOS and Linux; Windows users
//!   need Strawberry Perl or similar. A pure-Rust port of the tokenizer
//!   would remove this dep entirely and is the eventual plan.

use crate::{AnnotatedToken, Annotator};
use anyhow::{Context, Result, bail};
use corpust_core::Position;
use std::borrow::Cow;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// TreeTagger subprocess adapter.
pub struct TreeTagger {
    tagger_binary: PathBuf,
    tokenizer_script: PathBuf,
    abbreviations_file: PathBuf,
    model_file: PathBuf,
    language: &'static str,
    id: String,
}

impl TreeTagger {
    /// Configure an adapter from explicit paths. Useful for non-bundled
    /// installations (user-installed TreeTagger, custom layouts).
    pub fn new(
        tagger_binary: impl Into<PathBuf>,
        tokenizer_script: impl Into<PathBuf>,
        abbreviations_file: impl Into<PathBuf>,
        model_file: impl Into<PathBuf>,
        language: &'static str,
    ) -> Self {
        Self {
            tagger_binary: tagger_binary.into(),
            tokenizer_script: tokenizer_script.into(),
            abbreviations_file: abbreviations_file.into(),
            model_file: model_file.into(),
            language,
            id: format!("treetagger-{language}"),
        }
    }

    /// Locate a TreeTagger installation inside the repo's bundled layout:
    ///
    /// ```text
    /// <bundle_root>/
    /// ├── bin/<platform>/tree-tagger(.exe)
    /// ├── cmd/utf8-tokenize.perl
    /// └── lib/
    ///     ├── <language>-abbreviations
    ///     └── <language>.par
    /// ```
    ///
    /// `<platform>` is one of `macos-arm64`, `macos-x86_64`,
    /// `linux-x86_64`, `windows-x86_64`. `language` is the full
    /// TreeTagger language name (`"english"`, not `"en"`).
    pub fn from_bundle(bundle_root: &Path, language: &'static str) -> Result<Self> {
        let platform = current_platform_dir()?;
        let binary_name = if cfg!(target_os = "windows") {
            "tree-tagger.exe"
        } else {
            "tree-tagger"
        };

        let tagger = bundle_root.join("bin").join(platform).join(binary_name);
        let tokenizer = bundle_root.join("cmd").join("utf8-tokenize.perl");
        let abbr = bundle_root
            .join("lib")
            .join(format!("{language}-abbreviations"));
        let model = bundle_root.join("lib").join(format!("{language}.par"));

        for p in [&tagger, &tokenizer, &abbr, &model] {
            if !p.exists() {
                bail!(
                    "TreeTagger bundle missing: {} (bundle_root = {})",
                    p.display(),
                    bundle_root.display()
                );
            }
        }

        Ok(Self::new(tagger, tokenizer, abbr, model, language))
    }

    fn run(&self, text: &str) -> Result<Vec<RawTag>> {
        let tokenized = self.tokenize(text)?;
        let tagged = self.tag(&tokenized)?;
        Ok(parse_tsv(&tagged))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u8>> {
        let mut child = Command::new("perl")
            .arg(&self.tokenizer_script)
            .arg("-e")
            .arg("-a")
            .arg(&self.abbreviations_file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning perl tokenizer")?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .context("perl stdin unexpectedly closed")?;
            stdin
                .write_all(text.as_bytes())
                .context("writing to perl tokenizer")?;
        }
        drop(child.stdin.take());

        let output = child.wait_with_output().context("waiting for perl")?;
        if !output.status.success() {
            bail!(
                "tokenizer exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(output.stdout)
    }

    fn tag(&self, tokenized: &[u8]) -> Result<String> {
        let mut child = Command::new(&self.tagger_binary)
            .args(["-token", "-lemma", "-sgml"])
            .arg(&self.model_file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning {}", self.tagger_binary.display()))?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .context("tree-tagger stdin unexpectedly closed")?;
            stdin
                .write_all(tokenized)
                .context("writing to tree-tagger")?;
        }
        drop(child.stdin.take());

        let output = child
            .wait_with_output()
            .context("waiting for tree-tagger")?;
        if !output.status.success() {
            bail!(
                "tree-tagger exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        String::from_utf8(output.stdout).context("tree-tagger output was not valid UTF-8")
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

fn current_platform_dir() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok("macos-arm64"),
        ("macos", "x86_64") => Ok("macos-x86_64"),
        ("linux", "x86_64") => Ok("linux-x86_64"),
        ("windows", "x86_64") => Ok("windows-x86_64"),
        (os, arch) => bail!("unsupported platform: {os}-{arch}"),
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
            if word.is_empty() {
                return None;
            }
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
            None => (cursor, cursor),
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
        assert_eq!(tags[3].pos.as_deref(), Some("NN"));
    }

    #[test]
    fn parse_tsv_strips_unknown_lemma_sentinel() {
        let raw = "Xyzzy\tNNP\t<unknown>\n";
        let tags = parse_tsv(raw);
        assert_eq!(tags.len(), 1);
        assert!(tags[0].lemma.is_none());
    }

    #[test]
    fn parse_tsv_skips_empty_and_status_lines() {
        // TreeTagger sometimes interleaves tab-prefixed progress lines
        // if stderr got merged; be defensive.
        let raw = "\treading parameters\nThe\tDT\tthe\n\n\tfinished\n";
        let tags = parse_tsv(raw);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].word, "The");
    }

    #[test]
    fn align_to_source_straightforward() {
        let text = "The quick brown fox.";
        let tags = vec![
            raw("The", "DT", "the"),
            raw("quick", "JJ", "quick"),
            raw("brown", "JJ", "brown"),
            raw("fox", "NN", "fox"),
            raw(".", "SENT", "."),
        ];
        let aligned = align_to_source(tags, text);
        assert_eq!(aligned.len(), 5);
        assert_eq!(&text[aligned[0].byte_start..aligned[0].byte_end], "The");
        assert_eq!(aligned[3].byte_start, 16);
        assert_eq!(aligned[3].byte_end, 19);
        assert_eq!(aligned[4].byte_start, 19);
    }

    #[test]
    fn align_to_source_handles_contraction_split() {
        let text = "I don't know.";
        let tags = vec![
            raw("I", "PP", "I"),
            raw("do", "VVP", "do"),
            raw("n't", "RB", "not"),
            raw("know", "VV", "know"),
            raw(".", "SENT", "."),
        ];
        let aligned = align_to_source(tags, text);
        assert_eq!(aligned.len(), 5);
        assert_eq!(aligned[1].byte_start, 2);
        assert_eq!(aligned[1].byte_end, 4);
        assert_eq!(aligned[2].byte_start, 4);
        assert_eq!(aligned[2].byte_end, 7);
        assert_eq!(aligned[2].lemma.as_deref(), Some("not"));
    }

    #[test]
    fn align_to_source_missing_token_keeps_stream_intact() {
        let text = "hello world";
        let tags = vec![raw("hello", "X", "x"), raw("WIDGET", "X", "x"), raw("world", "X", "x")];
        let aligned = align_to_source(tags, text);
        assert_eq!(aligned.len(), 3);
        assert_eq!(aligned[1].byte_start, aligned[1].byte_end);
        assert_eq!(aligned[2].byte_start, 6);
    }

    #[test]
    fn from_bundle_succeeds_on_current_platform() {
        let bundle = bundle_path();
        if !bundle.exists() {
            eprintln!("bundle not at {}, skipping", bundle.display());
            return;
        }
        let tt = TreeTagger::from_bundle(&bundle, "english").unwrap();
        assert_eq!(tt.language, "english");
        assert!(tt.tagger_binary.exists());
    }

    /// End-to-end sanity: actually spawn the pipeline, verify POS +
    /// lemma come back correctly. Slow-ish (~2 s) because TreeTagger
    /// reloads its parameter file each call. Skipped when the bundle
    /// isn't present (e.g. a minimal clone without resources/).
    #[test]
    fn end_to_end_tags_an_english_sentence() {
        let bundle = bundle_path();
        if !bundle.exists() {
            eprintln!("bundle not at {}, skipping", bundle.display());
            return;
        }
        let tt = TreeTagger::from_bundle(&bundle, "english").unwrap();
        let tokens = tt
            .annotate("The quick brown fox jumps over the lazy dog.")
            .unwrap();
        assert!(!tokens.is_empty());

        let jumps = tokens.iter().find(|t| t.word.as_ref() == "jumps");
        assert!(jumps.is_some(), "expected a token for 'jumps'");
        assert_eq!(jumps.unwrap().lemma.as_deref(), Some("jump"));
        assert_eq!(jumps.unwrap().pos.as_deref(), Some("NNS"));

        let the = tokens.iter().find(|t| t.word.as_ref() == "The");
        assert!(the.is_some());
        assert_eq!(the.unwrap().pos.as_deref(), Some("DT"));
    }

    fn bundle_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("resources/treetagger")
    }

    fn raw(word: &str, pos: &str, lemma: &str) -> RawTag {
        RawTag {
            word: word.into(),
            pos: Some(pos.into()),
            lemma: Some(lemma.into()),
        }
    }
}
