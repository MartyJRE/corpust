//! TreeTagger subprocess adapter.
//!
//! Pipeline per document:
//!
//! ```text
//! raw text ──► perl utf8-tokenize.perl ──► one token per line
//!           ──► tree-tagger -token -lemma -sgml <english.par>
//!           ──► word\tPOS\tlemma TSV
//! ```
//!
//! Two subprocesses are spawned per call. TreeTagger is an external
//! C program that block-buffers its stdout when the writer is a pipe
//! (which we are), so it only flushes on EOF — meaning we can't drive
//! it persistently through plain pipes without a pseudoterminal.
//! That makes `spawn-per-document` the simple correct path: the tagger
//! exits after each document, its buffer flushes, we read full output.
//!
//! The ~14 MB model reload on every spawn is the obvious cost. Parallel
//! indexing (one `TreeTagger` per rayon worker, each with its own
//! per-doc lifecycle) amortizes it across cores. A persistent PTY-based
//! subprocess remains the long-term answer but is deferred.
//!
//! Output tokens are realigned back to byte spans in the source via a
//! forward scan, so downstream positional bookkeeping stays intact.

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
    /// Configure an adapter from explicit paths.
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

    /// Locate a TreeTagger installation inside the repo's bundled
    /// layout. See `resources/treetagger/README.md` for the expected
    /// file tree.
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
        let child = Command::new("perl")
            .arg(&self.tokenizer_script)
            .arg("-e")
            .arg("-a")
            .arg(&self.abbreviations_file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning perl tokenizer")?;

        run_subprocess(child, text.as_bytes(), "tokenizer")
    }

    fn tag(&self, tokenized: &[u8]) -> Result<String> {
        let child = Command::new(&self.tagger_binary)
            .args(["-token", "-lemma", "-sgml"])
            .arg(&self.model_file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning {}", self.tagger_binary.display()))?;

        let stdout = run_subprocess(child, tokenized, "tree-tagger")?;
        String::from_utf8(stdout).context("tree-tagger output was not valid UTF-8")
    }
}

/// Drive a subprocess safely: write `input` to its stdin on a side
/// thread while the main thread concurrently reads stdout and stderr.
/// Avoids the classic "full pipe → deadlock" when `input` is bigger
/// than the OS pipe buffer (~64 KB).
fn run_subprocess(mut child: std::process::Child, input: &[u8], label: &str) -> Result<Vec<u8>> {
    let mut stdin = child
        .stdin
        .take()
        .with_context(|| format!("{label}: stdin unexpectedly closed"))?;
    let input_owned = input.to_vec();

    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        stdin.write_all(&input_owned)?;
        drop(stdin);
        Ok(())
    });

    let output = child
        .wait_with_output()
        .with_context(|| format!("{label}: waiting for subprocess"))?;

    writer
        .join()
        .map_err(|_| anyhow::anyhow!("{label}: writer thread panicked"))?
        .with_context(|| format!("{label}: writing to stdin"))?;

    if !output.status.success() {
        bail!(
            "{label} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
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
            pos.as_ref()?;
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
        let raw = "The\tDT\tthe\nquick\tJJ\tquick\nfox\tNN\tfox\n";
        let tags = parse_tsv(raw);
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].pos.as_deref(), Some("DT"));
    }

    #[test]
    fn parse_tsv_strips_unknown_lemma_sentinel() {
        let raw = "Xyzzy\tNNP\t<unknown>\n";
        let tags = parse_tsv(raw);
        assert_eq!(tags.len(), 1);
        assert!(tags[0].lemma.is_none());
    }

    #[test]
    fn parse_tsv_skips_sgml_pass_through_lines() {
        let raw = "<MARKER/>\nThe\tDT\tthe\n";
        let tags = parse_tsv(raw);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].word, "The");
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
        assert_eq!(aligned[1].byte_start, 2);
        assert_eq!(aligned[2].byte_start, 4);
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
            return;
        }
        let tt = TreeTagger::from_bundle(&bundle, "english").unwrap();
        assert_eq!(tt.language, "english");
    }

    #[test]
    fn end_to_end_tags_an_english_sentence() {
        let bundle = bundle_path();
        if !bundle.exists() {
            return;
        }
        let tt = TreeTagger::from_bundle(&bundle, "english").unwrap();
        let tokens = tt
            .annotate("The quick brown fox jumps over the lazy dog.")
            .unwrap();
        assert!(!tokens.is_empty());
        let jumps = tokens.iter().find(|t| t.word.as_ref() == "jumps").unwrap();
        assert_eq!(jumps.lemma.as_deref(), Some("jump"));
    }

    fn raw(word: &str, pos: &str, lemma: &str) -> RawTag {
        RawTag {
            word: word.into(),
            pos: Some(pos.into()),
            lemma: Some(lemma.into()),
        }
    }

    fn bundle_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("resources/treetagger")
    }
}
