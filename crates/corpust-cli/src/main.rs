//! `corpust` CLI.
//!
//! Dev-loop tool for driving the library before the Tauri UI exists. Two
//! subcommands: `index` (build an index from a directory of `.txt` files) and
//! `kwic` (run a single-term concordance over an existing index).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use corpust_annotate::{Annotator, treetagger::TreeTagger};
use corpust_index::{CorpusIndex, DEFAULT_CONTEXT, DEFAULT_LIMIT, QueryLayer};
use corpust_query::{KwicRequest, kwic};
use corpust_tagger::Tagger as RustTagger;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_TAGGER_BUNDLE: &str = "./resources/treetagger";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TaggerArg {
    /// Bundled `tree-tagger` binary — spawns one subprocess per
    /// document and reloads the .par each time. Accurate (LancsBox
    /// parity) but slow.
    Subprocess,
    /// Pure-Rust in-process tagger (`corpust-tagger::Tagger`).
    /// ~200× faster per call, currently at ~92% POS accuracy
    /// because dtree Viterbi hasn't landed. Use for indexing where
    /// throughput matters more than exact parity.
    Rust,
}

#[derive(Parser)]
#[command(name = "corpust", version, about = "Corpus-linguistics CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build an index from a directory of .txt files.
    Index {
        /// Directory to scan recursively for .txt files.
        input: PathBuf,
        /// Where to write the index.
        #[arg(long)]
        out: PathBuf,
        /// Enable POS + lemma annotation during indexing.
        #[arg(long)]
        annotate: bool,
        /// Which tagger implementation to use when `--annotate` is
        /// set. Default is the pure-Rust in-process tagger, which is
        /// orders of magnitude faster than the subprocess.
        #[arg(long, value_enum, default_value_t = TaggerArg::Rust)]
        tagger: TaggerArg,
        /// Path to the TreeTagger bundle. Defaults to `./resources/treetagger`
        /// relative to the current working directory (repo layout).
        #[arg(long, default_value = DEFAULT_TAGGER_BUNDLE)]
        tagger_bundle: PathBuf,
        /// TreeTagger language name (as used in parameter-file names).
        #[arg(long, default_value = "english")]
        language: String,
    },
    /// Run a single-term KWIC concordance over an existing index.
    Kwic {
        /// Path to an index built by `corpust index`.
        #[arg(long)]
        index: PathBuf,
        /// Term to search for.
        term: String,
        /// Annotation layer to query.
        #[arg(long, value_enum, default_value_t = LayerArg::Word)]
        layer: LayerArg,
        /// Tokens of context on each side.
        #[arg(long, default_value_t = DEFAULT_CONTEXT)]
        context: usize,
        /// Maximum number of hits to return.
        #[arg(long, default_value_t = DEFAULT_LIMIT)]
        limit: usize,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LayerArg {
    /// Surface word form (always populated).
    Word,
    /// Lemma — requires annotation at index time.
    Lemma,
    /// Part-of-speech tag — requires annotation at index time.
    Pos,
}

impl From<LayerArg> for QueryLayer {
    fn from(arg: LayerArg) -> Self {
        match arg {
            LayerArg::Word => QueryLayer::Word,
            LayerArg::Lemma => QueryLayer::Lemma,
            LayerArg::Pos => QueryLayer::Pos,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index {
            input,
            out,
            annotate,
            tagger,
            tagger_bundle,
            language,
        } => run_index(input, out, annotate, tagger, tagger_bundle, language),
        Command::Kwic {
            index,
            term,
            layer,
            context,
            limit,
        } => run_kwic(index, &term, layer.into(), context, limit),
    }
}

fn run_index(
    input: PathBuf,
    out: PathBuf,
    annotate: bool,
    tagger_kind: TaggerArg,
    tagger_bundle: PathBuf,
    language: String,
) -> Result<()> {
    let t0 = Instant::now();
    let docs = corpust_io::read_text_dir(&input)
        .with_context(|| format!("reading corpus at {}", input.display()))?;
    let read_elapsed = t0.elapsed();
    let doc_count = docs.len();
    let byte_count: usize = docs.iter().map(|d| d.text.len()).sum();

    // Leak the string so we can keep the Annotator's `'static` constraint
    // satisfied — the value lives until process exit anyway.
    let lang_static: &'static str = Box::leak(language.into_boxed_str());
    let tagger: Option<Box<dyn Annotator + Sync>> = if annotate {
        Some(build_tagger(tagger_kind, &tagger_bundle, lang_static)?)
    } else {
        None
    };
    if let Some(t) = tagger.as_deref() {
        println!("annotation enabled: {}", t.id());
    }

    let t1 = Instant::now();
    let index = CorpusIndex::create(&out)
        .with_context(|| format!("creating index at {}", out.display()))?;
    index.add_documents(docs, tagger.as_deref())?;
    let index_elapsed = t1.elapsed();

    println!(
        "indexed {doc_count} doc(s) ({byte_count} bytes) in {:.2?} (read {:.2?} + index {:.2?})",
        t0.elapsed(),
        read_elapsed,
        index_elapsed
    );
    println!("index written to {}", out.display());
    Ok(())
}

fn build_tagger(
    kind: TaggerArg,
    bundle_root: &PathBuf,
    language: &'static str,
) -> Result<Box<dyn Annotator + Sync>> {
    match kind {
        TaggerArg::Subprocess => {
            let tt = TreeTagger::from_bundle(bundle_root, language).with_context(|| {
                format!("locating TreeTagger bundle at {}", bundle_root.display())
            })?;
            Ok(Box::new(tt))
        }
        TaggerArg::Rust => {
            let par = bundle_root.join("lib").join(format!("{language}.par"));
            let abbr_path = bundle_root.join("lib").join(format!("{language}-abbreviations"));
            let abbr: Vec<String> = if abbr_path.exists() {
                std::fs::read_to_string(&abbr_path)
                    .with_context(|| format!("reading {}", abbr_path.display()))?
                    .lines()
                    .filter_map(|l| {
                        let t = l.trim();
                        (!t.is_empty() && !t.starts_with('#')).then(|| t.to_owned())
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let tagger = RustTagger::load(&par, language, abbr)
                .with_context(|| format!("loading {}", par.display()))?;
            Ok(Box::new(tagger))
        }
    }
}

fn run_kwic(
    index_path: PathBuf,
    term: &str,
    layer: QueryLayer,
    context: usize,
    limit: usize,
) -> Result<()> {
    let index = CorpusIndex::open(&index_path)
        .with_context(|| format!("opening index at {}", index_path.display()))?;

    let t0 = Instant::now();
    let hits = kwic(
        &index,
        KwicRequest::new(term)
            .layer(layer)
            .context(context)
            .limit(limit),
    )?;
    let elapsed = t0.elapsed();

    let left_width = hits.iter().map(|h| h.left.chars().count()).max().unwrap_or(0);
    let hit_width = hits.iter().map(|h| h.hit.chars().count()).max().unwrap_or(0);

    for hit in &hits {
        let file = hit
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        println!(
            "{file:20} | {left:>lw$}  \x1b[1m{hit:^hw$}\x1b[0m  {right}",
            file = file,
            left = hit.left,
            hit = hit.hit,
            right = hit.right,
            lw = left_width,
            hw = hit_width,
        );
    }
    println!("\n{} hit(s) in {:.2?}", hits.len(), elapsed);
    Ok(())
}
