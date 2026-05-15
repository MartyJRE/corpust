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
        /// Explicit location for the index. Overrides the default,
        /// which drops the index into the platform data directory
        /// (`~/Library/Application Support/corpust/corpora/<slug>/index/`
        /// on macOS, `$XDG_DATA_HOME/corpust/…` on Linux, `%APPDATA%\corpust\…`
        /// on Windows).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Human-readable corpus name. Drives the on-disk slug when
        /// `--out` is omitted. Defaults to the input directory's
        /// basename.
        #[arg(long)]
        name: Option<String>,
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
    /// Annotation tooling (install language packs, …).
    Annotate {
        #[command(subcommand)]
        sub: AnnotateCmd,
    },
}

#[derive(Subcommand)]
enum AnnotateCmd {
    /// Download a TreeTagger parameter file from the upstream
    /// Stuttgart server and drop it into the platform data
    /// directory under `treetagger/lib/<lang>.par`.
    InstallLang {
        /// ISO 639-1 language code (`cs`, `de`, `fr`, `it`, `nl`,
        /// `pl`, `pt`, `ru`, `sk`, `es`). Mapped internally to the
        /// upstream language name.
        code: String,
        /// Force-overwrite an existing installed file.
        #[arg(long)]
        force: bool,
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
            name,
            annotate,
            tagger,
            tagger_bundle,
            language,
        } => run_index(input, out, name, annotate, tagger, tagger_bundle, language),
        Command::Kwic {
            index,
            term,
            layer,
            context,
            limit,
        } => run_kwic(index, &term, layer.into(), context, limit),
        Command::Annotate { sub } => match sub {
            AnnotateCmd::InstallLang { code, force } => run_install_lang(&code, force),
        },
    }
}

/// ISO 639-1 → upstream TreeTagger language name. Covers the
/// non-toy languages Stuttgart ships parameter files for. New entries
/// can be added freely; if an unknown code shows up the command bails
/// with a helpful message.
fn iso_to_treetagger_language(code: &str) -> Option<&'static str> {
    match code {
        "cs" => Some("czech"),
        "de" => Some("german"),
        "en" => Some("english"),
        "es" => Some("spanish"),
        "fr" => Some("french"),
        "it" => Some("italian"),
        "nl" => Some("dutch"),
        "pl" => Some("polish"),
        "pt" => Some("portuguese"),
        "ru" => Some("russian"),
        "sk" => Some("slovak"),
        _ => None,
    }
}

fn run_install_lang(code: &str, force: bool) -> Result<()> {
    let lang = iso_to_treetagger_language(code)
        .with_context(|| format!("unsupported ISO 639-1 code: {code:?}"))?;
    // Install under <data_root>/treetagger/lib/, mirroring the
    // bundled layout so `from_bundle(<data_root>/treetagger, lang)`
    // works once we wire that lookup. For now the file just lives in
    // a stable place users can point `--tagger-bundle` at.
    let data_root = corpust_io::paths::data_root().context("resolving data root")?;
    let lib_dir = data_root.join("treetagger").join("lib");
    std::fs::create_dir_all(&lib_dir).with_context(|| format!("creating {}", lib_dir.display()))?;
    let par_path = lib_dir.join(format!("{lang}.par"));
    if par_path.exists() && !force {
        eprintln!(
            "already installed: {} (use --force to overwrite)",
            par_path.display()
        );
        return Ok(());
    }
    let url =
        format!("https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/{lang}.par.gz");
    eprintln!("downloading {url}");
    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("requesting {url}"))?;
    let mut gz = response.into_reader();
    let mut decoder = flate2::read::GzDecoder::new(&mut gz);
    let tmp = par_path.with_extension("par.part");
    let written = {
        let mut out =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        std::io::copy(&mut decoder, &mut out)
            .with_context(|| format!("writing {}", tmp.display()))?
    };
    std::fs::rename(&tmp, &par_path)
        .with_context(|| format!("moving {} into place", par_path.display()))?;
    eprintln!(
        "installed {lang} parameter file ({written} bytes) to {}",
        par_path.display()
    );
    eprintln!(
        "TreeTagger will pick it up when `--tagger-bundle {}` is passed.",
        data_root.join("treetagger").display()
    );
    Ok(())
}

fn run_index(
    input: PathBuf,
    out: Option<PathBuf>,
    name: Option<String>,
    annotate: bool,
    tagger_kind: TaggerArg,
    tagger_bundle: PathBuf,
    language: String,
) -> Result<()> {
    // Track (display_name, slug, corpus_dir) so we can write the
    // metadata sidecar next to the index when the layout permits
    // (default platform-data-dir path) and a sensible best-effort
    // sidecar next to `--out` for explicit paths.
    let display_name = name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            input
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("corpus")
                .to_owned()
        });
    let (out, slug, corpus_dir) = match out {
        Some(p) => {
            // Explicit `--out`: write the sidecar alongside it. If the
            // user passed `…/foo/index`, sidecar goes to `…/foo/`.
            // Slug derives from the parent's basename or `--name`.
            let corpus_dir = p
                .parent()
                .filter(|pp| !pp.as_os_str().is_empty())
                .map(|pp| pp.to_path_buf())
                .unwrap_or_else(|| p.clone());
            let slug = corpust_io::paths::slugify(&display_name);
            (p, slug, corpus_dir)
        }
        None => {
            // Default: drop into the platform data directory, slug
            // derived from --name or the input folder's basename.
            let base = corpust_io::paths::slugify(&display_name);
            let slug = corpust_io::paths::unique_slug(&base)
                .with_context(|| format!("allocating slug for {display_name:?}"))?;
            let dir = corpust_io::paths::corpus_dir(&slug)
                .with_context(|| format!("resolving corpus dir for slug {slug:?}"))?;
            std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            (dir.join("index"), slug, dir)
        }
    };
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
    let tagger_id = tagger.as_deref().map(|t| t.id().to_owned());
    let index = CorpusIndex::create(&out)
        .with_context(|| format!("creating index at {}", out.display()))?;
    index.add_documents(docs, tagger.as_deref())?;
    let index_elapsed = t1.elapsed();
    let build_ms = t1.elapsed().as_millis() as u64;

    // Write the metadata.json sidecar next to the index so the
    // Tauri UI's `list_corpora` picks the corpus up. Mirrors the
    // structure the Tauri build path produces.
    use corpust_io::metadata::{CorpusMeta, dir_size, iso_now, write_metadata_file};
    let mut meta = CorpusMeta::stub(slug, display_name, out.to_string_lossy().into_owned());
    meta.source_path = input.to_string_lossy().into_owned();
    meta.annotated = annotate;
    meta.doc_count = doc_count as u64;
    // Token count is a byte-based approximation; a real pass over
    // the tantivy index would be more accurate but isn't needed
    // for the display-only header in the UI.
    meta.token_count = (byte_count / 6) as u64;
    meta.avg_doc_len = if doc_count > 0 {
        (byte_count / doc_count) as u64
    } else {
        0
    };
    meta.built_at = iso_now();
    meta.build_ms = build_ms;
    meta.size_on_disk = dir_size(&out).unwrap_or(0);
    meta.annotator = tagger_id.clone();
    meta.tagger_id = tagger_id;
    let metadata_path = corpus_dir.join("metadata.json");
    if let Err(e) = write_metadata_file(&metadata_path, &meta) {
        eprintln!("warning: couldn't write {}: {e:#}", metadata_path.display());
    }

    println!(
        "indexed {doc_count} doc(s) ({byte_count} bytes) in {:.2?} (read {:.2?} + index {:.2?})",
        t0.elapsed(),
        read_elapsed,
        index_elapsed
    );
    println!("index written to {}", out.display());
    println!("metadata written to {}", metadata_path.display());
    Ok(())
}

fn build_tagger(
    kind: TaggerArg,
    bundle_root: &std::path::Path,
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
            let abbr_path = bundle_root
                .join("lib")
                .join(format!("{language}-abbreviations"));
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

    let left_width = hits
        .iter()
        .map(|h| h.left.chars().count())
        .max()
        .unwrap_or(0);
    let hit_width = hits
        .iter()
        .map(|h| h.hit.chars().count())
        .max()
        .unwrap_or(0);

    for hit in &hits {
        let file = hit.path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
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
