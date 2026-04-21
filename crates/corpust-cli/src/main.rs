//! `corpust` CLI.
//!
//! Dev-loop tool for driving the library before the Tauri UI exists. Two
//! subcommands: `index` (build an index from a directory of `.txt` files) and
//! `kwic` (run a single-term concordance over an existing index).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use corpust_index::{CorpusIndex, DEFAULT_CONTEXT, DEFAULT_LIMIT};
use corpust_query::{KwicRequest, kwic};
use std::path::PathBuf;
use std::time::Instant;

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
    },
    /// Run a single-term KWIC concordance over an existing index.
    Kwic {
        /// Path to an index built by `corpust index`.
        #[arg(long)]
        index: PathBuf,
        /// Term to search for.
        term: String,
        /// Tokens of context on each side.
        #[arg(long, default_value_t = DEFAULT_CONTEXT)]
        context: usize,
        /// Maximum number of hits to return.
        #[arg(long, default_value_t = DEFAULT_LIMIT)]
        limit: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index { input, out } => run_index(input, out),
        Command::Kwic {
            index,
            term,
            context,
            limit,
        } => run_kwic(index, &term, context, limit),
    }
}

fn run_index(input: PathBuf, out: PathBuf) -> Result<()> {
    let t0 = Instant::now();
    let docs = corpust_io::read_text_dir(&input)
        .with_context(|| format!("reading corpus at {}", input.display()))?;
    let read_elapsed = t0.elapsed();
    let doc_count = docs.len();
    let byte_count: usize = docs.iter().map(|d| d.text.len()).sum();

    let t1 = Instant::now();
    let index = CorpusIndex::create(&out)
        .with_context(|| format!("creating index at {}", out.display()))?;
    index.add_documents(docs)?;
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

fn run_kwic(index_path: PathBuf, term: &str, context: usize, limit: usize) -> Result<()> {
    let index = CorpusIndex::open(&index_path)
        .with_context(|| format!("opening index at {}", index_path.display()))?;

    let t0 = Instant::now();
    let hits = kwic(&index, KwicRequest::new(term).context(context).limit(limit))?;
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
