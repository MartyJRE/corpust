# corpust

[![CI](https://github.com/MartyJRE/corpust/actions/workflows/ci.yml/badge.svg)](https://github.com/MartyJRE/corpust/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MartyJRE/corpust/branch/main/graph/badge.svg)](https://codecov.io/gh/MartyJRE/corpust)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A Rust corpus-linguistics toolkit — aiming for LancsBoxX-level functionality
with modern performance. Cross-platform desktop app + library + CLI, built for
billion-word corpora on commodity hardware.

> **Status:** pre-alpha. CLI works end-to-end (ingest → annotate → KWIC).
> Pure-Rust TreeTagger reimplementation reaches **99.80 % POS accuracy** on
> the 10 KB Gutenberg sample (bit-identical with the reference tagger on
> 722 / 722 contexts in the bigram tree). Tauri desktop app scaffolded.

## Quick start

### CLI

```sh
# Build everything (release mode for real numbers).
cargo build --release --workspace

# Index a directory of .txt files (word layer only).
cargo run --release -p corpust-cli -- index ./testdata --name "my corpus"

# With POS + lemma annotation (TreeTagger, English bundled).
cargo run --release -p corpust-cli -- index ./testdata --name "my corpus" --annotate

# Run a KWIC query — word / lemma / pos layers.
cargo run --release -p corpust-cli -- kwic --index <index-dir> the
cargo run --release -p corpust-cli -- kwic --index <index-dir> go  --layer lemma
cargo run --release -p corpust-cli -- kwic --index <index-dir> NN  --layer pos
```

When `--out` is omitted, the index lives in the platform data directory
(`~/Library/Application Support/corpust/corpora/<slug>/` on macOS,
`$XDG_DATA_HOME/corpust/` on Linux, `%APPDATA%\corpust\` on Windows). A
`metadata.json` sidecar lands next to it so the Tauri UI's corpus list picks
it up automatically.

Set `CORPUST_DATA_ROOT=/some/path` to relocate the whole tree — typically onto
an external drive.

### Multi-language tagging

The English model ships with the repo. For other languages, download from
upstream:

```sh
cargo run --release -p corpust-cli -- annotate install-lang de
cargo run --release -p corpust-cli -- annotate install-lang fr
cargo run --release -p corpust-cli -- annotate install-lang cs
```

Supported ISO 639-1 codes: `cs de en es fr it nl pl pt ru sk`. Files land in
`<data_root>/treetagger/lib/` and are picked up by `TreeTagger::from_bundle`
on next index build.

### Desktop app

```sh
cd app
npm install            # first run only
npm run dev            # browser preview at http://localhost:1420
npm run tauri:dev      # real Tauri desktop window
```

## Tagger parity

`corpust-tagger` is a pure-Rust reimplementation of Helmut Schmid's TreeTagger.
It parses the binary `.par` model file directly — no Perl, no subprocess — and
reaches bit-identical agreement with the reference tagger's bigram-tree output:

| sample                            | POS accuracy   | gap to reference  |
| --------------------------------- | -------------- | ----------------- |
| Gutenberg, 10 KB / 2 032 tokens   | **99.80 %**    | 4 errors          |
| Gutenberg, 50 KB / 9 990 tokens   | **98.82 %**    | 118 errors        |
| UNSC, 100 resolutions / 32 K tok  | **99.24 %**    | 244 errors        |
| UNSC, 500 resolutions / 161 K tok | **99.34 %**    | 1 071 errors      |

The `diff_bigram_tree_vs_oracle` regression test asserts **0 / 722** argmax
disagreement between our parsed dtree and `tree-tagger -print-prob-tree`,
with max per-tag absolute probability difference < 1 × 10⁻⁵. Any future drift
in dtree byte layout or traversal fails it.

## Performance

Measured on an M-series MacBook, release build with fat LTO. Numbers are for
the full 544-book / 79.5 M-word Project Gutenberg sample:

| Operation                      | Wall time | Throughput      |
| ------------------------------ | --------- | --------------- |
| Index build, word-only         | ~17 s     | ~4.5 M words/s  |
| Index build, with annotation   | ~3 min    | ~440 K words/s  |
| KWIC, common term (`the`)      | ~820 µs   | —               |
| KWIC, rare term in big book    | ~3.8 ms   | —               |
| KWIC on lemma / pos layer      | ~150 µs   | —               |

For reference, LancsBoxX on a 9 GB EU-resolutions corpus took 12+ hours to
reach ~50 % and crashed. Projected end-to-end time for the same corpus with
today's code is **~70 minutes** — about 20× faster on the whole job vs. a
linearly-extrapolated LancsBox run.

## Workspace layout

| Crate              | Role                                                                   |
| ------------------ | ---------------------------------------------------------------------- |
| `corpust-core`     | Domain types (`Document`, `Token`, ids). Pure, no deps.                |
| `corpust-tokenize` | Tokenizers. Pure-Rust port of `utf8-tokenize.perl` for TreeTagger.     |
| `corpust-io`       | Ingestion, paths, persisted corpus metadata.                           |
| `corpust-annotate` | `Annotator` trait + TreeTagger subprocess adapter.                     |
| `corpust-tagger`   | Pure-Rust TreeTagger reimplementation — `.par` parser + Viterbi.       |
| `corpust-index`    | Tantivy-backed multi-layer positional index. The hot crate.            |
| `corpust-query`    | Query layer over the index. KWIC + collocations.                       |
| `corpust-cli`      | `corpust` binary — dev tool / power-user entry point.                  |
| `app/src-tauri`    | `corpust-ui` — Tauri desktop app (React + Tailwind + shadcn/ui).       |

## Bundled assets

`resources/treetagger/` vendors TreeTagger binaries for macOS (arm64 + x86_64),
Linux (x86_64), Windows (x86_64), plus the English parameter file. See
[`resources/treetagger/README.md`](resources/treetagger/README.md) for layout
and license details. The pure-Rust tagger (`corpust-tagger`) doesn't need any
of these at runtime; the subprocess adapter (`corpust-annotate`) does, for
A/B comparison and for languages we haven't reimplemented yet.

## Releasing

See [RELEASING.md](RELEASING.md). Tags drive releases — pushing a `v*` tag
builds CLI binaries + Tauri bundles for every supported platform and attaches
them to a GitHub release.

## Roadmap

Open items live as GitHub issues. The two non-trivial ones:

- [#6](https://github.com/MartyJRE/corpust/issues/6) — UI polish pass for
  non-collocation views (KWIC table, frequency view, command palette, …).
- [#15](https://github.com/MartyJRE/corpust/issues/15) — merge case-insensitive
  lex entries to close the last ~0.6 pp of UNSC POS accuracy. Requires
  reverse-engineering tree-tagger's second lex table.

Further out: CQL parser + executor, keyness / dispersion stats, XML/TEI
ingestion, Tauri IPC wiring for the build dialog → real backend commands.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
