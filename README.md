# corpust

A Rust corpus-linguistics toolkit — aiming for LancsBoxX-level functionality with
modern performance. Cross-platform desktop app + library, built for billion-word
corpora on commodity hardware.

> **Status:** pre-alpha. CLI works end-to-end: ingest `.txt`, index with optional
> TreeTagger-driven POS + lemma annotation, run KWIC queries across word / lemma /
> POS layers. No GUI yet — Tauri shell is in the roadmap.

## Quick start

```sh
# Build everything (release mode for real numbers)
cargo build --release --workspace

# Index a directory of .txt files (word layer only)
cargo run --release -p corpust-cli -- index ./testdata --out ./testdata/index

# Or with POS + lemma annotation (TreeTagger, English bundled)
cargo run --release -p corpust-cli -- index ./testdata --out ./testdata/index --annotate

# Run a KWIC query — word / lemma / pos layers
cargo run --release -p corpust-cli -- kwic --index ./testdata/index the
cargo run --release -p corpust-cli -- kwic --index ./testdata/index go  --layer lemma
cargo run --release -p corpust-cli -- kwic --index ./testdata/index NN  --layer pos
```

## Workspace layout

| Crate              | Role                                                     |
| ------------------ | -------------------------------------------------------- |
| `corpust-core`     | Domain types (`Document`, `Token`, ids). Pure, no deps.  |
| `corpust-tokenize` | Tokenizers. Unicode word segmentation today.             |
| `corpust-io`       | Ingestion. Reads `.txt` directories into `Document`s.    |
| `corpust-annotate` | `Annotator` trait + TreeTagger subprocess adapter.       |
| `corpust-index`    | The hot crate. Tantivy-backed multi-layer positional index. |
| `corpust-query`    | Query layer over the index. KWIC today, CQL later.       |
| `corpust-cli`      | `corpust` binary — dev tool / power-user entry point.    |

## Bundled assets

`resources/treetagger/` vendors TreeTagger binaries for macOS (arm64 + x86_64),
Linux (x86_64), Windows (x86_64), plus `utf8-tokenize.perl` and the English
parameter file. See [`resources/treetagger/README.md`](resources/treetagger/README.md)
for layout and license details. Perl is required at runtime (preinstalled on
macOS/Linux; Strawberry Perl on Windows).

## Numbers

From today's measurement on a 544-book / 79.5 M-word Project Gutenberg sample
(release build, fat LTO, M-series MacBook):

| Operation                           | Wall time | Throughput        |
| ----------------------------------- | --------- | ----------------- |
| Index build, word-only              | 17.5 s    | 4.5 M words/sec   |
| Index build, with annotation        | 3:31      | 376 K words/sec   |
| KWIC, common term (e.g. `the`)      | ~820 µs   | —                 |
| KWIC, rare term in big book         | ~3.8 ms   | —                 |
| KWIC, lemma / pos layer             | ~150 µs   | —                 |

For comparison, LancsBoxX on a 9 GB EU-resolutions corpus took 12+ hours to
reach ~50% and then crashed — never finished. Our projected end-to-end time
on the same corpus with today's code is **~70 minutes** (~20× faster on the
whole-job comparison, since a linearly-extrapolated LancsBox run would have
been at least 24 hours). With the persistent-subprocess work
(issue [#3](https://github.com/MartyJRE/corpust/issues/3)) that drops to
**~30 minutes**, i.e. ~48× faster.

## Roadmap

Shipped:

- Multi-crate workspace with clean one-way dependency graph
- Tantivy-backed positional index with per-token byte offsets for O(context) KWIC
- TreeTagger subprocess adapter; bundled English model
- `--annotate` flag drives lemma + POS fields across indexing
- `--layer word | lemma | pos` flag on KWIC queries
- Rayon-parallel annotation

Open issues track the next steps:

- [#1](https://github.com/MartyJRE/corpust/issues/1) — store indexes in platform data dir
- [#3](https://github.com/MartyJRE/corpust/issues/3) — persistent TreeTagger via PTY
- [#4](https://github.com/MartyJRE/corpust/issues/4) — Rust port of `utf8-tokenize.perl`
- [#5](https://github.com/MartyJRE/corpust/issues/5) — `corpust annotate install-lang <code>`

Further out: CQL parser + executor, Tauri desktop shell, statistics (collocations,
keyness, dispersion), XML/TEI ingestion.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
