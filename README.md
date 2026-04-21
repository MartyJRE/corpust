# corpust

A Rust corpus-linguistics toolkit — aiming for LancsBoxX-level functionality with
modern performance. Cross-platform desktop app + library, built for billion-word
corpora on commodity hardware.

> **Status:** pre-alpha. Phase 0 scaffold — ingests plain text, builds a Tantivy
> index, runs simple KWIC queries from the CLI. No GUI yet.

## Phase 0 quick start

```sh
# Build everything
cargo build --workspace

# Index a directory of .txt files
cargo run -p corpust-cli -- index ./testdata --out ./testdata/index

# Run a KWIC query
cargo run -p corpust-cli -- kwic --index ./testdata/index the
```

## Workspace layout

| Crate              | Role                                                     |
| ------------------ | -------------------------------------------------------- |
| `corpust-core`     | Domain types (`Document`, `Token`, ids). Pure, no deps.  |
| `corpust-tokenize` | Tokenizers. Starts with a Unicode whitespace tokenizer.  |
| `corpust-io`       | Ingestion. Reads `.txt` directories into `Document`s.    |
| `corpust-index`    | The hot crate. Tantivy-backed positional index.          |
| `corpust-query`    | Query layer over the index. KWIC today, CQL later.       |
| `corpust-cli`      | `corpust` binary — dev tool / power-user entry point.    |

## Roadmap

Phase 0 (now): ingest `.txt` → Tantivy index → single-term KWIC via CLI.

Later:

1. CQL parser (`[lemma="go"] [pos="IN"]`) + executor.
2. Tauri desktop shell — corpus manager, KWIC view, frequency list.
3. Annotation pipeline (POS/lemma via TreeTagger, then pure-Rust taggers).
4. Statistics — collocations, keyness, dispersion.
5. XML/TEI ingestion + structural queries.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
