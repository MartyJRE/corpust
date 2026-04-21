# TreeTagger (bundled)

Vendored copy of [TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
by Helmut Schmid, Institut für maschinelle Sprachverarbeitung, University of
Stuttgart. Used by `corpust-annotate` for POS tagging and lemmatization.

See [`COPYRIGHT`](./COPYRIGHT) for the upstream license. TreeTagger is free
for research, teaching, and evaluation use; commercial use requires a
separate arrangement with the author.

## Layout

```
resources/treetagger/
├── README.md
├── COPYRIGHT                        # upstream license notice
├── bin/
│   ├── macos-arm64/tree-tagger      # 3.2.3 (Apple Silicon)
│   ├── macos-x86_64/tree-tagger     # 3.2.3 (Intel)
│   ├── linux-x86_64/tree-tagger     # 3.2.5
│   └── windows-x86_64/tree-tagger.exe   # 3.2.3a
├── cmd/
│   ├── utf8-tokenize.perl           # TreeTagger's own tokenizer (Perl)
│   └── tree-tagger-english          # upstream shell wrapper (unused by us; kept for reference)
└── lib/
    ├── english-abbreviations        # abbreviations list consumed by the tokenizer
    └── english.par                  # English parameter file (~14 MB, uncompressed)
```

## Runtime dependency: Perl

`utf8-tokenize.perl` runs under any Perl 5+ interpreter.

- **macOS / Linux** — preinstalled.
- **Windows** — install [Strawberry Perl](https://strawberryperl.com/) or
  bundle a portable distribution alongside the app.

A pure-Rust replacement for `utf8-tokenize.perl` is on the roadmap; it
would remove the Perl dependency entirely.

## Adding more languages

English is bundled by default. Other TreeTagger parameter files live at
<https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/#parfiles>.
Drop the uncompressed `<lang>.par` and `<lang>-abbreviations` (from the
`tagger-scripts.tar.gz` archive) into `lib/`, then reference them via
`TreeTagger::from_bundle(bundle_root, "<lang>")`.

A `corpust annotate install-lang <code>` subcommand that automates this
will land as a follow-up.

## Updating the bundle

Upstream archives:

- <https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-MacOSX-M1-3.2.3.tar.gz>
- <https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-MacOSX-Intel-3.2.3.tar.gz>
- <https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.5.tar.gz>
- <https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-windows-3.2.3a.zip>
- <https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz>
- <https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz>

Refreshing the bundle is a manual, version-pinned step — don't let CI
auto-update it.
