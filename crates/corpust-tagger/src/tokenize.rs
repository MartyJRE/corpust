//! Pure-Rust port of `utf8-tokenize.perl` (Helmut Schmid / Serge Sharoff).
//!
//! The upstream Perl script is the canonical pre-tokenizer for TreeTagger:
//! it splits raw text into one-token-per-line output, separating punctuation
//! and clitics the way TreeTagger's English model expects. Our job is to
//! reproduce its output byte-for-byte so the Rust `Tagger` can feed
//! TreeTagger-compatible streams without spawning Perl.
//!
//! Scope deliberately narrowed to **English** (the `-e` flag in the Perl
//! original). French / Italian / Catalan / Portuguese / Galician / Romanian
//! / Catalan clitic patterns are out of scope until those language models
//! land.
//!
//! Not ported:
//! - `-g` option (treat `NN.` as number rather than abbreviation).
//!   Re-add when we handle non-English text with numeric sentence-ends.
//! - BOM stripping on line 1 of input. Our input is always already
//!   UTF-8-clean `str`, not raw bytes off a file.
//!
//! Algorithm (closely mirroring the Perl):
//!
//! 1. Split `text` into chunks on `<...>` SGML tag boundaries, preserving
//!    tags as opaque units.
//! 2. For each non-tag chunk:
//!    a. Insert spaces around `...` and after `;!?` when followed by
//!       non-space. Split on whitespace.
//!    b. For each whitespace-delimited token, iteratively peel
//!       punctuation from front and back until stable:
//!         - `(` at start → emit, strip
//!         - `)` at end (when preceded by non-`(`) → queue as suffix
//!         - leading `PChar` char → emit, strip
//!         - trailing `FChar` char → queue as suffix
//!         - trailing `.` immediately after an `FChar` or `)` → queue `.`
//!    c. Check explicit abbreviation list — if matched, emit whole token.
//!    d. Check `(Xx-)*X.(Xx-)*X.` shape (`U.S.`, `A.B.C.`) — if matched,
//!       emit whole token.
//!    e. Disambiguate trailing period: if token ends `.` and isn't `...`,
//!       strip the `.` into suffix; re-check abbreviation list on the
//!       stripped form.
//!    f. Strip leading `--` as separate emitted token.
//!    g. Strip FClitic (English: `'s`, `'re`, `'ve`, `'d`, `'m`, `'em`,
//!       `'ll`, `n't`) from the end into suffix.
//!    h. Emit token + queued suffixes.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;

/// English tokenizer matching `utf8-tokenize.perl -e`.
///
/// Immutable after construction; can be shared across rayon workers via
/// a plain `&Tokenizer`.
#[derive(Debug)]
pub struct Tokenizer {
    abbreviations: HashSet<String>,
}

impl Tokenizer {
    /// Build with an explicit set of abbreviation tokens.
    ///
    /// Abbreviations are matched exactly (case-sensitive). A token on the
    /// list is always emitted as-is, skipping both the abbreviation-shape
    /// heuristic and the period-disambiguation step.
    pub fn new(abbreviations: impl IntoIterator<Item = String>) -> Self {
        Self {
            abbreviations: abbreviations.into_iter().collect(),
        }
    }

    /// Load abbreviations from a file — one per line, `#` starts a
    /// comment, blank lines ignored. Matches the Perl `-a` flag's
    /// expected format.
    pub fn from_abbreviations_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading abbreviations file {}", path.display()))?;
        let mut abbr = HashSet::new();
        for line in text.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            abbr.insert(t.to_owned());
        }
        Ok(Self { abbreviations: abbr })
    }

    /// Tokenize `text` into owned token strings, one per element.
    ///
    /// Output matches `utf8-tokenize.perl -e` byte-for-byte on valid
    /// UTF-8 input. SGML tags (matching `<...>`) are passed through as
    /// single tokens.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for chunk in split_sgml_chunks(text) {
            match chunk {
                Chunk::Sgml(t) => out.push(t.to_owned()),
                Chunk::Text(t) => self.tokenize_text_chunk(t, &mut out),
            }
        }
        out
    }

    fn tokenize_text_chunk(&self, text: &str, out: &mut Vec<String>) {
        // Step a: spacing around `...`, `;!?` before non-space. Done
        // via a single forward scan to avoid string reallocation loops.
        let padded = pad_punctuation(text);

        // Step b onwards: split on whitespace, process each word.
        for raw_word in padded.split_whitespace() {
            self.process_word(raw_word, out);
        }
    }

    fn process_word(&self, word: &str, out: &mut Vec<String>) {
        // Suffix queue: punctuation peeled off the end is emitted in
        // insertion order *after* the token proper. We push to the
        // front so the final order is token, trailing1, trailing2...
        let mut word = word.to_owned();
        let mut suffixes: Vec<String> = Vec::new();

        // Step b — iterative front/back peeling.
        loop {
            let before = word.clone();

            // Leading '(' with at least one trailing char → emit, strip.
            if let Some(stripped) = peel_leading_paren(&word) {
                out.push("(".to_owned());
                word = stripped;
                continue;
            }

            // Trailing ')' preceded by a non-'(' char → queue as suffix.
            if let Some(stripped) = peel_trailing_paren(&word) {
                suffixes.insert(0, ")".to_owned());
                word = stripped;
                continue;
            }

            // Leading PChar (one of the start-punctuation chars).
            if let Some((lead, rest)) = peel_leading_pchar(&word) {
                out.push(lead.to_string());
                word = rest;
                continue;
            }

            // Trailing FChar.
            if let Some((last, rest)) = peel_trailing_fchar(&word) {
                suffixes.insert(0, last.to_string());
                word = rest;
                continue;
            }

            // Trailing `.` after an FChar or `)`.
            if let Some(stripped) = peel_trailing_period_after_punct(&word) {
                suffixes.insert(0, ".".to_owned());
                word = stripped;
                continue;
            }

            if word == before {
                break;
            }
        }

        // Step c — explicitly listed abbreviations.
        if self.abbreviations.contains(&word) {
            out.push(word);
            out.extend(suffixes);
            return;
        }

        // Step d — X. or X.X.X. shape (letters + hyphens + periods).
        if is_letter_period_abbrev(&word) {
            out.push(word);
            out.extend(suffixes);
            return;
        }

        // Step e — disambiguate trailing period.
        if word.len() >= 2 && word.ends_with('.') && word != "..." {
            let stripped: String = word[..word.len() - 1].to_owned();
            // Perl pushes '.' after the current suffix list, i.e. the
            // period lands last in output order for this token.
            let period_suffix = ".".to_owned();
            if self.abbreviations.contains(&stripped) {
                // Known abbreviation: emit stripped form + period + any
                // previously queued trailing suffixes.
                out.push(stripped);
                out.push(period_suffix);
                out.extend(suffixes);
                return;
            }
            word = stripped;
            suffixes.insert(0, period_suffix);
        }

        // Step f — strip leading `--` as emitted tokens.
        while let Some(stripped) = word.strip_prefix("--") {
            if stripped.is_empty() {
                break;
            }
            out.push("--".to_owned());
            word = stripped.to_owned();
        }

        // Step g — trailing `--`.
        while let Some(stripped) = word.strip_suffix("--") {
            if stripped.is_empty() {
                break;
            }
            suffixes.insert(0, "--".to_owned());
            word = stripped.to_owned();
        }

        // Step g cont. — English FClitic stripping.
        loop {
            let before = word.clone();
            if let Some((cl, rest)) = peel_trailing_fclitic(&word) {
                suffixes.insert(0, cl.to_owned());
                word = rest.to_owned();
            }
            if word == before {
                break;
            }
        }

        if !word.is_empty() {
            out.push(word);
        }
        out.extend(suffixes);
    }
}

// ---------------------------------------------------------------------------
// SGML chunking
// ---------------------------------------------------------------------------

enum Chunk<'a> {
    Text(&'a str),
    Sgml(&'a str),
}

/// Split `text` into runs of plain text and `<...>` SGML tags.
/// Matches the Perl's `split(/\xff/)` after its tag-masking dance.
fn split_sgml_chunks(text: &str) -> Vec<Chunk<'_>> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut text_start = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Find matching '>' with no '<' or '>' in between.
            let mut j = i + 1;
            let mut ok = true;
            while j < bytes.len() && bytes[j] != b'>' {
                if bytes[j] == b'<' {
                    ok = false;
                    break;
                }
                j += 1;
            }
            if ok && j < bytes.len() {
                // Emit preceding text.
                if text_start < i {
                    chunks.push(Chunk::Text(&text[text_start..i]));
                }
                chunks.push(Chunk::Sgml(&text[i..j + 1]));
                i = j + 1;
                text_start = i;
                continue;
            }
        }
        // Advance by one UTF-8 char.
        i += utf8_char_len(bytes[i]);
    }
    if text_start < bytes.len() {
        chunks.push(Chunk::Text(&text[text_start..]));
    }
    chunks
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xC0 {
        1 // continuation; shouldn't start a char, but be lenient
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Punctuation spacing
// ---------------------------------------------------------------------------

fn pad_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4);
    out.push(' '); // leading space (matches Perl `$_ = " $_ "`)
    let mut chars: std::iter::Peekable<std::str::Chars<'_>> = text.chars().peekable();
    while let Some(c) = chars.next() {
        // `...` → ` ... `
        if c == '.' && matches!(chars.peek(), Some('.')) {
            let mut look = chars.clone();
            look.next();
            if let Some('.') = look.peek().copied() {
                // consume the two peeked dots
                chars.next();
                chars.next();
                if !out.ends_with(' ') {
                    out.push(' ');
                }
                out.push_str("...");
                if chars.peek().is_some() {
                    out.push(' ');
                }
                continue;
            }
        }

        // `;!?` before non-space → insert space.
        if matches!(c, ';' | '!' | '?') {
            out.push(c);
            if let Some(&n) = chars.peek() {
                if !n.is_whitespace() {
                    out.push(' ');
                }
            }
            continue;
        }

        out.push(c);
    }
    out.push(' ');
    out
}

// ---------------------------------------------------------------------------
// Peeling helpers — each returns `Option<...>` with the reduced form,
// None when the rule doesn't apply. They match the corresponding regex
// branches inside the Perl while-loop.
// ---------------------------------------------------------------------------

fn peel_leading_paren(word: &str) -> Option<String> {
    // Perl: `^(\()([^\)]*)(.)$` — starts with `(`, no `)` until the very
    // last character (which can be anything, `)` included). Strips the
    // leading `(` only; the trailing character (including `)`) stays for
    // the next peel iteration to deal with.
    if !word.starts_with('(') || word.chars().count() < 2 {
        return None;
    }
    let last_idx = word.len() - 1;
    for (i, c) in word.char_indices() {
        if c == ')' && i != last_idx {
            return None;
        }
    }
    Some(word[1..].to_owned())
}

fn peel_trailing_paren(word: &str) -> Option<String> {
    // Perl: `^([^(]+)(\))$` — word has no '(' and ends with ')'.
    if word.ends_with(')') && !word[..word.len() - 1].contains('(') && word.len() > 1 {
        Some(word[..word.len() - 1].to_owned())
    } else {
        None
    }
}

const PCHARS: &str = "[¿¡{'`\"‚„†‡‹‘’“”•–—›»«";
const FCHARS: &str = "]}'`\",;:!?؟%‚„…†‡‰‹‘’“”•–—›»«";

fn peel_leading_pchar(word: &str) -> Option<(char, String)> {
    // Perl: `s/^([$PChar])(.)/$2/` — leading PChar + at least one more char.
    let mut it = word.chars();
    let first = it.next()?;
    let second = it.clone().next()?;
    let _ = second;
    if PCHARS.contains(first) {
        let rest = word[first.len_utf8()..].to_owned();
        Some((first, rest))
    } else {
        None
    }
}

fn peel_trailing_fchar(word: &str) -> Option<(char, String)> {
    // Perl: `s/(.)([$FChar])$/$1/` — any char + FChar at end.
    let last = word.chars().next_back()?;
    if word.chars().count() < 2 {
        return None;
    }
    if FCHARS.contains(last) {
        let cut = word.len() - last.len_utf8();
        Some((last, word[..cut].to_owned()))
    } else {
        None
    }
}

fn peel_trailing_period_after_punct(word: &str) -> Option<String> {
    // Perl: `s/([$FChar]|\))\.$//` — FChar-or-')' then '.' at end.
    if !word.ends_with('.') || word.chars().count() < 2 {
        return None;
    }
    let without_period = &word[..word.len() - 1];
    let preceding = without_period.chars().next_back()?;
    if FCHARS.contains(preceding) || preceding == ')' {
        Some(without_period.to_owned())
    } else {
        None
    }
}

fn is_letter_period_abbrev(word: &str) -> bool {
    // Perl: /^([A-Za-z-]\.)+$/
    if word.is_empty() {
        return false;
    }
    let bytes = word.as_bytes();
    if bytes.len() % 2 != 0 {
        return false;
    }
    let mut i = 0;
    while i + 1 < bytes.len() {
        let c = bytes[i];
        let d = bytes[i + 1];
        let is_letter_or_hyphen = c.is_ascii_alphabetic() || c == b'-';
        if !(is_letter_or_hyphen && d == b'.') {
            return false;
        }
        i += 2;
    }
    true
}

fn peel_trailing_fclitic<'a>(word: &'a str) -> Option<(String, &'a str)> {
    // English FClitic: `['’´](s|re|ve|d|m|em|ll)|n['’´]t`
    //
    // Matched case-insensitively in the Perl (the `/i` flag on the `s///i`).
    // Returns (clitic_text, remaining_word).

    // Case 1: n + apostrophe + t (n't).
    // Work on the lowercased last 3 chars for the match, but return
    // the ORIGINAL case-preserved substring.
    if let Some(boundary) = word
        .char_indices()
        .rev()
        .nth(2)
        .map(|(i, _)| i)
    {
        let tail = &word[boundary..];
        let mut iter = tail.chars();
        let a = iter.next()?;
        let b = iter.next()?;
        let c = iter.next()?;
        let is_apostrophe = matches!(b, '\'' | '\u{2019}' | '\u{00B4}');
        if a.eq_ignore_ascii_case(&'n') && is_apostrophe && c.eq_ignore_ascii_case(&'t') {
            return Some((tail.to_owned(), &word[..boundary]));
        }
    }

    // Case 2: apostrophe + one of (s re ve d m em ll) — case-insensitive.
    for suf in ["'s", "'re", "'ve", "'d", "'m", "'em", "'ll",
                "\u{2019}s", "\u{2019}re", "\u{2019}ve", "\u{2019}d", "\u{2019}m", "\u{2019}em", "\u{2019}ll",
                "\u{00B4}s", "\u{00B4}re", "\u{00B4}ve", "\u{00B4}d", "\u{00B4}m", "\u{00B4}em", "\u{00B4}ll"] {
        if word.len() <= suf.len() {
            continue;
        }
        let tail_start = word.len() - suf.len();
        // Only accept if the tail starts on a char boundary.
        if !word.is_char_boundary(tail_start) {
            continue;
        }
        let tail = &word[tail_start..];
        if tail.eq_ignore_ascii_case(suf) {
            return Some((tail.to_owned(), &word[..tail_start]));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk() -> Tokenizer {
        let abbr = ["Mr.", "Mrs.", "Dr.", "U.S.", "etc.", "i.e.", "e.g.", "Ave."]
            .iter()
            .map(|s| s.to_string());
        Tokenizer::new(abbr)
    }

    #[test]
    fn plain_sentence() {
        let got = tk().tokenize("The quick brown fox jumps.");
        assert_eq!(
            got,
            vec!["The", "quick", "brown", "fox", "jumps", "."]
        );
    }

    #[test]
    fn splits_clitics() {
        let got = tk().tokenize("I don't know what he'll say.");
        assert_eq!(
            got,
            vec!["I", "do", "n't", "know", "what", "he", "'ll", "say", "."]
        );
    }

    #[test]
    fn abbreviation_preserved() {
        let got = tk().tokenize("Mr. Smith said hello.");
        assert_eq!(got, vec!["Mr.", "Smith", "said", "hello", "."]);
    }

    #[test]
    fn letter_period_abbrev() {
        let got = tk().tokenize("I live in the U.S.A. now.");
        // "U.S.A." passes the letter-period shape and should stay whole.
        assert_eq!(got, vec!["I", "live", "in", "the", "U.S.A.", "now", "."]);
    }

    #[test]
    fn ellipsis_separates() {
        let got = tk().tokenize("Well...maybe");
        assert_eq!(got, vec!["Well", "...", "maybe"]);
    }

    #[test]
    fn sgml_tag_preserved() {
        let got = tk().tokenize("Hello <b>world</b>!");
        assert_eq!(got, vec!["Hello", "<b>", "world", "</b>", "!"]);
    }

    #[test]
    fn quotes_and_commas() {
        let got = tk().tokenize("\"Hi,\" he said.");
        assert_eq!(got, vec!["\"", "Hi", ",", "\"", "he", "said", "."]);
    }

    #[test]
    fn parens_strip_out() {
        let got = tk().tokenize("See (above) for details.");
        assert_eq!(got, vec!["See", "(", "above", ")", "for", "details", "."]);
    }

    #[test]
    fn from_abbreviations_file_parses() {
        let tmp = tempfile_with_contents("# a comment\n\nMr.\nDr.\n  \nU.S.\n").unwrap();
        let tok = Tokenizer::from_abbreviations_file(&tmp).unwrap();
        assert!(tok.abbreviations.contains("Mr."));
        assert!(tok.abbreviations.contains("Dr."));
        assert!(tok.abbreviations.contains("U.S."));
        assert_eq!(tok.abbreviations.len(), 3);
        std::fs::remove_file(tmp).ok();
    }

    fn tempfile_with_contents(contents: &str) -> std::io::Result<std::path::PathBuf> {
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("corpust_tok_test_{}_{}.txt", pid, nanos));
        std::fs::write(&path, contents)?;
        Ok(path)
    }

    /// Byte-for-byte differential parity against the upstream
    /// `utf8-tokenize.perl -e`. Runs on the bundled testdata/sample.txt
    /// (≤1 KB) so it stays fast. Skipped when the bundle isn't
    /// available or Perl isn't on `PATH`.
    #[test]
    fn matches_perl_upstream_on_sample() {
        use std::path::{Path, PathBuf};
        use std::process::{Command, Stdio};

        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tok_script: PathBuf = repo_root.join("resources/treetagger/cmd/utf8-tokenize.perl");
        let abbr_file: PathBuf = repo_root.join("resources/treetagger/lib/english-abbreviations");
        let sample: PathBuf = repo_root.join("testdata/sample.txt");
        if !(tok_script.exists() && abbr_file.exists() && sample.exists()) {
            return;
        }
        let perl_ok = Command::new("perl")
            .arg("-v")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !perl_ok {
            return;
        }

        // Perl reference output.
        let perl_out = Command::new("perl")
            .arg(&tok_script)
            .args(["-e", "-a"])
            .arg(&abbr_file)
            .arg(&sample)
            .output()
            .expect("run perl tokenizer");
        assert!(perl_out.status.success(), "perl tokenizer failed: {:?}", perl_out);
        let perl_text = String::from_utf8(perl_out.stdout).unwrap();
        let perl_lines: Vec<&str> = perl_text.lines().collect();

        // Rust tokenizer output.
        let tok = Tokenizer::from_abbreviations_file(&abbr_file).unwrap();
        let text = std::fs::read_to_string(&sample).unwrap();
        let rust_tokens = tok.tokenize(&text);

        assert_eq!(
            rust_tokens.len(),
            perl_lines.len(),
            "token count mismatch — rust={} perl={}",
            rust_tokens.len(),
            perl_lines.len()
        );
        for (i, (r, p)) in rust_tokens.iter().zip(perl_lines.iter()).enumerate() {
            assert_eq!(
                r, p,
                "token #{i} mismatch — rust={r:?} perl={p:?}"
            );
        }
    }
}
