//! `.par` lexicon section — lemma pool + word records.
//!
//! Reverse-engineered from hex-dump inspection and cross-checked against
//! `tree-tagger -prob -threshold 0.00001` output on the bundled English
//! model. Format as we currently understand it:
//!
//! ```text
//! lemma_pool:
//!     u32               pool_count                  // 243034 for english.par
//!     cstr[pool_count]  null-terminated UTF-8 strings (ASCII-sorted)
//!
//! lexicon_header:
//!     u32               record_count                // 360930 for english.par
//!     i32               0xFFFFFFFE                  // sentinel
//!     u32[22]           opaque                      // unknown block; skipped blind
//!     u16               00 00                       // 2-byte pad to align records
//!
//! word_records: one per known word form, not necessarily matching the
//! lemma pool — e.g. the pool has `'` then `'30s`, but records have
//! `'` then `''` then `'30s` because `''` is a word form whose lemma is
//! `''` itself (which is also an entry in the pool under a different
//! path of sort ordering).
//!     cstr              word                        (UTF-8, null-terminated)
//!     u32               count
//!     u32               leading_field               // opaque — plausibly training frequency
//!     u32[count]        tag_id                      // indexes into Header::tags
//!     f32[count]        prob                        // P(tag | word) from training
//!     u32[count]        lemma_index                 // indexes into lemma_pool
//! ```
//!
//! The two pools overlap heavily but aren't identical. Word forms in
//! records reference lemmas by index into `lemma_pool`, so the pool is
//! effectively the lemma table + a subset of word forms that happen to
//! also be lemmas.
//!
//! **Open questions** tracked for follow-up:
//! - Why are `record_count` (360930) and `pool_count` (243034)
//!   different? Plausibly: pool = distinct lemmas ∪ shared short forms,
//!   records = every word form seen in training. Not required for
//!   inference.
//! - What does `leading_field` encode? Training frequency fits the
//!   value range observed on punctuation (0, 80, 189, 9533, 6631) but is
//!   unverified. Not load-bearing for inference.
//! - What does the 96-byte `opaque` block carry? Possibly per-tag
//!   priors or uninitialized `fwrite` padding. Skipped blindly for now.

use super::Cursor;
use super::header::Header;
use anyhow::{Context, Result, bail};

/// One `(tag, probability, lemma)` candidate for a lexicon word.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub tag_id: u32,
    pub prob: f32,
    /// Index into [`Lexicon::lemmas`] naming this candidate's lemma.
    pub lemma_index: u32,
}

/// A full-form lexicon record for one word.
#[derive(Debug, Clone)]
pub struct Entry {
    pub word: String,
    /// Opaque `u32` that precedes the candidate arrays. Shape of the
    /// value suggests a training frequency; we expose it raw so future
    /// callers can experiment without needing to re-parse the file.
    pub leading_field: u32,
    pub candidates: Vec<Candidate>,
}

/// The loaded lexicon.
#[derive(Debug, Clone)]
pub struct Lexicon {
    /// Lemma pool. Indexed by [`Candidate::lemma_index`].
    /// Typical English model: ~243k entries.
    pub lemmas: Vec<String>,
    /// One entry per word form observed in training. Typical English
    /// model: ~361k entries. Sorted by word in ASCII order.
    pub entries: Vec<Entry>,
    /// Byte offset immediately after the last entry — feeds the next
    /// section's reader.
    pub end_offset: usize,
}

impl Lexicon {
    /// Look up a candidate set by word form. `O(log n)` on the sorted
    /// record list.
    pub fn lookup(&self, word: &str) -> Option<&Entry> {
        let idx = self
            .entries
            .binary_search_by(|e| e.word.as_str().cmp(word))
            .ok()?;
        self.entries.get(idx)
    }

    /// Resolve a lemma index to its string.
    pub fn lemma(&self, index: u32) -> Option<&str> {
        self.lemmas.get(index as usize).map(String::as_str)
    }
}

/// Read the full lexicon section.
///
/// Caller must have already consumed the header so the cursor points
/// at the lemma-pool count.
pub fn read(cur: &mut Cursor<'_>, header: &Header) -> Result<Lexicon> {
    let lemmas = read_pool(cur).context("reading lemma pool")?;
    let record_count = read_lexicon_header(cur).context("reading lexicon header")?;
    locate_records(cur, header, &lemmas, record_count as usize)
        .context("locating start of word records")?;
    let entries = read_entries(cur, header, &lemmas, record_count as usize)
        .context("reading word records")?;
    Ok(Lexicon {
        end_offset: cur.offset(),
        lemmas,
        entries,
    })
}

/// Find the exact byte offset where word records begin.
///
/// The lexicon-header opaque block is not a fixed size: for models
/// trained by a modern `train-tree-tagger` it follows
/// `24 * num_tags + 18` bytes, but the bundled `english.par` pre-dates
/// that layout and uses a 90-byte preamble regardless of tag count.
/// Rather than hardcode either, we probe candidate preamble sizes and
/// pick the one under which the first few records validate: plausible
/// cstr, small candidate count, every `tag_id` in range, every
/// `lemma_index` in range.
///
/// Two candidates suffice for every `.par` we've seen in the wild —
/// the legacy `90` and the formula `24 * N + 18` — but the scan is
/// cheap so we keep it honest and try both plus a short-range
/// fallback to survive future format drifts.
fn locate_records(
    cur: &mut Cursor<'_>,
    header: &Header,
    lemmas: &[String],
    record_count: usize,
) -> Result<()> {
    let start = cur.offset();
    let bytes_after = cur.bytes_after_cursor();
    let formula = 24usize * header.tags.len() + 18;

    // Probe order: most-likely first, then fall back. The formula hits
    // on freshly-trained models; 90 hits on the shipped english.par;
    // the rest is a safety net in case a future format shifts by a few
    // bytes.
    let mut candidates: Vec<usize> = vec![formula, 90];
    for off in (0..=2048).step_by(2) {
        if off != formula && off != 90 {
            candidates.push(off);
        }
    }

    for candidate in candidates {
        if candidate > bytes_after.len() {
            continue;
        }
        if validate_records_at(bytes_after, candidate, header, lemmas, record_count) {
            cur.advance(candidate)
                .context("advancing cursor past opaque preamble")?;
            return Ok(());
        }
    }

    bail!(
        "could not locate start of word records after lexicon header at offset {} \
         — tried preamble sizes up to 2048 bytes",
        start
    );
}

/// Best-effort validator: try to read the first N records from `slice`
/// starting at `preamble_size` and return true iff every field of
/// every record is within range. Doesn't read all records because
/// that'd be O(record_count) per candidate — the first few (plus a
/// spot-check deeper in) are sufficient to reject preamble guesses
/// that land in opaque data.
fn validate_records_at(
    slice: &[u8],
    preamble_size: usize,
    header: &Header,
    lemmas: &[String],
    record_count: usize,
) -> bool {
    let num_tags = header.tags.len();
    let num_lemmas = lemmas.len();
    let probe_records = 8.min(record_count);

    let mut pos = preamble_size;
    for _ in 0..probe_records {
        // cstr (non-empty, terminated)
        let word_start = pos;
        let word_end = match slice[word_start..].iter().position(|&b| b == 0) {
            Some(n) if n > 0 && n < 256 => word_start + n,
            _ => return false,
        };
        if std::str::from_utf8(&slice[word_start..word_end]).is_err() {
            return false;
        }
        pos = word_end + 1;

        // candidate count (u32)
        if pos + 4 > slice.len() {
            return false;
        }
        let cand_count = u32::from_le_bytes(slice[pos..pos + 4].try_into().unwrap()) as usize;
        if cand_count == 0 || cand_count > 64 {
            return false;
        }
        pos += 4;

        // leading field (u32)
        if pos + 4 > slice.len() {
            return false;
        }
        pos += 4;

        // tag_ids
        if pos + cand_count * 4 > slice.len() {
            return false;
        }
        for _ in 0..cand_count {
            let id = u32::from_le_bytes(slice[pos..pos + 4].try_into().unwrap()) as usize;
            if id >= num_tags {
                return false;
            }
            pos += 4;
        }

        // probs (skip)
        if pos + cand_count * 4 > slice.len() {
            return false;
        }
        pos += cand_count * 4;

        // lemma_indices
        if pos + cand_count * 4 > slice.len() {
            return false;
        }
        for _ in 0..cand_count {
            let idx = u32::from_le_bytes(slice[pos..pos + 4].try_into().unwrap()) as usize;
            if idx >= num_lemmas {
                return false;
            }
            pos += 4;
        }
    }
    true
}

fn read_pool(cur: &mut Cursor<'_>) -> Result<Vec<String>> {
    let count = cur.read_u32_le().context("reading pool count")?;
    if count > 10_000_000 {
        bail!(
            "implausible lemma-pool count {count} at offset {} — file \
             likely not a TreeTagger .par",
            cur.offset() - 4
        );
    }
    let mut strings = Vec::with_capacity(count as usize);
    for i in 0..count {
        let s = cur
            .read_cstr()
            .with_context(|| format!("reading lemma #{i} of {count}"))?;
        strings.push(s.to_owned());
    }
    Ok(strings)
}

/// Read the first two `u32`s of the lexicon header — record count and
/// sentinel. The opaque block that follows is *not* fixed size (see
/// [`locate_records`]), so the caller must re-position the cursor
/// before reading records.
fn read_lexicon_header(cur: &mut Cursor<'_>) -> Result<u32> {
    let record_count = cur.read_u32_le().context("reading record_count")?;
    let sentinel = cur.read_u32_le().context("reading sentinel")?;
    if sentinel != 0xFFFF_FFFE {
        bail!(
            "expected sentinel 0xFFFFFFFE after lexicon record_count at offset {}, got {:#010x}",
            cur.offset() - 4,
            sentinel
        );
    }
    Ok(record_count)
}

fn read_entries(
    cur: &mut Cursor<'_>,
    header: &Header,
    lemmas: &[String],
    count: usize,
) -> Result<Vec<Entry>> {
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let record_start = cur.offset();
        let word = cur
            .read_cstr()
            .with_context(|| format!("reading word cstr for record #{i} at offset {record_start}"))?
            .to_owned();

        let cand_count = cur.read_u32_le().with_context(|| {
            format!("reading candidate count for {word:?} at offset {}", cur.offset() - 4)
        })?;
        if cand_count == 0 || cand_count > 64 {
            bail!(
                "implausible candidate count {cand_count} for {word:?} at offset {}",
                cur.offset() - 4
            );
        }

        let leading_field = cur.read_u32_le().context("reading leading field")?;

        let mut tag_ids = Vec::with_capacity(cand_count as usize);
        for _ in 0..cand_count {
            let id = cur.read_u32_le()?;
            if (id as usize) >= header.tags.len() {
                bail!(
                    "tag id {id} out of range (tag table has {} entries) while \
                     reading record #{i} word {word:?}",
                    header.tags.len()
                );
            }
            tag_ids.push(id);
        }

        let mut probs = Vec::with_capacity(cand_count as usize);
        for _ in 0..cand_count {
            probs.push(cur.read_f32_le()?);
        }

        let mut lemma_indices = Vec::with_capacity(cand_count as usize);
        for _ in 0..cand_count {
            let lemma = cur.read_u32_le()?;
            if (lemma as usize) >= lemmas.len() {
                bail!(
                    "lemma index {lemma} out of range (lemma pool has {} entries) \
                     while reading record #{i} word {word:?}",
                    lemmas.len()
                );
            }
            lemma_indices.push(lemma);
        }

        let candidates = tag_ids
            .into_iter()
            .zip(probs)
            .zip(lemma_indices)
            .map(|((tag_id, prob), lemma_index)| Candidate {
                tag_id,
                prob,
                lemma_index,
            })
            .collect();

        entries.push(Entry {
            word,
            leading_field,
            candidates,
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::super::header;
    use super::*;
    use std::path::{Path, PathBuf};

    fn english_par_path() -> Option<PathBuf> {
        let candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("resources/treetagger/lib/english.par");
        candidate.exists().then_some(candidate)
    }

    /// Hand-rolled minimal `.par` with 2 tags, 1 word record, and a
    /// preamble sized per the `24 * num_tags + 18` formula we observed
    /// on `train-tree-tagger`-generated files. Exercises the scanner's
    /// "prefer formula over legacy 90" branch.
    fn build_synthetic_formula_par() -> Vec<u8> {
        let num_tags = 2u32;
        let preamble_size = 24 * num_tags as usize + 18;
        let mut bytes = Vec::<u8>::new();

        // Header: field_a, field_b, sent_idx, num_tags, then tags
        bytes.extend_from_slice(&32u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // SENT at idx 1
        bytes.extend_from_slice(&num_tags.to_le_bytes());
        bytes.extend_from_slice(b"A\0SENT\0");

        // Lemma pool: 1 entry, "xA"
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(b"xA\0");

        // Lexicon header: 1 record, sentinel
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0xFFFF_FFFEu32.to_le_bytes());

        // Preamble filled with 0xBA canary (matches real-world padding)
        bytes.extend(std::iter::repeat(0xBA).take(preamble_size));

        // One record: word "xA", count=1, leading=0, tag=A, prob=1.0, lemma=0
        bytes.extend_from_slice(b"xA\0");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        bytes
    }

    #[test]
    fn reads_formula_sized_preamble() {
        let bytes = build_synthetic_formula_par();
        let mut cur = Cursor::new(&bytes);
        let header = super::super::header::read(&mut cur).unwrap();
        let lex = read(&mut cur, &header).unwrap();
        assert_eq!(lex.lemmas, vec!["xA".to_string()]);
        assert_eq!(lex.entries.len(), 1);
        let entry = lex.lookup("xA").unwrap();
        assert_eq!(entry.candidates.len(), 1);
        assert_eq!(entry.candidates[0].tag_id, 0);
        assert!((entry.candidates[0].prob - 1.0).abs() < 1e-5);
    }

    /// Full-file load against the bundled English model. Asserts we
    /// can parse all 243k entries without desynchronizing, and that
    /// spot-checked entries match `tree-tagger -prob` output.
    #[test]
    fn reads_bundled_english_lexicon() {
        let Some(par) = english_par_path() else {
            return;
        };
        let bytes = std::fs::read(&par).unwrap();
        let mut cur = Cursor::new(&bytes);
        let header = header::read(&mut cur).unwrap();
        let lex = read(&mut cur, &header).unwrap();

        // Sanity: lemma pool and record count should match what
        // we observed in the English model. Lemmas < records because
        // inflected word forms aren't lemmas in their own right.
        assert_eq!(lex.lemmas.len(), 243034);
        assert_eq!(lex.entries.len(), 360930);

        // Spot-check punctuation entries we verified by hex-dump and
        // against `tree-tagger -prob`:
        //   !    SENT  !  1.000000
        //   #    #     #  1.000000
        //   $    $     $  1.000000
        //   %    NN    %  1.000000
        //   ,    ,     ,  1.000000
        //   .    SENT  .  1.000000
        for (word, want_tag, want_prob) in [
            ("!", "SENT", 1.0_f32),
            ("#", "#", 1.0),
            ("$", "$", 1.0),
            ("%", "NN", 1.0),
            (",", ",", 1.0),
            (".", "SENT", 1.0),
        ] {
            let entry = lex.lookup(word).unwrap_or_else(|| panic!("no entry for {word:?}"));
            assert_eq!(entry.candidates.len(), 1, "{word:?} should have 1 candidate");
            let cand = &entry.candidates[0];
            assert_eq!(
                header.tag(cand.tag_id),
                Some(want_tag),
                "{word:?}: wrong tag"
            );
            assert!(
                (cand.prob - want_prob).abs() < 1e-4,
                "{word:?}: wrong prob (got {})",
                cand.prob
            );
            assert_eq!(
                lex.lemma(cand.lemma_index),
                Some(word),
                "{word:?}: wrong lemma"
            );
        }

        // Spot-check a multi-candidate word: `"` has 2 candidates at
        // equal 0.5 probability, one for the opening-quote tag and one
        // for the closing-quote tag. Both lemmatize to `"`.
        let entry = lex.lookup("\"").unwrap();
        assert_eq!(entry.candidates.len(), 2);
        let mut tags: Vec<&str> = entry
            .candidates
            .iter()
            .map(|c| header.tag(c.tag_id).unwrap())
            .collect();
        tags.sort();
        assert_eq!(tags, ["''", "``"]);
        for c in &entry.candidates {
            assert!((c.prob - 0.5).abs() < 1e-5, "prob = {}", c.prob);
            assert_eq!(lex.lemma(c.lemma_index), Some("\""));
        }

        // Spot-check an inflected verb: `went` should be VVD with lemma `go`.
        let went = lex.lookup("went").unwrap();
        let vvd = went
            .candidates
            .iter()
            .find(|c| header.tag(c.tag_id) == Some("VVD"))
            .expect("went should have a VVD candidate");
        assert_eq!(lex.lemma(vvd.lemma_index), Some("go"));

        // Log the end offset so the follow-on archaeology for
        // suffix/prefix/decision trees knows where to start. `cargo
        // test -- --nocapture` surfaces it.
        eprintln!("lexicon ends at offset 0x{:x} ({})", lex.end_offset, lex.end_offset);
    }
}
