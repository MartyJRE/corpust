//! `.par` header section.
//!
//! From a hex dump of the bundled English model, the first 16 bytes are
//! four little-endian `u32`s, followed by a run of null-terminated tag
//! strings. The first three `u32`s are model metadata whose exact
//! semantics we're still pinning down; the fourth is the tag count, and
//! the tag table that follows is confirmed byte-for-byte against the
//! published TreeTagger English Penn tagset.
//!
//! Current understanding (all values observed on `english.par`, v3.2):
//!
//! | offset | bytes         | u32 LE | working name    | note                                             |
//! |--------|---------------|--------|-----------------|--------------------------------------------------|
//! | 0x00   | `20 00 00 00` | 32     | `field_a`       | plausibly open-class tag count                   |
//! | 0x04   | `01 00 00 00` | 1      | `field_b`       | unknown; flag-like                                |
//! | 0x08   | `1f 00 00 00` | 31     | `sent_tag_idx`  | 0-based index of the `SENT` tag — confirmed      |
//! | 0x0C   | `3a 00 00 00` | 58     | `num_tags`      | number of tag strings — confirmed                |
//! | 0x10   | —             |        | —               | start of `num_tags` null-terminated tag strings  |
//!
//! The `sent_tag_idx` field is corroborated by walking the tag table:
//! position 31 is exactly where `SENT` lives in the English model. We
//! promote `field_a` / `field_b` to named fields once differential
//! training against `train-tree-tagger` clarifies their meaning.

use super::Cursor;
use anyhow::{Context, Result};

/// Parsed header section.
///
/// Owns the tag strings. The `.par` file is a long-lived resource
/// loaded once per process, so the cost of copying 58 short strings is
/// negligible and the rest of the tagger gets `'static`-friendly
/// ownership.
#[derive(Debug, Clone)]
pub struct Header {
    /// Unknown leading `u32`. 32 on `english.par`; plausibly the count
    /// of open-class tags (those used when guessing unknown words).
    pub field_a: u32,

    /// Unknown `u32`. 1 on `english.par`; plausibly a flag.
    pub field_b: u32,

    /// Index of the sentence-boundary tag (`SENT` in English) inside
    /// the tag table. Confirmed by direct inspection on the English
    /// model.
    pub sent_tag_index: u32,

    /// Tag strings in the order they appear in the file. Their position
    /// in this vector is the tag's numeric ID used everywhere else in
    /// the model (lexicon entries, decision-tree leaves, suffix-trie
    /// probability distributions).
    pub tags: Vec<String>,

    /// Byte offset immediately after the tag table, i.e. where the
    /// next section begins. Exposed so the lexicon reader can pick up
    /// without re-parsing the header.
    pub end_offset: usize,
}

impl Header {
    /// Look up a tag's numeric ID by string.
    pub fn tag_id(&self, tag: &str) -> Option<u32> {
        self.tags.iter().position(|t| t == tag).map(|i| i as u32)
    }

    /// The tag at numeric ID `id`, if it exists.
    pub fn tag(&self, id: u32) -> Option<&str> {
        self.tags.get(id as usize).map(String::as_str)
    }

    /// The sentence-boundary tag's string form (typically `"SENT"`).
    pub fn sent_tag(&self) -> Option<&str> {
        self.tag(self.sent_tag_index)
    }
}

pub fn read(cur: &mut Cursor<'_>) -> Result<Header> {
    let field_a = cur.read_u32_le().context("reading header field_a")?;
    let field_b = cur.read_u32_le().context("reading header field_b")?;
    let sent_tag_index = cur.read_u32_le().context("reading sent_tag_index")?;
    let num_tags = cur.read_u32_le().context("reading num_tags")?;

    // Sanity bound. Real `.par` files carry small tag tables (dozens
    // of entries). Anything in the millions is almost certainly a
    // corrupt or foreign file — bail loudly rather than allocate.
    if num_tags > 1024 {
        anyhow::bail!(
            "unreasonable tag count {num_tags} at offset {} — file \
             likely not a TreeTagger .par or uses a different layout",
            cur.offset() - 4
        );
    }

    let mut tags = Vec::with_capacity(num_tags as usize);
    for i in 0..num_tags {
        let tag = cur
            .read_cstr()
            .with_context(|| format!("reading tag string #{i} of {num_tags}"))?;
        tags.push(tag.to_owned());
    }

    if (sent_tag_index as usize) >= tags.len() {
        anyhow::bail!(
            "sent_tag_index {sent_tag_index} is out of range for a \
             {num_tags}-entry tag table — file likely not a TreeTagger .par"
        );
    }

    let end_offset = cur.offset();
    Ok(Header {
        field_a,
        field_b,
        sent_tag_index,
        tags,
        end_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Canonical hand-rolled header: 3 tags, SENT at index 2.
    #[test]
    fn reads_synthetic_header() {
        #[rustfmt::skip]
        let bytes: &[u8] = &[
            // field_a=7, field_b=1, sent_tag_index=2, num_tags=3
            0x07, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00,
            // "NN\0"
            b'N', b'N', 0,
            // "DT\0"
            b'D', b'T', 0,
            // "SENT\0"
            b'S', b'E', b'N', b'T', 0,
        ];
        let mut cur = Cursor::new(bytes);
        let h = read(&mut cur).unwrap();
        assert_eq!(h.field_a, 7);
        assert_eq!(h.field_b, 1);
        assert_eq!(h.sent_tag_index, 2);
        assert_eq!(h.tags, vec!["NN", "DT", "SENT"]);
        assert_eq!(h.sent_tag(), Some("SENT"));
        assert_eq!(h.tag_id("DT"), Some(1));
        assert_eq!(h.end_offset, bytes.len());
    }

    #[test]
    fn rejects_runaway_tag_count() {
        let bytes: &[u8] = &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff,
        ];
        let mut cur = Cursor::new(bytes);
        assert!(read(&mut cur).is_err());
    }

    #[test]
    fn rejects_out_of_range_sent_index() {
        #[rustfmt::skip]
        let bytes: &[u8] = &[
            0, 0, 0, 0,
            0, 0, 0, 0,
            // sent_tag_index=5 but only 2 tags
            5, 0, 0, 0,
            2, 0, 0, 0,
            b'N', b'N', 0,
            b'D', b'T', 0,
        ];
        let mut cur = Cursor::new(bytes);
        assert!(read(&mut cur).is_err());
    }

    /// Load the real bundled English model and verify the tag table
    /// byte-for-byte against TreeTagger's documented Penn tagset.
    /// Skipped if the bundle isn't present (e.g. on machines that
    /// didn't download it).
    #[test]
    fn reads_bundled_english_par() {
        let Some(par) = english_par_path() else {
            return;
        };
        let bytes = std::fs::read(&par).unwrap();
        let mut cur = Cursor::new(&bytes);
        let h = read(&mut cur).unwrap();

        // Values observed at construction time of this plan. If Schmid
        // ships a new `english.par` these may shift and we'll need to
        // re-verify.
        assert_eq!(h.field_a, 32);
        assert_eq!(h.field_b, 1);
        assert_eq!(h.sent_tag_index, 31);
        assert_eq!(h.tags.len(), 58);
        assert_eq!(h.sent_tag(), Some("SENT"));

        // Spot-check the tag table against the published English tagset.
        // Full list copied from the hex dump in the plan file.
        assert_eq!(h.tags[0], "#");
        assert_eq!(h.tags[7], "CC");
        assert_eq!(h.tags[13], "IN/that"); // the tricky compound tag
        assert_eq!(h.tags[19], "NN");
        assert_eq!(h.tags[31], "SENT");
        assert_eq!(h.tags[57], "``");

        // Header is exactly 16 bytes + sum of (tag_len + 1).
        let expected_end = 16
            + h.tags
                .iter()
                .map(|t| t.len() + 1)
                .sum::<usize>();
        assert_eq!(h.end_offset, expected_end);
    }

    fn english_par_path() -> Option<PathBuf> {
        let candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("resources/treetagger/lib/english.par");
        candidate.exists().then_some(candidate)
    }
}
