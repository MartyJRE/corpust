//! `.par` affix tries — prefix + suffix + shared prob-tag array.
//!
//! Structural layout (reverse-engineered on `english.par` 2026-04-23).
//!
//! The slab between the lexicon and the decision tree holds three
//! logical regions in fixed order:
//!
//! ```text
//! offset (slab-relative)   size            kind
//! -----------------------------------------------------------------
//! 0x000000 ..              ~0.5 KB         pre-trie prelude (opaque)
//! 0x0001d2 .. 0x008e72     ~36 KB          PREFIX trie, 3000 entries × 12 B
//! 0x008e73 .. 0x0141aa     ~46 KB          shared prob-tag array, 5735 × 8 B
//! 0x0141ac .. 0x01d254     ~37 KB          SUFFIX trie, 3086 entries × 12 B
//! ```
//!
//! (Prior plan notes had the trie offsets slightly off; the entries'
//! own `0xBABABABA` footer makes the real boundaries trivial to find
//! by scanning.)
//!
//! ## Entry layout — both tries
//!
//! ```text
//! 12 bytes per entry:
//!   u16  flag            // 0 = interior (has children),
//!                        // 1 = leaf (no children; `offset` is the
//!                        //          sentinel 47802)
//!   u16  char            // character (UTF-16 LE). Real chars for
//!                        //   deeper entries; the trie root has a
//!                        //   header entry with `char = 0x0101` as
//!                        //   a "root" marker — see [`Trie::root_index`].
//!   u16  count           // for flag=0: number of child entries.
//!                        //   for flag=1: undecoded (small int, possibly
//!                        //   a prob-record count — see the
//!                        //   "distributions are unresolved" note).
//!   u16  offset          // for flag=0: entries-array index of the
//!                        //   first child. Children are contiguous
//!                        //   in file order: entries[offset..offset+count].
//!   u32  0xBABABABA      // footer / canary.
//! ```
//!
//! ## Reading the suffix trie recursively yields **real English
//! word endings**: `s`, `ses`, `sesi` (= "...ises"), `seit`
//! (= "...ties"), `sred` (= "...ders"), etc. That's what pins down
//! the structural interpretation above.
//!
//! ## Distributions are still unresolved
//!
//! Each trie node should carry a `P(tag | this-suffix)` distribution
//! for unknown-word tagging. The `count` and `offset` fields of
//! flag=0 entries turned out to be structural (children-pointer), not
//! prob-tag-array pointers — so the linkage from a trie node to its
//! distribution isn't the simple `(count, offset)` the plan assumed.
//! The 5735-record prob-tag array has the distributions, but mapping
//! trie-node → prob-array-slice needs more archaeology.
//!
//! For now this module exposes the structural reader only. Unknown-
//! word tag guessing is still gated on the missing prob-array linkage.

use super::Cursor;
use anyhow::{Context, Result, bail};

/// One 12-byte trie entry.
#[derive(Debug, Clone, Copy)]
pub struct TrieEntry {
    pub flag: u16,
    /// Character matched at this edge. For the root marker it's
    /// `0x0101` (not a real char).
    pub char: u16,
    pub count: u16,
    pub offset: u16,
}

impl TrieEntry {
    pub fn is_leaf(&self) -> bool {
        self.flag == 1
    }

    pub fn as_char(&self) -> Option<char> {
        char::from_u32(self.char as u32).filter(|c| !c.is_control())
    }
}

/// A suffix or prefix trie.
#[derive(Debug, Clone)]
pub struct Trie {
    /// All entries in file order. Entry `root_index` is the root
    /// marker; entries `[root_index+1 .. root_index+1+count]` are
    /// the root's direct children.
    pub entries: Vec<TrieEntry>,
    /// The index into `entries` of the root-marker entry. Typically
    /// `0`.
    pub root_index: usize,
}

impl Trie {
    pub fn root(&self) -> &TrieEntry {
        &self.entries[self.root_index]
    }

    /// Iterate the direct children of `entry`. Honors the
    /// entries-index semantics of `offset` for flag=0 nodes.
    pub fn children(&self, entry: &TrieEntry) -> &[TrieEntry] {
        if entry.is_leaf() {
            return &[];
        }
        let start = entry.offset as usize;
        let end = start + entry.count as usize;
        if end > self.entries.len() {
            return &[];
        }
        &self.entries[start..end]
    }
}

/// One record in the shared prob-tag array.
///
/// Flag is always 0 on the records we've observed; `canary` is always
/// `0xBABA`. `prob` is `P(tag | some-trie-node-context)` — *which*
/// trie node owns which slice of this array is still open (see module
/// docs).
#[derive(Debug, Clone, Copy)]
pub struct ProbTagRecord {
    pub flag: u8,
    pub canary: u16,
    pub prob: f32,
    pub tag_id: u8,
}

/// Shared prob-tag array (one pool shared by both tries).
#[derive(Debug, Clone)]
pub struct ProbTagArray {
    pub records: Vec<ProbTagRecord>,
}

/// Fully loaded affix-tries slab.
#[derive(Debug, Clone)]
pub struct Tries {
    pub prefix: Trie,
    pub prob_tag: ProbTagArray,
    pub suffix: Trie,
}

/// Parse the tries slab from the cursor's current position to the
/// given end offset (== start of the decision-tree section).
pub fn read(cur: &mut Cursor<'_>, slab_end: usize) -> Result<Tries> {
    let slab_start = cur.offset();
    if slab_end <= slab_start {
        bail!("tries slab end {slab_end} must be past cursor offset {slab_start}");
    }
    let bytes = cur.bytes_after_cursor();
    let slab_len = slab_end - slab_start;
    if slab_len > bytes.len() {
        bail!("tries slab length {slab_len} exceeds remaining cursor bytes {}", bytes.len());
    }
    let slab = &bytes[..slab_len];

    // Scan for 3-in-a-row 12-byte entries with the 0xBABABABA footer.
    // That finds the first trie's start. Then walk entries until the
    // footer pattern breaks — that's the first trie's end.
    let prefix_start = find_entry_run(slab, 0)
        .context("could not locate prefix trie — no 3-run of 0xBABABABA-footer 12-byte entries")?;
    let prefix_entries = walk_entries(slab, prefix_start);
    let prefix_end = prefix_start + prefix_entries.len() * 12;

    // Prob-tag array runs from prefix_end to the suffix trie start.
    let suffix_start = find_entry_run(slab, prefix_end)
        .context("could not locate suffix trie after prefix trie + prob-tag array")?;
    let prob_tag_bytes = &slab[prefix_end..suffix_start];
    let prob_tag = parse_prob_tag_array(prob_tag_bytes);

    let suffix_entries = walk_entries(slab, suffix_start);
    let suffix_end = suffix_start + suffix_entries.len() * 12;

    if suffix_end > slab_len {
        bail!(
            "suffix trie at slab+{suffix_start} would run past slab end \
             (slab_len={slab_len}, suffix_end={suffix_end})"
        );
    }

    let prefix = build_trie(prefix_entries)?;
    let suffix = build_trie(suffix_entries)?;

    // Advance cursor to slab_end so downstream callers can pick up.
    cur.advance(slab_len)
        .context("advancing cursor past tries slab")?;

    Ok(Tries {
        prefix,
        prob_tag,
        suffix,
    })
}

fn is_entry_at(slab: &[u8], off: usize) -> bool {
    if off + 12 > slab.len() {
        return false;
    }
    let footer = u32::from_le_bytes([
        slab[off + 8],
        slab[off + 9],
        slab[off + 10],
        slab[off + 11],
    ]);
    footer == 0xBABABABA
}

fn find_entry_run(slab: &[u8], start: usize) -> Option<usize> {
    let mut off = start;
    while off + 36 <= slab.len() {
        if is_entry_at(slab, off) && is_entry_at(slab, off + 12) && is_entry_at(slab, off + 24) {
            return Some(off);
        }
        off += 1;
    }
    None
}

fn walk_entries(slab: &[u8], start: usize) -> Vec<TrieEntry> {
    let mut out = Vec::new();
    let mut off = start;
    while is_entry_at(slab, off) {
        let flag = u16::from_le_bytes([slab[off], slab[off + 1]]);
        let char = u16::from_le_bytes([slab[off + 2], slab[off + 3]]);
        let count = u16::from_le_bytes([slab[off + 4], slab[off + 5]]);
        let offset = u16::from_le_bytes([slab[off + 6], slab[off + 7]]);
        out.push(TrieEntry {
            flag,
            char,
            count,
            offset,
        });
        off += 12;
    }
    out
}

fn build_trie(entries: Vec<TrieEntry>) -> Result<Trie> {
    if entries.is_empty() {
        bail!("cannot build a trie from zero entries");
    }
    // The header / root entry is entry 0 by convention. Verify by
    // checking that its `offset` points just past itself (= 1) and
    // its `count` children fit inside the entries vector.
    let root = &entries[0];
    if root.flag != 0 {
        bail!(
            "trie root is marked as a leaf (flag=1), which makes no sense \
             for a multi-entry trie"
        );
    }
    let root_end = root.offset as usize + root.count as usize;
    if root_end > entries.len() {
        bail!(
            "trie root claims {} children starting at entries index {}, \
             but there are only {} entries total",
            root.count,
            root.offset,
            entries.len()
        );
    }
    Ok(Trie {
        entries,
        root_index: 0,
    })
}

fn parse_prob_tag_array(bytes: &[u8]) -> ProbTagArray {
    let mut records = Vec::with_capacity(bytes.len() / 8);
    let mut off = 0;
    while off + 8 <= bytes.len() {
        let flag = bytes[off];
        let canary = u16::from_le_bytes([bytes[off + 1], bytes[off + 2]]);
        let prob = f32::from_le_bytes([
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
        ]);
        let tag_id = bytes[off + 7];
        // Keep the record regardless of whether it looks clean; the
        // array is known to contain filler bytes (0xBA fill) between
        // real records. Callers filter by `flag == 0 && canary == 0xBABA`.
        records.push(ProbTagRecord {
            flag,
            canary,
            prob,
            tag_id,
        });
        off += 8;
    }
    ProbTagArray { records }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn english_par_path() -> Option<PathBuf> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("resources/treetagger/lib/english.par");
        p.exists().then_some(p)
    }

    #[test]
    fn reads_bundled_english_tries() {
        let Some(par) = english_par_path() else {
            return;
        };
        let bytes = std::fs::read(&par).unwrap();
        const SLAB_START: usize = 0xcf9cc3;
        const DTREE_START: usize = 0xd231bb;
        let mut cur = Cursor::new(&bytes);
        cur.advance(SLAB_START).unwrap();
        let tries = read(&mut cur, DTREE_START).unwrap();

        // Sizes matching earlier archaeology:
        assert_eq!(tries.prefix.entries.len(), 3000, "prefix trie entry count");
        assert_eq!(tries.suffix.entries.len(), 3086, "suffix trie entry count");
        // Prob-tag array: some thousands of records. Prior note:
        // plan said ~5738, count here is the raw 8-byte-stride parse.
        assert!(
            tries.prob_tag.records.len() > 5000,
            "expected >5k prob-tag records, got {}",
            tries.prob_tag.records.len()
        );

        // Suffix trie root: 61 children. The first few are the
        // most-frequent word-ending letters in English: 's', 'e',
        // 'n', 'd', 'y', ...
        let root = tries.suffix.root();
        assert_eq!(root.count, 61, "suffix trie root child count");
        let kids = tries.suffix.children(root);
        assert_eq!(kids.len(), 61);
        let first_chars: Vec<char> = kids.iter().take(5).filter_map(|e| e.as_char()).collect();
        assert_eq!(first_chars, vec!['s', 'e', 'n', 'd', 'y']);

        // Suffix trie should produce real English endings when walked.
        // Pick 's' → 'e' → 's' — walk should land at children that are
        // valid (flag=0) or leaves (flag=1) with character continuations
        // that build a real suffix.
        let s_node = kids[0]; // 's'
        assert_eq!(s_node.as_char(), Some('s'));
        let s_kids = tries.suffix.children(&s_node);
        // 's' has 23 children per `count`
        assert_eq!(s_kids.len(), 23);
        // First child of 's' is 'e' (suffix 'es')
        assert_eq!(s_kids[0].as_char(), Some('e'));
        let es_node = s_kids[0];
        let es_kids = tries.suffix.children(&es_node);
        // 'es' has 20 children
        assert_eq!(es_kids.len(), 20);
        // First child is 's' — so we've built the suffix "ses" walking s → e → s
        assert_eq!(es_kids[0].as_char(), Some('s'));

        // Prefix trie root sanity.
        let proot = tries.prefix.root();
        assert_eq!(
            proot.count, 61,
            "prefix trie root child count (same marker format as suffix)"
        );
    }
}
