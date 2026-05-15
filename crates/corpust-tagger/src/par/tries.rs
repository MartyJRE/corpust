//! `.par` affix tries — prefix + suffix + two prob-tag arrays.
//!
//! Structural layout (reverse-engineered on `english.par` 2026-04-23).
//!
//! The slab between the lexicon and the decision tree holds:
//!
//! ```text
//! offset (slab-rel.)   size       kind
//! ----------------------------------------------------------------------
//! 0x000000 ..          ~466 B     prelude: 58 × f64 (per-tag stats)
//! 0x0001d2 .. 0x008e72 ~36 KB     PREFIX trie, 3000 × 12-byte entries
//! 0x008e73 .. 0x0141aa ~46 KB     prob-array-1 (preceded by 1-byte 0x15)
//! 0x0141ac .. 0x01d254 ~37 KB     SUFFIX trie, 3086 × 12-byte entries
//! 0x01d255 .. 0x0294f8 ~50 KB     prob-array-2 (preceded by 1-byte 0x15)
//! ```
//!
//! Key facts:
//!
//! - **Each trie's entries form a binary parent-children tree.** The
//!   header entry (at index 0) has `count=61, offset=1`: 61 direct
//!   children starting at entries index 1. Any entry with `flag=0`
//!   is interior and has `count` children at `entries[offset .. offset+count]`.
//!   Entries with `flag=1` are leaves.
//! - **Each trie has its own prob-tag array.** Prefix trie's array is
//!   the one between the tries; suffix trie's array is the one between
//!   the suffix trie and the dtree section.
//! - **Each prob-array is preceded by a single `0x15` byte** that
//!   doesn't belong to any record — skip it before decoding.
//! - **Each prob-array segments cleanly into distributions.** Records
//!   within a distribution are sorted descending by probability; a
//!   new distribution begins when the probability jumps back up or
//!   the running sum reaches 1. Every segment sums to exactly 1.0.
//! - **Leaves map 1:1 to distributions by walk-order.** Pre-order DFS
//!   walk of the trie visits leaves in the same order the
//!   distributions appear in the prob-array. The N-th leaf
//!   encountered uses the N-th segment as its `P(tag | this suffix)`.
//!
//! ## Record layout (unchanged)
//!
//! Trie entries (12 bytes):
//!
//! ```text
//! u16 flag     // 0 = interior, 1 = leaf (offset sentinel 47802)
//! u16 char     // UTF-16 LE char on this edge
//! u16 count    // flag=0: child count.  flag=1: small, TBD.
//! u16 offset   // flag=0: entries-array index of first child
//! u32 0xBABABABA
//! ```
//!
//! Prob-array records (8 bytes):
//!
//! ```text
//! u8  flag=0
//! u16 canary=0xBABA
//! f32 prob
//! u8  tag_id
//! ```

use super::Cursor;
use super::header::Header;
use anyhow::{Context, Result, bail};

/// One trie-edge entry.
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

/// One `(tag, prob)` pair inside a distribution.
#[derive(Debug, Clone, Copy)]
pub struct TagProb {
    pub tag_id: u8,
    pub prob: f32,
}

/// A probability distribution — a slice of descending-sorted
/// `(tag, prob)` pairs summing to `~1.0`.
#[derive(Debug, Clone)]
pub struct Distribution {
    pub probs: Vec<TagProb>,
}

impl Distribution {
    /// Tag with the largest probability in the distribution.
    pub fn peak(&self) -> Option<TagProb> {
        self.probs.iter().max_by(|a, b| {
            a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal)
        }).copied()
    }
}

/// A suffix or prefix trie, with every leaf already pointing at its
/// distribution.
#[derive(Debug, Clone)]
pub struct Trie {
    /// All entries in file order.
    pub entries: Vec<TrieEntry>,
    /// Root marker index (typically 0).
    pub root_index: usize,
    /// For each entry index, the index into `distributions` if this
    /// entry is a leaf, else `None`.
    pub leaf_dist: Vec<Option<usize>>,
    /// Leaf distributions in walk-order. Parallel to the sequence
    /// of leaves produced by a pre-order DFS from `root_index`.
    pub distributions: Vec<Distribution>,
}

impl Trie {
    pub fn root(&self) -> &TrieEntry {
        &self.entries[self.root_index]
    }

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

    /// For an unknown word, walk the trie (suffix trie: consume the
    /// word's trailing characters; prefix trie: consume the leading
    /// characters) and return the distribution at the deepest leaf
    /// reached. Returns `None` if the walk never hits a leaf — e.g.
    /// the first char has no matching child, or the word is shorter
    /// than the deepest path.
    ///
    /// `chars` is an iterator of the characters in the order they
    /// should be matched (last-to-first for suffix trie,
    /// first-to-last for prefix trie).
    pub fn lookup<I: IntoIterator<Item = char>>(&self, chars: I) -> Option<&Distribution> {
        let mut cur = self.root_index;
        let mut best: Option<usize> = None;
        for ch in chars {
            let entry = &self.entries[cur];
            let kids = self.children(entry);
            let found = kids
                .iter()
                .position(|e| e.as_char() == Some(ch));
            let Some(pos) = found else { break };
            let child_idx = entry.offset as usize + pos;
            if let Some(d) = self.leaf_dist[child_idx] {
                best = Some(d);
                break; // leaves have no children to descend into
            }
            cur = child_idx;
        }
        best.and_then(|d| self.distributions.get(d))
    }
}

/// Shared header + the raw prob-array records for debugging.
#[derive(Debug, Clone)]
pub struct ProbTagArray {
    /// Raw 8-byte records as parsed. Includes any junk/filler.
    pub records: Vec<ProbTagRecord>,
}

/// One 8-byte record in a prob-tag array.
#[derive(Debug, Clone, Copy)]
pub struct ProbTagRecord {
    pub flag: u8,
    pub canary: u16,
    pub prob: f32,
    pub tag_id: u8,
}

/// Fully loaded affix-tries slab.
#[derive(Debug, Clone)]
pub struct Tries {
    pub prefix: Trie,
    pub suffix: Trie,
    /// Raw prob-array-1 records for diagnostic tooling. Segmented
    /// into distributions inside `prefix.distributions`.
    pub prob_array_1: ProbTagArray,
    /// Raw prob-array-2 records. Segmented into `suffix.distributions`.
    pub prob_array_2: ProbTagArray,
    /// Per-tag prelude — 58 `f64`s at the very start of the slab,
    /// one per tag in `Header::tags`. Range observed on `english.par`
    /// is roughly 7..21000, matching training-frequency counts. Used
    /// by callers as the true marginal `P(tag)` — divide by the
    /// total — for Bayes correction in inference.
    pub tag_prelude: Vec<f64>,
}

/// Parse the tries slab + its two prob-arrays + the one that lives
/// between the suffix trie and the decision tree.
///
/// `slab_start` is the cursor's current offset. `dtree_start` is the
/// absolute file offset where the decision tree begins — the second
/// prob-array extends from just-after-the-suffix-trie to just-before
/// the dtree.
pub fn read(cur: &mut Cursor<'_>, header: &Header, dtree_start: usize) -> Result<Tries> {
    let slab_start = cur.offset();
    if dtree_start <= slab_start {
        bail!(
            "dtree start {dtree_start} must be past slab start {slab_start}"
        );
    }
    let bytes = cur.bytes_after_cursor();
    let remaining = dtree_start - slab_start;
    if remaining > bytes.len() {
        bail!(
            "slab length {remaining} exceeds remaining cursor bytes {}",
            bytes.len()
        );
    }
    let slab = &bytes[..remaining];

    // Locate prefix trie entry run.
    let prefix_start = find_entry_run(slab, 0)
        .context("could not locate prefix trie")?;

    // Read the per-tag prelude — `num_tags` f64 values at the very
    // start of the slab, one per tag id. The remaining bytes between
    // the prelude and the prefix trie are alignment padding (2 bytes
    // on `english.par`).
    let n_tags = header.tags.len();
    let prelude_bytes = n_tags * 8;
    let mut tag_prelude = Vec::with_capacity(n_tags);
    if prelude_bytes <= prefix_start {
        for k in 0..n_tags {
            let off = k * 8;
            let v = f64::from_le_bytes([
                slab[off],
                slab[off + 1],
                slab[off + 2],
                slab[off + 3],
                slab[off + 4],
                slab[off + 5],
                slab[off + 6],
                slab[off + 7],
            ]);
            tag_prelude.push(v);
        }
    }
    let prefix_entries = walk_entries(slab, prefix_start);
    let prefix_end = prefix_start + prefix_entries.len() * 12;

    // 1-byte prelude (`0x15`) separates prefix trie from prob-array-1.
    // Skip it before decoding records.
    let pa1_start = prefix_end + 1;

    // Suffix trie: locate it via the same 3-run-of-footers scan,
    // starting somewhere past prob-array-1. We don't know the exact
    // end of prob-array-1 up-front, so just scan from the end of
    // the prefix trie.
    let suffix_start = find_entry_run(slab, prefix_end + 1)
        .context("could not locate suffix trie after prefix trie")?;
    let suffix_entries = walk_entries(slab, suffix_start);
    let suffix_end = suffix_start + suffix_entries.len() * 12;

    // 1-byte prelude again.
    let pa2_start = suffix_end + 1;
    let pa2_end = remaining;

    // prob-array-1: from pa1_start up to suffix_start.
    let prob_array_1 = parse_prob_tag_array(&slab[pa1_start..suffix_start]);
    let prob_array_2 = parse_prob_tag_array(&slab[pa2_start..pa2_end]);

    // Segment by leaf `count`: each trie leaf in pre-order DFS claims
    // exactly `count` records from the prob-array. This matches how
    // tree-tagger's runtime reads the file (verified via lldb on
    // suffix_lookup, see #14) and unlike the previous prob-curve-
    // based heuristic produces partial distributions that don't
    // always sum to 1.0 — which is exactly what Schmid's smoothing
    // expects from per-node distributions.
    let prefix_dists = segment_by_leaf_counts(&prob_array_1, &prefix_entries);
    let suffix_dists = segment_by_leaf_counts(&prob_array_2, &suffix_entries);

    let prefix = build_trie(prefix_entries, prefix_dists)?;
    let suffix = build_trie(suffix_entries, suffix_dists)?;

    cur.advance(remaining)
        .context("advancing cursor past tries slab")?;

    Ok(Tries {
        prefix,
        suffix,
        prob_array_1,
        prob_array_2,
        tag_prelude,
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
        if is_entry_at(slab, off)
            && is_entry_at(slab, off + 12)
            && is_entry_at(slab, off + 24)
        {
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
        out.push(TrieEntry {
            flag: u16::from_le_bytes([slab[off], slab[off + 1]]),
            char: u16::from_le_bytes([slab[off + 2], slab[off + 3]]),
            count: u16::from_le_bytes([slab[off + 4], slab[off + 5]]),
            offset: u16::from_le_bytes([slab[off + 6], slab[off + 7]]),
        });
        off += 12;
    }
    out
}

fn parse_prob_tag_array(bytes: &[u8]) -> ProbTagArray {
    let mut records = Vec::with_capacity(bytes.len() / 8);
    let mut off = 0;
    while off + 8 <= bytes.len() {
        records.push(ProbTagRecord {
            flag: bytes[off],
            canary: u16::from_le_bytes([bytes[off + 1], bytes[off + 2]]),
            prob: f32::from_le_bytes([
                bytes[off + 3],
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
            ]),
            tag_id: bytes[off + 7],
        });
        off += 8;
    }
    ProbTagArray { records }
}

/// Legacy prob-curve-based segmentation, kept around as a reference
/// for the previous algorithm. The current segmentation goes via
/// per-leaf `count` (see `segment_by_leaf_counts`).
#[allow(dead_code)]
fn segment_distributions(array: &ProbTagArray) -> Vec<Distribution> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut cur_sum = 0.0f32;
    let mut cur_prev = f32::MAX; // force a fresh start on first record

    for (idx, rec) in array.records.iter().enumerate() {
        // Only segment on clean records. Garbage-prob values at the
        // tail of a run (if any) get swept into the current segment
        // silently; real .par files seem not to have trailing junk.
        if idx > start
            && (rec.prob > cur_prev + 1e-4 || cur_sum + rec.prob > 1.001)
        {
            out.push(segment_to_dist(&array.records[start..idx]));
            start = idx;
            cur_sum = 0.0;
        }
        cur_sum += rec.prob;
        cur_prev = rec.prob;
    }
    if start < array.records.len() {
        out.push(segment_to_dist(&array.records[start..]));
    }
    out
}

/// Segment a prob-tag array using each trie leaf's `count` field as
/// the number of records to consume. Walks `entries` in pre-order DFS
/// (same order the file stores them) and pairs the i-th leaf with the
/// next `count` records from `array`.
///
/// Distributions produced this way are *partial* — they don't have
/// to sum to 1.0. Schmid's runtime computes the full P(t|w) by
/// interpolating partial distributions along the trie path.
fn segment_by_leaf_counts(array: &ProbTagArray, entries: &[TrieEntry]) -> Vec<Distribution> {
    if entries.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cursor = 0usize;
    let mut stack = vec![0usize];
    while let Some(idx) = stack.pop() {
        if idx >= entries.len() {
            continue;
        }
        let e = &entries[idx];
        if e.is_leaf() {
            let take = (e.count as usize).min(array.records.len() - cursor);
            let slice = &array.records[cursor..cursor + take];
            out.push(segment_to_dist(slice));
            cursor += take;
            continue;
        }
        let start = e.offset as usize;
        let end = start + e.count as usize;
        if end > entries.len() {
            continue;
        }
        // Push in reverse so we pop in natural pre-order.
        for c in (start..end).rev() {
            stack.push(c);
        }
    }
    out
}

fn segment_to_dist(records: &[ProbTagRecord]) -> Distribution {
    Distribution {
        probs: records
            .iter()
            .map(|r| TagProb {
                tag_id: r.tag_id,
                prob: r.prob,
            })
            .collect(),
    }
}

fn build_trie(entries: Vec<TrieEntry>, distributions: Vec<Distribution>) -> Result<Trie> {
    if entries.is_empty() {
        bail!("cannot build a trie from zero entries");
    }
    let root = &entries[0];
    if root.flag != 0 {
        bail!(
            "trie root is marked as a leaf (flag=1) — can't be, root must have children"
        );
    }
    let root_end = root.offset as usize + root.count as usize;
    if root_end > entries.len() {
        bail!(
            "trie root claims {} children at index {}, but only {} entries",
            root.count,
            root.offset,
            entries.len()
        );
    }

    // Walk leaves in pre-order DFS; the i-th leaf gets distribution i.
    let mut leaf_dist = vec![None; entries.len()];
    let mut stack = vec![0usize];
    let mut leaf_counter = 0usize;
    let n_dists = distributions.len();
    while let Some(idx) = stack.pop() {
        let e = &entries[idx];
        if e.flag == 1 {
            if leaf_counter < n_dists {
                leaf_dist[idx] = Some(leaf_counter);
            }
            leaf_counter += 1;
            continue;
        }
        // Push children in reverse so we pop them in natural order.
        let start = e.offset as usize;
        let end = start + e.count as usize;
        if end > entries.len() {
            continue;
        }
        for c in (start..end).rev() {
            stack.push(c);
        }
    }

    Ok(Trie {
        entries,
        root_index: 0,
        leaf_dist,
        distributions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::header;
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
        const DTREE_START: usize = 0xd231a3;

        // Need the real header so _header: &Header is available. Just
        // parse it in-place from file-start to keep the test self-
        // contained.
        let mut hcur = Cursor::new(&bytes);
        let hdr = header::read(&mut hcur).unwrap();

        let mut cur = Cursor::new(&bytes);
        cur.advance(SLAB_START).unwrap();
        let tries = read(&mut cur, &hdr, DTREE_START).unwrap();

        assert_eq!(tries.prefix.entries.len(), 3000, "prefix trie entries");
        assert_eq!(tries.suffix.entries.len(), 3086, "suffix trie entries");

        // Prob-arrays clean count
        assert!(
            tries.prob_array_1.records.iter().filter(|r| r.flag == 0 && r.canary == 0xBABA).count() > 5700,
            "prob-array-1 should have >5700 clean records"
        );
        assert!(
            tries.prob_array_2.records.iter().filter(|r| r.flag == 0 && r.canary == 0xBABA).count() > 6000,
            "prob-array-2 should have >6000 clean records"
        );

        // Distribution counts match leaf counts (1:1 file-order mapping).
        let prefix_leaves = tries.prefix.entries.iter().filter(|e| e.is_leaf()).count();
        let suffix_leaves = tries.suffix.entries.iter().filter(|e| e.is_leaf()).count();
        assert_eq!(prefix_leaves, 1728, "prefix leaves");
        assert_eq!(suffix_leaves, 2187, "suffix leaves");
        // Allow +/- 1 between distribution count and leaf count — the
        // segmentation heuristic sometimes picks up a trailing 1-record
        // "segment" past the last real distribution.
        assert!(
            (tries.prefix.distributions.len() as isize - prefix_leaves as isize).abs() <= 1,
            "prefix distributions ({}) should match leaf count ({})",
            tries.prefix.distributions.len(),
            prefix_leaves
        );
        assert!(
            (tries.suffix.distributions.len() as isize - suffix_leaves as isize).abs() <= 1,
            "suffix distributions ({}) should match leaf count ({})",
            tries.suffix.distributions.len(),
            suffix_leaves
        );

        // Distributions are *partial* — they don't have to sum to
        // 1.0. The runtime smooths multiple partial distributions
        // along the trie path. We just sanity-check that probs are
        // all in [0, 1].
        for (i, d) in tries.suffix.distributions.iter().enumerate().take(100) {
            for tp in &d.probs {
                assert!(
                    (0.0..=1.0 + 1e-5).contains(&tp.prob),
                    "suffix dist {i} has out-of-range prob: {}",
                    tp.prob
                );
            }
        }

        // Structural sanity: walk suffix trie from 's' → 'e' → 's'
        // (suffix "ses") and verify that each child lookup works.
        let root = tries.suffix.root();
        let kids = tries.suffix.children(root);
        let s_node = kids.iter().find(|e| e.as_char() == Some('s')).unwrap();
        let s_kids = tries.suffix.children(s_node);
        let es_node = s_kids.iter().find(|e| e.as_char() == Some('e')).unwrap();
        let es_kids = tries.suffix.children(es_node);
        let ses_node = es_kids.iter().find(|e| e.as_char() == Some('s')).unwrap();
        let _ = ses_node;

        // Suffix lookup for a word: walk the suffix trie in reverse
        // order of the word's characters. 'classes' → chars reversed
        // = 's', 'e', 's', 's', 'a', 'l', 'c'. Deepest matching leaf
        // gives an NN-heavy distribution.
        let dist = tries
            .suffix
            .lookup("classes".chars().rev())
            .expect("lookup for 'classes' should hit a leaf in the suffix trie");
        let peak = dist.peak().unwrap();
        // NN = 19 on english.par. 'sses' leaf had dist NN=0.98 in the
        // Python exploration — assert NN is the top tag and its prob
        // is high.
        assert_eq!(peak.tag_id, 19, "peak tag for 'classes' should be NN (tag 19)");
        assert!(peak.prob > 0.9, "peak prob for 'classes' should be > 0.9, was {}", peak.prob);
    }
}
