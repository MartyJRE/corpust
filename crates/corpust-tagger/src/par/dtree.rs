//! `.par` decision-tree section — partial reader.
//!
//! What's decoded (from differential training on toy models +
//! hex-dump inspection on `english.par`):
//!
//! - **Leaf record** (64 bytes on N=3, 720 bytes on N=58):
//!   ```text
//!   u32  node_id             // identifies which tree path leads here
//!   u32  1                   // type discriminator: 1 = leaf
//!   u32  num_tags
//!   u32  weight              // training sample count reaching this leaf
//!   [u32 tag_id, f64 prob] × num_tags   -- sums to 1.0
//!   ```
//!
//! - **Default fallback** (always last in the decision-tree section):
//!   ```text
//!   u32  1                   // type discriminator: 1 = default
//!   u32  num_tags
//!   u32  weight
//!   [u32 tag_id, f64 prob] × num_tags   -- P(tag) unconditional
//!   ```
//!   12-byte header (no `node_id`).
//!
//! What's **partially** decoded:
//!
//! - **Internal nodes are 12 bytes = 3 u32 fields.** Confirmed by
//!   differential training on minimum 2-tag models:
//!     ```text
//!     u32  offset_i         // 1 for cl=1 bigram, 2 for cl=2 trigram
//!                           // (matches Schmid's `tag_{-i} = t` test)
//!     u32  tag_id           // the tag t being tested (observed 0..1 on toys)
//!     u32  branch_info      // pointer/count for traversal — exact
//!                           //   semantics tbd: differs between cl=1
//!                           //   (value 0) and cl=2 (value 1) on the
//!                           //   same 3-node tree topology, so it's
//!                           //   encoding *something* about traversal
//!                           //   but not just a simple child offset.
//!     ```
//!   The count of internal records and their full traversal algorithm
//!   still need more work — `english.par`'s 44,692-byte internal blob
//!   doesn't divide cleanly by 12, so either there are other records
//!   interleaved (likely) or the internal layout changes with model
//!   size (less likely but possible).
//!
//! - The ~40 bytes of trie-like data just before the tree region on
//!   toy models — identical across balanced and skewed corpora at the
//!   same tagset shape, so structural rather than content. English
//!   models don't have this — their prefix/suffix tries live in the
//!   preceding 170 KB slab.
//!
//! Consequence for the in-process tagger: we can look up
//! `P(tag | word)` from the lexicon and `P(tag)` from the default,
//! but we can't route a context tag pair to its specific leaf. A
//! Viterbi built on this partial decode would use the default
//! distribution uniformly as the context-probability estimate — a
//! known-degraded model that sets a floor for accuracy before full
//! tree traversal lands.

use super::Cursor;
use super::header::Header;
use anyhow::{Context, Result};

/// One tag with its probability.
#[derive(Debug, Clone)]
pub struct TagProb {
    pub tag_id: u32,
    pub prob: f64,
}

/// Decision-tree leaf — a full `P(tag | context)` distribution plus
/// the node id that identifies its context path.
///
/// We don't yet know how to *reach* a specific leaf from a context
/// tag pair, so in the partial reader these are kept as a flat list
/// primarily for accounting and future use.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub node_id: u32,
    pub weight: u32,
    pub distribution: Vec<TagProb>,
}

/// Default distribution — always present, used as the fallback when
/// the decision tree doesn't yield a specific leaf.
#[derive(Debug, Clone)]
pub struct Default {
    pub weight: u32,
    pub distribution: Vec<TagProb>,
}

/// Partial reader output. `raw_internals` is the opaque blob that
/// encodes the tree structure we haven't fully decoded yet; kept
/// around so a future reader pass can crack it without re-loading the
/// file. Partial decode: 12-byte (offset_i, tag_id, branch_info)
/// records are known to live inside `raw_internals`, but their count
/// and traversal semantics aren't yet certain.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    pub leaves: Vec<Leaf>,
    pub default: Default,
    pub raw_internals: Vec<u8>,
}

/// A single internal (non-leaf) decision-tree node as we currently
/// understand it. Parsing is best-effort — the raw `branch_info`
/// field's semantics are still being pinned down so traversal isn't
/// yet possible from this record alone.
#[derive(Debug, Clone, Copy)]
pub struct Internal {
    /// Position to test. Schmid's paper calls this `i` in
    /// `tag_{-i} = t`. Observed values: `1` for bigram-context
    /// models, `2` for trigram-context.
    pub offset_i: u32,
    /// Tag ID being tested.
    pub tag_id: u32,
    /// Opaque pointer / count. Differs between tree topologies but
    /// exact semantics (yes-child offset? subtree size? a bitmap?)
    /// need more differential data to pin down.
    pub branch_info: u32,
}

/// Parse an 8- or 12-byte internal-node record from the given slice.
///
/// Returns `Some(node)` if the bytes look like a plausible internal
/// node (offset_i ∈ {1, 2, 3}, tag_id in range), else `None`. This
/// is a best-effort heuristic — we can't yet guarantee correctness
/// of every field, only that the header's shape is right.
pub fn try_parse_internal(bytes: &[u8], num_tags: u32) -> Option<Internal> {
    if bytes.len() < 12 {
        return None;
    }
    let offset_i = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
    let tag_id = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let branch_info = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    // Sanity: `offset_i` in a standard TreeTagger build is 1, 2, or
    // 3 (bi-/tri-/quatro-gram context lengths).
    if offset_i == 0 || offset_i > 3 {
        return None;
    }
    if tag_id >= num_tags {
        return None;
    }
    Some(Internal {
        offset_i,
        tag_id,
        branch_info,
    })
}

/// Parse the decision tree from `cur` to the end of the file.
///
/// Strategy:
/// 1. Locate the **default** by scanning backward from EOF for the
///    `[1, num_tags, weight]` header that precedes `num_tags` valid
///    `(tag_id, prob)` records summing to `~1.0`.
/// 2. Locate **leaves** by scanning forward from `cur` for
///    `[node_id, 1, num_tags, weight]` headers followed by valid
///    distributions.
/// 3. Everything between the cursor start, the leaf headers, and the
///    default is stashed into `raw_internals` verbatim.
pub fn read(cur: &mut Cursor<'_>, header: &Header) -> Result<DecisionTree> {
    let data = cur.bytes_after_cursor();
    let num_tags = header.tags.len() as u32;

    let default_off = find_default_start(data, num_tags)
        .context("could not locate decision-tree default distribution near EOF")?;

    let leaves = find_leaves(&data[..default_off], num_tags);

    // Build raw_internals from the gaps: before first leaf, between
    // consecutive leaves, and between last leaf and default.
    let mut raw_internals = Vec::new();
    let mut pos = 0usize;
    let leaf_size = 16 + (num_tags as usize) * 12;
    let mut sorted_leaves: Vec<_> = leaves.iter().map(|(o, _)| *o).collect();
    sorted_leaves.sort();
    for &off in &sorted_leaves {
        if off > pos {
            raw_internals.extend_from_slice(&data[pos..off]);
        }
        pos = off + leaf_size;
    }
    if default_off > pos {
        raw_internals.extend_from_slice(&data[pos..default_off]);
    }

    let default = parse_default(data, default_off, num_tags)?;

    let leaves: Vec<Leaf> = leaves
        .into_iter()
        .map(|(_, leaf)| leaf)
        .collect();

    // Advance cursor to EOF — the decision tree is the last section.
    cur.advance(data.len())
        .context("advancing cursor to EOF after decision tree")?;

    Ok(DecisionTree {
        leaves,
        default,
        raw_internals,
    })
}

fn find_default_start(data: &[u8], num_tags: u32) -> Option<usize> {
    let default_size = 12 + (num_tags as usize) * 12;
    if data.len() < default_size {
        return None;
    }
    // Walk candidate starts at 4-byte strides from the latest
    // possible position. Pick the one whose records parse and sum
    // close to 1.0.
    let latest = data.len() - default_size;
    for off in (0..=latest).rev().step_by(4) {
        if valid_default_at(data, off, num_tags) {
            return Some(off);
        }
    }
    None
}

fn valid_default_at(data: &[u8], off: usize, num_tags: u32) -> bool {
    if off + 12 + (num_tags as usize) * 12 > data.len() {
        return false;
    }
    let flag = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let n = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
    let _weight = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
    if flag != 1 || n != num_tags {
        return false;
    }
    let mut sum = 0.0f64;
    for k in 0..num_tags as usize {
        let rec = off + 12 + k * 12;
        let tag = u32::from_le_bytes(data[rec..rec + 4].try_into().unwrap());
        if tag != k as u32 {
            return false;
        }
        let prob = f64::from_le_bytes(data[rec + 4..rec + 12].try_into().unwrap());
        if !prob.is_finite() || prob < -1e-9 {
            return false;
        }
        sum += prob;
    }
    (sum - 1.0).abs() < 1e-5
}

fn parse_default(data: &[u8], off: usize, num_tags: u32) -> Result<Default> {
    let weight = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
    let distribution = (0..num_tags as usize)
        .map(|k| {
            let rec = off + 12 + k * 12;
            let tag = u32::from_le_bytes(data[rec..rec + 4].try_into().unwrap());
            let prob = f64::from_le_bytes(data[rec + 4..rec + 12].try_into().unwrap());
            TagProb { tag_id: tag, prob }
        })
        .collect();
    Ok(Default { weight, distribution })
}

fn find_leaves(data: &[u8], num_tags: u32) -> Vec<(usize, Leaf)> {
    let leaf_size = 16 + (num_tags as usize) * 12;
    let mut out = Vec::new();
    let mut i = 0;
    while i + leaf_size <= data.len() {
        if valid_leaf_at(data, i, num_tags) {
            let leaf = parse_leaf(data, i, num_tags);
            out.push((i, leaf));
            // Jump past — leaves don't overlap.
            i += leaf_size;
        } else {
            i += 4;
        }
    }
    out
}

fn valid_leaf_at(data: &[u8], off: usize, num_tags: u32) -> bool {
    if off + 16 + (num_tags as usize) * 12 > data.len() {
        return false;
    }
    let _node_id = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let flag = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
    let n = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
    let _weight = u32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap());
    if flag != 1 || n != num_tags {
        return false;
    }
    let mut sum = 0.0f64;
    for k in 0..num_tags as usize {
        let rec = off + 16 + k * 12;
        let tag = u32::from_le_bytes(data[rec..rec + 4].try_into().unwrap());
        if tag != k as u32 {
            return false;
        }
        let prob = f64::from_le_bytes(data[rec + 4..rec + 12].try_into().unwrap());
        if !prob.is_finite() || prob < -1e-9 {
            return false;
        }
        sum += prob;
    }
    (sum - 1.0).abs() < 1e-5
}

fn parse_leaf(data: &[u8], off: usize, num_tags: u32) -> Leaf {
    let node_id = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let weight = u32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap());
    let distribution = (0..num_tags as usize)
        .map(|k| {
            let rec = off + 16 + k * 12;
            let tag = u32::from_le_bytes(data[rec..rec + 4].try_into().unwrap());
            let prob = f64::from_le_bytes(data[rec + 4..rec + 12].try_into().unwrap());
            TagProb { tag_id: tag, prob }
        })
        .collect();
    Leaf {
        node_id,
        weight,
        distribution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn english_par_path() -> Option<PathBuf> {
        let candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("resources/treetagger/lib/english.par");
        candidate.exists().then_some(candidate)
    }

    #[test]
    fn parses_known_internal_shape() {
        // Bytes captured from m_cl1 (2-tag bigram) toy model — the
        // 12 bytes at file offset 0xdc which precede the only leaf.
        // Confirms our understanding that offset_i=1 on a bigram
        // model.
        let bytes = [
            0x01, 0x00, 0x00, 0x00, // offset_i = 1 (bigram)
            0x00, 0x00, 0x00, 0x00, // tag_id = 0
            0x00, 0x00, 0x00, 0x00, // branch_info = 0
        ];
        let node = try_parse_internal(&bytes, 2).unwrap();
        assert_eq!(node.offset_i, 1);
        assert_eq!(node.tag_id, 0);
        assert_eq!(node.branch_info, 0);

        // Bytes from m_cl2 (trigram): offset_i=2, branch_info=1
        let bytes = [
            0x02, 0x00, 0x00, 0x00, // offset_i = 2 (trigram)
            0x00, 0x00, 0x00, 0x00, // tag_id = 0
            0x01, 0x00, 0x00, 0x00, // branch_info = 1
        ];
        let node = try_parse_internal(&bytes, 2).unwrap();
        assert_eq!(node.offset_i, 2);
        assert_eq!(node.branch_info, 1);
    }

    #[test]
    fn rejects_garbage_as_internal() {
        // Zeros everywhere: offset_i=0 isn't valid (bigram/trigram/quatrogram only).
        assert!(try_parse_internal(&[0; 12], 58).is_none());
        // Huge offset_i.
        let bytes = [0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(try_parse_internal(&bytes, 58).is_none());
        // tag_id out of range.
        let bytes = [0x01, 0, 0, 0, 0x50, 0, 0, 0, 0, 0, 0, 0];
        assert!(try_parse_internal(&bytes, 58).is_none());
    }

    #[test]
    fn reads_bundled_english_decision_tree() {
        let Some(par) = english_par_path() else {
            return;
        };
        // Just verify we can parse by loading the whole file via the
        // top-level loader once dtree is wired in there. Here we
        // bypass by advancing a cursor manually from 0xd231bb (known
        // first-leaf offset) via a direct byte load.
        let bytes = std::fs::read(&par).unwrap();
        let tree_start = 0xd231bb;
        let mut cur = Cursor::new(&bytes);
        cur.advance(tree_start).unwrap();
        // Need a Header for num_tags. Build a stub with 58 tags.
        let header = super::super::header::Header {
            field_a: 0,
            field_b: 0,
            sent_tag_index: 31,
            tags: (0..58).map(|i| format!("T{i}")).collect(),
            end_offset: 0,
        };
        let tree = read(&mut cur, &header).unwrap();
        assert!(!tree.leaves.is_empty(), "no leaves found in english.par");
        assert_eq!(tree.default.distribution.len(), 58);
        let sum: f64 = tree.default.distribution.iter().map(|t| t.prob).sum();
        assert!((sum - 1.0).abs() < 1e-5, "default didn't sum to 1: {sum}");
        // Every leaf distribution sums to 1.0
        for (i, leaf) in tree.leaves.iter().enumerate() {
            let s: f64 = leaf.distribution.iter().map(|t| t.prob).sum();
            assert!((s - 1.0).abs() < 1e-5, "leaf {i} sum={s}");
        }
        eprintln!(
            "english.par: {} leaves, default weight={}, raw_internals={} bytes",
            tree.leaves.len(),
            tree.default.weight,
            tree.raw_internals.len()
        );
    }
}
