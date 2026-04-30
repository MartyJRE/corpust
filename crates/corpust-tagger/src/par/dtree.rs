//! `.par` decision-tree section — forward walker with typed records.
//!
//! Differential-training archaeology (see the pure-Rust TreeTagger
//! plan, *Section 4*) has pinned down **four** record shapes that
//! appear in this section. This reader walks the section in file
//! order, labels each record by kind, and fails loudly on anything
//! that doesn't match one of the four known shapes — that failure
//! mode is a feature: it exposes any fifth kind the next time we
//! meet one.
//!
//! ```text
//! kind                    | total bytes (N=num_tags)   | header layout
//! ------------------------|----------------------------|-----------------------------
//! Internal                | 12                         | [reserved=0, back_pos_i, test_tag_id]
//! Leaf                    | 16 + N*12                  | [node_id, 1, N, weight]
//! PrunedInternal          | 12 + N*12                  | [1, N, weight]
//! Default (always last)   | 12 + N*12                  | [1, N, weight]
//! ```
//!
//! **Internal predicate semantics** (reverse-engineered 2026-04-30):
//! `back_pos_i` is the zero-indexed back-position to test (0 means
//! tag_{-1}, 1 means tag_{-2}, etc.) and `test_tag_id` is the tag
//! being checked. The predicate evaluates as
//! `tag_at[-(back_pos_i + 1)] == test_tag_id`. On `english.par`
//! 724/781 internals test tag_{-1} and 57/781 test tag_{-2},
//! consistent with the model's cl=2 training. The `reserved` u32 is
//! always 0 in every model examined.
//!
//! Earlier plan drafts (see `pure-rust-treetagger.md` Section 4 prior
//! to 2026-04-30) had `back_pos_i` and `test_tag_id` swapped; the
//! mistake came from a toy1 (cl=1) vs toy2 (cl=2) differential in
//! which the change in `back_pos_i` (always 0 in toy1, 0 or 1 in
//! toy2) was misread as a change in test tag.
//!
//! Distribution payload (Leaf / PrunedInternal / Default) is always
//! `N × (u32 tag_id, f64 prob)` records with `tag_id == index` and
//! probabilities summing to `~1.0`.
//!
//! Note: **PrunedInternal and Default share a binary layout.** An
//! earlier plan draft said PrunedInternal had a `[1, N, weight, 0]`
//! header (16 bytes); the trailing `0` was actually `tag_id=0` of the
//! first distribution record. They're distinguished purely by
//! position — the Default is always the record flush with EOF.
//!
//! Disambiguation order at each cursor position `p`:
//!
//! 1. If `p == len - (12 + N*12)` AND header starts with `[1, N, _]`
//!    AND the N following distribution records validate → **Default**.
//! 2. If `u32[p+4] == 1` AND `u32[p+8] == N` AND the distribution at
//!    `p+16` validates → **Leaf**. The distribution sum-to-1.0 check
//!    is the strong test; the header shape alone isn't enough to rule
//!    out coincidence.
//! 3. If `u32[p] == 1` AND `u32[p+4] == N` AND the distribution at
//!    `p+12` validates → **PrunedInternal**.
//! 4. Else if `u32[p] ∈ {0,1,2,3}` AND `u32[p+4] < N` → **Internal**.
//!    `offset_i = 0` is observed on `english.par` — interspersed
//!    between Leaf/PrunedInternal records — so it's treated as a
//!    valid internal even though Schmid '94 §3 only references
//!    `i ∈ {1,2}`. Interpretation is unresolved (may be a
//!    root-pointer sentinel or a child-skip marker).
//! 5. Else bail with the offset and nearby bytes, so follow-up
//!    archaeology has somewhere concrete to start.
//!
//! Consumers get [`DecisionTree::records`] in order so they can
//! reconstruct tree topology later (preorder DFS with yes-child
//! implied by position, no-child pointer carried in `branch_info`)
//! once we've pinned down `branch_info` semantics in sub-task 2.
//!
//! Open question from this pass: on `english.par` the walker finds
//! **exactly one** `node_id`-headed ([`Leaf`]) record — at section
//! offset 0 — and 782 `PrunedInternal` records. That's suggestive
//! that what we're calling `Leaf` is really "the record with an
//! explicit node_id" (plausibly the tree root) rather than every
//! terminal node. All the actual distribution-carrying nodes use
//! the `[1, N, weight]` format we're labelling `PrunedInternal`.
//! The kind name is kept for continuity with the plan's taxonomy;
//! sub-task 3 (traversal) will resolve what each kind really
//! represents semantically.

use super::Cursor;
use super::header::Header;
use anyhow::{Context, Result, bail};

/// One tag with its probability.
#[derive(Debug, Clone, Copy)]
pub struct TagProb {
    pub tag_id: u32,
    pub prob: f64,
}

/// Distribution payload shared by Leaf / PrunedInternal / Default.
#[derive(Debug, Clone)]
pub struct Distribution {
    /// Training sample count reaching this node.
    pub weight: u32,
    /// `P(tag | this node's context)`. `probs[k].tag_id == k`;
    /// entries sum to `~1.0`.
    pub probs: Vec<TagProb>,
}

/// 12-byte record encoding a Schmid `tag_{-i}=t` test. Predicate
/// evaluates as `tag_at[-(back_pos_i + 1)] == test_tag_id`.
///
/// Cross-file evidence supporting this interpretation:
///
/// | model       | cl | back_pos_i values            | observed test_tag_id range |
/// |-------------|----|------------------------------|----------------------------|
/// | toy1        | 1  | {0} × 1 internal             | {0}                        |
/// | toy2        | 2  | {1} × 1 internal             | {0}                        |
/// | toy3        | 2  | {0}×9, {1}×9                 | 0..4                       |
/// | toy4        | 2  | {0}, {0}, {1}                | {0,1,2}                    |
/// | toy6        | 3  | {2} × 1 internal             | {0}                        |
/// | english.par | 2  | {0}×724, {1}×57              | 0..57                      |
///
/// `cl` is read from the dtree wrapper preamble (the second 12-byte
/// record at section offset 12, u32[8] of that record).
#[derive(Debug, Clone, Copy)]
pub struct Internal {
    /// `u32[0]` of the record. Always `0` in every model examined
    /// (toys 1–6, english.par). Possibly reserved/padding.
    pub reserved: u32,
    /// `u32[4]` — zero-indexed back-position. `0` means tag_{-1},
    /// `1` means tag_{-2}, etc. Bounded above by `cl - 1`.
    pub back_pos_i: u32,
    /// `u32[8]` — tag id being tested at position `-(back_pos_i + 1)`.
    /// Always in `0..num_tags`.
    pub test_tag_id: u32,
}

/// Decision-tree leaf — `P(tag | context)` at a terminal node.
#[derive(Debug, Clone)]
pub struct Leaf {
    /// Identifies which tree path reaches this leaf. Consumers don't
    /// have a way to resolve this to a path yet (see sub-task 3) so
    /// it's preserved verbatim.
    pub node_id: u32,
    pub distribution: Distribution,
}

/// Collapsed-subtree distribution — a branch that used to have
/// children but got pruned (all children were leaves, gain below
/// threshold; see Schmid '94 §3.2). Kept in the file to avoid
/// recomputing its probability vector at inference time.
#[derive(Debug, Clone)]
pub struct PrunedInternal {
    pub distribution: Distribution,
}

/// Unconditional `P(tag)` fallback. Always the last record in the
/// section.
#[derive(Debug, Clone)]
pub struct Default {
    pub distribution: Distribution,
}

/// One record from the decision-tree section, tagged by kind so
/// downstream code can tell leaves apart from pruned-internal
/// distributions (both 16+N*12 byte; previous reader fused them).
#[derive(Debug, Clone)]
pub enum DTreeRecord {
    Internal(Internal),
    Leaf(Leaf),
    PrunedInternal(PrunedInternal),
    Default(Default),
}

/// Just the variant tag of a [`DTreeRecord`]. Used for counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTreeKind {
    Internal,
    Leaf,
    PrunedInternal,
    Default,
}

impl DTreeRecord {
    pub fn kind(&self) -> DTreeKind {
        match self {
            DTreeRecord::Internal(_) => DTreeKind::Internal,
            DTreeRecord::Leaf(_) => DTreeKind::Leaf,
            DTreeRecord::PrunedInternal(_) => DTreeKind::PrunedInternal,
            DTreeRecord::Default(_) => DTreeKind::Default,
        }
    }
}

/// Per-kind record counts. Useful for sanity checks against what
/// `train-tree-tagger` reports ("Number of nodes: K").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KindCounts {
    pub internals: usize,
    pub leaves: usize,
    pub pruned_internals: usize,
    pub defaults: usize,
}

/// Parsed decision tree — records in file order.
///
/// Construction is via [`read`]. The tree structure itself (root
/// pointer, child links, traversal) isn't wired up yet — that's the
/// next sub-task once `branch_info` semantics are known.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    /// Every record in the order it appeared in the file.
    pub records: Vec<DTreeRecord>,
}

impl DecisionTree {
    pub fn kind_counts(&self) -> KindCounts {
        let mut c = KindCounts::default();
        for r in &self.records {
            match r.kind() {
                DTreeKind::Internal => c.internals += 1,
                DTreeKind::Leaf => c.leaves += 1,
                DTreeKind::PrunedInternal => c.pruned_internals += 1,
                DTreeKind::Default => c.defaults += 1,
            }
        }
        c
    }

    /// The unconditional fallback distribution. Always the last
    /// record when a tree is parsed successfully, so this is
    /// infallible on any `DecisionTree` built by [`read`].
    pub fn default(&self) -> &Default {
        match self.records.last() {
            Some(DTreeRecord::Default(d)) => d,
            _ => unreachable!("read() enforces a trailing Default"),
        }
    }

    pub fn leaves(&self) -> impl Iterator<Item = &Leaf> + '_ {
        self.records.iter().filter_map(|r| match r {
            DTreeRecord::Leaf(l) => Some(l),
            _ => None,
        })
    }

    pub fn internals(&self) -> impl Iterator<Item = &Internal> + '_ {
        self.records.iter().filter_map(|r| match r {
            DTreeRecord::Internal(i) => Some(i),
            _ => None,
        })
    }

    pub fn pruned_internals(&self) -> impl Iterator<Item = &PrunedInternal> + '_ {
        self.records.iter().filter_map(|r| match r {
            DTreeRecord::PrunedInternal(p) => Some(p),
            _ => None,
        })
    }

    /// Reconstruct the binary-tree topology from the flat record list
    /// by parsing it as a preorder DFS (yes-child first). Returns a
    /// forest because `english.par` is known to contain two
    /// back-to-back trees after stripping the wrapper records and
    /// trailing [`Default`]. Toy models produce a single-element
    /// forest.
    ///
    /// **Wrappers.** Two distinct preamble formats have been observed
    /// at the start of the dtree section:
    ///
    /// - `english.par` style — a single 16+N*12 byte record that
    ///   parses as a [`Leaf`] (with `node_id=21`, distribution
    ///   matching nothing structurally meaningful yet). One record.
    /// - Toy-model style — two consecutive 12-byte records. The
    ///   first has bytes `[0, 0, 0x01010001]`; the second has
    ///   `[0, 0, cl]` where `cl` is the model's context-length
    ///   parameter. These get walker-classified as [`Internal`]
    ///   because the byte shape matches, but their `branch_info`
    ///   value (`0x01010001` = 16842753) is far above the tag-id
    ///   ceiling that real Internals respect, so a `branch_info`
    ///   threshold cleanly distinguishes them.
    ///
    /// `nodes[0]` is the root of the first tree. `roots` lists the
    /// index into `nodes` of each tree's root so callers can walk
    /// each tree separately.
    ///
    /// We don't yet know what the fields of an [`Internal`] actually
    /// mean (sub-task 2 of `pure-rust-treetagger.md`), so
    /// [`TreeNode::Internal`] just carries the raw Internal fields
    /// unchanged plus yes/no child pointers. Once the predicate
    /// semantics are pinned down, `TreeNode::Internal` can grow a
    /// typed `predicate: Feature { back: u32, tag: u32 }` field
    /// without touching callers of the topology API.
    pub fn reconstruct(&self) -> Result<TreeForest> {
        let wrapper_records = self.detect_wrapper_records();
        let body = &self.records[wrapper_records..];

        let mut nodes: Vec<TreeNode> = Vec::new();
        let mut roots: Vec<usize> = Vec::new();
        let mut cursor = 0usize;
        while cursor < body.len() {
            let root = try_build_subtree(body, &mut cursor, &mut nodes).with_context(|| {
                format!(
                    "preorder-DFS reconstruction ran off the end of the body \
                     at record {} (wrapper_records={wrapper_records}, body_len={})",
                    cursor,
                    body.len()
                )
            })?;
            roots.push(root);
        }

        Ok(TreeForest {
            nodes,
            roots,
            wrapper_records,
        })
    }

    /// Detect leading wrapper / preamble records that aren't part of
    /// any tree's topology.
    fn detect_wrapper_records(&self) -> usize {
        match self.records.first() {
            // english.par-style Leaf wrapper.
            Some(DTreeRecord::Leaf(_)) => 1,
            // Toy-style 24-byte preamble — first record's u32[8] is
            // a sentinel (0x01010001) that exceeds anything a real
            // Internal's `test_tag_id` could carry.
            Some(DTreeRecord::Internal(i)) if i.test_tag_id > 0xFFFF => 2,
            _ => 0,
        }
    }
}

/// Cached topology + chosen entry-root for `predict`. Build once per
/// model; reuse across many predictions.
#[derive(Debug, Clone)]
pub struct Traversal {
    pub forest: TreeForest,
    /// Index into `forest.nodes` of the tree we descend from.
    pub root: usize,
    /// Marginal `P(tag)` distribution — weighted average of every
    /// leaf in `forest.nodes`. Used by Bayes-corrected callers as
    /// the prior; far more representative than any single leaf
    /// because it folds in all training data, not just the one path
    /// reached by a specific context.
    pub marginal: Distribution,
}

impl Traversal {
    /// `tag_at[-(back_pos_i + 1)] == test_tag_id` predicate evaluated
    /// against the supplied context (oldest tag first; e.g.
    /// `[..., tag_{-2}, tag_{-1}]`). Returns the leaf distribution
    /// reached by the **inference root** (last tree of the forest).
    pub fn predict<'a>(&'a self, context: &[u32]) -> &'a Distribution {
        traverse_tree(&self.forest, self.root, context)
    }

    /// Combined per-tag probability across **every tree** in the
    /// forest, multiplied. For `english.par`'s 2-tree forest this
    /// is `tree[0](t | ctx) × tree[1](t | ctx)` — empirically a
    /// far better estimate of `P(t | ctx)` than either tree alone.
    /// The two trees disagree wildly on common contexts (tree[0] is
    /// a shallow tag_{-1}-only switch, tree[1] is the full bigram
    /// tree); their disagreement multiplies out so each tree's
    /// confident choice gets reinforced and uncertain choices
    /// average. For single-tree forests (toys) this collapses to
    /// the only tree's distribution.
    pub fn predict_combined(&self, context: &[u32]) -> Vec<f64> {
        let n_tags = self.marginal.probs.len();
        let mut acc = vec![1.0f64; n_tags];
        for &root in &self.forest.roots {
            let dist = traverse_tree(&self.forest, root, context);
            for tp in &dist.probs {
                acc[tp.tag_id as usize] *= tp.prob;
            }
        }
        // Re-normalize so it's still a probability distribution.
        let total: f64 = acc.iter().sum();
        if total > 0.0 {
            for v in acc.iter_mut() {
                *v /= total;
            }
        }
        acc
    }

    /// Schmid 1995 / Brants 1995-style linear interpolation between
    /// the bigram tree (`forest.roots.last()`) and the unigram tree
    /// (`forest.roots.first()`):
    ///
    /// ```text
    /// P(t | t_{-1}, t_{-2}) = (1 - I) × tree[1](t | ctx) + I × tree[0](t | tag_{-1})
    /// ```
    ///
    /// where `I` is the interpolation weight (a backoff strength —
    /// higher `I` trusts the unigram more). For single-tree forests
    /// the interpolation collapses to the only tree's distribution.
    ///
    /// The unigram traversal uses only `tag_{-1}` (the last entry of
    /// `context`), since tree[0] of `english.par` is a switch chain
    /// over `tag_{-1}` and ignores deeper context anyway.
    pub fn predict_interpolated(&self, context: &[u32], i_weight: f64) -> Vec<f64> {
        let n_tags = self.marginal.probs.len();
        let mut acc = vec![0.0f64; n_tags];
        if self.forest.roots.len() <= 1 {
            // Only one tree — return its distribution directly.
            let dist = traverse_tree(&self.forest, self.root, context);
            for tp in &dist.probs {
                acc[tp.tag_id as usize] = tp.prob;
            }
            return acc;
        }
        let bigram = traverse_tree(&self.forest, self.root, context);
        // Pass only tag_{-1} to the unigram tree.
        let unigram_ctx: &[u32] = if let Some(last) = context.last() {
            std::slice::from_ref(last)
        } else {
            &[]
        };
        let unigram_root = self.forest.roots[0];
        let unigram = traverse_tree(&self.forest, unigram_root, unigram_ctx);
        for tp in &bigram.probs {
            acc[tp.tag_id as usize] += (1.0 - i_weight) * tp.prob;
        }
        for tp in &unigram.probs {
            acc[tp.tag_id as usize] += i_weight * tp.prob;
        }
        acc
    }
}

impl DecisionTree {
    /// Reconstruct the forest and pick the inference root. The rule:
    /// **last tree in preorder-DFS order**. For a single-tree forest
    /// (toys, most models) that's the only tree. For `english.par`'s
    /// 2-tree forest, that's tree[1] — empirically the inference
    /// tree (see issue #9 for the experimental evidence and the
    /// open question of tree[0]'s purpose).
    pub fn traversal(&self) -> Result<Traversal> {
        let forest = self.reconstruct()?;
        let &root = forest
            .roots
            .last()
            .context("forest has no trees — nothing to traverse")?;
        // Marginal P(tag) approximated by averaging every leaf's
        // distribution weighted by the leaf's training count. This
        // approximates the unconditional tag prior much better than
        // any single leaf (`predict(&[])` and `default()` both reach
        // a specific leaf, not the true marginal). Used by Viterbi
        // for the Bayes-corrected joint score.
        let n_tags = forest
            .nodes
            .iter()
            .find_map(|n| match n {
                TreeNode::Leaf { distribution, .. } => Some(distribution.probs.len()),
                _ => None,
            })
            .unwrap_or(0);
        let mut acc = vec![0.0f64; n_tags];
        let mut total_weight = 0.0f64;
        for n in &forest.nodes {
            if let TreeNode::Leaf { distribution, .. } = n {
                let w = distribution.weight as f64;
                if w == 0.0 {
                    continue;
                }
                total_weight += w;
                for tp in &distribution.probs {
                    acc[tp.tag_id as usize] += w * tp.prob;
                }
            }
        }
        let marginal = Distribution {
            weight: total_weight as u32,
            probs: acc
                .into_iter()
                .enumerate()
                .map(|(k, v)| TagProb {
                    tag_id: k as u32,
                    prob: if total_weight > 0.0 {
                        v / total_weight
                    } else {
                        0.0
                    },
                })
                .collect(),
        };
        Ok(Traversal {
            forest,
            root,
            marginal,
        })
    }
}

/// Walk one tree of a `TreeForest` from `root_idx` with `context`
/// = previous tags (oldest first). Each `Internal` predicate
/// `[back_pos_i, test_tag_id]` evaluates as
/// `context[len - 1 - back_pos_i] == test_tag_id`. Out-of-bounds
/// reads (early in a sentence) take the no-branch.
///
/// Returns the leaf distribution reached by the traversal.
pub fn traverse_tree<'a>(
    forest: &'a TreeForest,
    root_idx: usize,
    context: &[u32],
) -> &'a Distribution {
    let mut idx = root_idx;
    loop {
        match &forest.nodes[idx] {
            TreeNode::Leaf { distribution, .. } => return distribution,
            TreeNode::Internal {
                predicate, yes, no, ..
            } => {
                let back = predicate.back_pos_i as usize;
                let observed = context.len().checked_sub(back + 1).and_then(|k| context.get(k));
                idx = if observed.copied() == Some(predicate.test_tag_id) {
                    *yes
                } else {
                    *no
                };
            }
        }
    }
}

/// Topological view of the decision tree(s), reconstructed from the
/// flat preorder-DFS record list by [`DecisionTree::reconstruct`].
#[derive(Debug, Clone)]
pub struct TreeForest {
    /// All nodes in the forest, indexed by node id. Children of an
    /// internal node refer to indices into this same vector.
    pub nodes: Vec<TreeNode>,
    /// Indices into `nodes` of each tree's root.
    pub roots: Vec<usize>,
    /// How many leading records of the source `DecisionTree` were
    /// stripped as wrapper / preamble. See
    /// [`DecisionTree::reconstruct`] for the wrapper formats observed
    /// so far.
    pub wrapper_records: usize,
}

/// A node in a reconstructed decision tree.
///
/// Leaf and Default records map to [`TreeNode::Leaf`] — they both
/// carry a probability distribution and terminate a traversal path.
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Non-terminal: evaluate the predicate (still opaque) and
    /// descend into `yes` or `no`.
    Internal {
        predicate: Internal,
        yes: usize,
        no: usize,
    },
    /// Terminal: the distribution here is the answer.
    Leaf {
        /// `None` for the pre-tree wrapper Leaf (when present — only
        /// seen on `english.par` so far) or for the trailing Default.
        /// Pruned-internal leaves don't carry a node_id either.
        node_id: Option<u32>,
        distribution: Distribution,
    },
}

fn try_build_subtree(
    body: &[DTreeRecord],
    cursor: &mut usize,
    nodes: &mut Vec<TreeNode>,
) -> Option<usize> {
    let rec_idx = *cursor;
    if rec_idx >= body.len() {
        return None;
    }
    *cursor += 1;
    let my_idx = nodes.len();
    match &body[rec_idx] {
        DTreeRecord::Internal(internal) => {
            // Reserve this node's slot first so recursive calls can
            // push their own nodes without indexing conflict.
            nodes.push(TreeNode::Leaf {
                node_id: None,
                distribution: Distribution {
                    weight: 0,
                    probs: Vec::new(),
                },
            }); // placeholder, overwritten below
            let yes = try_build_subtree(body, cursor, nodes)?;
            let no = try_build_subtree(body, cursor, nodes)?;
            nodes[my_idx] = TreeNode::Internal {
                predicate: *internal,
                yes,
                no,
            };
        }
        DTreeRecord::PrunedInternal(p) => {
            nodes.push(TreeNode::Leaf {
                node_id: None,
                distribution: p.distribution.clone(),
            });
        }
        DTreeRecord::Leaf(l) => {
            nodes.push(TreeNode::Leaf {
                node_id: Some(l.node_id),
                distribution: l.distribution.clone(),
            });
        }
        DTreeRecord::Default(d) => {
            nodes.push(TreeNode::Leaf {
                node_id: None,
                distribution: d.distribution.clone(),
            });
        }
    }
    Some(my_idx)
}

/// Scan forward from `search_from` looking for the first byte offset
/// at which the remaining bytes parse cleanly as a complete dtree
/// section (terminating in a `Default` flush with EOF).
///
/// Used by exploration tooling on toy `.par` files where the dtree
/// start offset isn't known a priori. Linear in file length × section
/// length: fine for toys (~hundreds of bytes), don't call on
/// `english.par` — use the known constant there.
pub fn find_dtree_start(
    bytes: &[u8],
    header: &Header,
    search_from: usize,
) -> Option<usize> {
    let n_us = header.tags.len();
    let default_size = 12 + n_us * 12;
    if bytes.len() < default_size || search_from > bytes.len() - default_size {
        return None;
    }
    let max_start = bytes.len() - default_size;
    for off in search_from..=max_start {
        let mut cur = Cursor::new(bytes);
        cur.advance(off).ok()?;
        if read(&mut cur, header).is_ok() {
            return Some(off);
        }
    }
    None
}

/// Parse the decision-tree section from `cur` to EOF.
///
/// See the module docstring for the record-kind disambiguation order.
pub fn read(cur: &mut Cursor<'_>, header: &Header) -> Result<DecisionTree> {
    let data = cur.bytes_after_cursor();
    let n = header.tags.len() as u32;
    let n_us = n as usize;
    let len = data.len();
    let dist_bytes = n_us * 12;
    let leaf_size = 16 + dist_bytes;
    let default_size = 12 + dist_bytes;

    if len < default_size {
        bail!(
            "decision-tree section is {} bytes — too short to hold a \
             {}-tag default ({} bytes)",
            len,
            n,
            default_size
        );
    }
    let default_start = len - default_size;

    let mut records = Vec::new();
    let mut p = 0usize;

    while p < len {
        // 1. Default — only legal at the exact trailing offset.
        if p == default_start
            && u32_at(data, p) == Some(1)
            && u32_at(data, p + 4) == Some(n)
            && distribution_valid(data, p + 12, n_us)
        {
            let weight = u32_at(data, p + 8).unwrap();
            let probs = read_distribution_probs(data, p + 12, n_us);
            records.push(DTreeRecord::Default(Default {
                distribution: Distribution { weight, probs },
            }));
            p = len;
            continue;
        }

        // 2. Leaf — [node_id, 1, N, weight] + distribution.
        if p + leaf_size <= len
            && u32_at(data, p + 4) == Some(1)
            && u32_at(data, p + 8) == Some(n)
            && distribution_valid(data, p + 16, n_us)
        {
            let node_id = u32_at(data, p).unwrap();
            let weight = u32_at(data, p + 12).unwrap();
            let probs = read_distribution_probs(data, p + 16, n_us);
            records.push(DTreeRecord::Leaf(Leaf {
                node_id,
                distribution: Distribution { weight, probs },
            }));
            p += leaf_size;
            continue;
        }

        // 3. PrunedInternal — [1, N, weight] + distribution. Same
        // binary layout as Default; distinguished by position (Default
        // only fires at EOF, checked in step 1 above).
        if p + default_size <= len
            && u32_at(data, p) == Some(1)
            && u32_at(data, p + 4) == Some(n)
            && distribution_valid(data, p + 12, n_us)
        {
            let weight = u32_at(data, p + 8).unwrap();
            let probs = read_distribution_probs(data, p + 12, n_us);
            records.push(DTreeRecord::PrunedInternal(PrunedInternal {
                distribution: Distribution { weight, probs },
            }));
            p += default_size;
            continue;
        }

        // 4. Internal — 12 bytes, [reserved == 0, back_pos_i in {0..3}, test_tag_id < N].
        // `reserved` is always 0 in observed models; the {0..3} mask
        // is kept defensively in case future models use higher cl.
        // Test-tag bound `< N` is the strong test that excludes the
        // toy preamble's sentinel u32[8] (= 0x01010001).
        if p + 12 <= len {
            let reserved = u32_at(data, p).unwrap();
            let back_pos_i = u32_at(data, p + 4).unwrap();
            if matches!(reserved, 0 | 1 | 2 | 3) && back_pos_i < n {
                let test_tag_id = u32_at(data, p + 8).unwrap();
                records.push(DTreeRecord::Internal(Internal {
                    reserved,
                    back_pos_i,
                    test_tag_id,
                }));
                p += 12;
                continue;
            }
        }

        // None of the four known kinds fit — stop here so the next
        // archaeology session has a concrete offset to look at.
        let preview = preview_bytes(data, p, 32);
        bail!(
            "unrecognized decision-tree record at section-offset {p} \
             (remaining={}); next 32 bytes = {preview}",
            len - p
        );
    }

    if !matches!(records.last(), Some(DTreeRecord::Default(_))) {
        bail!(
            "decision tree didn't terminate with a Default record \
             (last kind: {:?})",
            records.last().map(|r| r.kind())
        );
    }

    cur.advance(len)
        .context("advancing cursor to EOF after decision tree")?;

    Ok(DecisionTree { records })
}

fn u32_at(data: &[u8], off: usize) -> Option<u32> {
    let slice = data.get(off..off + 4)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn f64_at(data: &[u8], off: usize) -> Option<f64> {
    let slice = data.get(off..off + 8)?;
    let bits = u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]);
    Some(f64::from_bits(bits))
}

/// Does `num_tags` records of `(u32 tag_id, f64 prob)` at `off`
/// form a valid distribution? The test is:
///
/// - `tag_id == k` for k in `0..num_tags`
/// - every `prob` is finite and in `[-1e-9, 1]` (allowing tiny
///   negative noise from f64 rounding, but nothing far off)
/// - probs sum to `1.0 ± 1e-5`
///
/// This is strong enough that random-looking file bytes have
/// essentially zero probability of passing — the tag-id ascending
/// constraint alone rejects almost everything, and the sum-to-1
/// constraint catches the rest.
fn distribution_valid(data: &[u8], off: usize, num_tags: usize) -> bool {
    if off + num_tags * 12 > data.len() {
        return false;
    }
    let mut sum = 0.0f64;
    for k in 0..num_tags {
        let rec = off + k * 12;
        let Some(tag) = u32_at(data, rec) else {
            return false;
        };
        if tag != k as u32 {
            return false;
        }
        let Some(prob) = f64_at(data, rec + 4) else {
            return false;
        };
        if !prob.is_finite() || prob < -1e-9 || prob > 1.0 + 1e-6 {
            return false;
        }
        sum += prob;
    }
    (sum - 1.0).abs() < 1e-5
}

fn read_distribution_probs(data: &[u8], off: usize, num_tags: usize) -> Vec<TagProb> {
    (0..num_tags)
        .map(|k| {
            let rec = off + k * 12;
            TagProb {
                tag_id: u32_at(data, rec).unwrap(),
                prob: f64_at(data, rec + 4).unwrap(),
            }
        })
        .collect()
}

fn preview_bytes(data: &[u8], off: usize, len: usize) -> String {
    let end = (off + len).min(data.len());
    let bytes = &data[off..end];
    let mut s = String::with_capacity(bytes.len() * 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        use std::fmt::Write as _;
        write!(&mut s, "{b:02x}").unwrap();
    }
    s
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

    /// Build a synthetic in-memory distribution payload: N records of
    /// `(tag_id, prob)` that validate as a distribution.
    fn synth_distribution(num_tags: u32) -> Vec<u8> {
        let n = num_tags as usize;
        let prob = 1.0f64 / n as f64;
        let mut out = Vec::with_capacity(n * 12);
        for k in 0..num_tags {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&prob.to_le_bytes());
        }
        out
    }

    fn synth_leaf(node_id: u32, num_tags: u32, weight: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&node_id.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&num_tags.to_le_bytes());
        out.extend_from_slice(&weight.to_le_bytes());
        out.extend_from_slice(&synth_distribution(num_tags));
        out
    }

    fn synth_pruned(num_tags: u32, weight: u32) -> Vec<u8> {
        // PrunedInternal and Default share the same layout:
        // [1, N, weight] + distribution.
        let mut out = Vec::new();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&num_tags.to_le_bytes());
        out.extend_from_slice(&weight.to_le_bytes());
        out.extend_from_slice(&synth_distribution(num_tags));
        out
    }

    fn synth_internal(reserved: u32, back_pos_i: u32, test_tag_id: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&reserved.to_le_bytes());
        out.extend_from_slice(&back_pos_i.to_le_bytes());
        out.extend_from_slice(&test_tag_id.to_le_bytes());
        out
    }

    fn synth_default(num_tags: u32, weight: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&num_tags.to_le_bytes());
        out.extend_from_slice(&weight.to_le_bytes());
        out.extend_from_slice(&synth_distribution(num_tags));
        out
    }

    fn stub_header(num_tags: u32) -> Header {
        Header {
            field_a: 0,
            field_b: 0,
            sent_tag_index: 0,
            tags: (0..num_tags).map(|i| format!("T{i}")).collect(),
            end_offset: 0,
        }
    }

    /// Walker round-trips every kind on a hand-assembled section.
    #[test]
    fn walks_all_four_kinds_in_order() {
        let n = 3u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&synth_internal(2, 1, 42));
        bytes.extend_from_slice(&synth_leaf(7, n, 10));
        bytes.extend_from_slice(&synth_internal(1, 0, 99));
        bytes.extend_from_slice(&synth_pruned(n, 20));
        bytes.extend_from_slice(&synth_leaf(8, n, 15));
        bytes.extend_from_slice(&synth_default(n, 100));

        let header = stub_header(n);
        let mut cur = Cursor::new(&bytes);
        let tree = read(&mut cur, &header).unwrap();

        let kinds: Vec<_> = tree.records.iter().map(|r| r.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                DTreeKind::Internal,
                DTreeKind::Leaf,
                DTreeKind::Internal,
                DTreeKind::PrunedInternal,
                DTreeKind::Leaf,
                DTreeKind::Default,
            ]
        );

        let counts = tree.kind_counts();
        assert_eq!(counts.internals, 2);
        assert_eq!(counts.leaves, 2);
        assert_eq!(counts.pruned_internals, 1);
        assert_eq!(counts.defaults, 1);

        // Internal fields round-trip.
        let first_internal = tree.internals().next().unwrap();
        assert_eq!(first_internal.reserved, 2);
        assert_eq!(first_internal.back_pos_i, 1);
        assert_eq!(first_internal.test_tag_id, 42);

        // Leaf node_id round-trips.
        let first_leaf = tree.leaves().next().unwrap();
        assert_eq!(first_leaf.node_id, 7);
        assert_eq!(first_leaf.distribution.weight, 10);

        // Default accessor works.
        assert_eq!(tree.default().distribution.weight, 100);
    }

    /// A minimal `.par` tail of just a default record parses.
    #[test]
    fn walks_default_only() {
        let n = 3u32;
        let bytes = synth_default(n, 55);
        let header = stub_header(n);
        let mut cur = Cursor::new(&bytes);
        let tree = read(&mut cur, &header).unwrap();
        assert_eq!(tree.records.len(), 1);
        assert!(matches!(tree.records[0], DTreeRecord::Default(_)));
    }

    /// Bytes that don't look like any of the four kinds bail with a
    /// useful error pointing at the offset.
    #[test]
    fn rejects_unrecognized_shape() {
        let n = 3u32;
        // 12 bytes where every candidate kind fails:
        //   offset_i = 5 (out of {0,1,2,3}) → not Internal
        //   u32[0]   = 5 (!= 1)             → not PrunedInternal / Default
        //   u32[4]   = 0 (!= 1)             → not Leaf
        // Followed by a trailing default so the section is still
        // terminated (otherwise we'd get the trailing-default error
        // instead of the record-kind error).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&synth_default(n, 1));
        let header = stub_header(n);
        let mut cur = Cursor::new(&bytes);
        let err = read(&mut cur, &header).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("unrecognized decision-tree record"),
            "expected a walker-error message, got: {msg}"
        );
        assert!(msg.contains("section-offset 0"), "missing offset in: {msg}");
    }

    /// Must terminate on a Default — otherwise inference has no
    /// fallback distribution and we should refuse to proceed.
    #[test]
    fn requires_trailing_default() {
        let n = 3u32;
        // Leaf then another leaf, no default. The walker should
        // parse both and then complain that there's no Default at
        // EOF. But the second leaf's trailing bytes exactly fill
        // the file, so the trailing-default check kicks in.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&synth_leaf(1, n, 5));
        bytes.extend_from_slice(&synth_leaf(2, n, 6));
        let header = stub_header(n);
        let mut cur = Cursor::new(&bytes);
        let err = read(&mut cur, &header).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("didn't terminate with a Default"),
            "expected trailing-default error, got: {msg}"
        );
    }

    /// Preorder-DFS reconstruction on a tree with known shape:
    /// Internal(Leaf, Internal(Leaf, Default)) — 2 internals, 3
    /// leaves, with the trailing Default acting as the rightmost
    /// leaf of the last tree.
    #[test]
    fn reconstructs_small_tree() {
        let n = 3u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&synth_internal(1, 0, 10)); // root
        bytes.extend_from_slice(&synth_pruned(n, 5)); // root yes-child (leaf)
        bytes.extend_from_slice(&synth_internal(2, 1, 20)); // root no-child (internal)
        bytes.extend_from_slice(&synth_pruned(n, 6)); // inner yes-child (leaf)
        bytes.extend_from_slice(&synth_default(n, 99)); // inner no-child (leaf, Default at EOF)

        let tree = read(&mut Cursor::new(&bytes), &stub_header(n)).unwrap();
        let forest = tree.reconstruct().unwrap();
        assert_eq!(forest.roots.len(), 1);

        let root = &forest.nodes[forest.roots[0]];
        match root {
            TreeNode::Internal { predicate, yes, no } => {
                assert_eq!((predicate.reserved, predicate.back_pos_i, predicate.test_tag_id), (1, 0, 10));
                match &forest.nodes[*yes] {
                    TreeNode::Leaf { distribution, .. } => {
                        assert_eq!(distribution.weight, 5);
                    }
                    _ => panic!("root.yes should be a leaf"),
                }
                match &forest.nodes[*no] {
                    TreeNode::Internal { predicate: inner, yes: iy, no: in_, .. } => {
                        assert_eq!((inner.reserved, inner.back_pos_i, inner.test_tag_id), (2, 1, 20));
                        match &forest.nodes[*iy] {
                            TreeNode::Leaf { distribution, .. } => {
                                assert_eq!(distribution.weight, 6);
                            }
                            _ => panic!("inner.yes should be a leaf"),
                        }
                        match &forest.nodes[*in_] {
                            TreeNode::Leaf { distribution, .. } => {
                                assert_eq!(distribution.weight, 99, "should be the Default acting as rightmost leaf");
                            }
                            _ => panic!("inner.no should be a leaf (the Default)"),
                        }
                    }
                    _ => panic!("root.no should be an internal"),
                }
            }
            _ => panic!("root should be an internal"),
        }
    }

    /// `predict()` follows yes/no branches according to the
    /// reconstructed predicate. Tree shape:
    ///
    /// ```text
    /// root: tag_{-1} == 1?
    ///   yes → leaf weight=100
    ///   no  → inner: tag_{-2} == 2?
    ///           yes → leaf weight=200
    ///           no  → Default weight=300
    /// ```
    #[test]
    fn predict_follows_yes_no_branches() {
        let n = 3u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&synth_internal(0, 0, 1)); // root: tag_{-1} == 1?
        bytes.extend_from_slice(&synth_pruned(n, 100));    // yes leaf
        bytes.extend_from_slice(&synth_internal(0, 1, 2)); // inner: tag_{-2} == 2?
        bytes.extend_from_slice(&synth_pruned(n, 200));    // inner yes leaf
        bytes.extend_from_slice(&synth_default(n, 300));   // inner no leaf

        let tree = read(&mut Cursor::new(&bytes), &stub_header(n)).unwrap();
        let traversal = tree.traversal().unwrap();

        // tag_{-1} == 1 → yes leaf
        assert_eq!(traversal.predict(&[0, 1]).weight, 100);
        // tag_{-1} != 1, tag_{-2} == 2 → inner yes leaf
        assert_eq!(traversal.predict(&[2, 0]).weight, 200);
        // tag_{-1} != 1, tag_{-2} != 2 → default leaf
        assert_eq!(traversal.predict(&[0, 0]).weight, 300);
        // empty context — predicates always evaluate false → all no
        assert_eq!(traversal.predict(&[]).weight, 300);
    }

    /// English.par is known to reconstruct as a forest of 2 trees
    /// (63-node chain + 1500-node main tree) after stripping the
    /// wrapper Leaf and trailing Default.
    #[test]
    fn reconstructs_english_as_two_trees() {
        let Some(par) = english_par_path() else {
            return;
        };
        let bytes = std::fs::read(&par).unwrap();
        let mut cur = Cursor::new(&bytes);
        cur.advance(0xd231bb).unwrap();
        let header = Header {
            field_a: 0,
            field_b: 0,
            sent_tag_index: 31,
            tags: (0..58).map(|i| format!("T{i}")).collect(),
            end_offset: 0,
        };
        let tree = read(&mut cur, &header).unwrap();
        let forest = tree.reconstruct().unwrap();
        assert_eq!(forest.roots.len(), 2, "english.par should be a 2-tree forest");
        let n0 = subtree_size(&forest.nodes, forest.roots[0]);
        let n1 = subtree_size(&forest.nodes, forest.roots[1]);
        eprintln!("english.par forest: tree[0]={n0} nodes, tree[1]={n1} nodes");
        assert_eq!(n0, 63, "first tree should be 63 nodes");
        // Wrapper stripped, Default kept as a leaf of the last tree.
        // Body = 1565 - 1 (wrapper) = 1564 records. n0 + n1 = 1564.
        assert_eq!(n0 + n1, 1564);
    }

    fn subtree_size(nodes: &[TreeNode], root: usize) -> usize {
        match &nodes[root] {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Internal { yes, no, .. } => {
                1 + subtree_size(nodes, *yes) + subtree_size(nodes, *no)
            }
        }
    }

    /// Real english.par: the bundled model parses without a walker
    /// error, ends on a Default, and every distribution sums to ~1.0.
    /// Also prints the kind counts so they're visible in `cargo test -- --nocapture`.
    #[test]
    fn reads_bundled_english_decision_tree() {
        let Some(par) = english_par_path() else {
            return;
        };
        let bytes = std::fs::read(&par).unwrap();
        let tree_start = 0xd231bb;
        let mut cur = Cursor::new(&bytes);
        cur.advance(tree_start).unwrap();
        let header = Header {
            field_a: 0,
            field_b: 0,
            sent_tag_index: 31,
            tags: (0..58).map(|i| format!("T{i}")).collect(),
            end_offset: 0,
        };
        let tree = read(&mut cur, &header).unwrap();
        let counts = tree.kind_counts();
        eprintln!(
            "english.par dtree records: {} internals, {} leaves, \
             {} pruned-internals, {} defaults ({} total)",
            counts.internals,
            counts.leaves,
            counts.pruned_internals,
            counts.defaults,
            tree.records.len()
        );
        assert_eq!(counts.defaults, 1);
        // Default is last.
        assert!(matches!(tree.records.last(), Some(DTreeRecord::Default(_))));
        // Every distribution sums to 1.0.
        for (i, r) in tree.records.iter().enumerate() {
            let dist = match r {
                DTreeRecord::Leaf(l) => &l.distribution,
                DTreeRecord::PrunedInternal(p) => &p.distribution,
                DTreeRecord::Default(d) => &d.distribution,
                DTreeRecord::Internal(_) => continue,
            };
            let s: f64 = dist.probs.iter().map(|tp| tp.prob).sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "record {i} ({:?}) distribution sum = {s}",
                r.kind()
            );
            assert_eq!(dist.probs.len(), 58);
        }
    }
}
