//! Bigram-Viterbi tagger over the dtree.
//!
//! Per-position score is `dcram/treetaggerj`'s `Tagger.java`
//! formula:
//!
//! ```text
//! argmax_t  P(t | w) × P(t | ctx)
//! ```
//!
//! where `P(t | w)` is the lexicon's per-word distribution and
//! `P(t | ctx)` is the dtree's prediction at the leaf reached by
//! `Traversal::predict` (the bigram tree alone — interpolation
//! between bigram and unigram trees regressed across every
//! Schmid 1995-style scheme tried, see #11 for the experimental
//! detail).
//!
//! **Lexicon-confidence pruning** is what makes this beat the
//! lex-only baseline: candidates with `lex_prob < threshold × max
//! lex_prob` are dropped before the dtree gets a vote. Without
//! this, words like "King" (lex(NP)=0.97, lex(NN)=0.03) get
//! flipped to NN by the dtree's domain-general NN preference.
//! Threshold 0.75 was the empirical sweep peak on the Gutenberg
//! 2032-token sample — 92.62% vs 92.42% lex-only baseline (+4
//! tokens).
//!
//! State space is bounded by `num_tags^2` ≈ 3.4k for english.par;
//! in practice only a small fraction is reachable per position so
//! the DP runs in milliseconds on a typical sentence. Initial
//! state has `tag_{-1}` and `tag_{-2}` both `None` — mirrors
//! treetaggerj's `getStartTag()` synthetic start marker that no
//! dtree internal can match.

use std::collections::HashMap;

use crate::par::dtree::Traversal;
use crate::par::header::Header;

/// One candidate tag for a single token. Lemma is pre-resolved at
/// candidate-collection time so the inner DP doesn't touch the
/// lemma pool.
#[derive(Debug, Clone)]
pub struct Cand {
    pub tag_id: u32,
    /// `P(tag | word)` from the lexicon, or `1.0` for unknown words
    /// (unknown-word path produces a single forced candidate).
    pub lex_prob: f64,
    pub lemma: Option<String>,
}

/// Output of one token's tagging step.
#[derive(Debug, Clone)]
pub struct Tagged {
    pub pos: Option<String>,
    pub lemma: Option<String>,
}

/// `(t_{-1}, t_{-2})`. `None` means "no tag at that position yet" —
/// happens at the start of the document.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct State {
    last: Option<u32>,
    second_last: Option<u32>,
}

impl State {
    fn shift(self, t: u32) -> Self {
        Self {
            last: Some(t),
            second_last: self.last,
        }
    }

    /// Materialize the state as a `[t_{-2}, t_{-1}]` slice (oldest
    /// first) for `Traversal::predict`. Missing tags are dropped from
    /// the left so the latest tag is always at the end of the slice
    /// — that keeps `back_pos_i=0` reading `t_{-1}` correctly.
    fn write_context(self, buf: &mut Vec<u32>) {
        buf.clear();
        if let Some(t2) = self.second_last {
            buf.push(t2);
        }
        if let Some(t1) = self.last {
            buf.push(t1);
        }
    }
}

/// Run Viterbi over the candidate lattice and return one Tagged per
/// input position. `cands_per_token` must have at least one
/// candidate per position; callers that can't produce any candidate
/// for a token should synthesize one (e.g. an unknown-word fallback)
/// rather than passing an empty vec.
pub fn tag_sequence(
    cands_per_token: &[Vec<Cand>],
    traversal: &Traversal,
    header: &Header,
) -> Vec<Tagged> {
    let n = cands_per_token.len();
    if n == 0 {
        return Vec::new();
    }

    let _ = header;

    // Initial state mirrors `dcram/treetaggerj`'s `getStartTag()` —
    // a synthetic start marker that no dtree internal can match,
    // so all predicates fail and the first token's prediction
    // comes from the all-no-path leaf. We model this with `None`
    // for both `tag_{-1}` and `tag_{-2}`, producing an empty context
    // slice into `Traversal::predict`.
    let mut dps: Vec<HashMap<State, (f64, Option<(usize, State)>)>> =
        Vec::with_capacity(n + 1);
    let mut init = HashMap::new();
    init.insert(
        State {
            last: None,
            second_last: None,
        },
        (0.0_f64, None),
    );
    dps.push(init);

    // When the lexicon is highly confident in one tag (e.g.
    // "King" → NP at 0.97, "How" → WRB at 0.999), the dtree's
    // out-of-domain preference for a more frequent tag can flip the
    // answer the wrong way under bare `lex × tree`. Pruning
    // candidates with `lex_prob < threshold × max_lex_prob` keeps
    // the dtree's vote restricted to genuine lexical ambiguities.
    // 0.75 was empirically optimal on the Gutenberg sample (sweep
    // peak at 92.62% vs 92.42% lex-only baseline, +4 tokens).
    let pruning_threshold = 0.75_f64;
    let mut ctx_buf: Vec<u32> = Vec::with_capacity(2);
    for i in 0..n {
        let cands = &cands_per_token[i];
        let max_lex = cands
            .iter()
            .map(|c| c.lex_prob)
            .fold(0.0_f64, f64::max);
        let cutoff = max_lex * pruning_threshold;
        let mut next: HashMap<State, (f64, Option<(usize, State)>)> = HashMap::new();
        for (state, &(score, _)) in &dps[i] {
            state.write_context(&mut ctx_buf);
            let cond = traversal.predict(&ctx_buf);
            for (cand_idx, c) in cands.iter().enumerate() {
                if c.lex_prob < cutoff {
                    continue;
                }
                let p_cond = cond
                    .probs
                    .iter()
                    .find(|tp| tp.tag_id == c.tag_id)
                    .map(|tp| tp.prob)
                    .unwrap_or(0.0);
                // Schmid's per-token formula is the bare HMM joint:
                // argmax_t  P(t | w) × P(t | ctx)
                // (treetaggerj's reference implementation uses this
                // exact form — see Tagger.java#getMostProbable.)
                let local = c.lex_prob * p_cond;
                if local <= 0.0 {
                    continue;
                }
                let new_score = score + local.ln();
                let new_state = state.shift(c.tag_id);
                let entry = next
                    .entry(new_state)
                    .or_insert((f64::NEG_INFINITY, None));
                if entry.0 < new_score {
                    *entry = (new_score, Some((cand_idx, *state)));
                }
            }
        }
        if next.is_empty() {
            // No path made it through this position — every candidate
            // produced a zero or negative joint. Fall back to a
            // lexicon-only pick so the trace-back can complete.
            let best_cand = (0..cands.len())
                .max_by(|&a, &b| {
                    cands[a]
                        .lex_prob
                        .partial_cmp(&cands[b].lex_prob)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            for (state, &(score, _)) in &dps[i] {
                let new_state = state.shift(cands[best_cand].tag_id);
                next.insert(new_state, (score, Some((best_cand, *state))));
                break; // any state suffices for the recovery path
            }
        }
        dps.push(next);
    }

    // Trace back from argmax of dps[n].
    let mut chosen = vec![0usize; n];
    if let Some(start) = dps[n]
        .iter()
        .max_by(|a, b| {
            a.1
                .0
                .partial_cmp(&b.1.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(s, _)| *s)
    {
        let mut cur_state = start;
        for i in (0..n).rev() {
            let (cand_idx, prev_state) = dps[i + 1][&cur_state]
                .1
                .expect("viterbi backpointer missing");
            chosen[i] = cand_idx;
            cur_state = prev_state;
        }
    }

    let mut out = Vec::with_capacity(n);
    for (i, idx) in chosen.into_iter().enumerate() {
        let c = &cands_per_token[i][idx];
        out.push(Tagged {
            pos: header.tag(c.tag_id).map(str::to_owned),
            lemma: c.lemma.clone(),
        });
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::dtree::{
        DTreeRecord, Distribution, Internal, Leaf, PrunedInternal, TagProb, TreeForest,
        TreeNode, Traversal,
    };

    fn synth_dist(weights: &[f64]) -> Distribution {
        let total: f64 = weights.iter().sum();
        let probs = weights
            .iter()
            .enumerate()
            .map(|(k, &w)| TagProb {
                tag_id: k as u32,
                prob: w / total,
            })
            .collect();
        Distribution {
            weight: total as u32,
            probs,
        }
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

    /// Synthesize a tiny TreeForest with one tree:
    ///
    ///   root: tag_{-1} == 1?
    ///     yes → leaf with high P(tag 0)
    ///     no  → leaf with high P(tag 1)
    ///
    /// Then verify viterbi picks the right tag for two-token
    /// sentences where the first word forces the test condition.
    #[test]
    fn viterbi_follows_dtree_yes_branch() {
        let n = 3u32;
        let yes_dist = synth_dist(&[0.7, 0.2, 0.1]); // favors tag 0
        let no_dist = synth_dist(&[0.1, 0.7, 0.2]); // favors tag 1

        // forest has 1 tree, 1 internal + 2 leaves.
        let forest = TreeForest {
            nodes: vec![
                TreeNode::Internal {
                    predicate: Internal {
                        reserved: 0,
                        back_pos_i: 0,
                        test_tag_id: 1,
                    },
                    yes: 1,
                    no: 2,
                },
                TreeNode::Leaf {
                    node_id: None,
                    distribution: yes_dist.clone(),
                },
                TreeNode::Leaf {
                    node_id: None,
                    distribution: no_dist.clone(),
                },
            ],
            roots: vec![0],
            wrapper_records: 0,
        };
        let marginal = synth_dist(&[1.0; 3]);
        let traversal = Traversal { forest, root: 0, marginal };
        let header = stub_header(n);

        // Token A: only candidate is tag 1 (forces context = [_, 1]).
        // Token B: candidates {tag 0, tag 1} with equal lex prob.
        // After token A=1, dtree predicts P(tag 0) = 0.7 → token B
        // should pick tag 0.
        let cands = vec![
            vec![Cand {
                tag_id: 1,
                lex_prob: 1.0,
                lemma: None,
            }],
            vec![
                Cand {
                    tag_id: 0,
                    lex_prob: 0.5,
                    lemma: None,
                },
                Cand {
                    tag_id: 1,
                    lex_prob: 0.5,
                    lemma: None,
                },
            ],
        ];
        let out = tag_sequence(&cands, &traversal, &header);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].pos.as_deref(), Some("T1"));
        assert_eq!(out[1].pos.as_deref(), Some("T0"));
    }

    /// Same setup but the first word forces tag 0 instead of 1, so
    /// the dtree's no-branch (favoring tag 1) applies.
    #[test]
    fn viterbi_follows_dtree_no_branch() {
        let n = 3u32;
        let yes_dist = synth_dist(&[0.7, 0.2, 0.1]);
        let no_dist = synth_dist(&[0.1, 0.7, 0.2]);
        let forest = TreeForest {
            nodes: vec![
                TreeNode::Internal {
                    predicate: Internal {
                        reserved: 0,
                        back_pos_i: 0,
                        test_tag_id: 1,
                    },
                    yes: 1,
                    no: 2,
                },
                TreeNode::Leaf {
                    node_id: None,
                    distribution: yes_dist,
                },
                TreeNode::Leaf {
                    node_id: None,
                    distribution: no_dist,
                },
            ],
            roots: vec![0],
            wrapper_records: 0,
        };
        let marginal = synth_dist(&[1.0; 3]);
        let traversal = Traversal { forest, root: 0, marginal };
        let header = stub_header(n);

        let cands = vec![
            vec![Cand {
                tag_id: 0,
                lex_prob: 1.0,
                lemma: None,
            }],
            vec![
                Cand {
                    tag_id: 0,
                    lex_prob: 0.5,
                    lemma: None,
                },
                Cand {
                    tag_id: 1,
                    lex_prob: 0.5,
                    lemma: None,
                },
            ],
        ];
        let out = tag_sequence(&cands, &traversal, &header);
        assert_eq!(out[0].pos.as_deref(), Some("T0"));
        assert_eq!(out[1].pos.as_deref(), Some("T1"));
    }

    /// Suppress dead-code warnings on imports we rely on for the
    /// surface API but don't use directly here.
    #[allow(dead_code)]
    fn _types_in_scope(_: DTreeRecord, _: Leaf, _: PrunedInternal) {}
}
