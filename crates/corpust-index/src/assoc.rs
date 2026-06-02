//! Collocation association measures.
//!
//! Given the windowed co-occurrence count of a collocate around a node
//! term plus the corpus marginals, compute the standard strength scores
//! a corpus-linguistics tool reports. Formulas follow the conventions
//! used by LancsBox / Stefan Evert's collocation literature:
//!
//! - **log Dice** = `14 + log2(2·O11 / (f(node) + f(collocate)))`
//!   (Rychlý 2008) — the LancsBox default; bounded and corpus-size
//!   independent.
//! - **MI** (pointwise mutual information) = `log2(O11 / E11)`.
//! - **z-score** = `(O11 − E11) / sqrt(E11)`.
//!
//! where the expected co-occurrence under independence is
//! `E11 = f(node) · span · f(collocate) / N` — `f(node)·span` is the
//! total number of collocate slots examined across every node
//! occurrence, and `f(collocate)/N` is the collocate's corpus-wide rate.
//!
//! Every result is guaranteed finite: degenerate inputs (a zero
//! marginal, no co-occurrence) yield `0.0` rather than `NaN`/`±∞`, which
//! matters because the scores cross a JSON IPC boundary where non-finite
//! floats would serialise to `null` and break the UI.

/// The three association scores the UI exposes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scores {
    pub log_dice: f64,
    pub mi: f64,
    pub z: f64,
}

/// Compute association scores for one collocate.
///
/// - `o11`  — observed co-occurrence count (collocate seen in the node's window)
/// - `f_node` — corpus frequency of the node term
/// - `f_coll` — corpus frequency of the collocate
/// - `n` — corpus size (total tokens on the surface layer)
/// - `span` — window width in token positions (`left + right`)
pub fn scores(o11: u64, f_node: u64, f_coll: u64, n: u64, span: u64) -> Scores {
    let o = o11 as f64;
    let f_n = f_node as f64;
    let f_c = f_coll as f64;
    let nn = n as f64;
    let s = span as f64;

    // log Dice — defined whenever there is co-occurrence and a non-zero
    // combined marginal. Bounded above by 14 (when every node and
    // collocate token co-occurs).
    let log_dice = if o11 > 0 && f_node + f_coll > 0 {
        14.0 + (2.0 * o / (f_n + f_c)).log2()
    } else {
        0.0
    };

    // Expected co-occurrence under independence.
    let e11 = if n > 0 { f_n * s * f_c / nn } else { 0.0 };

    let mi = if o11 > 0 && e11 > 0.0 {
        (o / e11).log2()
    } else {
        0.0
    };

    let z = if e11 > 0.0 {
        (o - e11) / e11.sqrt()
    } else {
        0.0
    };

    Scores { log_dice, mi, z }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All scores stay finite for degenerate inputs — they must never
    /// reach the JSON boundary as NaN/Inf.
    #[test]
    fn degenerate_inputs_are_finite_zeros() {
        for s in [
            scores(0, 100, 50, 1000, 10), // no co-occurrence
            scores(5, 0, 50, 1000, 10),   // node never occurs
            scores(5, 100, 0, 1000, 10),  // collocate never occurs
            scores(5, 100, 50, 0, 10),    // empty corpus
            scores(0, 0, 0, 0, 0),        // everything zero
        ] {
            assert!(s.log_dice.is_finite(), "log_dice not finite: {s:?}");
            assert!(s.mi.is_finite(), "mi not finite: {s:?}");
            assert!(s.z.is_finite(), "z not finite: {s:?}");
        }
    }

    #[test]
    fn mi_matches_hand_computation() {
        // O11=10, f_node=100, f_coll=200, N=10_000, span=10.
        // E11 = 100*10*200/10000 = 20. MI = log2(10/20) = -1.
        let s = scores(10, 100, 200, 10_000, 10);
        assert!((s.mi - (-1.0)).abs() < 1e-9, "mi = {}", s.mi);
        // z = (10 - 20)/sqrt(20) = -2.2360679...
        assert!((s.z - (-2.236_067_977)).abs() < 1e-6, "z = {}", s.z);
    }

    #[test]
    fn log_dice_matches_hand_computation() {
        // logDice = 14 + log2(2*O11/(f_node+f_coll))
        //         = 14 + log2(2*30/(100+200)) = 14 + log2(0.2) = 14 - 2.3219..
        let s = scores(30, 100, 200, 10_000, 10);
        let expected = 14.0 + (0.2_f64).log2();
        assert!(
            (s.log_dice - expected).abs() < 1e-9,
            "log_dice = {}",
            s.log_dice
        );
    }

    /// A collocate that co-occurs far more than chance scores positively
    /// on every measure; the inverse case (below chance) goes negative on
    /// MI/z — the case the scatter's y-axis must render.
    #[test]
    fn attraction_positive_repulsion_negative() {
        let attracted = scores(50, 100, 60, 100_000, 10);
        assert!(attracted.mi > 0.0 && attracted.z > 0.0);

        let repelled = scores(1, 100, 5000, 100_000, 10);
        assert!(repelled.mi < 0.0, "mi = {}", repelled.mi);
        assert!(repelled.z < 0.0, "z = {}", repelled.z);
    }
}
