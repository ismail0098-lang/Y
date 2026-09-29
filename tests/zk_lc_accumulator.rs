//! `LinearCombination::simplify` must not rescan an accumulator that never
//! stopped being sorted.
//!
//! The dot-product benchmark circuit (`sum = sum + a * b` in a loop) was
//! **O(N²)** in Y's emitter, and the whole of it was here. Each iteration
//! appends one fresh product wire to `sum`, then called `simplify`, whose
//! "already sorted" fast path still scans every term to decide it has nothing
//! to do. With `sum` holding `i` terms at iteration `i`, that is N²/2 term
//! visits — measured at N=40,000 as 119,999 calls but **800,259,997 terms**,
//! while the field-operation counters stayed perfectly linear and every
//! phase-level timer just showed `emit_circuit_entry` getting slower.
//!
//! Wire ids are allocated in increasing order, so appending a freshly allocated
//! wire to a sorted combination cannot break sortedness. `append_keeps_order`
//! decides that from the boundary term alone, in O(1).
//!
//! **These tests pin work, not wall time**, via `lc_simplify_stats` — a timing
//! assertion for an asymptotic property is either flaky or so loose it passes
//! through the regression it exists to catch.

#![cfg(feature = "zk")]

use std::collections::BTreeMap;

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{
    lc_simplify_stats, reset_lc_simplify_stats, Fr, LinearCombination, ZkEmitter,
};

/// Terms scanned by `simplify` while emitting the dot product at size `n`.
fn dot_product_simplify_terms(n: usize) -> u64 {
    let src = format!(
        "@unsafe\nfn main(x: I32, y: I32) -> I32 {{\n\
         \x20   let mut sum = 0;\n\
         \x20   let mut a = x;\n\
         \x20   let mut b = y;\n\
         \x20   for i in 0..{} {{\n\
         \x20       a = a + 1;\n\
         \x20       b = b + 1;\n\
         \x20       sum = sum + a * b;\n\
         \x20   }}\n\
         \x20   return sum;\n}}\n",
        n
    );
    std::env::set_var("Y_ZK_MAX_UNROLL", "40000000");
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);

    reset_lc_simplify_stats();
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("lower");
    let circuit = emitter.build_circuit();
    // Sanity: the circuit really is the size we asked for, so a "linear" result
    // cannot come from the emitter quietly dropping the loop.
    assert_eq!(circuit.constraints.len(), n + 1, "unexpected constraint count");
    lc_simplify_stats().1
}

#[test]
fn dot_product_simplify_work_is_linear() {
    let a = dot_product_simplify_terms(2_000);
    let b = dot_product_simplify_terms(4_000);
    let c = dot_product_simplify_terms(8_000);

    // Quadratic would be ~4x per doubling. Linear is ~2x. The gap is so wide
    // that a generous bound still catches the regression: before the fix these
    // were 2,001,999 / 8,003,999 / 32,007,999.
    assert!(
        b <= a * 5 / 2 && c <= b * 5 / 2,
        "simplify work is growing super-linearly: {} -> {} -> {} terms \
         for N = 2000 -> 4000 -> 8000",
        a,
        b,
        c
    );
    // And it should be genuinely small, not merely sub-quadratic: the emitter
    // does a bounded amount of simplify work per loop iteration.
    assert!(c < 8_000 * 32, "unexpectedly heavy: {} terms at N=8000", c);
}

/// Terms scanned emitting an accumulator that *also* adds an older wire.
///
/// `s = s + x` names an input wire, whose id is small, so it can never be
/// appended in order. Handled by `merge_in_place` — after `x` first appears
/// there is nothing to reorder, only a coefficient to add.
fn mixed_accumulator_simplify_terms(n: usize) -> u64 {
    let src = format!(
        "@unsafe\nfn main(x: I32, y: I32) -> I32 {{\n\
         \x20   let mut s = 0;\n\
         \x20   let mut a = x;\n\
         \x20   for i in 0..{} {{\n\
         \x20       a = a + 1;\n\
         \x20       s = s + a * y;\n\
         \x20       s = s + x;\n\
         \x20   }}\n\
         \x20   return s;\n}}\n",
        n
    );
    std::env::set_var("Y_ZK_MAX_UNROLL", "40000000");
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);
    reset_lc_simplify_stats();
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("lower");
    let _ = emitter.build_circuit();
    lc_simplify_stats().1
}

#[test]
fn accumulator_that_also_adds_an_input_wire_is_linear() {
    // This shape defeats the append shortcut entirely and was still O(N²) after
    // the first fix: 2,010,998 -> 8,021,998 -> 32,043,998 terms.
    let a = mixed_accumulator_simplify_terms(2_000);
    let b = mixed_accumulator_simplify_terms(4_000);
    let c = mixed_accumulator_simplify_terms(8_000);
    assert!(
        b <= a.max(64) * 5 / 2 && c <= b.max(64) * 5 / 2,
        "mixed accumulator is super-linear: {} -> {} -> {}",
        a,
        b,
        c
    );
}

#[test]
fn adding_a_wire_already_present_updates_in_place() {
    let mut lc = LinearCombination::zero();
    lc.add_term(3, Fr::one());
    lc.add_term(9, Fr::one());
    lc.simplify();
    assert!(lc.is_simplified);

    reset_lc_simplify_stats();
    // Out of order (3 < 9) but already present: no reordering is needed, so the
    // combination must stay simplified and `simplify` must stay idle.
    lc.add_term(3, Fr::from_u64(4));
    lc.simplify();
    assert_eq!(lc_simplify_stats(), (0, 0), "an in-place update forced a sort");
    assert_eq!(lc.terms, vec![(3, Fr::from_u64(5)), (9, Fr::one())]);
    assert!(lc.invariant_holds());

    // Same via `add_linear`, and with a coefficient that cancels to zero. A
    // zero coefficient breaks the invariant, so the flag must drop and the
    // following `simplify` must clean the term out.
    let mut neg = LinearCombination::zero();
    neg.add_term(3, Fr::zero().sub(&Fr::from_u64(5)));
    neg.simplify();
    lc.add_linear(&neg, Fr::one());
    assert!(!lc.is_simplified, "a cancelled coefficient was left claimed simplified");
    lc.simplify();
    assert_eq!(lc.terms, vec![(9, Fr::one())], "zero term was not removed");
    assert!(lc.invariant_holds());
}

#[test]
fn a_missing_wire_falls_back_and_leaves_the_combination_intact() {
    // `merge_in_place` must be all-or-nothing: if any wire of the addend is
    // absent it has to bail out *without* having applied the others, or the
    // fallback push would double-count them.
    let mut lc = LinearCombination::zero();
    lc.add_term(2, Fr::one());
    lc.add_term(6, Fr::one());
    lc.simplify();

    let mut addend = LinearCombination::zero();
    addend.add_term(2, Fr::one()); // present
    addend.add_term(4, Fr::one()); // absent -> forces the fallback
    addend.simplify();

    lc.add_linear(&addend, Fr::one());
    lc.simplify();
    assert_eq!(
        lc.terms,
        vec![(2, Fr::from_u64(2)), (4, Fr::one()), (6, Fr::one())],
        "wire 2 was counted twice, or wire 4 was lost"
    );
    assert!(lc.invariant_holds());
}

#[test]
fn appending_ascending_wires_keeps_the_combination_simplified() {
    // Exactly the shape the emitter builds: an accumulator fed one freshly
    // allocated (therefore larger) wire at a time.
    let mut acc = LinearCombination::zero();
    reset_lc_simplify_stats();
    for wire in 1..=500usize {
        acc.add_linear(&LinearCombination::variable(wire), Fr::one());
        acc.simplify();
    }
    assert_eq!(acc.terms.len(), 500);
    assert!(acc.invariant_holds());

    let (calls, terms) = lc_simplify_stats();
    assert_eq!(
        (calls, terms),
        (0, 0),
        "simplify did work it did not need to: {} calls, {} terms",
        calls,
        terms
    );
}

#[test]
fn out_of_order_append_is_not_claimed_simplified() {
    // The control for the test above. If `append_keeps_order` ever returned
    // `true` unconditionally, every assertion about the fast path would still
    // pass while `simplify` silently stopped sorting — so pin the negative.
    let mut lc = LinearCombination::variable(9);
    lc.add_term(4, Fr::one()); // descending: must invalidate
    assert!(!lc.terms.is_empty());
    let before = lc_simplify_stats().1;
    lc.simplify();
    assert!(
        lc_simplify_stats().1 > before,
        "simplify skipped a combination that was genuinely out of order"
    );
    assert_eq!(lc.terms[0].0, 4, "terms were not sorted");
    assert_eq!(lc.terms[1].0, 9);
    assert!(lc.invariant_holds());

    // A duplicate wire id must also invalidate, or coefficients stop merging.
    let mut dup = LinearCombination::variable(7);
    dup.add_term(7, Fr::one());
    dup.simplify();
    assert_eq!(dup.terms.len(), 1, "duplicate wires were not merged");
    assert_eq!(dup.terms[0].1, Fr::from_u64(2));

    // Wire 0 is the constant and sorts before everything, so appending it to a
    // non-empty combination must invalidate too.
    let mut with_const = LinearCombination::variable(3);
    with_const.add_constant(Fr::one());
    with_const.simplify();
    assert_eq!(with_const.terms[0].0, 0);
    assert!(with_const.invariant_holds());
}

/// Reference simplification: the obvious O(n log n) version, kept separate so
/// the fast path is checked against something that cannot share its bug.
fn reference_simplify(terms: &[(usize, Fr)]) -> Vec<(usize, Fr)> {
    let mut map: BTreeMap<usize, Fr> = BTreeMap::new();
    for (w, c) in terms {
        let e = map.entry(*w).or_insert_with(Fr::zero);
        *e = e.add(c);
    }
    map.into_iter().filter(|(_, c)| !c.is_zero()).collect()
}

#[test]
fn fast_path_agrees_with_a_reference_over_random_sequences() {
    // A hand-rolled LCG: this needs to be deterministic and the crate has no
    // rand dependency outside dev-only arkworks.
    let mut state: u64 = 0x5eed_1234_9abc_def0;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };

    for case in 0..400 {
        let mut lc = LinearCombination::zero();
        let mut shadow: Vec<(usize, Fr)> = Vec::new();
        let ops = 1 + next() % 12;
        for _ in 0..ops {
            match next() % 4 {
                0 => {
                    let w = next() % 10;
                    let v = Fr::from_u64((next() % 5) as u64);
                    lc.add_term(w, v);
                    if !v.is_zero() {
                        shadow.push((w, v));
                    }
                }
                1 => {
                    let v = Fr::from_u64((next() % 5) as u64);
                    lc.add_constant(v);
                    if !v.is_zero() {
                        shadow.push((0, v));
                    }
                }
                2 => {
                    // Ascending block, the emitter's own pattern.
                    let base = next() % 8;
                    let mut other = LinearCombination::zero();
                    for k in 0..(1 + next() % 3) {
                        other.add_term(base + k * 3 + 1, Fr::from_u64(1 + (next() % 4) as u64));
                    }
                    other.simplify();
                    let scale = Fr::from_u64(1 + (next() % 3) as u64);
                    lc.add_linear(&other, scale);
                    for (w, c) in &other.terms {
                        shadow.push((*w, c.mul(&scale)));
                    }
                }
                _ => {
                    lc.simplify();
                }
            }
        }
        lc.simplify();
        assert!(lc.invariant_holds(), "case {}: invariant broken", case);
        assert_eq!(
            lc.terms,
            reference_simplify(&shadow),
            "case {}: fast path disagrees with the reference",
            case
        );
    }
}

#[test]
fn scaling_preserves_the_simplified_flag_soundly() {
    // `scale` copies `is_simplified` across, which is only sound because a
    // non-zero factor cannot zero a coefficient or reorder wires in a field.
    let mut lc = LinearCombination::variable(2);
    lc.add_term(5, Fr::from_u64(3));
    lc.simplify();
    let scaled = lc.scale(Fr::from_u64(7));
    assert!(scaled.invariant_holds());
    assert_eq!(scaled.terms, vec![(2, Fr::from_u64(7)), (5, Fr::from_u64(21))]);

    // Scaling by zero must produce the empty combination, not a row of zero
    // coefficients that claim to be simplified.
    let zeroed = lc.scale(Fr::zero());
    assert!(zeroed.terms.is_empty());
    assert!(zeroed.invariant_holds());
}
