//! The division gadget's soundness, decided over its whole bounded domain.
//!
//! `tests/zk_integer_ops.rs::forged_quotient_is_rejected` exhibits ONE forgery
//! at ONE input: it forges `q = 7 * inv(2)`, `r = 0` on `7 % 2` and checks the
//! circuit rejects it. That is a sample, not the property. The property is
//! universal - *no* assignment satisfying the constraints has `q` other than
//! `a / b` - and a single counterexample-shaped test cannot say it.
//!
//! `CLAUDE.md` records that Z3 could NOT settle the corresponding question for
//! the comparison gadget: "254-bit modular arithmetic over `Int` defeats the
//! solver". That is true of the question as posed there, and it is not true
//! here, for a reason worth stating rather than rediscovering. The range
//! checks make the domain FINITE: `q`, `b` and `r` are all below `2^n`, so the
//! products in play are below `2^2n` and the modulus never enters. Every query
//! below is decided in well under a second.
//!
//! The chain is four steps and each licenses the next:
//!
//!   1. `q * b + r` cannot reach the modulus, so the field equation
//!      `q * b = a - r` IS an integer equation. Without this the rest is
//!      reasoning about the wrong structure.
//!   2. Over the integers that equation, plus `r < b`, pins `q` and `r`
//!      uniquely. This is the general form of the forged-quotient test.
//!   3. The largest dividend any witness can represent is exactly
//!      `max_representable_dividend()` - achievable, and nothing above it is.
//!   4. The compile-time guards the emitter applies to a folded `/` or `%`
//!      accept exactly the operand pairs the gadget can satisfy.
//!
//! Step 3 is a two-sided pin, which is why the test reads the emitter's own
//! constant rather than a copy. Move that constant UP and the achievability
//! query goes unsat; move it DOWN and the supremum query goes sat. Z3 is the
//! independent oracle, so sharing the constant with the emitter is what makes
//! a drift visible instead of moving both sides together - the failure mode
//! recorded for the generated-schedule gate.
//!
//! Run with:  cargo test --features zk --test zk_divmod_soundness
#![cfg(feature = "zk")]

use std::io::Write;
use std::process::{Command, Stdio};
use y::type_checker::z3_candidates;
use y::zk_emitter::{max_representable_dividend, ZK_COMPARISON_BITS};

/// BN254's Fr modulus. Only step 1 needs it, and only as an upper barrier.
const BN254_FR: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

/// The first Z3 that runs, using the compiler's own search path.
///
/// One search path, not two: `z3_candidates` exists because this repo had a
/// perfectly good solver in `venv/bin/z3` that the compiler could not see, and
/// a test with its own list would be free to develop the same blind spot.
fn z3() -> Option<String> {
    z3_candidates().into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// `sat`, `unsat`, or a panic. Never "the solver did not run".
fn solve(z3_bin: &str, name: &str, body: &str) -> String {
    let mut child = Command::new(z3_bin)
        .args(["-T:120", "-in"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawning z3 for `{}`: {}", name, e));
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(format!("{}\n(check-sat)\n", body).as_bytes())
        .expect("writing the query");
    let out = child.wait_with_output().expect("z3 did not finish");
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        text == "sat" || text == "unsat",
        "z3 answered `{}` for `{}` (stderr: {}). `unknown` or a timeout is a \
         FAILURE here, not a skip: the whole claim of this file is that these \
         queries are decidable because the range checks bound the domain.",
        text,
        name,
        String::from_utf8_lossy(&out.stderr).trim()
    );
    text
}

/// The declarations every query shares: one witness of the gadget, in range.
///
/// Stated from `ZK_COMPARISON_BITS`, so changing the gadget's width re-poses
/// every obligation at the new width instead of silently checking the old one.
fn a_witness_in_range() -> String {
    let bound = 1u64 << ZK_COMPARISON_BITS;
    format!(
        "(declare-const a Int)(declare-const b Int)\n\
         (declare-const q Int)(declare-const r Int)\n\
         ; emit_num2bits(q): the quotient is range-checked.\n\
         (assert (and (>= q 0) (< q {bound})))\n\
         ; emit_less_than(r, b): the remainder and the DIVISOR are, and r < b.\n\
         (assert (and (>= b 0) (< b {bound})))\n\
         (assert (and (>= r 0) (< r b)))\n\
         ; the one constraint tying them to the dividend.\n\
         (assert (= (+ (* q b) r) a))"
    )
}

fn max_dividend() -> String {
    max_representable_dividend().to_decimal_string()
}

#[test]
fn the_division_gadget_is_sound_over_its_whole_domain() {
    let Some(z3_bin) = z3() else {
        println!(
            "SKIP: no z3 found on any of {:?}. This file is the only universal \
             statement of the division gadget's soundness; without a solver the \
             property is asserted by one concrete forgery and nothing else.",
            z3_candidates()
        );
        return;
    };
    let bound = 1u64 << ZK_COMPARISON_BITS;
    let w = a_witness_in_range();

    // 1. NO WRAP. If `q*b + r` could reach the modulus, a second residue would
    //    satisfy the same constraint and every integer argument below would be
    //    about the wrong structure.
    assert_eq!(
        solve(
            &z3_bin,
            "no-wrap",
            &format!("{w}\n(assert (>= a {BN254_FR}))")
        ),
        "unsat",
        "some in-range witness reaches the modulus, so `q * b = a - r` is not \
         an integer equation and the uniqueness argument below does not apply"
    );

    // 2. UNIQUENESS - the general form of `forged_quotient_is_rejected`.
    assert_eq!(
        solve(
            &z3_bin,
            "uniqueness",
            &format!(
                "(declare-const a Int)(declare-const b Int)\n\
                 (declare-const q1 Int)(declare-const r1 Int)\n\
                 (declare-const q2 Int)(declare-const r2 Int)\n\
                 (assert (and (>= b 0) (< b {bound})))\n\
                 (assert (and (>= q1 0) (< q1 {bound}) (>= r1 0) (< r1 b)))\n\
                 (assert (and (>= q2 0) (< q2 {bound}) (>= r2 0) (< r2 b)))\n\
                 (assert (= (+ (* q1 b) r1) a))\n\
                 (assert (= (+ (* q2 b) r2) a))\n\
                 (assert (or (not (= q1 q2)) (not (= r1 r2))))"
            )
        ),
        "unsat",
        "two different (q, r) pairs satisfy the gadget for one (a, b) - a \
         prover could choose either, so the circuit does not compute division"
    );
}

#[test]
fn the_dividend_bound_is_exactly_the_supremum() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; the dividend bound is unchecked.");
        return;
    };
    let (w, max) = (a_witness_in_range(), max_dividend());

    // Achievable. Move the emitter's constant UP and this goes unsat.
    assert_eq!(
        solve(&z3_bin, "achievable", &format!("{w}\n(assert (= a {max}))")),
        "sat",
        "max_representable_dividend() = {max} is not reachable by any witness, \
         so the emitter refuses a range of dividends it did not need to"
    );

    // Supremum. Move the emitter's constant DOWN and this goes sat.
    assert_eq!(
        solve(&z3_bin, "supremum", &format!("{w}\n(assert (> a {max}))")),
        "unsat",
        "a dividend above max_representable_dividend() = {max} is satisfiable, \
         so `require_dividend_representable` refuses provable programs"
    );

    // ...and the boundary is ONE UNIT wide, not slack. Without this both
    // assertions above are satisfied by a bound that happens to sit in a gap.
    let one_below = max_representable_dividend()
        .sub(&y::zk_field::Fr::from_u64(1))
        .to_decimal_string();
    assert_eq!(
        solve(
            &z3_bin,
            "one-unit-wide",
            &format!("{w}\n(assert (> a {one_below}))")
        ),
        "sat",
        "nothing exceeds {one_below}, so the bound is not tight"
    );
}

#[test]
fn the_fold_guards_accept_exactly_what_the_gadget_can_satisfy() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; the fold/gadget agreement is unchecked.");
        return;
    };
    let (bound, max) = (1u64 << ZK_COMPARISON_BITS, max_dividend());

    // By uniqueness the gadget is satisfiable for (a, b) exactly when
    // `b != 0`, `b < 2^n` and `a div b < 2^n`. `require_quotient_range` tests
    // the last of those on the folded constants. So: can a pair pass the
    // quotient check and still exceed the dividend bound?
    //
    // unsat means the two guards OVERLAP - with both operands constant the
    // quotient check already refuses everything the dividend bound would.
    assert_eq!(
        solve(
            &z3_bin,
            "quotient-check-subsumes",
            &format!(
                "(declare-const a Int)(declare-const b Int)\n\
                 (assert (and (>= b 1) (< b {bound})))\n\
                 (assert (>= a 0))\n\
                 (assert (< (div a b) {bound}))\n\
                 (assert (> a {max}))"
            )
        ),
        "unsat",
        "a folded operand pair passes require_quotient_range and still exceeds \
         the dividend bound, so the two guards disagree about the same program"
    );

    // The converse must FAIL, or `require_quotient_range` is dead weight and
    // the dividend bound alone would do. sat is the answer that says it is
    // load-bearing.
    assert_eq!(
        solve(
            &z3_bin,
            "dividend-bound-is-not-enough",
            &format!(
                "(declare-const a Int)(declare-const b Int)\n\
                 (assert (and (>= b 1) (< b {bound})))\n\
                 (assert (and (>= a 0) (<= a {max})))\n\
                 (assert (>= (div a b) {bound}))"
            )
        ),
        "sat",
        "every dividend within the bound already has an in-range quotient, so \
         require_quotient_range refuses nothing and this test asserts nothing"
    );

    // The dividend bound's OWN live case: the divisor is not a constant, so
    // the quotient cannot be computed at compile time. unsat says a dividend
    // above the bound is unsatisfiable for EVERY divisor, which is exactly the
    // claim that guard makes - and the only case in which it does any work.
    assert_eq!(
        solve(
            &z3_bin,
            "dividend-bound-holds-for-every-divisor",
            &format!("{}\n(assert (> a {max}))", a_witness_in_range())
        ),
        "unsat",
        "some divisor makes an over-bound dividend provable, so refusing it at \
         compile time rejects a program the gadget would have accepted"
    );
}
