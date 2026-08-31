//! Every 32-bit gadget's soundness, decided over its whole bounded domain.
//!
//! `<` `<=` `>` `>=` `/` `%` `&` `|` `^` `<<` `>>` all rest on `emit_num2bits`:
//! `n` booleanity constraints plus one recomposition. That decomposition is
//! simultaneously the range proof, the bit supply the bitwise and shift gadgets
//! read, and the thing that makes an ordering claim meaningful in a field that
//! has no order. Nothing had stated what it guarantees.
//!
//! `tests/zk_integer_ops.rs::forged_quotient_is_rejected` exhibits ONE forgery
//! at ONE input: it forges `q = 7 * inv(2)`, `r = 0` on `7 % 2` and checks the
//! circuit rejects it. That is a sample, not the property. The property is
//! universal - *no* assignment satisfying the constraints has `q` other than
//! `a / b` - and a single counterexample-shaped test cannot say it.
//!
//! `CLAUDE.md` records that Z3 could NOT settle the corresponding question for
//! the comparison gadget: "254-bit modular arithmetic over `Int` defeats the
//! solver". THAT WAS A TIMEOUT, NOT AN UNDECIDABILITY, and the difference is
//! the whole shape of this file. Measured both ways, same question, same
//! solver:
//!
//! | posing                                          | result | time      |
//! |-------------------------------------------------|--------|-----------|
//! | whole gadget in the field, nothing bounded       | unsat  | 560,498ms |
//! | bounded, range checks as separately proved facts | unsat  |      15ms |
//!
//! **37,000x**, and the slow one does terminate. So the fix is not a better
//! solver, it is DECOMPOSITION: prove the bound first, then state the property
//! over the bounded domain. The bounded posing is faithful rather than weaker
//! precisely because the bounding steps are proved here too, and not assumed.
//! `the_field_posing_is_the_same_question_only_slower` is `--ignored` and
//! reproduces the left-hand column.
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
//! Run with:  cargo test --features zk --test zk_gadget_soundness
#![cfg(feature = "zk")]

use std::io::Write;
use std::process::{Command, Stdio};
use y::type_checker::z3_candidates;
use y::lexer::Lexer;
use y::parser::Parser;
use y::zk_emitter::{max_representable_dividend, ZkEmitter, ZK_COMPARISON_BITS};

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
///
/// Every bounded query here answers in milliseconds, so 120s is a generous
/// ceiling rather than a working budget - if one starts approaching it, the
/// posing has drifted back towards the field and the fix is to decompose it.
fn solve(z3_bin: &str, name: &str, body: &str) -> String {
    solve_within(z3_bin, name, body, 120)
}

fn solve_within(z3_bin: &str, name: &str, body: &str, secs: u32) -> String {
    let mut child = Command::new(z3_bin)
        .args([&format!("-T:{secs}"), "-in"])
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

// ---------------------------------------------------------------------------
// `emit_num2bits`, the decomposition every 32-bit gadget rests on.
//
// It emits `n` booleanity constraints (`b * b = b`) plus one recomposition
// (`sum 2^i b_i = value`). Three separate things depend on it and only the
// first is ever stated in the code:
//
//   * it is the RANGE PROOF - nothing above `2^n` has a decomposition;
//   * the decomposition is UNIQUE, which is what stops a prover choosing which
//     bits `&`, `|`, `^`, `<<` and `>>` get to read;
//   * out of range is UNPROVABLE rather than wrong, which is the fail-closed
//     claim every gadget's docstring makes and none of them establishes.
// ---------------------------------------------------------------------------

/// Declarations plus booleanity for `n` bits, and the recomposition sum.
fn decomposition(name: &str, n: u32) -> (String, String) {
    let mut decls = String::new();
    let mut terms = Vec::new();
    for i in 0..n {
        decls.push_str(&format!(
            "(declare-const {name}{i} Int)\n(assert (or (= {name}{i} 0) (= {name}{i} 1)))\n"
        ));
        terms.push(format!("(* {} {name}{i})", 1u64 << i));
    }
    (decls, format!("(+ {})", terms.join(" ")))
}

#[test]
fn the_range_check_really_is_a_range_proof() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; the range proof is unchecked.");
        return;
    };
    let n = ZK_COMPARISON_BITS;
    let (decls, sum) = decomposition("a", n);

    // Stated MODULARLY, because the recomposition constraint is an equation in
    // Fr and not over the integers. A prover picks the bits AND the value; the
    // question is whether any pair satisfies the constraint with the value
    // out of range.
    assert_eq!(
        solve(
            &z3_bin,
            "range-proof",
            &format!(
                "(declare-const X Int)\n{decls}\
                 (assert (and (>= X 0) (< X {BN254_FR})))\n\
                 (assert (= (mod {sum} {BN254_FR}) X))\n\
                 (assert (>= X {}))",
                1u64 << n
            )
        ),
        "unsat",
        "some field element at or above 2^{n} has an {n}-bit decomposition, so \
         `emit_num2bits` is not a range proof and every ordering claim built on \
         it is vacuous"
    );

    // The case the docstrings all name: a negative `I32` is `p - 1`.
    //
    // IN TWO STEPS, AND THIS FILE'S OWN THESIS IS WHY. Asked in one - is there
    // a bit assignment with `(mod (sum 2^i b_i) p) = p - 1` - z3 does not
    // answer in 120 seconds. Split into "the sum is bounded" and "a bounded
    // value is not p - 1", it is 5ms + 4ms. Same question, 13,000x, and the
    // second instance of the measurement in this file's header. A 254-bit
    // modulus over a 32-term symbolic sum is the thing to decompose away.
    assert_eq!(
        solve(
            &z3_bin,
            "recomposition-is-bounded",
            &format!("{decls}(assert (>= {sum} {}))", 1u64 << n)
        ),
        "unsat",
        "the recomposition can produce a value at or above 2^{n}"
    );
    assert_eq!(
        solve(
            &z3_bin,
            "negative-is-unprovable",
            &format!(
                "(declare-const S Int)\n\
                 (assert (and (>= S 0) (< S {})))\n\
                 (assert (= (mod S {BN254_FR}) (- {BN254_FR} 1)))",
                1u64 << n
            )
        ),
        "unsat",
        "`p - 1` is the residue of something below 2^{n}, so a negative operand \
         would be ANSWERED rather than refused - the opposite of the documented \
         fail-closed claim"
    );
}

#[test]
fn a_binary_decomposition_is_unique() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; decomposition uniqueness is unchecked.");
        return;
    };
    let n = ZK_COMPARISON_BITS;
    let (da, sa) = decomposition("a", n);
    let (db, sb) = decomposition("b", n);
    let differs = (0..n)
        .map(|i| format!("(not (= a{i} b{i}))"))
        .collect::<Vec<_>>()
        .join(" ");

    // THIS is the obligation the bitwise and shift gadgets actually need, and
    // it is not the range check. They read the BITS, so if one value admitted
    // two decompositions a prover could pick whichever gives the answer they
    // want - `a & b` would be under-constrained while every range check still
    // passed.
    assert_eq!(
        solve(
            &z3_bin,
            "decomposition-unique",
            &format!("{da}{db}(assert (= {sa} {sb}))\n(assert (or {differs}))")
        ),
        "unsat",
        "one value has two different {n}-bit decompositions, so `&`, `|`, `^`, \
         `<<` and `>>` are under-constrained: a prover chooses the bits"
    );

    // **The control.** Uniqueness comes from BOOLEANITY, not from the shape of
    // the sum - drop `b * b = b` and the same value has many decompositions.
    // Without this the test above could be passing for the wrong reason.
    let unconstrained: String = (0..n)
        .map(|i| format!("(declare-const c{i} Int)\n(declare-const d{i} Int)\n"))
        .collect();
    let sc = format!(
        "(+ {})",
        (0..n).map(|i| format!("(* {} c{i})", 1u64 << i)).collect::<Vec<_>>().join(" ")
    );
    let sd = format!(
        "(+ {})",
        (0..n).map(|i| format!("(* {} d{i})", 1u64 << i)).collect::<Vec<_>>().join(" ")
    );
    assert_eq!(
        solve(
            &z3_bin,
            "uniqueness-needs-booleanity",
            &format!(
                "{unconstrained}(assert (= {sc} {sd}))\n(assert (not (= c0 d0)))"
            )
        ),
        "sat",
        "the recomposition alone already forces uniqueness, so the booleanity \
         constraints are doing nothing and this file is testing the wrong thing"
    );
}

// ---------------------------------------------------------------------------
// The comparison gadget: `a + 2^n - b`, decomposed into n+1 bits, top bit clear
// exactly when `a < b`. Underneath `<`, `<=`, `>`, `>=` AND the `r < b` step of
// `emit_int_div_mod` above.
// ---------------------------------------------------------------------------

#[test]
fn the_comparison_difference_never_wraps() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; the no-wrap step is unchecked.");
        return;
    };
    let n = ZK_COMPARISON_BITS;
    let (b, top) = (1u64 << n, 1u64 << (n + 1));
    // Both operands are range-checked FIRST, which is what makes this true and
    // is why the gadget pays for two decompositions it never reads the bits of.
    assert_eq!(
        solve(
            &z3_bin,
            "no-wrap",
            &format!(
                "(declare-const A Int)(declare-const B Int)\n\
                 (assert (and (>= A 0) (< A {b})))\n\
                 (assert (and (>= B 0) (< B {b})))\n\
                 (assert (or (< (+ A {b} (- B)) 0) (>= (+ A {b} (- B)) {top})))"
            )
        ),
        "unsat",
        "`a + 2^{n} - b` can leave [0, 2^{}), so the field subtraction is not \
         integer subtraction and the top bit means nothing",
        n + 1
    );
}

#[test]
fn the_comparison_gadget_answers_exactly_the_ordering() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; comparison soundness rests on sampling.");
        return;
    };
    let n = ZK_COMPARISON_BITS;
    let b = 1u64 << n;
    let (dd, sd) = decomposition("d", n + 1);
    // The operand range checks are stated as the bounds they were just proved
    // to give, and the difference is integer arithmetic by the no-wrap theorem
    // above. Everything else is the emitted constraint set verbatim.
    let gadget = format!(
        "(declare-const A Int)(declare-const B Int)(declare-const out Int)\n\
         (assert (and (>= A 0) (< A {b})))\n\
         (assert (and (>= B 0) (< B {b})))\n\
         {dd}(assert (= {sd} (+ A {b} (- B))))\n\
         (assert (= out (- 1 d{n})))"
    );

    assert_eq!(
        solve(
            &z3_bin,
            "comparison-soundness",
            &format!("{gadget}\n(assert (not (= out (ite (< A B) 1 0))))")
        ),
        "unsat",
        "some satisfying assignment has `out != (a < b)`. This is THE question \
         `CLAUDE.md` recorded as defeating the solver; if it now returns sat, \
         the gadget is unsound rather than the posing being wrong"
    );

    // **The controls.** An `out` pinned to a constant satisfies the query above
    // vacuously, so both answers must be reachable.
    for want in [0, 1] {
        assert_eq!(
            solve(
                &z3_bin,
                "comparison-reachable",
                &format!("{gadget}\n(assert (= out {want}))")
            ),
            "sat",
            "the gadget can never answer {want}, so the soundness query above \
             holds for a degenerate reason",
            want = want
        );
    }
}

// ---------------------------------------------------------------------------
// The bitwise gadgets: per bit, `and = a*b`, `or = a + b - ab`,
// `xor = a + b - 2ab`, recomposed positionally.
// ---------------------------------------------------------------------------

/// **Note what this tie can and cannot see**, same limit as the shift test
/// below and confirmed by the same mutation. The three formulas are RESTATED
/// from `emit_bitwise`, so this checks that each formula is its operator - not
/// that the emitter uses that formula. Swapping the `|` and `^` scale factors
/// in the emitter passes this file and fails
/// `zk_integer_ops::bitwise_and_or_xor_match_rust`, which is the behavioural
/// tie. What this adds over that one is universality over the operands, and
/// the uniqueness result above, which is the part sampling cannot reach.
#[test]
fn the_bitwise_formulas_are_the_truth_tables() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; the bitwise formulas are unchecked.");
        return;
    };
    // Uniqueness (above) gives the gadget the right bits; the recomposition is
    // positional, so all that remains is that each formula IS its operator.
    // The reference is the truth table written out, not the same expression.
    for (name, formula, table) in [
        ("&", "(* x y)", "(ite (and (= x 1) (= y 1)) 1 0)"),
        ("|", "(- (+ x y) (* x y))", "(ite (or (= x 1) (= y 1)) 1 0)"),
        ("^", "(- (+ x y) (* 2 (* x y)))", "(ite (= x y) 0 1)"),
    ] {
        assert_eq!(
            solve(
                &z3_bin,
                name,
                &format!(
                    "(declare-const x Int)(declare-const y Int)\n\
                     (assert (or (= x 0) (= x 1)))\n\
                     (assert (or (= y 0) (= y 1)))\n\
                     (assert (not (= {formula} {table})))"
                )
            ),
            "unsat",
            "the per-bit formula for `{}` disagrees with its truth table",
            name
        );
    }

    // **The control.** The three formulas must be genuinely different, or a
    // single wrong one would pass by matching a neighbour's table.
    assert_eq!(
        solve(
            &z3_bin,
            "formulas-differ",
            "(declare-const x Int)(declare-const y Int)\n\
             (assert (or (= x 0) (= x 1)))\n\
             (assert (or (= y 0) (= y 1)))\n\
             (assert (not (= (* x y) (- (+ x y) (* x y)))))"
        ),
        "sat",
        "`&` and `|` agree on every bit pair, so the truth tables above do not \
         distinguish the operators"
    );
}

// ---------------------------------------------------------------------------
// The shift gadgets: a pure re-indexing of the decomposition.
// ---------------------------------------------------------------------------

/// **Note what this tie can and cannot see.** The index map below is RESTATED
/// from `emit_shift`, so it checks the map's arithmetic and not that the
/// emitter uses that map. `tests/zk_integer_ops.rs::shifts_match_rust` is what
/// ties it to the running emitter, by sampling; this is what makes the map
/// universal over the operand.
#[test]
fn the_shift_index_map_is_the_shift() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found; the shift index map is unchecked.");
        return;
    };
    let n = ZK_COMPARISON_BITS;
    let (da, sa) = decomposition("a", n);
    let modulus = 1u128 << n;

    // Every amount, including 0 and the `>= n` cases the emitter answers as 0.
    for k in 0..=n + 1 {
        for left in [true, false] {
            let mut terms = Vec::new();
            for i in 0..n {
                // Verbatim from `emit_shift`.
                let src = if left {
                    if i < k { continue; } else { i - k }
                } else {
                    let s = i + k;
                    if s >= n { continue; }
                    s
                };
                terms.push(format!("(* {} a{src})", 1u64 << i));
            }
            let out = if terms.is_empty() {
                "0".to_string()
            } else {
                format!("(+ {})", terms.join(" "))
            };
            let want = if k >= n {
                "0".to_string()
            } else if left {
                format!("(mod (* A {}) {modulus})", 1u128 << k)
            } else {
                format!("(div A {})", 1u128 << k)
            };
            let op = if left { "<<" } else { ">>" };
            assert_eq!(
                solve(
                    &z3_bin,
                    &format!("shift{op}{k}"),
                    &format!("(declare-const A Int)\n{da}(assert (= {sa} A))\n\
                              (assert (not (= {out} {want})))")
                ),
                "unsat",
                "`x {} {}` does not re-index the decomposition into the shift",
                op,
                k
            );
        }
    }
}

/// The measurement that decided this file's shape. `--ignored`: it takes about
/// nine and a half minutes, which is the point.
#[test]
#[ignore]
fn the_field_posing_is_the_same_question_only_slower() {
    let Some(z3_bin) = z3() else {
        println!("SKIP: no z3 found.");
        return;
    };
    let n = ZK_COMPARISON_BITS;
    let (da, sa) = decomposition("a", n);
    let (db, sb) = decomposition("b", n);
    let (dd, sd) = decomposition("d", n + 1);
    // The SAME question with nothing bounded: A and B are free elements of Fr
    // and only the constraints themselves confine them. This is the shape
    // recorded as defeating the solver. It does not - it takes ~560s where the
    // bounded posing takes 15ms.
    let t = std::time::Instant::now();
    let r = solve_within(
        &z3_bin,
        "field-posing",
        &format!(
            "(declare-const A Int)(declare-const B Int)(declare-const out Int)\n\
             (assert (and (>= A 0) (< A {BN254_FR})))\n\
             (assert (and (>= B 0) (< B {BN254_FR})))\n\
             {da}{db}{dd}\
             (assert (= (mod {sa} {BN254_FR}) A))\n\
             (assert (= (mod {sb} {BN254_FR}) B))\n\
             (assert (= (mod {sd} {BN254_FR}) (mod (+ A {} (- B)) {BN254_FR})))\n\
             (assert (= out (- 1 d{n})))\n\
             (assert (not (= out (ite (< A B) 1 0))))",
            1u64 << n
        ),
        1800,
    );
    println!(
        "field posing: {} in {:?} (the bounded posing answers the same question \
         in ~15ms)",
        r,
        t.elapsed()
    );
    assert_eq!(r, "unsat", "the two posings must agree");
}

// ---------------------------------------------------------------------------
// The tie. Everything above says what the gadgets MEAN; nothing above says the
// emitter builds the gadget the model describes.
// ---------------------------------------------------------------------------

/// The model's structure predicts the emitted constraint count exactly.
///
/// `tests/zk_integer_ops.rs::operator_costs_are_what_we_think` pins 101, 99 and
/// 34 as bare numbers with no derivation, so a STRUCTURAL change - decomposing
/// the comparison's difference into `n` bits instead of `n + 1`, dropping an
/// operand's range check, reading the wrong bit - moves the number and reads as
/// a cost change to be re-pinned. Derived here instead, from the same structure
/// the soundness queries above are stated over, so the two cannot drift apart
/// without this failing with the derivation attached.
///
/// `emit_num2bits(x, k)` is `k` booleanity constraints plus one recomposition;
/// each expression binds its result with one more.
///
/// **`/` and `%` are deliberately absent.** Two reduction passes delete
/// constraints from them, so their emitted count is not their structure - which
/// is exactly why their soundness is pinned by the queries above and by
/// `forged_quotient_is_rejected` rather than by a count.
#[test]
fn the_emitted_gadgets_have_the_shape_the_model_describes() {
    let n = ZK_COMPARISON_BITS as usize;
    let num2bits = |k: usize| k + 1;
    let bind = 1;

    for (expr, want, why) in [
        (
            "x < y",
            2 * num2bits(n) + num2bits(n + 1) + bind,
            "two operand range checks, plus an (n+1)-bit decomposition of \
             `a + 2^n - b` whose top bit is the answer",
        ),
        (
            "x & y",
            2 * num2bits(n) + n + bind,
            "two operand range checks, plus one AND product per bit; the \
             recomposition is a weighted sum and costs nothing",
        ),
        (
            "x << 3",
            num2bits(n) + bind,
            "one operand range check; the shift itself is a re-indexing of the \
             bits and is free",
        ),
    ] {
        let src = format!("@unsafe\nfn main(x: I32, y: I32) -> I32 {{ return {expr}; }}\n");
        let tokens = Lexer::new(&src).tokenize();
        let program = Parser::new(tokens).parse_program().expect("parse");
        let mut em = ZkEmitter::new();
        em.emit_program(&program).expect("lower");
        let got = em.build_circuit().constraints.len();
        assert_eq!(
            got, want,
            "`{expr}` emitted {got} constraints; the model says {want} - {why}. \
             Either the emitter no longer builds the gadget this file reasons \
             about, or the model is stale. Do not re-pin the number without \
             deciding which."
        );
    }
}
