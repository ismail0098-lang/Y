//! `return` and `if` in the ZK backend, checked against what the program means.
//!
//! Found with Z3, by parsing an emitted `.r1cs.txt` and asking whether any
//! satisfying assignment disagreed with the source. Two bugs came out of it,
//! both of the kind `CLAUDE.md`'s design rule is written about: the circuit
//! compiled cleanly, the emitter printed "Compilation Successful!", and the
//! artifact computed a different function than the program.
//!
//! 1. **`emit_block` recorded the LAST return instead of stopping at the
//!    first.** So `return 1; return 2;` emitted a circuit computing 2, and
//!    `if c { return x; } return y;` — the ordinary way to write a conditional
//!    — emitted one computing `y` UNCONDITIONALLY, after building and then
//!    discarding the ~100 constraints of the comparison. Z3 proved the
//!    resulting system had no satisfying assignment with a non-zero output:
//!    the circuit was the constant 0, and Groth16 would prove that about it
//!    perfectly happily. The LLVM backend compiles the same program to
//!    `ret i32 1`, so the two backends disagreed on what `return` means.
//!
//! 2. **A dynamic `if` multiplexed with `cond * (then - else) = out - else`
//!    and never constrained `cond` to be a bit.** That is a selector only for
//!    `cond` in {0, 1}; for anything else it linearly interpolates. `if a
//!    { return 5; } return 9;` emitted `9 - 4a`, so `a = 2` produced 1 — a
//!    value neither branch can return — and `a = 3` produced `-3`. A
//!    comparison already yields a booleanity-constrained bit so nothing real
//!    was affected, but the guarantee was absent rather than merely unused.
//!
//! The tests below state the *semantics*, not the constraint shapes, so they
//! survive a re-implementation of either fix.
//!
//! Run with:  cargo test --features zk --test zk_control_flow

#![cfg(feature = "zk")]

use y::zk_emitter::ZkEmitter;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

/// Compile a Y program to R1CS and solve it for `inputs`.
///
/// Returns `None` when the system has no satisfying assignment, which is a
/// meaningful outcome here rather than a failure: a refused `if` condition is
/// meant to be unprovable.
fn eval(source: &str, inputs: &[u64]) -> Option<String> {
    let tokens = y::lexer::Lexer::new(source).tokenize();
    let prog = y::parser::Parser::new(tokens)
        .parse_program()
        .expect("program did not parse");
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&prog).expect("program did not compile to R1CS");

    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (witness, satisfied) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    if !satisfied {
        return None;
    }
    let out = *circuit.outputs.first().expect("circuit has no output");
    Some(witness[out].to_decimal_string())
}

fn expect(source: &str, inputs: &[u64], want: u64, what: &str) {
    let got = eval(source, inputs);
    assert_eq!(
        got.as_deref(),
        Some(want.to_string().as_str()),
        "{}: inputs {:?} should give {}",
        what,
        inputs,
        want
    );
}

const IF_THEN_FALL: &str = r#"
fn main(a: I32, b: I32) -> I32 {
    if a < b { return 1; }
    return 0;
}
"#;

/// The headline case, and the one that was a constant.
#[test]
fn a_conditional_return_actually_depends_on_the_condition() {
    for (a, b) in [(3u64, 5u64), (0, 1), (5, 5), (5, 3), (9, 2)] {
        expect(IF_THEN_FALL, &[a, b], (a < b) as u64, "if/return/fall-through");
    }
}

/// **The control.** If the output were pinned to a constant — which is exactly
/// what the bug did — the test above would still pass on every case where the
/// answer happens to be that constant. This asserts the output actually takes
/// both values, so "always 0" and "always 1" both fail.
#[test]
fn the_conditional_return_is_not_secretly_constant() {
    let lo = eval(IF_THEN_FALL, &[3, 5]).expect("no witness");
    let hi = eval(IF_THEN_FALL, &[5, 3]).expect("no witness");
    assert_ne!(lo, hi, "the circuit's output does not depend on its inputs");
}

/// The fall-through value must be the value of what actually follows the `if`,
/// not zero. A fix that merely stopped `emit_block` at the first return would
/// pass every other test in this file and fail this one, because the old
/// `Stmt::If` arm substituted zero for the branch that does not return.
#[test]
fn the_fall_through_value_is_the_rest_of_the_block() {
    let src = r#"
fn main(a: I32, b: I32) -> I32 {
    if a < b { return 1; }
    return 7;
}
"#;
    expect(src, &[3, 5], 1, "taken branch");
    expect(src, &[5, 3], 7, "fall-through branch");
}

/// The case the Coq proof found, which survived the first fix.
///
/// `if c { if d { return 1; } } return 7;` with `c` true and `d` false: the
/// inner `if` falls through, so does the outer one, and the program returns 7.
/// The tail-multiplexing lowering answered 0, because the inner `if` reported
/// the zero its own empty tail supplied instead of telling the outer block
/// control had fallen past it. Writing the semantics down in
/// `proofs/ZkControlFlow.v` is what surfaced it; `low_tail_is_wrong_when_nested`
/// there is this program.
#[test]
fn a_nested_conditional_return_falls_through_correctly() {
    let src = r#"
fn main(a: I32, b: I32) -> I32 {
    if a < b {
        if b < a { return 1; }
    }
    return 7;
}
"#;
    // `a < b` and `b < a` cannot both hold, so the inner return is dead and
    // every input must reach `return 7`.
    for (a, b) in [(3u64, 5u64), (5, 3), (4, 4)] {
        expect(src, &[a, b], 7, "nested fall-through");
    }
}

/// Both branches returning has no fall-through and was always handled; it is
/// here so a future rewrite cannot fix one shape by breaking the other.
#[test]
fn if_else_with_returns_in_both_branches_still_works() {
    let src = r#"
fn main(a: I32, b: I32) -> I32 {
    if a < b { return 1; } else { return 0; }
}
"#;
    expect(src, &[3, 5], 1, "then");
    expect(src, &[5, 3], 0, "else");
}

/// Statements after a `return` are unreachable, and the LLVM backend agrees:
/// it emits `ret i32 1` for this program.
#[test]
fn code_after_a_return_is_unreachable() {
    let src = r#"
fn main(a: I32) -> I32 {
    return 1;
    return 2;
}
"#;
    expect(src, &[0], 1, "first return wins");
}

/// A `return` inside a `for` body is a conditional return too.
///
/// The loop is fully unrolled, so its iterations are a statement sequence and
/// the same predicated fold applies. The loop arm used to discard the body's
/// result entirely, so `for i in 0..4 { if c { return 1; } } return 9;`
/// emitted the constant 9 - the same bug as the flat case, in a different arm.
///
/// The `@invariant` is required by the type checker for any loop, and is
/// discharged by z3; it is not part of what this test is about.
#[test]
fn a_return_inside_a_loop_is_not_discarded() {
    let src = r#"
fn main(a: I32) -> I32 {
    @invariant(i >= 0)
    for i in 0..4 {
        if a < 2 { return 1; }
    }
    return 9;
}
"#;
    expect(src, &[0], 1, "loop returns on the first iteration");
    expect(src, &[1], 1, "loop returns on the first iteration");
    expect(src, &[2], 9, "loop never returns");
    expect(src, &[5], 9, "loop never returns");
}

/// A dynamic `if` condition is a SELECTOR, so it must be a bit.
///
/// The merge is `cond * (then - else) = out - else`. Nothing constrained
/// `cond`, so `if a { return 5; } return 9;` emitted `9 - 4a` and quietly
/// answered 1 for `a = 2`. Making the circuit unsatisfiable is the same
/// fail-closed choice the comparison gadget's range checks make: no proof can
/// be produced, rather than a proof of something the author did not write.
#[test]
fn a_non_boolean_if_condition_is_unprovable_not_wrong() {
    let src = r#"
fn main(a: I32) -> I32 {
    if a { return 5; }
    return 9;
}
"#;
    expect(src, &[0], 9, "false");
    expect(src, &[1], 5, "true");
    for a in [2u64, 3, 100] {
        assert_eq!(
            eval(src, &[a]),
            None,
            "a = {} is not a boolean, so the circuit must admit no witness",
            a
        );
    }
}
