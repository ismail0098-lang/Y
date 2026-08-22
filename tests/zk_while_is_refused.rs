// ============================================================
//  `while` is refused in ZK circuit mode, INCLUDING the bounded
//  `@max_iterations(N)` form the compiler used to recommend.
//
//  That form unrolled `N` times behind an "active mask"
//  (`active_{i+1} = active_i * cond_{i+1}`, every mutated variable
//  muxed on the mask). It did not work, and it failed three different
//  ways depending on the bound. Measured on
//
//      while i < p0 { acc = acc + 3; i = i + 1; }
//
//  solved for p0 = 0, 1, 2, 3:
//
//      @max_iterations(1)   3, 3, 3, 3     want 0, 3, 3, 3
//      @max_iterations(2)   0, 3, 6, 6     correct
//      @max_iterations(4)   0, UNSAT, UNSAT, UNSAT
//
//  The first row is the one that matters: the body ran although the
//  condition was false on entry, the circuit is SATISFIABLE, and
//  Groth16 proves that arithmetic as readily as the right kind. The
//  third row is merely unusable. **That the middle row is correct is
//  how it survived** -- any single probe at N=2 agrees with the source,
//  and N=2 is what anyone writes first.
//
//  It was found by building `tests/zk_llvm_differential.rs`: while
//  checking whether `while` could join that differential's program
//  space, the two backends were asked the same question and disagreed.
//
//  This file pins the refusal and its CONTROL -- `for` is fully
//  unrolled, correct, and must keep working, or "refuse everything"
//  would pass.
// ============================================================
#![cfg(feature = "zk")]

use y::zk_emitter::ZkEmitter;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

/// The error, or `None` if the program compiled. (`ZkEmitter` is not `Debug`,
/// so `expect_err` is unavailable.)
fn emit_err(src: &str) -> Option<String> {
    emit(src).err()
}

fn emit(src: &str) -> Result<ZkEmitter, String> {
    let tokens = y::lexer::Lexer::new(src).tokenize();
    let prog = y::parser::Parser::new(tokens)
        .parse_program()
        .map_err(|e| format!("parse: {e}"))?;
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&prog)?;
    Ok(emitter)
}

fn solve(src: &str, inputs: &[u64]) -> Option<String> {
    let emitter = emit(src).ok()?;
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (w, satisfied) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    if !satisfied {
        return None;
    }
    Some(w[*circuit.outputs.first()?].to_decimal_string())
}

const BOUNDED: &str = "\
fn main(p0: I32) -> I32 {
    let i: I32 = 0;
    let acc: I32 = 0;
    @max_iterations(2)
    while i < p0 {
        acc = acc + 3;
        i = i + 1;
    }
    return acc;
}
";

const UNBOUNDED: &str = "\
fn main(p0: I32) -> I32 {
    let i: I32 = 0;
    let acc: I32 = 0;
    while i < p0 {
        acc = acc + 3;
        i = i + 1;
    }
    return acc;
}
";

#[test]
fn a_bounded_while_is_refused_and_says_why() {
    let err = emit_err(BOUNDED).expect("`@max_iterations` while was accepted");
    assert!(
        err.contains("'while' is not supported"),
        "refusal does not name the construct: {err}"
    );
    assert!(
        err.contains("max_iterations"),
        "refusal does not mention the annotation it withdrew, so a user who \
         followed the old hint has nothing to go on: {err}"
    );
    assert!(
        err.contains("for"),
        "refusal does not point at the construct that does work: {err}"
    );
}

#[test]
fn an_unbounded_while_is_refused_too() {
    let err = emit_err(UNBOUNDED).expect("bare `while` was accepted");
    assert!(err.contains("'while' is not supported"), "{err}");
}

#[test]
fn a_for_loop_is_still_the_supported_bounded_loop() {
    // The control. Refusing every loop is sound and useless -- the same shape
    // as `ordinary_loop_bodies_still_verify` guarding the SMT encoding. `for`
    // is fully unrolled, and `tests/zk_llvm_differential.rs` checks it against
    // the LLVM backend on generated programs.
    let src = "\
fn main(p0: I32) -> I32 {
    let acc: I32 = 0;
    for i in 0..3 {
        acc = acc + p0;
    }
    return acc;
}
";
    assert_eq!(
        solve(src, &[7]).as_deref(),
        Some("21"),
        "a `for` loop over a witness value stopped working"
    );
}

#[test]
fn a_for_loop_with_a_conditional_return_still_works() {
    // The other half of the control: the `for` path this refusal pushes users
    // towards must handle the case `while` was reached for, which is a loop
    // that stops early.
    let src = "\
fn main(p0: I32) -> I32 {
    for i in 0..4 {
        if p0 < i {
            return i * 10;
        }
    }
    return 99;
}
";
    assert_eq!(solve(src, &[1]).as_deref(), Some("20"), "p0=1 stops at i=2");
    assert_eq!(solve(src, &[9]).as_deref(), Some("99"), "p0=9 never stops");
}
