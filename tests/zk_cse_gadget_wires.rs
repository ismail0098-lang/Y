//! Common-subexpression elimination must not orphan a gadget's witness recipe.
//!
//! `optimize_circuit` merges two constraints with identical `A` and `B` and a
//! single-intermediate-wire `C`, rewriting the loser's wire id everywhere in the
//! constraint system. It did not rewrite it in `ZkEmitter::witness_recipes` —
//! and those recipes hold `LinearCombination`s captured at emit time
//! (`emit_num2bits` keeps the value it decomposes, `emit_int_div_mod` its
//! dividend and divisor, the is-zero gadget its difference).
//!
//! So any gadget whose operand happened to be a common subexpression evaluated
//! a wire the pass had just deleted, read it as zero, and produced a witness
//! that did not satisfy the circuit. `let a = x * y; let b = x * y; return a ==
//! b;` — two provably equal values — came back unprovable.
//!
//! What makes this worth a dedicated file is how it presented: not a wrong
//! answer, just `satisfied = false`. It reads as "this circuit is unsatisfiable"
//! rather than "the compiler dropped a wire", and it only fires when the operand
//! is duplicated, so almost every hand-written test circuit misses it.
//!
//! Run with:  cargo test --features zk --test zk_cse_gadget_wires

#![cfg(feature = "zk")]

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Fr, ZkEmitter};
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

/// Compiles a circuit body and evaluates it, returning `None` when the circuit
/// cannot be witnessed — which is exactly how this bug presented.
fn eval_body(body: &str, x: u64, y: u64) -> Option<(String, usize)> {
    let src = format!("@unsafe\nfn main(x: I32, y: I32) -> I32 {{\n{}\n}}\n", body);
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);

    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("zk lowering");
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();

    let (witness, satisfied) = solve_r1cs_witness(
        &circuit.constraints,
        &ir,
        circuit.num_variables,
        &[],
        &[Fr::from_u64(x), Fr::from_u64(y)],
    );
    if !satisfied || check_r1cs_satisfiability(&circuit.constraints, &witness).is_err() {
        return None;
    }
    let out = *circuit.outputs.first().expect("output wire");
    Some((witness[out].to_decimal_string(), circuit.constraints.len()))
}

fn assert_eval(body: &str, x: u64, y: u64, expected: u64) {
    match eval_body(body, x, y) {
        None => panic!(
            "circuit was UNPROVABLE with x={}, y={} (expected {}):\n{}",
            x, y, expected, body
        ),
        Some((got, _)) => assert_eq!(
            got,
            expected.to_string(),
            "wrong value for x={}, y={}:\n{}",
            x,
            y,
            body
        ),
    }
}

/// The whole family. Every gadget carries a linear combination in its recipe, so
/// every gadget was affected; testing one would have left the others to rot.
#[test]
fn duplicated_operands_stay_provable_through_every_gadget() {
    // x = 3, y = 5, so a and b are both 15.
    let cases: &[(&str, u64)] = &[
        ("    let a = x * y;\n    let b = x * y;\n    return a < b;", 0),
        ("    let a = x * y;\n    let b = x * y;\n    return a <= b;", 1),
        ("    let a = x * y;\n    let b = x * y;\n    return a > b;", 0),
        ("    let a = x * y;\n    let b = x * y;\n    return a >= b;", 1),
        ("    let a = x * y;\n    let b = x * y;\n    return a == b;", 1),
        ("    let a = x * y;\n    let b = x * y;\n    return a != b;", 0),
        ("    let a = x * y;\n    let b = x * y;\n    return a & b;", 15),
        ("    let a = x * y;\n    let b = x * y;\n    return a | b;", 15),
        ("    let a = x * y;\n    let b = x * y;\n    return a ^ b;", 0),
        ("    let a = x * y;\n    let b = x * y;\n    return a / (b + 1);", 0),
        ("    let a = x * y;\n    let b = x * y;\n    return a % (b + 1);", 15),
        // Commuted, because the pass's bucket hash is deliberately symmetric in
        // (A, B): `y * x` merges with `x * y` and must remap the same way.
        ("    let a = x * y;\n    let b = y * x;\n    return a == b;", 1),
    ];
    for (body, expected) in cases {
        assert_eval(body, 3, 5, *expected);
    }
}

/// The control, and it is not optional.
///
/// A "fix" that simply stopped eliminating anything would make every test above
/// pass while silently disabling the optimisation. These bodies use two DISTINCT
/// products, so no merge is possible; the constraint counts below prove the
/// duplicated versions really are one constraint smaller, i.e. that CSE still
/// fires on exactly the cases the fix is about.
#[test]
fn distinct_operands_are_unaffected_and_cse_still_fires() {
    let pairs: &[(&str, &str, u64, u64)] = &[
        (
            "    let a = x * y;\n    let b = x * y;\n    return a == b;",
            "    let a = x * y;\n    let b = y * y;\n    return a == b;",
            1,
            0,
        ),
        (
            "    let a = x * y;\n    let b = x * y;\n    return a & b;",
            "    let a = x * y;\n    let b = y * y;\n    return a & b;",
            15,
            9, // 15 & 25
        ),
    ];
    for (dup, distinct, want_dup, want_distinct) in pairs {
        let (got_dup, n_dup) = eval_body(dup, 3, 5).expect("duplicated form must be provable");
        let (got_distinct, n_distinct) =
            eval_body(distinct, 3, 5).expect("distinct form must be provable");

        assert_eq!(got_dup, want_dup.to_string(), "{}", dup);
        assert_eq!(got_distinct, want_distinct.to_string(), "{}", distinct);
        assert_eq!(
            n_dup + 1,
            n_distinct,
            "CSE no longer eliminates the duplicated product - the pass has been \
             disabled rather than fixed ({} vs {} constraints)",
            n_dup,
            n_distinct
        );
    }
}

/// Poseidon hashes the same pair twice.
///
/// A hash is 241 constraints of shared structure, so this is the case where the
/// pass has the most to merge and the most recipes to keep in step with it. The
/// two digests must be equal *and* the circuit must still be witnessable.
#[test]
fn repeated_poseidon_still_witnesses() {
    let body = "    let a = poseidon_hash(x, y);\n    let b = poseidon_hash(x, y);\n    return a == b;";
    assert_eval(body, 7, 11, 1);

    let single = "    let a = poseidon_hash(x, y);\n    return a;";
    let (_, n_single) = eval_body(single, 7, 11).expect("single hash must be provable");
    let (_, n_double) = eval_body(body, 7, 11).expect("double hash must be provable");
    assert!(
        n_double < 2 * n_single,
        "the second identical Poseidon was not deduplicated at all ({} vs 2 x {})",
        n_double,
        n_single
    );
}
