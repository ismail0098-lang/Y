//! Integer and bitwise operators in ZK circuits, checked against Rust's own.
//!
//! These gadgets are the kind that fail silently. A field has no bits and no
//! order, so `&`, `%`, `>>` and friends are not primitive operations here -
//! each is a bit decomposition plus a recomposition, and a decomposition that
//! is off by one position, or a quotient that is not range-checked, still
//! produces a perfectly valid proof of the wrong number. So every case below
//! compares against the same expression evaluated natively.
//!
//! Note the width: `ZK_COMPARISON_BITS = 32`, and the decompositions treat
//! values as UNSIGNED 32-bit. A negative `I32` is `p - |v|` in the field, which
//! fails its range check and is therefore unprovable rather than wrong. Same
//! convention the comparison gadget already used.
//!
//! Run with:  cargo test --features zk --test zk_integer_ops
#![cfg(feature = "zk")]

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{BigUint, Fr, ZkEmitter};
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

/// Compiles `fn main(x, y) -> I32 { return <expr>; }` and evaluates it.
///
/// Returns `None` if the circuit is unsatisfiable for these inputs, which is
/// how the fail-closed cases (divide by zero, out-of-range operand) present.
fn eval(expr: &str, x: u64, y: u64) -> Option<String> {
    let src = format!(
        "@unsafe\nfn main(x: I32, y: I32) -> I32 {{\n    return {};\n}}\n",
        expr
    );
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
        &[Fr(BigUint::from_u64(x)), Fr(BigUint::from_u64(y))],
    );
    if !satisfied || check_r1cs_satisfiability(&circuit.constraints, &witness).is_err() {
        return None;
    }
    let out = *circuit.outputs.first().expect("output wire");
    Some(witness[out].0.to_decimal_string())
}

fn assert_eval(expr: &str, x: u64, y: u64, expected: u64) {
    let got = eval(expr, x, y)
        .unwrap_or_else(|| panic!("`{}` with x={}, y={} was unsatisfiable", expr, x, y));
    assert_eq!(
        got,
        expected.to_string(),
        "`{}` with x={}, y={}: circuit says {}, Rust says {}",
        expr,
        x,
        y,
        got,
        expected
    );
}

const PAIRS: [(u64, u64); 10] = [
    (0, 1),
    (1, 1),
    (7, 2),
    (2, 7),
    (255, 16),
    (1024, 33),
    (65535, 4096),
    (123456, 789),
    (4294967295, 65535),
    (1000000, 1000000),
];

#[test]
fn bitwise_and_or_xor_match_rust() {
    for (x, y) in PAIRS {
        assert_eval("x & y", x, y, x & y);
        assert_eval("x | y", x, y, x | y);
        assert_eval("x ^ y", x, y, x ^ y);
    }
}

#[test]
fn integer_div_and_mod_match_rust() {
    for (x, y) in PAIRS {
        if y == 0 {
            continue;
        }
        assert_eval("x / y", x, y, x / y);
        assert_eval("x % y", x, y, x % y);
    }
}

/// `/` and `%` must be consistent with each other.
///
/// `/` used to be FIELD division - `x * y^-1 mod p` - which agrees with integer
/// division only when `y` divides `x` exactly. This identity is the cheapest
/// way to catch a regression back to that.
#[test]
fn div_and_mod_reconstruct_the_dividend() {
    for (x, y) in PAIRS {
        if y == 0 {
            continue;
        }
        assert_eval("(x / y) * y + x % y", x, y, x);
    }
}

#[test]
fn shifts_match_rust() {
    for (x, _) in PAIRS {
        for k in [0u64, 1, 3, 8, 16, 31] {
            let mask = u32::MAX as u64;
            assert_eval(&format!("x << {}", k), x, 0, ((x << k) & mask) as u64);
            assert_eval(&format!("x >> {}", k), x, 0, (x & mask) >> k);
        }
    }
}

/// A shift past the width clears the value, and still range-checks the operand.
#[test]
fn oversized_shift_is_zero_not_wrapped() {
    assert_eval("x << 32", 12345, 0, 0);
    assert_eval("x >> 32", 12345, 0, 0);
}

#[test]
fn logical_and_or_match_rust() {
    // Operands must be boolean, so build them from comparisons.
    for (x, y) in PAIRS {
        let l = (x < y) as u64;
        let r = (y < 1000) as u64;
        assert_eval("(x < y) && (y < 1000)", x, y, l & r);
        assert_eval("(x < y) || (y < 1000)", x, y, l | r);
    }
}

/// `&&` on non-boolean operands must be unprovable, not silently `x*y`.
#[test]
fn logical_ops_reject_non_boolean_operands() {
    assert!(
        eval("x && y", 5, 3).is_none(),
        "`5 && 3` must fail its booleanity constraint rather than produce a value"
    );
}

/// Division by zero has no satisfying witness.
#[test]
fn division_by_zero_is_unprovable() {
    assert!(eval("x / y", 7, 0).is_none(), "7 / 0 must not be provable");
    assert!(eval("x % y", 7, 0).is_none(), "7 % 0 must not be provable");
}

/// What each operator costs, pinned.
///
/// These are not incidental numbers - they are the reason to think before
/// putting a `%` inside a loop. Everything that needs bits pays for a
/// decomposition, and a decomposition is one constraint per bit plus one:
///
///   `&`, `|`, `^`   two operand decompositions + one AND per bit  -> 3n+2
///   `<<`, `>>`      one decomposition, recomposition is free      -> n+1
///   `<`             two operands + the (n+1)-bit difference       -> 3n+5
///   `/`, `%`        a quotient range check + a `<` + the product  -> 4n+8
///
/// A rise here means a gadget started allocating wires for values that could
/// have stayed linear combinations. Add one for `main`'s output binding.
#[test]
fn operator_costs_are_what_we_think() {
    let cases: [(&str, usize); 12] = [
        ("x + y", 1),
        ("x * y", 2),
        ("x == y", 3),
        ("x < y", 101),
        ("x & y", 99),
        ("x | y", 99),
        ("x ^ y", 99),
        ("x << 3", 34),
        ("x >> 3", 34),
        ("x / y", 136),
        ("x % y", 136),
        ("(x < y) && (y < 9)", 204),
    ];
    for (expr, expected) in cases {
        let src = format!(
            "@unsafe\nfn main(x: I32, y: I32) -> I32 {{ return {}; }}\n",
            expr
        );
        let tokens = Lexer::new(&src).tokenize();
        let program = Parser::new(tokens).parse_program().expect("parse");
        TypeChecker::new().check_program(&program);
        let mut emitter = ZkEmitter::new();
        emitter.emit_program(&program).expect("lower");
        let got = emitter.build_circuit().constraints.len();
        assert_eq!(got, expected, "`{}` cost {} constraints, expected {}", expr, got, expected);
    }
}

/// The soundness hole that makes integer division delicate.
///
/// The natural encoding `q * b = a - r` with `r < b` is satisfiable with a
/// FORGED quotient: for `7 / 2`, a prover picks `r = 0` and
/// `q = 7 * 2^-1 mod p`, an enormous field element. Both constraints hold and
/// the circuit has proved `7 % 2 == 0`. The range check on `q` is what closes
/// it, and this test confirms the forged assignment is rejected.
#[test]
fn forged_quotient_is_rejected() {
    let src = "@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    return x % y;\n}\n";
    let tokens = Lexer::new(src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("lower");
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();

    let (mut witness, ok) = solve_r1cs_witness(
        &circuit.constraints,
        &ir,
        circuit.num_variables,
        &[],
        &[Fr(BigUint::from_u64(7)), Fr(BigUint::from_u64(2))],
    );
    assert!(ok, "honest witness should solve");

    // Find the quotient and remainder wires by name and forge them.
    let q = circuit
        .variables
        .iter()
        .position(|n| n.starts_with("intdiv_q"))
        .expect("quotient wire");
    let r = circuit
        .variables
        .iter()
        .position(|n| n.starts_with("intdiv_r"))
        .expect("remainder wire");
    assert_eq!(witness[q].0.to_decimal_string(), "3", "honest quotient");
    assert_eq!(witness[r].0.to_decimal_string(), "1", "honest remainder");

    // r = 0, q = 7 * inv(2): satisfies `q*b = a - r` over the field.
    witness[r] = Fr::zero();
    witness[q] = Fr::from_u64(7).mul(&Fr::from_u64(2).inv());

    assert!(
        check_r1cs_satisfiability(&circuit.constraints, &witness).is_err(),
        "a forged quotient must violate its range check - otherwise the circuit \
         proves 7 % 2 == 0"
    );
}
