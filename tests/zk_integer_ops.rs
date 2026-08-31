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
use y::zk_emitter::{Fr, ZkEmitter};
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
        &[Fr::from_u64(x), Fr::from_u64(y)],
    );
    if !satisfied || check_r1cs_satisfiability(&circuit.constraints, &witness).is_err() {
        return None;
    }
    let out = *circuit.outputs.first().expect("output wire");
    Some(witness[out].to_decimal_string())
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
        ("x * y", 1),
        ("x == y", 2),
        ("x < y", 101),
        ("x & y", 99),
        ("x | y", 99),
        ("x ^ y", 99),
        ("x << 3", 34),
        ("x >> 3", 34),
        // 136 -> 134 when the pass learned to pin a wire from a constraint
        // whose `C` is a constant (`k * lin = c0`), then -> 133 with the
        // pure-copy rename that collapses `<intermediate> * 1 = out`. The same
        // rename is why `x * y` and `x == y` each lost one.
        //
        // The gadget's SOUNDNESS is pinned separately by
        // `forged_quotient_is_rejected`, which is what makes a drop here safe
        // to accept rather than a reason to investigate.
        ("x / y", 133),
        ("x % y", 133),
        ("(x < y) && (y < 9)", 203),
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
        &[Fr::from_u64(7), Fr::from_u64(2)],
    );
    assert!(ok, "honest witness should solve");

    // Find the quotient and remainder wires and forge them.
    //
    // `x % y` returns the remainder, so the pure-copy rename in
    // `substitute_linear_constraints` collapses `intdiv_r * 1 = out` and the
    // remainder IS the output wire - it no longer exists under its own name.
    // Falling back to the output is not a weakening of the test: the wire being
    // forged is the same one, and the property under test is that the range
    // check on `q` rejects the forgery either way.
    let by_name = |prefix: &str| circuit.variables.iter().position(|n| n.starts_with(prefix));
    let q = by_name("intdiv_q").expect("quotient wire");
    let r = by_name("intdiv_r").unwrap_or_else(|| circuit.outputs[0]);
    assert_eq!(witness[q].to_decimal_string(), "3", "honest quotient");
    assert_eq!(witness[r].to_decimal_string(), "1", "honest remainder");

    // r = 0, q = 7 * inv(2): satisfies `q*b = a - r` over the field.
    witness[r] = Fr::zero();
    witness[q] = Fr::from_u64(7).mul(&Fr::from_u64(2).inv());

    assert!(
        check_r1cs_satisfiability(&circuit.constraints, &witness).is_err(),
        "a forged quotient must violate its range check - otherwise the circuit \
         proves 7 % 2 == 0"
    );
}

// ---------------------------------------------------------------------------
// The fold path for `/` and `%` must accept exactly what the gadget accepts.
//
// `emit_int_div_mod` range-checks the QUOTIENT, the remainder and the DIVISOR.
// It does NOT range-check the dividend, which `q * b = a - r` pins instead - so
// a dividend up to `(2^n - 1)^2 + 2^n - 2` is provable. The fold used to apply
// `require_gadget_range2`, refusing any constant operand above `2^n` including
// the dividend, with the message "no witness could satisfy it". One does.
//
// Found by the generative fuzzer's metamorphic oracle within seconds of the
// generator learning to write a `/` at all, and independently by hand before
// that. Fail-closed, so never an unsound proof - but the same program meant two
// different things depending on whether the compiler could see the value, which
// is the property that oracle exists to check.
// ---------------------------------------------------------------------------

/// Compiles `fn main() -> I32 { return <expr>; }` with no parameters, so every
/// operand is a literal and the emitter takes its constant-folding path.
fn compile_folded(expr: &str) -> Result<(), String> {
    let src = format!("@unsafe\nfn main() -> I32 {{\n    return {};\n}}\n", expr);
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    ZkEmitter::new().emit_program(&program).map(|_| ())
}

#[test]
fn a_dividend_above_the_gadget_width_still_folds() {
    // 2^33 / 4 = 2^31, an in-range quotient. The gadget proves it - verified
    // directly by `eval`, which drives the parameterised path - so the fold
    // must not refuse it.
    assert_eq!(
        eval("x / y", 8589934592, 4).as_deref(),
        Some("2147483648"),
        "the gadget path must prove a 2^33 dividend with an in-range quotient"
    );
    assert!(
        compile_folded("8589934592 / 4").is_ok(),
        "the folded form of a program the gadget proves must not be refused"
    );
    assert!(
        compile_folded("8589934592 % 4").is_ok(),
        "`%` shares the gadget and must agree with `/`"
    );
}

#[test]
fn an_unrepresentable_quotient_is_refused_when_folded() {
    // 2^34 / 4 = 2^32, one past the quotient's range check, so the gadget is
    // unsatisfiable. Dropping `require_quotient_range` makes the fold answer
    // 4294967296 instead - a value no witness of this gadget can hold.
    //
    // The fuzzer CANNOT catch this: its folding oracle deliberately permits a
    // folded form to be provable where the parameterised form is not, because
    // branch pruning legitimately does exactly that. So it needs a pin here.
    assert_eq!(
        eval("x / y", 17179869184, 4),
        None,
        "a quotient of 2^32 fails its range check, so the gadget is unsatisfiable"
    );
    let err = compile_folded("17179869184 / 4").expect_err("must be refused");
    assert!(
        err.contains("quotient"),
        "the refusal must name the quotient, which is what is out of range - \
         not the dividend, which is not. Got: {}",
        err
    );
    // One unit below, the same expression is fine. Without this the test is
    // satisfied by refusing every large dividend, which is the original bug.
    assert!(
        compile_folded("17179869180 / 4").is_ok(),
        "4 * (2^32 - 1) has an in-range quotient and must still fold"
    );
}

#[test]
fn a_negative_dividend_is_still_refused_by_name() {
    // `-1` is `p - 1`, far above any representable dividend, so no divisor
    // makes it provable. That is the documented behaviour of every 32-bit
    // gadget here and it must survive the relaxation above - which it does for
    // a different reason than before: not a range check on the dividend, but
    // the fact that `(p - 1) / b` overflows the quotient's.
    let err = compile_folded("(0 - 1) / 4").expect_err("must be refused");
    assert!(
        err.contains("dividend") || err.contains("quotient"),
        "a negative dividend must be refused with a reason. Got: {}",
        err
    );
    // And with a VARIABLE divisor, where the quotient cannot be computed at
    // compile time. This is the only case in which the dividend bound does any
    // work - Z3 shows the quotient check subsumes it whenever both operands are
    // constant (`tests/zk_divmod_soundness.rs`).
    let src = "@unsafe\nfn main(y: I32) -> I32 {\n    return (0 - 1) / y;\n}\n";
    let tokens = Lexer::new(src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    let err = ZkEmitter::new()
        .emit_program(&program)
        .expect_err("a constant dividend above the bound is unprovable for every divisor");
    assert!(
        err.contains("dividend"),
        "with a variable divisor only the dividend bound can fire. Got: {}",
        err
    );
}
