//! Two bugs found by `tests/zk_fuzz_differential.rs`, pinned as regressions.
//!
//! Both were silent in the way `CLAUDE.md`'s design rule is about, and neither
//! is in the same subsystem as the other — which is the argument for a
//! *generative* fuzzer over the surface syntax rather than another targeted
//! test. One is in the ZK emitter's constant folding, one is in the parser.
//!
//! Run with:  cargo test --features zk --test zk_gadget_range_and_parsing

#![cfg(feature = "zk")]

use y::zk_emitter::ZkEmitter;
use y::zk_fuzz::{run_circuit, Outcome};

fn compile(src: &str) -> Result<usize, String> {
    let tokens = y::lexer::Lexer::new(src).tokenize();
    let prog = y::parser::Parser::new(tokens).parse_program()?;
    let mut em = ZkEmitter::new();
    em.emit_program(&prog)?;
    Ok(em.build_circuit().constraints.len())
}

// ---------------------------------------------------------------------------
// 1. The 32-bit range check is part of a gadget's meaning, folded or not.
// ---------------------------------------------------------------------------

/// Every gadget that treats its operands as 32-bit integers had a
/// constant-folding fast path beside it that skipped the range check the gadget
/// enforces. The same program therefore meant two different things depending on
/// whether the compiler could see the values.
#[test]
fn an_out_of_range_constant_operand_is_refused_by_every_gadget() {
    // 2^32 exactly: the first value the 32-bit decomposition cannot represent.
    for (name, src) in [
        ("<", "fn main(p: I32) -> I32 { return (p < 4294967296); }"),
        (">", "fn main(p: I32) -> I32 { return (p > 4294967296); }"),
        ("<=", "fn main(p: I32) -> I32 { return (p <= 4294967296); }"),
        (">=", "fn main(p: I32) -> I32 { return (p >= 4294967296); }"),
        // `/` and `%` in the DIVISOR position. The dividend position is a
        // different question and has its own test below - this list had only
        // ever exercised the dividend, so the divisor's range check was
        // covered by nothing.
        ("/", "fn main(p: I32) -> I32 { return (p / 4294967296); }"),
        ("%", "fn main(p: I32) -> I32 { return (p % 4294967296); }"),
        ("&", "fn main(p: I32) -> I32 { return (p & 4294967296); }"),
        ("|", "fn main(p: I32) -> I32 { return (p | 4294967296); }"),
        ("^", "fn main(p: I32) -> I32 { return (p ^ 4294967296); }"),
        ("<<", "fn main(p: I32) -> I32 { return (4294967296 << 1); }"),
        (">>", "fn main(p: I32) -> I32 { return (4294967296 >> 1); }"),
    ] {
        let r = compile(src);
        assert!(
            r.is_err(),
            "`{}` accepted an operand of 2^32, which its gadget cannot range-check: {:?}",
            name,
            r
        );
    }
}

/// `/` and `%` do NOT range-check their dividend, and must not pretend to.
///
/// `emit_int_div_mod` range-checks the quotient, the remainder and the divisor.
/// The dividend is pinned by `q * b = a - r` instead, so it may be as large as
/// `(2^n - 1)^2 + 2^n - 2` and still be provable. The list above used to assert
/// `4294967296 / p` was refused "which its gadget cannot range-check" - but the
/// gadget proves it: at `p = 2` the quotient is `2^31`, comfortably in range.
///
/// So this is not a relaxation of the test, it is a correction of it. The
/// obligation the dividend really carries is the quotient's, and that is
/// asserted here by exhibiting both sides of it.
#[test]
fn the_dividend_is_bounded_by_its_quotient_not_by_the_gadget_width() {
    for (name, src, input, want) in [
        ("/", "fn main(p: I32) -> I32 { return (4294967296 / p); }", 2u64, Some("2147483648")),
        ("%", "fn main(p: I32) -> I32 { return (4294967296 % p); }", 3, Some("1")),
        // p = 1 makes the quotient 2^32, one past its range check. Same
        // dividend, same program, unprovable - which is what says the bound
        // being enforced is the quotient's and not the dividend's.
        ("/", "fn main(p: I32) -> I32 { return (4294967296 / p); }", 1, None),
    ] {
        assert!(
            compile(src).is_ok(),
            "`{}` refused a 2^32 dividend at compile time, but the gadget does \
             not range-check the dividend - only its quotient, which is a \
             runtime property of the divisor",
            name
        );
        let got = match run_circuit(src, &[input]) {
            Outcome::Value(v) => Some(v.to_decimal_string()),
            _ => None,
        };
        assert_eq!(
            got.as_deref(),
            want,
            "`{}` with a 2^32 dividend at p = {}",
            name,
            input
        );
    }
}

/// **The control.** Refusing everything would satisfy the test above. The
/// largest representable operand must still compile.
#[test]
fn the_largest_in_range_operand_still_compiles() {
    for (name, src) in [
        ("<", "fn main(p: I32) -> I32 { return (p < 4294967295); }"),
        ("/", "fn main(p: I32) -> I32 { return (p / 4294967295); }"),
        ("&", "fn main(p: I32) -> I32 { return (p & 4294967295); }"),
        ("<<", "fn main(p: I32) -> I32 { return (4294967295 << 1); }"),
    ] {
        assert!(
            compile(src).is_ok(),
            "`{}` refused 2^32 - 1, which is in range",
            name
        );
    }
}

/// The specific shape the fuzzer's metamorphic oracle caught: an underflow.
///
/// `0 - 1` is `p - 1` in the field, so the fold answered a 254-bit canonical
/// ordering question — `p - 1 >= 0` is **true** — where the gadget refuses the
/// operand outright. This is the same inversion circomlib's `LessThan(32)` has,
/// arrived at by a different route.
#[test]
fn a_folded_underflow_does_not_answer_a_254_bit_ordering_question() {
    let r = compile("fn main(p: I32) -> I32 { return ((0 - 1) >= 0); }");
    assert!(
        r.is_err(),
        "an underflowed constant was compared as a 254-bit field element: {:?}",
        r
    );
}

/// The two paths must agree: a program's meaning cannot depend on whether its
/// inputs arrived as parameters or as literals.
#[test]
fn folded_and_gadget_paths_agree_on_in_range_values() {
    let parameterised = "fn main(a: I32, b: I32) -> I32 { return (a < b); }";
    for (a, b) in [(3u64, 5u64), (5, 3), (5, 5), (0, 4294967295), (4294967295, 0)] {
        let folded = format!("fn main(u: I32) -> I32 {{ return ({} < {}); }}", a, b);
        let x = run_circuit(parameterised, &[a, b]);
        let y = run_circuit(&folded, &[0]);
        assert_eq!(x, y, "`{} < {}` differs between the two paths", a, b);
        assert!(matches!(x, Outcome::Value(_)), "{} < {} refused", a, b);
    }
}

// ---------------------------------------------------------------------------
// 2. `if <ident> { }` is a branch, not a struct literal.
// ---------------------------------------------------------------------------

/// `p0 { }` is a well-formed empty struct literal, so the condition parser
/// swallowed it and the `if` was left with no block. The reported error pointed
/// at the *next* statement, which is why this survived: it reads as a mistake
/// in the line below.
#[test]
fn an_empty_block_after_a_bare_identifier_condition_parses() {
    // Parse only. `while` is asserted here rather than through `compile`
    // because the ZK emitter refuses an unbounded `while` (Z0010) for its own
    // good reasons, which would make this test pass for the wrong one.
    for (name, src) in [
        ("if", "fn main(p0: I32) -> I32 {\n    if p0 {\n    }\n    return 59;\n}\n"),
        (
            "if/else",
            "fn main(p0: I32) -> I32 {\n    if p0 {\n    } else {\n    }\n    return 59;\n}\n",
        ),
        (
            "while",
            "fn main(p0: I32) -> I32 {\n    while p0 {\n    }\n    return 59;\n}\n",
        ),
        (
            "nested if",
            "fn main(p0: I32) -> I32 {\n    if p0 {\n        if p0 {\n        }\n    }\n    return 59;\n}\n",
        ),
    ] {
        let tokens = y::lexer::Lexer::new(src).tokenize();
        assert!(
            y::parser::Parser::new(tokens).parse_program().is_ok(),
            "`{}` with an empty body failed to parse",
            name
        );
    }
}

/// The fix must not cost the construct it was suppressing. A struct literal is
/// unambiguous inside brackets, so it stays available there.
#[test]
fn struct_literals_still_parse_where_they_are_unambiguous() {
    let src = r#"
struct P { x: I32 }
fn take(v: P) -> I32 { return 1; }
fn main(a: I32) -> I32 {
    let q: P = P { x: 3 };
    if take(P { x: 4 }) {
    }
    return 0;
}
"#;
    let tokens = y::lexer::Lexer::new(src).tokenize();
    assert!(
        y::parser::Parser::new(tokens).parse_program().is_ok(),
        "struct literals stopped parsing in a position where they are unambiguous"
    );
}

/// And the branch still behaves like a branch after the parse fix, rather than
/// merely parsing.
#[test]
fn the_repaired_branch_still_computes_the_right_value() {
    let src = "fn main(p0: I32) -> I32 {\n    if p0 {\n    }\n    return 59;\n}\n";
    for v in [0u64, 1] {
        match run_circuit(src, &[v]) {
            Outcome::Value(x) => assert_eq!(x.to_decimal_string(), "59", "p0 = {}", v),
            other => panic!("p0 = {} gave {:?}", v, other),
        }
    }
    // The condition is still a selector, so a non-bit is still unprovable.
    assert_eq!(run_circuit(src, &[2]), Outcome::Unprovable);
}
