//! Unsigned integers in `@invariant` proofs.
//!
//! Every unsigned variable under an invariant was refused ("does not have a
//! supported signed integer type"), so every loop with a `U32` bound was
//! too, because the backends had not agreed on unsigned operators. They
//! agree now: `llvm_unsigned_ops` and `llvm_integer_widths` pin the LLVM
//! backend (and the JIT, which compiles its IR), `ptx_integer_datapath` runs
//! every integer operator against a CPU reference on the GPU, and
//! `for_header_semantics` pins how each backend reads a `U32` loop bound.
//!
//! The model is the signed one's: exact values, every intermediate proved to
//! fit its type, here `0..=2^n-1`, so no execution that wraps is ever
//! described. What differs is what the backends could read differently,
//! and each case below is one of those, with the reading the machine uses:
//!
//! * a negative literal beside an unsigned value - the machine compares
//!   against its bits, 0xFFFFFFFF;
//! * a signed and an unsigned value together - a `U32` 4294967295 and an
//!   `I32` -1 have the same bits, so `x != d` is FALSE on the machine;
//! * a `U32` loop bound of 2^31 or more - every backend compares its bits as
//!   an I32, so the loop does not run.
use std::process::Command;
use std::sync::OnceLock;
use y::type_checker::{z3_candidates, TypeChecker};

#[path = "common/verification.rs"]
mod verification;

fn solver_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let found = z3_candidates().iter().any(|candidate| {
            Command::new(candidate).arg("-version").output().is_ok_and(|out| out.status.success())
        });
        verification::prerequisite_available(found, "executable Z3 for unsigned invariant proofs")
    })
}

fn check(source: &str) -> Vec<String> {
    assert_ne!(
        std::env::var("Y_ALLOW_UNVERIFIED_INVARIANTS").as_deref(),
        Ok("1"),
        "these regressions require actual invariant verification"
    );
    let program = y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
        .parse_program()
        .expect("parse");
    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    assert!(
        checker.errors.iter().all(|e| !e.contains("SMT solver could not be run")),
        "a discovered Z3 must run: {:?}",
        checker.errors
    );
    checker.errors
}

fn accept(label: &str, source: &str) {
    let errors = check(source);
    assert!(errors.is_empty(), "{label}: {errors:?}\n{source}");
}

/// Rejected by the verifier itself - a failed obligation or a refusal to
/// model - and not by some earlier check.
fn reject(label: &str, source: &str) {
    let errors = check(source);
    assert!(
        errors.iter().any(|e| e.contains("SMT Safety Verification Failed") || e.contains("Cannot verify invariant")),
        "{label}: expected the verifier to reject it, got {errors:?}\n{source}"
    );
}

#[test]
fn an_unsigned_value_is_proved_in_its_own_range() {
    if !solver_available() {
        return;
    }
    // A parameter has no entry interval: only its type says it is >= 0. A
    // signed range for it would make this unprovable.
    for ty in ["U8", "U16", "U32", "U64"] {
        accept(ty, &format!("fn f(x: {ty}) {{ @invariant(x >= 0) for i in 0..1 {{ }} }}"));
    }
    // Above 2^31: an unsigned quotient of an unsigned value.
    accept(
        "high U32",
        "fn main() { let x: U32 = 3000000000; @invariant(x / 2 == 1500000000 && x % 7 == 4) for i in 0..1 { } }",
    );
}

#[test]
fn an_unsigned_counter_can_be_related_to_the_index() {
    if !solver_available() {
        return;
    }
    // `c == i` compares a U32 with the I32 index: both are proved to lie in
    // 0..=2^31-1, where every reading of the two agrees.
    for ty in ["U32", "U8"] {
        accept(
            ty,
            &format!("fn main() {{ let c: {ty} = 0; @invariant(c == i && i >= 0) for i in 0..10 {{ c = c + 1; }} }}"),
        );
    }
    // The control: a false relation over the same counter.
    reject("off by one", "fn main() { let c: U32 = 0; @invariant(c == i + 1) for i in 0..10 { c = c + 1; } }");
}

#[test]
fn unsigned_wraparound_is_never_modelled() {
    if !solver_available() {
        return;
    }
    // `x >= 0` holds on the machine, but only because 0 - 1 wraps; the
    // model describes no execution that wraps, signed or unsigned.
    reject("underflow", "fn main() { let x: U32 = 0; @invariant(x >= 0) for i in 0..3 { x = x - 1; } }");
    reject("overflow", "fn f(y: U32) { let x: U32 = y; @invariant(x >= 0) for i in 0..3 { x = x + 1; } }");
}

#[test]
fn a_negative_literal_beside_an_unsigned_value_is_refused() {
    if !solver_available() {
        return;
    }
    // True in the integers; false on every backend, which compares x with
    // 0xFFFFFFFF.
    reject("x > -1", "fn main() { let x: U32 = 5; @invariant(x > -1) for i in 0..1 { } }");
    reject("negated", "fn f(x: U32) { @invariant(-x <= 0) for i in 0..1 { } }");
}

#[test]
fn equal_bits_are_not_equal_values() {
    if !solver_available() {
        return;
    }
    // In the integers 4294967295 != -1. On the machine the two have the same
    // bits and compare equal, so this invariant is FALSE there.
    reject(
        "U32 max against I32 -1",
        "fn main() { let x: U32 = 4294967295; let d: I32 = -1; @invariant(x != d) for i in 0..1 { } }",
    );
    // Inside the range both readings share, the same comparison is proved.
    accept("in range", "fn main() { let x: U32 = 7; let d: I32 = 6; @invariant(x != d) for i in 0..1 { } }");
}

#[test]
fn a_u32_bound_is_read_as_the_loop_reads_it() {
    if !solver_available() {
        return;
    }
    // At n >= 2^31 the loop does not run on any backend, so `c` stays 0.
    // That is provable only by reading the bound's bits as an I32: taken as
    // its value the loop would run and `c == 0` would fail, and requiring
    // it to fit I32 is unprovable for a parameter.
    accept(
        "empty above 2^31",
        "fn f(n: U32) { let c: I32 = 0; @invariant((n <= 2147483647 || c == 0) && c >= 0 && c <= i) for i in 0..n { c = c + 1; } }",
    );
    reject(
        "the control: below 2^31 it runs",
        "fn f(n: U32) { let c: I32 = 0; @invariant(c == 0 && i >= 0) for i in 0..n { c = c + 1; } }",
    );
}
