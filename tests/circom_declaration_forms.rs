//! Three circom spellings Y's front end did not accept, found by compiling
//! real production circuits rather than by reading the grammar.
//!
//! None of them is exotic. `var a, b;` is in semaphore's `Semaphore` template,
//! `c.in <== [a, b]` is the line above it, and `n \= 2;` is inside
//! zk-email's `log2Ceil`. Each was a hard refusal, so nothing was ever
//! silently wrong -- but each blocked a circuit outright.
//!
//! The `var a, b;` one is worth reading. It parsed to a `Stmt::Block`, and
//! `exec_stmt` opens a scope for a block and pops it on the way out, so both
//! names were gone by the next statement: "`a` is not a variable in scope".
//! `signal x, y;` had the same shape and survived only by accident, because
//! `Frame::signals` is not scoped at all. The fix is a `Stmt::Seq` that groups
//! statements WITHOUT opening a scope -- block scoping itself is correct and
//! matches circom, which is what `a_block_still_opens_a_scope` pins.
//!
//! Run with:  cargo test --features zk --test circom_declaration_forms

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_emitter::ZkEmitter;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/circom").join(name)
}

fn compile(name: &str) -> Result<ZkEmitter, String> {
    compile_file(&fixture(name), &[root().join("circomlib/circuits")])
}

fn compile_ok(name: &str) -> ZkEmitter {
    compile(name).unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e))
}

fn r1cs_bytes(name: &str, emitter: &ZkEmitter) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!(
        "y_declforms_{}_{}",
        std::process::id(),
        name.replace('.', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.r1cs");
    emitter
        .write_r1cs_binary(path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: could not write r1cs: {}", name, e));
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn solve_one(name: &str, privs: &[Fr]) -> (bool, Option<u64>) {
    let emitter = compile_ok(name);
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let (w, sat) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], privs);
    assert_eq!(circuit.outputs.len(), 1, "{}: expected one output", name);
    (sat, w[circuit.outputs[0]].to_u64())
}

// ────────────────────────────────────────────────────────
// `var a, b;`
// ────────────────────────────────────────────────────────

#[test]
fn several_variables_in_one_declaration_stay_in_scope() {
    let (sat, out) = solve_one("multi_var_decl.circom", &[Fr::from_u64(4)]);
    assert!(sat, "the solved witness does not satisfy its own circuit");
    assert_eq!(out, Some(48), "o should be (5 + 7) * 4 = 48");
}

/// The control. Removing block scoping would also make `var a, b;` work, and
/// would be wrong: circom rejects a read of a block-local with error[T2021],
/// and so must Y.
#[test]
fn a_block_still_opens_a_scope() {
    match compile("block_still_scopes.circom") {
        Ok(_) => panic!(
            "a variable declared inside `{{ }}` escaped its block. circom refuses \
             this with error[T2021]; the `var a, b;` fix must not have been made \
             by deleting block scoping."
        ),
        Err(e) => assert!(
            e.contains("hidden"),
            "refused, but not for the block-local being out of scope: {}",
            e
        ),
    }
}

// ────────────────────────────────────────────────────────
// Whole-array signal assignment
// ────────────────────────────────────────────────────────

/// `c.in <== [a, b]` and `c.in <== other` must be exactly the element-wise
/// form. circom's own two files are byte-identical here too, checked with
/// circom 2.2.3.
#[test]
fn driving_a_whole_signal_array_is_the_element_wise_form() {
    let bulk = r1cs_bytes("bulk", &compile_ok("array_signal_assign.circom"));
    let each = r1cs_bytes("each", &compile_ok("array_signal_assign_explicit.circom"));
    assert!(
        bulk == each,
        "`c.in <== [a, b]` did not produce the same circuit as assigning the \
         elements one at a time"
    );
}

#[test]
fn the_array_assignment_computes_what_the_source_says() {
    // i = 10 gives 10 + 11 + 12 = 33 from the literal; pair = 1,2,3 gives 6.
    let ins = [
        Fr::from_u64(10),
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
    ];
    let (sat, out) = solve_one("array_signal_assign.circom", &ins);
    assert!(sat, "the solved witness does not satisfy its own circuit");
    assert_eq!(out, Some(39), "o should be (10+11+12) + (1+2+3) = 39");
}

/// A length mismatch must be refused, not padded or truncated. circom:
/// error[T3001]. Silently dropping the extra element would leave a signal
/// unconstrained, which is the one failure mode in this backend that still
/// produces a valid-looking proof.
#[test]
fn an_array_assignment_of_the_wrong_length_is_refused() {
    match compile("array_signal_assign_wrong_length.circom") {
        Ok(_) => panic!("a 2-element literal was accepted for a 3-signal array"),
        Err(e) => assert!(
            e.contains("2 element(s) but the signal array has 3"),
            "refused for the wrong reason: {}",
            e
        ),
    }
}

// ────────────────────────────────────────────────────────
// `\=`
// ────────────────────────────────────────────────────────

/// `\=` is compound INTEGER division, not field division.
///
/// The distinction is the whole point: `/` in circom is multiplication by the
/// modular inverse, so `100 / 2 \ ... ` and `100 \ 2` agree here only because
/// 2 divides 100 -- and the loop keeps halving until it does not (25 \ 2 = 12,
/// where field division would give a 254-bit number and the `n > 0` test would
/// never end). A `\=` lowered to `Div` therefore hangs rather than answering
/// wrongly, but it is still the wrong operator.
#[test]
fn compound_integer_division_is_integer_division() {
    let (sat, out) = solve_one("int_div_assign.circom", &[Fr::from_u64(3)]);
    assert!(sat, "the solved witness does not satisfy its own circuit");
    // 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1 -> 0 is seven halvings.
    assert_eq!(out, Some(21), "o should be 7 * 3 = 21");
}
