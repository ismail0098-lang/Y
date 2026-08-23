//! `assert` over signals is a WITNESS-domain construct, and refusing it cost
//! three real circuits.
//!
//! Y used to refuse `assert` over any value it could not evaluate, with the
//! reason "it is a constraint on the witness, so ignoring it would emit a
//! circuit weaker than the source". **That premise is false, and it was
//! measured rather than argued**: compiled with circom 2.2.3, a circuit with
//! `assert(a >= 0)` over a signal and the same circuit without it have
//! identical non-linear, linear AND wire counts. circom emits nothing for it.
//!
//! So it is a witness-time check, exactly like `<--` is a witness-time
//! assignment, and the same argument that makes `Val::Opaque` safe applies
//! unchanged: the emitted `.r1cs` is byte-for-byte what a compiler that
//! honoured the assert would emit, because honouring it emits nothing.
//!
//! `circom-ecdsa`'s `ModSubThree` is the motivating case and it says so in the
//! source - `assert(a - b - c + (1 << n) >= 0)` sits under a comment reading
//! "assume a - b - c + 2**n >= 0". It documents a PRECONDITION on the caller.
//! It blocked `test_bigsub_15`, `test_bigsub_23` and `test_bigsubmodp_32`.
//!
//! **What Y does NOT do is evaluate it**, and that is a real difference from
//! circom, whose witness calculator aborts on a false assert. Y warns instead
//! of going quiet - `report_unchecked_asserts` - because learning this from a
//! divergence against circom later is the degradation this repo refuses.
//!
//! Run with:  cargo test --features zk --test circom_assert_is_witness_domain

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/circom").join(name)
}

fn compile(name: &str) -> Result<y::zk_emitter::ZkEmitter, String> {
    compile_file(&fixture(name), &[root().join("circomlib/circuits")])
}

fn compile_ok(name: &str) -> y::zk_emitter::ZkEmitter {
    compile(name).unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e))
}

/// Serialize the circuit and return the bytes, so two circuits can be compared
/// as ARTIFACTS rather than as constraint counts. A count is a weaker claim:
/// two different circuits can have the same one.
fn r1cs_bytes(name: &str, emitter: &y::zk_emitter::ZkEmitter) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!(
        "y_assert_{}_{}",
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

// ────────────────────────────────────────────────────────
// The claim the relaxation rests on
// ────────────────────────────────────────────────────────

/// The emitted artifact must be IDENTICAL with and without the assert.
///
/// This is the whole soundness argument in one assertion. If it ever fails,
/// the relaxation is dropping something real and must be reverted, not patched.
#[test]
fn an_assert_over_signals_changes_the_r1cs_not_at_all() {
    let with = compile_ok("assert_over_signals.circom");
    let without = compile_ok("assert_over_signals_removed.circom");

    let a = r1cs_bytes("with", &with);
    let b = r1cs_bytes("without", &without);

    assert_eq!(
        a.len(),
        b.len(),
        "the .r1cs changed SIZE when the assert was removed ({} vs {} bytes) - \
         the assert is emitting something, so it is not witness-domain",
        a.len(),
        b.len()
    );
    assert!(
        a == b,
        "the .r1cs differs with and without the assert, so the assert emits \
         constraints and skipping it weakens the circuit"
    );
}

/// ...and the circuit still computes what it should.
///
/// Identical-to-nothing would also be satisfied by emitting nothing at all, so
/// this pins that the surrounding circuit survived.
#[test]
fn the_circuit_around_the_assert_still_works() {
    let emitter = compile_ok("assert_over_signals.circom");
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();

    // a = 7, b = 3  ->  out = a*a + b = 52. Cross-checked against circom
    // 2.2.3's own wasm witness calculator on the same source.
    let privs = [Fr::from_u64(7), Fr::from_u64(3)];
    let (w, sat) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);

    assert!(sat, "the solved witness does not satisfy its own circuit");
    assert_eq!(circuit.outputs.len(), 1, "expected one output");
    assert_eq!(
        w[circuit.outputs[0]].to_u64(),
        Some(52),
        "out should be 7*7 + 3 = 52 (circom 2.2.3 agrees)"
    );
}

// ────────────────────────────────────────────────────────
// The control
// ────────────────────────────────────────────────────────

/// A compile-time-false assert must STILL be a hard error.
///
/// Without this, "delete the whole `Stmt::Assert` arm" passes every other test
/// in the file. The relaxation is only for values the compiler cannot evaluate;
/// where it CAN, circom itself fails the build and so must Y.
#[test]
fn a_false_compile_time_assert_is_still_refused() {
    let e = compile("assert_false_at_compile_time.circom")
        .err()
        .expect("a statically false `assert` must not compile");
    assert!(
        e.contains("assertion failed at compile time"),
        "refused, but not by the assert check - so the control proves nothing.\n{}",
        e
    );
}
