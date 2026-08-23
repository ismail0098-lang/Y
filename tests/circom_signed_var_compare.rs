//! circom compares `var`s by their SIGNED representative; Y compared them
//! canonically, and the same source computed two different functions.
//!
//! A circom `var` holds a field element, and circom orders those by the
//! representative in `[-(p-1)/2, (p-1)/2]`: a value above `(p-1)/2` denotes
//! `v - p`, a negative number. Y's `Fr: Ord` is CANONICAL and deliberately so -
//! `zk_emitter` folds Y's OWN `<`/`<=`/`>`/`>=` through it, where the operands
//! carry a 32-bit range check that makes the canonical order the right one.
//! The circom front end borrowed that ordering, and the two disagree on every
//! pair straddling `(p-1)/2`.
//!
//! Measured against circom 2.2.3, on this source:
//!
//! ```text
//! var a = 0 - 1;
//! if (a < 1) { out <== 111; } else { out <== 222; }
//! ```
//!
//! circom emits a circuit computing **111** and Y emitted one computing
//! **222**. Both compile. Both are one constraint. Both produce a valid
//! Groth16 proof - of different statements. That is the worst failure shape
//! this repo has a name for, and it is invisible to a constraint count, to
//! satisfiability, and to `snarkjs verify`.
//!
//! Only the four comparisons are signed. `\`, `%`, `<<`, `>>` and the bitwise
//! operators were probed at `p-1` against circom 2.2.3 and all return the
//! UNSIGNED result, so they are deliberately left alone - `the_arithmetic_
//! operators_stay_unsigned` pins that, because "make the whole front end
//! signed" would otherwise pass every other test here.
//!
//! Run with:  cargo test --features zk --test circom_signed_var_compare

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

/// Compile a fixture and solve it; returns the output values in order.
fn outputs_of(name: &str) -> Vec<Fr> {
    let emitter = compile_file(&fixture(name), &[root().join("circomlib/circuits")])
        .unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e));
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let (w, sat) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &[]);
    assert!(sat, "{}: the solved witness does not satisfy its own circuit", name);
    circuit.outputs.iter().map(|i| w[*i]).collect()
}

fn as_u64s(v: &[Fr]) -> Vec<u64> {
    v.iter()
        .map(|f| f.to_u64().expect("output did not fit in u64"))
        .collect()
}

// ────────────────────────────────────────────────────────
// The fix
// ────────────────────────────────────────────────────────

/// Every discriminating comparison must match circom 2.2.3.
///
/// The expected vector is not derived from Y - it was produced by compiling
/// `signed_var_compare.circom` with circom 2.2.3 and running its own wasm
/// witness calculator. Deriving it from Y would make this an internal
/// consistency check, which is exactly what failed to catch the bug.
#[test]
fn var_comparisons_use_circoms_signed_order() {
    // circom 2.2.3, `circom signed_var_compare.circom --r1cs --wasm` then
    // `node generate_witness.js`:  o = [1, 0, 0, 0, 1, 0, 1, 1, 1]
    const CIRCOM: [u64; 9] = [1, 0, 0, 0, 1, 0, 1, 1, 1];

    let got = as_u64s(&outputs_of("signed_var_compare.circom"));
    assert_eq!(
        got.len(),
        CIRCOM.len(),
        "fixture changed shape; expected {} outputs",
        CIRCOM.len()
    );

    let labels = [
        "(p-1) <  1",
        "H     <  H+1",
        "0     <  (p-1)",
        "(p-1) >  1",
        "(p+1)/2 < 0",
        "(p-1) >= 0",
        "(p-1) <= (p-1)   [control]",
        "H     >  0       [control]",
        "2     <  5       [control]",
    ];
    let mut wrong = Vec::new();
    for (i, (g, w)) in got.iter().zip(CIRCOM.iter()).enumerate() {
        if g != w {
            wrong.push(format!("  o[{}]  {}   circom {}  Y {}", i, labels[i], w, g));
        }
    }
    assert!(
        wrong.is_empty(),
        "Y disagrees with circom on {} of {} comparisons:\n{}",
        wrong.len(),
        got.len(),
        wrong.join("\n")
    );
}

/// The control that stops the canonical order from passing this file.
///
/// Six of the nine cases were chosen because the two orderings disagree on
/// them. If a future change made `signed_cmp` canonical again - or made the
/// fixture stop straddling `(p-1)/2` - this asserts loudly rather than letting
/// the test above keep passing on three controls.
#[test]
fn the_canonical_order_really_would_disagree() {
    // What Y emitted BEFORE the fix, i.e. `Fr: Ord` applied directly.
    const CANONICAL: [u64; 9] = [0, 1, 1, 1, 0, 1, 1, 1, 1];
    const CIRCOM: [u64; 9] = [1, 0, 0, 0, 1, 0, 1, 1, 1];

    let differ = CANONICAL
        .iter()
        .zip(CIRCOM.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differ, 6,
        "the fixture must keep six discriminating cases; got {}",
        differ
    );

    let got = as_u64s(&outputs_of("signed_var_compare.circom"));
    assert_ne!(
        got, CANONICAL,
        "Y is still using the canonical order - the fix is not in effect"
    );
}

/// `\`, `%` and `>>` are NOT signed in circom, and must not be swept up.
///
/// Probed against circom 2.2.3 at `p-1`: `(p-1) \ 2` and `(p-1) >> 1` both
/// return `(p-1)/2` (the unsigned answer; the signed one would be 0 or -1),
/// and `(p-1) % 2` returns 0. A change that made the whole front end signed
/// would pass every other test in this file and fail this one.
#[test]
fn the_arithmetic_operators_stay_unsigned() {
    let got = as_u64s_or_big(&outputs_of("unsigned_var_arith.circom"));
    let half = "10944121435919637611123202872628637544274182200208017171849102093287904247808";
    assert_eq!(got[0], half, "(p-1) \\ 2 should be the UNSIGNED quotient");
    assert_eq!(got[1], "0", "(p-1) % 2");
    assert_eq!(got[2], half, "(p-1) >> 1 should be the UNSIGNED shift");
}

fn as_u64s_or_big(v: &[Fr]) -> Vec<String> {
    v.iter().map(|f| f.to_decimal_string()).collect()
}

// ────────────────────────────────────────────────────────
// Scope: the signed rule is circom's, not Y's
// ────────────────────────────────────────────────────────

/// Y's own `.ysu` language must NOT have acquired circom's ordering.
///
/// Y's comparison gadget range-checks both operands to 32 bits, so a value
/// like `p-1` is out of range and the program is refused at compile time
/// (`require_gadget_range`). That is a different and deliberate design - a
/// field has no order, so Y makes an out-of-range ordering claim unprovable
/// rather than answering it. This asserts the fix did not leak across.
#[test]
fn y_s_own_language_still_range_checks_instead() {
    let src = "fn main(a: I32) -> I32 { if 4294967296 < a { return 1; } return 0; }";
    let dir = std::env::temp_dir().join(format!("y_signed_scope_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scope.ysu");
    std::fs::write(&path, src).unwrap();

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--target=r1cs")
        .current_dir(&dir)
        .output()
        .expect("failed to run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        !out.status.success(),
        "Y accepted an out-of-range comparison operand; the circom signed rule \
         may have leaked into zk_emitter.\n{}",
        text
    );
}
