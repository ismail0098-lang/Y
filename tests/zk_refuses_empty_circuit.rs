//! A constraint system with no constraints asserts nothing, and must be refused.
//!
//! An empty R1CS is satisfied by EVERY assignment. A Groth16 proof over it
//! verifies unconditionally and proves nothing at all — so it is the one
//! artifact this backend must never hand a user while reporting success.
//!
//! It did, four ways: `fn main() {}`, a `main` whose body only declares an
//! unused local, a body that is entirely `@ghost`, and — the case that matters —
//! every program whose work lives in a `kernel` rather than in `main`. Measured
//! over `tests/`: **all 57 programs the backend accepted produced 1 wire, 0
//! constraints, no inputs and no outputs**, under "Compilation Successful!" and
//! exit 0, with `.r1cs`, `.sym` and `.r1cs.txt` written to disk.
//!
//! This is the empty-artifact bug `tests/backends_refuse_empty_artifacts.rs`
//! records for the PTX and co-processor backends, in the one backend where the
//! artifact is supposed to carry a soundness guarantee. An empty `.ptx` does
//! nothing; an empty `.r1cs` ASSERTS nothing while looking exactly like a
//! circuit that does.
//!
//! ## Two arms, one guard
//!
//! The `.ysu` path and the circom path build the circuit through different code
//! and write it through different code. The first fix went into the `.ysu` arm
//! only, and an empty circom template still compiled clean — the circom arm even
//! PRINTS its own constraint count one line above the write without looking at
//! it. Both call `refuse_if_no_constraints` now, and both are tested here,
//! because "fixed one arm of a match" is the recurring shape in CLAUDE.md's
//! design-rule table.
//!
//! ## What is deliberately NOT refused
//!
//! The test is the constraint count alone. A circuit with no PUBLIC inputs is
//! legitimate — that is a proof of knowledge of a witness — and so is one with
//! no outputs, which is what a body of pure assertions emits. Refusing either
//! would break real circuits; zero constraints is the one shape that cannot mean
//! anything.
#![cfg(feature = "zk")]

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Run {
    ok: bool,
    text: String,
    artifacts: Vec<String>,
}

/// Compile `name` (already written into a private dir) with `--target=r1cs`.
fn run(name: &str, src: &str, ext: &str, extra: &[&str]) -> Run {
    let dir = std::env::temp_dir().join(format!("y_zkempty_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.{}", name, ext));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--target=r1cs")
        .args(extra)
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let artifacts = ["r1cs", "sym", "r1cs.txt"]
        .iter()
        .map(|e| dir.join(format!("{}.{}", name, e)))
        .filter(|p| p.exists())
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    Run { ok: out.status.success(), text, artifacts }
}

fn assert_refused(r: &Run, what: &str) {
    assert!(!r.ok, "{} compiled and exited 0:\n{}", what, r.text);
    assert!(
        r.text.contains("no constraints"),
        "{} was refused, but not for being empty:\n{}",
        what,
        r.text
    );
    // The refusal has to land before anything is written, or the user is left
    // holding a `.r1cs` that a prover will happily accept.
    assert!(
        r.artifacts.is_empty(),
        "{} was refused but still wrote {:?}",
        what,
        r.artifacts
    );
}

#[test]
fn a_void_main_is_refused() {
    assert_refused(&run("voidmain", "fn main() {\n}\n", "ysu", &[]), "`fn main() {}`");
}

#[test]
fn a_body_that_emits_nothing_is_refused() {
    assert_refused(
        &run("deadlocal", "fn main() {\n    let x: I32 = 5;\n}\n", "ysu", &[]),
        "a body with only an unused local",
    );
}

/// The case that produced all 57, in the exact shape the corpus has it.
///
/// A GPU program's work is in a `kernel`, and the R1CS backend compiles `main`.
/// A `kernel` with NO `main` at all is already refused correctly ("No entry
/// function 'main' or 'circuit' found") — but every one of these files ends
/// with a literal `fn main() {}` stub, and that stub is precisely what turns a
/// correct refusal into an empty circuit. `tests/bn254_sub_vec.ysu:101` is
/// `fn main() {}`, and so are the other 56.
#[test]
fn a_kernel_plus_an_empty_main_stub_is_refused() {
    let src = "\
kernel touch_it(Out: GlobalMemory<U32>, N: U32) {
    let tid: U32 = thread_idx_x();
    block_ptr2d_store(Out, 0, tid, 1, 1, N, tid);
}

fn main() {}
";
    assert_refused(
        &run("kernelstub", src, "ysu", &[]),
        "a kernel with an empty `main` stub",
    );
}

/// And the neighbouring case stays refused for its own reason, so the two are
/// not confused: no `main` at all is a missing entry point, not an empty
/// circuit. Checked because my first version of the test above used this
/// shape and passed for the wrong reason.
#[test]
fn a_kernel_with_no_main_is_still_a_missing_entry_point() {
    let src = "\
kernel touch_it(Out: GlobalMemory<U32>, N: U32) {
    let tid: U32 = thread_idx_x();
    block_ptr2d_store(Out, 0, tid, 1, 1, N, tid);
}
";
    let r = run("nomain", src, "ysu", &[]);
    assert!(!r.ok, "a program with no entry function compiled:\n{}", r.text);
    assert!(
        r.text.contains("No entry function"),
        "refused, but not as a missing entry point:\n{}",
        r.text
    );
}

/// The control. "Refuse everything" satisfies every test above and deletes the
/// backend — the shape `ordinary_loop_bodies_still_verify` exists for.
#[test]
fn a_real_circuit_still_compiles() {
    let src = "\
fn main(a: I32, b: I32) -> I32 {
    let c: I32 = a * b;
    return c + a;
}
";
    let r = run("realcircuit", src, "ysu", &[]);
    assert!(r.ok, "a real circuit was refused:\n{}", r.text);
    let mut got = r.artifacts.clone();
    got.sort();
    assert_eq!(
        got,
        vec!["r1cs".to_string(), "r1cs.txt".to_string(), "sym".to_string()]
            .into_iter()
            .map(|e| format!("realcircuit.{}", e))
            .collect::<Vec<_>>(),
        "a successful compile did not write all three artifacts"
    );
}

// ── the circom arm ────────────────────────────────────────

#[test]
fn the_circom_front_end_refuses_an_empty_circuit_too() {
    let src = "\
pragma circom 2.0.0;
template Empty() {
    signal input a;
}
component main = Empty();
";
    assert_refused(&run("emptycircom", src, "circom", &[]), "an empty circom template");
}

/// The circom control, and the reason it is here rather than assumed: the first
/// version of this fix guarded only the `.ysu` arm and this exact circuit
/// compiled clean, so the arm needs a positive case as well as a negative one.
#[test]
fn a_real_circom_circuit_still_compiles() {
    let src = "\
pragma circom 2.0.0;
template Mul() {
    signal input a;
    signal input b;
    signal output c;
    c <== a * b;
}
component main = Mul();
";
    let r = run("realcircom", src, "circom", &[]);
    assert!(r.ok, "a real circom circuit was refused:\n{}", r.text);
    assert!(
        r.artifacts.iter().any(|a| a.ends_with(".r1cs")),
        "a successful circom compile wrote no .r1cs, only {:?}",
        r.artifacts
    );
}

/// A circuit with no PUBLIC inputs is a proof of knowledge of a witness, which
/// is the whole point of the scheme. Pinning it so a future "tighten the empty
/// check" does not start refusing real circuits.
#[test]
fn a_circuit_with_no_public_inputs_is_not_refused() {
    let src = "\
fn main(secret: I32) -> I32 {
    return secret * secret;
}
";
    let r = run("noPublic", src, "ysu", &[]);
    assert!(
        r.ok,
        "a circuit with only private inputs was refused; that is a proof of \
         knowledge, not an empty circuit:\n{}",
        r.text
    );
}
