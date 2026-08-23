//! `--witness` reads circom's own `input.json`, and the witness it produces
//! agrees with circom's calculator element for element.
//!
//! Until this landed, "Y compiles this circuit" did not imply "Y can run it".
//! Three separate things stopped it, and all three were invisible from Y's own
//! side because every check Y ran used Y's own naming:
//!
//!   1. **The top-level prefix had no dot.** `signal input a` was wired as
//!      `maina` while a nested one was `mainc.x`, so `{"a": "5"}` -- the file
//!      circom's own witness calculator takes -- was rejected with
//!      `input "maina" is missing`. It is `main.a` now, which is circom's
//!      spelling and consistent with Y's own nested one.
//!   2. **The input reader was scalar-only.** `parse_scalar_map` refused a JSON
//!      array, and a circom input file is routinely `{"in": ["1", "2"]}`. Y's
//!      wires behind that are `main.in[0]` and `main.in[1]`, so reading the file
//!      means flattening it exactly as `alloc_signal` flattens the declaration.
//!   3. **Public inputs were never bound.** `solve_and_write_witness` passed
//!      `&[]` for `pub_in`, so every signal named in `{public [...]}` was solved
//!      at zero. Y's own language has no `public` keyword and never populates
//!      `public_inputs`, which is why this was invisible until the circom front
//!      end existed -- and why the fixture here has a public input that an
//!      OUTPUT depends on, so leaving it at zero is a wrong answer rather than
//!      a missing one.
//!
//! **The acceptance test is circom's own witness, recorded**: for this fixture
//! circom 2.2.3's calculator produces `[1, 196, 14, 7, 2, 3, 4, 1, 2, 3, 4]`
//! and Y must produce the same list. That is stronger than "it solved" -- a
//! circuit solved with a public input at zero still solves, it just proves
//! something else.
//!
//! Run with:  cargo test --features zk --test circom_witness_inputs

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/circom").join(name)
}

fn workdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("y_wtns_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run the real binary. `--witness` lives in `main.rs`, not the library, so a
/// subprocess is the only way to exercise the path a user actually takes.
fn run(circuit: &str, inputs: &str, tag: &str) -> (bool, String, PathBuf) {
    let dir = workdir(tag);
    let out_r1cs = dir.join("c.r1cs");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(fixture(circuit))
        .arg("--target=r1cs")
        .arg("--witness")
        .arg(fixture(inputs))
        .arg("-o")
        .arg(&out_r1cs)
        .output()
        .expect("failed to run the Y binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text, dir.join("c.wtns"))
}

/// Read an iden3 `.wtns` as decimal strings, in witness order.
///
/// Decimal strings rather than integers because a field element does not fit a
/// `u64` and this test must be able to compare a full-width one.
fn read_wtns(path: &Path) -> Vec<String> {
    let b = std::fs::read(path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    assert_eq!(&b[0..4], b"wtns", "not a .wtns file");
    let n_sections = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
    let mut off = 12usize;
    let mut sections = std::collections::HashMap::new();
    for _ in 0..n_sections {
        let t = u32::from_le_bytes(b[off..off + 4].try_into().unwrap());
        let len = u64::from_le_bytes(b[off + 4..off + 12].try_into().unwrap()) as usize;
        off += 12;
        sections.insert(t, (off, len));
        off += len;
    }
    let (ho, _) = sections[&1];
    let fs = u32::from_le_bytes(b[ho..ho + 4].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(b[ho + 4 + fs..ho + 8 + fs].try_into().unwrap()) as usize;
    let (dof, _) = sections[&2];
    (0..n)
        .map(|i| {
            let mut acc = y::zk_field::BigUint::zero();
            let base = y::zk_field::BigUint::from_u64(256);
            for byte in b[dof + i * fs..dof + (i + 1) * fs].iter().rev() {
                acc = acc.mul(&base).add(&y::zk_field::BigUint::from_u64(*byte as u64));
            }
            acc.to_decimal_string()
        })
        .collect()
}

/// What circom 2.2.3's own witness calculator produces for
/// `witness_inputs.circom` fed `witness_inputs.json`.
///
/// `sum` = 1*2 + 2*3 + 3*4 + 10*1 + 11*2 + 20*3 + 21*4 = 196.
/// `scaled` = scale * in[0] = 7 * 2 = 14, and `scale` is the PUBLIC input.
const CIRCOM_WITNESS: [&str; 11] =
    ["1", "196", "14", "7", "2", "3", "4", "1", "2", "3", "4"];

// ────────────────────────────────────────────────────────
// The claim
// ────────────────────────────────────────────────────────

#[test]
fn a_circom_input_file_is_accepted_verbatim() {
    let (ok, text, wtns) = run("witness_inputs.circom", "witness_inputs.json", "plain");
    assert!(ok, "compiling with --witness failed:\n{}", text);
    let got = read_wtns(&wtns);
    assert_eq!(
        got, CIRCOM_WITNESS,
        "Y's witness differs from the one circom 2.2.3's calculator produces \
         from the same input file"
    );
}

/// The `.sym` name is the other half of the same fix, and it is what a user
/// greps to find out what an input is called.
#[test]
fn the_symbol_file_uses_circoms_names() {
    let (ok, text, wtns) = run("witness_inputs.circom", "witness_inputs.json", "sym");
    assert!(ok, "{}", text);
    let sym = std::fs::read_to_string(wtns.with_file_name("c.sym")).unwrap();

    for expected in ["main.scale", "main.in[2]", "main.nested[1][0]", "main.sum"] {
        assert!(
            sym.lines().any(|l| l.rsplit(',').next() == Some(expected)),
            "`{}` is not a symbol name in the .sym file:\n{}",
            expected,
            sym
        );
    }
    // The uniquifying `_<id>` suffix must be gone -- circom's `.sym` carries
    // the source name, and so must Y's if the two files are to be read
    // interchangeably.
    assert!(
        !sym.contains("main.scale_"),
        "the .sym still carries the internal `_<id>` suffix:\n{}",
        sym
    );
    // circom emits no row for the constant-1 wire and starts at label 1.
    //
    // Asserted on the LABEL COLUMN, not on the name. The first version of this
    // looked for the literal `const_1` and was vacuous: `display_name` strips
    // the trailing `_1`, so the row would have read `0,0,0,const` and the
    // assertion passed with the row present. Found by mutation, not by review.
    assert!(
        sym.lines().all(|l| !l.is_empty() && !l.starts_with("0,")),
        "the .sym has a row for wire 0, the constant 1, which circom does not \
         emit:\n{}",
        sym
    );
}

/// The fully-qualified spelling works too, so a name copied out of the `.sym`
/// can be pasted into an input file.
#[test]
fn the_fully_qualified_name_is_accepted_as_well() {
    let (ok, text, wtns) = run(
        "witness_inputs.circom",
        "witness_inputs_qualified.json",
        "qualified",
    );
    assert!(ok, "{}", text);
    assert_eq!(read_wtns(&wtns), CIRCOM_WITNESS);
}

// ────────────────────────────────────────────────────────
// Refusals
// ────────────────────────────────────────────────────────

fn refused(inputs: &str, needle: &str, tag: &str) {
    let (ok, text, _) = run("witness_inputs.circom", inputs, tag);
    assert!(!ok, "{} was accepted; it should not be:\n{}", inputs, text);
    assert!(
        text.contains(needle),
        "{}: refused for the wrong reason.\n  expected to mention: {}\n  got: {}",
        inputs,
        needle,
        text
    );
}

#[test]
fn a_missing_input_is_named() {
    refused(
        "witness_inputs_missing.json",
        "\"nested[0][0]\" is missing",
        "missing",
    );
}

/// A short array is a missing input, not a silently padded one. Padding would
/// solve a circuit whose inputs the author did not supply.
#[test]
fn a_short_array_is_a_missing_input_not_a_padded_one() {
    refused(
        "witness_inputs_short_array.json",
        "\"in[2]\" is missing",
        "short",
    );
}

/// An input the circuit does not have is an error, not ignored -- it is
/// usually a typo for one that IS an input, and ignoring it would leave that
/// one missing instead.
#[test]
fn an_input_the_circuit_does_not_have_is_refused() {
    refused(
        "witness_inputs_unknown.json",
        "\"notAnInput\", which is not an input",
        "unknown",
    );
}

#[test]
fn a_json_object_is_not_a_field_element() {
    refused(
        "witness_inputs_object.json",
        "is a JSON object",
        "object",
    );
}
