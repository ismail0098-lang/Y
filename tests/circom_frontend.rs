//! Y's circom front end, end to end.
//!
//! This is the piece that makes the R1CS back end reachable by anyone who has
//! not written a line of Y. Its acceptance test is not "does it parse" — it is
//! **does circomlib's `Poseidon(2)` compile through it and produce the digest
//! circomlib itself produces**, which is a published test vector and is already
//! pinned independently in `zk_poseidon_interop.rs` for Y's own front end.
//!
//! Getting that digest right requires the whole chain to be correct: the
//! three-character signal operators, template instantiation, component arrays,
//! signal arrays, compile-time `for` unrolling, functions returning constant
//! arrays, the constant/linear/quadratic value model, and the R1CS lowering.
//! Almost nothing can be subtly wrong and still land on it.
//!
//! Run with:  cargo test --features zk --test circom_frontend

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_emitter::ZkEmitter;
use y::zk_field::Fr;
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/circom").join(name)
}

/// circomlib lives in the repo, and its circuits `include` each other by bare
/// filename.
fn circomlib() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("circomlib/circuits")
}

struct Compiled {
    emitter: ZkEmitter,
    constraints: usize,
    wires: usize,
}

fn compile(name: &str) -> Result<Compiled, String> {
    let emitter = compile_file(&fixture(name), &[circomlib()])?;
    let v = emitter.view();
    let (constraints, wires) = (v.constraints.len(), v.num_variables);
    Ok(Compiled { emitter, constraints, wires })
}

/// Compile, solve the witness, and return the first output signal.
fn eval(name: &str, inputs: &[u64]) -> String {
    let c = compile(name).unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e));
    let circuit = c.emitter.build_circuit();
    let ir = c.emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (w, satisfied) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    assert!(satisfied, "{}: no satisfying witness found", name);
    check_r1cs_satisfiability(&circuit.constraints, &w)
        .unwrap_or_else(|e| panic!("{}: witness does not satisfy the circuit: {}", name, e));
    let out = *circuit.outputs.first().expect("circuit has no output signal");
    w[out].to_decimal_string()
}

#[test]
fn multiplier_computes_the_product() {
    assert_eq!(eval("multiplier.circom", &[3, 5]), "15");
    let c = compile("multiplier.circom").unwrap();
    assert_eq!(c.constraints, 1, "`c <== a * b` is exactly one R1CS constraint");
}

#[test]
fn subcomponents_arrays_and_loops() {
    // 1 + 4 + 9 + 16
    assert_eq!(eval("sum_squares.circom", &[1, 2, 3, 4]), "30");
    assert_eq!(eval("sum_squares.circom", &[0, 0, 0, 7]), "49");
}

/// **The acceptance test.**
///
/// circomlib's own `Poseidon(2)`, compiled by Y's circom front end, against the
/// digests circomlib's wasm witness calculator produces. These are the same four
/// vectors `zk_poseidon_interop.rs` pins for Y's native front end, so if this
/// passes, two completely independent paths through this compiler agree with
/// circomlib.
///
/// If it fails, the front end is wrong somewhere. **Do not update these values.**
#[test]
fn circomlib_poseidon_matches_its_published_digests() {
    let vectors: &[(u64, u64, &str)] = &[
        (1, 2, "7853200120776062878684798364095072458815029376092732009249414926327459813530"),
        (3, 4, "14763215145315200506921711489642608356394854266165572616578112107564877678998"),
        (0, 0, "14744269619966411208579211824598458697587494354926760081771325075741142829156"),
        (7, 0, "10402197090275139279073177788985849389816807868761640028215734431067655199248"),
    ];
    for (a, b, expected) in vectors {
        let got = eval("poseidon2.circom", &[*a, *b]);
        assert_eq!(
            &got, expected,
            "Poseidon({}, {}) through the circom front end disagrees with circomlib",
            a, b
        );
    }
}

/// The two circuits must agree with each other, not merely with a stored string.
#[test]
fn circom_and_native_front_ends_agree_on_poseidon() {
    use y::lexer::Lexer;
    use y::parser::Parser;

    let src = "@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    return poseidon_hash(x, y);\n}\n";
    let program = Parser::new(Lexer::new(src).tokenize()).parse_program().expect("parse");
    let mut native = ZkEmitter::new();
    native.emit_program(&program).expect("native lowering");
    let circuit = native.build_circuit();
    let ir = native.build_witness_ir();
    let (w, sat) = solve_r1cs_witness(
        &circuit.constraints,
        &ir,
        circuit.num_variables,
        &[],
        &[Fr::from_u64(5), Fr::from_u64(6)],
    );
    assert!(sat, "native poseidon witness");
    let native_digest = w[*circuit.outputs.first().unwrap()].to_decimal_string();

    assert_eq!(
        eval("poseidon2.circom", &[5, 6]),
        native_digest,
        "Y's circom front end and Y's own language disagree on the same hash"
    );
}

/// Structural metadata must match circom's, because that is what a verifier and
/// a witness file are indexed by.
///
/// Skipped with a notice when `circom` is not installed, like the `ptxas` and
/// `solc` gates elsewhere in this suite.
#[test]
fn r1cs_metadata_matches_circom() {
    let Ok(out) = std::process::Command::new("circom").arg("--version").output() else {
        eprintln!("note: `circom` not installed; skipping the differential metadata check");
        return;
    };
    if !out.status.success() {
        eprintln!("note: `circom --version` failed; skipping");
        return;
    }

    let tmp = std::env::temp_dir().join("y_circom_frontend_diff");
    let _ = std::fs::create_dir_all(&tmp);

    for name in ["multiplier.circom", "sum_squares.circom", "poseidon2.circom"] {
        let status = std::process::Command::new("circom")
            .arg(fixture(name))
            .arg("--r1cs")
            .arg("-o")
            .arg(&tmp)
            .arg("-l")
            .arg(circomlib())
            .output()
            .expect("run circom");
        if !status.status.success() {
            eprintln!("note: circom could not compile {}; skipping it", name);
            continue;
        }
        let their = read_r1cs_header(&tmp.join(name.replace(".circom", ".r1cs")));

        let c = compile(name).unwrap_or_else(|e| panic!("{}: {}", name, e));
        let v = c.emitter.view();

        // Public/private input and output COUNTS must agree: they define the
        // public signal layout a verifier checks and the order a `.wtns` is
        // written in. The wire and constraint totals legitimately differ — both
        // compilers reduce, and Y reduces harder (286/289 against circom's
        // 517/520 on `Poseidon(2)`) — so those are reported, not asserted.
        assert_eq!(
            their.pub_out, v.outputs.len(),
            "{}: output signal count differs from circom",
            name
        );
        assert_eq!(
            their.pub_in, v.public_inputs.len(),
            "{}: public input count differs from circom",
            name
        );
        assert_eq!(
            their.priv_in, v.private_inputs.len(),
            "{}: private input count differs from circom",
            name
        );
        eprintln!(
            "{:24} circom {:>6} constraints / {:>6} wires    Y {:>6} / {:>6}",
            name, their.constraints, their.wires, c.constraints, c.wires
        );
    }
}

struct R1csHeader {
    wires: usize,
    pub_out: usize,
    pub_in: usize,
    priv_in: usize,
    constraints: usize,
}

fn read_r1cs_header(path: &Path) -> R1csHeader {
    let d = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    assert_eq!(&d[0..4], b"r1cs", "{} is not an r1cs file", path.display());
    let n_sections = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
    let mut off = 12;
    for _ in 0..n_sections {
        let stype = u32::from_le_bytes(d[off..off + 4].try_into().unwrap());
        let ssize = u64::from_le_bytes(d[off + 4..off + 12].try_into().unwrap()) as usize;
        off += 12;
        if stype == 1 {
            let b = &d[off..off + ssize];
            let fs = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
            let g = |i: usize| {
                u32::from_le_bytes(b[4 + fs + i * 4..4 + fs + i * 4 + 4].try_into().unwrap()) as usize
            };
            let constraints =
                u32::from_le_bytes(b[4 + fs + 24..4 + fs + 28].try_into().unwrap()) as usize;
            return R1csHeader {
                wires: g(0),
                pub_out: g(1),
                pub_in: g(2),
                priv_in: g(3),
                constraints,
            };
        }
        off += ssize;
    }
    panic!("no header section in {}", path.display());
}

/// Unsupported constructs must be refused by name.
///
/// A front end that quietly drops what it does not understand emits a circuit
/// with fewer constraints than the source describes — which still proves, just
/// something weaker. Nothing downstream records the difference, so the only
/// place it can be caught is here.
#[test]
fn out_of_subset_constructs_are_refused_with_a_reason() {
    let e = compile("nonquadratic.circom").err().expect("a*b*c exceeds degree 2 and must be refused");
    assert!(
        e.contains("non-quadratic"),
        "the refusal should use circom's own word for this, got: {}",
        e
    );

    let e = compile("signal_condition.circom")
        .err()
        .expect("a branch on a signal value decides which constraints exist and must be refused");
    assert!(
        e.contains("compile time"),
        "the refusal should explain that the condition must be compile-time, got: {}",
        e
    );
    assert!(
        e.contains("multiplexer"),
        "the refusal should point at the standard workaround, got: {}",
        e
    );

    // `==` over signals is a gadget, not an operator. Saying so - and naming
    // the circomlib file that implements it - is the difference between a
    // five-minute fix and an afternoon.
    let e = compile("signal_compare.circom")
        .err()
        .expect("`==` over signals has no R1CS form and must be refused");
    assert!(
        e.contains("non-quadratic") && e.contains("comparators"),
        "the refusal should name the problem and point at circomlib, got: {}",
        e
    );
}

// ── `<--` witness hints: bitify / comparators ────────────────────────────────
//
// `Num2Bits` computes its witness with `out[i] <-- (in >> i) & 1` and then
// constrains it with `out[i] * (out[i] - 1) === 0` plus the recomposition
// `lc1 === in`. The shift has no R1CS form and does not need one — a `<--`
// right-hand side never becomes a constraint.
//
// This front end used to evaluate `<--` through the *constraint* value model
// and so refused the shift, which made circomlib's own `bitify.circom`
// uncompilable — and with it `comparators.circom`, `aliascheck.circom`, and
// every range check and comparison built on them. That is most of a real
// circuit, so these tests are the coverage gate for the fix.
//
// They assert values, not just satisfiability: `Num2Bits`'s recomposition
// constraint already makes a wrong bit *unsatisfiable*, but only a value check
// catches bits that are individually valid and in the wrong ORDER.

#[test]
fn num2bits_decomposition_recomposes_to_its_input() {
    for x in [0u64, 1, 2, 255, 256, 12345, 65535] {
        assert_eq!(
            eval("num2bits_sum.circom", &[x]),
            x.to_string(),
            "Num2Bits(16) did not recompose {}",
            x
        );
    }
}

#[test]
fn num2bits_bits_are_lsb_first() {
    // Bit 3 is set exactly when (x >> 3) & 1 == 1. A big-endian decomposition
    // satisfies every constraint in the circuit and fails this.
    for x in [0u64, 7, 8, 9, 15, 16, 65535] {
        assert_eq!(
            eval("num2bits_bit.circom", &[x]),
            ((x >> 3) & 1).to_string(),
            "bit 3 of {} is wrong — decomposition is not LSB-first",
            x
        );
    }
}

#[test]
fn lessthan_computes_the_right_answer() {
    for (a, b) in [(3u64, 5u64), (9, 5), (5, 5), (0, 1), (1, 0), (65535, 65536)] {
        assert_eq!(
            eval("lessthan.circom", &[a, b]),
            if a < b { "1" } else { "0" },
            "LessThan(32) is wrong for ({}, {})",
            a,
            b
        );
    }
}
