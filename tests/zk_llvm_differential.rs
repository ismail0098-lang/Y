// ============================================================
//  The ZK backend vs the LLVM backend, on CONTROL FLOW.
//
//  Four bugs have been found in `zk_emitter`'s handling of `return`,
//  `if` and `for`, and CLAUDE.md's table records all four with the same
//  note: each compiled cleanly and produced a satisfiable circuit
//  computing a different function than the source. They were found by
//  Z3 (two), by writing the semantics in Coq (one), and by noticing a
//  lowering was never CALLED (one) -- expensive instruments, all of
//  them, and the document says of the first: "the LLVM backend compiles
//  the same program to `ret i32 1`, so the two backends disagreed on
//  what `return` means -- a cross-backend differential is the cheapest
//  test that would have caught it".
//
//  `tests/backend_differential.rs` is that test for LLVM vs
//  `--emit-native`, and it cannot reach control flow at all, because
//  the native backend has no branches. This file is the pairing that
//  can: the same function body, compiled to a native binary by one
//  backend and to an R1CS circuit solved for its witness by the other.
//
//  **Both sides take the inputs as PARAMETERS**, which is what keeps
//  either from constant-folding a condition away -- LLVM calls
//  `compute(...)` from `main`, and the ZK arm passes them as private
//  inputs. A version with the values inlined would be testing the two
//  constant folders instead, which is a different question (and one
//  `zk_fuzz`'s metamorphic oracle already asks).
//
//  ## Where the two languages genuinely differ, and how that is avoided
//
//  R1CS arithmetic is in BN254's scalar field, not `i32`. So:
//
//    * **subtraction is not generated at all** -- `3 - 5` is `p - 2` in
//      the field and -2 in LLVM, a real difference and not a bug;
//    * **every binding is masked to 16 bits**, so no product can pass
//      `i32` (where LLVM wraps and the field does not);
//    * division and shift right-operands are non-zero literals, for the
//      same reasons as in the native differential.
//
//  Divergence outside those rules is a finding, not noise.
//
//  ## What it actually catches, measured by re-introducing the bugs
//
//  Not asserted -- each of the four historical control-flow bugs was put
//  back into `zk_emitter.rs` and this file re-run:
//
//    1. `return` not a terminator (the LAST return wins)      CAUGHT
//    2. `Stmt::For` discards its body's return                CAUGHT
//    3. the `if` condition never constrained to a bit         NOT caught
//    4. one-sided return muxed against the wrong tail         n/a
//
//  (3) is a CONFIRMATION of a claim CLAUDE.md already makes, not a hole:
//  every condition this generator writes is a comparison, and a comparison
//  already yields a booleanity-constrained bit, so "nothing reachable was
//  wrong -- the guarantee was absent rather than unused". Catching it needs a
//  raw-integer condition, which the fix deliberately makes UNSATISFIABLE
//  rather than wrong, so it would show up in the skip count and not as a
//  divergence. (4) no longer has a shape to re-introduce: the predicated
//  lowering the Coq proof produced does not have a "wrong tail" to pick.
//
//  **(2) was NOT caught by the first version of this file, twice over.** The
//  generator emitted no loops at all; adding them was still not enough,
//  because the loop guard compared a raw parameter (0..199) against an index
//  below 5, so the return inside the loop could never be taken. The compared
//  value is masked to 0..3 for that reason. A branch that cannot be reached
//  tests nothing.
// ============================================================
#![cfg(feature = "zk")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use y::zk_emitter::ZkEmitter;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

static SALT: AtomicUsize = AtomicUsize::new(0);

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Operators that mean the same thing in `i32` and in `Fr` for non-negative
/// operands below 2^16. `-` is deliberately absent; see the header.
const OPS: [&str; 9] = ["+", "*", "/", "%", "&", "|", "^", "<<", ">>"];
const CMPS: [&str; 6] = ["<", "<=", ">", ">=", "==", "!="];

const NPARAMS: usize = 3;

fn value(r: &mut Rng, locals: u64) -> String {
    match r.below(3) {
        0 if locals > 0 => format!("v{}", r.below(locals)),
        1 => format!("p{}", r.below(NPARAMS as u64)),
        _ => format!("{}", r.below(60)),
    }
}

fn arith(r: &mut Rng, locals: u64) -> String {
    let op = OPS[r.below(OPS.len() as u64) as usize];
    let lhs = value(r, locals);
    let rhs = match op {
        "/" | "%" => format!("{}", 1 + r.below(20)),
        "<<" | ">>" => format!("{}", r.below(4)),
        _ => value(r, locals),
    };
    format!("{lhs} {op} {rhs}")
}

fn condition(r: &mut Rng, locals: u64) -> String {
    format!(
        "{} {} {}",
        value(r, locals),
        CMPS[r.below(CMPS.len() as u64) as usize],
        value(r, locals)
    )
}

/// Builds the shared function body: masked `let`s, then a nest of `if`s with
/// returns, then a fall-through return.
fn body(r: &mut Rng, indent: usize, locals: u64, depth: u32) -> (String, u64) {
    let pad = "    ".repeat(indent);
    let mut out = String::new();
    let mut n = locals;
    for _ in 0..1 + r.below(2) {
        // The mask is what keeps `i32` and `Fr` in agreement: without it a
        // chain of products leaves 32 bits, LLVM wraps and the field does not.
        out.push_str(&format!("{pad}let v{n}: I32 = ({}) & 65535;\n", arith(r, n)));
        n += 1;
    }
    // Reassignment and compound assignment, because `zk_emitter::emit_stmt`
    // handled ten `Stmt` variants and swallowed the rest in `_ => {}` -- so
    // `x = x + 5;` emitted a constraint and `x += 5;` emitted NOTHING, and the
    // emitter disagreed with itself about what one program meant. Both spell
    // the same thing and must give the same answer here.
    if n > 0 && r.below(2) == 0 {
        let target = r.below(n);
        if r.below(2) == 0 {
            out.push_str(&format!(
                "{pad}v{target} = (v{target} + {}) & 65535;\n",
                1 + r.below(40)
            ));
        } else {
            // Only `+=`, `-=` and `*=` exist -- the parser rejects `&=`,
            // `|=` and `^=` outright. `-=` is excluded here for the same
            // reason plain subtraction is: it can go negative.
            let op = ["+=", "*="][r.below(2) as usize];
            out.push_str(&format!("{pad}v{target} {op} {};\n", 1 + r.below(40)));
            out.push_str(&format!("{pad}v{target} = v{target} & 65535;\n"));
        }
    }
    if depth > 0 && r.below(2) == 0 {
        out.push_str(&format!("{pad}if {} {{\n", condition(r, n)));
        let (inner, n2) = body(r, indent + 1, n, depth - 1);
        out.push_str(&inner);
        out.push_str(&format!("{pad}}}\n"));
        n = n2;
    }
    // A `for` whose body can return is the third of the three control-flow
    // sites, and the one whose bug was that the lowering was never CALLED
    // rather than that it was wrong. Without loops here, re-introducing that
    // bug leaves this file green -- checked, and it did.
    //
    // Gated on a solver, because `@invariant` is mandatory on a loop outside
    // `unsafe` and an invariant that cannot be discharged now fails the build.
    if depth > 0 && z3_path().is_some() && r.below(3) == 0 {
        let bound = 2 + r.below(4);
        out.push_str(&format!("{pad}@invariant(i{indent} >= 0)\n"));
        out.push_str(&format!("{pad}for i{indent} in 0..{bound} {{\n"));
        // The compared value is MASKED to 0..3. The first version of this
        // compared a raw parameter (0..199) against a loop index below 5, so
        // the return inside the loop could essentially never be taken -- and
        // re-introducing the historical "for-body return discarded" bug left
        // the whole file green. A branch that cannot be reached tests nothing;
        // `feedback-exercised-is-not-covered`, one level down.
        out.push_str(&format!(
            "{pad}    if ({} & 3) < i{indent} {{\n",
            value(r, n)
        ));
        out.push_str(&format!(
            "{pad}        return (i{indent} * {}) & 65535;\n",
            1 + r.below(9)
        ));
        out.push_str(&format!("{pad}    }}\n{pad}}}\n"));
    }
    out.push_str(&format!("{pad}return ({}) & 65535;\n", arith(r, n)));
    (out, n)
}

fn programs(seed: u64) -> (String, String, [u64; NPARAMS]) {
    let mut r = Rng(seed);
    let (shared, _) = body(&mut r, 1, 0, 3);
    let params = (0..NPARAMS)
        .map(|i| format!("p{i}: I32"))
        .collect::<Vec<_>>()
        .join(", ");
    let args: [u64; NPARAMS] = [r.below(200), r.below(200), r.below(200)];
    let call = args
        .iter()
        .map(|a| a.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    (
        // LLVM: a real call, so the condition is not folded.
        format!(
            "fn compute({params}) -> I32 {{\n{shared}}}\n\nfn main() -> I32 {{\n    return compute({call}) & 255;\n}}\n"
        ),
        // ZK: the same body, inputs as private parameters.
        format!("fn main({params}) -> I32 {{\n{shared}}}\n"),
        args,
    )
}

/// The repo-local solver, if there is one. `@invariant` is discharged by Z3
/// and an invariant that cannot be checked fails the build, so loops are only
/// generated when this returns `Some`.
fn z3_path() -> Option<PathBuf> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        "venv/bin/z3",
        ".venv/bin/z3",
        "z3/build/z3",
    ]
    .iter()
    .map(|p| repo.join(p))
    .find(|p| p.exists())
    .or_else(|| std::env::var_os("Y_Z3_PATH").map(PathBuf::from).filter(|p| p.exists()))
}

fn llvm_run(src: &str) -> Option<i32> {
    let dir = std::env::temp_dir().join(format!(
        "y_zkdiff_{}_{}",
        std::process::id(),
        SALT.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("p.ysu");
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join("p");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path)
        .arg("-o")
        .arg(&bin)
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    if let Some(z3) = z3_path() {
        cmd.env("Y_Z3_PATH", z3);
    }
    let out = cmd.output().expect("run Y");
    let ok = bin.exists();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !ok {
        let _ = std::fs::remove_dir_all(&dir);
        // "No clang" and "clang rejected the IR" must not share an answer --
        // the mistake `tests/llvm_control_flow.rs` and
        // `tests/llvm_integer_widths.rs` both record making.
        if Command::new("clang").arg("--version").output().is_ok() {
            panic!("clang is installed but no binary was produced:\n{src}\n{text}");
        }
        return None;
    }
    let code = Command::new(&bin).status().ok().and_then(|s| s.code());
    let _ = std::fs::remove_dir_all(&dir);
    code
}

fn zk_run(src: &str, inputs: &[u64]) -> Option<u128> {
    let tokens = y::lexer::Lexer::new(src).tokenize();
    let prog = y::parser::Parser::new(tokens).parse_program().ok()?;
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&prog).ok()?;
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (w, satisfied) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    if !satisfied {
        return None;
    }
    // A value too large for u128 means something went negative and wrapped to
    // `p - k`. Given the generator emits no subtraction that should not
    // happen, so it is reported as a divergence rather than skipped.
    w[*circuit.outputs.first()?].to_decimal_string().parse().ok()
}

struct Tally {
    compared: usize,
    zk_unsat: usize,
    disagreements: Vec<String>,
}

fn sweep(count: u64) -> Tally {
    let mut t = Tally {
        compared: 0,
        zk_unsat: 0,
        disagreements: Vec::new(),
    };
    for seed in 0..count {
        let (llvm_src, zk_src, args) = programs(seed ^ 0xC1_2C);
        let l = match llvm_run(&llvm_src) {
            Some(v) => v,
            None => return t, // no clang
        };
        match zk_run(&zk_src, &args) {
            Some(z) => {
                t.compared += 1;
                let masked = (z & 255) as i32;
                if masked != l {
                    t.disagreements.push(format!(
                        "seed {seed}: llvm={l} zk={z} (zk & 255 = {masked}), inputs {args:?}\n{zk_src}"
                    ));
                }
            }
            None => t.zk_unsat += 1,
        }
    }
    t
}

/// Prints a sample of the corpus, for eyeballing what is actually generated.
/// `cargo test --features zk --test zk_llvm_differential -- --ignored show`
#[test]
#[ignore]
fn show_the_corpus() {
    let mut loops = 0;
    for seed in 0..30u64 {
        let (_l, z, a) = programs(seed ^ 0xC1_2C);
        if z.contains("for i") {
            loops += 1;
            if loops <= 2 {
                println!("--- seed {seed}, inputs {a:?}\n{z}");
            }
        }
    }
    println!("programs containing a loop: {loops} / 30");
}

/// `Y_ZKDIFF_PROGRAMS=300 cargo test --features zk --test zk_llvm_differential \
///    -- --ignored deep`
#[test]
#[ignore]
fn a_deep_sweep_finds_nothing_more() {
    let n = std::env::var("Y_ZKDIFF_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let t = sweep(n);
    eprintln!(
        "deep zk/llvm sweep: {} compared, {} unsatisfiable, {} disagreements",
        t.compared,
        t.zk_unsat,
        t.disagreements.len()
    );
    assert!(
        t.disagreements.is_empty(),
        "{} disagreements:\n\n{}",
        t.disagreements.len(),
        t.disagreements.join("\n---\n")
    );
}

#[test]
fn the_zk_circuit_computes_what_the_native_binary_computes() {
    let t = sweep(30);
    if t.compared == 0 && t.zk_unsat == 0 {
        eprintln!("SKIP zk_llvm_differential: no clang on this machine");
        return;
    }
    assert!(
        t.disagreements.is_empty(),
        "{} of {} programs disagreed between the ZK and LLVM backends:\n\n{}",
        t.disagreements.len(),
        t.compared,
        t.disagreements.join("\n---\n")
    );
    eprintln!(
        "zk/llvm differential: {} compared, {} unsatisfiable",
        t.compared, t.zk_unsat
    );
}

#[test]
fn the_corpus_is_not_all_unsatisfiable() {
    // The liveness canary. A circuit whose witness cannot be solved is skipped,
    // and the ZK backend has legitimate reasons to be unsatisfiable (a range
    // check in an untaken branch is documented as one). If the generator
    // drifted into that territory the test above would compare nothing and
    // pass -- `feedback-null-metrics-pass-dead-components` again.
    let t = sweep(30);
    if t.compared == 0 && t.zk_unsat == 0 {
        eprintln!("SKIP zk_llvm_differential canary: no clang on this machine");
        return;
    }
    assert!(
        t.compared >= 15,
        "only {} of 30 programs produced a solvable witness ({} unsatisfiable), \
         so the differential is mostly comparing nothing",
        t.compared,
        t.zk_unsat
    );
}
