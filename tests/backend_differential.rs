// ============================================================
//  LLVM vs `--emit-native`: the same program, two backends, one answer.
//
//  CLAUDE.md records that the ZK backend's `return` was not a terminator --
//  `if c { return x; } return y;` emitted a circuit computing `y`
//  unconditionally -- and notes that "the LLVM backend compiles the same
//  program to `ret i32 1`, so the two backends disagreed on what `return`
//  means; a cross-backend differential is the cheapest test that would have
//  caught it". No such test existed. This is it, for the two backends that
//  both produce a RUNNABLE artifact.
//
//  Scope is the intersection of what they support: straight-line integer
//  `let` bindings and the 16 integer binary operators. `native_emitter` has
//  no branches at all and refuses everything outside that subset, which is
//  what makes the intersection well-defined.
//
//  Two things about the comparison, stated because they bound what it proves:
//
//    * **A process exit status carries 8 bits.** So one program compares one
//      byte of a 32-bit answer. Each generated program therefore returns
//      `(expr >> s) & 255` for an `s` drawn from {0, 8, 16, 24}, spreading the
//      comparison across the whole word. This leans on `>>` and `&` agreeing
//      between the backends -- which is itself part of what is compared, and a
//      bug present identically in both is invisible to any differential.
//    * **Refusals are counted, not silently skipped.** `native_emitter`
//      refuses a great deal, and a differential where every case is skipped
//      passes perfectly while comparing nothing. `the_corpus_is_not_all_skips`
//      asserts a floor on how many programs were actually run on both sides.
// ============================================================

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Both tests in this file run a sweep, cargo runs them concurrently, and the
/// temp directory name used to be `y_bdiff_<pid>_<tag>` -- identical between
/// them. One sweep then deleted the directory the other was compiling into,
/// which showed up as an intermittent failure with no disagreement listed.
/// Exactly the `.ptx` race CLAUDE.md records for the GPU test harness.
static SALT: AtomicUsize = AtomicUsize::new(0);

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A cheap deterministic PRNG, so the corpus needs no dev-dependency and is
/// reproducible from its seed.
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

/// The 16 integer binary operators both backends claim to implement.
const OPS: [&str; 16] = [
    "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "<", "<=", ">", ">=", "==", "!=",
];

/// Builds one straight-line integer expression over the locals `v0..vn`.
fn expr(r: &mut Rng, upto: u64, depth: u32) -> String {
    if depth == 0 || r.below(3) == 0 {
        return if r.below(2) == 0 && upto > 0 {
            format!("v{}", r.below(upto))
        } else {
            format!("{}", 1 + r.below(90))
        };
    }
    let op = OPS[r.below(OPS.len() as u64) as usize];
    let lhs = expr(r, upto, depth - 1);
    // A zero divisor traps rather than disagreeing, and a shift wider than the
    // type is undefined in LLVM and merely wraps on x86 -- a real difference,
    // but one about UB rather than about this compiler. Both are pinned to a
    // safe literal, which is also why the right operand is not recursive here.
    let rhs = match op {
        "/" | "%" => format!("{}", 1 + r.below(20)),
        "<<" | ">>" => format!("{}", r.below(8)),
        _ => expr(r, upto, depth - 1),
    };
    format!("({lhs} {op} {rhs})")
}

/// Builds one straight-line integer program.
///
/// Values are kept small on purpose. The point is to compare the two
/// backends' *operators*, and overflow behaviour is a separate question that
/// `tests/llvm_integer_widths.rs` and `native_emitter`'s own width refusal
/// already cover -- letting a product run past `i32` here would produce
/// disagreements that say nothing about the operator.
fn program(seed: u64) -> (String, u32) {
    let mut r = Rng(seed);

    // Zero to two helper functions, each taking up to the six integer
    // parameters System V passes in registers. `native_emitter`'s recorded bug
    // was that "a function with parameters read stack garbage; nothing stored
    // rdi/rsi/... anywhere", so the call boundary is the part of its subset
    // most worth differentiating -- and the generator could not write a call
    // at all until now.
    let nfuncs = r.below(3) as usize;
    let mut prelude = String::new();
    let mut arities = Vec::new();
    for f in 0..nfuncs {
        let arity = 1 + r.below(6) as usize;
        arities.push(arity);
        let params = (0..arity)
            .map(|i| format!("p{i}: I32"))
            .collect::<Vec<_>>()
            .join(", ");
        // The body sees its parameters as `p0..`, which `expr` names `v0..`.
        let body = expr(&mut r, arity as u64, 2).replace('v', "p");
        prelude.push_str(&format!("fn f{f}({params}) -> I32 {{\n    return {body};\n}}\n\n"));
    }

    let nlocals = 2 + r.below(3) as usize;
    let mut body = String::new();
    for i in 0..nlocals {
        if i == 0 {
            body.push_str(&format!("    let v0: I32 = {};\n", 1 + r.below(90)));
        } else if !arities.is_empty() && r.below(3) == 0 {
            let f = r.below(arities.len() as u64) as usize;
            let args = (0..arities[f])
                .map(|_| expr(&mut r, i as u64, 1))
                .collect::<Vec<_>>()
                .join(", ");
            body.push_str(&format!("    let v{i}: I32 = f{f}({args});\n"));
        } else {
            body.push_str(&format!("    let v{i}: I32 = {};\n", expr(&mut r, i as u64, 2)));
        }
    }

    // A process exit status carries 8 bits, so one program compares one byte
    // of a 32-bit answer; the shift spreads the comparison over the word.
    let shift = [0u32, 8, 16, 24][r.below(4) as usize];
    body.push_str(&format!(
        "    let out: I32 = (v{} >> {shift}) & 255;\n    return out;\n",
        nlocals - 1
    ));
    (
        format!("{prelude}fn main() -> I32 {{\n{body}}}\n"),
        shift,
    )
}

enum Outcome {
    Ran(i32),
    Refused(String),
    NoToolchain,
}

fn build_and_run(tag: &str, src: &str, native: bool) -> Outcome {
    let dir = std::env::temp_dir().join(format!(
        "y_bdiff_{}_{}_{}",
        std::process::id(),
        SALT.fetch_add(1, Ordering::SeqCst),
        tag
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("p.ysu");
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join("p");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path).current_dir(repo());
    if native {
        cmd.arg("--emit-native").arg(format!("--output={}", bin.display()));
    } else {
        cmd.arg("-o").arg(&bin);
    }
    let out = cmd.output().expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    if !bin.exists() {
        let _ = std::fs::remove_dir_all(&dir);
        // Only the LLVM path depends on an external toolchain.
        if !native && Command::new("clang").arg("--version").output().is_err() {
            return Outcome::NoToolchain;
        }
        return Outcome::Refused(text);
    }
    let status = Command::new(&bin).status().expect("run the built binary");
    let _ = std::fs::remove_dir_all(&dir);
    match status.code() {
        Some(c) => Outcome::Ran(c),
        None => Outcome::Refused(format!("{tag} did not exit normally")),
    }
}

struct Tally {
    compared: usize,
    native_refused: usize,
    disagreements: Vec<String>,
}

fn sweep(count: u64) -> Tally {
    let mut t = Tally {
        compared: 0,
        native_refused: 0,
        disagreements: Vec::new(),
    };
    for seed in 0..count {
        let (src, _shift) = program(seed ^ 0x5eed);
        let llvm = build_and_run(&format!("llvm_{seed}"), &src, false);
        let l = match llvm {
            Outcome::Ran(c) => c,
            Outcome::NoToolchain => return t,
            Outcome::Refused(why) => panic!(
                "the LLVM backend refused a program inside its own subset \
                 (seed {seed}):\n{src}\n{why}"
            ),
        };
        match build_and_run(&format!("nat_{seed}"), &src, true) {
            Outcome::Ran(n) => {
                t.compared += 1;
                if n != l {
                    t.disagreements
                        .push(format!("seed {seed}: llvm={l} native={n}\n{src}"));
                }
            }
            // A refusal is the native backend's documented answer for anything
            // outside its subset, and it names what it refused. It is not a
            // disagreement -- but it is counted, so that "everything was
            // refused" cannot look like "everything agreed".
            Outcome::Refused(_) => t.native_refused += 1,
            Outcome::NoToolchain => unreachable!("the native backend links nothing"),
        }
    }
    t
}

/// A deep sweep, off by default because it costs two compiles per program.
/// `cargo test --test backend_differential -- --ignored --nocapture`
#[test]
#[ignore]
fn a_deep_sweep_finds_nothing_more() {
    // `Y_DIFF_PROGRAMS=5000` to push it further. Each program costs two
    // compiles and two runs, so ~11 per second here.
    let n = std::env::var("Y_DIFF_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let t = sweep(n);
    eprintln!(
        "deep sweep: {} compared, {} refused, {} disagreements",
        t.compared,
        t.native_refused,
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
fn the_two_runnable_backends_compute_the_same_answer() {
    let t = sweep(40);
    if t.compared == 0 && t.native_refused == 0 {
        eprintln!("SKIP backend_differential: no clang on this machine");
        return;
    }
    assert!(
        t.disagreements.is_empty(),
        "{} of {} programs disagreed between the LLVM and native backends:\n\n{}",
        t.disagreements.len(),
        t.compared,
        t.disagreements.join("\n---\n")
    );
    eprintln!(
        "backend differential: {} compared, {} refused by --emit-native",
        t.compared, t.native_refused
    );
}

#[test]
fn the_corpus_is_not_all_skips() {
    // The liveness canary. `native_emitter` refuses everything outside a
    // straight-line integer subset, so a generator that drifted out of that
    // subset would leave the test above comparing NOTHING and passing -- the
    // shape recorded in `feedback-null-metrics-pass-dead-components`, where a
    // "count of bad things == 0" metric is scored perfectly by a component
    // that never ran.
    let t = sweep(40);
    if t.compared == 0 && t.native_refused == 0 {
        eprintln!("SKIP backend_differential canary: no clang on this machine");
        return;
    }
    assert!(
        t.compared >= 20,
        "only {} of 40 generated programs ran on BOTH backends ({} refused by \
         --emit-native), so the differential is mostly comparing nothing",
        t.compared,
        t.native_refused
    );
}
