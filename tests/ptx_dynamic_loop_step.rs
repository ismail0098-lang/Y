//! `for i in a..b step <runtime value>` must step by that value.
//!
//! It used to step by **one**. `Stmt::For` read the step with
//!
//! ```ignore
//! match step {
//!     Some(Expr::IntLit(step, _)) if *step > 0 => *step as u32,
//!     _ => 1,
//! }
//! ```
//!
//! so any step the arm could not read fell through to a plausible default
//! rather than being refused or emitted. That is the exact shape of every row
//! in `CLAUDE.md`'s design-rule table, and it lands on the single most common
//! GPU idiom there is: the **grid-stride loop**,
//! `for i in worker..count step nworkers`, which is how a kernel is written when
//! its launch geometry is a tuning parameter rather than part of its meaning.
//!
//! Stepping by 1 there is not a slowdown, it is a wrong answer: every thread
//! walks the entire range, so a reduction counts each element once per thread.
//! It compiled cleanly and printed "Compilation Successful!".
//!
//! Found while building the deterministic reduction of
//! `docs/deterministic_inference.md` M4 step 2 — which is a grid-stride loop
//! by construction, since the whole claim is that the answer does not depend
//! on the partition.

use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn compile(name: &str, src: &str) -> (bool, String, String) {
    // A fresh directory holding a PINNED profile. This wrote into the shared
    // `CARGO_TARGET_TMPDIR` and compiled with `current_dir` at the repository,
    // i.e. for this machine's card.
    let dir = pinned::pinned_scratch(name, pinned::SM_PINNED);
    let ysu = dir.join(format!("{name}.ysu"));
    std::fs::write(&ysu, src).unwrap();

    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let out = Command::new(bin.join("Y"))
        .arg(&ysu)
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join(format!("{name}.ptx"))).unwrap_or_default();
    (out.status.success() && log.contains("Compilation Successful"), ptx, log)
}

const GRID_STRIDE: &str = r#"
kernel gs(Src: GlobalMemory<I32>, Out: GlobalMemory<I32>, N: I32) {
    // Bound the runtime trip count so stepping by up to 1024 cannot wrap.
    @bounds(min=0, max=1024) let count: I32 = N;
    let worker: I32 = thread_idx_x();
    let nworkers: I32 = block_dim_x();
    @invariant(i >= 0)
    for i in worker..count step nworkers {
        Out[i] = i;
    }
}
"#;

/// The increment after the loop body must be a REGISTER, not the literal 1.
/// Use a single-block launch: unrestricted grid dimensions can overflow I32,
/// so the former global-worker calculation did not establish `i >= 0`.
#[test]
fn a_runtime_step_is_emitted_rather_than_silently_becoming_one() {
    let (ok, ptx, log) = compile("dyn_step", GRID_STRIDE);
    assert!(ok, "the grid-stride kernel did not compile:\n{log}");

    let incr: Vec<&str> = ptx
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("add.u32") && {
            // `add.u32 d, a, b` where d and a are the same register: the loop
            // latch.
            let ops: Vec<&str> = l
                .trim_end_matches(';')
                .split_whitespace()
                .skip(1)
                .map(|t| t.trim_end_matches(','))
                .collect();
            ops.len() == 3 && ops[0] == ops[1]
        })
        .collect();
    assert!(
        !incr.is_empty(),
        "no loop-latch increment found in the emitted PTX:\n{ptx}"
    );
    for line in &incr {
        let rhs = line
            .trim_end_matches(';')
            .split(',')
            .next_back()
            .unwrap()
            .trim();
        assert!(
            rhs.starts_with('%'),
            "the loop steps by the literal `{rhs}`, not by its runtime step. A \
             grid-stride loop that steps by 1 walks the whole range on every \
             thread — it compiles clean and computes the wrong answer.\n{line}"
        );
    }
}

/// A literal step must still be folded, or the fix above would have cost the
/// vectorising pass (which keys off `step == 4`) and every existing kernel.
#[test]
fn a_literal_step_is_still_a_constant_and_still_vectorises() {
    let src = r#"
kernel lit(Out: GlobalMemory<F32>, N: I32) {
    @bounds(min=0, max=1024) let count: I32 = N;
    @invariant(i >= 0)
    for i in 0..count step 4 {
        Out[i] = 1.0;
    }
}
"#;
    let (ok, ptx, log) = compile("lit_step", src);
    assert!(ok, "the literal-step kernel did not compile:\n{log}");
    assert!(
        ptx.contains("add.u32") && ptx.lines().any(|l| l.trim_end_matches(';').ends_with(", 4")),
        "a literal step of 4 is no longer emitted as a constant:\n{ptx}"
    );
    assert!(
        ptx.contains("VECTORIZING PASS"),
        "the step-4 vectorising pass stopped firing, so the dynamic-step fix \
         regressed the literal path:\n{ptx}"
    );
}

/// A step that cannot terminate is refused rather than quietly turned into 1.
#[test]
fn a_non_positive_literal_step_is_refused() {
    for bad in ["0", "0 - 2"] {
        let src = format!(
            r#"
kernel bad(Out: GlobalMemory<I32>, N: I32) {{
    @invariant(i >= 0)
    for i in 0..8 step {bad} {{
        Out[i] = i;
    }}
}}
"#
        );
        let (ok, _, log) = compile("bad_step", &src);
        assert!(
            !ok,
            "a non-positive step `{bad}` compiled successfully:\n{log}"
        );
    }
}
