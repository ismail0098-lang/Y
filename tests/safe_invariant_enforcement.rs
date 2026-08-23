//! `@invariant` inside `@safe` is a proof obligation, and an unverified one
//! must fail the build.
//!
//! It did not used to. When the SMT solver could not be spawned, the type
//! checker printed `[Warning] SMT Solver execution failed` to stdout and
//! continued, so on any machine without z3 - the default - every invariant in
//! every `@safe` block was accepted unchecked. This compiled cleanly and
//! reported "Compilation Successful!":
//!
//! ```text
//! @safe {
//!     @invariant(i > 1000)     // never true for i in 0..10
//!     for i in 0..10 { ... }
//! }
//! ```
//!
//! Two things made it hard to notice. The message went to stdout among a lot of
//! other build chatter, and the search path for the solver was three entries
//! long - so a machine with a perfectly good `venv/bin/z3` (this repo has one)
//! reported it as missing and waved everything through.
//!
//! These tests drive the real `Y` binary as a subprocess rather than calling
//! `TypeChecker` directly, because the behaviour under test depends on the
//! environment and mutating `std::env` from a test races every other test in
//! the process.
//!
//! Run with:  cargo test --test safe_invariant_enforcement

use std::path::PathBuf;
use std::process::Command;

/// A `@safe` loop whose invariant is false: `i` ranges over `0..10`.
const FALSE_INVARIANT: &str = r#"
fn main() {
    @safe {
        let x: I32 = 0;
        @invariant(i > 1000)
        for i in 0..10 {
            print_int(i);
        }
    }
}
"#;

/// The same shape, with an invariant that actually holds.
const TRUE_INVARIANT: &str = r#"
fn main() {
    @safe {
        let x: I32 = 0;
        @invariant(i >= 0)
        for i in 0..10 {
            print_int(i);
        }
    }
}
"#;

fn write_source(name: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("y_safe_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, body).expect("write source");
    path
}

/// Runs the compiler on `src`. `solver_visible` false strips the environment
/// the solver could be found through.
/// These tests assert on "Front-end analysis complete.", NOT on "Compilation
/// Successful!".
///
/// The two used to be the same string: the success banner was printed before
/// the backend dispatch ran, so it meant "the front end accepted this". It now
/// means what it says - the selected backend produced its artifact - and these
/// programs deliberately do not get that far. They call undeclared stubs, use
/// `match` (which the PTX backend refuses by name), and are compiled without a
/// linkable runtime, because the property under test is entirely a front-end
/// one.
///
/// The negative assertions got STRONGER in the same move: `!contains(banner)`
/// used to be satisfied by a program the front end ACCEPTED and a backend then
/// refused, which would have read as "the front end rejected it".
fn compile(src: &PathBuf, solver_visible: bool, allow_unverified: bool) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(src);
    cmd.env_remove("Y_Z3_PATH");
    cmd.env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS");
    if !solver_visible {
        // No `z3` on PATH, and no `$HOME/.local/bin/z3`. The relative
        // candidates resolve against the working directory, so run somewhere
        // that has no `venv/` or `z3/` in it.
        cmd.env("PATH", "/nonexistent-path");
        cmd.env("HOME", "/nonexistent-home");
        cmd.current_dir(src.parent().expect("temp dir"));
    }
    if allow_unverified {
        cmd.env("Y_ALLOW_UNVERIFIED_INVARIANTS", "1");
    }
    let out = cmd.output().expect("run Y");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Without a solver, an invariant cannot be discharged - so the build fails.
#[test]
fn unverifiable_invariant_fails_the_build() {
    let src = write_source("false_inv", FALSE_INVARIANT);
    let out = compile(&src, false, false);
    assert!(
        out.contains("Could not verify invariant"),
        "a @safe invariant that could not be checked must be an error, not a warning.\n{}",
        out
    );
    assert!(
        !out.contains("Front-end analysis complete"),
        "compilation must not succeed with an undischarged proof obligation.\n{}",
        out
    );
    // The message has to tell the user how to fix it, both ways.
    assert!(out.contains("Y_Z3_PATH"), "error should mention Y_Z3_PATH:\n{}", out);
    assert!(
        out.contains("Y_ALLOW_UNVERIFIED_INVARIANTS"),
        "error should name the opt-out:\n{}",
        out
    );
}

/// The escape hatch restores the old behaviour, loudly.
#[test]
fn opt_out_downgrades_to_a_warning() {
    let src = write_source("optout", FALSE_INVARIANT);
    let out = compile(&src, false, true);
    assert!(
        out.contains("was NOT verified"),
        "opt-out should still say plainly that nothing was proved.\n{}",
        out
    );
    assert!(
        !out.contains("Could not verify invariant"),
        "opt-out should not also raise the error.\n{}",
        out
    );
}

/// With a solver available, the checking is real in both directions.
///
/// Skipped when no z3 can be found, in the same spirit as the `ptxas` gate on
/// the PTX emitter - but note this test is the only one that proves the
/// verification does anything at all, so a CI without z3 is CI that is not
/// testing `@safe`'s central claim.
#[test]
fn with_a_solver_false_invariants_are_caught_and_true_ones_pass() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let z3 = ["venv/bin/z3", ".venv/bin/z3", "z3/build/z3"]
        .iter()
        .map(|p| repo.join(p))
        .find(|p| p.exists())
        .or_else(|| {
            std::env::var("PATH").ok().and_then(|paths| {
                std::env::split_paths(&paths)
                    .map(|d| d.join("z3"))
                    .find(|p| p.exists())
            })
        });
    let Some(z3) = z3 else {
        eprintln!("skipping: no z3 binary found");
        return;
    };

    let bad = write_source("smt_false", FALSE_INVARIANT);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&bad)
        .env("Y_Z3_PATH", &z3)
        .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
        .output()
        .expect("run Y");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.contains("SMT Safety Verification Failed"),
        "z3 must reject `i > 1000` on a 0..10 loop.\n{}",
        out
    );

    let good = write_source("smt_true", TRUE_INVARIANT);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&good)
        .env("Y_Z3_PATH", &z3)
        .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
        .output()
        .expect("run Y");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.contains("SMT Safety Verification Failed"),
        "`i >= 0` holds on a 0..10 loop and must be accepted.\n{}",
        out
    );
}

// ─────────────────────── soundness of the SMT encoding ───────────────────────
//
// The tests above establish that an invariant is checked at all. These
// establish that the thing being checked is the actual program.
//
// The verifier translates the loop body into SMT-LIB. Every construct it does
// not translate used to be skipped silently - `trace_body_statements` ended in
// `_ => {}`, and `expr_to_smt` ended in `_ => "0"` with `_ => "+"` for unknown
// operators. Dropping a statement's effects makes the preservation obligation
// strictly EASIER, so the failure was in the unsound direction: it accepted
// invariants that are false.

/// Runs `src` with a solver available and returns the compiler's output.
fn compile_with_solver(name: &str, src: &str) -> Option<String> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let z3 = ["venv/bin/z3", ".venv/bin/z3", "z3/build/z3"]
        .iter()
        .map(|p| repo.join(p))
        .find(|p| p.exists())
        .or_else(|| {
            std::env::var("PATH").ok().and_then(|paths| {
                std::env::split_paths(&paths).map(|d| d.join("z3")).find(|p| p.exists())
            })
        })?;
    let src_path = write_source(name, src);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src_path)
        .env("Y_Z3_PATH", &z3)
        .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
        .current_dir(&repo)
        .output()
        .expect("run Y");
    Some(format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

/// Wrapping a violation in a trivially-true `if` must not launder it.
///
/// This is the exact program that used to compile clean. `i = i - 100;` was
/// correctly rejected, and `if i >= 0 { i = i - 100; }` - the same violation -
/// reported "Compilation Successful!", because `Stmt::If` had no arm in the
/// body tracer and the assignment was never modelled.
///
/// Branches are now havoc'd: every variable a branch might assign gets a fresh
/// unconstrained value, which is a sound over-approximation and can only make
/// preservation harder to prove.
#[test]
fn nested_if_cannot_hide_a_false_invariant() {
    let hidden = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            if i >= 0 {\n                i = i - 100;\n            }\n        }\n    }\n}\n";
    let plain = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            i = i - 100;\n        }\n    }\n}\n";

    let Some(hidden_out) = compile_with_solver("hidden_violation", hidden) else {
        eprintln!("skipping: no z3 binary found");
        return;
    };
    let plain_out = compile_with_solver("plain_violation", plain).expect("z3 was found above");

    assert!(
        !plain_out.contains("Front-end analysis complete"),
        "the control must be rejected:\n{}",
        plain_out
    );
    assert!(
        !hidden_out.contains("Front-end analysis complete"),
        "a false invariant was accepted because the violating assignment sat inside an `if`. \
         The body tracer is skipping a construct instead of over-approximating it.\n{}",
        hidden_out
    );
}

/// A loop that only uses modelled constructs must still verify.
///
/// Rejecting everything unmodelled is sound but useless; this is the other half
/// of the requirement. A call with no reference argument cannot write a tracked
/// integer local, and a branch is havoc'd rather than refused, so ordinary loop
/// bodies still pass.
#[test]
fn ordinary_loop_bodies_still_verify() {
    let src = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            let y: I32 = 0;\n            if i >= 5 {\n                y = y + 1;\n            }\n            print_int(i);\n        }\n    }\n}\n";
    let Some(out) = compile_with_solver("ordinary_body", src) else {
        eprintln!("skipping: no z3 binary found");
        return;
    };
    assert!(
        out.contains("Front-end analysis complete"),
        "a true invariant over supported constructs must still verify:\n{}",
        out
    );
}

/// An operator with no sound encoding must be refused, not mistranslated.
///
/// `&`, `|`, `^`, `<<` and `>>` have no counterpart in the integer theory the
/// verifier uses. They used to fall through to `_ => "+"`, so `x & y` was
/// proven as `x + y` - a different function, not a coarser one.
#[test]
fn unencodable_operator_is_refused_not_mistranslated() {
    let src = "fn main() {\n    @safe {\n        @invariant((i & 1) >= 0)\n        for i in 0..10 {\n            print_int(i);\n        }\n    }\n}\n";
    let Some(out) = compile_with_solver("bitwise_invariant", src) else {
        eprintln!("skipping: no z3 binary found");
        return;
    };
    assert!(
        !out.contains("Front-end analysis complete"),
        "an invariant using an operator with no sound encoding must be refused:\n{}",
        out
    );
    assert!(
        out.contains("no sound encoding"),
        "the error should say why it cannot be encoded:\n{}",
        out
    );
}

// ── Loops whose bounds are variables ───────────────────────────────────────
//
// These were unverifiable, and the message blamed the wrong thing. A loop
// `for k in lo..hi` puts `lo_0` into the assertions, but `lo` is not one of
// the tracked variables the declaration pass walks, so z3 received an unknown
// constant, exited 1, and the compiler reported "the SMT solver could not be
// run". The solver ran fine; it was handed a malformed query. Every existing
// test used literal bounds, which is why none of them saw it.

/// A bound the verifier genuinely knows NOTHING about. `Lo` is a kernel
/// parameter, so it may be any i32 including a negative one.
///
/// This used to be `block_idx_x()`, which stopped being unconstrained when the
/// interval domain learned the GPU index intrinsics' hardware ranges (see
/// `gpu_index_interval`). `k >= 0` became provable there — correctly, since
/// `%ctaid.x` really is non-negative — which would have quietly turned this
/// test into a no-op. The property it exists to protect is unchanged: an
/// unknown bound must not be *assumed* non-negative.
const UNKNOWN_BOUNDS: &str = r#"
kernel probe(A: GlobalMemory<F32>, N: I32, Lo: I32) {
    let lo: I32 = Lo;
    let hi: I32 = lo + 8;
    @invariant(k >= lo)
    for k in lo..hi {
        let v: F32 = block_ptr2d_load(A, 0, k, N, 1, N);
        block_ptr2d_store(A, 0, k, N, 1, N, v);
    }
}
fn main() {}
"#;

const VAR_BOUNDS: &str = r#"
kernel probe(A: GlobalMemory<F32>, N: I32) {
    let lo: I32 = block_idx_x();
    let hi: I32 = lo + 8;
    @invariant(k >= lo)
    for k in lo..hi {
        let v: F32 = block_ptr2d_load(A, 0, k, N, 1, N);
        block_ptr2d_store(A, 0, k, N, 1, N, v);
    }
}
fn main() {}
"#;

#[test]
fn a_loop_with_variable_bounds_can_be_verified() {
    let src = write_source("var_bounds_ok", VAR_BOUNDS);
    let out = compile(&src, true, false);
    assert!(
        !out.contains("could not be run"),
        "a variable-bounded loop still reports the solver as unrunnable:\n{}",
        out
    );
    assert!(
        out.contains("Front-end analysis complete"),
        "a true invariant on a variable-bounded loop was not accepted:\n{}",
        out
    );
}

/// The control, and it is the one that matters: `declare_free_symbols` gives an
/// undeclared symbol an UNCONSTRAINED value, which must make obligations
/// harder, never easier. If it had instead made them vacuously true, this
/// false invariant would now pass.
#[test]
fn a_false_invariant_on_a_variable_bounded_loop_is_still_rejected() {
    let src = write_source(
        "var_bounds_false",
        &VAR_BOUNDS.replace("@invariant(k >= lo)", "@invariant(k > 1000)"),
    );
    let out = compile(&src, true, false);
    assert!(
        out.contains("Verification Failed") || out.contains("may not hold"),
        "a false invariant on a variable-bounded loop was accepted:\n{}",
        out
    );
    assert!(
        !out.contains("Front-end analysis complete"),
        "the build succeeded despite a false invariant:\n{}",
        out
    );
}

/// An unconstrained bound must NOT be assumed non-negative. `k >= 0` on
/// `for k in lo..hi` with `lo` unknown is genuinely unprovable, and saying so
/// is correct behaviour rather than a gap - it is what caught the real
/// modelling limit in the MSM kernel.
#[test]
fn an_unprovable_invariant_on_an_unknown_bound_is_reported_as_such() {
    let src = write_source(
        "var_bounds_unknown",
        &UNKNOWN_BOUNDS.replace("@invariant(k >= lo)", "@invariant(k >= 0)"),
    );
    let out = compile(&src, true, false);
    assert!(
        !out.contains("could not be run"),
        "this must be a verification result, not a solver failure:\n{}",
        out
    );
    assert!(
        out.contains("may not hold") || out.contains("Verification Failed"),
        "an unprovable invariant was accepted:\n{}",
        out
    );
}


/// The flip side of the test above: a bound derived from a GPU index intrinsic
/// IS non-negative, and the verifier now knows it.
///
/// Without this, `for i in worker..N step nworkers` — the grid-stride loop,
/// which is how every kernel in `docs/deterministic_inference.md` is written —
/// could not be given any invariant at all, because `worker` was a havoc and
/// even `i >= 0` was refutable. The facts asserted are CUDA launch limits
/// (`gpu_index_interval`), and they are the only place in the type checker
/// that makes an obligation easier rather than harder.
#[test]
fn a_bound_from_a_gpu_index_intrinsic_is_known_non_negative() {
    let src = write_source(
        "var_bounds_intrinsic",
        &VAR_BOUNDS.replace("@invariant(k >= lo)", "@invariant(k >= 0)"),
    );
    let out = compile(&src, true, false);
    if out.contains("could not be run") {
        eprintln!("SKIP: no z3 on this machine");
        return;
    }
    assert!(
        out.contains("Front-end analysis complete"),
        "`k >= 0` on `for k in block_idx_x()..hi` was not provable, so a \
         grid-stride loop still cannot carry an invariant:\n{}",
        out
    );
}

/// ...and the intrinsic's range must not be over-claimed. `%ctaid.x` is
/// non-negative, but nothing says it is non-ZERO, so a strict `> 0` must still
/// be refuted.
#[test]
fn the_intrinsic_range_is_not_over_claimed() {
    let src = write_source(
        "var_bounds_strict",
        &VAR_BOUNDS.replace("@invariant(k >= lo)", "@invariant(k > 0)"),
    );
    let out = compile(&src, true, false);
    if out.contains("could not be run") {
        eprintln!("SKIP: no z3 on this machine");
        return;
    }
    assert!(
        !out.contains("Front-end analysis complete"),
        "`k > 0` was accepted on a loop starting at `block_idx_x()`, which is \
         zero on the first CTA of every launch. The interval bounds are too \
         strong:\n{}",
        out
    );
}

// ── A reference handed to a callee, anywhere in the body ────────────────
//
// The SMT model rests on exactly one assumption about calls: Y has no way for
// a callee to write a caller's local integer scalar unless the caller hands
// over a reference. `takes_reference` enforces it - and used to be consulted
// at ONE site, the top-level `Stmt::Expr` arm of `trace_body_statements`. The
// assumption is about the whole loop body, so everywhere else the reference
// was invisible and the invariant was "verified" against a variable the callee
// was free to overwrite.
//
// This is the same one-level-deep shape as `nested_if_cannot_hide_a_false_
// invariant` above: the construct is refused when written plainly and was
// accepted the moment it sat inside anything.
//
// Each case asserts on its OWN diagnostic rather than merely on the absence of
// success, so a fixture that is stopped by some earlier pass fails instead of
// passing for the wrong reason. The positive controls below are what prove
// the fixtures reach this check at all.

/// `bump(&i);` at the top of the loop body, the one spelling that was already
/// refused. The control for everything under it.
const REF_PLAIN: &str = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            bump(&i);\n        }\n    }\n}\n";

fn assert_reference_refused(name: &str, src: &str, site: &str) {
    let Some(out) = compile_with_solver(name, src) else {
        eprintln!("skipping: no z3 binary found");
        return;
    };
    assert!(
        out.contains("passes a reference to a call"),
        "a reference handed to a callee {} was not refused. The invariant was \
         discharged against a variable the callee could have overwritten.\n{}",
        site,
        out
    );
}

#[test]
fn a_reference_at_the_top_of_the_body_is_refused() {
    assert_reference_refused("ref_plain", REF_PLAIN, "at the top of the loop body");
}

#[test]
fn a_reference_inside_a_branch_is_refused() {
    let src = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            if i >= 0 {\n                bump(&i);\n            }\n        }\n    }\n}\n";
    assert_reference_refused("ref_in_if", src, "inside an `if`");
}

#[test]
fn a_reference_in_an_assignment_value_is_refused() {
    let src = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            let y: I32 = 0;\n            y = bump(&i);\n        }\n    }\n}\n";
    assert_reference_refused("ref_in_assign", src, "in an assignment's right-hand side");
}

#[test]
fn a_reference_in_a_let_initialiser_is_refused() {
    let src = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            let y: I32 = bump(&i);\n        }\n    }\n}\n";
    assert_reference_refused("ref_in_let", src, "in a `let` initialiser");
}

#[test]
fn a_reference_in_a_match_arm_is_refused() {
    let src = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            if i >= 0 {\n                match i {\n                    _ => bump(&i)\n                }\n            }\n        }\n    }\n}\n";
    assert_reference_refused("ref_in_match", src, "in a `match` arm");
}

/// Mutation-verified as a CONFIRMATION, not as a guard on the outer walk.
///
/// Deleting `Stmt::For`'s body traversal from `stmts_take_reference` leaves
/// this test green, and that is correct: `check_stmt` requires an `@invariant`
/// on every loop outside an `unsafe` context, so the inner loop is itself
/// verified and runs its own reference check first. The outer arm is kept
/// anyway - it is not sound for this check to depend on a rule enforced by a
/// different pass - but do not read this test as pinning it.
#[test]
fn a_reference_in_a_nested_loop_is_refused() {
    let src = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            @invariant(j >= 0)\n            for j in 0..2 {\n                bump(&i);\n            }\n        }\n    }\n}\n";
    assert_reference_refused("ref_in_nested_loop", src, "in a nested loop's body");
}

/// The controls, and they carry as much weight as the six cases above.
///
/// Refusing every loop body that contains a call would pass all six and be
/// useless - `print_int(i)` is the ordinary shape. Each of these is the
/// corresponding negative case with the `&` removed, so it also proves the
/// fixture reaches the reference check rather than being stopped by an earlier
/// pass. The `match` control matters most: if `match` were refused inside
/// `@safe` for some unrelated reason, that negative case would be vacuous.
#[test]
fn calls_that_hand_out_no_reference_still_verify() {
    let cases = [
        ("ok_ref_plain", "bump(i);", ""),
        ("ok_ref_in_if", "if i >= 0 {\n                print_int(i);\n            }", ""),
        ("ok_ref_in_assign", "let y: I32 = 0;\n            y = bump(i);", ""),
        ("ok_ref_in_let", "let y: I32 = bump(i);", ""),
        (
            "ok_ref_in_match",
            "if i >= 0 {\n                match i {\n                    _ => print_int(i)\n                }\n            }",
            "",
        ),
        (
            "ok_ref_in_nested_loop",
            "@invariant(j >= 0)\n            for j in 0..2 {\n                print_int(i);\n            }",
            "",
        ),
    ];
    for (name, body, _) in cases {
        let src = format!(
            "fn main() {{\n    @safe {{\n        @invariant(i >= 0)\n        for i in 0..10 {{\n            {}\n        }}\n    }}\n}}\n",
            body
        );
        let Some(out) = compile_with_solver(name, &src) else {
            eprintln!("skipping: no z3 binary found");
            return;
        };
        assert!(
            out.contains("Front-end analysis complete"),
            "`{}` hands out no reference and its invariant is true, so it must \
             still verify. Refusing every body containing a call would satisfy \
             every negative case in this section and check nothing.\n{}",
            name,
            out
        );
    }
}

/// A reference inside a struct literal inside a call argument.
///
/// `takes_reference` ended in `_ => false`, and `Expr::StructLit` fell into it -
/// so `bump(P { x: &i })` was a reference the check could not see even at the
/// one site where the check was run. It is exhaustive over `Expr` now.
#[test]
fn a_reference_inside_a_struct_literal_is_refused() {
    let src = "struct P { x: I32 }\nfn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            bump(P { x: &i });\n        }\n    }\n}\n";
    assert_reference_refused("ref_in_struct_lit", src, "inside a struct literal");
}

// ── The entry state is what the INITIATION obligation is about ──────────
//
// `check_stmt` clears the interval of every variable a loop body assigns
// before it verifies the loop, and it has to: `check_block` reasons about
// `@bounds` inside the body, where a range measured before the loop no longer
// holds, and so does the PRESERVATION obligation. But INITIATION is a
// statement about the state on ENTRY, where that range is still exactly true.
// Clearing it first made every useful invariant unprovable:
//
//     let acc: I32 = 0;
//     @invariant(acc >= 0)
//     for i in 0..4 { acc = acc + 1; }     // "initiation check failed"
//
// An invariant about a variable the body does not touch is trivial, so the
// only kind worth writing was the only kind that could not be stated. `while`
// was unusable outright -- its induction variable is always body-assigned --
// and `for` worked only for invariants over its own induction variable, whose
// range `verify_for_loop_invariant` re-derives from `start`/`end`.
// `tests/math.ysu` and `tests/safe_test.ysu` were both refused by this.

fn phase(name: &str, src: &str) -> Option<&'static str> {
    let out = compile_with_solver(name, src)?;
    Some(if out.contains("initiation check failed") {
        "initiation"
    } else if out.contains("preservation") {
        "preservation"
    } else if out.contains("Front-end analysis complete") {
        "accepted"
    } else {
        "other"
    })
}

/// A true invariant over a variable the body assigns must verify.
#[test]
fn an_invariant_over_a_body_assigned_variable_can_be_verified() {
    let cases = [
        (
            "entry_for",
            "fn main() {\n    let acc: I32 = 0;\n    @invariant(acc >= 0)\n    for i in 0..4 {\n        acc = acc + 1;\n    }\n}\n",
        ),
        (
            "entry_while",
            "fn main() {\n    let i: I32 = 0;\n    @invariant(i >= 0)\n    while i < 4 {\n        i = i + 1;\n    }\n}\n",
        ),
    ];
    for (name, src) in cases {
        let Some(p) = phase(name, src) else {
            eprintln!("skipping: no z3 binary found");
            return;
        };
        assert_eq!(
            p, "accepted",
            "`{}` states a true invariant about a variable its body assigns, \
             which is the only kind worth writing. It failed the {} check.",
            name, p
        );
    }
}

/// ...and the entry fact must be able to REFUTE one, not only support it.
/// This is what shows the snapshot actually reaches the initiation query
/// rather than the check having been weakened into always passing.
#[test]
fn an_invariant_false_on_entry_fails_the_initiation_check() {
    let src = "fn main() {\n    let acc: I32 = 0;\n    @invariant(acc > 5)\n    for i in 0..4 {\n        acc = acc + 1;\n    }\n}\n";
    let Some(p) = phase("entry_false", src) else {
        eprintln!("skipping: no z3 binary found");
        return;
    };
    assert_eq!(
        p, "initiation",
        "`acc > 5` is false at `acc = 0`, so the INITIATION check is what must \
         reject it. Getting {:?} instead means the entry state is not reaching \
         that query.",
        p
    );
}

/// The soundness half, and the reason this fix is narrow rather than "pass the
/// intervals to both queries".
///
/// A fact true on entry is NOT true after an iteration, so the snapshot must
/// not reach the preservation query. With `acc = 0` on entry:
///
///     @invariant(acc <= 1)
///     for i in 0..4 { acc = acc + 1; }
///
/// is FALSE — `acc` reaches 4. Preservation refutes it only because it treats
/// `acc` as unconstrained: pinned to its entry value 0, one body step gives 1,
/// and `1 <= 1` holds. Verified by mutation — leaking the snapshot into the
/// preservation query makes this exact program compile clean.
#[test]
fn the_entry_state_does_not_leak_into_the_preservation_check() {
    let src = "fn main() {\n    let acc: I32 = 0;\n    @invariant(acc <= 1)\n    for i in 0..4 {\n        acc = acc + 1;\n    }\n}\n";
    let Some(p) = phase("entry_leak", src) else {
        eprintln!("skipping: no z3 binary found");
        return;
    };
    assert_eq!(
        p, "preservation",
        "`acc <= 1` is false — `acc` reaches 4 — and only the PRESERVATION \
         check can see that. Getting {:?} means the entry interval is pinning \
         `acc` to 0 inside the loop, which proves false invariants.",
        p
    );
}

/// Two more in the same shape, so the fix is not pinned by a single program.
#[test]
fn body_violations_are_still_caught() {
    let cases = [
        (
            "entry_eq",
            "fn main() {\n    let acc: I32 = 0;\n    @invariant(acc == 0)\n    for i in 0..4 {\n        acc = acc + 1;\n    }\n}\n",
        ),
        (
            "entry_dec",
            "fn main() {\n    let acc: I32 = 0;\n    @invariant(acc >= 0)\n    for i in 0..4 {\n        acc = acc - 1;\n    }\n}\n",
        ),
        (
            "entry_while_dec",
            "fn main() {\n    let i: I32 = 0;\n    @invariant(i >= 0)\n    while i < 4 {\n        i = i - 1;\n    }\n}\n",
        ),
    ];
    for (name, src) in cases {
        let Some(p) = phase(name, src) else {
            eprintln!("skipping: no z3 binary found");
            return;
        };
        assert_eq!(
            p, "preservation",
            "`{}` states an invariant its body breaks; the preservation check \
             must reject it. Got {:?}.",
            name, p
        );
    }
}
