//! `PtxEmitter::emit_expr` ended in `_ => "".into()`, and an empty string is
//! spliced straight into instruction text by every caller.
//!
//! This is the design-rule table's commonest row, found again in the arm that
//! was left when its neighbours were fixed. CLAUDE.md already records the same
//! shape being repaired twice in this function - for an unknown CALL name, and
//! for a short intrinsic argument list - and both fixes named their own arm and
//! stopped there.
//!
//! Three defects, all reachable by typing a documented flag:
//!
//! 1. **`Expr::BoolLit` had no arm at all**, so `if true { .. }` was refused.
//!    That took out `cargo run -- tests/test_drift.ysu --emit-ptx`, a command
//!    CLAUDE.md documents and that a previous session had already repaired
//!    once. A bool is a u32 holding 0/1 - exactly what `Stmt::If` expects,
//!    since it lowers a condition as `setp.eq.u32 %p, <cond>, 0`.
//!
//! 2. **A string literal in an argument position emitted a missing operand.**
//!    `mul.lo.s32 %r5, , %r1;` and `setp.lt.u32 %p0, , %r2;`, under
//!    "Compilation Successful!" and exit 0. `ptxas` rejects the file with a
//!    parse error - on whatever machine tries to use it later.
//!
//! 3. **An unbound identifier returned ITSELF.** `let z: u32 = ZeroInit;`
//!    emitted `mov.u32 %r0, ZeroInit;`, which `ptxas` rejects with
//!    `Unknown symbol 'ZeroInit'`. Every name this backend defines is inserted
//!    into `variables` when it is bound, so a bare name reaching the emitter
//!    means the program named something that does not exist.
//!
//! Nothing in the repo could see 2 or 3: the compiler exits 0, so
//! `success_banner_means_success` agrees with itself, and the emitted file is
//! never assembled by the compile path. The corpus assemble sweep passes too,
//! because no committed program contains these shapes.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Outcome {
    ok: bool,
    output: String,
    ptx: Option<String>,
}

fn emit_ptx(name: &str, src: &str) -> Outcome {
    let dir = std::env::temp_dir().join(format!("y_exprcov_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("run Y");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    let ptx = std::fs::read_to_string(dir.join(format!("{}.ptx", name))).ok();
    Outcome {
        ok: out.status.success(),
        output: text,
        ptx,
    }
}

/// Every operand position in the emitted module must be filled. A bare `,` or a
/// trailing `,` before `;` is the signature of an empty `emit_expr` return, and
/// it is what `ptxas` reports as a parse error.
fn has_empty_operand(ptx: &str) -> bool {
    ptx.lines().map(str::trim).any(|l| {
        let code = l.split("//").next().unwrap_or("").trim();
        code.contains(", ,") || code.contains(",,") || code.ends_with(", ;") || code.contains(", ;")
    })
}

#[test]
fn a_boolean_literal_condition_lowers() {
    let r = emit_ptx(
        "boolcond",
        "kernel k(A: GlobalMemory<F32>) {\n    if true { let x: u32 = 1; }\n}\n",
    );
    assert!(
        r.ok,
        "`if true` must lower - a bool is a u32 holding 0/1, which is what the \
         `if` lowering already expects. Output:\n{}",
        r.output
    );
    let ptx = r.ptx.expect("a .ptx should have been written");
    assert!(
        !has_empty_operand(&ptx),
        "the emitted module has an empty operand:\n{ptx}"
    );
    // The EFFECT, not the vocabulary: the condition must actually reach a
    // `setp` against zero, which is how the branch is decided.
    assert!(
        ptx.contains("setp.eq.u32"),
        "expected the condition to be tested against zero:\n{ptx}"
    );
}

/// The documented command. CLAUDE.md carries an explicit note that this example
/// "did not compile for two more years' worth of reasons", and that a
/// documented command which does not run is a bug in the same class as a doc
/// describing an absent optimisation. It regressed again, so it is pinned here.
#[test]
fn the_documented_drift_example_still_compiles() {
    let src = repo().join("tests/test_drift.ysu");
    assert!(src.exists(), "tests/test_drift.ysu is missing");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "`Y tests/test_drift.ysu --emit-ptx` is a documented command and must \
         run. stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn a_string_literal_is_refused_rather_than_left_as_an_empty_operand() {
    let r = emit_ptx(
        "strlit",
        "kernel k(A: GlobalMemory<F32>) {\n    \
         let t: u32 = block_ptr2d_load(A, \"hi\", 0, 0, 1, 1, 1);\n}\n",
    );
    assert!(
        !r.ok,
        "a string literal has no PTX lowering and must be refused. It used to \
         emit `mul.lo.s32 %r5, , %r1;` under a success banner. Output:\n{}",
        r.output
    );
    assert!(
        r.output.contains("string literal"),
        "the refusal must name the construct, so the user does not have to \
         bisect their kernel. Output:\n{}",
        r.output
    );
    if let Some(ptx) = r.ptx {
        assert!(
            !has_empty_operand(&ptx),
            "a refused compile must not leave a module with empty operands:\n{ptx}"
        );
    }
}

#[test]
fn an_undefined_name_is_refused_rather_than_spliced_into_the_ptx() {
    let r = emit_ptx(
        "undefname",
        "kernel k(A: GlobalMemory<F32>) {\n    let z: u32 = NoSuchThing;\n}\n",
    );
    assert!(
        !r.ok,
        "an unbound name used to return itself, emitting \
         `mov.u32 %r0, NoSuchThing;` - which ptxas rejects with `Unknown \
         symbol`. Output:\n{}",
        r.output
    );
    assert!(
        r.output.contains("NoSuchThing"),
        "the refusal must name the identifier. Output:\n{}",
        r.output
    );
}

/// The control, and it is what stops "refuse everything" from passing this
/// file. Refusing every unlowerable expression is sound and useless if it also
/// refuses the programs that work - the same shape as
/// `ordinary_loop_bodies_still_verify`.
///
/// This asserts on the real corpus rather than a fixture, because the fixtures
/// above are all one-liners and could not detect a refusal that fires on a
/// realistic kernel.
#[test]
fn the_real_kernels_still_compile() {
    let kernels = [
        "tests/paged_decode_attention_128_32_8_16.ysu",
        "tests/bn254_fr_mul_fast.ysu",
        "tests/rope_64.ysu",
    ];
    let mut compiled = 0;
    for k in kernels {
        let src = repo().join(k);
        if !src.exists() {
            continue;
        }
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&src)
            .arg("--emit-ptx")
            .current_dir(repo())
            .output()
            .expect("run Y");
        assert!(
            out.status.success(),
            "{k} must still compile. stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        compiled += 1;
    }
    assert!(
        compiled >= 2,
        "the control compiled only {compiled} kernels; it is close to vacuous"
    );
}
