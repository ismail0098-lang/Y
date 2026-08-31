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

// ─────────────────────────── zero-initialisers ───────────────────────────
//
// `let x: T = {};` had NO lowering in this backend: it fell to `emit_expr`'s
// `_ => ""` and then to the "initialiser produced no value" refusal. The LLVM
// backend memsets it, `type_checker` types it, and `tests/bounds_test.ysu`
// uses it, so it is a real language construct rather than a corner.
//
// It is what `python/tests/test_gpu_architect_features.py` uses to declare a
// tile, which is why one of the two documented Python test commands had been
// failing. That suite lives in a directory `cargo test` never runs, so these
// three cases are the gate.

#[test]
fn a_block_tile_declaration_lowers_and_assembles() {
    let r = emit_ptx(
        "tiledecl",
        "kernel k(A: GlobalMemory<F32>, B: GlobalMemory<F32>) {\n    \
         let tile: BlockTile<F32, 128> = {};\n    \
         let val: F32 = block_tile_load(A, 10, 128);\n    \
         block_tile_store(B, 10, val, 128);\n}\n",
    );
    assert!(
        r.ok,
        "a `BlockTile` declaration is a DECLARATION - the tile intrinsics take \
         the buffer directly and never read the name. Output:\n{}",
        r.output
    );
    let ptx = r.ptx.expect("a .ptx should have been written");
    assert!(
        ptx.contains("BLOCK TILE LOAD"),
        "the tile machinery must still run:\n{ptx}"
    );
    assert!(
        !has_empty_operand(&ptx),
        "the emitted module has an empty operand:\n{ptx}"
    );
}

#[test]
fn a_scalar_zero_initialiser_is_actually_zeroed() {
    let r = emit_ptx(
        "scalarzero",
        "kernel k(A: GlobalMemory<F32>) {\n    \
         let n: u32 = {};\n    \
         let t: F32 = block_ptr2d_load(A, n, 0, 1, 1, 1);\n}\n",
    );
    assert!(r.ok, "`let n: u32 = {{}}` must lower. Output:\n{}", r.output);
    let ptx = r.ptx.expect("a .ptx should have been written");
    // The EFFECT: a register really is set to zero, not merely declared.
    assert!(
        ptx.lines()
            .any(|l| l.trim().starts_with("mov.u32") && l.trim().ends_with(", 0;")),
        "expected the scalar to be zeroed:\n{ptx}"
    );
}

/// The refusal that keeps the other two honest. This backend has no local
/// aggregate storage - a `let` binds ONE register - so zeroing "the array"
/// would zero a single scalar and call it the array.
#[test]
fn an_aggregate_zero_initialiser_is_refused_by_name() {
    let r = emit_ptx(
        "aggzero",
        "kernel k(A: GlobalMemory<F32>) {\n    let arr: [I32; 5] = {};\n}\n",
    );
    assert!(
        !r.ok,
        "an array local has no storage in this backend and must be refused. \
         Output:\n{}",
        r.output
    );
    assert!(
        r.output.contains("aggregate") || r.output.contains("no local array"),
        "the refusal must say why, not merely fail. Output:\n{}",
        r.output
    );
}

/// `Expr::MemberAccess` was the last arm in `emit_expr` still falling through
/// to `"".into()`.
///
/// The hole was closed for `Expr::Ident`, for `Expr::BoolLit` and for the `_`
/// arm, and never for this one - which is the whole reason the design-rule
/// table says to enumerate SITES rather than variants. `emit_expr`'s callers
/// splice the result straight into instruction text, so a field read in an
/// ARGUMENT position emitted
///
/// ```text
///     cvt.rn.f32.s32 %f1, ;
/// ```
///
/// under "Compilation Successful!" and exit 0, with `ptxas` rejecting the file
/// afterwards - on whatever machine tries to run it, which is not the one that
/// compiled it.
#[test]
fn a_struct_field_access_is_refused_rather_than_spliced_as_an_empty_operand() {
    let d = std::env::temp_dir().join(format!("y_ptx_member_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("temp dir");
    let src = d.join("ma.ysu");
    std::fs::write(
        &src,
        r#"
kernel k(A: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32) {
    let x: F32 = block_ptr2d_load(A, 0, 0, N, M, N);
    block_ptr2d_store(C, 0, 0, N, M, N, x.lane);
}
fn main() {}
"#,
    )
    .expect("write source");

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .output()
        .expect("run Y");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));

    assert!(
        !out.status.success(),
        "a field access this backend cannot lower must fail the build:\n{text}"
    );
    assert!(
        text.contains("x.lane"),
        "the refusal must name the field it could not lower:\n{text}"
    );
    assert!(
        !text.contains("Compilation Successful"),
        "the success banner appeared over a refusal:\n{text}"
    );
    assert!(
        !d.join("ma.ptx").exists(),
        "a .ptx was written for a program the backend refused; that file is what \
         `ptxas` rejects on somebody else's machine"
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// THE CONTROL. Refusing every member access satisfies the test above and
/// deletes the `v4` lane surface, which is how eight-limb field arithmetic is
/// written in this repo. `.x/.y/.z/.w` on a value bound by a vector load must
/// still lower.
#[test]
fn a_v4_lane_is_still_a_member_access_that_works() {
    let src = repo().join("tests/bn254_fr_mul_fast.ysu");
    assert!(src.exists(), "the v4 fixture is missing");
    // Compile a COPY in a per-process temp directory. `--emit-ptx` writes next
    // to its source, and THREE test binaries compile this same fixture - this
    // one, `zk_gpu_field.rs`, and `zk_gpu_groth16.rs` through `common/qap.rs`.
    // `cargo test` runs them in parallel, so in-place compilation had them
    // truncating and reading one repo path at once; this test read a torn file
    // and reported "the v4 kernel emitted no loads" after many clean runs,
    // which is the documented signature of that race. The per-process mutex in
    // the `ptx_for` helpers cannot help across processes.
    let dir = std::env::temp_dir().join(format!("y_v4_lane_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let tmp = dir.join("bn254_fr_mul_fast.ysu");
    std::fs::copy(&src, &tmp).expect("copy the v4 fixture");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&tmp)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("run Y");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    assert!(
        out.status.success(),
        "a kernel reading `.x/.y/.z/.w` off a v4 load must still compile:\n{text}"
    );
    let ptx = std::fs::read_to_string(dir.join("bn254_fr_mul_fast.ptx"))
        .expect("emitted PTX");
    assert!(
        ptx.lines().filter(|l| l.contains("ld.global")).count() > 0,
        "the v4 kernel emitted no loads, so the lanes reached nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The class-wide structural gate: an emitted instruction may never have a
/// MISSING operand.
///
/// Every bug of this shape - `Expr::Ident`, the `_` arm, `mbarrier_*`,
/// `mma_sync`, and now `MemberAccess` - produces the same artifact signature,
/// a comma with nothing after it. Checking the signature catches the next one
/// without anyone having to guess which arm it will be in.
#[test]
fn no_emitted_instruction_has_a_missing_operand() {
    let mut checked = 0usize;
    for entry in std::fs::read_dir(repo().join("tests")).expect("read tests/") {
        let path = entry.expect("dir entry").path();
        if path.extension().map(|e| e != "ptx").unwrap_or(true) {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read ptx");
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").trim();
            if code.is_empty() {
                continue;
            }
            // `, ;` - an operand list that ends in a comma. `, ,` - one with a
            // hole in the middle.
            assert!(
                !code.contains(", ;") && !code.contains(",;") && !code.contains(", ,"),
                "{}:{}: `{}` has a missing operand. Some `emit_expr` arm returned \
                 an empty string and a caller spliced it into instruction text.",
                path.display(),
                i + 1,
                code
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 20,
        "only {checked} committed .ptx files were scanned; a sweep that finds \
         nothing reports no missing operands perfectly"
    );
}
