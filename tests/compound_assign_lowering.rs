//! `x op= e` must compute `x = x op e`.
//!
//! Outside a `@ZeroDrift` accumulator it computed NOTHING. `emit_stmt` had one
//! `Stmt::CompoundAssign` arm, guarded on the target being a drift
//! accumulator; every other compound assignment fell through every arm below it
//! and landed in the function's `_ => {}`, so
//!
//! ```text
//!     let mut acc: U32 = 1;
//!     acc += 7;
//!     ... store acc          ->   the kernel stores 1
//! ```
//!
//! compiled clean, exited 0, wrote a `.ptx`, and `ptxas` accepted it. No
//! assemble gate can see a MISSING instruction, which is gotcha #8 stated about
//! this exact shape.
//!
//! It survived because **no kernel in this corpus uses `+=` in a body**: the
//! 256 occurrences in `bn254_g1_add.ysu` and the 272 in `bn254_ntt4_fused.ysu`
//! are all comments (`// t += a * b_i`), and the two sources that really use it
//! -- `hello.ysu`, `simple_test.ysu` -- are host programs the PTX backend
//! refuses for declaring no `kernel`. Reachable from the surface syntax,
//! reached by nothing that runs. *Find these while the path is still dead.*
//!
//! The load-bearing test needs no knowledge of what correct PTX looks like: the
//! two spellings of one statement must emit BYTE-IDENTICAL kernels.
//!
//! It is paired with a non-vacuity assertion, and **measured, the two are
//! redundant for the shipped defect**: that bug made the spellings DIFFER (one
//! emitted nothing, the other emitted the adds), so byte-identity catches it on
//! its own. Non-vacuity is kept because it covers a future byte-identity cannot
//! see -- anything that makes BOTH spellings emit nothing, such as the desugar
//! becoming a no-op or `Stmt::Assign` itself regressing. Two identically-empty
//! kernels are byte-identical.

use std::path::Path;
use std::process::Command;

fn compile(fixture: &str) -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    // A COPY, in a directory whose name carries an atomic counter as well as a
    // tag: `--emit-ptx` writes next to its input, and three tests in this file
    // compile fixtures whose names alone would collide across parallel runs.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_compound_{}_{}_{}",
        std::process::id(),
        fixture.replace('/', "_").replace('.', "_"),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("k.ysu");
    std::fs::copy(repo.join(fixture), &src).expect("fixture missing");
    let _ = std::fs::copy(repo.join(".ysu_hw_profile"), dir.join(".ysu_hw_profile"));

    let out = Command::new(bin.join("Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "{} did not compile:\n{}{}",
        fixture,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).expect("no .ptx emitted");
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

/// The equality. Needs no model of correct PTX -- and is worth nothing without
/// the non-vacuity test below, which is what the bug demonstrates: before the
/// fix these two also agreed, on a kernel that did no arithmetic at all.
#[test]
fn the_two_spellings_emit_byte_identical_kernels() {
    let compound = compile("tests/compound_assign.ysu");
    let longhand = compile("tests/compound_assign_desugared.ysu");
    assert_eq!(
        compound, longhand,
        "`x += e` and `x = x + e` are the same statement and must emit the same \
         kernel; a second implementation is how two spellings come to compute \
         different functions"
    );
}

/// NON-VACUITY. `acc += 7; acc -= 2; acc *= 3;` must leave three arithmetic
/// instructions in the kernel. "Emit nothing for both spellings" passes the
/// equality above perfectly; that is NOT the state that shipped (the bug made
/// them differ), so this is redundancy against a different future rather than
/// the sole catcher -- verified by mutation.
#[test]
fn the_compound_form_really_computes() {
    let ptx = compile("tests/compound_assign.ysu");
    for op in ["add", "sub", "mul"] {
        assert!(
            ptx.lines().any(|l| {
                let t = l.trim();
                !t.starts_with("//") && t.contains(&format!("{}.", op)) && t.contains("%r")
            }),
            "no `{}` over a 32-bit register: the compound assignment emitted \
             nothing and the kernel keeps its initial value\n{}",
            op,
            ptx
        );
    }
}

/// THE CONTROL: a drift accumulator must still be accumulated exactly.
///
/// The obvious claim -- "the general arm must sit BELOW the `@ZeroDrift` one,
/// or a drift accumulator goes through the float path" -- was **tested and is
/// FALSE**: moving it above emits a BYTE-IDENTICAL kernel. The desugared
/// `sum = sum + v` lands on the drift `Stmt::Assign` arm, which exists because
/// an earlier increment had to make those two spellings agree, so both routes
/// are drift-aware. The ordering is kept anyway (the guarded arm is the direct
/// one), and this test pins the EXACTNESS rather than the ordering.
#[test]
fn a_drift_accumulator_still_takes_its_exact_path() {
    let ptx = compile("tests/compound_assign_drift.ysu");
    assert!(
        ptx.contains("[Y ZERO DRIFT]"),
        "the drift accumulator lost its marker, so the desugaring shadowed its \
         arm\n{}",
        ptx
    );
    assert!(
        ptx.lines().any(|l| l.trim().starts_with("add.s64")),
        "the accumulation is not an exact 64-bit integer add\n{}",
        ptx
    );
    assert!(
        !ptx.lines().any(|l| l.trim().starts_with("add.f32")),
        "the drift accumulator is being added in FLOAT, which is precisely what \
         @ZeroDrift forbids\n{}",
        ptx
    );
}

/// Assembling is necessary and not sufficient -- it cannot see the missing
/// instruction this file is about, which is why it is the last test and not the
/// first. It is here so a desugaring that emits ill-formed PTX fails by name.
#[test]
fn the_emitted_kernel_assembles() {
    let ptx = compile("tests/compound_assign.ysu");
    let dir = std::env::temp_dir().join(format!("y_compound_asm_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let f = dir.join("k.ptx");
    std::fs::write(&f, &ptx).unwrap();
    match Command::new("ptxas").arg("-arch=sm_89").arg(&f).arg("-o").arg(dir.join("k.cubin")).output() {
        Ok(out) => assert!(
            out.status.success(),
            "ptxas rejected the desugared kernel:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(_) => eprintln!("SKIP: ptxas unavailable"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}
