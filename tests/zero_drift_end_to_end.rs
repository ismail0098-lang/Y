//! `@ZeroDrift` must actually remove the drift - proven by running the binary.
//!
//! The annotation used to do nothing at all: parsed, counted, printed as an
//! advisory, and read by no backend, so compiling with and without it produced
//! byte-identical output. Checking that the emitted IR now *mentions* an `i64`
//! would be a weak replacement for that, because the property being claimed is
//! numerical, not syntactic.
//!
//! So this compiles a Y program with `clang`, runs it, and checks the number it
//! prints. The program sums the same terms in two different orders. Floating
//! point addition is not associative, so a `f32` or `f64` accumulator gives two
//! different answers; an exact fixed-point accumulator gives one.
//!
//! The control matters as much as the experiment. `adversarial_sequence_really_
//! does_drift_in_f32` computes the identical sums in Rust and asserts they
//! DISAGREE - without it, a test that merely observes "the two orders matched"
//! would pass just as happily on a sequence too benign to expose anything.
//!
//! Requires `clang`; skipped with a notice otherwise, like the `ptxas` and
//! `solc` gates elsewhere in the suite.
//!
//! Run with:  cargo test --test zero_drift_end_to_end

use std::path::PathBuf;
use std::process::Command;

/// One large term followed by many tiny ones.
///
/// Summed largest-first in `f32`, each tiny term is smaller than an ulp of the
/// running total and vanishes entirely. Summed smallest-first they accumulate
/// into something the large term cannot swallow. That gap is the drift.
const BIG: f64 = 1.0;
const SMALL: f64 = 1.0e-7;
const N_SMALL: usize = 4000;

// Y's lexer has no scientific notation, so the generated source spells these
// out in full decimal.
/// Q32.32's scale, used to read the difference back out in exact units.
const SCALE: f64 = 4294967296.0;

fn y_source() -> String {
    let mut s = String::from("fn main() {\n");

    // Largest first.
    s.push_str("    @bounds(min=0, max=1000)\n    @ZeroDrift\n    let fwd: F32 = 0.0;\n");
    s.push_str(&format!("    fwd += {:.1};\n", BIG));
    for _ in 0..N_SMALL {
        s.push_str(&format!("    fwd += {:.10};\n", SMALL));
    }

    // Smallest first - same terms, opposite order.
    s.push_str("    @bounds(min=0, max=1000)\n    @ZeroDrift\n    let rev: F32 = 0.0;\n");
    for _ in 0..N_SMALL {
        s.push_str(&format!("    rev += {:.10};\n", SMALL));
    }
    s.push_str(&format!("    rev += {:.1};\n", BIG));

    // Report the difference in exact Q32.32 units, so a discrepancy of a single
    // representable step shows up as 1 rather than rounding away to 0.
    s.push_str(&format!(
        "    let diff: I32 = (fwd - rev) * {:.1};\n    print_int(diff);\n}}\n",
        SCALE
    ));
    s
}

/// Builds and runs `source`, returning stdout.
fn build_and_run(name: &str, source: &str) -> Option<String> {
    if Command::new("clang").arg("--version").output().is_err() {
        return None;
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("y_zd_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ysu = dir.join(format!("{}.ysu", name));
    std::fs::write(&ysu, source).expect("write source");

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&ysu)
        .arg("--emit-llvm")
        .current_dir(&repo)
        .output()
        .expect("run Y");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ll = dir.join(format!("{}.ll", name));
    assert!(ll.exists(), "Y did not emit IR:\n{}", log);

    // A two-line shim rather than the repo's runtime.c. The program only calls
    // `print_int`, and `c_src/runtime.c` currently fails to compile for an
    // unrelated reason (a static-after-non-static declaration in
    // shadowplay_gui.h). Linking it would make this test fail for something it
    // is not testing.
    let shim = dir.join("shim.c");
    std::fs::write(
        &shim,
        "#include <stdio.h>\nvoid print_int(long long v) { printf(\"%lld\\n\", v); }\nvoid ysu_main(void);\nint main(void) { ysu_main(); return 0; }\n",
    )
    .expect("write shim");

    let bin = dir.join("prog");
    let cc = Command::new("clang")
        .arg("-O2")
        .arg("-o")
        .arg(&bin)
        .arg(&ll)
        .arg(&shim)
        .arg("-lm")
        .output()
        .expect("run clang");
    assert!(
        cc.status.success(),
        "clang failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let run = Command::new(&bin).output().expect("run program");
    Some(String::from_utf8_lossy(&run.stdout).trim().to_string())
}

/// The control: this sequence genuinely drifts in `f32`.
///
/// Without this the main test proves nothing - "both orders agreed" is trivial
/// if nothing could have made them disagree.
#[test]
fn adversarial_sequence_really_does_drift_in_f32() {
    let mut fwd = BIG as f32;
    for _ in 0..N_SMALL {
        fwd += SMALL as f32;
    }
    let mut rev = 0f32;
    for _ in 0..N_SMALL {
        rev += SMALL as f32;
    }
    rev += BIG as f32;

    assert_ne!(
        fwd.to_bits(),
        rev.to_bits(),
        "the test sequence must actually be order-sensitive in f32, otherwise the \
         zero-drift result below is vacuous"
    );
}

/// The experiment: with `@ZeroDrift`, the two orders agree exactly.
#[test]
fn zero_drift_accumulation_is_order_independent() {
    let Some(out) = build_and_run("drift", &y_source()) else {
        eprintln!("skipping: no clang");
        return;
    };
    let printed: i64 = out
        .lines()
        .last()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or_else(|| panic!("expected a single integer, got: {:?}", out));

    assert_eq!(
        printed, 0,
        "summing the same terms in opposite orders differed by {} Q32.32 steps - \
         the accumulation is not exact",
        printed
    );
}

/// `@ZeroDrift` must change the generated code.
///
/// This is the assertion that would have failed for the entire life of the
/// annotation before now: the old implementation printed an advisory and left
/// codegen untouched, so the IR was byte-identical with and without it.
#[test]
fn annotation_changes_the_generated_ir() {
    let with = "fn main() {\n    @bounds(min=0, max=1000)\n    @ZeroDrift\n    let acc: F32 = 0.0;\n    acc += 1.0;\n    print_int(1);\n}\n";
    let without = "fn main() {\n    @bounds(min=0, max=1000)\n    let acc: F32 = 0.0;\n    acc += 1.0;\n    print_int(1);\n}\n";

    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let emit = |name: &str, src: &str| -> String {
        let dir = std::env::temp_dir().join(format!("y_zdir_{}_{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let ysu = dir.join(format!("{}.ysu", name));
        std::fs::write(&ysu, src).expect("write");
        let _ = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&ysu)
            .arg("--emit-llvm")
            .current_dir(&repo)
            .output()
            .expect("run Y");
        std::fs::read_to_string(dir.join(format!("{}.ll", name))).expect("read IR")
    };

    let ir_with = emit("with", with);
    let ir_without = emit("without", without);

    assert_ne!(ir_with, ir_without, "@ZeroDrift must affect codegen");
    assert!(
        ir_with.contains("add i64"),
        "the accumulator should be accumulating in exact integer arithmetic:\n{}",
        ir_with
    );
    assert!(
        !ir_without.contains("add i64"),
        "the unannotated version should not be using a fixed-point accumulator"
    );
}

/// The PTX path must lower `@ZeroDrift` too, and the result must ASSEMBLE.
///
/// GPU kernels are where reduction-order nondeterminism actually bites - the
/// summation order is decided by launch geometry, so retuning a tile can change
/// the answer. String-matching the PTX would not be enough here: this repo has
/// already shipped PTX containing instructions that exist on no hardware, and
/// the gate that caught it was running `ptxas`. Skipped when `ptxas` is absent.
#[test]
fn ptx_zero_drift_lowers_and_assembles() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("y_zdptx_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ysu = dir.join("acc.ysu");
    std::fs::write(
        &ysu,
        "@require(sm >= 89)\nkernel drift_acc(A: GlobalMemory<F32>, C: GlobalMemory<F32>) {\n\
         \x20   @bounds(min=0, max=1000)\n    @ZeroDrift\n    let acc: F32 = 0.0;\n\
         \x20   acc += 1.0;\n    acc += 0.25;\n    acc += 0.125;\n}\n",
    )
    .expect("write source");

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&ysu)
        .arg("--emit-ptx")
        .current_dir(&repo)
        .output()
        .expect("run Y");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx_path = dir.join("acc.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|_| panic!("no PTX:\n{}", log));

    // Exact integer accumulation, one `add.s64` per `+=`.
    assert!(
        ptx.contains("[Y ZERO DRIFT]"),
        "the annotation left no trace in the PTX:\n{}",
        ptx
    );
    assert_eq!(
        ptx.matches("add.s64").count(),
        3,
        "expected one exact accumulate per `+=`:\n{}",
        ptx
    );
    assert!(
        !ptx.contains("add.f32 %f"),
        "the accumulator must not fall back to float addition:\n{}",
        ptx
    );

    // And it has to be real PTX.
    let Ok(res) = Command::new("ptxas").arg("-arch=sm_89").arg(&ptx_path).arg("-o").arg("/dev/null").output() else {
        eprintln!("skipping ptxas gate: ptxas not on PATH");
        return;
    };
    assert!(
        res.status.success(),
        "emitted PTX does not assemble:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
}
