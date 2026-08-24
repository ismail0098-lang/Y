//! `@ZeroDrift` must mean the same thing in every backend that lowers it.
//!
//! `acc += e` had a drift-aware arm in both backends. `acc = acc + e` - the
//! SAME statement, and the form the recognised GEMM nest is written in - had an
//! arm in neither. The LLVM backend was fixed while wiring Phase 0; the PTX
//! backend was not, and nothing noticed for the obvious reason that nothing
//! looked. What it emitted for an exact `I64` accumulator was:
//!
//! ```text
//!     // [Y ZERO DRIFT] sum: I64 accumulated exactly as I64
//!     cvt.rn.f64.s64 %fd0, %rd3;      ; the accumulator, as f64
//!     cvt.rn.f32.f64 %f0, %fd0;       ; ...then as f32
//!     cvt.rn.f32.s64 %f1, %rd12;      ; the product, as f32
//!     add.f32 %f2, %f0, %f1;          ; ACCUMULATE IN F32
//!     cvt.rzi.u64.f32 %rd13, %f2;     ; back, UNSIGNED, truncating
//! ```
//!
//! Three separate wrongs under a comment claiming exactness: the accumulation
//! is f32, so everything above 2^24 rounds; the write-back is unsigned, so a
//! negative sum is not representable; and the whole round trip is pointless for
//! a representation that already holds the integer. It assembles and it
//! launches, which is the signature of every bug in this repo's design-rule
//! table.
//!
//! **The rule now lives in `zero_drift::running_sum`, which both backends
//! call.** A second copy is how they came to disagree, so the tests below check
//! agreement rather than checking each backend against a list.

use std::path::PathBuf;
use std::process::Command;

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn dir(tag: &str) -> PathBuf {
    // Per-test directory: these run in one binary and share a pid.
    let d = std::env::temp_dir().join(format!("y_drift_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

/// The exact integer nest, with the accumulation written either way.
fn integer_nest(accumulate: &str) -> String {
    format!(
        r#"
kernel y_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {{
    @invariant(i >= 0)
    for i in 0..M step 1 {{
        @invariant(j >= 0)
        for j in 0..N step 1 {{
            @ZeroDrift
            let mut sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {{
                @bounds(min=-1024, max=1024)
                let a_val: I64 = block_ptr2d_load(A, i, k, K, M, K);
                @bounds(min=-1024, max=1024)
                let b_val: I64 = block_ptr2d_load(B, k, j, N, K, N);
                {accumulate}
            }}
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }}
    }}
}}

fn main() {{
}}
"#
    )
}

/// Compile `src` with `flag`; returns (exit ok, combined output, artifact text).
fn compile(d: &PathBuf, name: &str, src: &str, flag: &str, ext: &str) -> (bool, String, Option<String>) {
    let path = d.join(format!("{name}.ysu"));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg(flag)
        .output()
        .expect("run Y");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    let artifact = std::fs::read_to_string(d.join(format!("{name}.{ext}"))).ok();
    (out.status.success(), text, artifact)
}

/// Instruction lines only - comments and register declarations move around for
/// reasons that are not the accumulation.
fn body(ptx: &str) -> Vec<String> {
    ptx.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .map(|l| l.to_string())
        .collect()
}

/// The load-bearing test: the two spellings are one statement, so they must
/// compile to one kernel. This is the assertion that fails on the original bug
/// and it needs no knowledge of what the right PTX looks like.
#[test]
fn the_two_spellings_of_an_exact_accumulation_emit_the_same_ptx() {
    let d = dir("spell");
    let (ok_c, out_c, ptx_c) = compile(
        &d,
        "compound",
        &integer_nest("sum += a_val * b_val;"),
        "--emit-ptx",
        "ptx",
    );
    let (ok_r, out_r, ptx_r) = compile(
        &d,
        "running",
        &integer_nest("sum = sum + a_val * b_val;"),
        "--emit-ptx",
        "ptx",
    );
    assert!(ok_c, "the `+=` form must compile:\n{out_c}");
    assert!(ok_r, "the running-sum form must compile:\n{out_r}");

    let a = body(&ptx_c.expect("compound artifact"));
    let b = body(&ptx_r.expect("running artifact"));
    assert_eq!(
        a, b,
        "`sum += e` and `sum = sum + e` are the same statement and emitted \
         different kernels. That is how the running-sum form came to accumulate \
         in f32 while `+=` accumulated exactly."
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// What the agreement above is agreement ON. Both forms must reach `add.s64`
/// and neither may touch a float - an equality test alone is satisfied by two
/// kernels that are identically wrong.
#[test]
fn an_integer_drift_accumulator_never_touches_a_float() {
    let d = dir("nofloat");
    for (name, form) in [
        ("compound", "sum += a_val * b_val;"),
        ("running", "sum = sum + a_val * b_val;"),
    ] {
        let (ok, out, ptx) = compile(&d, name, &integer_nest(form), "--emit-ptx", "ptx");
        assert!(ok, "{name} must compile:\n{out}");
        let ptx = ptx.expect("artifact");
        assert!(
            ptx.contains("add.s64"),
            "{name}: an exact I64 accumulator must accumulate with `add.s64`; \
             the emitted kernel has no such instruction:\n{ptx}"
        );
        // Not a list of banned opcodes - a list would have to anticipate which
        // float the emitter reaches for, and the first version of this test did
        // exactly that and missed `cvt.rn.f64.s64`. THIS PROGRAM CONTAINS NO
        // FLOAT AT ALL: every operand, the accumulator and the output are
        // integers. So no instruction in it may mention one. (`.reg .f32`
        // declarations are skipped - the register pool is emitted unconditionally
        // and declaring an unused one is not an arithmetic claim.)
        let floats: Vec<&str> = ptx
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with("//") && !l.starts_with('.'))
            .filter(|l| l.contains(".f32") || l.contains(".f64"))
            .collect();
        assert!(
            floats.is_empty(),
            "{name}: the accumulator is an exact I64 and this program has no \
             float in it, yet the kernel contains:\n  {}\nRouting an exact \
             accumulation through a float is what `@ZeroDrift` exists to prevent.",
            floats.join("\n  ")
        );
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// THE CONTROL. "Never emit a float for a drift accumulator" is satisfied by
/// breaking the fixed-point path, which genuinely needs one: a `Q32.32`
/// accumulator holds a SCALED float and must convert on every access. Without
/// this, deleting the fixed-point branch entirely passes the test above.
#[test]
fn a_fixed_point_accumulator_still_scales_through_a_float() {
    let d = dir("fixed");
    let src = r#"
kernel k(A: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @ZeroDrift
        @bounds(min=-1024.0, max=1024.0)
        let mut acc: F32 = 0.0;
        @invariant(k >= 0)
        for k in 0..K step 1 {
            let v: F32 = block_ptr2d_load(A, i, k, K, M, K);
            acc = acc + v;
        }
        block_ptr2d_store(C, i, 0, N, M, N, acc);
    }
}
fn main() {}
"#;
    let (ok, out, ptx) = compile(&d, "fx", src, "--emit-ptx", "ptx");
    assert!(ok, "the fixed-point nest must compile:\n{out}");
    let ptx = ptx.expect("artifact");
    assert!(
        ptx.contains("Q32.32") || ptx.contains("Q16.16"),
        "an F32 accumulator with @bounds should select a Q format:\n{ptx}"
    );
    assert!(
        ptx.contains("mul.f64"),
        "a fixed-point accumulator encodes a float as a scaled integer, so the \
         scale multiply must still be there. If this is gone, the float path was \
         removed rather than confined to the case that needs it:\n{ptx}"
    );
    // ...and the accumulation itself is still exact, on the scaled integer.
    assert!(ptx.contains("add.s64"), "the accumulate is still an integer add");
    let _ = std::fs::remove_dir_all(&d);
}

/// Refusing is the fix, not a stopgap: lowering this as an ordinary assignment
/// is precisely the bug. Both backends must refuse, because the rule is now
/// shared - a backend that quietly accepts it emits a kernel whose exactness
/// claim is false.
#[test]
fn both_backends_refuse_an_assignment_that_is_not_an_accumulation() {
    let d = dir("refuse");
    let src = integer_nest("sum = sum * a_val;");
    for (flag, ext) in [("--emit-ptx", "ptx"), ("--emit-llvm", "ll")] {
        let name = format!("bad{}", ext);
        let (ok, out, artifact) = compile(&d, &name, &src, flag, ext);
        assert!(
            !ok,
            "{flag} accepted `sum = sum * a_val` on a @ZeroDrift accumulator. \
             Multiplication reintroduces the rounding the directive removes.\n{out}"
        );
        assert!(
            out.contains("@ZeroDrift accumulator"),
            "{flag} must refuse BY NAME, saying which binding and why:\n{out}"
        );
        assert!(
            !out.contains("Compilation Successful"),
            "{flag} printed the success banner over a refusal:\n{out}"
        );
        assert!(
            artifact.is_none(),
            "{flag} wrote an artifact for a program it refused"
        );
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// The other half of the agreement: both backends ACCEPT the running-sum form.
/// "Refuse everything" satisfies the test above and deletes a working path -
/// the same shape as `ordinary_loop_bodies_still_verify`.
#[test]
fn both_backends_accept_the_running_sum_form() {
    let d = dir("accept");
    let src = integer_nest("sum = sum + a_val * b_val;");
    for (flag, ext) in [("--emit-ptx", "ptx"), ("--emit-llvm", "ll")] {
        let name = format!("good{}", ext);
        let (ok, out, artifact) = compile(&d, &name, &src, flag, ext);
        assert!(ok, "{flag} refused the running-sum form:\n{out}");
        assert!(
            artifact.is_some_and(|a| !a.trim().is_empty()),
            "{flag} produced no artifact for a program it accepted"
        );
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// An assemble gate cannot see a missing instruction, but it does catch a
/// register the emitter never declared - which is what an untyped accumulator
/// produced when the write-back changed width.
#[test]
fn the_exact_kernel_assembles() {
    if !have("ptxas") {
        eprintln!("skipping: ptxas not found - the emitted PTX is NOT being assembled");
        return;
    }
    let d = dir("ptxas");
    let (ok, out, ptx) = compile(
        &d,
        "asm",
        &integer_nest("sum = sum + a_val * b_val;"),
        "--emit-ptx",
        "ptx",
    );
    assert!(ok, "must compile:\n{out}");
    let path = d.join("asm.ptx");
    assert!(ptx.is_some());
    let res = Command::new("ptxas")
        .args(["-arch=sm_89", path.to_str().unwrap(), "-o", "/dev/null"])
        .output()
        .expect("run ptxas");
    assert!(
        res.status.success(),
        "ptxas rejected the emitted kernel:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
    let _ = std::fs::remove_dir_all(&d);
}
