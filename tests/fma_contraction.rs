//! The emitter says what the machine does: `a*b + c` over F32 is one
//! `fma.rn.f32`, not a `mul.f32` and an `add.f32`.
//!
//! `ptxas` is free to contract a `mul.f32` feeding an `add.f32` into a single
//! `FFMA`, which rounds ONCE where the PTX asks for two roundings, and on this
//! corpus it takes that freedom every time it is offered. So the shipped
//! artifact did not mean what the hardware performed, and
//! `tools/ptxas_tval/loopval.py` REFUTED `naive_gemm_f32` for exactly that -
//! `BASE`, `STEP`, `LOOPCOND` and `ENTRY` all proved, and the accumulated
//! VALUE could not be shown equal.
//!
//! There are two repairs and the difference between them is the finding.
//! Forbidding the fusion (`mul.rn.f32` + `add.rn.f32`) also validates, and it
//! is the worse one: it changes the instruction stream and leaves the kernel
//! rounding twice where the hardware rounds once. STATING the fusion emits a
//! byte-identical instruction stream, which is what makes this free.
//!
//! Four things are pinned here, and the second is a regression rather than a
//! feature. The recognition fires on floats; it does NOT disturb the integer
//! `a*b + c`, whose multiply must stay where it was - deciding the shape after
//! emitting `c` moved `mul.lo.s32` past `c`'s own instructions and changed the
//! emitted PTX of FOURTEEN integer kernels that contain no float at all. The
//! two committed artifacts that carry the shape say `fma.rn.f32`. And the
//! byte-identity claim is MEASURED with `ptxas` rather than asserted, with a
//! control: the forbidding form must NOT be byte-identical, or the comparison
//! is passing because nothing distinguishes any of the three.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Compiles a kernel source to PTX in a private directory.
///
/// The tag is in the SIGNATURE because a per-process temp path is not enough:
/// two tests in one binary sharing a directory race on `remove_dir_all`, which
/// this repository has now been bitten by seven times. The counter is what
/// makes the tag sufficient rather than merely conventional - a tag is for
/// legibility when a run leaves a directory behind.
fn emit_ptx(tag: &str, src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_fma_{}_{}_{}",
        tag,
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    // Pin the profile: the emitted target must not be the build machine's card.
    std::fs::write(
        dir.join(".ysu_hw_profile"),
        "SM_VERSION=8.9\nGPU_NAME=FmaGate\nSM_COUNT=66\n",
    )
    .expect("pin the profile");
    let f = dir.join("k.ysu");
    std::fs::write(&f, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&f)
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "the fixture did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).expect("no .ptx written");
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

/// The kernel body only, with comments stripped.
///
/// A raw substring search over the whole module matches the emitter's own
/// comments - including the ones explaining the very instruction being looked
/// for - which is how a gate in this directory once reported the opposite of
/// the truth.
fn body(ptx: &str) -> String {
    let start = ptx.find(".visible .entry").expect("no entry point");
    ptx[start..]
        .lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const FLOAT_SRC: &str = r#"
kernel probe(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let i: I32 = block_idx_x() * 128 + thread_idx_x();
    let a: F32 = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: F32 = block_ptr2d_load(B, 0, i, N, 1, N);
    let c: F32 = block_ptr2d_load(C, 0, i, N, 1, N);
    let lhs: F32 = a * b + c;
    let rhs: F32 = c + a * b;
    block_ptr2d_store(C, 0, i, N, 1, N, lhs + rhs);
}
fn main() {}
"#;

/// One `a*b + c` and nothing else.
///
/// `FLOAT_SRC` cannot serve the `ptxas` comparison: its two expressions compute
/// the SAME value, so `ptxas` common-subexpression-eliminates the pair on one
/// arm and not on the other, and the two arms then differ for a reason that has
/// nothing to do with contraction. That is the same trap `fpsem_abi.py` records
/// - the translator under test rewriting the question.
const ONE_FMA_SRC: &str = r#"
kernel probe(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let i: I32 = block_idx_x() * 128 + thread_idx_x();
    let a: F32 = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: F32 = block_ptr2d_load(B, 0, i, N, 1, N);
    let c: F32 = block_ptr2d_load(C, 0, i, N, 1, N);
    block_ptr2d_store(C, 0, i, N, 1, N, a * b + c);
}
fn main() {}
"#;

#[test]
fn a_float_multiply_feeding_an_add_is_emitted_as_one_fma() {
    let b = body(&emit_ptx("float", FLOAT_SRC));
    let fmas = b.matches("fma.rn.f32").count();
    // BOTH operand orders, because the multiply may sit on either side of the
    // `+` and only one of them needed care about emission order.
    assert_eq!(
        fmas, 2,
        "`a*b + c` and `c + a*b` must each emit one fma.rn.f32; got {fmas}:\n{b}"
    );
    assert!(
        !b.contains("mul.f32"),
        "a `mul.f32` survived, so one of the two orders is still asking for a \
         rounding the hardware does not perform:\n{b}"
    );
}

/// Only `+`. A subtraction is NOT this shape, and folding it into the same
/// instruction computes a different function.
///
/// `a*b - c` is `fma.rn.f32 d, a, b, -c` and `c - a*b` is
/// `fma.rn.f32 d, -a, b, c`: each needs a `neg.f32` the two-instruction form
/// does not, so there is no instruction to save and there IS a sign to get
/// wrong. Recognising `Sub` alongside `Add` without negating anything emits
/// `a*b + c` for `a*b - c` -- a wrong answer under a green banner, which is
/// the failure class this repository's design rule is about. It survived
/// every other suite in the mutation sweep, including this file before this
/// test existed.
#[test]
fn a_float_multiply_feeding_a_subtract_is_not_fused() {
    let src = r#"
kernel probe(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let i: I32 = block_idx_x() * 128 + thread_idx_x();
    let a: F32 = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: F32 = block_ptr2d_load(B, 0, i, N, 1, N);
    let c: F32 = block_ptr2d_load(C, 0, i, N, 1, N);
    block_ptr2d_store(C, 0, i, N, 1, N, a * b - c);
}
fn main() {}
"#;
    let b = body(&emit_ptx("sub", src));
    assert!(
        b.contains("mul.f32") && b.contains("sub.f32"),
        "`a*b - c` must stay a multiply and a subtract:\n{b}"
    );
    assert!(
        !b.contains("fma."),
        "`a*b - c` was folded into an fma. Unless the addend is negated that \
         computes `a*b + c`, which is a different function:\n{b}"
    );
}

#[test]
fn an_integer_multiply_feeding_an_add_stays_exactly_where_it_was() {
    // `block_idx_x() * block_idx_y()` is a multiply the emitter cannot fold,
    // and `thread_idx_x()` is an addend that costs an instruction of its own -
    // which is what makes the ORDER observable at all.
    let src = r#"
kernel probe(O: GlobalMemory<U32>, N: I32) {
    let k: I32 = block_idx_x() * block_idx_y() + thread_idx_x();
    block_ptr2d_store(O, 0, 0, N, 1, N, k);
}
fn main() {}
"#;
    let b = body(&emit_ptx("int", src));
    assert!(
        !b.contains("fma."),
        "an integer `a*b + c` must not be fused; there is no integer fma here:\n{b}"
    );
    let mul = b.find("mul.lo.s32").expect("no integer multiply emitted");
    let tid = b.find("%tid.x").expect("no thread index emitted");
    assert!(
        mul < tid,
        "the integer multiply moved PAST the instruction computing its addend. \
         That is a reordering, not a fusion, and it silently changed the \
         emitted PTX of fourteen kernels with no float in them:\n{b}"
    );
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

#[test]
fn the_committed_artifacts_say_what_the_hardware_does() {
    // Exactly the artifacts whose SOURCE writes a float `a*b + c`. Hand-written
    // kernel bodies elsewhere in the emitter still spell the pair out; they are
    // not reached by expression lowering and are named in the residue rather
    // than swept up here.
    let mut checked = 0;
    for k in ["naive_gemm_f32", "y_cpu_matmul"] {
        let p = repo().join(format!("tests/{k}.ptx"));
        let b = body(&std::fs::read_to_string(&p).expect("committed artifact"));
        assert!(
            b.contains("fma.rn.f32"),
            "tests/{k}.ptx does not say fma.rn.f32, so it claims a rounding \
             `ptxas` does not perform"
        );
        assert!(
            !b.contains("mul.f32"),
            "tests/{k}.ptx still contains a mul.f32"
        );
        checked += 1;
    }
    assert_eq!(checked, 2, "the artifact list emptied itself");
}

/// `ptxas` is what decides whether this change is free, so it is measured.
///
/// Skips when `ptxas` is absent - and the three tests above do not, which is
/// what stops the whole file going quiet on a machine with no CUDA toolkit.
#[test]
fn stating_the_fusion_is_free_and_forbidding_it_is_not() {
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("SKIP: no ptxas");
        return;
    }
    let ptx = emit_ptx("sass", ONE_FMA_SRC);
    assert_eq!(
        body(&ptx).matches("fma.rn.f32").count(),
        1,
        "this fixture must contain exactly ONE fma, or the rewrite below \
         replaces one of several and the arms differ for an unrelated reason"
    );
    // Derive the two other readings from the shipped one, so all three are the
    // same kernel and nothing else varies.
    let one = ptx
        .lines()
        .find(|l| l.trim_start().starts_with("fma.rn.f32"))
        .expect("no fma.rn.f32 to rewrite")
        .to_string();
    let ops: Vec<&str> = one
        .trim()
        .trim_start_matches("fma.rn.f32")
        .trim_end_matches(';')
        .split(',')
        .map(|s| s.trim())
        .collect();
    assert_eq!(ops.len(), 4, "unexpected fma operand count in `{one}`");
    let (d, a, b, c) = (ops[0], ops[1], ops[2], ops[3]);
    // Splitting one instruction into two needs a register the kernel does not
    // declare; a body naming a register outside its declared pool is a bug this
    // repository has shipped before.
    let pool = ptx
        .lines()
        .find(|l| l.trim_start().starts_with(".reg .f32 %f<"))
        .expect("no f32 register pool")
        .to_string();
    let n: usize = pool
        .trim()
        .trim_start_matches(".reg .f32 %f<")
        .trim_end_matches(">;")
        .parse()
        .expect("register pool count");
    let grown = ptx.replace(pool.trim(), &format!(".reg .f32 %f<{}>;", n + 1));
    let prod = format!("%f{n}");
    let split = |m: &str, p: &str| {
        grown.replacen(
            one.trim(),
            &format!("{m} {prod}, {a}, {b};\n    {p} {d}, {c}, {prod};"),
            1,
        )
    };
    let sass = |name: &str, src: &str| -> String {
        let dir = std::env::temp_dir().join(format!("y_fma_sass_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let p = dir.join(format!("{name}.ptx"));
        std::fs::write(&p, src).expect("write ptx");
        let cu = dir.join(format!("{name}.cubin"));
        let o = Command::new("ptxas")
            .args(["-O1", "-arch=sm_89", "-o"])
            .arg(&cu)
            .arg(&p)
            .output()
            .expect("ptxas");
        assert!(
            o.status.success(),
            "ptxas rejected the {name} form:\n{}",
            String::from_utf8_lossy(&o.stderr)
        );
        let d = Command::new("nvdisasm")
            .arg("-c")
            .arg(&cu)
            .output()
            .expect("nvdisasm");
        String::from_utf8_lossy(&d.stdout).to_string()
    };
    let shipped = sass("fma", &ptx);
    let muladd = sass("muladd", &split("mul.f32", "add.f32"));
    let forbid = sass("rn", &split("mul.rn.f32", "add.rn.f32"));
    assert!(
        !shipped.is_empty(),
        "nvdisasm produced nothing, so the comparison below is between two \
         empty strings and asserts nothing"
    );
    assert_eq!(
        shipped, muladd,
        "the emitted `fma.rn.f32` is NOT byte-identical to the `mul.f32`+\
         `add.f32` it replaced, so this change is not free and the claim that \
         it only makes the artifact truthful is wrong"
    );
    // The control. Without it, three identical strings satisfy the assertion
    // above for a reason that has nothing to do with contraction.
    assert_ne!(
        shipped, forbid,
        "forbidding the fusion produced the SAME instruction stream, so \
         `ptxas` is not contracting here and this fixture measures nothing"
    );
}
