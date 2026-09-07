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
    emit_ptx_at(tag, src, "sm_89")
}

/// The same, at a named target: an artifact's own `.target` and never the
/// build machine's card, which is the bug `ptx_portability.rs` exists to stop.
fn emit_ptx_at(tag: &str, src: &str, target: &str) -> String {
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
        {
            let d = target.trim_start_matches("sm_");
            format!(
                "SM_VERSION={}.{}\nGPU_NAME=FmaGate\nSM_COUNT=66\n",
                &d[..d.len() - 1],
                &d[d.len() - 1..]
            )
        },
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

/// The two subtraction directions, one expression each.
///
/// Each is ASYMMETRIC in the operand that carries the sign - `a`, `b` and `c`
/// are three distinct loads - so negating the wrong one is observable. A
/// fixture reusing a register for two of the three could not tell `fma(a,b,-c)`
/// from `fma(-a,b,c)`.
const SUB_SRC: &str = r#"
kernel probe(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let i: I32 = block_idx_x() * 128 + thread_idx_x();
    let a: F32 = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: F32 = block_ptr2d_load(B, 0, i, N, 1, N);
    let c: F32 = block_ptr2d_load(C, 0, i, N, 1, N);
    block_ptr2d_store(C, 0, i, N, 1, N, a * b - c);
}
fn main() {}
"#;

const RSUB_SRC: &str = r#"
kernel probe(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let i: I32 = block_idx_x() * 128 + thread_idx_x();
    let a: F32 = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: F32 = block_ptr2d_load(B, 0, i, N, 1, N);
    let c: F32 = block_ptr2d_load(C, 0, i, N, 1, N);
    block_ptr2d_store(C, 0, i, N, 1, N, c - a * b);
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

/// A subtraction is the SAME instruction with one operand negated, and which
/// operand carries the sign depends on which side the multiply is on:
///
///   `a*b - c`  ->  fma( a, b, -c)   negate the ADDEND
///   `c - a*b`  ->  fma(-a, b,  c)   negate a MULTIPLICAND
///
/// Recognising `Sub` without negating anything emits `a*b + c` for `a*b - c`,
/// a wrong answer under a green banner - which is why this was left unfused
/// when the `Add` case landed, and why both directions are pinned here rather
/// than one. A fixture that is symmetric in the negated operand cannot see a
/// sign error at all.
///
/// The reason to fuse it after all is that `ptxas` CONTRACTS the subtraction
/// too - `FFMA R9, R9, R8, -R10` for the first shape and
/// `FFMA R9, R9, -R8, R10` for the second - so leaving it split shipped an
/// artifact claiming a rounding the hardware does not perform, which is the
/// whole defect this file exists downstream of. The recorded price was "a
/// `neg.f32` PTX cannot fold"; the `neg` REPLACES the `mul.f32` the `fma`
/// absorbs, so the instruction count is unchanged and the SASS is
/// byte-identical (measured below, and on the three `rope_*` artifacts).
#[test]
fn a_float_multiply_feeding_a_subtract_is_fused_with_the_sign_on_the_right_operand() {
    // `a*b - c`: the ADDEND is negated, and the multiplicands are untouched.
    let b = body(&emit_ptx("sub", SUB_SRC));
    assert!(
        !b.contains("sub.f32"),
        "`a*b - c` still emits a sub.f32, so it claims a rounding ptxas does \
         not perform:\n{b}"
    );
    let fma = b
        .lines()
        .find(|l| l.trim_start().starts_with("fma.rn.f32"))
        .unwrap_or_else(|| panic!("`a*b - c` was not fused:\n{b}"))
        .to_string();
    let neg = b
        .lines()
        .find(|l| l.trim_start().starts_with("neg.f32"))
        .unwrap_or_else(|| panic!("`a*b - c` was fused with NO negation, which \
             computes `a*b + c` - a different function:\n{b}"))
        .to_string();
    // The negated register must be the fma's ADDEND (operand 4), not either
    // multiplicand. Asserting only that a `neg` exists is satisfied by
    // negating the wrong operand, which is the wrong function.
    let negged = neg
        .trim()
        .trim_start_matches("neg.f32")
        .trim_end_matches(';')
        .split(',')
        .nth(0)
        .map(|s| s.trim().to_string())
        .expect("neg destination");
    let ops: Vec<String> = fma
        .trim()
        .trim_start_matches("fma.rn.f32")
        .trim_end_matches(';')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    assert_eq!(ops.len(), 4, "unexpected fma operand count in `{fma}`");
    assert_eq!(
        ops[3], negged,
        "`a*b - c` must negate the ADDEND. `{negged}` is not the fma's addend \
         in `{fma}`, so this computes something other than `a*b - c`"
    );
    assert!(
        ops[1] != negged && ops[2] != negged,
        "a multiplicand was negated for `a*b - c` in `{fma}`"
    );
}

#[test]
fn a_subtract_from_a_float_multiply_negates_a_multiplicand_instead() {
    // `c - a*b` is the OTHER direction and needs the sign somewhere else.
    // Without this case, negating the addend unconditionally passes the test
    // above and emits `a*b - c` for `c - a*b` - the sign inverted.
    let b = body(&emit_ptx("rsub", RSUB_SRC));
    assert!(
        !b.contains("sub.f32"),
        "`c - a*b` still emits a sub.f32:\n{b}"
    );
    let fma = b
        .lines()
        .find(|l| l.trim_start().starts_with("fma.rn.f32"))
        .unwrap_or_else(|| panic!("`c - a*b` was not fused:\n{b}"))
        .to_string();
    let neg = b
        .lines()
        .find(|l| l.trim_start().starts_with("neg.f32"))
        .unwrap_or_else(|| panic!("`c - a*b` was fused with no negation:\n{b}"))
        .to_string();
    let negged = neg
        .trim()
        .trim_start_matches("neg.f32")
        .trim_end_matches(';')
        .split(',')
        .nth(0)
        .map(|s| s.trim().to_string())
        .expect("neg destination");
    let ops: Vec<String> = fma
        .trim()
        .trim_start_matches("fma.rn.f32")
        .trim_end_matches(';')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    assert_eq!(ops.len(), 4, "unexpected fma operand count in `{fma}`");
    assert_eq!(
        ops[1], negged,
        "`c - a*b` must negate a MULTIPLICAND. `{negged}` is not the fma's \
         first multiplicand in `{fma}`, so the sign is on the wrong operand"
    );
    assert!(
        ops[3] != negged,
        "the addend was negated for `c - a*b` in `{fma}`, which emits \
         `a*b - c` - the sign inverted"
    );
}

/// The operator gate is a WHITELIST, and nothing else may reach the fusion.
///
/// The shape this recognises is "a `Mul` on one side of a binary operator",
/// and that is true of `a*b / c` and `a*b * c` as well. Widening the gate to
/// every operator emits `fma.rn.f32 d, a, b, c` for a DIVISION - `a*b + c`
/// where the source says `a*b / c`, with no division left in the artifact at
/// all. It compiles clean, `ptxas` accepts it, and it survived every other
/// test in this file: the `Add` and `Sub` cases pin what those two operators
/// do and say nothing about a third.
///
/// This is the over-refusal control read the other way round. Without it,
/// "fuse everything" passes.
#[test]
fn no_operator_other_than_add_and_subtract_reaches_the_fusion() {
    for (name, expr, want) in [
        ("div", "a * b / c", "div."),
        ("mulmul", "a * b * c", "mul.f32"),
    ] {
        let src = format!(
            r#"
kernel probe(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {{
    let i: I32 = block_idx_x() * 128 + thread_idx_x();
    let a: F32 = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: F32 = block_ptr2d_load(B, 0, i, N, 1, N);
    let c: F32 = block_ptr2d_load(C, 0, i, N, 1, N);
    block_ptr2d_store(C, 0, i, N, 1, N, {expr});
}}
fn main() {{}}
"#
        );
        let bd = body(&emit_ptx(name, &src));
        assert!(
            !bd.contains("fma."),
            "`{expr}` was folded into an fma. The fusion is only correct for \
             `+` and `-`; for anything else it computes a different function \
             entirely:\n{bd}"
        );
        assert!(
            bd.contains(want),
            "`{expr}` lost its `{want}`, so it is not being lowered as itself:\n{bd}"
        );
    }
}

/// The subtraction must not disturb the INTEGER path, exactly as the addition
/// does not. An integer `a*b - c` has no fma to fold into and the multiply
/// must stay where it was.
#[test]
fn an_integer_multiply_feeding_a_subtract_stays_exactly_where_it_was() {
    let src = r#"
kernel probe(O: GlobalMemory<U32>, N: I32) {
    let k: I32 = block_idx_x() * block_idx_y() - thread_idx_x();
    block_ptr2d_store(O, 0, 0, N, 1, N, k);
}
fn main() {}
"#;
    let b = body(&emit_ptx("isub", src));
    assert!(
        !b.contains("fma."),
        "an integer `a*b - c` must not be fused:\n{b}"
    );
    assert!(
        b.contains("sub.s32") || b.contains("sub.u32"),
        "the integer subtract disappeared:\n{b}"
    );
    let mul = b.find("mul.lo.s32").expect("no integer multiply emitted");
    let tid = b.find("%tid.x").expect("no thread index emitted");
    assert!(
        mul < tid,
        "the integer multiply moved PAST the instruction computing its \
         subtrahend. That is a reordering, not a fusion:\n{b}"
    );
}

/// `c - a*b` in the INTEGER case must keep its operand ORDER. Subtraction does
/// not commute, so finishing it through the ordinary path with the operands
/// the wrong way round emits `a*b - c` - a wrong answer with no float in it.
#[test]
fn an_integer_subtract_from_a_multiply_keeps_its_operand_order() {
    let src = r#"
kernel probe(O: GlobalMemory<U32>, N: I32) {
    let k: I32 = thread_idx_x() - block_idx_x() * block_idx_y();
    block_ptr2d_store(O, 0, 0, N, 1, N, k);
}
fn main() {}
"#;
    let b = body(&emit_ptx("irsub", src));
    let sub = b
        .lines()
        .find(|l| l.trim_start().starts_with("sub.s32") || l.trim_start().starts_with("sub.u32"))
        .unwrap_or_else(|| panic!("no integer subtract emitted:\n{b}"))
        .to_string();
    let mul = b
        .lines()
        .find(|l| l.trim_start().starts_with("mul.lo.s32"))
        .unwrap_or_else(|| panic!("no integer multiply emitted:\n{b}"))
        .to_string();
    let prod = mul
        .trim()
        .trim_start_matches("mul.lo.s32")
        .trim_end_matches(';')
        .split(',')
        .nth(0)
        .map(|s| s.trim().to_string())
        .expect("mul destination");
    let ops: Vec<String> = sub
        .trim()
        .trim_start_matches("sub.s32")
        .trim_start_matches("sub.u32")
        .trim_end_matches(';')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    assert_eq!(ops.len(), 3, "unexpected sub operand count in `{sub}`");
    assert_eq!(
        ops[2], prod,
        "`c - a*b` emitted `{sub}` with the product as the MINUEND. \
         Subtraction does not commute; that computes `a*b - c`"
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

/// The hand-written kernel bodies state it too.
///
/// Expression lowering never reaches these: each is a literal PTX string in
/// `emit_rmsnorm_residual_kernel`, `emit_rope_kernel` and the int8 GEMM's
/// dequantise epilogue. They were the whole of the remaining contraction set.
///
/// ALL FIVE now go to zero: forbidding the fusion with `.rn` changes not one
/// byte of any of their SASS, because there is no longer a fusion to forbid.
/// That empties the corpus-wide contraction set, so every artifact this
/// repository ships states every rounding the hardware performs.
///
/// The rope counts are TWO per rotated pair - `out0 = x0*cos - x1*sin` and
/// `out1 = x0*sin + x1*cos` - where they used to be one, because the subtract
/// half now states its fusion as well.
#[test]
fn the_hand_written_kernels_state_their_fusions() {
    let want: &[(&str, usize)] = &[
        ("rmsnorm_residual_4096", 9),
        ("int8_gemm_scaled", 4),
        ("rope_64", 2),
        ("rope_128", 4),
        ("rope_256", 8),
    ];
    for (k, n) in want {
        let shipped = std::fs::read_to_string(repo().join(format!("tests/{k}.ptx"))).expect(k);
        let got = body(&shipped).matches("fma.rn.f32").count();
        assert_eq!(
            got, *n,
            "tests/{k}.ptx states {got} fusions where it should state {n}; a \
             `mul.f32` feeding an `add.f32` claims a rounding `ptxas` does not \
             perform"
        );
        // ...and the EMITTER must still produce that file. Reading the shipped
        // artifact alone is satisfied by an emitter that has drifted away from
        // it: reverting one of these four sites left every suite green, because
        // the committed `.ptx` still said `fma` and nothing compared the two.
        //
        // Byte-identity is claimable for exactly these five: emitted PTX
        // varies with the hardware profile and the autotune cache in general,
        // and these were checked to reproduce at their own declared target.
        let target = shipped
            .lines()
            .find_map(|l| l.trim().strip_prefix(".target "))
            .expect("no .target")
            .trim()
            .to_string();
        let src = std::fs::read_to_string(repo().join(format!("tests/{k}.ysu"))).expect("source");
        let fresh = emit_ptx_at(k, &src, &target);
        assert_eq!(
            fresh, shipped,
            "tests/{k}.ptx is not what the emitter produces from its source, so \
             the shipped kernel and the compiler have drifted apart"
        );
    }
}

/// The rope rotation states its subtract, with the sign on the ADDEND.
///
/// `x0*cos - x1*sin` is `fma.rn.f32 d, x0, cos, -(x1*sin)`, and `ptxas`
/// contracts it either way - into `FFMA R7, R4, R5, -R7`, with the negation as
/// an operand modifier the SASS encoding carries for free. So leaving it split
/// shipped an artifact claiming a rounding the machine does not perform.
///
/// The negation must be on the ADDEND and not on a multiplicand. Both compute
/// the same value, but `fma(-x1, sin, x0*cos)` is NOT byte-identical here - it
/// reorders the schedule - so which side carries the sign was measured rather
/// than chosen, and this pins the measured one.
///
/// Without this the count test above is satisfied by "fuse everything", which
/// is the over-recognition that computes `a*b + c` for `a*b - c`.
#[test]
fn the_rope_rotation_states_its_subtract_on_the_addend() {
    let mut checked = 0;
    for k in ["rope_64", "rope_128", "rope_256"] {
        let b = body(&std::fs::read_to_string(repo().join(format!("tests/{k}.ptx"))).expect(k));
        assert!(
            !b.contains("sub.f32"),
            "tests/{k}.ptx still splits the rotation's `x0*cos - x1*sin` half, \
             so it claims a rounding `ptxas` does not perform"
        );
        // Every `neg.f32` destination must be the addend of some `fma.rn.f32`,
        // and never one of its multiplicands. Asserting only that a `neg`
        // exists is satisfied by negating the wrong operand.
        let negs: Vec<String> = b
            .lines()
            .filter(|l| l.trim_start().starts_with("neg.f32"))
            .map(|l| {
                l.trim()
                    .trim_start_matches("neg.f32")
                    .split(',')
                    .next()
                    .unwrap()
                    .trim()
                    .to_string()
            })
            .collect();
        assert!(!negs.is_empty(), "tests/{k}.ptx has no neg.f32, so the \
             subtract was fused without its sign - a different function");
        let fmas: Vec<Vec<String>> = b
            .lines()
            .filter(|l| l.trim_start().starts_with("fma.rn.f32"))
            .map(|l| {
                l.trim()
                    .trim_start_matches("fma.rn.f32")
                    .trim_end_matches(';')
                    .split(',')
                    .map(|o| o.trim().to_string())
                    .collect()
            })
            .collect();
        for n in &negs {
            assert!(
                fmas.iter().any(|o| o.len() == 4 && &o[3] == n),
                "tests/{k}.ptx negates {n}, which is not the addend of any \
                 fma.rn.f32 - the sign is on the wrong operand"
            );
            assert!(
                !fmas.iter().any(|o| o.len() == 4 && (&o[1] == n || &o[2] == n)),
                "tests/{k}.ptx negates {n} and uses it as a MULTIPLICAND. \
                 That computes the same value and is not byte-identical here"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 7, "expected 1 + 2 + 4 rotated pairs across the three \
         rope kernels; the fixture list or the schedule moved");
}

/// Forbidding the fusion is a NO-OP for every kernel that states all of it.
///
/// This is the claim the count above is a proxy for, measured: if `.rn` cannot
/// change the SASS, the artifact and the machine agree about every rounding.
///
/// All FIVE hand-written kernels are here now. The three `rope_*` joined when
/// the rotation's subtract half started stating its own fusion, which empties
/// the corpus-wide contraction set that `tools/ptxas_tval/contract.py`
/// measures - so this is the whole of it rather than a subset.
#[test]
fn forbidding_the_fusion_changes_nothing_where_it_is_all_stated() {
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("SKIP: no ptxas");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_fma_noop_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let mut checked = 0;
    for k in [
        "rmsnorm_residual_4096",
        "int8_gemm_scaled",
        "rope_64",
        "rope_128",
        "rope_256",
    ] {
        let src = std::fs::read_to_string(repo().join(format!("tests/{k}.ptx"))).expect(k);
        let arch = src
            .lines()
            .find_map(|l| l.trim().strip_prefix(".target "))
            .expect("no .target")
            .trim()
            .to_string();
        // Rewrite every plain f32 mul/add/sub to its .rn form, which forbids
        // contraction and nothing else.
        let rn: String = src
            .lines()
            .map(|l| {
                let t = l.trim_start();
                for op in ["mul.f32 ", "add.f32 ", "sub.f32 "] {
                    if t.starts_with(op) {
                        return l.replacen(op, &format!("{}rn.f32 ", &op[..4]), 1);
                    }
                }
                l.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let sass = |name: &str, text: &str| -> String {
            let p = dir.join(format!("{k}_{name}.ptx"));
            std::fs::write(&p, text).expect("write");
            let cu = dir.join(format!("{k}_{name}.cubin"));
            let o = Command::new("ptxas")
                .arg(format!("-arch={arch}"))
                .arg("-o")
                .arg(&cu)
                .arg(&p)
                .output()
                .expect("ptxas");
            assert!(
                o.status.success(),
                "ptxas rejected {k} ({name}):\n{}",
                String::from_utf8_lossy(&o.stderr)
            );
            let d = Command::new("nvdisasm")
                .arg("-c")
                .arg(&cu)
                .output()
                .expect("nvdisasm");
            String::from_utf8_lossy(&d.stdout).to_string()
        };
        let plain = sass("plain", &src);
        assert!(!plain.is_empty(), "nvdisasm produced nothing for {k}");
        assert_eq!(
            plain,
            sass("rn", &rn),
            "forbidding the fusion changes {k}'s SASS, so it still contains a \
             contraction its PTX does not state"
        );
        checked += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(checked, 5, "the kernel list emptied itself");
}
