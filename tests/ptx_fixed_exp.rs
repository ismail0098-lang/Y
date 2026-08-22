//! The integer `exp2` on the device must agree with `y::fixed_exp` **bit for
//! bit**, because that agreement is the whole cross-architecture claim.
//!
//! `ex2.approx.f32` is a fixed-function unit whose rounding belongs to the SM
//! generation. It is deterministic per launch and per architecture, which is
//! enough for batch invariance (the step is per-element and cannot see the
//! partition) and not enough for `docs/deterministic_inference.md` M5's "same
//! answer on two different GPUs".
//!
//! `src/fixed_exp.rs` computes the same function in pure integer arithmetic.
//! Integers are specified, not approximated: any machine that implements 32-
//! and 64-bit integer ops produces the identical bit pattern. This file is
//! what makes that a checked property rather than an assertion — a device
//! against a host, two completely different implementations of the same
//! integer recipe.
//!
//! If it ever fails, the two have diverged and the M5 claim is void until they
//! agree again. Do not "fix" it by loosening the comparison.
//!
//! Requires a CUDA driver; skipped with a notice otherwise.

use y::cuda_runtime::CudaContext;
use y::fixed_exp::{exp2_neg_q16_16, EXP2_TABLE, LN2_Q30, RECIP6_Q32};

fn ptx(n: usize) -> String {
    let table = EXP2_TABLE
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    SRC.replace("$TABLE", &table)
        .replace("$LN2", &LN2_Q30.to_string())
        .replace("$R6", &RECIP6_Q32.to_string())
        .replace("$N", &n.to_string())
}

const SRC: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.const .align 4 .u32 exp2_tbl[64] = { $TABLE };

.visible .entry fixed_exp_probe(
    .param .u64 a0,   // T   u32 [N]  argument, Q16.16, non-negative
    .param .u64 a1    // Out u32 [N]  result,   Q0.28
)
{
    .reg .pred %p<4>;
    .reg .u32  %r<32>;
    .reg .u64  %rd<32>;

    ld.param.u64 %rd1, [a0];
    ld.param.u64 %rd2, [a1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, $N;
    @%p1 bra END;

    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.u32 %r5, [%rd4];          // t

    shr.u32 %r6, %r5, 16;               // n = integer part
    mov.u32 %r20, 0;                    // result, 0 if it underflows
    setp.ge.u32 %p2, %r6, 30;
    @%p2 bra STORE;

    and.b32 %r7, %r5, 65535;            // f
    shr.u32 %r8, %r7, 10;               // table index (top 6 bits)
    mul.wide.u32 %rd5, %r8, 4;
    mov.u64 %rd6, exp2_tbl;
    add.s64 %rd7, %rd6, %rd5;
    ld.const.u32 %r9, [%rd7];
    cvt.u64.u32 %rd8, %r9;              // base, Q0.30

    and.b32 %r10, %r7, 1023;            // delta * 2^16
    cvt.u64.u32 %rd9, %r10;
    mov.u64 %rd10, $LN2;
    mul.lo.u64 %rd11, %rd9, %rd10;
    shr.u64 %rd11, %rd11, 16;           // y = delta*ln2, Q0.30

    mul.lo.u64 %rd12, %rd11, %rd11;
    shr.u64 %rd12, %rd12, 30;           // y^2
    mul.lo.u64 %rd13, %rd12, %rd11;
    shr.u64 %rd13, %rd13, 30;           // y^3
    mov.u64 %rd14, $R6;
    mul.lo.u64 %rd15, %rd13, %rd14;
    shr.u64 %rd15, %rd15, 32;           // y^3/6, exact for y^3 < 2^12

    mov.u64 %rd16, 1073741824;          // 1 << 30
    sub.s64 %rd17, %rd16, %rd11;
    shr.u64 %rd18, %rd12, 1;
    add.s64 %rd17, %rd17, %rd18;
    sub.s64 %rd17, %rd17, %rd15;        // corr = 1 - y + y^2/2 - y^3/6

    mul.lo.u64 %rd19, %rd8, %rd17;
    shr.u64 %rd19, %rd19, 30;           // g = 2^30 * 2^-f

    add.u32 %r11, %r6, 2;               // shift: Q0.30 -> Q0.28, then >> n
    sub.u32 %r12, %r11, 1;
    mov.u64 %rd20, 1;
    shl.b64 %rd21, %rd20, %r12;         // rounding half
    add.s64 %rd22, %rd19, %rd21;
    shr.u64 %rd23, %rd22, %r11;
    cvt.u32.u64 %r20, %rd23;
STORE:
    mul.wide.u32 %rd24, %r4, 4;
    add.s64 %rd25, %rd2, %rd24;
    st.global.u32 [%rd25], %r20;
END:
    ret;
}

// The SAME function through the hardware unit, for the portability argument.
// `ex2.approx.f32` is a fixed-function approximation: the PTX ISA gives it a
// tolerance, not a value, so which result inside that tolerance you get is a
// property of the SM generation and not of the input.
.visible .entry hw_exp_probe(
    .param .u64 b0,   // T   u32 [N]  argument, Q16.16, non-negative
    .param .u64 b1    // Out u32 [N]  result,   Q0.28
)
{
    .reg .pred %p<4>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .u64  %rd<16>;

    ld.param.u64 %rd1, [b0];
    ld.param.u64 %rd2, [b1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, $N;
    @%p1 bra HEND;

    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.u32 %r5, [%rd4];          // t, Q16.16

    cvt.rn.f32.u32 %f1, %r5;
    mul.f32 %f2, %f1, 0f37800000;       // * 2^-16
    neg.f32 %f3, %f2;                   // x = -t / 65536
    ex2.approx.f32 %f4, %f3;
    mul.f32 %f5, %f4, 0f4D800000;       // * 2^28
    cvt.rni.u32.f32 %r6, %f5;

    add.s64 %rd5, %rd2, %rd3;
    st.global.u32 [%rd5], %r6;
HEND:
    ret;
}
"#;

/// Every argument that matters: all 2^16 fractional values (so the table and
/// the series are covered exhaustively), every integer part (so the shift and
/// the saturation are covered), and the boundaries.
/// Every argument the integer exp can be reached with, not a sample.
///
/// `exact_attention` clamps its Q16.16 input to [0, 2^30] and the function
/// zeroes everything with `n = t >> 16 >= 30`, so the reachable domain is
/// `0 .. 30 * 65536` -- 1,966,080 values, plus the above-domain edges. That is
/// 8 MB in and 8 MB out; there is no reason to sample it.
///
/// `arguments()` below is the ~100k structured sample, kept for the host-only
/// comparisons where 2M host-side `exp2` calls would dominate the test's
/// runtime for no extra coverage.
fn every_reachable_argument() -> Vec<u32> {
    let mut t: Vec<u32> = (0..(30u32 << 16)).collect();
    t.extend_from_slice(&[30 << 16, (30 << 16) + 1, (1u32 << 30) - 1, 1 << 30, u32::MAX]);
    t
}

fn arguments() -> Vec<u32> {
    let mut t: Vec<u32> = (0..(1u32 << 16)).collect();
    for n in 0..34u32 {
        for k in 0..1024u32 {
            t.push((n << 16) | (k * 64));
        }
        t.push((n << 16) | 0xFFFF);
        t.push(n << 16);
    }
    t.extend_from_slice(&[
        0,
        1,
        0xFFFF,
        29 << 16,
        (29 << 16) | 0xFFFF,
        30 << 16,
        u32::MAX,
    ]);
    t
}

#[test]
fn the_device_integer_exp_agrees_with_the_host_bit_for_bit() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — cross-architecture exp was not demonstrated.");
        return;
    };
    // EXHAUSTIVE, not a sample. This is the cross-architecture claim, and it
    // is cheap: x86-64 and sm_89 are far more different than two NVIDIA cards,
    // so agreeing on every reachable argument is the stronger measurement.
    let t = every_reachable_argument();
    let n = t.len();
    let module = ctx
        .load_ptx(&ptx(n), "fixed_exp_probe")
        .expect("fixed_exp_probe failed to load");

    let d_t = ctx.alloc(n * 4).unwrap();
    let d_o = ctx.alloc(n * 4).unwrap();
    let raw: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
    ctx.memcpy_htod_at(&d_t, 0, &raw).unwrap();
    ctx.memset_u8(&d_o, 0xAB).unwrap();

    let args = vec![d_t.device_ptr(), d_o.device_ptr()];
    ctx.launch(
        &module,
        ((n as u32).div_ceil(256), 1, 1),
        (256, 1, 1),
        0,
        &args,
    )
    .expect("launch failed");
    ctx.synchronize().unwrap();

    let mut out = vec![0u8; n * 4];
    ctx.memcpy_dtoh_at(&mut out, &d_o, 0).unwrap();
    let got: Vec<u32> = out
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut mismatches = 0usize;
    let mut first = None;
    for (i, &arg) in t.iter().enumerate() {
        let want = exp2_neg_q16_16(arg);
        if got[i] != want {
            mismatches += 1;
            first.get_or_insert((arg, want, got[i]));
        }
    }
    if let Some((arg, want, g)) = first {
        panic!(
            "device and host integer exp2 disagree on {mismatches} of {n} arguments \
             (first: t = {arg}, host {want}, device {g}). The cross-architecture \
             determinism claim rests on these being the same computation — do not \
             loosen this comparison to make it pass."
        );
    }
    eprintln!("device == host on all {n} arguments, bit for bit");

    // A control: the comparison must be capable of failing. If the device
    // buffer were never written, every entry would still be the 0xABABABAB
    // fill, so a vacuous pass is not possible -- but assert the outputs are
    // actually varied, in case the kernel returned a constant.
    let distinct = {
        let mut v = got.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert!(
        distinct > 1000,
        "the device produced only {distinct} distinct results over {n} arguments; \
         the kernel is probably not computing anything"
    );
    eprintln!("{distinct} distinct results, so the agreement is not over a constant");
}

/// `ex2.approx.f32` and the integer path must disagree *somewhere*, or there
/// would be nothing to fix and this module would be dead weight.
///
/// This is the justification for the whole file: it measures how much an f32
/// exp2 differs from a correctly-rounded result, which is the size of the
/// cross-GPU exposure being removed.
///
/// **Note what this does NOT touch.** `hw` here is Rust's own f32 `exp2` on the
/// host — a software implementation. The docstring used to call it "the
/// hardware unit", which it is not. The device's `ex2.approx.f32` is measured
/// by `the_hardware_exp_is_implementation_defined` below, and the two are
/// different numbers for a reason that is the whole portability argument.
#[test]
fn the_hardware_approximation_really_is_looser_than_the_integer_one() {
    let mut worst_hw = 0.0f64;
    let mut worst_int = 0.0f64;
    let mut differ = 0usize;
    let total = 1u32 << 16;
    for t in 0..total {
        let x = -(t as f64) / 65536.0;
        let want = (1u64 << 28) as f64 * x.exp2();
        // What the f32 path produces, rounded to the same fixed point.
        let hw = ((-(t as f32) / 65536.0).exp2() * 268_435_456.0).round() as u32;
        let int = exp2_neg_q16_16(t);
        worst_hw = worst_hw.max((hw as f64 - want).abs());
        worst_int = worst_int.max((int as f64 - want).abs());
        if hw != int {
            differ += 1;
        }
    }
    eprintln!("f32 exp2 -> Q0.28 : worst {worst_hw:.2} ulp");
    eprintln!("integer exp2      : worst {worst_int:.2} ulp");
    eprintln!("they differ on {differ} of {total} arguments");
    assert!(
        differ > 0,
        "the f32 and integer paths agree everywhere, so there is no cross-GPU \
         exposure to remove and src/fixed_exp.rs is unnecessary"
    );
    assert!(
        worst_int <= worst_hw,
        "the integer exp ({worst_int:.2} ulp) is LOOSER than the f32 one \
         ({worst_hw:.2} ulp) it replaces -- it would be trading accuracy for \
         portability rather than getting both"
    );
}

/// The portability argument, made on one GPU.
///
/// Finding 11 showed that a **fixed-order float** path is batch-invariant too,
/// so exactness is not what buys determinism. The obvious next challenge is
/// "then write a fast fixed-order float kernel and skip the quantisation" — and
/// the answer is that a float softmax still needs a transcendental, while
/// `ex2.approx.f32` is specified by a *tolerance*, not a value. Which result
/// inside that tolerance you get is a property of the SM generation.
///
/// A second GPU would demonstrate that directly and there is only one here. What
/// CAN be shown on one card is the premise: the device's approximation and a
/// software f32 `exp2` are two implementations of the same nominal function, and
/// they disagree. A value that two implementations disagree about is not pinned
/// by the specification, so nothing obliges the next architecture to reproduce
/// it — which is exactly the exposure `src/fixed_exp.rs` removes, and which the
/// integer path is separately shown to have no trace of (`device == host on all
/// arguments, bit for bit`, above).
///
/// This is evidence for the premise, not a measurement of cross-architecture
/// divergence. Do not report it as the latter.
#[test]
fn the_hardware_exp_is_implementation_defined() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the hardware exp was not sampled.");
        return;
    };
    let t = arguments();
    let n = t.len();
    let module = ctx
        .load_ptx(&ptx(n), "hw_exp_probe")
        .expect("hw_exp_probe failed to load");

    let d_t = ctx.alloc(n * 4).unwrap();
    let d_o = ctx.alloc(n * 4).unwrap();
    let raw: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
    ctx.memcpy_htod_at(&d_t, 0, &raw).unwrap();
    ctx.memset_u8(&d_o, 0xAB).unwrap();

    let args = vec![d_t.device_ptr(), d_o.device_ptr()];
    ctx.launch(&module, ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().unwrap();

    let mut out = vec![0u8; n * 4];
    ctx.memcpy_dtoh_at(&mut out, &d_o, 0).unwrap();
    let dev: Vec<u32> = out
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // The same computation in host f32: a DIFFERENT implementation of `exp2`.
    let mut differ = 0usize;
    let mut worst_gap = 0i64;
    let mut worst_dev_err = 0.0f64;
    for (i, &arg) in t.iter().enumerate() {
        let host = ((-(arg as f32) / 65536.0).exp2() * 268_435_456.0).round() as u32;
        if dev[i] != host {
            differ += 1;
            worst_gap = worst_gap.max((dev[i] as i64 - host as i64).abs());
        }
        let want = (1u64 << 28) as f64 * (-(arg as f64) / 65536.0).exp2();
        worst_dev_err = worst_dev_err.max((dev[i] as f64 - want).abs());
    }
    eprintln!(
        "device ex2.approx.f32 vs host f32 exp2: differ on {differ} of {n} \
         arguments, worst gap {worst_gap} ulp of Q0.28"
    );
    eprintln!("device ex2.approx.f32 vs correctly rounded: worst {worst_dev_err:.2} ulp");

    let distinct = {
        let mut v = dev.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert!(
        distinct > 1000,
        "the device produced only {distinct} distinct results; the hardware \
         probe is probably not computing anything"
    );
    assert!(
        differ > 0,
        "the device's ex2.approx.f32 agrees with host f32 exp2 on all {n} \
         arguments. That would make the float path reproducible across these \
         two implementations, and the portability argument for the integer exp \
         would need a different justification than this test provides"
    );
    // And the contrast that gives the number meaning: the integer path has no
    // such disagreement, which the test above asserts bit for bit.
    assert!(
        worst_dev_err > 1.0,
        "the hardware approximation is within 1 ulp of correctly rounded, so it \
         is effectively pinned and there is little cross-architecture exposure \
         to remove"
    );
}

/// The portability argument as a STATIC property, on any machine with a CUDA
/// toolkit and no GPU at all.
///
/// Cross-architecture identity is the one headline claim in
/// `docs/bit_identical_decode.md` with no measurement behind it, and a second
/// GPU is what it needs. But most of it does not need one: if every instruction
/// in the emitted SASS is exactly specified by the ISA, the result is
/// determined, and the only instructions that are *not* exactly specified are
/// the approximate ones — the multi-function unit (`MUFU`), which computes
/// `ex2`/`lg2`/`rcp`/`sqrt` to a tolerance rather than to a value.
///
/// So: assemble both probes and assert the integer one contains no `MUFU` and
/// no floating-point instruction, while the float one contains a `MUFU`. The
/// second half is what stops this passing vacuously — if the detector saw
/// nothing in either kernel it would report a clean integer path forever.
///
/// This does not prove two architectures agree; it proves the integer path
/// contains nothing they are permitted to disagree about.
#[test]
fn the_integer_exp_compiles_to_no_architecture_defined_instruction() {
    let n = 4096usize;
    let dir = std::env::temp_dir().join("y_fixed_exp_sass");
    let _ = std::fs::create_dir_all(&dir);
    let ptx_path = dir.join("probe.ptx");
    let cubin = dir.join("probe.cubin");
    if std::fs::write(&ptx_path, ptx(n)).is_err() {
        eprintln!("SKIP: could not write the probe PTX");
        return;
    }
    let asm = std::process::Command::new("ptxas")
        .args(["-arch=sm_89"])
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin)
        .output();
    let Ok(asm) = asm else {
        eprintln!("SKIP: ptxas not found — the SASS property was not checked.");
        return;
    };
    assert!(
        asm.status.success(),
        "the probe module no longer assembles: {}",
        String::from_utf8_lossy(&asm.stderr)
    );
    let dump = std::process::Command::new("cuobjdump")
        .arg("--dump-sass")
        .arg(&cubin)
        .output();
    let Ok(dump) = dump else {
        eprintln!("SKIP: cuobjdump not found — the SASS property was not checked.");
        return;
    };
    let sass = String::from_utf8_lossy(&dump.stdout).to_string();

    // Split by function so the two kernels are judged separately.
    let mut blocks = sass.split("Function : ").skip(1).map(|b| {
        let name = b.lines().next().unwrap_or("").trim().to_string();
        (name, b.to_string())
    });
    let mut integer = None;
    let mut hardware = None;
    for (name, body) in &mut blocks {
        if name.contains("fixed_exp_probe") {
            integer = Some(body);
        } else if name.contains("hw_exp_probe") {
            hardware = Some(body);
        }
    }
    let integer: String = integer.expect("fixed_exp_probe missing from the SASS");
    let hardware: String = hardware.expect("hw_exp_probe missing from the SASS");

    let has_mufu = |b: &str| b.contains("MUFU");
    // Any single-precision op: the SASS mnemonics all start with F.
    let fp_ops = |b: &str| -> Vec<String> {
        b.lines()
            .filter_map(|l| l.split_whitespace().find(|w| {
                w.starts_with("FADD") || w.starts_with("FMUL") || w.starts_with("FFMA")
                    || w.starts_with("FSETP") || w.starts_with("FSEL")
                    || w.starts_with("F2I") || w.starts_with("I2F")
            }))
            .map(|w| w.to_string())
            .collect()
    };

    // The control FIRST: if the detector cannot see the float path's MUFU, a
    // clean integer result means nothing.
    assert!(
        has_mufu(&hardware),
        "no MUFU found in hw_exp_probe, which is an `ex2.approx.f32` kernel. \
         The detector is broken, so the integer result below proves nothing."
    );

    let stray = fp_ops(&integer);
    assert!(
        !has_mufu(&integer) && stray.is_empty(),
        "the integer exp compiled to architecture-defined instructions: \
         MUFU={} floating-point={stray:?}. Its whole purpose is that every \
         instruction is exactly specified by the ISA, so the result cannot \
         depend on the SM generation",
        has_mufu(&integer)
    );
    eprintln!(
        "integer exp SASS: no MUFU, no floating-point instruction; \
         hw exp SASS: MUFU present (the control)"
    );
}
