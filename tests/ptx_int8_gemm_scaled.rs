//! The int8 Tensor Core GEMM with a fused scaled epilogue, run on the device.
//!
//! `C = acc * Sa[row] * Sb[col] + Bias[col]`, where `acc` is the exact int32
//! `mma.sync.m16n8k32` accumulation. This is the shape `w8a8_matmul` needs and
//! the largest single item in the launch census: `_w8a8_gemm` is 145 launches
//! and 38.7% of a decode step.
//!
//! **Assembling proves nothing here** -- gotcha #8. A wrong row/column mapping
//! in the epilogue produces PTX that `ptxas` accepts, launches cleanly, and
//! returns plausible numbers, because scaling by the wrong row's scale is still
//! a scale. Only comparing against a CPU reference settles it.
//!
//! Two comparisons, because they catch different things:
//!
//!   * **arbitrary scales, compared to a few ULP.** This is the indexing test:
//!     a transposed or off-by-8 scale lookup is wrong by orders of magnitude,
//!     not by an ULP. A tolerance is right here because `ptxas` is free to
//!     contract the epilogue's `mul` + `add` into an `fma`, which changes the
//!     last bit legitimately.
//!   * **power-of-two scales and integral bias, compared BIT-EXACTLY.** Then
//!     every epilogue operation is exact in f32 whether or not it is contracted,
//!     so the tolerance can be removed entirely and the comparison becomes a
//!     statement about the value rather than about its neighbourhood.

use y::cuda_runtime::CudaContext;

const MM: usize = 64; // must match the @tile in tests/int8_gemm_scaled.ysu
const NN: usize = 32;
const KK: usize = 128;

fn ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        use std::path::Path;
        use std::process::Command;
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut bin = std::env::current_exe().unwrap();
        bin.pop();
        if bin.ends_with("deps") {
            bin.pop();
        }
        let out = Command::new(bin.join("Y"))
            .arg(repo.join("tests/int8_gemm_scaled.ysu"))
            .arg("--emit-ptx")
            .current_dir(repo)
            .output()
            .expect("run Y");
        assert!(
            out.status.success(),
            "int8_gemm_scaled.ysu did not compile:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        std::fs::read_to_string(repo.join("tests/int8_gemm_scaled.ptx")).expect("no .ptx")
    })
}

fn operands() -> (Vec<i8>, Vec<i8>) {
    let a: Vec<i8> = (0..MM * KK)
        .map(|i| (((i * 31 + 5) % 251) as i32 - 125) as i8)
        .collect();
    let b: Vec<i8> = (0..NN * KK)
        .map(|i| (((i * 67 + 13) % 251) as i32 - 125) as i8)
        .collect();
    (a, b)
}

fn accum(a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0i32; MM * NN];
    for m in 0..MM {
        for n in 0..NN {
            c[m * NN + n] = (0..KK).map(|k| a[m * KK + k] as i32 * b[n * KK + k] as i32).sum();
        }
    }
    c
}

fn run(ctx: &CudaContext, a: &[i8], b: &[i8], sa: &[f32], sb: &[f32], bias: &[f32]) -> Vec<f32> {
    let module = ctx
        .load_ptx(ptx(), "int8_gemm_scaled")
        .expect("PTX failed to load");
    let d_a = ctx.alloc(MM * KK).unwrap();
    let d_b = ctx.alloc(NN * KK).unwrap();
    let d_sa = ctx.alloc(MM * 4).unwrap();
    let d_sb = ctx.alloc(NN * 4).unwrap();
    let d_bi = ctx.alloc(NN * 4).unwrap();
    let d_c = ctx.alloc(MM * NN * 4).unwrap();

    let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_sa, 0, &f32b(sa)).unwrap();
    ctx.memcpy_htod_at(&d_sb, 0, &f32b(sb)).unwrap();
    ctx.memcpy_htod_at(&d_bi, 0, &f32b(bias)).unwrap();
    // POISON, not zero. This shape STORES rather than reducing (the grid is 2-D,
    // so K is not split and each lane owns its outputs outright), so a kernel
    // that writes nothing must fail loudly instead of inheriting a zero that
    // might coincidentally match a bias-only expectation.
    ctx.memset_u8(&d_c, 0xAB).unwrap();

    let args = vec![
        d_a.device_ptr(),
        d_b.device_ptr(),
        d_sa.device_ptr(),
        d_sb.device_ptr(),
        d_bi.device_ptr(),
        d_c.device_ptr(),
    ];
    ctx.launch(&module, ((NN / 8) as u32, (MM / 16) as u32, 1), (32, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let mut raw = vec![0u8; MM * NN * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_scaled_epilogue_matches_a_cpu_reference() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver -- the scaled int8 GEMM was emitted but not executed.");
        return;
    };
    let (a, b) = operands();
    let acc = accum(&a, &b);
    // Deliberately NOT uniform: a constant scale vector would pass with the
    // row and column lookups swapped, and with an off-by-8 row index.
    let sa: Vec<f32> = (0..MM).map(|i| 1e-3 + i as f32 * 7e-5).collect();
    let sb: Vec<f32> = (0..NN).map(|i| 2e-3 - i as f32 * 3e-5).collect();
    let bias: Vec<f32> = (0..NN).map(|i| i as f32 * 0.25 - 4.0).collect();

    let got = run(&ctx, &a, &b, &sa, &sb, &bias);
    let mut worst = 0f32;
    let mut first = None;
    for m in 0..MM {
        for n in 0..NN {
            let want = acc[m * NN + n] as f32 * sa[m] * sb[n] + bias[n];
            let g = got[m * NN + n];
            let rel = (g - want).abs() / want.abs().max(1e-6);
            if rel > worst {
                worst = rel;
                if rel > 1e-5 && first.is_none() {
                    first = Some(format!("C[{m}][{n}]: GPU {g}, expected {want}"));
                }
            }
        }
    }
    assert!(
        worst < 1e-5,
        "worst relative error {worst:e}; {}",
        first.unwrap_or_default()
    );
}

#[test]
fn with_dyadic_scales_the_epilogue_is_bit_exact() {
    // Powers of two and integral bias: every epilogue operation is exact in
    // f32, contracted or not, so the tolerance can go. This is the difference
    // between "the answer is close" and "the answer is the answer".
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let (a, b) = operands();
    let acc = accum(&a, &b);
    let sa: Vec<f32> = (0..MM).map(|i| (2f32).powi(-((i % 5) as i32) - 8)).collect();
    let sb: Vec<f32> = (0..NN).map(|i| (2f32).powi(-((i % 3) as i32) - 4)).collect();
    let bias: Vec<f32> = (0..NN).map(|i| (i as f32) - 16.0).collect();

    let got = run(&ctx, &a, &b, &sa, &sb, &bias);
    let mut bad = 0usize;
    let mut first = None;
    for m in 0..MM {
        for n in 0..NN {
            let want = acc[m * NN + n] as f32 * sa[m] * sb[n] + bias[n];
            if got[m * NN + n].to_bits() != want.to_bits() {
                bad += 1;
                if first.is_none() {
                    first = Some(format!(
                        "C[{m}][{n}]: GPU {} ({:#x}), expected {want} ({:#x})",
                        got[m * NN + n],
                        got[m * NN + n].to_bits(),
                        want.to_bits()
                    ));
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {} differ; {}", MM * NN, first.unwrap_or_default());
}

#[test]
fn the_epilogue_is_not_a_no_op() {
    // The control. Every assertion above compares against a reference that
    // includes the scales, so a kernel applying NO epilogue would fail them --
    // but only because the reference is right. This asserts the weaker,
    // independent thing: changing a scale changes the output, and changing
    // one ROW's scale changes only that row.
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let (a, b) = operands();
    let sa: Vec<f32> = vec![1.0 / 256.0; MM];
    let sb: Vec<f32> = vec![1.0 / 64.0; NN];
    let bias: Vec<f32> = vec![0.0; NN];
    let base = run(&ctx, &a, &b, &sa, &sb, &bias);

    let mut sa2 = sa.clone();
    sa2[7] *= 2.0;
    let moved = run(&ctx, &a, &b, &sa2, &sb, &bias);

    let changed: Vec<usize> = (0..MM)
        .filter(|&m| (0..NN).any(|n| base[m * NN + n] != moved[m * NN + n]))
        .collect();
    assert_eq!(
        changed,
        vec![7],
        "doubling Sa[7] must change row 7 and only row 7; rows that moved: {changed:?}"
    );
}
