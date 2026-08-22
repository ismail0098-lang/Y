//! M3: the same GPU kernel, the same inputs, **every launch geometry** — one
//! bit-identical answer.
//!
//! This is the product claim of `docs/deterministic_inference.md` stated on the
//! hardware the market cares about. `docs/deterministic_inference.md` M0 showed
//! it on the CPU; this shows it where the nondeterminism people actually
//! complain about lives.
//!
//! ## Why this is not a trivial result
//!
//! The kernel splits K across `gridDim.z` and combines the partial sums with
//! `red.global.add.s32`. **An atomic add is the canonical reason GPU results
//! are not reproducible** — CTAs finish in whatever order the scheduler
//! chooses, so with a float accumulator the answer changes between launches of
//! the *identical* binary on the *identical* input. That is the bug behind
//! "why does my eval score move by 0.1 when I rerun it".
//!
//! Integer addition is associative and commutative, so the same atomic is
//! order-independent by construction. Nothing is serialised, no ordering is
//! imposed, and no determinism flag is set: the reduction simply cannot care.
//!
//! ## What is swept
//!
//! - **Split-K factor** `gridDim.z` ∈ {1, 2, 3, 5, 8, 16, 32} — a different
//!   number of CTAs contributing to every output element, striped so no
//!   divisibility is required.
//! - **Repeat launches** at each geometry, because atomic contention order is
//!   not reproducible run to run; a single launch per geometry would not
//!   exercise the property being claimed.
//!
//! ## The control
//!
//! Bit-identity is only interesting if the same structure in floating point
//! would *not* be. The control adds the same partial sums in f32 in several
//! orders on the CPU and asserts they DISAGREE — without it, this file would
//! keep passing on data too benign to reassociate, which is the trap the first
//! version of `cpu_gemm_vnni_micro.rs`'s control fell into.
//!
//! Requires a CUDA driver; skipped with a notice otherwise.
//!
//! Run with:  cargo test --release --test gpu_batch_invariance -- --nocapture

use std::path::Path;
use std::process::Command;

const M: usize = 64;
const N: usize = 32;
const K: usize = 128;

fn compile_fixture() -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let out = Command::new(bin.join("Y"))
        .arg(repo.join("tests/int8_gemm.ysu"))
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "int8_gemm.ysu did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(repo.join("tests/int8_gemm.ptx")).expect("no .ptx")
}

#[test]
fn the_int8_gemm_is_bit_identical_across_every_split_k_geometry() {
    use y::cuda_runtime::CudaContext;

    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — GPU batch invariance was not demonstrated.");
        return;
    };
    let ptx = compile_fixture();
    let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");

    let a: Vec<i8> = (0..M * K)
        .map(|i| (((i * 31 + 5) % 251) as i32 - 125) as i8)
        .collect();
    let b: Vec<i8> = (0..N * K)
        .map(|i| (((i * 67 + 13) % 251) as i32 - 125) as i8)
        .collect();

    let d_a = ctx.alloc(M * K).unwrap();
    let d_b = ctx.alloc(N * K).unwrap();
    let d_c = ctx.alloc(M * N * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>())
        .unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>())
        .unwrap();

    let run = |splits: u32| -> Vec<i32> {
        ctx.memset_u8(&d_c, 0).unwrap();
        let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
        ctx.launch(
            &module,
            ((N / 8) as u32, (M / 16) as u32, splits),
            (32, 1, 1),
            0,
            &args,
        )
        .expect("launch failed");
        ctx.synchronize().expect("kernel did not complete");
        let mut raw = vec![0u8; M * N * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        (0..M * N)
            .map(|i| {
                i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
            })
            .collect()
    };

    // Ground truth, so "all geometries agree" cannot pass by all being wrong
    // in the same way.
    let mut want = vec![0i32; M * N];
    for m in 0..M {
        for n in 0..N {
            want[m * N + n] = (0..K)
                .map(|k| a[m * K + k] as i32 * b[n * K + k] as i32)
                .sum();
        }
    }

    let reference = run(1);
    assert_eq!(
        reference, want,
        "the single-split launch does not match the CPU reference; nothing below means anything"
    );

    // Repeat launches matter: atomic contention order is not reproducible run
    // to run, so one launch per geometry would not exercise the claim.
    for splits in [1u32, 2, 3, 5, 8, 16, 32] {
        for rep in 0..3 {
            let got = run(splits);
            let diffs = got
                .iter()
                .zip(&reference)
                .filter(|(g, r)| g != r)
                .count();
            assert_eq!(
                diffs, 0,
                "split-K = {splits} (repeat {rep}) differs from split-K = 1 on {diffs} of {} \
                 elements. Exact integer accumulation is supposed to make the launch geometry \
                 invisible — this is the product claim failing.",
                M * N
            );
        }
        eprintln!("split-K {splits:2}: bit-identical across 3 launches");
    }
}

/// The control: the same partial-sum structure in f32 must NOT be reproducible.
///
/// Split-K in floating point means each CTA's partial sum is rounded before the
/// combine, and the combine order is whatever the scheduler chose. Reproducing
/// that on the CPU — same partials, different combine orders — must disagree,
/// or the result above is a statement about this dataset rather than about
/// exact accumulation.
#[test]
fn the_same_split_structure_in_f32_is_not_reproducible() {
    let a: Vec<f32> = (0..M * K)
        .map(|i| (((i * 31 + 5) % 251) as f32 - 125.0) * 1.000_001)
        .collect();
    let b: Vec<f32> = (0..N * K)
        .map(|i| (((i * 67 + 13) % 251) as f32 - 125.0) * 0.999_997)
        .collect();

    // Partial sums for one output element under a striped split, combined in
    // two different orders — exactly what different `gridDim.z` values and
    // different CTA completion orders produce.
    let mut disagreements = 0usize;
    for m in 0..M {
        for n in 0..N {
            let partial = |splits: usize| -> Vec<f32> {
                (0..splits)
                    .map(|z| {
                        let mut acc = 0.0f32;
                        let mut k = z;
                        while k < K {
                            acc += a[m * K + k] * b[n * K + k];
                            k += splits;
                        }
                        acc
                    })
                    .collect()
            };
            let p4 = partial(4);
            let fwd: f32 = p4.iter().fold(0.0, |s, v| s + v);
            let rev: f32 = p4.iter().rev().fold(0.0, |s, v| s + v);
            let p8: f32 = partial(8).iter().fold(0.0, |s, v| s + v);
            if fwd != rev || fwd != p8 {
                disagreements += 1;
            }
        }
    }
    assert!(
        disagreements > 0,
        "the f32 control did not disagree under ANY reordering, so the GPU \
         bit-identity result proves nothing about this data"
    );
    eprintln!(
        "control: f32 split-K disagrees on {disagreements} of {} elements",
        M * N
    );
}
