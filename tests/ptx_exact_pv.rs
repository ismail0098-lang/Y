//! The softmax-weight times value product, accumulated in int64 on the device.
//!
//! `tests/exact_pv.ysu` replaces the PyTorch prototype's digit-split
//! `exact_pv`, which computes the same thing as `ceil(29/dbits)` separate fp32
//! matmuls because an fp32 mantissa cannot hold a Q0.28 weight times an int8
//! activation. The split is a workaround for the accumulator; an int64
//! accumulator has nothing to split.
//!
//! Three things have to be true and each has its own test:
//!
//!   1. it equals an exact integer reference, at both extremes of `p`;
//!   2. an fp32 accumulator over the same data DISAGREES -- otherwise the
//!      exactness is decorative and the digit split never had a reason;
//!   3. the answer does not depend on summation order, which is the property
//!      the whole deterministic path is built on.
//!
//! Mutation-verified, six mutations. Caught: V declared `U8` (zero-extends a
//! negative activation), an `I32` accumulator, dropping the multiply, and a
//! loop stopping one iteration short. NOT caught, and both are confirmations
//! rather than holes: declaring `P` as `I32`, and annotating the loaded weight
//! `I32` instead of `I64`. A Q0.28 weight is at most 2^28 and 2^28 < 2^31, so
//! signedness is unobservable on the reachable domain, and the emitter inserts
//! a `cvt.s64.s32` and still multiplies at 64 bits. Checked in the PTX rather
//! than assumed from the pass/fail.

use y::cuda_runtime::CudaContext;
use y::verified_exact_pv::{ExactPvShape, ValidatedExactPv};
use sha2::{Digest, Sha256};

#[path = "common/validator.rs"]
mod validator;

#[path = "common/pinned.rs"]
mod pinned;

const B: usize = 8; // batch * KV heads
const T: usize = 67; // a key length, deliberately not a power of two
const D: usize = 64; // head_dim
// Query rows per KV row. `exact_attention` reshapes to [b, nkv, rep*q_len, tk],
// so this is >1 for every grouped-query model -- Q = 1 would leave the shared-V
// indexing untested, which is the whole reason the grid has two dimensions.
const Q: usize = 3;
const P_BITS: u32 = 28;

/// Compile and translation-validate once in an isolated directory. Every
/// device test loads the retained cubin through its pinned receipt, so a
/// driver JIT cannot replace the compilation these tests claim to exercise.
fn validated_artifact() -> &'static ValidatedExactPv {
    static ARTIFACT: std::sync::OnceLock<ValidatedExactPv> = std::sync::OnceLock::new();
    ARTIFACT.get_or_init(compile)
}

fn compile() -> ValidatedExactPv {
    use std::path::Path;
    use std::process::Command;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    // A COPY, against a PINNED profile. This compiled the fixture IN PLACE
    // with the repository as working directory - rewriting the committed
    // `tests/exact_pv.ptx` for this machine's card. The pin is sm_89, not the
    // floor: the validated artifact format accepts an unqualified
    // `.target sm_89` only.
    let work = pinned::pinned_scratch("exact_pv", pinned::SM_FP8);
    let source = pinned::copy_fixture(&work, "tests/exact_pv.ysu");
    let out = Command::new(bin.join("Y"))
        .arg(&source)
        .arg("--emit-ptx")
        .current_dir(&work)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "exact_pv.ysu did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx_path = work.join("exact_pv.ptx");
    let ptx = std::fs::read(&ptx_path).expect("no emitted .ptx");
    let expected_ptx_sha256 = format!("{:x}", Sha256::digest(&ptx));
    let bundle = work.join("validated");
    let python = validator::validator_python(repo);
    let out = Command::new(python)
        .arg(repo.join("tools/ptxas_tval/exact_pv_artifact.py"))
        .arg("build")
        .arg(&ptx_path)
        .arg(&bundle)
        .current_dir(&work)
        .output()
        .expect("run exact_pv artifact validator (Y_TVAL_PYTHON must provide z3)");
    assert!(
        out.status.success(),
        "exact_pv translation validation failed; verified execution is forbidden:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    ValidatedExactPv::open(&bundle, Some(&expected_ptx_sha256))
        .expect("validated exact_pv artifact binding failed")
}

/// A cheap deterministic PRNG, so the fixture needs no dev-dependency.
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

/// `p` in [0, 2^28] and `v` in [-127, 127], with both extremes of each forced
/// in: the top bit of `p` is the one a short digit loop drops, and a negative
/// `v` is the one a zero-extending load destroys.
fn inputs(seed: u64) -> (Vec<u32>, Vec<i8>) {
    let mut s = seed;
    let mut p: Vec<u32> =
        (0..B * Q * T).map(|_| (lcg(&mut s) % ((1u64 << P_BITS) + 1)) as u32).collect();
    let mut v: Vec<i8> = (0..B * T * D).map(|_| ((lcg(&mut s) % 255) as i64 - 127) as i8).collect();
    for b in 0..B {
        for q in 0..Q {
            p[(b * Q + q) * T] = 1 << P_BITS; // exactly 2^28
            p[(b * Q + q) * T + 1] = 0;
        }
        for d in 0..D {
            v[(b * T) * D + d] = -127;
            v[(b * T + 1) * D + d] = 127;
        }
    }
    (p, v)
}

fn reference(p: &[u32], v: &[i8]) -> Vec<i64> {
    let mut out = vec![0i64; B * Q * D];
    for b in 0..B {
        for q in 0..Q {
            for d in 0..D {
                let mut acc = 0i64;
                for t in 0..T {
                    acc += p[(b * Q + q) * T + t] as i64 * v[(b * T + t) * D + d] as i64;
                }
                out[(b * Q + q) * D + d] = acc;
            }
        }
    }
    out
}

fn run(ctx: &CudaContext, p: &[u32], v: &[i8]) -> Vec<i64> {
    let module = ctx.load_checked_exact_pv(validated_artifact())
        .expect("validated exact_pv cubin failed to load; PTX JIT is forbidden");
    let d_p = ctx.alloc(p.len() * 4).unwrap();
    let d_v = ctx.alloc(v.len()).unwrap();
    let mut d_o = ctx.alloc(B * Q * D * 8).unwrap();

    let raw_p: Vec<u8> = p.iter().flat_map(|x| x.to_le_bytes()).collect();
    let raw_v: Vec<u8> = v.iter().map(|x| *x as u8).collect();
    ctx.memcpy_htod_at(&d_p, 0, &raw_p).unwrap();
    ctx.memcpy_htod_at(&d_v, 0, &raw_v).unwrap();
    ctx.memset_u8(&d_o, 0xAB).unwrap();

    let shape = ExactPvShape::new(B as i64, Q as i64, T as i64, D as i64)
        .expect("device fixture must satisfy the arithmetic theorem's domain");
    ctx.launch_checked_exact_pv(&module, shape, &d_p, &d_v, &mut d_o)
        .expect("checked exact_pv launch failed");

    let mut bytes = vec![0u8; B * Q * D * 8];
    ctx.memcpy_dtoh_at(&mut bytes, &d_o, 0).unwrap();
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_device_kernel_equals_an_exact_integer_reference() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver -- verified exact_pv was not executed.");
        return;
    };
    let (p, v) = inputs(0x5eed_1234);
    let got = run(&ctx, &p, &v);
    let want = reference(&p, &v);
    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(
        bad, 0,
        "{bad} of {} outputs differ; first got {:?} want {:?}",
        got.len(),
        got.iter().zip(&want).find(|(a, b)| a != b).map(|(a, _)| *a),
        got.iter().zip(&want).find(|(a, b)| a != b).map(|(_, b)| *b)
    );
    // Not vacuous: the answer must not be all zeros, which memset 0xAB would
    // also not be, but a kernel storing a constant would satisfy "equals the
    // reference" only if the reference were constant too.
    let distinct: std::collections::HashSet<i64> = got.iter().copied().collect();
    assert!(
        distinct.len() > B * Q * D / 2,
        "only {} distinct outputs of {} -- the kernel is not computing per-lane \
         results",
        distinct.len(),
        got.len()
    );
}

#[test]
fn an_fp32_accumulator_over_the_same_data_disagrees() {
    // The control for the control. If fp32 got the same answer, neither the
    // digit split nor this kernel would have a reason to exist, and test 1
    // would be checking nothing about EXACTNESS -- only about indexing.
    let (p, v) = inputs(0x5eed_1234);
    let want = reference(&p, &v);
    let mut differ = 0usize;
    let mut worst = 0i64;
    for b in 0..B {
        for q in 0..Q {
            for d in 0..D {
                let mut acc = 0f32;
                for t in 0..T {
                    acc += p[(b * Q + q) * T + t] as f32 * v[(b * T + t) * D + d] as f32;
                }
                let exact = want[(b * Q + q) * D + d];
                if acc as i64 != exact {
                    differ += 1;
                    worst = worst.max((acc as i64 - exact).abs());
                }
            }
        }
    }
    assert!(
        differ > B * Q * D / 4,
        "an fp32 accumulator agreed with the exact one on all but {differ} of \
         {} lanes -- this data does not exercise the precision the kernel is \
         for, so test 1 proves nothing about exactness",
        B * D
    );
    eprintln!(
        "fp32 accumulation differs on {differ} of {} lanes, worst by {worst}",
        B * Q * D
    );
}

#[test]
fn the_result_does_not_depend_on_summation_order() {
    // The property the whole deterministic path rests on. Reversing `t`
    // reverses the order the device sums in for a fixed lane; integer addition
    // is associative, so the answer must be bit-identical.
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let (p, v) = inputs(0xa5a5_0f0f);
    let forward = run(&ctx, &p, &v);

    let mut pr = vec![0u32; p.len()];
    let mut vr = vec![0i8; v.len()];
    for b in 0..B {
        for t in 0..T {
            for q in 0..Q {
                pr[(b * Q + q) * T + t] = p[(b * Q + q) * T + (T - 1 - t)];
            }
            for d in 0..D {
                vr[(b * T + t) * D + d] = v[(b * T + (T - 1 - t)) * D + d];
            }
        }
    }
    let reversed = run(&ctx, &pr, &vr);
    assert_eq!(
        forward, reversed,
        "reversing the summation order changed the result -- the accumulation \
         is not associative, so this kernel cannot carry the determinism claim"
    );
}
