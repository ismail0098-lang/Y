//! The int8 tensor-core GEMM's exactness licence, and the tie to
//! `proofs/Int8GemmExact.v`.
//!
//! ## The defect this file exists for, measured before the guard was written
//!
//! `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` accumulates into int32.
//! This kernel has **no flush** — the CPU's exact GEMM widens to int64 every
//! `Fl` k-pairs, and there is nowhere to widen to here because the OUTPUT is
//! int32 too. So the whole contraction must fit:
//!
//! ```text
//!   | sum over k < K of A[r][k] * B[c][k] |  <=  K * 127^2  <=  i32::MAX
//! ```
//!
//! `floor(i32::MAX / 127^2)` is 133_144, and `K % 32 == 0` is already the
//! kernel's shape precondition, so the largest admissible K is **133_120**.
//!
//! **Nothing checked it.** Not `emit_int8_gemm_kernel`, whose only refusal was
//! on M % 16 / N % 8 / K % 32; not `proofs/Int8GemmSchedule.v`, which proves
//! the schedule and says nothing about the accumulator's range; not any test.
//! Measured on the device, M=16 N=8, every element of A and B set to 127, one
//! warp, grid (1,1,1):
//!
//! | K       | exact          | device         |         |
//! |---------|----------------|----------------|---------|
//! | 133_088 | 2_146_576_352  | 2_146_576_352  | ok      |
//! | 133_120 | 2_147_092_480  | 2_147_092_480  | ok      |
//! | 133_152 | 2_147_608_608  | -2_147_358_688 | WRAPPED |
//! | 133_184 | 2_148_124_736  | -2_146_842_560 | WRAPPED |
//!
//! One K step wide, and the wrapped value is exactly two's complement —
//! `Int8GemmExact.the_measured_overflow_is_two_s_complement` reproduces that
//! third row from `wrap32` alone, so the model is refereed against the silicon
//! rather than asserted to describe it.
//!
//! It is the one GPU GEMM in this repository whose whole claim is an exact
//! answer, and past the bound it returned a NEGATIVE number under a green
//! banner with `red.global.add.s32` summing it.
//!
//! **Latent rather than live**: the largest K in the corpus is 16_384, four
//! orders below the bound. That is the reason to fix it now — this
//! repository's own rule is to find these while the path is still dead.
//!
//! ## What each test covers, and what it cannot
//!
//! The device test SKIPS without a CUDA driver, so it cannot be the only
//! cover. Three tests run everywhere: the boundary is one K step wide in the
//! compiler; the compiler's constant and the proof's agree; and an ordinary K
//! still compiles, which is what stops "refuse every int8 GEMM" from passing.
//!
//! The device test asserts the admitted MAXIMUM is exact rather than that the
//! rejected one wraps — the compiler can no longer emit the rejected one, and
//! that is the fix. Its refutation lives in Coq, pinned to the measurement
//! above.

use std::path::Path;
use std::process::Command;

/// The compiler's own bound. Parsed from the source rather than restated: a
/// second copy of the constant is the drift this tie exists to prevent.
fn emitter_bound() -> u32 {
    let src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ptx_emitter.rs"),
    )
    .expect("read ptx_emitter.rs");
    let line = src
        .lines()
        .find(|l| l.trim_start().starts_with("const INT8_MAX_EXACT_K"))
        .expect("no `const INT8_MAX_EXACT_K` in src/ptx_emitter.rs");
    line.split('=')
        .nth(1)
        .and_then(|v| v.trim().trim_end_matches(';').replace('_', "").parse().ok())
        .unwrap_or_else(|| panic!("could not parse a value from `{line}`"))
}

/// Compile a fixture at `(m, n, k)` in a per-test temp directory. The tag is in
/// the SIGNATURE rather than a comment asking the next author to remember:
/// this helper materialises files in a temp dir and that race has fired six
/// times in this repository.
fn emit(tag: &str, m: usize, n: usize, k: usize) -> Result<String, String> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let dir = std::env::temp_dir().join(format!("i8ex_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    // `--emit-ptx` writes next to its input, so compile a COPY: a gate that
    // emits must never rewrite the committed artifacts it is checking.
    if let Ok(p) = std::fs::read(repo.join(".ysu_hw_profile")) {
        let _ = std::fs::write(dir.join(".ysu_hw_profile"), p);
    }
    let src = dir.join("ex.ysu");
    std::fs::write(
        &src,
        format!(
            "@tile({m}, {n}, {k})\n\
             kernel int8_gemm(A: GlobalMemory<I8>, B: GlobalMemory<I8>, C: GlobalMemory<I32>) {{\n}}\n\
             fn main() {{}}\n"
        ),
    )
    .unwrap();
    let out = Command::new(bin.join("Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let r = if out.status.success() {
        Ok(std::fs::read_to_string(dir.join("ex.ptx")).expect("no .ptx"))
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    };
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// The boundary is ONE K STEP wide, in both directions.
///
/// A one-sided assertion is satisfied by a compiler that refuses everything,
/// and a bound that is merely "somewhere around 133k" would hide an off-by-one
/// that costs a whole K step of exactness — or, in the other direction, admits
/// a K whose product does not fit.
#[test]
fn the_licence_boundary_is_one_k_step_wide() {
    let bound = emitter_bound();
    assert_eq!(
        bound % 32,
        0,
        "the bound must be a multiple of 32: the kernel's own shape refusal \
         already requires K % 32 == 0, so a bound that is not one can never be \
         the largest admissible K."
    );
    // The arithmetic the bound comes from, re-derived rather than copied.
    assert!(
        (bound as u64) * 127 * 127 <= i32::MAX as u64,
        "K = {bound} does not fit: {} > i32::MAX",
        (bound as u64) * 127 * 127
    );
    assert!(
        ((bound + 32) as u64) * 127 * 127 > i32::MAX as u64,
        "K = {} would also fit, so the bound is not the largest one",
        bound + 32
    );

    emit("at", 16, 8, bound as usize).unwrap_or_else(|e| {
        panic!("the largest admissible K = {bound} must compile, got:\n{e}")
    });
    let over = emit("over", 16, 8, (bound + 32) as usize)
        .err()
        .unwrap_or_else(|| panic!("K = {} must be refused, but it compiled", bound + 32));
    assert!(
        over.contains("exceeds this kernel's exact range"),
        "K = {} is refused, but not for the licence reason. The message a user \
         gets must name the accumulator, not a shape:\n{over}",
        bound + 32
    );
    assert!(
        over.contains("127^2") && over.contains("i32::MAX"),
        "the refusal must state the derivation, so it can be acted on rather \
         than merely obeyed:\n{over}"
    );
}

/// The compiler's bound and the proof's are the same number.
///
/// `proofs/Int8GemmExact.v` states the exactness theorem under
/// `K * m^2 <= i32::MAX`; if the compiler admitted a larger K the theorem
/// would be about a kernel Y does not emit.
#[test]
fn the_emitter_and_the_proof_agree_on_the_bound() {
    let v = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("proofs/Int8GemmExact.v"),
    )
    .expect("read proofs/Int8GemmExact.v");

    assert!(
        v.contains("Definition MAX_EXACT_K : Z := MC.I32MAX / (127 * 127)."),
        "the proof must DERIVE the bound rather than state a numeral, or the \
         two sides can drift while both looking right"
    );
    assert!(
        v.contains("MAX_EXACT_K = 133144") && v.contains("MAX_EXACT_K_STEPS = 133120"),
        "`the_bound_is_one_k_step_wide` must pin both the raw quotient and the \
         step-granular bound; only the second is what the emitter uses"
    );
    assert_eq!(
        emitter_bound(),
        133_120,
        "the emitter's INT8_MAX_EXACT_K must be the proof's MAX_EXACT_K_STEPS"
    );

    // The refutation must state the DEVICE's number, not merely that it
    // overflows. A model that only said "it wraps" would agree with any wrong
    // answer; this one reproduces -2147358688 from `wrap32` alone.
    assert!(
        v.contains("MC.wrap32 2147608608 = -2147358688"),
        "`the_measured_overflow_is_two_s_complement` must reproduce the value \
         the device returned, or the proof is not refereed against the silicon"
    );
}

/// The control that stops "refuse every int8 GEMM" from passing every
/// assertion above.
#[test]
fn an_ordinary_shape_is_still_emitted() {
    let ptx = emit("ok", 64, 32, 128).expect("an ordinary int8 GEMM must still compile");
    assert!(
        ptx.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"),
        "the emitted kernel must still contain the instruction the licence is \
         about"
    );
}

/// The behavioural half: the largest ADMITTED K really is exact on the device.
///
/// This is what says the bound is tight rather than merely safe. The rejected
/// side cannot be tested here — the compiler no longer emits it, which is the
/// fix — so its refutation lives in `Int8GemmExact.v`, pinned to the
/// measurement in this file's header.
#[test]
fn the_largest_admitted_k_is_exact_on_the_device() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!(
            "SKIP: no CUDA driver. The three source-level tests in this file \
             still cover the licence and its tie to the proof."
        );
        return;
    };
    let k = emitter_bound() as usize;
    let (m, n) = (16usize, 8usize);
    let ptx = emit("dev", m, n, k).expect("the fixture must compile");
    let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");

    let d_a = ctx.alloc(m * k).unwrap();
    let d_b = ctx.alloc(n * k).unwrap();
    let d_c = ctx.alloc(m * n * 4).unwrap();
    // 127 is the WORST case, not a convenient one: the bound is a worst case
    // over the declared operand type, so anything smaller would pass with a
    // bound that is too large.
    ctx.memset_u8(&d_a, 127).unwrap();
    ctx.memset_u8(&d_b, 127).unwrap();
    ctx.memset_u8(&d_c, 0).unwrap();
    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
    ctx.launch(&module, (1, 1, 1), (32, 1, 1), 0, &args).unwrap();
    ctx.synchronize().expect("kernel faulted");

    let mut raw = vec![0u8; m * n * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
    let want = (k as i64) * 127 * 127;
    assert!(
        want <= i32::MAX as i64,
        "the fixture is not a test of exactness if it exceeds the bound"
    );
    for i in 0..m * n {
        let v = i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
        assert_eq!(
            v as i64, want,
            "C[{i}] = {v} at K = {k}, want {want}. At one K step further the \
             device returns -2147358688 (measured); the licence exists to make \
             that unreachable, and this asserts it does not refuse a K that is \
             still exact."
        );
    }
}
