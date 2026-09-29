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
    // The tag is for legibility when a run leaves a directory behind; the
    // COUNTER is what makes the path unique. A per-test tag in the signature
    // makes the requirement visible and does not enforce it - two tests in
    // this file both passed "over", which is the same temp-dir race this
    // repository has now hit seven times, caused here by reusing a string.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let uniq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "i8ex_{}_{}_{}",
        std::process::id(),
        tag,
        uniq
    ));
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

// ---------------------------------------------------------------------------
// The SPLIT-K accumulation, and the hypothesis writing its theorem forced out.
// ---------------------------------------------------------------------------

/// The positive case for `Int8GemmExact.the_split_k_accumulation_is_exact_in_int32`.
///
/// The capstone's int32 conjunct used to be stated for the flat accumulation
/// only. At `gridDim.z > 1` the kernel performs TWO wrapping folds — each CTA
/// accumulates its own residue class in an int32 register, and
/// `red.global.add.s32` then combines the partials in int32 in memory, in
/// whatever order they land. This runs both at the licensed maximum K, which
/// is the case where every partial of both folds is at its worst.
#[test]
fn the_split_k_accumulation_is_exact_at_the_licensed_maximum() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the split-K accumulation was not demonstrated.");
        return;
    };
    let k = emitter_bound() as usize;
    let (m, n) = (16usize, 8usize);
    let ptx = emit("splitk", m, n, k).expect("the fixture must compile");
    let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");

    let d_a = ctx.alloc(m * k).unwrap();
    let d_b = ctx.alloc(n * k).unwrap();
    let d_c = ctx.alloc(m * n * 4).unwrap();
    ctx.memset_u8(&d_a, 127).unwrap();
    ctx.memset_u8(&d_b, 127).unwrap();
    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
    let want = (k as i64) * 127 * 127;

    // 17 is deliberately not a divisor of the K step count: the theorem has no
    // divisibility precondition and a sweep of powers of two would not say so.
    for nz in [1u32, 2, 3, 8, 17, 64] {
        ctx.memset_u8(&d_c, 0).unwrap();
        ctx.launch(&module, (1, 1, nz), (32, 1, 1), 0, &args).unwrap();
        ctx.synchronize().expect("kernel faulted");
        let mut raw = vec![0u8; m * n * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        for i in 0..m * n {
            let v = i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
                as i64;
            assert_eq!(
                v, want,
                "C[{i}] = {v} at K = {k}, split {nz}, want {want}. Both wrapping folds are \
                 covered by one licence hypothesis because it bounds the sum of ABSOLUTE \
                 values and every partial of either fold is a sum over a subset."
            );
        }
    }
}

/// The refutation, and the reason it is a test rather than a remark: **the
/// licence is sufficient only for a ZEROED `C`, and nothing in the compiler
/// can check that.**
///
/// This kernel accumulates into `C` — which is what lets `gridDim.z` split the
/// contraction — so a caller who instead splits K across LAUNCHES into the
/// same int32 buffer is doing the obvious thing with that property. Every
/// launch is individually licensed; the accumulation is not.
///
/// Asserting the wrap makes the precondition load-bearing rather than
/// defensive, the same way `the_add_formula_really_is_incomplete` pins that
/// `add(P, P)` degenerates. The two exact launches before it are the control:
/// without them this would pass for a kernel that was simply broken.
#[test]
fn accumulating_across_launches_wraps_although_each_launch_is_licensed() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the zeroed-C precondition was not demonstrated.");
        return;
    };
    let k = (emitter_bound() / 2) as usize;
    assert_eq!(k % 32, 0, "half the bound must still be a legal K");
    let (m, n) = (16usize, 8usize);
    let per = (k as i64) * 127 * 127;
    assert!(
        per <= i32::MAX as i64,
        "each launch must be inside the licence, or this tests nothing"
    );

    // It compiles: the compiler accepts every one of these launches.
    let ptx = emit("half", m, n, k).expect("half the bound must compile");
    let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");

    let d_a = ctx.alloc(m * k).unwrap();
    let d_b = ctx.alloc(n * k).unwrap();
    let d_c = ctx.alloc(m * n * 4).unwrap();
    ctx.memset_u8(&d_a, 127).unwrap();
    ctx.memset_u8(&d_b, 127).unwrap();
    ctx.memset_u8(&d_c, 0).unwrap();
    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
    let read = |ctx: &CudaContext| -> i64 {
        let mut raw = vec![0u8; 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64
    };

    let mut seen = Vec::new();
    for _ in 0..3 {
        ctx.launch(&module, (1, 1, 1), (32, 1, 1), 0, &args).unwrap();
        ctx.synchronize().expect("kernel faulted");
        seen.push(read(&ctx));
    }
    assert_eq!(seen[0], per, "launch 1 into a zeroed C must be exact");
    assert_eq!(seen[1], 2 * per, "launch 2 is still inside int32 and must be exact");

    // Two's complement, computed here rather than pasted, so the expectation
    // moves with the bound instead of pinning one card's answer.
    let exact3 = 3 * per;
    let wrapped = ((exact3 - i32::MIN as i64).rem_euclid(1i64 << 32)) + i32::MIN as i64;
    assert!(
        exact3 > i32::MAX as i64,
        "three launches must exceed int32, or there is nothing to refute"
    );
    assert_eq!(
        seen[2], wrapped,
        "launch 3 gave {}, and the model says {wrapped} (exact {exact3}). Each launch was \
         licensed; the ACCUMULATION is not, because the bound is on C_initial + sum.",
        seen[2]
    );
}

/// The source-level half, which runs with no GPU: the compiler must warn about
/// the precondition, and the proof must state it as a hypothesis with its
/// refutation rather than assuming a zeroed destination silently.
#[test]
fn the_zeroed_destination_precondition_is_stated_in_both_places() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));

    // The message a user sees. It used to say only "accumulate the partials in
    // a wider type on the host", which is right and does not warn against the
    // reading that fails.
    let over = emit("overmsg", 16, 8, (emitter_bound() + 32) as usize)
        .expect_err("a K past the bound must be refused");
    for needle in ["accumulates into C", "C_initial + sum", "WIDER TYPE ON THE HOST"] {
        assert!(
            over.contains(needle),
            "the refusal no longer names the zeroed-C precondition (missing {needle:?}):\n{over}"
        );
    }

    let v = std::fs::read_to_string(repo.join("proofs/Int8GemmExact.v"))
        .expect("proofs/Int8GemmExact.v");

    // The combine is stated from zero, and the theorem that says that matters.
    assert!(
        v.contains("wcombine 0 (map (fun w => wclass f w n S) order) = GS.sum_upto Z.add f S"),
        "the split-K int32 theorem no longer combines from a zeroed destination"
    );
    assert!(
        v.contains("Theorem the_combine_needs_a_zeroed_destination"),
        "the refutation that makes the hypothesis load-bearing was deleted"
    );
    assert!(
        v.contains("Theorem from_zero_the_same_partial_is_exact"),
        "the control that stops the refutation reading as `the combine is broken` was deleted"
    );

    // Both folds must actually WRAP. Neither theorem's name nor `coqc` can see
    // this: with the wrap removed the definitions become ordinary integer
    // folds, every theorem above still holds, and the file still reports
    // "Closed under the global context" — a proof about int32 that says
    // nothing about int32. The guard belongs on the definition's text.
    assert!(
        v.contains("if Nat.eqb (k mod n) w then MC.wrap32 (wclass f w n k + f k)"),
        "a CTA's own accumulator no longer wraps, so the class fold is not int32"
    );
    assert!(
        v.contains("| x :: r => wcombine (MC.wrap32 (c0 + x)) r"),
        "the atomic combine no longer wraps, so the memory fold is not int32"
    );

    // The fixture magnitude is DERIVED from the emitter's bound in the proof
    // too, so a change to the bound cannot leave a stale numeral behind.
    assert!(
        v.contains("Definition LICENSED_HALF : Z := (MAX_EXACT_K_STEPS / 2) * (127 * 127)."),
        "the proof's launch magnitude is no longer derived from the licensed maximum"
    );
    assert_eq!(
        (emitter_bound() as i64 / 2) * 127 * 127,
        1_073_546_240,
        "the proof pins LICENSED_HALF = 1073546240; the emitter's bound no longer gives it"
    );
}
