// ============================================================
//  Y  —  Packed AVX-512 SGEMM codegen for the LLVM/CPU backend
//  cpu_gemm.rs
//
//  Recognises the canonical Y matmul loop nest and emits a
//  BLIS-shaped blocked kernel in its place: pack A and B into
//  cache-resident panels, then run a register-blocked
//  MR x NR micro-kernel over explicit <16 x float> vectors.
//
//  The blocking parameters below are measured, not assumed —
//  see `docs/cpu_gemm_tuning.md` for the sweep they came from.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use crate::zero_drift::OperandBounds;
use std::fmt::Write;

/// Register-blocking and cache-blocking parameters.
///
/// These were constants, and being constants is why `NRV` was never swept:
/// changing it meant editing and rebuilding the compiler, so every recorded
/// sweep moved `MR` alone and left the `MR x NRV` *shape* at 12x2 by default
/// rather than by measurement. They are overridable at emit time now —
/// `Y_GEMM_TILE=mr,nr,kc,mc,nc` — so a sweep is a shell loop.
///
/// The override behaves like `Y_CTA_OVERRIDE` on the PTX side and carries the
/// same trap: **it persists in whatever `.ll` was emitted while it was set.**
/// Regenerate `tests/y_cpu_matmul.ll` after sweeping or later measurements are
/// silently attributed to the default tile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tile {
    /// Rows of C held in registers by the micro-kernel.
    pub mr: usize,
    /// Columns of C held in registers, in floats. A multiple of 16.
    pub nr: usize,
    /// K-panel depth. `nr * kc * 4` should sit in L1d with the A micro-panel.
    pub kc: usize,
    /// M-block. `mc * kc * 4` should fit L2.
    pub mc: usize,
    /// N-block. `kc * nc * 4` should stream through L3.
    pub nc: usize,
}

impl Tile {
    /// Vectors per micro-kernel row: `nr / 16`.
    pub fn nrv(&self) -> usize {
        self.nr / 16
    }

    /// zmm registers the micro-kernel needs: accumulators, the B vectors, and
    /// two A broadcasts live at once.
    pub fn regs(&self) -> usize {
        self.mr * self.nrv() + self.nrv() + 2
    }

    /// Reject rather than emit a kernel that cannot work. A tile over the
    /// register budget compiles and runs — it just spills every accumulator to
    /// the stack in the innermost loop, which reads as a mysterious 3x slowdown
    /// rather than as a bad tile.
    pub fn check(&self) -> Result<(), String> {
        if self.nr % 16 != 0 || self.nr == 0 {
            return Err(format!("nr={} must be a non-zero multiple of 16", self.nr));
        }
        if self.mr == 0 || self.kc == 0 || self.mc == 0 || self.nc == 0 {
            return Err("mr, kc, mc and nc must all be non-zero".into());
        }
        if self.regs() > 32 {
            return Err(format!(
                "tile {}x{} needs {} zmm registers (max 32); accumulators would spill",
                self.mr,
                self.nr,
                self.regs()
            ));
        }
        if self.mr > MR_MAX || self.nr > NR_MAX || self.kc > KC_MAX
            || self.mc > MC_MAX || self.nc > NC_MAX
        {
            return Err(format!(
                "tile exceeds the scratch reserved by SCRATCH_FLOATS \
                 (max mr={} nr={} kc={} mc={} nc={})",
                MR_MAX, NR_MAX, KC_MAX, MC_MAX, NC_MAX
            ));
        }
        Ok(())
    }
}

/// Measured default: see `docs/cpu_gemm_tuning.md`.
pub const DEFAULT_TILE: Tile = Tile { mr: 6, nr: 64, kc: 256, mc: 192, nc: 2048 };

/// Floats of packed B the whole process may hold across all threads.
///
/// 32 MB against this part's 64 MB L3 — half, because the A panels, the C
/// block being accumulated and OpenBLAS-style streaming all want room too.
/// Divided by `nthr * nc` at runtime to pick the K-panel depth; see the long
/// comment at the `g.pc` loop in `emit_driver`.
pub const L3_PANEL_FLOATS: usize = 8 * 1024 * 1024;

/// Floor for the runtime K-panel depth. Below this the per-panel pack and the
/// C read-modify-write stop being amortised by anything.
pub const KC_MIN: usize = 64;

/// Upper bounds the static scratch is sized for. A tile past these is refused
/// rather than allowed to run off the end of the buffer.
pub const MR_MAX: usize = 16;
pub const NR_MAX: usize = 64;
pub const KC_MAX: usize = 1024;
pub const MC_MAX: usize = 384;
pub const NC_MAX: usize = 2048;

/// The tile in force for this compilation, resolved once from the environment.
pub fn tl() -> Tile {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Tile> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let raw = match std::env::var("Y_GEMM_TILE") {
            Ok(v) => v,
            Err(_) => return DEFAULT_TILE,
        };
        let f: Vec<usize> = raw
            .split(',')
            .map(|x| x.trim().parse().unwrap_or(0))
            .collect();
        if f.len() != 5 {
            eprintln!(
                "[Y CPU GEMM] Y_GEMM_TILE=\"{}\" is not mr,nr,kc,mc,nc — using the default tile",
                raw
            );
            return DEFAULT_TILE;
        }
        let t = Tile { mr: f[0], nr: f[1], kc: f[2], mc: f[3], nc: f[4] };
        if let Err(e) = t.check() {
            eprintln!("[Y CPU GEMM] Y_GEMM_TILE rejected: {} — using the default tile", e);
            return DEFAULT_TILE;
        }
        eprintln!(
            "[Y CPU GEMM] tile override: mr={} nr={} kc={} mc={} nc={} ({} zmm)",
            t.mr, t.nr, t.kc, t.mc, t.nc, t.regs()
        );
        t
    })
}

/// The GEMM shape a recognised loop nest computes: `C = A * B` with the
/// operand names, the three extents and the three leading dimensions, all as
/// Y identifiers.
///
/// `lda`/`ldb`/`ldc` are the row strides the nest actually indexes with. They
/// are usually the same identifiers as `k`/`n`/`n` — that is what a packed
/// row-major matmul writes, and what every shape measured in
/// `docs/cpu_gemm_tuning.md` uses — but they are recorded SEPARATELY so a
/// submatrix (`lda > K`) is a legal, fast input rather than a silent fallback
/// to scalar lowering. Without that distinction this kernel cannot implement
/// `sgemm` at all, since the BLAS API is defined in terms of leading
/// dimensions.
///
/// The recogniser deliberately does NOT check `lda >= K`: the strides are
/// runtime values, and that is the caller's precondition exactly as it is in
/// BLAS. What it does still enforce is the *indexing order* — A by `[i, k]`
/// and B by `[k, j]` — so a transposed operand is refused rather than
/// reinterpreted.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmShape {
    pub a: String,
    pub b: String,
    pub c: String,
    pub m: String,
    pub n: String,
    pub k: String,
    pub lda: String,
    pub ldb: String,
    pub ldc: String,
    /// Set when the accumulator carries `@ZeroDrift`, in which case the
    /// reduction must be EXACT and the f32 kernel is not a legal lowering.
    pub drift: Option<DriftAccumulator>,
    /// The declared type of the two operand `let`s, when both name the same
    /// one.
    ///
    /// **This decides whether the product truncates**, and a substitution that
    /// gets it wrong computes a different function. With `I16` buffers,
    /// `let a_val: I16 = ...` makes `a_val * b_val` an `i16` multiply that
    /// OVERFLOWS - 1024*1024 is 2^20 - and the naive nest then accumulates the
    /// truncated product. `vpdpwssd` widens internally, so substituting it for
    /// that nest replaces a truncating reduction with a widening one. Declaring
    /// the operands `I64` sign-extends at the load and the multiply is `i64`,
    /// which is what the kernel computes.
    pub operand_ty: Option<String>,
}

/// A `@ZeroDrift` accumulator's declared type and stated range, carried from
/// the recognised nest to the emitter so it can pick a representation.
///
/// The range is the load-bearing part. An exact fixed-point format buys
/// order-independence — integer addition is associative, so a tiled, threaded,
/// K-split reduction produces a bit-identical result to the naive loop — and it
/// pays for that with a bounded range. `@bounds(min, max)` is what makes the
/// choice satisfiable, so a missing range is not a detail: it is the
/// difference between a representation that can be selected and one that
/// cannot. See `docs/proof_carrying_kernels.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftAccumulator {
    /// Declared type of the accumulator, e.g. `F32`.
    pub ty: String,
    /// `@bounds(min, max)` resolved to constants, when the source gave one.
    pub bounds: Option<(f64, f64)>,
    /// `@bounds` on the statement loading the A operand, when the source gave
    /// one. See [`DriftAccumulator::operand_bounds`] for why the accumulator's
    /// own bound cannot stand in for this.
    pub a_bounds: Option<(f64, f64)>,
    /// `@bounds` on the statement loading the B operand.
    pub b_bounds: Option<(f64, f64)>,
}

impl DriftAccumulator {
    /// The magnitude bound on the operands, which is what an exact `vpdpwssd`
    /// kernel actually needs licensed — and which is **not** implied by
    /// [`Self::bounds`].
    ///
    /// `@bounds` on the accumulator constrains `C[i, j]`, the *sum*. The
    /// overflow obligation is about `A[i, k]` and `B[k, j]`, the *terms*, and a
    /// bound on a sum implies nothing about its terms because they cancel: a
    /// result bounded by 1.0 is perfectly consistent with operands of 1e9. So
    /// this returns `None` unless BOTH operands were bounded at their load, and
    /// the caller must refuse rather than substitute a guess.
    ///
    /// The two bounds are combined by taking the larger magnitude, because the
    /// overflow derivation in [`crate::zero_drift::VnniExact`] assumes a single
    /// bound covering both operands (`m^2` per product). Using the larger is
    /// the conservative direction; using the smaller, or their product, would
    /// licence a nest that can overflow.
    pub fn operand_bounds(&self) -> Option<OperandBounds> {
        let mag = |(lo, hi): (f64, f64)| lo.abs().max(hi.abs());
        match (self.a_bounds, self.b_bounds) {
            (Some(a), Some(b)) => Some(OperandBounds {
                max_magnitude: mag(a).max(mag(b)),
            }),
            _ => None,
        }
    }
}

/// Whether the exact `vpdpwssd` GEMM kernel may be emitted for a recognised
/// `@ZeroDrift` nest.
///
/// **`Unavailable` is not an error, and conflating the two would break working
/// programs.** A nest that cannot use the fast path is still compiled exactly —
/// scalar lowering honours `@ZeroDrift` by selecting an exact representation
/// (`llvm_emitter.rs`, `Stmt::Let` with `zero_drift`), it is simply slower. The
/// distinction the emitter has to preserve is:
///
/// - **no exact representation exists at all** → `emit_errors`, the build fails,
///   because the guarantee the source asked for cannot be delivered;
/// - **the exact representation exists but this one fast kernel is not licensed
///   for it** → `drift_report`, an advisory, because the guarantee *is*
///   delivered and only the speed is lost.
///
/// Reporting the second as an error would refuse programs that compile
/// correctly today.
#[derive(Debug, Clone, PartialEq)]
pub enum ExactGemmPlan {
    /// The exact `vpdpwssd` kernel is sound for this nest.
    Vnni {
        scheme: crate::zero_drift::VnniExact,
        /// The operand magnitude the licence was granted against.
        operand_magnitude: f64,
    },
    /// The fast path is unavailable, with the reason. The nest remains valid.
    Unavailable(String),
}

/// Decide whether the exact `vpdpwssd` kernel may be emitted for `drift`.
///
/// The whole decision is delegated to [`crate::zero_drift::license_vnni_exact`]
/// so that there is exactly ONE place that knows when the scheme is sound. A
/// second copy of the overflow arithmetic here is the failure the `optimize_circuit`
/// and constant-folding rows of `CLAUDE.md`'s design-rule table both describe:
/// two implementations of one rule, drifting apart, with the looser one
/// deciding what actually ships.
/// Whether the GEMM recogniser has been switched off for this process.
///
/// Set `Y_NO_GEMM_RECOGNISER=1` to lower a recognised nest as written. The
/// point is to be able to ask the compiler for both readings of one source -
/// the substituted kernel and the naive nest - so they can be compared against
/// each other rather than against a reference someone typed into a test.
///
/// Read from the environment on every call rather than cached: a test process
/// compiles the same source both ways, and a `OnceLock` would freeze whichever
/// arm ran first.
pub fn recogniser_disabled() -> bool {
    matches!(
        std::env::var("Y_NO_GEMM_RECOGNISER").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

pub fn plan_exact_gemm(drift: &DriftAccumulator) -> ExactGemmPlan {
    let operands = drift.operand_bounds();
    match crate::zero_drift::license_vnni_exact(
        operands,
        crate::zero_drift::VnniExact::DEFAULT_FLUSH_K_PAIRS,
    ) {
        Ok(scheme) => ExactGemmPlan::Vnni {
            scheme,
            // `license_vnni_exact` only returns `Ok` when `operands` was `Some`,
            // so this cannot be reached with `None` - but it is written as a
            // fallible match rather than an `unwrap` because that coupling is
            // an invariant of another function, not of this one.
            operand_magnitude: operands.map(|o| o.max_magnitude).unwrap_or(0.0),
        },
        Err(reason) => ExactGemmPlan::Unavailable(reason),
    }
}

// ── Exact VNNI micro-kernel ─────────────────────────────────

/// Name of the emitted exact micro-kernel.
pub const VNNI_MICRO_NAME: &str = "__y_gemm_micro_vnni";

/// Rows of C the micro-kernel holds in registers.
pub const VNNI_MR: usize = 6;

/// `<16 x i32>` accumulator groups per row. 4 groups = 64 columns, which is the
/// same 24-register footprint as the f32 kernel's `mr=6, nr=64` — the point of
/// `vpdpwssd` over the int64 formulation is that an int32 accumulator holds 16
/// lanes like a float, so the tile does NOT have to halve.
pub const VNNI_NRV: usize = 4;

/// Columns of C the micro-kernel covers.
pub const VNNI_NR: usize = VNNI_NRV * 16;

/// The LLVM intrinsic, with the signature **derived from clang**, not assumed.
///
/// Note the operand types: the accumulator is `<16 x i32>` but the multiplicands
/// are `<32 x i16>`. Writing `<16 x i32>` for the operands — the obvious guess,
/// since that is what `_mm512_dpwssd_epi32` takes in C — produces IR that fails
/// to verify. Established by compiling `tests/probes/vnni_kernels.c` with
/// `clang -mavx512vnni -emit-llvm` and reading the `declare` it generated.
const VPDPWSSD: &str =
    "declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <32 x i16>, <32 x i16>)";

/// Emit the flush: widen every int32 accumulator to int64, add it into `C`, and
/// zero it.
///
/// **This is what makes the kernel exact, and it is not an optimisation
/// detail.** `vpdpwssd` accumulates into int32, which overflows; the licence in
/// [`crate::zero_drift::VnniExact`] is precisely the promise that it cannot do
/// so *within one flush interval*. Widening into an int64 running sum is what
/// makes the interval bound sufficient rather than merely likely.
///
/// The int64 sum is where order-independence lives: int64 addition is
/// associative, so however the k-range is split across tiles or threads, the
/// partial sums combine to the same value.
fn emit_vnni_flush(out: &mut String, prefix: &str) {
    for i in 0..VNNI_MR {
        writeln!(out, "  %{p}row{i} = mul i64 %ldc, {i}", p = prefix, i = i).unwrap();
        for v in 0..VNNI_NRV {
            let acc = format!("%acc{}_{}", i, v);
            let t = format!("%{}f{}_{}", prefix, i, v);
            writeln!(out, "  {t}a = load <16 x i32>, ptr {acc}, align 64").unwrap();
            // Split the 16 int32 lanes into two halves and sign-extend each to
            // int64. `sext` is mandatory: these are signed products, and a
            // `zext` would turn every negative partial sum into a huge positive
            // one - a wrong answer that still runs.
            writeln!(
                out,
                "  {t}lo = shufflevector <16 x i32> {t}a, <16 x i32> poison, \
                 <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>"
            )
            .unwrap();
            writeln!(
                out,
                "  {t}hi = shufflevector <16 x i32> {t}a, <16 x i32> poison, \
                 <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>"
            )
            .unwrap();
            writeln!(out, "  {t}lo64 = sext <8 x i32> {t}lo to <8 x i64>").unwrap();
            writeln!(out, "  {t}hi64 = sext <8 x i32> {t}hi to <8 x i64>").unwrap();

            writeln!(
                out,
                "  {t}off = add i64 %{p}row{i}, {c}",
                p = prefix,
                i = i,
                c = v * 16
            )
            .unwrap();
            writeln!(out, "  {t}p = getelementptr inbounds i64, ptr %C, i64 {t}off").unwrap();
            writeln!(out, "  {t}c0 = load <8 x i64>, ptr {t}p, align 8").unwrap();
            writeln!(out, "  {t}s0 = add <8 x i64> {t}c0, {t}lo64").unwrap();
            writeln!(out, "  store <8 x i64> {t}s0, ptr {t}p, align 8").unwrap();
            writeln!(out, "  {t}p8 = getelementptr inbounds i64, ptr {t}p, i64 8").unwrap();
            writeln!(out, "  {t}c1 = load <8 x i64>, ptr {t}p8, align 8").unwrap();
            writeln!(out, "  {t}s1 = add <8 x i64> {t}c1, {t}hi64").unwrap();
            writeln!(out, "  store <8 x i64> {t}s1, ptr {t}p8, align 8").unwrap();
            writeln!(out, "  store <16 x i32> zeroinitializer, ptr {acc}, align 64").unwrap();
        }
    }
}

/// The exact `vpdpwssd` micro-kernel, as a standalone LLVM IR module.
///
/// Computes `C[0..MR][0..NR] += A^T B` over `kpairs` k-pairs, where every
/// operand is int16 and the result is int64. Layouts, which the packing must
/// match exactly:
///
/// - `Ap`: `i32`, indexed `Ap[p * MR + i]`. Each `i32` is **two int16 k-values**
///   for row `i`, because `vpdpwssd` consumes a k-pair per lane. This is why
///   the loop counts k-*pairs* rather than k.
/// - `Bp`: `i16`, indexed `Bp[p * NR * 2 + ..]`, four `<32 x i16>` vectors per
///   k-pair.
/// - `C`: `i64`, row stride `ldc`. Accumulated INTO, not overwritten, so a
///   caller may split the k-range across calls and sum the pieces — which is
///   exactly the order-independence being sold.
///
/// `kpairs` is not required to be a multiple of `flush_k_pairs`: the loop
/// flushes on the interval boundary and again at exit, so a partial final
/// interval is carried out correctly rather than dropped.
pub fn emit_vnni_micro_module(flush_k_pairs: u32) -> String {
    let mut out = String::new();
    writeln!(out, "; Exact vpdpwssd micro-kernel, MR={VNNI_MR} NR={VNNI_NR}, flush every {flush_k_pairs} k-pairs.").unwrap();
    writeln!(out, "; See docs/deterministic_inference.md M0.").unwrap();
    writeln!(out, "{VPDPWSSD}\n").unwrap();
    writeln!(
        out,
        "define void @{VNNI_MICRO_NAME}(ptr noalias %Ap, ptr noalias %Bp, ptr noalias %C, \
         i64 %ldc, i64 %kpairs) #0 {{"
    )
    .unwrap();
    writeln!(out, "entry:").unwrap();
    for i in 0..VNNI_MR {
        for v in 0..VNNI_NRV {
            writeln!(out, "  %acc{i}_{v} = alloca <16 x i32>, align 64").unwrap();
            writeln!(
                out,
                "  store <16 x i32> zeroinitializer, ptr %acc{i}_{v}, align 64"
            )
            .unwrap();
        }
    }
    writeln!(out, "  br label %ohead\n").unwrap();

    // The flush lives in the OUTER loop, and that placement is the whole
    // performance story. See `emit_vnni_micro_module`'s doc comment.
    writeln!(out, "ohead:").unwrap();
    writeln!(out, "  %c = phi i64 [ 0, %entry ], [ %cnext, %olatch ]").unwrap();
    writeln!(out, "  %ogo = icmp slt i64 %c, %kpairs").unwrap();
    writeln!(out, "  br i1 %ogo, label %obody, label %done\n").unwrap();

    writeln!(out, "obody:").unwrap();
    writeln!(out, "  %cend0 = add i64 %c, {flush_k_pairs}").unwrap();
    writeln!(out, "  %clt = icmp slt i64 %cend0, %kpairs").unwrap();
    writeln!(
        out,
        "  %cend = select i1 %clt, i64 %cend0, i64 %kpairs"
    )
    .unwrap();
    for i in 0..VNNI_MR {
        for v in 0..VNNI_NRV {
            writeln!(
                out,
                "  store <16 x i32> zeroinitializer, ptr %acc{i}_{v}, align 64"
            )
            .unwrap();
        }
    }
    writeln!(out, "  br label %head\n").unwrap();

    writeln!(out, "head:").unwrap();
    writeln!(out, "  %p = phi i64 [ %c, %obody ], [ %pnext, %body ]").unwrap();
    writeln!(out, "  %go = icmp slt i64 %p, %cend").unwrap();
    writeln!(out, "  br i1 %go, label %body, label %flush\n").unwrap();

    writeln!(out, "body:").unwrap();
    // B: four <32 x i16> vectors at Bp + p * NR * 2 int16 elements.
    writeln!(out, "  %bidx = mul i64 %p, {}", VNNI_NR * 2).unwrap();
    writeln!(out, "  %bp = getelementptr inbounds i16, ptr %Bp, i64 %bidx").unwrap();
    for v in 0..VNNI_NRV {
        writeln!(
            out,
            "  %bp{v} = getelementptr inbounds i16, ptr %bp, i64 {}",
            v * 32
        )
        .unwrap();
        writeln!(out, "  %b{v} = load <32 x i16>, ptr %bp{v}, align 2").unwrap();
    }
    // A: one i32 per row, broadcast across all 16 lanes then reinterpreted as
    // 32 int16 - the pattern clang generates for `_mm512_set1_epi32`.
    writeln!(out, "  %aidx = mul i64 %p, {VNNI_MR}").unwrap();
    for i in 0..VNNI_MR {
        writeln!(out, "  %ai{i} = add i64 %aidx, {i}").unwrap();
        writeln!(out, "  %ap{i} = getelementptr inbounds i32, ptr %Ap, i64 %ai{i}").unwrap();
        writeln!(out, "  %a{i} = load i32, ptr %ap{i}, align 4").unwrap();
        writeln!(
            out,
            "  %av{i} = insertelement <16 x i32> poison, i32 %a{i}, i64 0"
        )
        .unwrap();
        writeln!(
            out,
            "  %as{i} = shufflevector <16 x i32> %av{i}, <16 x i32> poison, \
             <16 x i32> zeroinitializer"
        )
        .unwrap();
        writeln!(out, "  %ab{i} = bitcast <16 x i32> %as{i} to <32 x i16>").unwrap();
        for v in 0..VNNI_NRV {
            writeln!(
                out,
                "  %old{i}_{v} = load <16 x i32>, ptr %acc{i}_{v}, align 64"
            )
            .unwrap();
            writeln!(
                out,
                "  %new{i}_{v} = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(\
                 <16 x i32> %old{i}_{v}, <32 x i16> %ab{i}, <32 x i16> %b{v})"
            )
            .unwrap();
            writeln!(
                out,
                "  store <16 x i32> %new{i}_{v}, ptr %acc{i}_{v}, align 64"
            )
            .unwrap();
        }
    }
    writeln!(out, "  %pnext = add i64 %p, 1").unwrap();
    writeln!(out, "  br label %head\n").unwrap();

    // One flush per chunk, after the inner loop has finished. `cend` is clamped
    // to `kpairs`, so the final partial chunk is flushed by this same code -
    // there is no separate tail case to forget.
    writeln!(out, "flush:").unwrap();
    emit_vnni_flush(&mut out, "F");
    writeln!(out, "  br label %olatch\n").unwrap();

    writeln!(out, "olatch:").unwrap();
    writeln!(out, "  %cnext = add i64 %c, {flush_k_pairs}").unwrap();
    writeln!(out, "  br label %ohead\n").unwrap();

    writeln!(out, "done:").unwrap();
    writeln!(out, "  ret void").unwrap();
    writeln!(out, "}}\n").unwrap();
    writeln!(
        out,
        "attributes #0 = {{ \"target-features\"=\"+avx512f,+avx512bw,+avx512vnni\" }}"
    )
    .unwrap();
    out
}

/// Name of the emitted exact blocked GEMM.
pub const VNNI_GEMM_NAME: &str = "__y_gemm_exact_vnni";

/// Pack an `MR x kc` panel of int16 `A` into the micro-kernel's `Ap` layout.
///
/// `Ap[(p * MR + i) * 2 + h] = A[i][2p + h]` — two consecutive k-values of one
/// row, adjacent, so the pair reads back as the single `i32` that `vpdpwssd`
/// broadcasts. Rows past `mrows` and k past `kc` are written as **zero**, which
/// is exact rather than approximate: a zero operand contributes a zero product,
/// so padding cannot perturb the sum. That is a property of exact accumulation
/// specifically — with a float accumulator, padding still perturbs nothing, but
/// nothing else about the reduction would be reproducible either.
fn emit_vnni_pack_a() -> String {
    let mut b = IrBuilder::new();
    let (src, lda, mrows, kc, dst) = ("%src", "%lda", "%mrows", "%kc", "%dst");

    // kpairs = ceil(kc / 2). An odd kc leaves the final high half zero.
    let kc1 = b.add(kc, "1");
    let kpairs = b.t();
    b.w(&format!("{} = sdiv i64 {}, 2", kpairs, kc1));

    let pl = b.loop_begin("va.p", "0", &kpairs, "1");
    let p = b.iv(&pl);
    let k0 = b.mul(&p, "2");
    let dbase = b.mul(&p, &(VNNI_MR * 2).to_string());

    for i in 0..VNNI_MR {
        let ic = i.to_string();
        let in_row = b.t();
        b.w(&format!("{} = icmp slt i64 {}, {}", in_row, ic, mrows));
        let row_off = b.mul(&ic, lda);
        for h in 0..2usize {
            let k = b.add(&k0, &h.to_string());
            let in_k = b.t();
            b.w(&format!("{} = icmp slt i64 {}, {}", in_k, k, kc));
            let ok = b.t();
            b.w(&format!("{} = and i1 {}, {}", ok, in_row, in_k));

            let off = b.add(&row_off, &k);
            let safe = b.t();
            b.w(&format!("{} = select i1 {}, i64 {}, i64 0", safe, ok, off));
            let sp = b.gep("i16", src, &safe);
            let raw = b.t();
            b.w(&format!("{} = load i16, ptr {}, align 2", raw, sp));
            let val = b.t();
            b.w(&format!(
                "{} = select i1 {}, i16 {}, i16 0",
                val, ok, raw
            ));

            let doff = b.add(&dbase, &(i * 2 + h).to_string());
            let dp = b.gep("i16", dst, &doff);
            b.w(&format!("store i16 {}, ptr {}, align 2", val, dp));
        }
    }
    b.loop_end(pl);
    b.finish(&format!(
        "define internal void @__y_gemm_vnni_pack_a(ptr noalias {}, i64 {}, i64 {}, i64 {}, \
         ptr noalias {})",
        src, lda, mrows, kc, dst
    ))
}

/// Pack a `kc x NR` panel of int16 `B` into the micro-kernel's `Bp` layout.
///
/// `Bp[p * NR * 2 + (j / 16) * 32 + (j % 16) * 2 + h] = B[2p + h][j]`.
///
/// The `(j / 16, j % 16)` split is not cosmetic: it is the lane layout
/// `vpdpwssd` imposes. Accumulator group `v = j / 16` is one `<32 x i16>`
/// vector, and within it lane `l = j % 16` consumes int16 elements `2l` and
/// `2l + 1` — so the two k-values for a column must be **adjacent**, and
/// consecutive columns are 2 int16 apart rather than 1. Getting this wrong
/// produces a kernel that runs at full speed and computes a different function;
/// `tests/cpu_gemm_vnni_micro.rs` mutates the stride to prove the tests see it.
fn emit_vnni_pack_b() -> String {
    let mut b = IrBuilder::new();
    let (src, ldb, kc, ncols, dst) = ("%src", "%ldb", "%kc", "%ncols", "%dst");

    let kc1 = b.add(kc, "1");
    let kpairs = b.t();
    b.w(&format!("{} = sdiv i64 {}, 2", kpairs, kc1));

    let pl = b.loop_begin("vb.p", "0", &kpairs, "1");
    let p = b.iv(&pl);
    let k0 = b.mul(&p, "2");
    let dbase = b.mul(&p, &(VNNI_NR * 2).to_string());

    let jl = b.loop_begin("vb.j", "0", &VNNI_NR.to_string(), "1");
    let j = b.iv(&jl);
    let in_col = b.t();
    b.w(&format!("{} = icmp slt i64 {}, {}", in_col, j, ncols));

    // v = j >> 4, l = j & 15  ->  lane offset v*32 + l*2
    let v = b.t();
    b.w(&format!("{} = lshr i64 {}, 4", v, j));
    let l = b.t();
    b.w(&format!("{} = and i64 {}, 15", l, j));
    let v32 = b.mul(&v, "32");
    let l2 = b.mul(&l, "2");
    let lane = b.add(&v32, &l2);
    let dlane = b.add(&dbase, &lane);

    for h in 0..2usize {
        let k = b.add(&k0, &h.to_string());
        let in_k = b.t();
        b.w(&format!("{} = icmp slt i64 {}, {}", in_k, k, kc));
        let ok = b.t();
        b.w(&format!("{} = and i1 {}, {}", ok, in_col, in_k));

        let row_off = b.mul(&k, ldb);
        let off = b.add(&row_off, &j);
        let safe = b.t();
        b.w(&format!("{} = select i1 {}, i64 {}, i64 0", safe, ok, off));
        let sp = b.gep("i16", src, &safe);
        let raw = b.t();
        b.w(&format!("{} = load i16, ptr {}, align 2", raw, sp));
        let val = b.t();
        b.w(&format!("{} = select i1 {}, i16 {}, i16 0", val, ok, raw));

        let doff = b.add(&dlane, &h.to_string());
        let dp = b.gep("i16", dst, &doff);
        b.w(&format!("store i16 {}, ptr {}, align 2", val, dp));
    }

    b.loop_end(jl);
    b.loop_end(pl);
    b.finish(&format!(
        "define internal void @__y_gemm_vnni_pack_b(ptr noalias {}, i64 {}, i64 {}, i64 {}, \
         ptr noalias {})",
        src, ldb, kc, ncols, dst
    ))
}

/// The blocked exact GEMM: `C += A * B`, int16 in, int64 out.
///
/// `C` is **accumulated into, not overwritten**, and that is the whole
/// interface. It is what lets a caller split the k-range across calls, tiles or
/// threads and sum the pieces — the partial sums are int64, whose addition is
/// associative, so every such split yields the identical result. Overwriting
/// would force one call to own the whole reduction and throw the property away.
///
/// Scratch is passed in rather than allocated:
///
/// - `Apanel`: `ceil(M/MR) * ceil(K/2) * MR * 2` int16 — **the whole of A**,
///   packed once before either loop.
/// - `Bpanel`: `ceil(K/2) * NR * 2` int16 — one column panel, repacked per `j`.
/// - `Ctile`: `MR * NR` int64.
///
/// Making the caller own them keeps this function free of an allocation policy
/// it is not yet ready to have — the `mc`/`nc` blocking that would cap
/// `Apanel` at a cache-resident size is a later step, not this one.
fn emit_vnni_gemm_driver() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c) = ("%A", "%B", "%C");
    let (m, n, k) = ("%M", "%N", "%K");
    let (lda, ldb, ldc) = ("%lda", "%ldb", "%ldc");
    let (ap, bp, ct) = ("%Apanel", "%Bpanel", "%Ctile");

    let kc1 = b.add(k, "1");
    let kpairs = b.t();
    b.w(&format!("{} = sdiv i64 {}, 2", kpairs, kc1));

    // Pack ALL of A once, before either loop.
    //
    // The first version packed A inside the i-loop and B inside the j-loop
    // nested within it, so B was re-packed once per ROW panel — `M/MR` times
    // over, which is `M*N*K/MR` packing work against the `M*N*K` of arithmetic.
    // At MR=6 that is a sixth of the total run spent copying B. Packing A once
    // up front and B once per column panel makes packing `M*K + N*K`, i.e.
    // asymptotically free.
    let panel_stride = b.mul(&kpairs, &(VNNI_MR * 2).to_string());
    let pal = b.loop_begin("vg.pa", "0", m, &VNNI_MR.to_string());
    let pai = b.iv(&pal);
    let pa_rem = b.sub(m, &pai);
    let pa_mw = b.imin(&pa_rem, &VNNI_MR.to_string());
    let pa_row = b.mul(&pai, lda);
    let pa_src = b.gep("i16", a, &pa_row);
    let pa_idx = b.t();
    b.w(&format!(
        "{} = sdiv i64 {}, {}",
        pa_idx, pai, VNNI_MR
    ));
    let pa_off = b.mul(&pa_idx, &panel_stride);
    let pa_dst = b.gep("i16", ap, &pa_off);
    b.w(&format!(
        "call void @__y_gemm_vnni_pack_a(ptr {}, i64 {}, i64 {}, i64 {}, ptr {})",
        pa_src, lda, pa_mw, k, pa_dst
    ));
    b.loop_end(pal);

    let jl = b.loop_begin("vg.j", "0", n, &VNNI_NR.to_string());
    let j0 = b.iv(&jl);
    let rem_n = b.sub(n, &j0);
    let nw = b.imin(&rem_n, &VNNI_NR.to_string());
    let bsub = b.gep("i16", bb, &j0);
    b.w(&format!(
        "call void @__y_gemm_vnni_pack_b(ptr {}, i64 {}, i64 {}, i64 {}, ptr {})",
        bsub, ldb, k, nw, bp
    ));

    let il = b.loop_begin("vg.i", "0", m, &VNNI_MR.to_string());
    let i0 = b.iv(&il);
    let rem_m = b.sub(m, &i0);
    let mw = b.imin(&rem_m, &VNNI_MR.to_string());
    let ai_idx = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", ai_idx, i0, VNNI_MR));
    let ai_off = b.mul(&ai_idx, &panel_stride);
    let apan = b.gep("i16", ap, &ai_off);

    // The micro-kernel always writes a full MR x NR tile, so it accumulates
    // into scratch and only the live part is folded back. Letting it write C
    // directly would run past the last row and column whenever M or N is not a
    // multiple of the tile - an out-of-bounds WRITE, not a wrong number.
    let zl = b.loop_begin("vg.z", "0", &(VNNI_MR * VNNI_NR).to_string(), "1");
    let z = b.iv(&zl);
    let zp = b.gep("i64", ct, &z);
    b.w(&format!("store i64 0, ptr {}, align 8", zp));
    b.loop_end(zl);

    b.w(&format!(
        "call void @{}(ptr {}, ptr {}, ptr {}, i64 {}, i64 {})",
        VNNI_MICRO_NAME, apan, bp, ct, VNNI_NR, kpairs
    ));

    let fl = b.loop_begin("vg.fi", "0", &mw, "1");
    let fi = b.iv(&fl);
    let crow = b.add(&i0, &fi);
    let coff = b.mul(&crow, ldc);
    let coff2 = b.add(&coff, &j0);
    let trow = b.mul(&fi, &VNNI_NR.to_string());

    let fjl = b.loop_begin("vg.fj", "0", &nw, "1");
    let fj = b.iv(&fjl);
    let cidx = b.add(&coff2, &fj);
    let cp = b.gep("i64", c, &cidx);
    let tidx = b.add(&trow, &fj);
    let tp = b.gep("i64", ct, &tidx);
    let cv = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", cv, cp));
    let tv = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", tv, tp));
    let sv = b.add(&cv, &tv);
    b.w(&format!("store i64 {}, ptr {}, align 8", sv, cp));
    b.loop_end(fjl);
    b.loop_end(fl);

    b.loop_end(il);
    b.loop_end(jl);

    b.finish(&format!(
        "define void @{}(ptr noalias {}, ptr noalias {}, ptr noalias {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, \
         ptr noalias {}, ptr noalias {}, ptr noalias {}) #0",
        VNNI_GEMM_NAME, a, bb, c, m, n, k, lda, ldb, ldc, ap, bp, ct
    ))
}

/// Name of the threaded entry point that splits K across workers.
pub const VNNI_THREADED_NAME: &str = "__y_gemm_exact_vnni_threaded";

/// The K-split threaded wrapper around [`VNNI_GEMM_NAME`].
///
/// This is what makes the exact path's central claim observable rather than
/// merely argued: integer addition is associative, so partitioning K across
/// threads and summing the partial sums gives a BIT-IDENTICAL result whatever
/// the thread count and however ragged the cuts are. An f32 kernel cannot say
/// that, which is the whole reason `docs/proof_carrying_kernels.md` is built on
/// exact accumulation.
///
/// Deliberately plain `pthread_create` / `pthread_join` rather than the f32
/// path's persistent pool. The pool is tuned around a task struct shaped for
/// that kernel and around dispatch costs this kernel has not been measured
/// against; forking per call costs tens of microseconds, which is noise at any
/// size where threading pays at all. Correctness first - the property under
/// test here is bit-identity, not peak throughput.
///
/// `need_libc_decls` is false when the f32 module is also being emitted, since
/// it declares the same libc entry points and **a duplicate `declare` is an
/// invalid redefinition in LLVM, not a duplicate that gets merged**.
pub fn emit_vnni_threaded_module(need_libc_decls: bool) -> String {
    let mut s = String::new();
    if need_libc_decls {
        let _ = writeln!(&mut s, "declare ptr @getenv(ptr)");
        let _ = writeln!(&mut s, "declare i32 @atoi(ptr)");
        let _ = writeln!(&mut s, "declare i64 @sysconf(i32)");
        let _ = writeln!(&mut s, "declare i32 @pthread_create(ptr, ptr, ptr, ptr)");
        let _ = writeln!(&mut s, "declare i32 @pthread_join(i64, ptr)");
    }
    let _ = writeln!(
        &mut s,
        "@.y_exact_env = private unnamed_addr constant [15 x i8] c\"Y_NUM_THREADS\\00\\00\""
    );
    let _ = writeln!(
        &mut s,
        "@__y_gemm_exact_nthreads = internal global i64 0, align 8"
    );

    // Job slots, 8 bytes each: A B C M N K lda ldb ldc Ap Bp Ct.
    let job_bytes = 96usize;
    let mr = VNNI_MR;
    let nr = VNNI_NR;

    let _ = write!(
        &mut s,
        r#"
define internal i64 @__y_gemm_exact_threads(i64 %K) {{
entry:
  %c = load i64, ptr @__y_gemm_exact_nthreads, align 8
  %need = icmp eq i64 %c, 0
  br i1 %need, label %resolve, label %have
resolve:
  %e = call ptr @getenv(ptr @.y_exact_env)
  %has = icmp ne ptr %e, null
  br i1 %has, label %fromenv, label %fromsys
fromenv:
  %ei = call i32 @atoi(ptr %e)
  %e64 = sext i32 %ei to i64
  br label %clamp
fromsys:
  %sc = call i64 @sysconf(i32 84)
  br label %clamp
clamp:
  %raw = phi i64 [ %e64, %fromenv ], [ %sc, %fromsys ]
  %lo = icmp slt i64 %raw, 1
  %r1 = select i1 %lo, i64 1, i64 %raw
  %hi = icmp sgt i64 %r1, {maxthr}
  %r2 = select i1 %hi, i64 {maxthr}, i64 %r1
  store i64 %r2, ptr @__y_gemm_exact_nthreads, align 8
  br label %have
have:
  %ceil = phi i64 [ %c, %entry ], [ %r2, %clamp ]
  ; Never hand out a K-band shorter than {minband}: the per-thread zero-fill and
  ; the reduction are independent of K, so a sliver costs more than it saves.
  %byk = sdiv i64 %K, {minband}
  %small = icmp slt i64 %byk, %ceil
  %n = select i1 %small, i64 %byk, i64 %ceil
  %z = icmp slt i64 %n, 1
  %out = select i1 %z, i64 1, i64 %n
  ret i64 %out
}}

define internal ptr @__y_gemm_exact_worker(ptr %arg) {{
entry:
  %pa = getelementptr i8, ptr %arg, i64 0
  %A = load ptr, ptr %pa, align 8
  %pb = getelementptr i8, ptr %arg, i64 8
  %B = load ptr, ptr %pb, align 8
  %pc = getelementptr i8, ptr %arg, i64 16
  %C = load ptr, ptr %pc, align 8
  %pm = getelementptr i8, ptr %arg, i64 24
  %M = load i64, ptr %pm, align 8
  %pn = getelementptr i8, ptr %arg, i64 32
  %N = load i64, ptr %pn, align 8
  %pk = getelementptr i8, ptr %arg, i64 40
  %K = load i64, ptr %pk, align 8
  %pla = getelementptr i8, ptr %arg, i64 48
  %lda = load i64, ptr %pla, align 8
  %plb = getelementptr i8, ptr %arg, i64 56
  %ldb = load i64, ptr %plb, align 8
  %plc = getelementptr i8, ptr %arg, i64 64
  %ldc = load i64, ptr %plc, align 8
  %pap = getelementptr i8, ptr %arg, i64 72
  %Ap = load ptr, ptr %pap, align 8
  %pbp = getelementptr i8, ptr %arg, i64 80
  %Bp = load ptr, ptr %pbp, align 8
  %pct = getelementptr i8, ptr %arg, i64 88
  %Ct = load ptr, ptr %pct, align 8
  call void @{gemm}(ptr %A, ptr %B, ptr %C, i64 %M, i64 %N, i64 %K, i64 %lda, i64 %ldb, i64 %ldc, ptr %Ap, ptr %Bp, ptr %Ct)
  ret ptr null
}}

define void @{threaded}(ptr %A, ptr %B, ptr %C, i64 %M, i64 %N, i64 %K, i64 %lda, i64 %ldb, i64 %ldc) {{
entry:
  ; `C` is ASSIGNED by the nest this replaces, while the kernel accumulates
  ; INTO it - which is exactly what lets the K-bands be summed.
  %cn = mul i64 %M, %N
  %cb = mul i64 %cn, 8
  %rowb = mul i64 %N, 8
  br label %zero.head

; Zero the LIVE M x N rectangle ONLY, a row at a time at the caller's stride.
;
; This was one flat `memset(C, 0, M*N*8)`, which is right if and only if
; `ldc == N` - and every caller in the compiler passes exactly that, so nothing
; ever exercised it. With a padded C it zeroed into the row padding of the early
; rows and left the last rows' live cells UNZEROED, and since this kernel
; ACCUMULATES into C that is a wrong answer, not a cosmetic one.
zero.head:
  %zi = phi i64 [ 0, %entry ], [ %zinext, %zero.body ]
  %zmore = icmp slt i64 %zi, %M
  br i1 %zmore, label %zero.body, label %zero.done

zero.body:
  %zrow = mul i64 %zi, %ldc
  %zp = getelementptr i64, ptr %C, i64 %zrow
  call void @llvm.memset.p0.i64(ptr %zp, i8 0, i64 %rowb, i1 false)
  %zinext = add i64 %zi, 1
  br label %zero.head

zero.done:
  %nthr = call i64 @__y_gemm_exact_threads(i64 %K)
  %mtiles0 = add i64 %M, {mr_m1}
  %mtiles = sdiv i64 %mtiles0, {mr}
  %one = icmp sle i64 %nthr, 1
  br i1 %one, label %single, label %many

single:
  %kp1 = add i64 %K, 1
  %kps = sdiv i64 %kp1, 2
  %apn = mul i64 %mtiles, %kps
  %apn2 = mul i64 %apn, {mr2}
  %apb = mul i64 %apn2, 2
  %bpn = mul i64 %kps, {nr2}
  %bpb = mul i64 %bpn, 2
  %ap1 = call ptr @malloc(i64 %apb)
  %bp1 = call ptr @malloc(i64 %bpb)
  %ct1 = call ptr @malloc(i64 {ctb})
  call void @llvm.memset.p0.i64(ptr %ap1, i8 0, i64 %apb, i1 false)
  call void @llvm.memset.p0.i64(ptr %bp1, i8 0, i64 %bpb, i1 false)
  call void @llvm.memset.p0.i64(ptr %ct1, i8 0, i64 {ctb}, i1 false)
  call void @{gemm}(ptr %A, ptr %B, ptr %C, i64 %M, i64 %N, i64 %K, i64 %lda, i64 %ldb, i64 %ldc, ptr %ap1, ptr %bp1, ptr %ct1)
  call void @free(ptr %ap1)
  call void @free(ptr %bp1)
  call void @free(ptr %ct1)
  ret void

many:
  %jobsb = mul i64 %nthr, {jobb}
  %jobs = call ptr @malloc(i64 %jobsb)
  %tidsb = mul i64 %nthr, 8
  %tids = call ptr @malloc(i64 %tidsb)
  %base = sdiv i64 %K, %nthr
  %rem = srem i64 %K, %nthr
  br label %spawn.head

spawn.head:
  %t = phi i64 [ 0, %many ], [ %tnext, %spawn.next ]
  %off = phi i64 [ 0, %many ], [ %offnext, %spawn.next ]
  %more = icmp slt i64 %t, %nthr
  br i1 %more, label %spawn.body, label %joinloop.head

spawn.body:
  ; The first `rem` bands get one extra k, so the cuts are uneven by
  ; construction and never line up with the flush interval.
  %extra = icmp slt i64 %t, %rem
  %inc = select i1 %extra, i64 1, i64 0
  %klen = add i64 %base, %inc

  %kp1b = add i64 %klen, 1
  %kpsb = sdiv i64 %kp1b, 2
  %apnb = mul i64 %mtiles, %kpsb
  %apn2b = mul i64 %apnb, {mr2}
  %apbb = mul i64 %apn2b, 2
  %bpnb = mul i64 %kpsb, {nr2}
  %bpbb = mul i64 %bpnb, 2
  %apt = call ptr @malloc(i64 %apbb)
  %bpt = call ptr @malloc(i64 %bpbb)
  %ctt = call ptr @malloc(i64 {ctb})
  %cpt = call ptr @malloc(i64 %cb)
  call void @llvm.memset.p0.i64(ptr %apt, i8 0, i64 %apbb, i1 false)
  call void @llvm.memset.p0.i64(ptr %bpt, i8 0, i64 %bpbb, i1 false)
  call void @llvm.memset.p0.i64(ptr %ctt, i8 0, i64 {ctb}, i1 false)
  call void @llvm.memset.p0.i64(ptr %cpt, i8 0, i64 %cb, i1 false)

  ; A is [M, K] with row stride lda, so a k-band starts at A + off and keeps
  ; lda. B is [K, N] with row stride ldb, so its band starts at B + off*ldb.
  %aoff = getelementptr i16, ptr %A, i64 %off
  %boffi = mul i64 %off, %ldb
  %boff = getelementptr i16, ptr %B, i64 %boffi

  %jb = mul i64 %t, {jobb}
  %j = getelementptr i8, ptr %jobs, i64 %jb
  %s0 = getelementptr i8, ptr %j, i64 0
  store ptr %aoff, ptr %s0, align 8
  %s1 = getelementptr i8, ptr %j, i64 8
  store ptr %boff, ptr %s1, align 8
  %s2 = getelementptr i8, ptr %j, i64 16
  store ptr %cpt, ptr %s2, align 8
  %s3 = getelementptr i8, ptr %j, i64 24
  store i64 %M, ptr %s3, align 8
  %s4 = getelementptr i8, ptr %j, i64 32
  store i64 %N, ptr %s4, align 8
  %s5 = getelementptr i8, ptr %j, i64 40
  store i64 %klen, ptr %s5, align 8
  %s6 = getelementptr i8, ptr %j, i64 48
  store i64 %lda, ptr %s6, align 8
  %s7 = getelementptr i8, ptr %j, i64 56
  store i64 %ldb, ptr %s7, align 8
  %s8 = getelementptr i8, ptr %j, i64 64
  ; The worker's C is `%cpt`, a private M*N buffer - COMPACT, whatever the
  ; caller's stride is. Passing the caller's `%ldc` here made the worker write
  ; at that stride into a buffer sized for N, i.e. `(M-1)*(ldc-N)` elements past
  ; the end: a heap overflow, reported as `double free or corruption`.
  store i64 %N, ptr %s8, align 8
  %s9 = getelementptr i8, ptr %j, i64 72
  store ptr %apt, ptr %s9, align 8
  %s10 = getelementptr i8, ptr %j, i64 80
  store ptr %bpt, ptr %s10, align 8
  %s11 = getelementptr i8, ptr %j, i64 88
  store ptr %ctt, ptr %s11, align 8

  %tb = mul i64 %t, 8
  %tp = getelementptr i8, ptr %tids, i64 %tb
  %rc = call i32 @pthread_create(ptr %tp, ptr null, ptr @__y_gemm_exact_worker, ptr %j)
  ; A thread that fails to start must still be accounted for, or the join
  ; below waits on a tid that was never written. Run its band inline instead.
  %failed = icmp ne i32 %rc, 0
  br i1 %failed, label %spawn.inline, label %spawn.next

spawn.inline:
  call ptr @__y_gemm_exact_worker(ptr %j)
  store i64 0, ptr %tp, align 8
  br label %spawn.next

spawn.next:
  %tnext = add i64 %t, 1
  %offnext = add i64 %off, %klen
  br label %spawn.head

joinloop.head:
  %jt = phi i64 [ 0, %spawn.head ], [ %jtnext, %joinloop.skip ]
  %jmore = icmp slt i64 %jt, %nthr
  br i1 %jmore, label %joinloop.body, label %reduce.head

joinloop.body:
  %jtb = mul i64 %jt, 8
  %jtp = getelementptr i8, ptr %tids, i64 %jtb
  %tid = load i64, ptr %jtp, align 8
  %live = icmp ne i64 %tid, 0
  br i1 %live, label %joinloop.do, label %joinloop.skip

joinloop.do:
  %jrc = call i32 @pthread_join(i64 %tid, ptr null)
  br label %joinloop.skip

joinloop.skip:
  %jtnext = add i64 %jt, 1
  br label %joinloop.head

reduce.head:
  %rt = phi i64 [ 0, %joinloop.head ], [ %rtnext, %reduce.done ]
  %rmore = icmp slt i64 %rt, %nthr
  br i1 %rmore, label %reduce.body, label %cleanup

reduce.body:
  %rjb = mul i64 %rt, {jobb}
  %rj = getelementptr i8, ptr %jobs, i64 %rjb
  %rs2 = getelementptr i8, ptr %rj, i64 16
  %rcp = load ptr, ptr %rs2, align 8
  br label %reduce.row

; The destination and the source have DIFFERENT row strides - `%ldc` for the
; caller's C, `%N` for the worker's compact private one. The flat loop this
; replaces walked both with one index, which is only correct when they are
; equal.
reduce.row:
  %ri = phi i64 [ 0, %reduce.body ], [ %rinext, %reduce.rowend ]
  %rimore = icmp slt i64 %ri, %M
  br i1 %rimore, label %reduce.rowbody, label %reduce.free

reduce.rowbody:
  %drow = mul i64 %ri, %ldc
  %srow = mul i64 %ri, %N
  br label %reduce.inner

reduce.inner:
  %q = phi i64 [ 0, %reduce.rowbody ], [ %qnext, %reduce.inner.body ]
  %qmore = icmp slt i64 %q, %N
  br i1 %qmore, label %reduce.inner.body, label %reduce.rowend

reduce.inner.body:
  %didx = add i64 %drow, %q
  %sidx = add i64 %srow, %q
  %dstp = getelementptr i64, ptr %C, i64 %didx
  %srcp = getelementptr i64, ptr %rcp, i64 %sidx
  %dv = load i64, ptr %dstp, align 8
  %sv = load i64, ptr %srcp, align 8
  ; Integer addition, so the order these partials are summed in cannot change
  ; the total. That is the property the whole exact path exists to provide.
  %sum = add i64 %dv, %sv
  store i64 %sum, ptr %dstp, align 8
  %qnext = add i64 %q, 1
  br label %reduce.inner

reduce.rowend:
  %rinext = add i64 %ri, 1
  br label %reduce.row

reduce.free:
  %fs9 = getelementptr i8, ptr %rj, i64 72
  %fap = load ptr, ptr %fs9, align 8
  call void @free(ptr %fap)
  %fs10 = getelementptr i8, ptr %rj, i64 80
  %fbp = load ptr, ptr %fs10, align 8
  call void @free(ptr %fbp)
  %fs11 = getelementptr i8, ptr %rj, i64 88
  %fct = load ptr, ptr %fs11, align 8
  call void @free(ptr %fct)
  call void @free(ptr %rcp)
  br label %reduce.done

reduce.done:
  %rtnext = add i64 %rt, 1
  br label %reduce.head

cleanup:
  call void @free(ptr %jobs)
  call void @free(ptr %tids)
  ret void
}}
"#,
        maxthr = 64,
        minband = KSPLIT_MIN_BAND,
        mr = mr,
        mr_m1 = mr - 1,
        mr2 = mr * 2,
        nr2 = nr * 2,
        ctb = mr * nr * 8,
        jobb = job_bytes,
        gemm = VNNI_GEMM_NAME,
        threaded = VNNI_THREADED_NAME,
    );
    s
}

/// The complete exact GEMM module: micro-kernel, both packers, and the driver.
pub fn emit_vnni_gemm_module(flush_k_pairs: u32) -> String {
    let mut out = emit_vnni_micro_module(flush_k_pairs);
    out.push('\n');
    out.push_str(&emit_vnni_pack_a());
    out.push('\n');
    out.push_str(&emit_vnni_pack_b());
    out.push('\n');
    out.push_str(&emit_vnni_gemm_driver());
    out
}

// ── Recognition ─────────────────────────────────────────────

fn ident_of(e: &Expr) -> Option<&str> {
    match e {
        Expr::Ident(n, _) => Some(n.as_str()),
        _ => None,
    }
}

fn is_zero_lit(e: &Expr) -> bool {
    matches!(e, Expr::IntLit(0, _)) || matches!(e, Expr::FloatLit(v, _) if *v == 0.0)
}

/// A `step` clause that is absent or literally 1.
fn is_unit_step(step: &Option<Expr>) -> bool {
    match step {
        None => true,
        Some(Expr::IntLit(1, _)) => true,
        _ => false,
    }
}

fn call_parts<'a>(e: &'a Expr, name: &str) -> Option<&'a [Expr]> {
    let Expr::Call { func, args, .. } = e else {
        return None;
    };
    if ident_of(func) == Some(name) {
        Some(args.as_slice())
    } else {
        None
    }
}

/// Matches `block_ptr2d_load(buf, row, col, stride, _, _)` and returns
/// `(buf, row, col, stride)`.
fn match_load(e: &Expr) -> Option<(&str, &str, &str, &str)> {
    let args = call_parts(e, "block_ptr2d_load")?;
    if args.len() != 6 {
        return None;
    }
    Some((
        ident_of(&args[0])?,
        ident_of(&args[1])?,
        ident_of(&args[2])?,
        ident_of(&args[3])?,
    ))
}

/// Recognises exactly this nest, and refuses anything else:
///
/// ```text
/// for i in 0..M { for j in 0..N {
///     let mut sum = 0.0;
///     for k in 0..K { sum = sum + load(A,i,k,K,..) * load(B,k,j,N,..); }
///     store(C, i, j, N, .., sum);
/// } }
/// ```
///
/// The match is deliberately strict. Anything that does not line up falls
/// through to the ordinary scalar lowering, which is correct — so a near-miss
/// costs performance, never an answer. A looser match that "mostly" fit would
/// substitute a different computation for the one that was written.
pub fn recognize_gemm(body: &Block) -> Option<GemmShape> {
    let [Stmt::For {
        loop_var: i,
        start: i_start,
        end: i_end,
        step: i_step,
        body: i_body,
        ..
    }] = body.stmts.as_slice()
    else {
        return None;
    };
    if !is_zero_lit(i_start) || !is_unit_step(i_step) {
        return None;
    }

    let [Stmt::For {
        loop_var: j,
        start: j_start,
        end: j_end,
        step: j_step,
        body: j_body,
        ..
    }] = i_body.stmts.as_slice()
    else {
        return None;
    };
    if !is_zero_lit(j_start) || !is_unit_step(j_step) {
        return None;
    }

    // let sum = 0.0;  /  for k ...  /  store(...)
    let [Stmt::Let {
        name: sum,
        init: Some(sum_init),
        zero_drift,
        ty: sum_ty,
        bounds: sum_bounds,
        ..
    }, Stmt::For {
        loop_var: k,
        start: k_start,
        end: k_end,
        step: k_step,
        body: k_body,
        ..
    }, Stmt::Expr(store_call)] = j_body.stmts.as_slice()
    else {
        return None;
    };
    if !is_zero_lit(sum_init) || !is_zero_lit(k_start) || !is_unit_step(k_step) {
        return None;
    }
    // A `@ZeroDrift` accumulator asks for an EXACT reduction, which the f32
    // kernel cannot provide — substituting it would silently discard the
    // guarantee, so this used to refuse outright. It is recorded instead now,
    // and `try_emit_gemm_kernel` selects a representation and dispatches to an
    // exact kernel; refusing still happens there if no representation is
    // satisfiable. The property that makes this worth doing is that integer
    // addition is associative, so the tiled, threaded, K-split reduction is
    // bit-identical to the naive loop rather than merely close to it.
    // See `docs/proof_carrying_kernels.md`.
    // sum = sum + load(A,i,k,K,..) * load(B,k,j,N,..)
    //
    // The operand loads' own `@bounds` are captured here, not just the
    // accumulator's. An exact `vpdpwssd` kernel's overflow obligation is stated
    // over A and B, and the accumulator's bound does not imply it — see
    // `DriftAccumulator::operand_bounds`.
    let [Stmt::Let {
        name: a_val,
        init: Some(a_init),
        bounds: a_bounds_attr,
        ty: a_val_ty,
        ..
    }, Stmt::Let {
        name: b_val,
        init: Some(b_init),
        bounds: b_bounds_attr,
        ty: b_val_ty,
        ..
    }, Stmt::Assign { target, value, .. }] = k_body.stmts.as_slice()
    else {
        return None;
    };

    let const_bounds = |b: &Option<BoundsAttr>| -> Option<(f64, f64)> {
        b.as_ref().and_then(|b| {
            match (const_f64_of(&b.min), const_f64_of(&b.max)) {
                (Some(lo), Some(hi)) => Some((lo, hi)),
                _ => None,
            }
        })
    };

    let drift = zero_drift.as_ref().map(|_| DriftAccumulator {
        ty: match sum_ty {
            Some(Type::Primitive(n, _)) | Some(Type::Ident(n, _)) => n.clone(),
            _ => "F32".to_string(),
        },
        bounds: const_bounds(sum_bounds),
        a_bounds: const_bounds(a_bounds_attr),
        b_bounds: const_bounds(b_bounds_attr),
    });

    let (a_buf, a_row, a_col, a_stride) = match_load(a_init)?;
    let (b_buf, b_row, b_col, b_stride) = match_load(b_init)?;

    // A is indexed [i, k] with row stride K; B is indexed [k, j] with row
    // stride N. Both must be row-major and non-transposed.
    if (a_row, a_col) != (i.as_str(), k.as_str()) {
        return None;
    }
    if (b_row, b_col) != (k.as_str(), j.as_str()) {
        return None;
    }

    if ident_of(target)? != sum.as_str() {
        return None;
    }
    let Expr::BinaryOp {
        op: BinaryOp::Add,
        left,
        right,
        ..
    } = value
    else {
        return None;
    };
    if ident_of(left)? != sum.as_str() {
        return None;
    }
    let Expr::BinaryOp {
        op: BinaryOp::Mul,
        left: ml,
        right: mr,
        ..
    } = &**right
    else {
        return None;
    };
    if ident_of(ml)? != a_val.as_str() || ident_of(mr)? != b_val.as_str() {
        return None;
    }

    // store(C, i, j, N, _, _, sum)
    let sargs = call_parts(store_call, "block_ptr2d_store")?;
    if sargs.len() != 7 {
        return None;
    }
    let c_buf = ident_of(&sargs[0])?;
    if ident_of(&sargs[1])? != i.as_str()
        || ident_of(&sargs[2])? != j.as_str()
        || ident_of(&sargs[6])? != sum.as_str()
    {
        return None;
    }
    let c_stride = ident_of(&sargs[3])?;

    let m = ident_of(i_end)?;
    let n = ident_of(j_end)?;
    let k_ext = ident_of(k_end)?;

    // The strides are RECORDED, not required to equal the extents. A row
    // stride larger than the extent is a submatrix, which is a legal and
    // common BLAS input; requiring `lda == K` here is what used to send every
    // such call to scalar lowering. The indexing order was already checked
    // above (A by `[i, k]`, B by `[k, j]`), which is the part that would
    // change the computation rather than just its addressing.
    //
    // A stride must still be a plain identifier — `match_load` and
    // `ident_of` enforce that — so an expression like `K + 1` is refused
    // rather than evaluated here, where there is no scope to evaluate it in.

    Some(GemmShape {
        a: a_buf.to_string(),
        b: b_buf.to_string(),
        c: c_buf.to_string(),
        m: m.to_string(),
        n: n.to_string(),
        k: k_ext.to_string(),
        lda: a_stride.to_string(),
        ldb: b_stride.to_string(),
        ldc: c_stride.to_string(),
        drift,
        // Both operands must be declared the SAME width for the product's type
        // to be unambiguous; a mismatch is not a shape this recogniser claims
        // to understand, so it reports None and the caller refuses the fast
        // path rather than guessing which one wins.
        operand_ty: match (a_val_ty, b_val_ty) {
            (Some(Type::Primitive(x, _)) | Some(Type::Ident(x, _)),
             Some(Type::Primitive(y, _)) | Some(Type::Ident(y, _))) if x == y => Some(x.clone()),
            _ => None,
        },
    })
}

/// Constant-folds the literals `@bounds(min, max)` accepts. Deliberately tiny:
/// a bound that is not a literal cannot be resolved here, and guessing one
/// would licence a representation whose range was never established.
fn const_f64_of(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::IntLit(v, _) => Some(*v as f64),
        Expr::FloatLit(v, _) => Some(*v),
        Expr::UnaryOp { op: crate::ast::UnaryOp::Neg, operand, .. } => {
            const_f64_of(operand).map(|v| -v)
        }
        _ => None,
    }
}

// ── IR emission ─────────────────────────────────────────────

/// Small helper for emitting structured loops as LLVM IR text.
///
/// Induction variables live in `alloca`s rather than `phi` nodes. That is not
/// laziness: `mem2reg` promotes them in the first function pass, and it keeps
/// this generator free of the SSA bookkeeping that hand-written phi chains
/// need. The vector width, the FMA structure and the blocking are explicit —
/// those are what the performance comes from — while register allocation is
/// left to the backend, which is better at it.
pub struct IrBuilder {
    pub out: String,
    n: usize,
    entry: Vec<String>,
}

pub struct LoopCtx {
    var: String,
    step: String,
    cond: String,
    body: String,
    end: String,
}

impl IrBuilder {
    pub fn new() -> Self {
        IrBuilder {
            out: String::new(),
            n: 0,
            entry: Vec::new(),
        }
    }

    fn t(&mut self) -> String {
        self.n += 1;
        format!("%g{}", self.n)
    }

    fn l(&mut self, tag: &str) -> String {
        self.n += 1;
        format!("{}.{}", tag, self.n)
    }

    fn w(&mut self, s: &str) {
        self.out.push_str("  ");
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// Declare an alloca that must sit in the entry block.
    fn entry_alloca(&mut self, name: &str, ty: &str, align: usize) {
        self.entry
            .push(format!("  {} = alloca {}, align {}", name, ty, align));
    }

    /// `for (var = start; var < end; var += step)`, signed.
    fn loop_begin(&mut self, tag: &str, start: &str, end: &str, step: &str) -> LoopCtx {
        self.n += 1;
        let var = format!("%iv{}", self.n);
        let cond = self.l(&format!("{}.cond", tag));
        let body = self.l(&format!("{}.body", tag));
        let end_l = self.l(&format!("{}.end", tag));
        self.entry_alloca(&var, "i64", 8);
        self.w(&format!("store i64 {}, ptr {}", start, var));
        self.w(&format!("br label %{}", cond));
        self.out.push_str(&format!("{}:\n", cond));
        let cur = self.t();
        self.w(&format!("{} = load i64, ptr {}", cur, var));
        let c = self.t();
        self.w(&format!("{} = icmp slt i64 {}, {}", c, cur, end));
        self.w(&format!("br i1 {}, label %{}, label %{}", c, body, end_l));
        self.out.push_str(&format!("{}:\n", body));
        LoopCtx {
            var,
            step: step.to_string(),
            cond,
            body,
            end: end_l,
        }
    }

    /// Current value of a loop's induction variable.
    fn iv(&mut self, ctx: &LoopCtx) -> String {
        let t = self.t();
        self.w(&format!("{} = load i64, ptr {}", t, ctx.var));
        t
    }

    fn loop_end(&mut self, ctx: LoopCtx) {
        let cur = self.t();
        self.w(&format!("{} = load i64, ptr {}", cur, ctx.var));
        let nx = self.t();
        self.w(&format!("{} = add i64 {}, {}", nx, cur, ctx.step));
        self.w(&format!("store i64 {}, ptr {}", nx, ctx.var));
        self.w(&format!("br label %{}", ctx.cond));
        self.out.push_str(&format!("{}:\n", ctx.end));
        let _ = ctx.body;
    }

    /// `max(a, b)` on i64.
    fn imax(&mut self, a: &str, b: &str) -> String {
        let c = self.t();
        self.w(&format!("{} = icmp sgt i64 {}, {}", c, a, b));
        let r = self.t();
        self.w(&format!("{} = select i1 {}, i64 {}, i64 {}", r, c, a, b));
        r
    }

    /// `min(a, b)` on i64.
    fn imin(&mut self, a: &str, b: &str) -> String {
        let c = self.t();
        self.w(&format!("{} = icmp slt i64 {}, {}", c, a, b));
        let r = self.t();
        self.w(&format!("{} = select i1 {}, i64 {}, i64 {}", r, c, a, b));
        r
    }

    fn gep(&mut self, ty: &str, base: &str, off: &str) -> String {
        let r = self.t();
        self.w(&format!(
            "{} = getelementptr inbounds {}, ptr {}, i64 {}",
            r, ty, base, off
        ));
        r
    }

    fn mul(&mut self, a: &str, b: &str) -> String {
        let r = self.t();
        self.w(&format!("{} = mul nsw i64 {}, {}", r, a, b));
        r
    }

    fn add(&mut self, a: &str, b: &str) -> String {
        let r = self.t();
        self.w(&format!("{} = add nsw i64 {}, {}", r, a, b));
        r
    }

    fn sub(&mut self, a: &str, b: &str) -> String {
        let r = self.t();
        self.w(&format!("{} = sub nsw i64 {}, {}", r, a, b));
        r
    }

    /// Finish a function: splice the entry-block allocas in ahead of the body.
    ///
    /// Every function gets `#0`, the target-cpu/target-features attribute set.
    /// Without it a helper that does not get inlined into the `#0` kernel is
    /// compiled for baseline x86-64, and the `<16 x float>` masked loads and
    /// stores scalarise into per-lane bit-test chains on 4-wide `xmm`. That is
    /// a ~7x slowdown that shows up only after an unrelated change stops the
    /// inliner from firing, which is how it was found.
    fn finish(&mut self, signature: &str) -> String {
        let mut s = String::new();
        let _ = writeln!(&mut s, "{} #0 {{", signature);
        let _ = writeln!(&mut s, "entry:");
        for a in &self.entry {
            let _ = writeln!(&mut s, "{}", a);
        }
        s.push_str(&self.out);
        let _ = writeln!(&mut s, "  ret void");
        let _ = writeln!(&mut s, "}}");
        s
    }
}

/// Name of the emitted kernel.
pub const KERNEL_NAME: &str = "__y_sgemm_f32_avx512";

/// Upper bound on worker threads. Also the size of the stack-allocated task
/// and `pthread_t` arrays, so it is a hard cap, not a hint.
///
/// Capped at physical-core count rather than `_SC_NPROCESSORS_ONLN`: a GEMM
/// micro-kernel saturates a core's FMA pipes on its own, so the second SMT
/// thread on a core adds scheduling pressure and scratch footprint for no
/// extra throughput.
pub const MAX_THREADS: usize = 16;

/// Floats of scratch each thread needs: one A panel plus one B panel.
///
/// Sized for the LARGEST tile `Tile::check` will accept, not for the tile in
/// force, because it sizes a static array and the tile is a runtime value. The
/// buffer is BSS, so the pages a smaller tile never touches are never faulted
/// in and cost nothing resident.
pub const SCRATCH_FLOATS: usize =
    (MC_MAX + MR_MAX) * KC_MAX + KC_MAX * (NC_MAX + NR_MAX);

/// Multiply-adds a thread must be given before it is worth adding.
///
/// OpenBLAS's rule is `nthreads = M*N*K / (SMP_THRESHOLD_MIN *
/// GEMM_MULTITHREAD_THRESHOLD)` with that product equal to `65536 * 4`
/// (`interface/gemm.c`). Using their constant here asks for 16 threads on a
/// 256^3 GEMM, which measures 391 GFLOPS against 465 on four — their figure is
/// calibrated for a per-thread kernel that does roughly half the work per
/// cycle that this one does, and for a driver that shares one packed B panel
/// between threads. This one re-packs B per thread, so each extra thread adds
/// real work rather than only splitting it.
///
/// Measured optima on a clean machine (Y GFLOPS by thread count):
///
/// | shape | 2 | 4 | 8 | 16 |
/// |---|---|---|---|---|
/// | 256^3 | 326 | **465** | 387 | 391 |
/// | 250^3 | 325 | **480** | 449 | 384 |
/// | 512^3 | 372 | 487 | **576** | 483 |
/// | 64x64x32768 | 168 | **257** | 160 | 157 |
/// | 17x4096x4096 | 111 | 168 | 234 | **318** |
pub const WORK_PER_THREAD: usize = 1 << 16;

/// Multiply-adds a shape must have before it is threaded AT ALL when the pool
/// is cold — i.e. when workers are blocked on the condvar rather than spinning.
///
/// `WORK_PER_THREAD` was measured in a throughput harness, which calls the
/// kernel back-to-back in a tight loop. The workers never exhaust their spin
/// window there, so a dispatch is a release store and threading is nearly free.
/// **That is not the only regime, and the harness is structurally incapable of
/// seeing the other one.** A caller that does anything at all between GEMMs —
/// even a `nanosleep(0)` — lets the workers park, and then a dispatch costs a
/// futex wake and a scheduler round trip.
///
/// Measured with `tests/cold_call_cpu_gemm.c`, which times a SINGLE call after
/// a gap, median over 150 isolated calls, 16 threads against 1:
///
/// | shape | work | 1 thread | 16 threads | |
/// |---|---|---|---|---|
/// | 56³ | 175,616 | 2.78 us | 24.50 us | **0.11x** |
/// | 72³ | 373,248 | 6.63 | 23.17 | 0.29x |
/// | 104³ | 1,124,864 | 14.03 | 26.91 | 0.52x |
/// | 128³ | 2,097,152 | 19.00 | 27.02 | 0.70x |
/// | 160³ | 4,096,000 | 43.71 | 32.33 | **1.35x** |
/// | 200³ | 8,000,000 | 97.11 | 40.71 | **2.39x** |
///
/// The 16-thread column is nearly FLAT at ~23-27 us from 56³ to 128³: that is a
/// fixed ~20 us dispatch cost, and it is the same whether `nt` is 2 or 16, so
/// it is per-dispatch and not per-thread. Threading therefore pays only once
/// the single-threaded time exceeds roughly twice it, which the table puts
/// between 128³ and 160³.
///
/// **Raising `POOL_SPIN` does not fix this and makes it worse.** Every larger
/// value is fast while the caller's gap fits inside the spin window and
/// catastrophic just past it — at 128³, spin 32768 measures 4.6 us at a 200 us
/// gap and **175 us** at a 500 us gap, against a uniform 22-28 us for the
/// shipped 512. A library cannot know the caller's gap, so that is tuning to an
/// unknown; the cliff is worse than the cost it removes.
///
/// So the thread count is chosen from the pool's actual state instead. Only two
/// data points bracket this constant (128³ loses, 160³ wins), so it is a
/// bracket, not a fitted value.
pub const COLD_MIN_WORK: usize = 3 << 20;

/// Nanoseconds since the previous GEMM within which a caller counts as a
/// throughput caller, so small shapes are still threaded.
///
/// The pool's own spin window (`POOL_SPIN`, ~3us) is far too short to use as
/// this signal, and the pool's parked state is the wrong signal entirely
/// because it latches — see `emit_threads`. This is a property of the CALLER,
/// not of the pool: 100us is comfortably longer than any back-to-back GEMM
/// sequence and far shorter than the gap that made isolated calls measure
/// 0.11x, so nothing in either measured regime sits near the boundary.
pub const HOT_WINDOW_NS: usize = 100_000;

/// Work above which threads share one packed B panel instead of each packing
/// their own.
///
/// Sharing is **not** a general win, which is the opposite of what the traffic
/// arithmetic predicts. Packing B once instead of `nthreads` times obviously
/// moves fewer bytes, but the panel is then produced by one thread and consumed
/// by all, so it travels through L3 rather than staying in the packer's own L2
/// — and the two barriers per block expose whatever load imbalance the M-split
/// left. Measured over four full runs each (mean Y/OB over 18 shapes):
///
/// | | private B | shared B |
/// |---|---|---|
/// | mean | **0.98** | 0.89 |
/// | 1024³ | 1230 GF | 960 GF |
/// | 1000³ | 1298 GF | 903 GF |
/// | 2048³ | 1770 GF | **2050 GF** |
///
/// So it pays only once the operands stop fitting comfortably in a 32 MB L3,
/// where the redundant re-reads become real DRAM traffic. 2048³ moves 16.8 MB
/// per operand; 1024³ moves 4 MB and simply stays cached however many times it
/// is re-read.
pub const SHARE_B_WORK: usize = 2 << 30;

/// Widest `N` the copy-free tiny path can serve.
///
/// This one is STRUCTURAL, not tuned: `N` lives in `N/16` accumulator vectors
/// per row and there are 32 zmm registers. Raising it past 64 does not make the
/// kernel slower by a little, it makes LLVM spill the accumulators — which is
/// the entire thing the path exists to avoid.
pub const TINY_MAX_N: usize = 64;

/// `(vectors of N, rows blocked per tile)`, one specialised body each.
///
/// The product is held at 24 accumulators, leaving the other 8 registers for
/// the `nv` B vectors and the A broadcast. `nv` cannot be a runtime value: the
/// accumulator count decides the register allocation, so a data-dependent one
/// puts C back in memory.
pub const TINY_BLOCK: [(usize, usize); 4] = [(1, 16), (2, 8), (3, 6), (4, 4)];

/// Multiply-adds below which the copy-free path replaces the packed one.
///
/// MEASURED, and the first guess (`1 << 18`) was wrong at its own boundary.
/// Against the packed path, 16 threads, four interleaved launches:
///
/// | shape | work | copy-free | packed | ratio |
/// |---|---|---|---|---|
/// | 48^3 | 110,592 | **230.1** | 127.2 | **1.81x** |
/// | 64^3 | 262,144 | 251.9 | **334.4** | **0.75x** |
///
/// So the crossover is bracketed between those two, and `1 << 18` = 262,144 sat
/// exactly on the losing side of it.
///
/// The reason is NOT the packing, which is what the original argument assumed.
/// **This path is single-threaded by construction** — C lives in registers
/// across the whole K loop, which is the entire point and is why it cannot be
/// split — so its advantage ends where the shape becomes worth threading at
/// all. With `WORK_PER_THREAD` at `1 << 16`, a shape of 262,144 multiply-adds
/// gets four threads, and four threads on the packed path beat one thread on a
/// better kernel.
///
/// That makes this constant and `WORK_PER_THREAD` related in fact, and it is
/// deliberately NOT written as a formula over it. A constant that is also a
/// guard reading another constant is how `SM_PU = 8` came to look
/// catastrophic; if `WORK_PER_THREAD` moves, **re-measure this** rather than
/// letting it drift silently.
///
/// Set to the conservative end of the bracket: routing a shape to the packed
/// path too early costs 0.75x at worst (measured at 64^3), while routing it
/// there too late costs 0.55x (measured at 48^3).
pub const TINY_MAX_WORK: usize = 1 << 17;

/// Depth of K below which the small-M path is not worth splitting along K.
///
/// The split costs one `M x N` zero-fill and one `nthr`-deep reduction per
/// call, both independent of K, so it only pays once the K-band each thread
/// gets is long enough to amortise them. `flatK 4096x4096x8` is the shape this
/// protects: K = 8 there, and every thread would reduce the whole of C to
/// avoid re-reading a B that is only 128 KB in the first place.
pub const KSPLIT_MIN_K: usize = 512;

/// Shortest K-band worth giving a thread. Caps the thread count when K is
/// modest, rather than handing out slivers whose reduction costs more than the
/// accumulation they saved.
pub const KSPLIT_MIN_BAND: usize = 128;

/// Largest K-split thread count the emitted wrapper will use. Mirrors the
/// `maxthr` substituted into `emit_vnni_threaded_module`.
pub const KSPLIT_MAX_THREADS: usize = 64;

/// The K-split thread count, as `__y_gemm_exact_threads` computes it.
///
/// This is a **transcription of emitted LLVM**, not a second decision. From
/// `emit_vnni_threaded_module`:
///
/// ```text
///   %r1  = select i1 %lo, i64 1, i64 %raw       ; clamp the request up to 1
///   %r2  = select i1 %hi, i64 64, i64 %r1       ; ...and down to maxthr
///   %byk = sdiv i64 %K, 128                     ; never a band under minband
///   %n   = select i1 %small, i64 %byk, i64 %ceil
///   %out = select i1 %z, i64 1, i64 %n
/// ```
///
/// It exists so the model can PREDICT an observable: the `--wrap=pthread_create`
/// canary in `tests/exact_gemm_thread_invariance.rs` counts real spawns, and
/// `the_min_band_floor_is_what_the_model_says` asserts the count is this
/// function's answer. That is the one place the model touches the shipped code
/// behaviourally rather than by inspection.
pub fn ksplit_threads(requested: usize, k: usize) -> usize {
    let ceil = requested.clamp(1, KSPLIT_MAX_THREADS);
    let by_k = k / KSPLIT_MIN_BAND;
    let n = by_k.min(ceil);
    if n < 1 { 1 } else { n }
}

/// Where `pack_a` puts row `i`, half `h` of a k-pair, inside its 2*MR slot group.
///
/// `Ap[p*(2*MR) + pack_a_slot(i, h)] = A[i][2p+h]`, zero outside the live tile.
/// Transcribed from `emit_vnni_pack_a`; `proofs/ExactGemmPacking.v` proves it a
/// bijection onto `[0, 2*MR)`, so no slot is written twice and none keeps a
/// value from the previous tile - the panel buffer really is reused.
pub fn pack_a_slot(i: usize, h: usize) -> usize {
    2 * i + h
}

/// Where `pack_b` puts column `j`, half `h`, inside its 2*NR slot group.
///
/// `Bp[p*(2*NR) + pack_b_slot(j, h)] = B[2p+h][j]`, zero outside the live tile.
///
/// The `(j / 16, j % 16)` split spells out the `vpdpwssd` lane layout: group
/// `v = j / 16` is one `<32 x i16>` vector and lane `l = j % 16` inside it
/// consumes int16 elements `2l` and `2l + 1`, so a column's two k-values are
/// ADJACENT and consecutive columns sit 2 int16 apart rather than 1.
///
/// **It is arithmetically the plain interleave `2*j + h`**, since
/// `16*(j/16) + (j%16) == j` - the decomposition folds away and computes
/// nothing extra. It is kept in that form because it is the derivation, and
/// because an inconsistent pair of constants (the `32` and the `16`) stops
/// it equalling `2*j + h`, which both `slot_b_is_the_plain_interleave` in
/// `proofs/ExactGemmPacking.v` and the model test assert.
///
/// **No proof here pins the lane layout**, and not merely because two
/// bijections are indistinguishable - there is no arithmetic difference to
/// distinguish. That lane `l` consumes elements `2l`/`2l+1` is an ISA fact;
/// `tests/cpu_gemm_vnni_micro.rs` covers it by mutating the stride against a
/// scalar reference on the real instruction.
pub fn pack_b_slot(j: usize, h: usize) -> usize {
    (j / 16) * 32 + (j % 16) * 2 + h
}

/// The flush chunking, transcribed from `emit_vnni_micro_module`'s outer loop.
///
/// The emitted loop is `for c = 0; c < kpairs; c += flush`, with
/// `cend = select (c + flush < kpairs), c + flush, kpairs` — so a chunk is
/// `[c, min(c + flush, kpairs))` and the final partial chunk is carried by the
/// same clamp rather than by a separate epilogue. `proofs/ExactGemmMicro.v`
/// proves this decomposition sums to the whole range (`flush_exact`), which is
/// the flush's analogue of the K-split's `bands_tile`.
///
/// This is the same clamped-tile shape as [`mn_tiles`], NOT the K-split's
/// uneven-band shape: the chunks are uniform with a short tail, because the
/// interval is a fixed overflow budget rather than a work split.
pub fn flush_chunks(kpairs: usize, flush: usize) -> Vec<(usize, usize)> {
    assert!(flush > 0, "a zero flush interval would not terminate");
    let mut out = Vec::new();
    let mut c = 0usize;
    while c < kpairs {
        out.push((c, kpairs.min(c + flush) - c));
        c += flush;
    }
    out
}

/// Which `<32 x i16>` vector, and which of its 16 lanes, carries panel slot `s`.
///
/// `vpdpwssd` treats a `<32 x i16>` operand as 16 lanes of two int16, so lane
/// `l` consumes elements `2l` and `2l + 1` of its own vector.
pub fn vec_of_slot(s: usize) -> usize {
    s / 32
}

/// See [`vec_of_slot`].
pub fn lane_of_slot(s: usize) -> usize {
    (s % 32) / 2
}

/// The i32 element index the micro-kernel loads row `i`'s k-pair `p` from.
///
/// The emitter writes `%aidx = mul i64 %p, MR` then `%ai{i} = add i64 %aidx, i`
/// and indexes `Ap` as **i32** - one load fetches both halves of the pair.
pub fn a_i32_element(p: usize, i: usize) -> usize {
    p * VNNI_MR + i
}

/// The two int16 panel slots that i32 load aliases, low half first.
///
/// `proofs/ExactGemmRegisterTile.v::the_i32_load_is_the_packed_pair` proves
/// these are exactly `pack_a_slot(i, 0)` and `pack_a_slot(i, 1)` inside k-pair
/// group `p`. Which half is `2p` rather than `2p+1` is little-endianness, an
/// ISA fact - and `swapping_the_pair_halves_computes_a_different_function`
/// shows it is load-bearing, not decorative.
pub fn a_pair_slots(p: usize, i: usize) -> (usize, usize) {
    let base = p * (VNNI_MR * 2);
    (base + pack_a_slot(i, 0), base + pack_a_slot(i, 1))
}

/// The two int16 B panel slots accumulator `acc[i][v]` lane `l` consumes.
///
/// `vpdpwssd` lane `l` reads elements `2l` and `2l+1` of the `<32 x i16>`
/// loaded at `Bp + p*NR*2 + v*32`, which are the two slots of column
/// `column_of_lane(v, l)` - proved as `the_lane_consumes_its_own_column`.
pub fn b_pair_slots_for_lane(p: usize, v: usize, l: usize) -> (usize, usize) {
    let base = p * (VNNI_NR * 2);
    let j = column_of_lane(v, l);
    (base + pack_b_slot(j, 0), base + pack_b_slot(j, 1))
}

/// Which column of the `MR x NR` tile the store reads back out of vector `v`,
/// lane `l` — the inverse leg of the round trip [`pack_b_slot`] opens.
///
/// `the_packed_column_is_the_stored_column` in `proofs/ExactGemmMicro.v` proves
/// the composition is the identity. A mismatch is a correctly-summed but
/// column-PERMUTED result, which no bound or bijection elsewhere can see.
pub fn column_of_lane(v: usize, l: usize) -> usize {
    16 * v + l
}

/// The output tiling, as `emit_vnni_gemm_driver`'s `vg.j` / `vg.i` loops
/// compute it: uniform tiles of `tile` with a CLAMPED ragged tail.
///
/// ```text
///   %rem_n = sub i64 %N, %j0
///   %nw    = call i64 @llvm.smin.i64(i64 %rem_n, i64 <NR>)
/// ```
///
/// This is the Rust transcription of `tw` / `toff` in
/// `proofs/ExactGemmTiling.v`, where `tiles_cover` proves the tiles account
/// for `[0, extent)` exactly and `tile_index_injective` /
/// `tile_index_surjective` prove `(tile, offset)` is a BIJECTION onto it - the
/// "written exactly once" obligation, which a coverage count alone does not
/// give (a tiling that writes one element twice and another never still
/// covers the right total).
///
/// Different decomposition from [`ksplit_bands`], deliberately: that one
/// spreads the remainder one k at a time across all bands, this one clamps a
/// single ragged tail. Do not unify them.
///
/// Returns `(offset, width)` per tile. A `tile` of 0 has no meaning and panics.
pub fn mn_tiles(extent: usize, tile: usize) -> Vec<(usize, usize)> {
    assert!(tile > 0, "a tiling with zero-width tiles does not terminate");
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < extent {
        out.push((off, (extent - off).min(tile)));
        off += tile;
    }
    out
}

/// The K-band decomposition, as the emitted wrapper's spawn loop computes it:
/// `base = K / nthr`, `rem = K % nthr`, and the first `rem` bands take one
/// extra k so the cuts are uneven.
///
/// This is the Rust transcription of `blen` / `boff` in
/// `proofs/ExactGemmKsplit.v`, where `bands_tile` proves the bands cover
/// `[0, K)` exactly for every `K` and every positive `nthr`, and
/// `ksplit_exact` proves that summing their partials equals the naive sum.
/// `tests/exact_gemm_ksplit_model.rs` checks this transcription against that
/// theorem over a finite range.
///
/// Returns `(offset, len)` per band. Panics on `nthr == 0`, which the emitted
/// code cannot produce - `ksplit_threads` floors at 1.
pub fn ksplit_bands(nthr: usize, k: usize) -> Vec<(usize, usize)> {
    assert!(nthr > 0, "a K-split with no workers has no bands");
    let base = k / nthr;
    let rem = k % nthr;
    let mut out = Vec::with_capacity(nthr);
    let mut off = 0usize;
    for t in 0..nthr {
        let len = base + usize::from(t < rem);
        out.push((off, len));
        off += len;
    }
    out
}

const SCRATCH: &str = "@__y_gemm_scratch";
const SCRATCH_CAP: &str = "@__y_gemm_scratch_threads";
const NTHREADS_CACHE: &str = "@__y_gemm_nthreads";
/// Statically reserved single-thread panel, used only if `malloc` fails.
const FALLBACK: &str = "@__y_gemm_fallback";
const POOL_N: &str = "@__y_pool_n";
const POOL_GEN: &str = "@__y_pool_gen";
const POOL_DONE: &str = "@__y_pool_done";
const POOL_TASK: &str = "@__y_pool_task";
const POOL_IDS: &str = "@__y_pool_ids";

const POOL_MUTEX: &str = "@__y_pool_mutex";
const POOL_COND: &str = "@__y_pool_cond";
const POOL_BARRIER: &str = "@__y_pool_barrier";
const BARRIER_N: &str = "@__y_barrier_n";
/// Nanosecond timestamp of the previous GEMM call, used to tell a throughput
/// caller from an isolated one. See `COLD_MIN_WORK`.
const POOL_LAST_NS: &str = "@__y_pool_last_ns";
/// One B panel shared by all threads when the split is along M.
const SHARED_B: &str = "@__y_shared_b";

/// Pause iterations a worker spins before blocking on the condition variable.
///
/// Short on purpose. The spin only exists to cover the gap between
/// back-to-back dispatches; past that a worker must genuinely block. An
/// earlier version spun 8192 times and then polled `usleep(100)` forever,
/// which left 15 threads waking 10k times a second and spinning hot into
/// whatever ran next — it corrupted the very A/B benchmark used to tune it,
/// reading OpenBLAS at 15 GFLOPS on a shape where it does 460.
const POOL_SPIN: usize = 512;

/// Module-level declarations the kernel needs.
///
/// Packing scratch is one heap block of `SCRATCH_FLOATS` per thread, allocated
/// once and reused. It used to be two fixed globals, which made the kernel
/// single-threaded by construction — threads sharing a packed panel corrupt
/// each other's data rather than merely contending for it.
pub fn emit_globals() -> String {
    let mut s = String::new();
    let _ = writeln!(&mut s, "; --- Y CPU GEMM: shared state ---");
    let _ = writeln!(
        &mut s,
        // Fields 17-19 (lda, ldb, ldc) are APPENDED rather than placed next to
        // M/N/K, so every existing index in `emit_worker` and the task-fill
        // loop keeps its meaning. A struct whose field numbering shifts is the
        // one change here that would compile, run, and silently pass the wrong
        // value to a worker.
        "%y_gemm_task = type {{ ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64 }}"
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global ptr null, align 8   ; per-thread packing scratch",
        SCRATCH
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8      ; threads the scratch was sized for",
        SCRATCH_CAP
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8      ; 0 = not yet resolved",
        NTHREADS_CACHE
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x float] zeroinitializer, align 64  ; malloc-failure panel",
        FALLBACK, SCRATCH_FLOATS
    );
    let _ = writeln!(
        &mut s,
        "@.y_gemm_env = private unnamed_addr constant [15 x i8] c\"Y_NUM_THREADS\\00\\00\""
    );
    let _ = writeln!(
        &mut s,
        "declare <16 x float> @llvm.fmuladd.v16f32(<16 x float>, <16 x float>, <16 x float>)"
    );
    let _ = writeln!(
        &mut s,
        "declare void @llvm.masked.store.v16f32.p0(<16 x float>, ptr, i32 immarg, <16 x i1>)"
    );
    let _ = writeln!(
        &mut s,
        "declare <16 x float> @llvm.masked.load.v16f32.p0(ptr, i32 immarg, <16 x i1>, <16 x float>)"
    );
    // `malloc`/`free` are already declared by the emitter prelude; redeclaring
    // them is an invalid redefinition, not a duplicate that LLVM merges.
    let _ = writeln!(&mut s, "declare ptr @getenv(ptr)");
    let _ = writeln!(&mut s, "declare i32 @atoi(ptr)");
    let _ = writeln!(&mut s, "declare i64 @sysconf(i32)");
    let _ = writeln!(&mut s, "declare i32 @clock_gettime(i32, ptr)");
    let _ = writeln!(
        &mut s,
        "declare i32 @pthread_create(ptr, ptr, ptr, ptr)"
    );
    let _ = writeln!(&mut s, "declare i32 @pthread_join(i64, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_mutex_init(ptr, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_mutex_lock(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_mutex_unlock(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_cond_init(ptr, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_cond_wait(ptr, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_cond_broadcast(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_barrier_init(ptr, ptr, i32)");
    let _ = writeln!(&mut s, "declare i32 @pthread_barrier_destroy(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_barrier_wait(ptr)");
    let _ = writeln!(
        &mut s,
        "{} = internal global [64 x i8] zeroinitializer, align 16",
        POOL_BARRIER
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8   ; barrier width, 0 = uninitialised",
        BARRIER_N
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8   ; ns timestamp of the last call",
        POOL_LAST_NS
    );
    // Shared B panel. When every thread walks the same (jc, pc) blocks — the
    // `ntn == 1` column of the grid — they all want the same packed B, and
    // packing it once instead of once per thread is the difference between
    // adding a thread that splits the work and adding a thread that also
    // re-reads all of B.
    //
    // Sized for the LARGEST `kc` the runtime rule can produce, not for the
    // tile's `kc`, exactly as `SCRATCH_FLOATS` is. It used to be
    // `tl().kc * (tl().nc + tl().nr)`, which is only correct while `kc` stays
    // at its compile-time 256 — and `kc` has been chosen at runtime since the
    // `L3_PANEL_FLOATS` rule landed. `pack_b` writes `kc * roundup(nc, NR)`
    // floats, and `kc*nc` is pinned to `L3_PANEL_FLOATS / nthr` by that rule,
    // so at `nthr = 16` the write is 524,288 floats against a 532,480-float
    // buffer: it fits by 1.5%, and the roundup to a whole NR panel is enough to
    // break it. `192 x 513 x 22000` overflows this global by ~92 KB into the
    // pool mutex and condvar that follow it. Nothing in the benchmark set has a
    // ragged N at a work size past `SHARE_B_WORK`, so nothing caught it.
    //
    // The bound is now structural rather than arithmetic: `kc <= KC_MAX` and
    // `roundup(nc, NR) <= NC_MAX + NR_MAX` hold whatever the runtime rule does.
    // It is BSS, so the pages a smaller panel never touches are never faulted
    // in and cost nothing resident.
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x float] zeroinitializer, align 64",
        SHARED_B,
        KC_MAX * (NC_MAX + NR_MAX)
    );
    // Over-sized and initialised by call rather than by relying on the
    // all-zero glibc static initialisers, so the layout is not assumed.
    let _ = writeln!(
        &mut s,
        "{} = internal global [64 x i8] zeroinitializer, align 16",
        POOL_MUTEX
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [64 x i8] zeroinitializer, align 16",
        POOL_COND
    );
    let _ = writeln!(&mut s, "declare void @llvm.x86.sse2.pause()");
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8   ; workers running (0 = pool not started)",
        POOL_N
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8   ; bumped once per dispatch",
        POOL_GEN
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x i64] zeroinitializer, align 64  ; generation each worker finished",
        POOL_DONE, MAX_THREADS
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x %y_gemm_task] zeroinitializer, align 64",
        POOL_TASK, MAX_THREADS
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x i64] zeroinitializer, align 8    ; each worker's index",
        POOL_IDS, MAX_THREADS
    );
    s
}

/// `<16 x i1>` mask selecting the first `rem` lanes (saturating at 0 and 16).
fn lane_mask(b: &mut IrBuilder, rem: &str) -> String {
    let t = b.t();
    b.w(&format!(
        "{} = insertelement <16 x i64> poison, i64 {}, i32 0",
        t, rem
    ));
    let sp = b.t();
    b.w(&format!(
        "{} = shufflevector <16 x i64> {}, <16 x i64> poison, <16 x i32> zeroinitializer",
        sp, t
    ));
    let m = b.t();
    b.w(&format!(
        "{} = icmp slt <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, \
         i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, {}",
        m, sp
    ));
    m
}

/// Splat a scalar float across `<16 x float>`.
fn splat(b: &mut IrBuilder, v: &str) -> String {
    let t = b.t();
    b.w(&format!(
        "{} = insertelement <16 x float> poison, float {}, i32 0",
        t, v
    ));
    let s = b.t();
    b.w(&format!(
        "{} = shufflevector <16 x float> {}, <16 x float> poison, <16 x i32> zeroinitializer",
        s, t
    ));
    s
}

/// `packB(src, ldb, kc, nc)` — copy a `kc x nc` block of B into
/// `[panel][k][NR]` order, zero-filling the N tail so the micro-kernel never
/// needs an N-tail branch.
fn emit_pack_b() -> String {
    let mut b = IrBuilder::new();
    let (src, ldb, kc, nc, dst, tid, nthr) =
        ("%src", "%ldb", "%kc", "%nc", "%dst", "%tid", "%nthr");

    // Panel `p` is packed by thread `p % nthr`. With nthr = 1 this is the
    // ordinary sequential pack.
    let start = b.mul(tid, &tl().nr.to_string());
    let stride = b.mul(nthr, &tl().nr.to_string());
    let jl = b.loop_begin("pb.j", &start, nc, &stride);
    let j0 = b.iv(&jl);
    let rem_n = b.sub(nc, &j0);
    let jw = b.imin(&rem_n, &tl().nr.to_string());

    // dst = packB + j0 * kc
    let dbase = b.mul(&j0, kc);
    let dst0 = b.gep("float", dst, &dbase);

    let pl = b.loop_begin("pb.p", "0", kc, "1");
    let p = b.iv(&pl);
    let srow = b.mul(&p, ldb);
    let soff = b.add(&srow, &j0);
    let sp = b.gep("float", src, &soff);
    let drow = b.mul(&p, &tl().nr.to_string());
    let dp = b.gep("float", &dst0, &drow);

    // Copy jw lanes, zero the rest. NR is 32 floats = two vectors.
    for v in 0..tl().nrv() {
        let lo = (v * 16).to_string();
        let rem = b.sub(&jw, &lo);
        let mask = lane_mask(&mut b, &rem);
        let s = b.gep("float", &sp, &lo);
        let d = b.gep("float", &dp, &lo);
        let val = b.t();
        b.w(&format!(
            "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, <16 x i1> {}, \
             <16 x float> zeroinitializer)",
            val, s, mask
        ));
        // Store all 16 lanes: the masked-off ones are zero, which is exactly
        // the padding the micro-kernel needs.
        b.w(&format!("store <16 x float> {}, ptr {}, align 4", val, d));
    }

    b.loop_end(pl);
    b.loop_end(jl);
    b.finish(&format!(
        "define internal void @__y_gemm_pack_b(ptr noalias {}, i64 {}, i64 {}, i64 {}, \
         ptr noalias {}, i64 {}, i64 {})",
        src, ldb, kc, nc, dst, tid, nthr
    ))
}

/// `packA(src, lda, mc, kc)` — copy an `mc x kc` block of A into
/// `[panel][k][MR]` order. This is a transpose: A is row-major, so the
/// micro-kernel would otherwise walk it by column, at one cache miss per
/// element.
fn emit_pack_a() -> String {
    let mut b = IrBuilder::new();
    let (src, lda, mc, kc, dst) = ("%src", "%lda", "%mc", "%kc", "%dst");

    let il = b.loop_begin("pa.i", "0", mc, &tl().mr.to_string());
    let i0 = b.iv(&il);
    let rem_m = b.sub(mc, &i0);
    let iw = b.imin(&rem_m, &tl().mr.to_string());

    let dbase = b.mul(&i0, kc);
    let dst0 = b.gep("float", dst, &dbase);

    let pl = b.loop_begin("pa.p", "0", kc, "1");
    let p = b.iv(&pl);
    let drow = b.mul(&p, &tl().mr.to_string());
    let dp = b.gep("float", &dst0, &drow);

    for i in 0..tl().mr {
        let ic = i.to_string();
        let inb = b.t();
        b.w(&format!("{} = icmp slt i64 {}, {}", inb, ic, iw));
        // Row i0+i of A, column p. Out-of-range rows read element 0 and are
        // then forced to zero, so the padded rows contribute nothing.
        let row = b.add(&i0, &ic);
        let ro = b.mul(&row, lda);
        let so = b.add(&ro, &p);
        let sso = b.t();
        b.w(&format!(
            "{} = select i1 {}, i64 {}, i64 0",
            sso, inb, so
        ));
        let sp = b.gep("float", src, &sso);
        let raw = b.t();
        b.w(&format!("{} = load float, ptr {}, align 4", raw, sp));
        let val = b.t();
        b.w(&format!(
            "{} = select i1 {}, float {}, float 0.0",
            val, inb, raw
        ));
        let d = b.gep("float", &dp, &ic);
        b.w(&format!("store float {}, ptr {}, align 4", val, d));
    }

    b.loop_end(pl);
    b.loop_end(il);
    b.finish(&format!(
        "define internal void @__y_gemm_pack_a(ptr noalias {}, i64 {}, i64 {}, i64 {}, \
         ptr noalias {})",
        src, lda, mc, kc, dst
    ))
}

/// The micro-kernel: `C[mw x jw] (+)= Ap[kc x MR] * B[kc x jw]`, with all
/// `MR * NRV` accumulators live in registers across the whole K loop.
///
/// `ldb` is a parameter rather than a constant so one kernel serves both
/// sources of B: the packed panel (`ldb = NR`, zero-padded, so the loads are
/// always full width) and B in place (`ldb = N`, unpadded, so the final column
/// panel must be masked). Packing B only pays when it is reused across several
/// row panels; when M is small there is one panel and packing is pure loss.
fn emit_micro() -> String {
    let mut b = IrBuilder::new();
    let (ap, bp, ldb, c, ldc, kc, mw, jw, first) = (
        "%ap", "%bp", "%ldb", "%c", "%ldc", "%kc", "%mw", "%jw", "%first",
    );

    // Accumulators. mem2reg promotes these to registers; MR*NRV = 24 vectors
    // plus NRV operands and one broadcast fits the 32 zmm registers.
    for i in 0..tl().mr {
        for v in 0..tl().nrv() {
            let name = format!("%acc{}_{}", i, v);
            b.entry_alloca(&name, "<16 x float>", 64);
            b.w(&format!(
                "store <16 x float> zeroinitializer, ptr {}, align 64",
                name
            ));
        }
    }

    // Two copies of the K loop: full-width loads when the panel is complete,
    // masked loads only for the ragged final panel. Emitting one masked loop
    // would put a mask computation in the innermost loop of every shape to
    // serve the tail of one panel.
    let full = b.t();
    b.w(&format!("{} = icmp eq i64 {}, {}", full, jw, tl().nr));
    let full_l = b.l("mk.full");
    let tail_l = b.l("mk.tail");
    let epi_l = b.l("mk.epi");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        full, full_l, tail_l
    ));

    for (masked, entry) in [(false, full_l), (true, tail_l)] {
        b.out.push_str(&format!("{}:\n", entry));
        let pl = b.loop_begin("mk.p", "0", kc, "1");
        let p = b.iv(&pl);

        // B row: jw contiguous floats at stride ldb.
        let bo = b.mul(&p, ldb);
        let brow = b.gep("float", bp, &bo);
        let mut bv = Vec::new();
        for v in 0..tl().nrv() {
            let off = (v * 16).to_string();
            let q = b.gep("float", &brow, &off);
            let t = b.t();
            if masked {
                let rem = b.sub(jw, &off);
                let mask = lane_mask(&mut b, &rem);
                b.w(&format!(
                    "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                     <16 x i1> {}, <16 x float> zeroinitializer)",
                    t, q, mask
                ));
            } else {
                b.w(&format!("{} = load <16 x float>, ptr {}, align 4", t, q));
            }
            bv.push(t);
        }

        // A panel row: MR contiguous floats, each broadcast.
        let ao = b.mul(&p, &tl().mr.to_string());
        let arow = b.gep("float", ap, &ao);
        for i in 0..tl().mr {
            let off = i.to_string();
            let q = b.gep("float", &arow, &off);
            let s = b.t();
            b.w(&format!("{} = load float, ptr {}, align 4", s, q));
            let a = splat(&mut b, &s);
            for v in 0..tl().nrv() {
                let name = format!("%acc{}_{}", i, v);
                let cur = b.t();
                b.w(&format!(
                    "{} = load <16 x float>, ptr {}, align 64",
                    cur, name
                ));
                let nv = b.t();
                b.w(&format!(
                    "{} = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> {}, \
                     <16 x float> {}, <16 x float> {})",
                    nv, a, bv[v], cur
                ));
                b.w(&format!(
                    "store <16 x float> {}, ptr {}, align 64",
                    nv, name
                ));
            }
        }
        b.loop_end(pl);
        b.w(&format!("br label %{}", epi_l));
    }
    b.out.push_str(&format!("{}:\n", epi_l));

    // Write out the mw live rows, masked to jw columns.
    //
    // The two cases are emitted as separate blocks under one branch rather than
    // as a `select` over both values. A select needs the accumulate operand,
    // which means reading C back on *every* K-block including the first — and
    // when K fits in one block that read is pure waste proportional to the size
    // of C. At 4096x4096x8 it was a 64 MB read supporting 268 MFLOP.
    let acc_l = b.l("mk.acc");
    let st_l = b.l("mk.first");
    let done_l = b.l("mk.done");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        first, st_l, acc_l
    ));

    for (accumulate, entry) in [(false, st_l), (true, acc_l)] {
        b.out.push_str(&format!("{}:\n", entry));
        for i in 0..tl().mr {
            let ic = i.to_string();
            let live = b.t();
            b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
            let do_l = b.l("mk.st");
            let skip_l = b.l("mk.sk");
            b.w(&format!(
                "br i1 {}, label %{}, label %{}",
                live, do_l, skip_l
            ));
            b.out.push_str(&format!("{}:\n", do_l));

            let ro = b.mul(&ic, ldc);
            let crow = b.gep("float", c, &ro);
            for v in 0..tl().nrv() {
                let lo = (v * 16).to_string();
                let rem = b.sub(jw, &lo);
                let mask = lane_mask(&mut b, &rem);
                let q = b.gep("float", &crow, &lo);
                let name = format!("%acc{}_{}", i, v);
                let acc = b.t();
                b.w(&format!(
                    "{} = load <16 x float>, ptr {}, align 64",
                    acc, name
                ));
                let val = if accumulate {
                    let old = b.t();
                    b.w(&format!(
                        "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                         <16 x i1> {}, <16 x float> zeroinitializer)",
                        old, q, mask
                    ));
                    let sum = b.t();
                    b.w(&format!("{} = fadd <16 x float> {}, {}", sum, acc, old));
                    sum
                } else {
                    acc
                };
                b.w(&format!(
                    "call void @llvm.masked.store.v16f32.p0(<16 x float> {}, ptr {}, i32 4, \
                     <16 x i1> {})",
                    val, q, mask
                ));
            }
            b.w(&format!("br label %{}", skip_l));
            b.out.push_str(&format!("{}:\n", skip_l));
        }
        b.w(&format!("br label %{}", done_l));
    }
    b.out.push_str(&format!("{}:\n", done_l));

    b.finish(&format!(
        "define internal void @__y_gemm_micro(ptr noalias {}, ptr noalias {}, i64 {}, ptr {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i1 {})",
        ap, bp, ldb, c, ldc, kc, mw, jw, first
    ))
}

/// Rows of C the small-M kernel keeps broadcast in registers.
pub const SM_MR: usize = 8;
/// Above this many rows the packed path wins and this one is not used.
///
/// Measured by sweeping the threshold over the 4096-wide skinny family:
/// against OpenBLAS this path is 4.05x at M=1, 0.88x at M=4 and 0.74x at M=8,
/// but at M=17 it measures 0.467 where the packed path measures 0.663 — B is
/// re-read once per 8-row block, and by M=17 that costs more than the packing
/// it avoids. Thresholds of 8, 16 and 24 measure 1.622 / 1.638 / 1.599 mean
/// ratio over the whole shape set, so this is the conservative end of a flat
/// optimum, not a knife edge.
pub const SM_MAX_M: usize = 8;

/// K-steps the small-M kernel folds into one pass over its C strip.
///
/// The inner loop's cost is not the FMAs, it is the read-modify-write of C:
/// one B vector drives `mw` C loads, `mw` FMAs and `mw` stores, so the load and
/// store ports run out long before the FMA pipes do. Holding C in a register
/// across `SM_PU` consecutive K-steps divides that traffic by `SM_PU` while the
/// FMA count stays the same.
///
/// **4, and the register-pressure argument for a smaller value is wrong.** The
/// A broadcasts are `SM_MR * SM_PU` values live across the j loop, so 8x4 is 32
/// vectors and cannot fit alongside B and the accumulator — LLVM spills. It is
/// still faster, because a spilled broadcast reloads from L1 while the C
/// traffic it removes was missing L1 entirely. Measured against `SM_PU = 2`,
/// three interleaved launches, best of each:
///
/// | shape | 1 thread | 16 threads |
/// |---|---|---|
/// | decode 8x4096x4096 | **1.19x** | **1.18x** |
/// | decode 4x4096x4096 | **1.41x** | **1.26x** |
/// | gemv 1x4096x4096 | 0.98x | 0.98x |
/// | gemv 1x8192x8192 | 0.98x | 1.00x |
///
/// An earlier revision of this comment claimed 4 measured *worse* (207 against
/// 568 GFLOPS). That was two different probe variants compared across two runs
/// — one padded, one not — on a shape whose 64 MB B sits exactly on this part's
/// 64 MB L3, where probe absolutes move 2x between runs of an identical
/// protocol. **Sweep this through the real emitter and the real harness, never
/// through the probe.**
///
/// 8 is not better: it wins on `decode 8` (1.06x / 1.14x, inside the spread)
/// and loses `decode 4` outright — see `SM_PU_MIN_ROWS` for why that is the
/// gate rather than the depth.
pub const SM_PU: usize = 4;

/// Live rows below which the K-unroll is skipped entirely.
///
/// **Deliberately NOT `SM_PU`.** Tying the two together is what made
/// `SM_PU = 8` look catastrophic on `decode 4x4096x4096` — 0.47x at one thread
/// — because a 4-row shape then failed `mspan >= 8` and fell all the way to the
/// un-unrolled loop. The measurement was of the gate, not of the depth, and
/// reading it as "8 is too deep" would have been the wrong conclusion.
///
/// 2 is where the unroll starts paying: at one live row there is no C traffic
/// to divide (a `1 x N` strip is L1-resident already) while the `SM_MR * SM_PU`
/// broadcast block is emitted regardless, and `gemv 1x4096x4096` measured
/// 31.2 GFLOPS ungated against 27.4 unrolled. M = 2 and M = 3 are not covered
/// by any benchmark shape; they take the unrolled path on the strength of
/// M = 4, and correctness for them is pinned by `tests/cpu_gemm_threaded.rs`.
pub const SM_PU_MIN_ROWS: usize = 2;

/// Small-M kernel: `k` outer, `j` inner.
///
/// The packed path is wrong for a near-GEMM shape for two compounding reasons:
/// `MR = 12` rows are computed for however few exist, and B gets packed with
/// almost no reuse. Inverting the loops fixes both. B is then read exactly
/// once in perfect linear order — it is the large operand and the shape is
/// bandwidth-bound — while C, only `M * N` with M small, stays resident in
/// cache across the whole K loop and absorbs the repeated accumulation.
///
/// `ldc` is C's row stride and `n` remains B's, because they are no longer the
/// same buffer: under the K-split each thread accumulates into a private
/// `M x N` panel and only the reduction writes the caller's C. `[k0, k1)` is
/// this thread's band of K; the single-threaded and N-split callers pass
/// `0, K`.
fn emit_small_m() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c, _m, n, k, ldc, m0, m1, n0, n1, k0, k1) = (
        "%A", "%B", "%C", "%M", "%N", "%K", "%ldc", "%m0", "%m1", "%n0", "%n1", "%k0", "%k1",
    );
    // This path reads A and B IN PLACE — it is the copy-free small-M kernel,
    // so there is no packing step to absorb a stride and these must be used
    // directly at every address computation.
    let (lda, ldb) = ("%lda", "%ldb");

    // C is accumulated into across all of K, so this thread's slice of it
    // starts at zero. Only the owned columns are cleared: another thread may
    // be writing the rest of the same rows concurrently.
    let zw = b.sub(n1, n0);
    let zb = b.t();
    b.w(&format!("{} = shl i64 {}, 2", zb, zw));
    let zl = b.loop_begin("sm.z", m0, m1, "1");
    let zi = b.iv(&zl);
    let zro = b.mul(&zi, ldc);
    let zoff = b.add(&zro, n0);
    let zp = b.gep("float", c, &zoff);
    b.w(&format!(
        "call void @llvm.memset.p0.i64(ptr {}, i8 0, i64 {}, i1 false)",
        zp, zb
    ));
    b.loop_end(zl);

    // Full 16-wide columns, then a masked tail. Keeping the mask out of the
    // main j loop matters: it runs M*K*N/16 times. Loop-invariant in i and p,
    // so it is computed once here rather than per K-step.
    let span = b.sub(n1, n0);
    let span_full = b.t();
    b.w(&format!("{} = and i64 {}, -16", span_full, span));
    let n_full = b.add(n0, &span_full);

    // Split [k0, k1) into a run of whole SM_PU groups and a scalar remainder.
    let kspan = b.sub(k1, k0);
    let kwhole = b.t();
    b.w(&format!(
        "{} = and i64 {}, -{}",
        kwhole, kspan, SM_PU
    ));

    // At one live row the unroll is a pure loss and it was measured as one:
    // `gemv 1x4096x4096` single-threaded measured 31.2 GFLOPS un-unrolled
    // against 27.4 unrolled (best of three interleaved launches, same session).
    // What the unroll buys is divided C traffic, and a `1 x N` strip of C is
    // L1-resident already, so there is nothing to divide — while the cost, an
    // `SM_MR * SM_PU` block of A broadcasts, is emitted regardless.
    //
    // Gated on the row span, computed ABOVE the i loop rather than on `mw`
    // inside it. `mw` is defined inside the i loop, and deriving the loop
    // bounds below from it stops LLVM unswitching the per-row `i < mw`
    // branches out of the j loop, where they would run once per row per 16
    // columns. See `SM_PU_MIN_ROWS` for why the threshold is not `SM_PU`.
    let mspan = b.sub(m1, m0);
    let use_pu = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", use_pu, mspan, SM_PU_MIN_ROWS));
    let kmain = b.add(k0, &kwhole);

    let yes_l = b.l("sm.pu.yes");
    let no_l = b.l("sm.pu.no");
    let end_l = b.l("sm.pu.end");
    b.w(&format!("br i1 {}, label %{}, label %{}", use_pu, yes_l, no_l));

    // Two whole loop nests rather than one nest with a selected `kmain`.
    //
    // `kmain` is the unrolled loop's exit bound and the remainder loop's entry
    // bound. Choosing it with a `select` hides from LLVM that the two ranges
    // partition `[k0, k1)` and that the remainder is short, so the remainder
    // stops being peeled and the shared nest is compiled for a general trip
    // count. Branching here instead keeps both bounds affine in `k0`/`k1`, at
    // the cost of duplicating a function that is out-of-line anyway.
    //
    // Measured at `SM_PU = 2`, best of three launches in one session:
    // `decode 8x4096x4096` single-threaded 69.4 GFLOPS with the bound affine
    // against 59.8 with the `select`. Caveat on that figure: the arms were not
    // order-alternated, and it has NOT been re-measured since `SM_PU` moved to
    // 4. The structural reason stands either way; treat the 14% as indicative.
    for (tag, passes) in [
        (
            yes_l.as_str(),
            vec![
                (SM_PU, "sm.pu", k0.to_string(), kmain.clone()),
                (1usize, "sm.p1", kmain.clone(), k1.to_string()),
            ],
        ),
        (
            no_l.as_str(),
            vec![(1usize, "sm.s1", k0.to_string(), k1.to_string())],
        ),
    ] {
    b.out.push_str(&format!("{}:\n", tag));
    let il = b.loop_begin("sm.i", m0, m1, &SM_MR.to_string());
    let i0 = b.iv(&il);
    let rm = b.sub(m1, &i0);
    let mw = b.imin(&rm, &SM_MR.to_string());
    let crow_base = b.mul(&i0, ldc);

    for (nu, ptag, pstart, pend) in passes {
        let pl = b.loop_begin(ptag, &pstart, &pend, &nu.to_string());
        let p = b.iv(&pl);

        // Broadcast this group's A columns once, outside the j loop.
        let mut av: Vec<Vec<String>> = Vec::new();
        let mut brow: Vec<String> = Vec::new();
        for u in 0..nu {
            let pu_ = b.add(&p, &u.to_string());
            let mut row_av = Vec::new();
            for i in 0..SM_MR {
                let ic = i.to_string();
                let live = b.t();
                b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
                let row = b.add(&i0, &ic);
                let ro = b.mul(&row, lda);
                let off = b.add(&ro, &pu_);
                // Rows past mw read element 0 and are then zeroed, so they
                // contribute nothing and the load stays in bounds.
                let soff = b.t();
                b.w(&format!("{} = select i1 {}, i64 {}, i64 0", soff, live, off));
                let q = b.gep("float", a, &soff);
                let raw = b.t();
                b.w(&format!("{} = load float, ptr {}, align 4", raw, q));
                let val = b.t();
                b.w(&format!(
                    "{} = select i1 {}, float {}, float 0.0",
                    val, live, raw
                ));
                row_av.push(splat(&mut b, &val));
            }
            av.push(row_av);
            let brow_off = b.mul(&pu_, ldb);
            brow.push(b.gep("float", bb, &brow_off));
        }

        for (masked, tag) in [(false, "sm.j"), (true, "sm.t")] {
            let (start, end) = if masked {
                (n_full.clone(), n1.to_string())
            } else {
                (n0.to_string(), n_full.clone())
            };
            let jl = b.loop_begin(tag, &start, &end, "16");
            let j = b.iv(&jl);

            let mask = if masked {
                let rem = b.sub(n1, &j);
                Some(lane_mask(&mut b, &rem))
            } else {
                None
            };
            let mut bvv = Vec::new();
            for u in 0..nu {
                let bq = b.gep("float", &brow[u], &j);
                let v = b.t();
                match &mask {
                    Some(mk) => b.w(&format!(
                        "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                         <16 x i1> {}, <16 x float> zeroinitializer)",
                        v, bq, mk
                    )),
                    None => b.w(&format!("{} = load <16 x float>, ptr {}, align 4", v, bq)),
                }
                bvv.push(v);
            }

            for i in 0..SM_MR {
                let ic = i.to_string();
                let live = b.t();
                b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
                let do_l = b.l("sm.do");
                let sk_l = b.l("sm.sk");
                b.w(&format!("br i1 {}, label %{}, label %{}", live, do_l, sk_l));
                b.out.push_str(&format!("{}:\n", do_l));

                let ro = b.mul(&ic, ldc);
                let co = b.add(&crow_base, &ro);
                let co2 = b.add(&co, &j);
                let cq = b.gep("float", c, &co2);
                let mut acc = b.t();
                match &mask {
                    Some(mk) => b.w(&format!(
                        "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                         <16 x i1> {}, <16 x float> zeroinitializer)",
                        acc, cq, mk
                    )),
                    None => b.w(&format!("{} = load <16 x float>, ptr {}, align 4", acc, cq)),
                }
                // The whole point of the unroll: C is loaded once and stored
                // once for `nu` K-steps instead of once for each.
                for u in 0..nu {
                    let nv = b.t();
                    b.w(&format!(
                        "{} = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> {}, \
                         <16 x float> {}, <16 x float> {})",
                        nv, av[u][i], bvv[u], acc
                    ));
                    acc = nv;
                }
                match &mask {
                    Some(mk) => b.w(&format!(
                        "call void @llvm.masked.store.v16f32.p0(<16 x float> {}, ptr {}, \
                         i32 4, <16 x i1> {})",
                        acc, cq, mk
                    )),
                    None => b.w(&format!("store <16 x float> {}, ptr {}, align 4", acc, cq)),
                }
                b.w(&format!("br label %{}", sk_l));
                b.out.push_str(&format!("{}:\n", sk_l));
            }
            b.loop_end(jl);
        }

        b.loop_end(pl);
    }
    b.loop_end(il);
    b.w(&format!("br label %{}", end_l));
    }
    b.out.push_str(&format!("{}:\n", end_l));

    b.finish(&format!(
        "define internal void @__y_gemm_small_m(ptr noalias {}, ptr noalias {}, \
         ptr {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {})",
        a, bb, c, _m, n, k, ldc, m0, m1, n0, n1, k0, k1, lda, ldb
    ))
}

/// `__y_gemm_tiny(A, B, C, M, N, K)` — the copy-free path: no packing at all,
/// A and B read in place, and the whole `MR' x N` block of C held in registers
/// across the entire K loop.
///
/// The packed path exists to make B's access pattern linear and to amortise
/// that cost over many row panels. Below roughly `MC x NC` there are no row
/// panels to amortise against, and at `48^3` the pack moves 4,608 floats to
/// support 110,592 multiply-adds — while OpenBLAS, whose own `SMALL_MATRIX_OPT`
/// is disabled on ZEN so it is using its *ordinary* packed path, was still
/// 2.3x ahead. So the pack is not the whole story and the arithmetic says so:
/// what this kernel removes as well is every C round trip. The packed
/// micro-kernel writes C once per `kc`-block per micro-panel; here C is loaded
/// never and stored exactly once.
///
/// `N` is capped at `TINY_MAX_N` because it is held in registers: `nv = N/16`
/// accumulator vectors per row, `MR' x nv <= 24` of the 32 zmm registers, with
/// the rest for the `nv` B vectors and the A broadcast. Hence one specialised
/// body per `nv`, with its own row block — a runtime `nv` would put the
/// accumulators in memory and defeat the entire point.
///
/// Rows past `mw` in the final row block are not branched around; their A
/// element is forced to zero and the FMA runs anyway, so the K loop stays a
/// straight-line register chain. The waste is bounded by one row block on one
/// tile, and only when `M % MR' != 0`.
fn emit_tiny() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c, m, n, k) = ("%A", "%B", "%C", "%M", "%N", "%K");
    // Copy-free: A, B and C are all the caller's memory, read and written in
    // place, so all three strides are live here. `n` remains the EXTENT that
    // selects which specialised body runs.
    let (lda, ldb, ldc) = ("%lda", "%ldb", "%ldc");

    for (nv, mrv) in TINY_BLOCK {
        for i in 0..mrv {
            for v in 0..nv {
                b.entry_alloca(&format!("%ty{}a{}_{}", nv, i, v), "<16 x float>", 64);
            }
        }
    }

    let done_l = b.l("ty.done");
    let entries: Vec<String> = TINY_BLOCK
        .iter()
        .map(|(nv, _)| b.l(&format!("ty.v{}", nv)))
        .collect();

    // Pick the body by N. The caller has already checked `N <= TINY_MAX_N`, so
    // the final `else` is the widest body and needs no test of its own.
    for (idx, entry) in entries.iter().enumerate().take(TINY_BLOCK.len() - 1) {
        let next = b.l("ty.pick");
        let cnd = b.t();
        b.w(&format!(
            "{} = icmp sle i64 {}, {}",
            cnd,
            n,
            (idx + 1) * 16
        ));
        b.w(&format!(
            "br i1 {}, label %{}, label %{}",
            cnd, entry, next
        ));
        b.out.push_str(&format!("{}:\n", next));
    }
    b.w(&format!("br label %{}", entries[TINY_BLOCK.len() - 1]));

    for (idx, (nv, mrv)) in TINY_BLOCK.iter().enumerate() {
        let (nv, mrv) = (*nv, *mrv);
        b.out.push_str(&format!("{}:\n", entries[idx]));

        // Only the LAST vector of a row can be ragged, and its mask is
        // loop-invariant, so it is built once out here rather than per K-step.
        let rem = b.sub(n, &((nv - 1) * 16).to_string());
        let mask = lane_mask(&mut b, &rem);

        let il = b.loop_begin("ty.i", "0", m, &mrv.to_string());
        let i0 = b.iv(&il);
        let rm = b.sub(m, &i0);
        let mw = b.imin(&rm, &mrv.to_string());
        for i in 0..mrv {
            for v in 0..nv {
                b.w(&format!(
                    "store <16 x float> zeroinitializer, ptr %ty{}a{}_{}, align 64",
                    nv, i, v
                ));
            }
        }

        let pl = b.loop_begin("ty.p", "0", k, "1");
        let p = b.iv(&pl);

        // One row of B, in place: N contiguous floats at row stride ldb.
        let bo = b.mul(&p, ldb);
        let brow = b.gep("float", bb, &bo);
        let mut bv = Vec::new();
        for v in 0..nv {
            let q = b.gep("float", &brow, &(v * 16).to_string());
            let t = b.t();
            if v == nv - 1 {
                b.w(&format!(
                    "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                     <16 x i1> {}, <16 x float> zeroinitializer)",
                    t, q, mask
                ));
            } else {
                b.w(&format!("{} = load <16 x float>, ptr {}, align 4", t, q));
            }
            bv.push(t);
        }

        // One column of A, in place: stride K between rows, broadcast per row.
        for i in 0..mrv {
            let ic = i.to_string();
            let live = b.t();
            b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
            let row = b.add(&i0, &ic);
            let ro = b.mul(&row, lda);
            let off = b.add(&ro, &p);
            // A dead row reads element 0 and is zeroed, so the load stays in
            // bounds and the FMA below contributes nothing.
            let soff = b.t();
            b.w(&format!("{} = select i1 {}, i64 {}, i64 0", soff, live, off));
            let q = b.gep("float", a, &soff);
            let raw = b.t();
            b.w(&format!("{} = load float, ptr {}, align 4", raw, q));
            let val = b.t();
            b.w(&format!(
                "{} = select i1 {}, float {}, float 0.0",
                val, live, raw
            ));
            let av = splat(&mut b, &val);
            for v in 0..nv {
                let name = format!("%ty{}a{}_{}", nv, i, v);
                let cur = b.t();
                b.w(&format!("{} = load <16 x float>, ptr {}, align 64", cur, name));
                let nx = b.t();
                b.w(&format!(
                    "{} = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> {}, \
                     <16 x float> {}, <16 x float> {})",
                    nx, av, bv[v], cur
                ));
                b.w(&format!("store <16 x float> {}, ptr {}, align 64", nx, name));
            }
        }
        b.loop_end(pl);

        // C is STORED, never loaded: this path owns all of K, so there is no
        // previous K-block to accumulate onto.
        for i in 0..mrv {
            let ic = i.to_string();
            let live = b.t();
            b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
            let st_l = b.l("ty.st");
            let sk_l = b.l("ty.sk");
            b.w(&format!("br i1 {}, label %{}, label %{}", live, st_l, sk_l));
            b.out.push_str(&format!("{}:\n", st_l));
            let row = b.add(&i0, &ic);
            let ro = b.mul(&row, ldc);
            for v in 0..nv {
                let off = b.add(&ro, &(v * 16).to_string());
                let q = b.gep("float", c, &off);
                let name = format!("%ty{}a{}_{}", nv, i, v);
                let cur = b.t();
                b.w(&format!("{} = load <16 x float>, ptr {}, align 64", cur, name));
                if v == nv - 1 {
                    b.w(&format!(
                        "call void @llvm.masked.store.v16f32.p0(<16 x float> {}, ptr {}, \
                         i32 4, <16 x i1> {})",
                        cur, q, mask
                    ));
                } else {
                    b.w(&format!("store <16 x float> {}, ptr {}, align 4", cur, q));
                }
            }
            b.w(&format!("br label %{}", sk_l));
            b.out.push_str(&format!("{}:\n", sk_l));
        }
        b.loop_end(il);
        b.w(&format!("br label %{}", done_l));
    }

    b.out.push_str(&format!("{}:\n", done_l));
    b.finish(&format!(
        // `C` is deliberately NOT `noalias`, matching `__y_gemm_run`. It buys
        // nothing here — the accumulators are allocas and every C store is
        // outside the K loop, so no B load can be clobbered by one — and a
        // `noalias` the caller does not honour is a silent miscompile.
        "define internal void @__y_gemm_tiny(ptr noalias {}, ptr noalias {}, \
         ptr {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, lda, ldb, ldc
    ))
}

/// `__y_gemm_small_reduce` — sum the per-thread K-band panels into C.
///
/// Each of `nthr` threads has accumulated a private `M x N` panel at
/// `base + t*SCRATCH_FLOATS`; this sums them into `C[m0..m1) x [n0..n1)`. The
/// caller partitions the columns, so every thread reduces its own band and no
/// two threads write the same cache line.
fn emit_small_reduce() -> String {
    let mut b = IrBuilder::new();
    let (base, c, n, m0, m1, n0, n1, nthr) =
        ("%base", "%C", "%N", "%m0", "%m1", "%n0", "%n1", "%nthr");
    let ldc = "%ldc";

    let span = b.sub(n1, n0);
    let span_full = b.t();
    b.w(&format!("{} = and i64 {}, -16", span_full, span));
    let n_full = b.add(n0, &span_full);
    b.entry_alloca("%sr.acc", "<16 x float>", 64);

    let il = b.loop_begin("sr.i", m0, m1, "1");
    let i = b.iv(&il);
    // Two DIFFERENT row strides, and conflating them is the whole hazard here.
    // The per-thread panels are freshly packed `M x N` blocks in scratch, so
    // their stride is `n`; C belongs to the caller and may be a submatrix, so
    // its stride is `ldc`. They coincided before leading dimensions existed.
    let roff = b.mul(&i, n);
    let croff = b.mul(&i, ldc);

    for (masked, tag) in [(false, "sr.j"), (true, "sr.t")] {
        let (start, end) = if masked {
            (n_full.clone(), n1.to_string())
        } else {
            (n0.to_string(), n_full.clone())
        };
        let jl = b.loop_begin(tag, &start, &end, "16");
        let j = b.iv(&jl);
        let mask = if masked {
            let rem = b.sub(n1, &j);
            Some(lane_mask(&mut b, &rem))
        } else {
            None
        };
        let coff = b.add(&roff, &j);
        let ccoff = b.add(&croff, &j);

        // acc = sum over t of base[t*SCRATCH_FLOATS + i*N + j]
        b.w("store <16 x float> zeroinitializer, ptr %sr.acc, align 64");
        let tl = b.loop_begin("sr.t2", "0", nthr, "1");
        let t = b.iv(&tl);
        let toff = b.mul(&t, &SCRATCH_FLOATS.to_string());
        let tbase = b.gep("float", base, &toff);
        let sp = b.gep("float", &tbase, &coff);
        let v = b.t();
        match &mask {
            Some(mk) => b.w(&format!(
                "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                 <16 x i1> {}, <16 x float> zeroinitializer)",
                v, sp, mk
            )),
            None => b.w(&format!("{} = load <16 x float>, ptr {}, align 4", v, sp)),
        }
        let cur = b.t();
        b.w(&format!(
            "{} = load <16 x float>, ptr %sr.acc, align 64",
            cur
        ));
        let sum = b.t();
        b.w(&format!(
            "{} = fadd <16 x float> {}, {}",
            sum, cur, v
        ));
        b.w(&format!("store <16 x float> {}, ptr %sr.acc, align 64", sum));
        b.loop_end(tl);

        let fin = b.t();
        b.w(&format!(
            "{} = load <16 x float>, ptr %sr.acc, align 64",
            fin
        ));
        let cq = b.gep("float", c, &ccoff);
        match &mask {
            Some(mk) => b.w(&format!(
                "call void @llvm.masked.store.v16f32.p0(<16 x float> {}, ptr {}, \
                 i32 4, <16 x i1> {})",
                fin, cq, mk
            )),
            None => b.w(&format!("store <16 x float> {}, ptr {}, align 4", fin, cq)),
        }
        b.loop_end(jl);
    }
    b.loop_end(il);

    b.finish(&format!(
        "define internal void @__y_gemm_small_reduce(ptr noalias {}, ptr {}, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {})",
        base, c, n, m0, m1, n0, n1, nthr, ldc
    ))
}

/// The blocked driver: the five-deep BLIS loop nest around the micro-kernel.
fn emit_driver() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c, m, n, k, m0, m1, n0, n1, scratch, tid, nthr, shared, k0, k1, ksplit) = (
        "%A", "%B", "%C", "%M", "%N", "%K", "%m0", "%m1", "%n0", "%n1", "%scratch",
        "%tid", "%nthr", "%shared", "%k0", "%k1", "%ksplit",
    );
    // The row strides of A, B and C. These are the ONLY thing that may be used
    // to address the caller's memory; `k` and `n` are extents and describe how
    // much of each row is live, not how far apart two rows are. Before leading
    // dimensions existed the two coincided, so the distinction is easy to lose
    // — every place below that indexes A, B or C must use these.
    let (lda, ldb_c, ldc) = ("%lda", "%ldb", "%ldc");

    // Nothing to do, and — importantly — no barrier to enter. A pool slot past
    // the active thread count must return before the cooperative section, or
    // it would be counted in a barrier it was never sized into.
    // Under the K-split `[n0, n1)` is this thread's band of the REDUCTION, not
    // of the work, and it is allowed to be empty — the thread still has a K-band
    // to accumulate and, decisively, still has to arrive at the barrier. Only
    // `m0 >= m1` marks a slot as unused there, which is how `emit_entry`
    // switches a pool slot past the active thread count off.
    let empty_m = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", empty_m, m0, m1));
    let empty_n = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", empty_n, n0, n1));
    let not_ks = b.t();
    b.w(&format!("{} = icmp eq i64 {}, 0", not_ks, ksplit));
    let empty_n2 = b.t();
    b.w(&format!("{} = and i1 {}, {}", empty_n2, empty_n, not_ks));
    let empty = b.t();
    b.w(&format!("{} = or i1 {}, {}", empty, empty_m, empty_n2));
    let done_l = b.l("g.empty");
    let go_l = b.l("g.go");
    b.w(&format!("br i1 {}, label %{}, label %{}", empty, done_l, go_l));
    b.out.push_str(&format!("{}:\n", done_l));
    b.w("ret void");
    b.out.push_str(&format!("{}:\n", go_l));

    // This thread's A panel. B is either shared with the other threads (when
    // the split is along M, so every thread wants the same packed block) or
    // private (when the split is along N, where the blocks are disjoint and
    // there is nothing to share).
    let packa = scratch;
    // The B panel starts past the LARGEST A panel the scratch is sized for,
    // not past the current tile's. `kc` is chosen at runtime below and can
    // exceed `tl().kc`, so an offset computed from the tile would put the A
    // panel's tail on top of the B panel — a silent wrong answer on exactly
    // the shapes the runtime `kc` exists to speed up. `SCRATCH_FLOATS` already
    // reserves the maximum, so this only makes the two agree.
    let privb = b.gep("float", scratch, &((MC_MAX + MR_MAX) * KC_MAX).to_string());
    let coop = b.t();
    b.w(&format!("{} = icmp ne i64 {}, 0", coop, shared));
    let packb = b.t();
    b.w(&format!(
        "{} = select i1 {}, ptr {}, ptr {}",
        packb, coop, SHARED_B, privb
    ));

    // Route a near-GEMV shape to the k-outer kernel; see `SM_MAX_M`.
    let small = b.t();
    b.w(&format!("{} = icmp sle i64 {}, {}", small, m, SM_MAX_M));
    let sm_l = b.l("g.small");
    let big_l = b.l("g.big");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        small, sm_l, big_l
    ));
    b.out.push_str(&format!("{}:\n", sm_l));
    // Two ways to run the small-M kernel, and which one is used is decided by
    // `emit_entry` (it owns the partition and the barrier width).
    //
    //   ksplit = 0  this thread owns columns [n0, n1) of the real C and sweeps
    //               all of K.
    //   ksplit = 1  this thread owns K-band [k0, k1), accumulates a private
    //               M x N panel in its scratch, and after a barrier reduces
    //               columns [n0, n1) of C across every thread's panel.
    //
    // The K-split exists because the N-split starves the memory system on this
    // kernel: with `p` outer and `j` inner, a thread given 1/16th of the
    // columns reads a 1 KB slice of every 16 KB row of B, so it touches all
    // 64 MB of B's address span to consume 4 MB of it and the prefetcher can
    // never stream. A K-band is contiguous.
    //
    // Isolated in a standalone probe (identical inner loop, only the partition
    // differing) on 8x4096x4096 at 16 threads, three runs: N-split 62/74/66
    // GFLOPS against K-split 405/427/371. **Read the ratio, not the values** —
    // B is 64 MB against this part's 64 MB L3, so the probe's absolutes move
    // ~2x between runs of an identical protocol. End to end through the real
    // harness the shape goes 128.7 -> 584.2 GFLOPS, but that figure also
    // contains the K-unroll.
    let ks = b.t();
    b.w(&format!("{} = icmp ne i64 {}, 0", ks, ksplit));
    let ksp_l = b.l("g.ksplit");
    let nsp_l = b.l("g.nsplit");
    b.w(&format!("br i1 {}, label %{}, label %{}", ks, ksp_l, nsp_l));

    b.out.push_str(&format!("{}:\n", nsp_l));
    // Writes the caller's C directly, so C's stride is the caller's `ldc`.
    b.w(&format!(
        "call void @__y_gemm_small_m(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 0, i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, ldc, m0, m1, n0, n1, k, lda, ldb_c
    ));
    b.w("ret void");

    b.out.push_str(&format!("{}:\n", ksp_l));
    // Private panel: the thread's own packing scratch, which this path does not
    // otherwise use. `emit_entry` has already checked `M*N <= SCRATCH_FLOATS`.
    // Accumulates into this thread's PRIVATE `M x N` scratch panel, which is
    // freshly packed and therefore `n`-strided. Passing the caller's `ldc`
    // here would stride a buffer that has nothing to do with C — and since
    // `ldc >= n` it would run off the end of the thread's scratch slot into
    // the next thread's. A and B are still the caller's, so they keep theirs.
    b.w(&format!(
        "call void @__y_gemm_small_m(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 0, i64 {}, i64 0, i64 {}, i64 {}, i64 {}, i64 {}, i64 {})",
        a, bb, scratch, m, n, k, n, m, n, k0, k1, lda, ldb_c
    ));
    b.w(&format!(
        "call i32 @pthread_barrier_wait(ptr {})",
        POOL_BARRIER
    ));
    let sbase = b.t();
    b.w(&format!("{} = load ptr, ptr {}, align 8", sbase, SCRATCH));
    b.w(&format!(
        "call void @__y_gemm_small_reduce(ptr {}, ptr {}, i64 {}, i64 0, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {})",
        sbase, c, n, m, n0, n1, nthr, ldc
    ));
    b.w("ret void");

    b.out.push_str(&format!("{}:\n", big_l));

    let jcl = b.loop_begin("g.jc", n0, n1, &tl().nc.to_string());
    let jc = b.iv(&jcl);
    let rn = b.sub(n1, &jc);
    let nc = b.imin(&rn, &tl().nc.to_string());

    // K-panel depth, chosen at RUNTIME from the thread count.
    //
    // This is the one blocking parameter whose optimum is not a property of
    // the machine alone, and a compile-time constant therefore cannot be right
    // for both cases:
    //
    //   - One thread wants `kc` LARGE. C is read and written once per K-panel,
    //     so C traffic scales as `K/kc`. Raising kc 256 -> 1024 measured
    //     2048^3 at 207 -> 298 GFLOPS single-threaded, +44%.
    //   - Sixteen threads want `kc` SMALL. Each thread packs a private
    //     `kc x nc` B panel, so the aggregate footprint is `nthr*kc*nc*4`
    //     bytes. At kc=1024, nc=2048 that is 8.4 MB per thread and 134 MB
    //     across 16 — against a 64 MB L3. Measured with the single-threaded
    //     optimum baked in, 1024^3 collapsed from 2983 to 851 GFLOPS (0.34x
    //     OpenBLAS). Tuning this on one thread and shipping it is a real trap.
    //
    // So budget the aggregate panel at `L3_PANEL_FLOATS` and divide:
    //     kc = clamp(L3_PANEL_FLOATS / (nthr * nc), KC_MIN, KC_MAX)
    // At nc=2048 that yields 1024 for one thread and 256 for sixteen — both of
    // which are the values the sweeps independently picked, which is the only
    // reason to trust the formula rather than a table.
    let denom = b.mul(nthr, &nc);
    let raw = b.t();
    b.w(&format!(
        "{} = udiv i64 {}, {}",
        raw, L3_PANEL_FLOATS, denom
    ));
    let capped = b.imin(&raw, &KC_MAX.to_string());
    let kc_max_eff = b.imax(&capped, &KC_MIN.to_string());

    let pcl = b.loop_begin("g.pc", "0", k, &kc_max_eff);
    let pc = b.iv(&pcl);
    let rk = b.sub(k, &pc);
    let kc = b.imin(&rk, &kc_max_eff);

    // packB(B + pc*N + jc, N, kc, nc)
    //
    // Skipping this pack when M is small was tried and is NOT a win: measured
    // at 4x4096x4096 it moved 0.29x -> 0.33x, nowhere near the 3x the traffic
    // arithmetic suggested. Packing is not overhead to be avoided here — it is
    // what turns B's access pattern linear. Reading B in place walks a
    // 32-float column panel down 256 rows at stride N, touching a new page
    // every row and revisiting each page once per panel. Small M is handled by
    // `__y_gemm_small_m` instead, which changes the loop order rather than the
    // packing.
    let bro = b.mul(&pc, ldb_c);
    let bo = b.add(&bro, &jc);
    let bsrc = b.gep("float", bb, &bo);
    // Cooperative: each thread packs every nthr'th panel of the shared block,
    // then all wait. Private: this thread packs the whole block alone.
    let pk_t = b.t();
    b.w(&format!("{} = select i1 {}, i64 {}, i64 0", pk_t, coop, tid));
    let pk_n = b.t();
    b.w(&format!("{} = select i1 {}, i64 {}, i64 1", pk_n, coop, nthr));
    b.w(&format!(
        "call void @__y_gemm_pack_b(ptr {}, i64 {}, i64 {}, i64 {}, ptr {}, i64 {}, i64 {})",
        bsrc, ldb_c, kc, nc, packb, pk_t, pk_n
    ));
    let bar1_l = b.l("g.bar1");
    let after1_l = b.l("g.after1");
    b.w(&format!("br i1 {}, label %{}, label %{}", coop, bar1_l, after1_l));
    b.out.push_str(&format!("{}:\n", bar1_l));
    b.w(&format!(
        "call i32 @pthread_barrier_wait(ptr {})",
        POOL_BARRIER
    ));
    b.w(&format!("br label %{}", after1_l));
    b.out.push_str(&format!("{}:\n", after1_l));

    let first = b.t();
    b.w(&format!("{} = icmp eq i64 {}, 0", first, pc));

    let icl = b.loop_begin("g.ic", m0, m1, &tl().mc.to_string());
    let ic = b.iv(&icl);
    let rm = b.sub(m1, &ic);
    let mc = b.imin(&rm, &tl().mc.to_string());

    // packA(A + ic*lda + pc, lda, mc, kc)
    let aro = b.mul(&ic, lda);
    let ao = b.add(&aro, &pc);
    let asrc = b.gep("float", a, &ao);
    b.w(&format!(
        "call void @__y_gemm_pack_a(ptr {}, i64 {}, i64 {}, i64 {}, ptr {})",
        asrc, lda, mc, kc, packa
    ));

    let jrl = b.loop_begin("g.jr", "0", &nc, &tl().nr.to_string());
    let jr = b.iv(&jrl);
    let rjw = b.sub(&nc, &jr);
    let jw = b.imin(&rjw, &tl().nr.to_string());

    let bpo = b.mul(&jr, &kc);
    let bpp = b.gep("float", &packb, &bpo);
    let bld = tl().nr.to_string();

    let irl = b.loop_begin("g.ir", "0", &mc, &tl().mr.to_string());
    let ir = b.iv(&irl);
    let rmw = b.sub(&mc, &ir);
    let mw = b.imin(&rmw, &tl().mr.to_string());
    let apo = b.mul(&ir, &kc);
    let app = b.gep("float", packa, &apo);

    // C + (ic+ir)*N + jc + jr
    // C column is jc + jr: `jc` is the absolute start of this NC block and
    // `jr` the offset of the micro-panel within it.
    let crow = b.add(&ic, &ir);
    let cro = b.mul(&crow, ldc);
    let cc1 = b.add(&cro, &jc);
    let cc2 = b.add(&cc1, &jr);
    let cp = b.gep("float", c, &cc2);

    b.w(&format!(
        "call void @__y_gemm_micro(ptr {}, ptr {}, i64 {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i1 {})",
        app, bpp, bld, cp, ldc, kc, mw, jw, first
    ));

    b.loop_end(irl);
    b.loop_end(jrl);
    b.loop_end(icl);

    // Second barrier: the next pc iteration overwrites the shared panel, so no
    // thread may start repacking until every thread has finished reading it.
    let bar2_l = b.l("g.bar2");
    let after2_l = b.l("g.after2");
    b.w(&format!("br i1 {}, label %{}, label %{}", coop, bar2_l, after2_l));
    b.out.push_str(&format!("{}:\n", bar2_l));
    b.w(&format!(
        "call i32 @pthread_barrier_wait(ptr {})",
        POOL_BARRIER
    ));
    b.w(&format!("br label %{}", after2_l));
    b.out.push_str(&format!("{}:\n", after2_l));

    b.loop_end(pcl);
    b.loop_end(jcl);

    b.finish(&format!(
        "define internal void @__y_gemm_run(ptr noalias {}, ptr noalias {}, ptr {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, ptr noalias {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, m0, m1, n0, n1, scratch, tid, nthr, shared, k0, k1, ksplit,
        lda, ldb_c, ldc
    ))
}

/// `__y_gemm_worker(ptr task) -> ptr` — pthread entry. Unpacks the task
/// record and runs the driver over this thread's slice.
fn emit_worker() -> String {
    let mut b = IrBuilder::new();
    let t = "%t";

    // %y_gemm_task = { A, B, C, M, N, K, m0, m1, n0, n1, scratch, tid, nthr,
    //                  shared, k0, k1, ksplit, lda, ldb, ldc }
    let names = [
        ("a", "ptr", 0),
        ("bb", "ptr", 1),
        ("c", "ptr", 2),
        ("m", "i64", 3),
        ("n", "i64", 4),
        ("k", "i64", 5),
        ("m0", "i64", 6),
        ("m1", "i64", 7),
        ("n0", "i64", 8),
        ("n1", "i64", 9),
        ("scr", "ptr", 10),
        ("tid", "i64", 11),
        ("nthr", "i64", 12),
        ("shared", "i64", 13),
        ("k0", "i64", 14),
        ("k1", "i64", 15),
        ("ksplit", "i64", 16),
        ("lda", "i64", 17),
        ("ldb", "i64", 18),
        ("ldc", "i64", 19),
    ];
    let mut vals = Vec::new();
    for (_, ty, idx) in names {
        let p = b.t();
        b.w(&format!(
            "{} = getelementptr inbounds %y_gemm_task, ptr {}, i64 0, i32 {}",
            p, t, idx
        ));
        let v = b.t();
        b.w(&format!("{} = load {}, ptr {}, align 8", v, ty, p));
        vals.push(v);
    }

    b.w(&format!(
        "call void @__y_gemm_run(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, ptr {}, i64 {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {})",
        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8],
        vals[9], vals[10], vals[11], vals[12], vals[13], vals[14], vals[15], vals[16],
        vals[17], vals[18], vals[19]
    ));
    b.w("ret ptr null");

    let mut s = b.finish(&format!(
        "define internal ptr @__y_gemm_worker(ptr {})",
        t
    ));
    // `finish` appends `ret void`, which is wrong for a ptr-returning function.
    s = s.replace("  ret void\n}", "}");
    s
}

/// `__y_gemm_threads(M, N, K) -> i64` — how many threads to use.
///
/// This is OpenBLAS's shape, including its constants: refuse to thread below
/// `SMP_THRESHOLD`, then scale linearly with the work until the core count is
/// reached. `Y_NUM_THREADS` overrides the ceiling, and is resolved once and
/// cached because `getenv` on every GEMM would show up on the small shapes
/// this function exists to protect.
fn emit_threads() -> String {
    let mut b = IrBuilder::new();
    let (m, n, k) = ("%M", "%N", "%K");

    // Resolve the ceiling once.
    let cached = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", cached, NTHREADS_CACHE));
    let need = b.t();
    b.w(&format!("{} = icmp eq i64 {}, 0", need, cached));
    let res_l = b.l("nt.resolve");
    let have_l = b.l("nt.have");
    b.w(&format!("br i1 {}, label %{}, label %{}", need, res_l, have_l));

    b.out.push_str(&format!("{}:\n", res_l));
    let env = b.t();
    b.w(&format!("{} = call ptr @getenv(ptr @.y_gemm_env)", env));
    let has_env = b.t();
    b.w(&format!("{} = icmp ne ptr {}, null", has_env, env));
    let env_l = b.l("nt.env");
    let sys_l = b.l("nt.sys");
    let set_l = b.l("nt.set");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        has_env, env_l, sys_l
    ));

    b.out.push_str(&format!("{}:\n", env_l));
    let ev32 = b.t();
    b.w(&format!("{} = call i32 @atoi(ptr {})", ev32, env));
    let ev = b.t();
    b.w(&format!("{} = sext i32 {} to i64", ev, ev32));
    b.w(&format!("br label %{}", set_l));

    b.out.push_str(&format!("{}:\n", sys_l));
    // _SC_NPROCESSORS_ONLN = 84 on Linux.
    let sc = b.t();
    b.w(&format!("{} = call i64 @sysconf(i32 84)", sc));
    b.w(&format!("br label %{}", set_l));

    b.out.push_str(&format!("{}:\n", set_l));
    let raw = b.t();
    b.w(&format!(
        "{} = phi i64 [ {}, %{} ], [ {}, %{} ]",
        raw, ev, env_l, sc, sys_l
    ));
    let atleast1 = b.imax(&raw, "1");
    let capped = b.imin(&atleast1, &MAX_THREADS.to_string());
    b.w(&format!(
        "store i64 {}, ptr {}, align 8",
        capped, NTHREADS_CACHE
    ));
    b.w(&format!("br label %{}", have_l));

    b.out.push_str(&format!("{}:\n", have_l));
    let ceiling = b.t();
    b.w(&format!(
        "{} = load i64, ptr {}, align 8",
        ceiling, NTHREADS_CACHE
    ));

    // work = M*N*K, saturating: the product overflows i64 only at sizes that
    // cannot be allocated, but clamping keeps the comparison meaningful.
    let mn = b.mul(m, n);
    let work = b.mul(&mn, k);
    // Threading is only nearly free while the workers are still spinning; once
    // they park, a dispatch costs ~20us of futex wake and scheduling, flat in
    // the thread count. So a shape below `COLD_MIN_WORK` is threaded only when
    // this caller is issuing GEMMs often enough to keep the pool warm.
    //
    // **The signal is call FREQUENCY, not the pool's parked state.** Counting
    // parked workers was tried first and it latches: a small shape reads the
    // pool as cold, drops to one thread, which means the workers are never
    // woken, so it reads cold forever. It measured 0.21x on `128^3` in the
    // throughput harness for exactly that reason. Elapsed time since the
    // previous call cannot latch — a caller in a tight loop looks hot whether
    // or not the last call used the pool, so the first threaded call pays the
    // wake once and every one after it is warm.
    //
    // `clock_gettime` is a vDSO call, ~25ns, and the tiny path returns before
    // reaching here so it never pays it. This is a heuristic; a wrong answer
    // costs a suboptimal thread count for one call, never correctness.
    b.entry_alloca("%.ts", "{ i64, i64 }", 8);
    b.w("call i32 @clock_gettime(i32 1, ptr %.ts)");
    let tsec = b.t();
    b.w(&format!(
        "{} = load i64, ptr %.ts, align 8",
        tsec
    ));
    let nsp = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds {{ i64, i64 }}, ptr %.ts, i64 0, i32 1",
        nsp
    ));
    let tns = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", tns, nsp));
    let sec_ns = b.mul(&tsec, "1000000000");
    let now = b.add(&sec_ns, &tns);
    let last = b.t();
    b.w(&format!(
        "{} = load atomic i64, ptr {} monotonic, align 8",
        last, POOL_LAST_NS
    ));
    b.w(&format!(
        "store atomic i64 {}, ptr {} monotonic, align 8",
        now, POOL_LAST_NS
    ));
    let since = b.sub(&now, &last);
    let cold = b.t();
    b.w(&format!("{} = icmp sgt i64 {}, {}", cold, since, HOT_WINDOW_NS));
    let floor = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        floor, cold, COLD_MIN_WORK, WORK_PER_THREAD
    ));
    let big = b.t();
    b.w(&format!("{} = icmp sgt i64 {}, {}", big, work, floor));
    let scale_l = b.l("nt.scale");
    let one_l = b.l("nt.one");
    let out_l = b.l("nt.out");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        big, scale_l, one_l
    ));

    b.out.push_str(&format!("{}:\n", scale_l));
    let by_work = b.t();
    b.w(&format!(
        "{} = sdiv i64 {}, {}",
        by_work, work, WORK_PER_THREAD
    ));
    let scaled = b.imin(&by_work, &ceiling);
    let scaled = b.imax(&scaled, "1");
    b.w(&format!("br label %{}", out_l));

    b.out.push_str(&format!("{}:\n", one_l));
    b.w(&format!("br label %{}", out_l));

    b.out.push_str(&format!("{}:\n", out_l));
    let r = b.t();
    b.w(&format!(
        "{} = phi i64 [ {}, %{} ], [ 1, %{} ]",
        r, scaled, scale_l, one_l
    ));
    b.w(&format!("ret i64 {}", r));

    let mut s = b.finish("define internal i64 @__y_gemm_threads(i64 %M, i64 %N, i64 %K)");
    s = s.replace("  ret void\n}", "}");
    s
}

/// `__y_pool_worker(ptr idx)` — a persistent worker. Waits for the dispatch
/// generation to change, runs its slice, publishes completion, waits again.
///
/// This replaces a `pthread_create`/`pthread_join` pair per GEMM. That cost
/// ~25 us per thread, which at 16 threads is ~400 us of fork/join against a
/// 256^3 GEMM that takes 135 us single-threaded — so threading made the small
/// shapes several times *slower* than not threading. OpenBLAS has had a
/// persistent thread server (`driver/others/blas_server.c`) for exactly this
/// reason; dispatch here is one release store and a spin on an acquire load.
///
/// Idle workers spin briefly and then park in `usleep`, so a pool that is not
/// in use costs no CPU and does not perturb anything measured next to it.
fn emit_pool_worker() -> String {
    let mut b = IrBuilder::new();
    let idxp = "%idxp";

    let my = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", my, idxp));
    let donep = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 {}",
        donep, MAX_THREADS, POOL_DONE, my
    ));
    let taskp = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x %y_gemm_task], ptr {}, i64 0, i64 {}",
        taskp, MAX_THREADS, POOL_TASK, my
    ));

    b.entry_alloca("%.last", "i64", 8);
    b.entry_alloca("%.spin", "i64", 8);
    b.entry_alloca("%.gen", "i64", 8);
    b.w("store i64 0, ptr %.last, align 8");

    let outer = b.l("w.outer");
    let wait = b.l("w.wait");
    let work = b.l("w.work");
    let park = b.l("w.park");
    b.w(&format!("br label %{}", outer));

    b.out.push_str(&format!("{}:\n", outer));
    b.w("store i64 0, ptr %.spin, align 8");
    b.w(&format!("br label %{}", wait));

    b.out.push_str(&format!("{}:\n", wait));
    let g = b.t();
    b.w(&format!(
        "{} = load atomic i64, ptr {} acquire, align 8",
        g, POOL_GEN
    ));
    let last = b.t();
    b.w(&format!("{} = load i64, ptr %.last, align 8", last));
    let changed = b.t();
    b.w(&format!("{} = icmp ne i64 {}, {}", changed, g, last));
    b.w(&format!("store i64 {}, ptr %.gen, align 8", g));
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        changed, work, park
    ));

    b.out.push_str(&format!("{}:\n", park));
    let sp = b.t();
    b.w(&format!("{} = load i64, ptr %.spin, align 8", sp));
    let sp1 = b.add(&sp, "1");
    b.w(&format!("store i64 {}, ptr %.spin, align 8", sp1));
    let hot = b.t();
    b.w(&format!("{} = icmp slt i64 {}, {}", hot, sp1, POOL_SPIN));
    let spin_l = b.l("w.spin");
    let block_l = b.l("w.block");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        hot, spin_l, block_l
    ));
    b.out.push_str(&format!("{}:\n", spin_l));
    b.w("call void @llvm.x86.sse2.pause()");
    b.w(&format!("br label %{}", wait));

    // Blocked: hold the mutex, re-check under it, and wait. An idle pool costs
    // nothing here — no wakeups, no cache traffic, no stolen issue slots.
    b.out.push_str(&format!("{}:\n", block_l));
    b.w(&format!("call i32 @pthread_mutex_lock(ptr {})", POOL_MUTEX));
    let recheck = b.l("w.recheck");
    b.w(&format!("br label %{}", recheck));
    b.out.push_str(&format!("{}:\n", recheck));
    let g2 = b.t();
    b.w(&format!(
        "{} = load atomic i64, ptr {} acquire, align 8",
        g2, POOL_GEN
    ));
    let last2 = b.t();
    b.w(&format!("{} = load i64, ptr %.last, align 8", last2));
    let ready = b.t();
    b.w(&format!("{} = icmp ne i64 {}, {}", ready, g2, last2));
    let wake_l = b.l("w.wake");
    let sleep_l = b.l("w.sleep");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        ready, wake_l, sleep_l
    ));
    b.out.push_str(&format!("{}:\n", sleep_l));
    // Spurious wakeups are why this re-checks in a loop rather than once.
    b.w(&format!(
        "call i32 @pthread_cond_wait(ptr {}, ptr {})",
        POOL_COND, POOL_MUTEX
    ));
    b.w(&format!("br label %{}", recheck));
    b.out.push_str(&format!("{}:\n", wake_l));
    b.w(&format!("call i32 @pthread_mutex_unlock(ptr {})", POOL_MUTEX));
    b.w(&format!("store i64 {}, ptr %.gen, align 8", g2));
    b.w(&format!("br label %{}", work));

    b.out.push_str(&format!("{}:\n", work));
    let gw = b.t();
    b.w(&format!("{} = load i64, ptr %.gen, align 8", gw));
    b.w(&format!("store i64 {}, ptr %.last, align 8", gw));
    b.w(&format!("call void @__y_gemm_worker(ptr {})", taskp));
    // Publish the generation this iteration actually ran, not the one the
    // spin path happened to read: a worker that reached here after blocking
    // read its stale pre-block value there, so publishing that left `done[]`
    // one generation behind and the dispatcher waiting forever.
    b.w(&format!(
        "store atomic i64 {}, ptr {} release, align 8",
        gw, donep
    ));
    b.w(&format!("br label %{}", outer));

    let mut s = b.finish(&format!(
        "define internal ptr @__y_pool_worker(ptr {})",
        idxp
    ));
    s = s.replace("  ret void\n}", "}");
    s
}

/// The public entry: pick a thread count, size the scratch, split the work,
/// dispatch to the pool, run one slice inline, wait.
///
/// The split is 1-D and adaptive. OpenBLAS partitions in both M and N with a
/// `switch_ratio` guard so neither dimension is cut below a useful width; the
/// same concern applies here with only one axis to choose, so the axis is
/// picked by which one has enough width to give every thread a whole number of
/// micro-tiles. Cutting M when M is small would hand threads empty slices —
/// which is exactly the skinny/decode family.
fn emit_entry() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c, m, n, k) = ("%A", "%B", "%C", "%M", "%N", "%K");
    // Every dispatch decision below (tiny/small-M/threaded, the partition, the
    // thread count) is made from the EXTENTS. A leading dimension changes only
    // how memory is addressed, never how much arithmetic there is, so it must
    // not enter any of those tests — a submatrix of a 4096-wide buffer is the
    // same amount of work as a packed one.
    let (lda, ldb_c, ldc) = ("%lda", "%ldb", "%ldc");

    // The copy-free path, before anything else looks at this shape.
    //
    // Deliberately ahead of the thread count, the pool and the partition: a
    // shape this small is single-threaded anyway (`WORK_PER_THREAD` alone
    // settles that), and putting the test here means the tiny path cannot
    // interact with the barrier or the grid at all. `M > SM_MAX_M` keeps it
    // clear of `__y_gemm_small_m`, which is already copy-free and is tuned for
    // the very different `M <= 8` shape.
    let mn0 = b.mul(m, n);
    let work0 = b.mul(&mn0, k);
    let tw = b.t();
    b.w(&format!(
        "{} = icmp sle i64 {}, {}",
        tw, work0, TINY_MAX_WORK
    ));
    let tn = b.t();
    b.w(&format!("{} = icmp sle i64 {}, {}", tn, n, TINY_MAX_N));
    let tm = b.t();
    b.w(&format!("{} = icmp sgt i64 {}, {}", tm, m, SM_MAX_M));
    let t0 = b.t();
    b.w(&format!("{} = and i1 {}, {}", t0, tw, tn));
    let is_tiny = b.t();
    b.w(&format!("{} = and i1 {}, {}", is_tiny, t0, tm));
    let tiny_l = b.l("e.tiny");
    let notiny_l = b.l("e.notiny");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        is_tiny, tiny_l, notiny_l
    ));
    b.out.push_str(&format!("{}:\n", tiny_l));
    b.w(&format!(
        "call void @__y_gemm_tiny(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, lda, ldb_c, ldc
    ));
    b.w("ret void");
    b.out.push_str(&format!("{}:\n", notiny_l));

    let nt = b.t();
    b.w(&format!(
        "{} = call i64 @__y_gemm_threads(i64 {}, i64 {}, i64 {})",
        nt, m, n, k
    ));

    // Single-threaded shapes skip the pool entirely: no dispatch, no scratch
    // growth, no generation bump.
    let solo = b.t();
    b.w(&format!("{} = icmp sle i64 {}, 1", solo, nt));
    let solo_l = b.l("e.solo");
    let par_l = b.l("e.par");
    b.w(&format!("br i1 {}, label %{}, label %{}", solo, solo_l, par_l));

    b.out.push_str(&format!("{}:\n", solo_l));
    let s0 = b.t();
    b.w(&format!("{} = load ptr, ptr {}, align 8", s0, SCRATCH));
    let has0 = b.t();
    b.w(&format!("{} = icmp ne ptr {}, null", has0, s0));
    let scr0 = b.t();
    b.w(&format!(
        "{} = select i1 {}, ptr {}, ptr {}",
        scr0, has0, s0, FALLBACK
    ));
    b.w(&format!(
        "call void @__y_gemm_run(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 0, i64 {}, i64 0, i64 {}, ptr {}, i64 0, i64 1, i64 0, i64 0, i64 {}, i64 0, \
         i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, m, n, scr0, k, lda, ldb_c, ldc
    ));
    b.w("ret void");

    b.out.push_str(&format!("{}:\n", par_l));

    // Start the pool once, sized to the cap, and give it scratch to match.
    let started = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", started, POOL_N));
    let cold = b.t();
    b.w(&format!("{} = icmp eq i64 {}, 0", cold, started));
    let start_l = b.l("e.start");
    let hot_l = b.l("e.hot");
    b.w(&format!("br i1 {}, label %{}, label %{}", cold, start_l, hot_l));

    b.out.push_str(&format!("{}:\n", start_l));
    let cap = b.t();
    b.w(&format!(
        "{} = load i64, ptr {}, align 8",
        cap, NTHREADS_CACHE
    ));
    let bytes = b.mul(&cap, &(SCRATCH_FLOATS * 4).to_string());
    let fresh = b.t();
    b.w(&format!("{} = call ptr @malloc(i64 {})", fresh, bytes));
    b.w(&format!("store ptr {}, ptr {}, align 8", fresh, SCRATCH));
    let ok = b.t();
    b.w(&format!("{} = icmp ne ptr {}, null", ok, fresh));
    let oom_l = b.l("e.oom");
    let spawn_l = b.l("e.spawn0");
    b.w(&format!("br i1 {}, label %{}, label %{}", ok, spawn_l, oom_l));

    b.out.push_str(&format!("{}:\n", oom_l));
    // Fall back to one thread on the statically reserved panel. This must not
    // be an `alloca`: an entry-block alloca of this size is emitted on every
    // call, not just on the path that uses it, and a 2.3 MB stack frame per
    // GEMM costs more than the whole kernel at small sizes.
    b.w(&format!(
        "call void @__y_gemm_run(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 0, i64 {}, i64 0, i64 {}, ptr {}, i64 0, i64 1, i64 0, i64 0, i64 {}, i64 0, \
         i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, m, n, FALLBACK, k, lda, ldb_c, ldc
    ));
    b.w("ret void");

    b.out.push_str(&format!("{}:\n", spawn_l));
    b.w(&format!(
        "call i32 @pthread_mutex_init(ptr {}, ptr null)",
        POOL_MUTEX
    ));
    b.w(&format!(
        "call i32 @pthread_cond_init(ptr {}, ptr null)",
        POOL_COND
    ));
    b.entry_alloca("%.tid", "i64", 8);
    let sl = b.loop_begin("e.spawn", "1", &cap, "1");
    let si = b.iv(&sl);
    let idp = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 {}",
        idp, MAX_THREADS, POOL_IDS, si
    ));
    b.w(&format!("store i64 {}, ptr {}, align 8", si, idp));
    b.w(&format!(
        "call i32 @pthread_create(ptr %.tid, ptr null, ptr @__y_pool_worker, ptr {})",
        idp
    ));
    b.loop_end(sl);
    b.w(&format!("store i64 {}, ptr {}, align 8", cap, POOL_N));
    b.w(&format!("br label %{}", hot_l));

    b.out.push_str(&format!("{}:\n", hot_l));
    let pool_n = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", pool_n, POOL_N));
    let scratch = b.t();
    b.w(&format!("{} = load ptr, ptr {}, align 8", scratch, SCRATCH));

    // Partition C over a 2-D `ntm x ntn` grid of threads, and cap the thread
    // count by the number of whole micro-tiles the two extents can supply.
    //
    // The rule before this one cut exactly ONE axis, choosing whichever yielded
    // more whole micro-tiles. That caps a shape small in both extents at
    // `max(M/MR, N/NR)` threads, and — more expensively on the large square
    // shapes — it makes the packing redundancy maximal. Under a 16-way M-split
    // every thread packs the SAME full-width B, so B is packed 16 times and 16
    // copies of the packed panel are live in L3 at once; under a 16-way N-split
    // the same is true of A. A 4x4 grid packs each operand 4 times instead.
    //
    // This is exactly what OpenBLAS does (`driver/level3/level3_thread.c`,
    // `nthreads_m x nthreads_n` with the `switch_ratio` guard), including the
    // objective function used to pick the factorisation: minimise
    // `N*ntm + M*ntn`, which is the sum of the two per-thread block edges and
    // so prefers square blocks. At 2048^3 on 16 threads their rule and this one
    // both choose 4x4.
    //
    // `pm`/`pn` are whole micro-tiles, so a factor can never be handed a band
    // narrower than one micro-panel.
    let pm = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", pm, m, tl().mr));
    let pm = b.imax(&pm, "1");
    let pn = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", pn, n, tl().nr));
    let pn = b.imax(&pn, "1");
    // The 1-D cap was `max(pm, pn)`; a grid can use the product.
    let par_mn = b.mul(&pm, &pn);

    // Cut K instead, for the shapes that go to `__y_gemm_small_m`.
    //
    // Neither M nor N is the right axis there. M is `SM_MAX_M` or less, so it
    // cannot be cut at all; and cutting N leaves each thread reading a
    // `N/nthr`-wide slice of every row of B — 1 KB out of every 16 KB at
    // 8x4096x4096 on 16 threads, so the thread walks B's whole 64 MB address
    // span to consume 4 MB of it and no prefetcher can follow. A K-band is
    // contiguous: thread `t` reads rows `[t*K/nt, (t+1)*K/nt)` end to end.
    //
    // The price is a private `M x N` accumulator per thread plus a reduction,
    // because every thread now contributes a partial sum to every element of C.
    // At 8x4096x4096 that is 128 KB per thread and a 2 MB reduction against
    // B's 64 MB — 3% of the traffic, for a partition that measured 221 GFLOPS
    // where the N-split measured 77.
    //
    // Three conditions, all necessary:
    //   - the shape actually routes to the small-M kernel (`M <= SM_MAX_M`),
    //   - the panel fits the scratch already allocated per thread,
    //   - K is deep enough that a band is worth more than the reduction.
    let is_small = b.t();
    b.w(&format!("{} = icmp sle i64 {}, {}", is_small, m, SM_MAX_M));
    let panel = b.mul(m, n);
    let fits = b.t();
    b.w(&format!(
        "{} = icmp sle i64 {}, {}",
        fits, panel, SCRATCH_FLOATS
    ));
    let deep = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", deep, k, KSPLIT_MIN_K));
    let ks1 = b.t();
    b.w(&format!("{} = and i1 {}, {}", ks1, is_small, fits));
    let cut_k = b.t();
    b.w(&format!("{} = and i1 {}, {}", cut_k, ks1, deep));
    let ksplit = b.t();
    b.w(&format!("{} = zext i1 {} to i64", ksplit, cut_k));

    // A K-band shorter than this leaves the reduction dominating, so cap the
    // thread count by it rather than handing out slivers.
    let pk = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", pk, k, KSPLIT_MIN_BAND));
    let pk = b.imax(&pk, "1");
    let par = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        par, cut_k, pk, par_mn
    ));
    let nt = b.imin(&nt, &par);

    // Factorise the thread count into the grid `ntm x ntn`.
    //
    // `ntm` is searched over EVERY value up to `MAX_THREADS`, not over the
    // powers of two. Powers of two were tried first and are wrong for a reason
    // worth keeping: with `nt = 3` the only reachable factorisations are 1x3
    // and 2x1, so the M-split `3x1` cannot be expressed and the search picks a
    // pure N-split. `ragged 250^3` gets `nt = 3`, and that is exactly what
    // happened to it — **0.65x**, at a 5.4% spread, on the shape the 1-D rule
    // had been cutting along M all along. Sixteen candidates is a few hundred
    // straight-line instructions once per GEMM call; the shape that pays it
    // takes tens of microseconds.
    //
    // `ntn` is `nt / ntm`, clipped to the whole micro-tiles N can supply.
    //
    // Two scoring keys, in this order:
    //   1. threads actually used. Never leave a core idle to get a prettier
    //      block; `nt` is already the work-derived count and giving some of it
    //      back is a straight loss. This is also why `ntm` need not divide
    //      `nt` — a candidate that wastes a thread simply loses on this key.
    //   2. OpenBLAS's objective `N*ntm + M*ntn`. Each thread reads `M/ntm` rows
    //      of A and `N/ntn` columns of B, so the total operand traffic is
    //      `K*(N*ntm + M*ntn)` and this key minimises it directly.
    //
    // Ties go to the LARGER `ntm`, which is what the `<=` on the cost does.
    // M and N are symmetric in the objective but not in the code: M bands snap
    // to `MR = 12` and N bands to `NR = 32`, so an M cut balances nearly three
    // times finer and a tie should be spent there.
    let mut best_m = "1".to_string();
    let mut best_n = "1".to_string();
    let mut best_used = "0".to_string();
    let mut best_cost = i64::MAX.to_string();
    for cand in 1..=MAX_THREADS {
        let cs = cand.to_string();
        // `cand = 1` is always valid (`nt >= 2` on this path and `pm >= 1`), so
        // the running best is guaranteed to be initialised by the first
        // iteration and the `best_used = 0` seed can never survive.
        let fit_nt = b.t();
        b.w(&format!("{} = icmp sle i64 {}, {}", fit_nt, cs, nt));
        let fit_m = b.t();
        b.w(&format!("{} = icmp sle i64 {}, {}", fit_m, cs, pm));
        let valid = b.t();
        b.w(&format!("{} = and i1 {}, {}", valid, fit_nt, fit_m));
        let raw_n = b.t();
        b.w(&format!("{} = sdiv i64 {}, {}", raw_n, nt, cs));
        let cn = b.imin(&raw_n, &pn);
        let cn = b.imax(&cn, "1");
        let used = b.mul(&cs, &cn);
        let cost_m = b.mul(n, &cs);
        let cost_n = b.mul(m, &cn);
        let cost = b.add(&cost_m, &cost_n);
        let more = b.t();
        b.w(&format!("{} = icmp sgt i64 {}, {}", more, used, best_used));
        let same = b.t();
        b.w(&format!("{} = icmp eq i64 {}, {}", same, used, best_used));
        let cheaper = b.t();
        b.w(&format!("{} = icmp sle i64 {}, {}", cheaper, cost, best_cost));
        let tie = b.t();
        b.w(&format!("{} = and i1 {}, {}", tie, same, cheaper));
        let win0 = b.t();
        b.w(&format!("{} = or i1 {}, {}", win0, more, tie));
        let win = b.t();
        b.w(&format!("{} = and i1 {}, {}", win, win0, valid));
        let nm = b.t();
        b.w(&format!("{} = select i1 {}, i64 {}, i64 {}", nm, win, cs, best_m));
        let nn = b.t();
        b.w(&format!("{} = select i1 {}, i64 {}, i64 {}", nn, win, cn, best_n));
        let nu = b.t();
        b.w(&format!(
            "{} = select i1 {}, i64 {}, i64 {}",
            nu, win, used, best_used
        ));
        let ncst = b.t();
        b.w(&format!(
            "{} = select i1 {}, i64 {}, i64 {}",
            ncst, win, cost, best_cost
        ));
        best_m = nm;
        best_n = nn;
        best_used = nu;
        best_cost = ncst;
    }

    // The K-split OVERRIDES the grid, and every consumer has to see that.
    //
    // It is expressed AS a grid rather than beside one: the K-split is the
    // `1 x nt` case, whose N bands are the bands of the *reduction* and whose M
    // band is all of M. Folding it in this way is deliberate — the previous
    // shape of this code carried a separate `cut_m` boolean that had to be
    // corrected by `!cut_k` at each of its five consumers, and missing one of
    // them deadlocked. `cut_m` was `pm >= pn`, and at `N = 33` with `nr = 64`
    // both sides are 1, so it came out *true* for a shape whose M is 8; the
    // task fill then assigned the reduction's COLUMN bounds to the M range,
    // thread 0's band came out empty, it took the driver's early return, never
    // reached a barrier sized for three threads, and the dispatcher spun
    // forever. Every benchmark shape has `N >= nr` and hid it; it is
    // `tests/cpu_gemm_threaded.rs` at N=33 and N=15 that catches it.
    //
    // With one grid there is nothing left to subtract: the bands, the barrier
    // width, the `inuse` cutoff and the shared-B gate are all derived from
    // `ntm`/`ntn` and cannot disagree about which mode is in force.
    let not_ck = b.t();
    b.w(&format!("{} = xor i1 {}, true", not_ck, cut_k));
    let mn2 = b.mul(m, n);
    let work2 = b.mul(&mn2, k);
    let big_enough = b.t();
    b.w(&format!(
        "{} = icmp sge i64 {}, {}",
        big_enough, work2, SHARE_B_WORK
    ));

    // Shared packed B collapses the grid to ONE column, and is preferred over
    // the grid where it applies.
    //
    // These are two ways to cut the same redundancy and they do not compose: a
    // thread can share a packed B panel only with threads that walk the same
    // `(jc, pc)` blocks, which is the `ntn == 1` column. Measured at 2048^3 on
    // 16 threads, four interleaved launches, best of each — and note the two
    // unshared arms are the honest comparison of the partitions:
    //
    // | arm                     | GFLOPS | vs 1-D+share |
    // |-------------------------|--------|--------------|
    // | 1-D M-split, shared B   | 3170   | 1.00         |
    // | 1-D M-split, private B  | 2319   | 0.73         |
    // | 4x4 grid, private B     | 2580   | 0.81         |
    //
    // So the grid beats the 1-D split on its own terms (2580 vs 2319, 1.11x)
    // and sharing beats the grid (1.37x over the same 1-D split). Below
    // `SHARE_B_WORK` sharing is a measured loss and the grid runs; above it the
    // grid stands down. `SHARE_B_WORK` is therefore no longer "share or don't"
    // but "share or grid", which is why it had to be re-measured here rather
    // than carried over.
    //
    // `pm >= nt` is required, not cosmetic: one column means `ntm = nt`, and an
    // M band per thread only exists if M supplies `nt` whole micro-panels.
    // Without it a thread gets an empty band, returns before the cooperative
    // section, and never reaches a barrier sized to include it.
    let m_fits = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", m_fits, pm, nt));
    let share0 = b.t();
    b.w(&format!("{} = and i1 {}, {}", share0, big_enough, m_fits));
    let share_mode = b.t();
    b.w(&format!("{} = and i1 {}, {}", share_mode, share0, not_ck));

    let gm = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        gm, share_mode, nt, best_m
    ));
    let gn = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 1, i64 {}",
        gn, share_mode, best_n
    ));
    let ntm = b.t();
    b.w(&format!("{} = select i1 {}, i64 1, i64 {}", ntm, cut_k, gm));
    let ntn = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        ntn, cut_k, nt, gn
    ));
    // The participating thread count is the grid, not `nt`: the factorisation
    // may use fewer than `nt` threads, and the barrier must be sized for the
    // threads that actually arrive at it.
    let nt = b.mul(&ntm, &ntn);

    // The reduction's column band is snapped to a whole vector; an ordinary
    // N band is snapped to a whole micro-panel. The K-band itself has no
    // alignment to respect and the M band always uses `mr`.
    let gran_m = tl().mr.to_string();
    let gran_n = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 16, i64 {}",
        gran_n, cut_k, tl().nr
    ));

    // Threads sharing one packed B panel must be exactly the threads the
    // barrier is sized for, and must all want the SAME packed B. `share_mode`
    // is what SELECTED the single-column grid above, so it is also exactly the
    // condition under which sharing is legal — the two cannot drift apart the
    // way a separately-computed `cut_m` did.
    let do_share = share_mode.clone();
    let share = b.t();
    b.w(&format!("{} = zext i1 {} to i64", share, do_share));
    let resize_l = b.l("e.bar");
    let barok_l = b.l("e.barok");
    let bn = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", bn, BARRIER_N));
    let stale = b.t();
    b.w(&format!("{} = icmp ne i64 {}, {}", stale, bn, nt));
    // Both cooperative modes rendezvous on this barrier: the shared packed-B
    // path twice per K-block, and the K-split once between accumulating the
    // private panels and reducing them. Sizing it for one and running the other
    // hangs the dispatcher, so the two conditions have to be OR'd here.
    let wants_bar = b.t();
    b.w(&format!("{} = or i1 {}, {}", wants_bar, do_share, cut_k));
    let need_bar = b.t();
    b.w(&format!("{} = and i1 {}, {}", need_bar, stale, wants_bar));
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        need_bar, resize_l, barok_l
    ));
    b.out.push_str(&format!("{}:\n", resize_l));
    let had = b.t();
    b.w(&format!("{} = icmp ne i64 {}, 0", had, bn));
    let destroy_l = b.l("e.bardestroy");
    let init_l = b.l("e.barinit");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        had, destroy_l, init_l
    ));
    b.out.push_str(&format!("{}:\n", destroy_l));
    b.w(&format!(
        "call i32 @pthread_barrier_destroy(ptr {})",
        POOL_BARRIER
    ));
    b.w(&format!("br label %{}", init_l));
    b.out.push_str(&format!("{}:\n", init_l));
    let nt32 = b.t();
    b.w(&format!("{} = trunc i64 {} to i32", nt32, nt));
    b.w(&format!(
        "call i32 @pthread_barrier_init(ptr {}, ptr null, i32 {})",
        POOL_BARRIER, nt32
    ));
    b.w(&format!("store i64 {}, ptr {}, align 8", nt, BARRIER_N));
    b.w(&format!("br label %{}", barok_l));
    b.out.push_str(&format!("{}:\n", barok_l));

    // Fill every pool slot, not just the first `nt`: all workers wake on the
    // generation bump, so a stale record would have one recomputing an old
    // slice on top of this call's output. Slots past `nt` get an empty range.
    let fl = b.loop_begin("e.fill", "0", &pool_n, "1");
    let ti = b.iv(&fl);
    // This slot's position in the grid. Row-major, so the `ntn` threads sharing
    // a row of the grid are consecutive ids.
    let gi = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", gi, ti, ntn));
    let gi_n0 = b.mul(&gi, &ntn);
    let gj = b.sub(&ti, &gi_n0);

    // Partition the GRANULE COUNT, not the extent.
    //
    // The obvious form — cut at `idx*extent/count` and floor to the tile
    // granularity — snaps each band's *position*, which dumps the whole
    // accumulated rounding slack onto one band. In 1-D that is at most one
    // granule out of a full-width band and it never mattered. In 2-D it is one
    // granule out of a band `count` times narrower, and the errors on the two
    // axes MULTIPLY. Measured at 16 threads on a 4x4 grid, busiest thread over
    // idlest:
    //
    // | shape  | N bands (gran 32)     | M bands (gran 12)   | imbalance |
    // |--------|-----------------------|---------------------|-----------|
    // | 1024^3 | 256, 256, 256, 256    | 252, 252, 252, 268  | 1.06x     |
    // | 1000^3 | 224, 256, 256, 264    | 240, 252, 252, 256  | 1.26x     |
    // | 1021^3 | 224, 256, 256, 285    | 252, 252, 252, 265  | 1.34x     |
    //
    // and that is the whole reason 1024^3 measured 1.39x on this grid while its
    // two ragged neighbours measured 0.82x and 0.79x. A 2-D partition is simply
    // far more sensitive to this than a 1-D one; the rule was not wrong before,
    // it was merely never stressed.
    //
    // Counting granules instead spreads the remainder: `G = ceil(extent/gran)`
    // and band `idx` takes granules `[idx*G/count, (idx+1)*G/count)`, so the
    // widths differ by at most one granule anywhere. 1000^3 becomes
    // 256/256/256/232 by 252/252/252/244 (1.14x) and 1021^3 1.08x.
    //
    // Every band is still non-empty whenever `count <= extent/gran`, which the
    // candidate search guarantees for the grid (`ntm <= pm`, `ntn <= pn`). The
    // K-split deliberately does NOT guarantee it — `ntn = nt` is uncapped there
    // — and its empty reduction bands are expected; see the driver's early
    // return.
    let band = |b: &mut IrBuilder, idx: &str, count: &str, extent: &str, gran: &str| {
        let gm1 = b.sub(gran, "1");
        let up = b.add(extent, &gm1);
        let g = b.t();
        b.w(&format!("{} = sdiv i64 {}, {}", g, up, gran));
        let lo0 = b.mul(idx, &g);
        let lo1 = b.t();
        b.w(&format!("{} = sdiv i64 {}, {}", lo1, lo0, count));
        let lo2 = b.mul(&lo1, gran);
        let lo = b.imin(&lo2, extent);
        let nx = b.add(idx, "1");
        let hi0 = b.mul(&nx, &g);
        let hi1 = b.t();
        b.w(&format!("{} = sdiv i64 {}, {}", hi1, hi0, count));
        let hi2 = b.mul(&hi1, gran);
        let hi3 = b.imin(&hi2, extent);
        let is_last = b.t();
        b.w(&format!("{} = icmp eq i64 {}, {}", is_last, nx, count));
        let hi = b.t();
        b.w(&format!(
            "{} = select i1 {}, i64 {}, i64 {}",
            hi, is_last, extent, hi3
        ));
        (lo, hi)
    };
    let (m_from, m_to) = band(&mut b, &gi, &ntm, m, &gran_m);
    let (n_from, n_to) = band(&mut b, &gj, &ntn, n, &gran_n);

    // A slot past the grid must present an EMPTY range on BOTH axes. Its `gi`
    // and `gj` are computed from a `ti` the grid does not cover, so `gj` wraps
    // back to a live column band and `gi` runs past the last row — neither is
    // meaningful, and under the K-split, where the driver's early return tests
    // only the M range, an unclamped M band would have a spare thread recompute
    // the whole of C on top of this call's output.
    let inuse = b.t();
    b.w(&format!("{} = icmp slt i64 {}, {}", inuse, ti, nt));
    let slot = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x %y_gemm_task], ptr {}, i64 0, i64 {}",
        slot, MAX_THREADS, POOL_TASK, ti
    ));
    let m_to2 = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        m_to2, inuse, m_to, m_from
    ));
    let n_to2 = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        n_to2, inuse, n_to, n_from
    ));
    // This slot's K-band, for the K-split. Unlike the M/N bands there is no
    // tile granularity to snap to, and the last thread takes the remainder so
    // no row of B is dropped. Under the K-split the grid is `1 x nt`, so `ti`
    // indexes the K-bands directly.
    let tn = b.add(&ti, "1");
    let k_last = b.t();
    b.w(&format!("{} = icmp eq i64 {}, {}", k_last, tn, nt));
    let kl0 = b.mul(&ti, k);
    let kfrom = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", kfrom, kl0, nt));
    let kl1 = b.mul(&tn, k);
    let kto0 = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", kto0, kl1, nt));
    let kto = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        kto, k_last, k, kto0
    ));

    let scr_off = b.mul(&ti, &SCRATCH_FLOATS.to_string());
    let scr = b.gep("float", &scratch, &scr_off);
    for (idx, ty, val) in [
        (0, "ptr", a.to_string()),
        (1, "ptr", bb.to_string()),
        (2, "ptr", c.to_string()),
        (3, "i64", m.to_string()),
        (4, "i64", n.to_string()),
        (5, "i64", k.to_string()),
        (6, "i64", m_from),
        (7, "i64", m_to2),
        (8, "i64", n_from),
        (9, "i64", n_to2),
        (10, "ptr", scr),
        (11, "i64", ti.clone()),
        (12, "i64", nt.clone()),
        (13, "i64", share.clone()),
        (14, "i64", kfrom),
        (15, "i64", kto),
        (16, "i64", ksplit.clone()),
        (17, "i64", lda.to_string()),
        (18, "i64", ldb_c.to_string()),
        (19, "i64", ldc.to_string()),
    ] {
        let f = b.t();
        b.w(&format!(
            "{} = getelementptr inbounds %y_gemm_task, ptr {}, i64 0, i32 {}",
            f, slot, idx
        ));
        b.w(&format!("store {} {}, ptr {}, align 8", ty, val, f));
    }
    b.loop_end(fl);

    // Publish. The release pairs with the workers' acquire load, so every task
    // record above is visible before any worker observes the new generation.
    let gcur = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", gcur, POOL_GEN));
    let gnext = b.add(&gcur, "1");
    b.w(&format!("call i32 @pthread_mutex_lock(ptr {})", POOL_MUTEX));
    b.w(&format!(
        "store atomic i64 {}, ptr {} release, align 8",
        gnext, POOL_GEN
    ));
    b.w(&format!("call i32 @pthread_cond_broadcast(ptr {})", POOL_COND));
    b.w(&format!("call i32 @pthread_mutex_unlock(ptr {})", POOL_MUTEX));

    // Slice 0 on this thread, so a core is never idle waiting on its own fork.
    let own = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x %y_gemm_task], ptr {}, i64 0, i64 0",
        own, MAX_THREADS, POOL_TASK
    ));
    b.w(&format!("call void @__y_gemm_worker(ptr {})", own));

    let jl = b.loop_begin("e.join", "1", &pool_n, "1");
    let ji = b.iv(&jl);
    let dp = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 {}",
        dp, MAX_THREADS, POOL_DONE, ji
    ));
    let spin = b.l("e.spinwait");
    let done = b.l("e.done");
    b.w(&format!("br label %{}", spin));
    b.out.push_str(&format!("{}:\n", spin));
    let dv = b.t();
    b.w(&format!(
        "{} = load atomic i64, ptr {} acquire, align 8",
        dv, dp
    ));
    let fin = b.t();
    b.w(&format!("{} = icmp eq i64 {}, {}", fin, dv, gnext));
    let again = b.l("e.again");
    b.w(&format!("br i1 {}, label %{}, label %{}", fin, done, again));
    b.out.push_str(&format!("{}:\n", again));
    // Without this the waiting thread hammers the very cache line the workers
    // are publishing into, and on an SMT sibling it steals issue slots from a
    // core still doing useful FMAs.
    b.w("call void @llvm.x86.sse2.pause()");
    b.w(&format!("br label %{}", spin));
    b.out.push_str(&format!("{}:\n", done));
    b.loop_end(jl);

    b.finish(&format!(
        "define internal void @{}(ptr noalias {}, ptr noalias {}, ptr noalias {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {})",
        KERNEL_NAME, a, bb, c, m, n, k, lda, ldb_c, ldc
    ))
}

/// `x - 1`, as a fresh temporary.
fn b_sub1(b: &mut IrBuilder, x: &str) -> String {
    let r = b.t();
    b.w(&format!("{} = sub nsw i64 {}, 1", r, x));
    r
}

/// The whole GEMM support module: globals, packing routines, micro-kernel and
/// blocked driver.
pub fn emit_kernel_module() -> String {
    let mut s = String::new();
    s.push_str(&emit_globals());
    s.push('\n');
    s.push_str(&emit_pack_b());
    s.push('\n');
    s.push_str(&emit_pack_a());
    s.push('\n');
    s.push_str(&emit_micro());
    s.push('\n');
    s.push_str(&emit_small_m());
    s.push('\n');
    s.push_str(&emit_small_reduce());
    s.push_str("\n");
    s.push_str(&emit_tiny());
    s.push('\n');
    s.push_str(&emit_driver());
    s.push('\n');
    s.push_str(&emit_worker());
    s.push('\n');
    s.push_str(&emit_threads());
    s.push('\n');
    s.push_str(&emit_pool_worker());
    s.push('\n');
    s.push_str(&emit_entry());
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn kernel_body(src: &str) -> Block {
        let mut lx = Lexer::new(src);
        let toks = lx.tokenize();
        let mut p = Parser::new(toks);
        let prog = p.parse_program().expect("test source should parse");
        for item in prog.items {
            if let Item::Kernel(k) = item {
                return k.body;
            }
        }
        panic!("no kernel in source");
    }

    const CANONICAL: &str = r#"
kernel mm(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    for i in 0..M step 1 {
        for j in 0..N step 1 {
            let mut sum: F32 = 0.0;
            for k in 0..K step 1 {
                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);
                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}
"#;

    /// `@bounds` on the operand loads reaches the emitter, and BOTH are needed.
    ///
    /// This is the front-end half of the exact-VNNI licence. The overflow
    /// obligation is stated over `A[i,k]` and `B[k,j]`, so the recogniser has
    /// to carry the bound from the statement that loads each of them — the
    /// accumulator's own `@bounds` describes the sum and cannot substitute.
    #[test]
    fn operand_bounds_are_carried_from_the_loads() {
        let bounded = CANONICAL
            .replace(
                "            let mut sum: F32 = 0.0;",
                "            @ZeroDrift\n            let mut sum: F32 = 0.0;",
            )
            .replace(
                "                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
                "                @bounds(min=-1024, max=1024)\n                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
            )
            .replace(
                "                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);",
                "                @bounds(min=-512, max=768)\n                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);",
            );

        let shape = recognize_gemm(&kernel_body(&bounded)).expect("still a GEMM");
        let d = shape.drift.expect("@ZeroDrift must reach the emitter");
        assert_eq!(d.a_bounds, Some((-1024.0, 1024.0)));
        assert_eq!(d.b_bounds, Some((-512.0, 768.0)));

        // Combined by the LARGER magnitude: the overflow derivation uses one
        // bound covering both operands (`m^2` per product), so taking the
        // smaller — or their product — would licence a nest that can overflow.
        let ob = d.operand_bounds().expect("both operands are bounded");
        assert_eq!(ob.max_magnitude, 1024.0);

        // And that licences the measured probe configuration.
        crate::zero_drift::license_vnni_exact(
            Some(ob),
            crate::zero_drift::VnniExact::DEFAULT_FLUSH_K_PAIRS,
        )
        .expect("operands bounded by 1024 are licensed at the default interval");
    }

    /// `plan_exact_gemm` distinguishes "not licensed" from "not exact", and the
    /// caller must be able to tell them apart.
    ///
    /// An `Unavailable` plan is not an error condition: the nest still compiles
    /// and still gets an exact accumulator through scalar lowering. Only the
    /// fast kernel is lost. The emitter routes this to `drift_report`, and
    /// routing it to `emit_errors` instead would fail programs that are
    /// correct — which is why the two are separate variants rather than an
    /// `Option`.
    #[test]
    fn an_unlicensed_plan_is_not_an_error() {
        let unbounded = DriftAccumulator {
            ty: "F32".into(),
            bounds: Some((-1000.0, 1000.0)),
            a_bounds: None,
            b_bounds: None,
        };
        match plan_exact_gemm(&unbounded) {
            ExactGemmPlan::Unavailable(reason) => {
                assert!(reason.contains("cancel"), "must say why: {reason}");
            }
            ExactGemmPlan::Vnni { .. } => {
                panic!("an unbounded nest must not be licensed")
            }
        }

        let bounded = DriftAccumulator {
            ty: "F32".into(),
            bounds: None,
            a_bounds: Some((-1024.0, 1024.0)),
            b_bounds: Some((-1024.0, 1024.0)),
        };
        match plan_exact_gemm(&bounded) {
            ExactGemmPlan::Vnni { scheme, operand_magnitude } => {
                assert_eq!(scheme.flush_k_pairs, 64);
                assert_eq!(operand_magnitude, 1024.0);
            }
            ExactGemmPlan::Unavailable(r) => panic!("1024 must be licensed: {r}"),
        }

        // Note the second case has NO accumulator bound and is still licensed:
        // the licence is about the operands only. That is not an oversight, it
        // is the separation this whole path exists to enforce.
    }

    /// One bounded operand is not enough, and the missing one is a REFUSAL.
    ///
    /// Bounding only A says nothing about B, and `m^2` per product needs both.
    /// Treating the stated one as covering the pair is the silent-approximation
    /// failure the design rule exists for.
    #[test]
    fn one_bounded_operand_does_not_licence_the_pair() {
        let half = CANONICAL
            .replace(
                "            let mut sum: F32 = 0.0;",
                "            @ZeroDrift\n            let mut sum: F32 = 0.0;",
            )
            .replace(
                "                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
                "                @bounds(min=-1024, max=1024)\n                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
            );

        let shape = recognize_gemm(&kernel_body(&half)).expect("still a GEMM");
        let d = shape.drift.expect("@ZeroDrift must reach the emitter");
        assert_eq!(d.a_bounds, Some((-1024.0, 1024.0)));
        assert_eq!(d.b_bounds, None);
        assert!(
            d.operand_bounds().is_none(),
            "one bounded operand must not licence the pair"
        );

        let err = crate::zero_drift::license_vnni_exact(
            d.operand_bounds(),
            crate::zero_drift::VnniExact::DEFAULT_FLUSH_K_PAIRS,
        )
        .expect_err("an unbounded operand must refuse");
        assert!(err.contains("cancel"), "must explain why: {err}");
    }

    /// Operands too large for the int32 accumulator are refused, not truncated.
    #[test]
    fn oversized_operand_bounds_are_refused() {
        let big = CANONICAL
            .replace(
                "            let mut sum: F32 = 0.0;",
                "            @ZeroDrift\n            let mut sum: F32 = 0.0;",
            )
            .replace(
                "                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
                "                @bounds(min=-30000, max=30000)\n                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
            )
            .replace(
                "                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);",
                "                @bounds(min=-30000, max=30000)\n                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);",
            );

        let shape = recognize_gemm(&kernel_body(&big)).expect("still a GEMM");
        let d = shape.drift.expect("@ZeroDrift must reach the emitter");
        let ob = d.operand_bounds().expect("both bounded");
        assert_eq!(ob.max_magnitude, 30000.0);

        // 30000 fits int16 but not the flush interval's product budget, so the
        // refusal must be the overflow one and must suggest a real fix.
        let err = crate::zero_drift::license_vnni_exact(
            Some(ob),
            crate::zero_drift::VnniExact::DEFAULT_FLUSH_K_PAIRS,
        )
        .expect_err("30000 overflows int32 at 64 k-pairs");
        assert!(err.contains("overflow"), "expected the overflow reason: {err}");
    }

    #[test]
    fn canonical_matmul_is_recognized() {
        let shape = recognize_gemm(&kernel_body(CANONICAL)).expect("should recognize");
        assert_eq!(shape.a, "A");
        assert_eq!(shape.b, "B");
        assert_eq!(shape.c, "C");
        assert_eq!(shape.m, "M");
        assert_eq!(shape.n, "N");
        assert_eq!(shape.k, "K");
        // The packed case: the strides ARE the extents, and are recorded as
        // such rather than assumed.
        assert_eq!(shape.lda, "K");
        assert_eq!(shape.ldb, "N");
        assert_eq!(shape.ldc, "N");
    }

    /// A `@ZeroDrift` accumulator is RECORDED now rather than refused, so an
    /// exact kernel can be selected for it. The refusal used to live in the
    /// recogniser, which meant the request never reached the emitter at all.
    ///
    /// The bounds matter as much as the flag: an exact fixed-point format buys
    /// order-independence and pays for it with a bounded range, so the stated
    /// range is what makes a representation selectable.
    #[test]
    fn a_zero_drift_accumulator_is_recorded_with_its_bounds() {
        let drifted = CANONICAL.replace(
            "            let mut sum: F32 = 0.0;",
            "            @bounds(min=-1000, max=1000)\n            @ZeroDrift\n            let mut sum: F32 = 0.0;",
        );
        let shape = recognize_gemm(&kernel_body(&drifted)).expect("a drifted nest is still a GEMM");
        let d = shape.drift.expect("the @ZeroDrift request must reach the emitter");
        assert_eq!(d.ty, "F32");
        assert_eq!(d.bounds, Some((-1000.0, 1000.0)));

        // Control: without the directive there is no obligation to carry, and
        // the ordinary f32 kernel remains a legal lowering.
        let plain = recognize_gemm(&kernel_body(CANONICAL)).expect("should recognize");
        assert!(plain.drift.is_none());

        // The accumulator's own bound is NOT an operand bound, and stating only
        // it must not licence anything. See `DriftAccumulator::operand_bounds`.
        assert!(
            d.operand_bounds().is_none(),
            "an accumulator bound alone must not stand in for operand bounds"
        );
    }

    /// A row stride larger than the extent is a submatrix — a legal, common
    /// BLAS input, and the whole reason this kernel can implement `sgemm`.
    /// It must be RECOGNISED with the stride recorded, not refused.
    ///
    /// This replaced a near-miss test asserting the opposite. That test was
    /// pinning the restriction, not a correctness property: requiring
    /// `lda == K` sent every submatrix call to scalar lowering, which is a
    /// ~100x slowdown with no diagnostic.
    #[test]
    fn a_leading_dimension_is_recorded_not_refused() {
        let strided = CANONICAL
            .replace(
                "kernel mm(A: GlobalMemory<F32>, B: GlobalMemory<F32>, \
                 C: GlobalMemory<F32>, M: I32, N: I32, K: I32)",
                "kernel mm(A: GlobalMemory<F32>, B: GlobalMemory<F32>, \
                 C: GlobalMemory<F32>, M: I32, N: I32, K: I32, LDA: I32)",
            )
            .replace(
                "block_ptr2d_load(A, i, k, K, M, K)",
                "block_ptr2d_load(A, i, k, LDA, M, K)",
            );
        let shape = recognize_gemm(&kernel_body(&strided)).expect("submatrix A is legal");
        assert_eq!(shape.lda, "LDA");
        // The extent is still K — only the addressing changed.
        assert_eq!(shape.k, "K");
        assert_eq!(shape.ldb, "N");
    }

    /// The recogniser decides to compute something *different* from what was
    /// written, so every one of these near-misses must be refused. A false
    /// positive here is a wrong answer that no test of the emitted IR would
    /// catch, because the IR would be a correct GEMM — just not this program.
    #[test]
    fn near_misses_are_refused() {
        // B indexed [j, k] instead of [k, j] — that is A * Bᵀ.
        let transposed = CANONICAL.replace(
            "block_ptr2d_load(B, k, j, N, K, N)",
            "block_ptr2d_load(B, j, k, N, K, N)",
        );
        assert!(recognize_gemm(&kernel_body(&transposed)).is_none());

        // Subtraction, not accumulation.
        let minus = CANONICAL.replace("sum = sum + a_val", "sum = sum - a_val");
        assert!(recognize_gemm(&kernel_body(&minus)).is_none());

        // A non-unit step visits a strict subset of K.
        let step2 = CANONICAL.replace("for k in 0..K step 1", "for k in 0..K step 2");
        assert!(recognize_gemm(&kernel_body(&step2)).is_none());

        // A stride that is not a plain identifier cannot be lowered: there is
        // no scope here to evaluate an expression in, and guessing a value
        // would address the wrong memory. (A stride that IS an identifier but
        // differs from the extent is a submatrix and is now accepted — see
        // `a_leading_dimension_is_recorded_not_refused`.)
        let expr_stride = CANONICAL.replace(
            "block_ptr2d_load(A, i, k, K, M, K)",
            "block_ptr2d_load(A, i, k, K + 1, M, K)",
        );
        assert!(recognize_gemm(&kernel_body(&expr_stride)).is_none());

        // An accumulator that starts from something other than zero.
        let init = CANONICAL.replace("let mut sum: F32 = 0.0;", "let mut sum: F32 = 1.0;");
        assert!(recognize_gemm(&kernel_body(&init)).is_none());
    }

    #[test]
    fn emitted_module_has_the_expected_pieces() {
        let ir = emit_kernel_module();
        assert!(ir.contains(&format!("define internal void @{}", KERNEL_NAME)));
        assert!(ir.contains("@__y_gemm_pack_a"));
        assert!(ir.contains("@__y_gemm_pack_b"));
        assert!(ir.contains("@__y_gemm_micro"));
        assert!(ir.contains("llvm.fmuladd.v16f32"));
        // MR * NRV accumulators, all distinct.
        for i in 0..tl().mr {
            for v in 0..tl().nrv() {
                assert!(ir.contains(&format!("%acc{}_{} = alloca <16 x float>", i, v)));
            }
        }
    }

    /// A tile over the register budget must be REFUSED, not emitted.
    ///
    /// An over-budget tile compiles and runs — it just spills every
    /// accumulator to the stack in the innermost loop. That reads as a
    /// mysterious 3x slowdown rather than as a bad tile, which is exactly the
    /// failure mode `Y_GEMM_TILE` would otherwise invite during a sweep.
    #[test]
    fn over_budget_tiles_are_refused() {
        // 16x4: 64 accumulators + 4 B + 2 A against 32 zmm.
        let t = Tile { mr: 16, nr: 64, kc: 256, mc: 192, nc: 2048 };
        assert!(t.regs() > 32);
        assert!(t.check().is_err(), "16x64 needs {} registers", t.regs());

        // The shipped default and every tile named in docs/cpu_gemm_tuning.md
        // must pass, or the sweep table describes tiles that cannot be built.
        for t in [
            DEFAULT_TILE,
            Tile { mr: 12, nr: 32, kc: 256, mc: 192, nc: 2048 },
            Tile { mr: 8, nr: 48, kc: 256, mc: 192, nc: 2048 },
            Tile { mr: 6, nr: 64, kc: 1024, mc: 384, nc: 2048 },
        ] {
            assert!(t.check().is_ok(), "{:?} should be accepted: {:?}", t, t.check());
            assert!(t.regs() <= 32);
        }

        // A non-vector-width nr is refused rather than silently truncated by
        // the `nr / 16` division.
        assert!(Tile { mr: 6, nr: 40, kc: 256, mc: 192, nc: 2048 }.check().is_err());
        // And anything past what SCRATCH_FLOATS reserves.
        assert!(Tile { mr: 6, nr: 64, kc: KC_MAX + 1, mc: 192, nc: 2048 }.check().is_err());
    }

    /// The packed-A region must not be able to reach the packed-B region, for
    /// ANY `kc` the runtime rule can pick — not merely for the current tile.
    ///
    /// `kc` is chosen at runtime from the thread count, so a B offset derived
    /// from `tl().kc` puts the A panel's tail on top of B whenever the runtime
    /// value is larger. That is a wrong answer, not a crash, and it appears
    /// only on the shapes the runtime `kc` exists to speed up.
    #[test]
    fn scratch_layout_holds_for_the_largest_runtime_kc() {
        let b_offset = (MC_MAX + MR_MAX) * KC_MAX;
        let widest_a = (MC_MAX + MR_MAX) * KC_MAX;
        assert!(b_offset >= widest_a, "A panel can overrun into B");

        let widest_b = KC_MAX * (NC_MAX + NR_MAX);
        assert!(
            SCRATCH_FLOATS >= b_offset + widest_b,
            "SCRATCH_FLOATS={} cannot hold A({}) + B({})",
            SCRATCH_FLOATS,
            b_offset,
            widest_b
        );

        // The emitted offset must be the max-derived one. A tile-derived
        // offset would still pass the arithmetic above while being wrong.
        let ir = emit_kernel_module();
        assert!(
            ir.contains(&format!("i64 {}", b_offset)),
            "packed-B offset is not the SCRATCH_FLOATS-derived one"
        );

        // And the runtime rule must actually be emitted, with the budget the
        // docs quote. If this constant moves, the sweep table is stale.
        assert!(
            ir.contains(&format!("udiv i64 {}", L3_PANEL_FLOATS)),
            "the thread-count-dependent kc rule is not in the emitted IR"
        );

        // The SHARED B panel needs the identical argument, and did not get it:
        // it was sized `tl().kc * (tl().nc + tl().nr)` while this very test was
        // asserting that the private one must not be. `pack_b` writes
        // `kc * roundup(nc, NR)` floats into whichever of the two it was
        // handed, so one bound covers both. 192x513x22000 overflowed the
        // tile-sized version by ~98 KB, into the pool mutex and condvar that
        // follow it in BSS; `tests/cpu_gemm_threaded.rs` carries that shape.
        assert!(
            ir.contains(&format!(
                "{} = internal global [{} x float]",
                SHARED_B, widest_b
            )),
            "the shared packed-B global is not sized for the largest runtime kc"
        );
    }

    /// The runtime `kc` rule must reproduce both measured optima.
    ///
    /// These two values were found by independent sweeps before the formula
    /// existed; the formula is only trustworthy because it lands on them.
    #[test]
    fn runtime_kc_rule_matches_the_measured_optima() {
        let rule = |nthr: usize, nc: usize| {
            (L3_PANEL_FLOATS / (nthr * nc)).min(KC_MAX).max(KC_MIN)
        };
        assert_eq!(rule(1, 2048), 1024, "one thread should get the deep panel");
        assert_eq!(rule(16, 2048), 256, "sixteen threads should get the shallow one");
        // Never below the floor, however many threads or however wide nc is.
        assert_eq!(rule(64, 2048), KC_MIN);
    }
}
