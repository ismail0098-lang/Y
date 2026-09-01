// ============================================================
//  Y  —  Empirical (measured) autotuning
//  empirical_autotune.rs
//
//  `Autotuner::score_candidate` (src/autotuner.rs) ranks candidate tile
//  configurations with a hand-fitted analytic model. That model's
//  coefficients were fitted to 15 configurations measured at M=N=K=4096 on
//  one specific GPU and needed three rounds of correction before it stopped
//  regressing the 256/512/1024 shapes - and it only ever agreed with reality
//  after reality had been measured. It also cannot adapt to a different GPU
//  at all: every coefficient in it encodes this card's register file, its
//  shared-memory ceiling and its warp-scheduler count.
//
//  This module replaces guessing with measuring. For a given (M, N, K) it:
//
//    1. compiles each candidate through the REAL codegen path - the same
//       `PtxEmitter::emit_program` a `--emit-ptx` invocation runs, driven by
//       a synthesized `@tile` kernel identical in shape to
//       `tests/gemm_f16_<N>.ysu`, so what gets measured is the kernel that
//       will actually ship, not a proxy. (`src/bin/autotune_verify.rs`
//       measures a separate hand-written CUDA C++ WMMA kernel parameterised
//       the same way; that answers a different, weaker question.)
//    2. JIT-compiles it with the installed driver (`cuModuleLoadDataEx`),
//       which is the same loader path the Python benchmark harnesses and any
//       real embedder use,
//    3. checks it against a CPU reference before believing any timing from
//       it,
//    4. times it under the clock-ramp + A/B-interleave discipline described
//       below,
//    5. returns the ranking, measured.
//
//  MEASUREMENT DISCIPLINE (all of this is load-bearing, none of it is
//  ceremony - see `RAMP_SECONDS` and `measure_round_robin` for the specific
//  failures each part prevents):
//
//    * The SM clock on this project's dev GPU idles at ~210 MHz and needs
//      ~3s of sustained load to reach ~2670 MHz, a 12.7x swing, and the box
//      has no permission to lock clocks. Timing anything before that ramp
//      measures the clock, not the kernel.
//    * Candidates must be interleaved, not measured one-after-another to
//      completion, or later candidates are systematically favoured by a
//      hotter clock. The rotation in `measure_round_robin` additionally
//      removes position-within-round bias.
//    * A candidate that produces wrong results is not a fast candidate. Each
//      one is correctness-checked before it is allowed into the ranking; a
//      kernel that writes nothing is otherwise extremely fast.
// ============================================================


use std::time::Instant;

use crate::autotuner::AutotuneCandidate;
use crate::cuda_runtime::{f16_bits_to_f32, random_f16_bits, CudaContext, DeviceBuffer};
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::ptx_emitter::PtxEmitter;
use crate::sentinel::HardwareProfile;

/// Sustained-load seconds before any measurement is taken. See this module's
/// header comment - without it the same kernel has been observed to measure
/// anywhere in a 9x band purely from clock state.
const RAMP_SECONDS: f64 = 3.0;

/// Interleaved rounds in the cheap screening pass and in the final pass.
/// Screening only has to separate "plausible" from "hopeless", so it runs
/// few rounds over many candidates; the final pass runs many rounds over the
/// few survivors, where the differences are small and the noise floor
/// matters.
///
/// `FINAL_ROUNDS` was raised from 7 after measuring: at M=N=K=512 the top
/// two candidates differ by ~3%, and across three forced remeasurements at 7
/// rounds they swapped rank every time (9.71/10.03/10.28us against
/// 10.03/9.96/10.01us). Seven rounds could not resolve them, which is a
/// statement about the harness, not about the kernels. The cost of the extra
/// rounds is a few hundred milliseconds on a result that is then cached.
const SCREEN_ROUNDS: u32 = 3;
const FINAL_ROUNDS: u32 = 15;

/// How many screening survivors get the full treatment.
const FINALISTS: usize = 5;

/// Target wall time for one timed batch, in microseconds. Iteration counts
/// are calibrated per candidate to hit roughly this, so that CUDA event
/// overhead (~5-10us) stays negligible against the batch even for decode
/// shapes where a single launch is only tens of microseconds.
const SCREEN_BATCH_US: f64 = 1_500.0;
const FINAL_BATCH_US: f64 = 10_000.0;

/// Relative L2 error over the sampled output positions above which a
/// candidate is rejected as incorrect.
///
/// Both the kernel and the reference read bit-identical f16 inputs (see
/// `cuda_runtime::random_f16_bits`) and both accumulate in f32, so the only
/// legitimate difference is f32 summation ORDER, worth ~1e-5 relative at the
/// deepest K this compiler supports. This threshold is three orders of
/// magnitude above that and still four orders below what any real bug
/// produces (a kernel that writes nothing scores exactly 1.0).
const CORRECTNESS_REL_L2_TOL: f64 = 2e-2;

/// Output positions sampled for the correctness check. Sampling rather than
/// pulling back the whole matrix keeps the check O(S*K) instead of O(M*N*K)
/// and avoids a 1GB device-to-host copy at M=N=16384.
const CORRECTNESS_SAMPLES: usize = 48;

const SEED_A: u64 = 0xA5A5_1234_5678_9ABC;
const SEED_B: u64 = 0xB1B1_0FED_CBA9_8765;

const PROBE_KERNEL: &str = "y_autotune_probe";

/// Launch geometry read back out of the emitted PTX, so the launch is
/// guaranteed to match what the kernel was compiled for rather than being a
/// second, independently derived guess that can silently drift from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid_x: u32,
    pub grid_y: u32,
    pub threads: u32,
    pub dyn_smem_bytes: u32,
}

/// One candidate's measured time across the interleaved rounds.
///
/// RANKING USES `best_us` (the fastest round), NOT the median. That is not
/// the usual advice, and it was not the original design here - it is what
/// the data said. Two forced remeasurements of M=N=K=512, 15 rounds each:
///
///   config              min (run1 / run2)     median (run1 / run2)
///   64x64x64 2x2 s3      9.69 / 9.69           9.76 / 10.29
///   64x64x64 2x2 s4      9.97 / 9.94          10.05 / 10.30
///   64x64x64 2x2 s2     10.40 / 10.40         10.68 / 11.03
///   64x64x32 2x2 s4     10.49 / 10.48         10.49 / 11.15
///   64x64x32 2x2 s3     10.76 / 10.76         10.85 / 11.35
///
/// The minima reproduce to within 0.3% across independent processes while
/// the medians drift by up to 5% - enough to swap the top two ranks, which
/// it did on an earlier 7-round version of this harness. The reason is that
/// the contamination here is ONE-SIDED: a clock dip, a competing process or
/// a scheduler hiccup can only ever make a round slower than the kernel
/// really is, never faster. So the fastest round is the best available
/// estimate of the kernel's actual cost, and the median mostly measures how
/// disturbed the machine was.
///
/// The median and max are still reported, because the spread is what tells a
/// reader whether a ranking is resolved at all.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// Fastest observed round. The ranking statistic - see above.
    pub best_us: f64,
    pub median_us: f64,
    pub max_us: f64,
}

/// Floor on the tie band, as a fraction of the measured time.
///
/// Only matters on an unusually quiet machine where a candidate's median
/// lands exactly on its best, which would otherwise claim infinite
/// resolution and declare a 0.01% gap decisive.
const TIE_FLOOR_FRACTION: f64 = 0.0025;

impl Timing {
    /// Whether the harness can actually separate these two candidates.
    ///
    /// The band is derived from THIS RUN'S OWN dispersion (`median - best`
    /// per candidate), not from a fixed constant, because the noise floor is
    /// a property of the regime, not of the harness. Measured, on the same
    /// machine and the same 15-round harness:
    ///
    ///   * M=N=K=512, compute-bound and L2-resident: `best_us` reproduces
    ///     across independent processes to ~0.3%.
    ///   * M=1, N=K=4096, streaming a 201 MB working set: `best_us`
    ///     reproduces to ~4%. Four forced remeasurements put the same
    ///     candidate at 56.10 / 56.18 / 58.14 / 55.94us, while the entire
    ///     top-five spread was 4.8% - i.e. the run-to-run noise is the same
    ///     size as the differences being ranked. At the DRAM roofline these
    ///     kernels are genuinely not separable, and any harness that reports
    ///     an ordering there is reporting noise.
    ///
    /// A constant band calibrated on the first regime is off by an order of
    /// magnitude in the second. An earlier revision used a flat 1%, which
    /// declared a 1.27% gap at M=1 decisive and selected a 32-thread CTA on
    /// it; that gap was smaller than the candidate's own run-to-run spread.
    fn tied_with(&self, other: &Timing) -> bool {
        let noise = (self.median_us - self.best_us)
            .max(other.median_us - other.best_us)
            .max(TIE_FLOOR_FRACTION * self.best_us.min(other.best_us));
        (self.best_us - other.best_us).abs() <= noise
    }
}

#[derive(Debug, Clone)]
pub struct MeasuredCandidate {
    pub candidate: AutotuneCandidate,
    pub timing: Timing,
    pub tflops: f64,
    /// False when this candidate was screened out and never re-measured in
    /// the final pass, so its time is the coarse screening estimate.
    pub finalist: bool,
}

impl MeasuredCandidate {
    /// The ranking statistic. See `Timing`.
    pub fn us(&self) -> f64 {
        self.timing.best_us
    }
}

/// Why a shape could not be measured. Every variant is a fall-back-to-the-
/// heuristic condition, not an error to propagate: the compiler must still
/// produce a kernel on a machine with no GPU.
#[derive(Debug)]
pub enum TuneFailure {
    NoCudaDevice,
    NoUsableCandidate(String),
    Cuda(String),
}

impl std::fmt::Display for TuneFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuneFailure::NoCudaDevice => write!(f, "no CUDA device/driver available"),
            TuneFailure::NoUsableCandidate(s) => write!(f, "no candidate compiled and verified: {}", s),
            TuneFailure::Cuda(s) => write!(f, "CUDA error: {}", s),
        }
    }
}

// ── candidate PTX generation ───────────────────────────────

/// True when `emit_tensor_core_gemm_kernel` can actually lower this
/// candidate at this problem shape.
///
/// These mirror the conditions that function asserts on entry, so the
/// measurement loop never asks codegen for something codegen has already
/// declared out of contract.
///
/// The K condition is `k % 8 == 0`, NOT `k % cta_k == 0`. Since
/// `emit_tensor_core_gemm_kernel` gained a real K tail (`k_tiles` rounds up
/// and the partial tile is zero-filled by the existing per-chunk masking),
/// `cta_k` no longer has to divide K - only A's 16-byte cp.async chunk
/// granularity does. Keeping the old, stricter test here had a bad
/// consequence that outlived the codegen limit it was written for: at any
/// ragged shape EVERY candidate was filtered out, `tune_gemm_f16` reported
/// "candidate list was empty", and the compiler silently fell back to the
/// analytic `score_candidate` heuristic. Measured at M=N=K=1000, that
/// fallback picks `128x128x32` where a measured search picks `64x128x32` -
/// 36.0 us against 31.6 us, i.e. **0.77x vs 0.89x of cuBLAS**. So ragged
/// shapes were not merely un-tuned, they were tuned by the one path this
/// module exists to replace.
pub fn is_emittable(c: &AutotuneCandidate, k: u32) -> bool {
    if c.warps_m == 0 || c.warps_n == 0 || c.cta_k == 0 {
        return false;
    }
    let per_warp_m = c.cta_m / c.warps_m;
    let per_warp_n = c.cta_n / c.warps_n;
    (per_warp_m / 16) * 16 * c.warps_m == c.cta_m
        && (per_warp_n / 16) * 16 * c.warps_n == c.cta_n
        && per_warp_m >= 16
        && per_warp_n >= 16
        && c.cta_k % 16 == 0
        && k % 8 == 0
}

/// Synthesizes the `.ysu` source for a probe kernel of exactly the shape the
/// shipping GEMM tests use (compare `tests/gemm_f16_1024.ysu`), so the
/// emitter takes its normal `@tile` dispatch path.
fn probe_source(m: u32, n: u32, k: u32) -> String {
    format!(
        "// Generated by empirical_autotune.rs - not written to disk.\n\
         @tile({}, {}, {})\n\
         kernel {}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {{\n\
         }}\n\
         \n\
         fn main() {{\n\
         }}\n",
        m, n, k, PROBE_KERNEL
    )
}

/// Runs the real front-end and PTX backend for one forced candidate.
///
/// The forced-config hook in `autotuner` is what makes this terminate:
/// `emit_tensor_core_gemm_kernel` calls `Autotuner::autotune` itself, so
/// without it, emitting a candidate would re-enter tuning and recurse
/// forever.
pub fn emit_candidate_ptx(
    m: u32,
    n: u32,
    k: u32,
    candidate: &AutotuneCandidate,
    hw_profile: &HardwareProfile,
) -> Result<(String, LaunchConfig), String> {
    let src = probe_source(m, n, k);
    let ptx = crate::autotuner::with_forced_config(candidate, || {
        let mut lexer = Lexer::new(&src);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let ast = parser
            .parse_program()
            .map_err(|e| format!("probe source failed to parse: {}", e))?;
        let mut emitter = PtxEmitter::new_with_profile(hw_profile);
        Ok::<String, String>(emitter.emit_program(&ast, hw_profile))
    })?;

    let launch = parse_launch_config(&ptx, m, n)?;
    Ok((ptx, launch))
}

/// Recovers grid/block/dynamic-shared-memory from the emitter's own
/// `[Y TENSOR CORE GEMM]` PTX comment.
///
/// Deliberately parsed from the emitted text rather than recomputed from the
/// candidate: the emitter clamps the requested stage count against smem and
/// `k_tiles`, and it sizes the dynamic allocation as the max of the pipeline
/// buffers and the epilogue scratch tile. Recomputing any of that here would
/// be a second implementation of it, free to drift.
fn parse_launch_config(ptx: &str, m: u32, n: u32) -> Result<LaunchConfig, String> {
    let marker = "// [Y TENSOR CORE GEMM]";
    let line = ptx
        .lines()
        .find(|l| l.contains(marker))
        .ok_or_else(|| {
            "emitted PTX has no '[Y TENSOR CORE GEMM]' comment - the @tile GEMM path did not \
             fire for the probe kernel (it falls back to generic scalar lowering silently)"
                .to_string()
        })?;

    // "... | CTA 64x128x32 | 2x2 warps | ..."
    let cta_field = field_between(line, "| CTA ", " |")
        .ok_or_else(|| format!("could not find the CTA field in: {}", line.trim()))?;
    let cta: Vec<u32> = cta_field.split('x').filter_map(|s| s.trim().parse().ok()).collect();
    if cta.len() != 3 {
        return Err(format!("malformed CTA field '{}'", cta_field));
    }
    let warp_field = field_between(line, "| ", " warps")
        .and_then(|f| f.rsplit("| ").next().map(|s| s.to_string()))
        .ok_or_else(|| format!("could not find the warps field in: {}", line.trim()))?;
    let warps: Vec<u32> = warp_field.split('x').filter_map(|s| s.trim().parse().ok()).collect();
    if warps.len() != 2 {
        return Err(format!("malformed warps field '{}'", warp_field));
    }

    // The second comment line carries the byte count, but ONLY on the
    // cp.async-pipelined path - the K-too-small synchronous fallback emits no
    // such clause, and 0 is the correct answer there (no dynamic smem).
    let dyn_smem_bytes = ptx
        .lines()
        .find_map(|l| field_between(l, "Dynamic shared memory required: ", " bytes"))
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);

    let (cta_m, cta_n) = (cta[0], cta[1]);
    Ok(LaunchConfig {
        grid_x: (n + cta_n - 1) / cta_n,
        grid_y: (m + cta_m - 1) / cta_m,
        threads: warps[0] * warps[1] * 32,
        dyn_smem_bytes,
    })
}

fn field_between(haystack: &str, start: &str, end: &str) -> Option<String> {
    let s = haystack.find(start)? + start.len();
    let rest = &haystack[s..];
    let e = rest.find(end)?;
    Some(rest[..e].to_string())
}

// ── device-side probe harness ──────────────────────────────

/// Owns the CUDA context and the A/B/C buffers every candidate for one
/// (M, N, K) shares.
///
/// Field order matters: `DeviceBuffer` frees through a copy of the driver
/// table and must be dropped while the context is still alive, and Rust
/// drops struct fields in declaration order.
struct GemmProbe {
    a: DeviceBuffer,
    /// Several distinct copies of B, rotated across launches. See
    /// `plan_weight_replicas`.
    b: Vec<DeviceBuffer>,
    c: DeviceBuffer,
    ctx: CudaContext,
    m: u32,
    n: u32,
    k: u32,
    /// Reported so the caller can say what regime the numbers came from.
    l2_bytes: usize,
}

/// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`. Verified against this project's dev
/// GPU, which must report 50331648 (48 MB) - the value the sentinel's own
/// profile and `tests/benchmark_y_decode_gemm.py`'s `L2_BYTES` both record.
/// An earlier revision of this file used 76 here, which is a different
/// attribute; it returned 0, `plan_weight_replicas` correctly refused to
/// guess from that, and the L2 rotation silently never engaged.
const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: i32 = 38;

/// Target working set as a multiple of L2, matching the convention already
/// established by `tests/benchmark_y_decode_gemm.py` (6 buffers x 33.55 MB =
/// ~4x this card's 48 MB L2).
const L2_OVERSUBSCRIBE: usize = 4;

/// Hard cap on weight replicas, and on the fraction of free device memory
/// the probe is willing to occupy.
const MAX_WEIGHT_REPLICAS: usize = 8;
const MAX_FREE_MEM_FRACTION: f64 = 0.6;

/// Decides how many distinct copies of B to rotate over.
///
/// A timing loop that hammers ONE weight buffer measures L2, not DRAM. This
/// is not hypothetical here: before this rotation existed, the probe timed
/// M=1, N=K=4096 at 39.65us over a 33.55 MB B, which is ~845 GB/s effective
/// - 1.26x ABOVE this card's 672 GB/s theoretical DRAM peak, and therefore
/// only reachable out of cache. Real autoregressive decode has a distinct
/// weight matrix per layer and nothing resident between tokens, so tuning
/// against the cached regime tunes for a situation the kernel never runs in
/// - and the axis that matters most there (`cta_k`, i.e. bytes in flight per
/// CTA) is exactly the one the cached regime cannot see.
///
/// Returns 1 whenever B alone already oversubscribes L2 (every large square
/// shape), so this costs nothing where it buys nothing.
fn plan_weight_replicas(b_bytes: usize, a_bytes: usize, c_bytes: usize, l2_bytes: usize, free_bytes: usize) -> usize {
    if b_bytes == 0 || l2_bytes == 0 {
        return 1;
    }
    let target = l2_bytes.saturating_mul(L2_OVERSUBSCRIBE);
    if a_bytes + b_bytes >= target {
        return 1;
    }
    let wanted = (target - a_bytes).div_ceil(b_bytes).max(1).min(MAX_WEIGHT_REPLICAS);

    // Never let the L2 defeat push the probe into an allocation failure.
    let budget = (free_bytes as f64 * MAX_FREE_MEM_FRACTION) as usize;
    let fixed = a_bytes + c_bytes;
    if budget <= fixed + b_bytes {
        return 1;
    }
    let affordable = (budget - fixed) / b_bytes;
    wanted.min(affordable.max(1))
}

impl GemmProbe {
    fn new(m: u32, n: u32, k: u32) -> Result<Self, TuneFailure> {
        let ctx = CudaContext::new().ok_or(TuneFailure::NoCudaDevice)?;

        let a_bytes = m as usize * k as usize * 2;
        let b_bytes = k as usize * n as usize * 2;
        let c_bytes = m as usize * n as usize * 4;

        let l2_bytes = ctx
            .device_attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .unwrap_or(0)
            .max(0) as usize;
        let free_bytes = ctx.mem_info().map(|(free, _)| free).unwrap_or(0);
        let replicas = plan_weight_replicas(b_bytes, a_bytes, c_bytes, l2_bytes, free_bytes);

        let a = ctx.alloc(a_bytes).map_err(TuneFailure::Cuda)?;
        let c = ctx.alloc(c_bytes).map_err(TuneFailure::Cuda)?;

        let mut b = Vec::with_capacity(replicas);
        b.push(ctx.alloc(b_bytes).map_err(TuneFailure::Cuda)?);
        fill_f16_buffer(&ctx, &a, m as usize * k as usize, SEED_A).map_err(TuneFailure::Cuda)?;
        fill_f16_buffer(&ctx, &b[0], k as usize * n as usize, SEED_B).map_err(TuneFailure::Cuda)?;
        // Replicas only need to be DIFFERENT memory, not different data, so
        // they are cloned on-device rather than regenerated and re-uploaded.
        for _ in 1..replicas {
            let extra = ctx.alloc(b_bytes).map_err(TuneFailure::Cuda)?;
            ctx.memcpy_dtod(&extra, &b[0]).map_err(TuneFailure::Cuda)?;
            b.push(extra);
        }

        Ok(GemmProbe { a, b, c, ctx, m, n, k, l2_bytes })
    }

    /// One argument set per weight replica, cycled across launches.
    fn arg_sets(&self) -> Vec<Vec<u64>> {
        self.b
            .iter()
            .map(|b| vec![self.a.device_ptr(), b.device_ptr(), self.c.device_ptr()])
            .collect()
    }

    /// The single argument set used for correctness, always against replica 0
    /// (the only one whose contents the CPU reference knows how to predict -
    /// though they are all identical, this keeps the dependency explicit).
    fn correctness_args(&self) -> Vec<u64> {
        vec![self.a.device_ptr(), self.b[0].device_ptr(), self.c.device_ptr()]
    }

    /// Zeroes C, runs the kernel once, and scores the sampled output against
    /// a CPU reference. Returns the relative L2 error.
    fn correctness_rel_l2(
        &self,
        kernel: &crate::cuda_runtime::KernelModule,
        launch: &LaunchConfig,
    ) -> Result<f64, String> {
        self.ctx.memset_u8(&self.c, 0)?;
        self.ctx.launch(
            kernel,
            (launch.grid_x, launch.grid_y, 1),
            (launch.threads, 1, 1),
            launch.dyn_smem_bytes,
            &self.correctness_args(),
        )?;
        self.ctx.synchronize()?;

        let (m, n, k) = (self.m as usize, self.n as usize, self.k as usize);
        let mut num = 0.0f64; // sum of squared differences
        let mut den = 0.0f64; // sum of squared reference values

        for s in 0..CORRECTNESS_SAMPLES {
            // Corners first (boundary/masking bugs live there), then a
            // coprime-strided spread across the whole output.
            let (row, col) = match s {
                0 => (0, 0),
                1 => (0, n - 1),
                2 => (m - 1, 0),
                3 => (m - 1, n - 1),
                _ => ((s * 7919) % m, (s * 104729) % n),
            };

            let mut reference = 0.0f64;
            for kk in 0..k {
                let av = f16_bits_to_f32(random_f16_bits((row * k + kk) as u64, SEED_A)) as f64;
                let bv = f16_bits_to_f32(random_f16_bits((kk * n + col) as u64, SEED_B)) as f64;
                reference += av * bv;
            }

            let mut got_bytes = [0u8; 4];
            let offset = (row * n + col) * 4;
            self.ctx.memcpy_dtoh_at(&mut got_bytes, &self.c, offset)?;
            let got = f32::from_le_bytes(got_bytes) as f64;

            num += (got - reference) * (got - reference);
            den += reference * reference;
        }

        if den == 0.0 {
            return Err("degenerate correctness reference (all sampled references are zero)".into());
        }
        Ok((num / den).sqrt())
    }

    /// Picks an iteration count that makes one timed batch last roughly
    /// `target_us`, so event overhead is negligible regardless of shape.
    fn calibrate_iters(
        &self,
        kernel: &crate::cuda_runtime::KernelModule,
        launch: &LaunchConfig,
        target_us: f64,
    ) -> Result<u32, String> {
        let per_launch = self.ctx.time_launches(
            kernel,
            (launch.grid_x, launch.grid_y, 1),
            (launch.threads, 1, 1),
            launch.dyn_smem_bytes,
            &self.arg_sets(),
            3,
        )?;
        if per_launch <= 0.0 {
            return Ok(20);
        }
        Ok(((target_us / per_launch).ceil() as u32).clamp(3, 500))
    }

    fn time(
        &self,
        kernel: &crate::cuda_runtime::KernelModule,
        launch: &LaunchConfig,
        iters: u32,
    ) -> Result<f64, String> {
        self.ctx.time_launches(
            kernel,
            (launch.grid_x, launch.grid_y, 1),
            (launch.threads, 1, 1),
            launch.dyn_smem_bytes,
            &self.arg_sets(),
            iters,
        )
    }
}

/// Uploads deterministic f16 test data without ever materialising the whole
/// buffer host-side (A and B are 512MB each at M=N=K=16384).
fn fill_f16_buffer(
    ctx: &CudaContext,
    buf: &DeviceBuffer,
    elements: usize,
    seed: u64,
) -> Result<(), String> {
    const CHUNK_ELEMS: usize = 4 * 1024 * 1024; // 8 MiB of f16
    let mut host = vec![0u8; CHUNK_ELEMS.min(elements) * 2];
    let mut done = 0usize;
    while done < elements {
        let this_chunk = CHUNK_ELEMS.min(elements - done);
        for i in 0..this_chunk {
            let bits = random_f16_bits((done + i) as u64, seed);
            host[i * 2] = (bits & 0xff) as u8;
            host[i * 2 + 1] = (bits >> 8) as u8;
        }
        ctx.memcpy_htod_at(buf, done * 2, &host[..this_chunk * 2])?;
        done += this_chunk;
    }
    Ok(())
}

// ── the tuning loop ────────────────────────────────────────

struct Compiled {
    candidate: AutotuneCandidate,
    launch: LaunchConfig,
    kernel: crate::cuda_runtime::KernelModule,
}

/// Measures every emittable candidate at this shape and returns them sorted
/// fastest-first.
///
/// `verbose` prints a per-candidate table; the CLI turns it on so a user who
/// asked for autotuning can see what it actually found, rather than being
/// told to trust it.
pub fn tune_gemm_f16(
    m: u32,
    n: u32,
    k: u32,
    hw_profile: &HardwareProfile,
    candidates: &[AutotuneCandidate],
    verbose: bool,
) -> Result<Vec<MeasuredCandidate>, TuneFailure> {
    let probe = GemmProbe::new(m, n, k)?;
    if verbose {
        let working_set = (m as usize * k as usize * 2)
            + (k as usize * n as usize * 2) * probe.b.len();
        println!(
            "         [Y autotuner] measuring on {} - M={} N={} K={} | {} weight buffer(s), \
             {:.1} MB working set vs {:.1} MB L2 ({})",
            probe.ctx.device_name(),
            m, n, k,
            probe.b.len(),
            working_set as f64 / 1e6,
            probe.l2_bytes as f64 / 1e6,
            // An unknown L2 size is reported as unknown. Calling a 33.6 MB
            // working set "DRAM-streaming" because the L2 query returned 0
            // is exactly the kind of confidently wrong label this harness
            // exists to avoid.
            if probe.l2_bytes == 0 {
                "L2 size unknown - rotation disabled"
            } else if working_set > probe.l2_bytes {
                "DRAM-streaming"
            } else {
                "L2-resident"
            },
        );
    }

    // ---- 1. compile every candidate, dedup identical codegen ----
    //
    // Candidates that differ only in requested `num_stages` frequently emit
    // byte-identical PTX, because the emitter clamps the request against the
    // real shared-memory budget and against `k_tiles`. Measuring the same
    // kernel three times under three different labels would waste most of
    // the tuning budget and make the ranking look more resolved than it is.
    let mut compiled: Vec<Compiled> = Vec::new();
    let mut seen_ptx: Vec<u64> = Vec::new();
    let mut rejects: Vec<String> = Vec::new();

    for cand in candidates {
        if !is_emittable(cand, k) {
            continue;
        }
        let (ptx, launch) = match emit_candidate_ptx(m, n, k, cand, hw_profile) {
            Ok(v) => v,
            Err(e) => {
                rejects.push(format!("{} -> codegen: {}", describe(cand), e));
                continue;
            }
        };

        let digest = fnv1a64(ptx.as_bytes());
        if seen_ptx.contains(&digest) {
            continue;
        }

        let kernel = match probe.ctx.load_ptx(&ptx, PROBE_KERNEL) {
            Ok(kmod) => kmod,
            Err(e) => {
                rejects.push(format!("{} -> JIT: {}", describe(cand), e));
                continue;
            }
        };
        if launch.dyn_smem_bytes > 0 {
            if let Err(e) = kernel.set_max_dynamic_smem(launch.dyn_smem_bytes) {
                rejects.push(format!("{} -> smem opt-in: {}", describe(cand), e));
                continue;
            }
        }

        seen_ptx.push(digest);
        compiled.push(Compiled { candidate: cand.clone(), launch, kernel });
    }

    if compiled.is_empty() {
        return Err(TuneFailure::NoUsableCandidate(
            rejects.first().cloned().unwrap_or_else(|| "candidate list was empty".into()),
        ));
    }

    // ---- 2. correctness gate, before any timing is believed ----
    let mut verified: Vec<Compiled> = Vec::new();
    for c in compiled {
        match probe.correctness_rel_l2(&c.kernel, &c.launch) {
            Ok(err) if err <= CORRECTNESS_REL_L2_TOL => verified.push(c),
            Ok(err) => rejects.push(format!(
                "{} -> INCORRECT (relative L2 error {:.3e} > {:.0e})",
                describe(&c.candidate),
                err,
                CORRECTNESS_REL_L2_TOL
            )),
            Err(e) => rejects.push(format!("{} -> correctness check: {}", describe(&c.candidate), e)),
        }
    }
    if verified.is_empty() {
        return Err(TuneFailure::NoUsableCandidate(
            rejects.first().cloned().unwrap_or_else(|| "every candidate failed".into()),
        ));
    }

    // ---- 3. clock ramp ----
    //
    // Runs a real candidate rather than a synthetic load so the ramp warms
    // the same units the measurement will use.
    let ramp_target = &verified[0];
    let ramp_iters = probe
        .calibrate_iters(&ramp_target.kernel, &ramp_target.launch, 10_000.0)
        .map_err(TuneFailure::Cuda)?;
    let ramp_start = Instant::now();
    while ramp_start.elapsed().as_secs_f64() < RAMP_SECONDS {
        probe
            .time(&ramp_target.kernel, &ramp_target.launch, ramp_iters)
            .map_err(TuneFailure::Cuda)?;
    }

    // ---- 4. screening pass over everything ----
    let all: Vec<usize> = (0..verified.len()).collect();
    let screen = measure_round_robin(&probe, &verified, &all, SCREEN_ROUNDS, SCREEN_BATCH_US)
        .map_err(TuneFailure::Cuda)?;
    let mut order: Vec<usize> = all.clone();
    order.sort_by(|&a, &b| {
        screen[a]
            .best_us
            .partial_cmp(&screen[b].best_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ---- 5. final pass over the survivors, more rounds, longer batches ----
    let finalists: Vec<usize> = order.iter().take(FINALISTS).copied().collect();
    let final_timing = measure_round_robin(&probe, &verified, &finalists, FINAL_ROUNDS, FINAL_BATCH_US)
        .map_err(TuneFailure::Cuda)?;

    // Finalists carry their re-measured time; anything screened out keeps its
    // (coarser) screening time, which is only ever used for reporting - it
    // was already slower than every finalist under identical conditions.
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let mut times: Vec<Timing> = screen.clone();
    let mut is_finalist = vec![false; verified.len()];
    for (rank, &i) in finalists.iter().enumerate() {
        times[i] = final_timing[rank];
        is_finalist[i] = true;
    }
    let mut results: Vec<MeasuredCandidate> = verified
        .iter()
        .enumerate()
        .map(|(i, c)| MeasuredCandidate {
            candidate: c.candidate.clone(),
            timing: times[i],
            tflops: flops / (times[i].best_us * 1e-6) / 1e12,
            finalist: is_finalist[i],
        })
        .collect();
    results.sort_by(|a, b| a.us().partial_cmp(&b.us()).unwrap_or(std::cmp::Ordering::Equal));

    let winner = select_winner(&results);

    if verbose {
        println!(
            "         [Y autotuner] {} distinct kernels compiled, {} verified, {} rejected",
            seen_ptx.len(),
            verified.len(),
            rejects.len()
        );
        for (i, r) in results.iter().take(FINALISTS).enumerate() {
            println!(
                "           {}. {:<20} {:>8.2} us (median {:.2}, worst {:.2})  {:>6.1} TFLOPS{}",
                i + 1,
                describe(&r.candidate),
                r.timing.best_us,
                r.timing.median_us,
                r.timing.max_us,
                r.tflops,
                if i == winner { "   <- selected" } else { "" }
            );
        }
        // Say so when the pick is not actually resolved. A sub-1% gap is
        // inside this harness's own reproducibility, and reporting it as a
        // winner would be reporting noise as a result.
        if winner != 0 {
            println!(
                "           note: #1..#{} are inside this run's own noise floor \
                 (+/-{:.2} us); selected #{} on warp count then K depth, not on the measured gap",
                results.iter().filter(|r| r.timing.tied_with(&results[0].timing)).count(),
                results[0].timing.median_us - results[0].timing.best_us,
                winner + 1
            );
        } else if results.len() >= 2 && results[0].timing.tied_with(&results[1].timing) {
            println!(
                "           note: #1 and #2 are inside this run's own noise floor \
                 (+/-{:.2} us); #1 also wins the structural tie-break",
                results[0].timing.median_us - results[0].timing.best_us
            );
        }
        for rej in rejects.iter().take(3) {
            println!("           rejected: {}", rej);
        }
    }

    // Winner first, so callers can keep taking `[0]`.
    results.swap(0, winner);
    Ok(results)
}

/// Times every candidate `rounds` times, interleaved, with the starting
/// position rotated each round.
///
/// Interleaving is what makes the comparison valid: measuring all of A then
/// all of B gives B a systematically hotter clock, which is a bias, not
/// noise, and no amount of repetition removes it. The per-round rotation
/// additionally cancels any residual "first one measured in a round is
/// slower" effect. The median across rounds, not the mean, is what gets
/// reported - a single scheduler hiccup should not decide a ranking.
/// `subset` indexes into `candidates`; the returned times are parallel to
/// `subset`, not to `candidates`.
fn measure_round_robin(
    probe: &GemmProbe,
    candidates: &[Compiled],
    subset: &[usize],
    rounds: u32,
    target_batch_us: f64,
) -> Result<Vec<Timing>, String> {
    let mut iters = Vec::with_capacity(subset.len());
    for &i in subset {
        iters.push(probe.calibrate_iters(&candidates[i].kernel, &candidates[i].launch, target_batch_us)?);
    }

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds as usize); subset.len()];
    for round in 0..rounds {
        for offset in 0..subset.len() {
            let slot = (offset + round as usize) % subset.len();
            let c = &candidates[subset[slot]];
            let us = probe.time(&c.kernel, &c.launch, iters[slot])?;
            samples[slot].push(us);
        }
    }

    Ok(samples
        .into_iter()
        .map(|mut s| {
            let best_us = s.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_us = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Timing { best_us, median_us: median(&mut s), max_us }
        })
        .collect())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        f64::INFINITY
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Picks the winner from a time-sorted ranking, breaking statistical ties on
/// structure rather than on noise. Returns an index into `results`.
///
/// At the DRAM-bandwidth roofline every candidate is limited by the same
/// wall, so they genuinely tie: measured at M=1, N=K=4096 streaming a 201 MB
/// working set, the whole top five spans under 5% while the SAME candidate
/// varies by 4% across independent runs (see `Timing::tied_with`). Taking
/// the fastest there is taking whichever candidate the noise happened to
/// favour. Two structural preferences break those ties instead, both
/// grounded in hardware rather than fitted to data:
///
///   1. FOUR WARPS PER CTA, preferring whichever candidate is closest to it
///      from either side. An SM has four warp schedulers: a one-warp CTA can
///      only keep them fed if several CTAs happen to be co-resident, and past
///      four the extra warps add `bar.sync` participants without adding
///      schedulers. Both directions are measured on this hardware. Below
///      four: without this preference, M=1/4/8 selected a 32-thread
///      `16x32x32 1x1` CTA on a 0.4% margin. Above four, at a fixed
///      128x128x32 tile, 2x2 warps runs 75.8 TFLOPS against 70.5 for 4x2 and
///      52.5 for 4x4 (ncu barrier-stall ratio 6.27 for the 512-thread config
///      vs 1.27 for cuBLAS's 128-thread one) - and nVidia's own kernel for
///      this shape is likewise 128 threads.
///
///      This preference must peak at four rather than saturate there. An
///      earlier revision used `num_warps.min(4)`, which scores a 4-warp and
///      an 8-warp CTA identically and so handed the decision to preference 2
///      below; at M=N=K=4096 that selected `128x128x64 4x2 s4` over
///      `128x128x32 2x2 s2`, which a clean interleaved A/B then measured at
///      **0.88x** (70.7 vs 80.0 TFLOPS).
///   2. DEEPER K TILE. A wider `cta_k` puts more bytes in flight per CTA per
///      `cp.async` stage, which is the axis that matters in exactly this
///      DRAM-bound regime - already established by measurement in this repo
///      (`generate_candidates`: 16x64x32 -> 16x64x64 went 97.9us -> 76.8us).
///
/// Remaining ties fall through to the measured time and then to the tile
/// dimensions, so the selection is fully deterministic and a re-tune of an
/// unchanged machine cannot flip the persisted answer.
pub fn select_winner(results: &[MeasuredCandidate]) -> usize {
    if results.is_empty() {
        return 0;
    }
    let best = results[0].timing;
    let mut winner = 0usize;
    for (i, r) in results.iter().enumerate().skip(1) {
        if !r.timing.tied_with(&best) {
            continue;
        }
        let key = |c: &MeasuredCandidate| {
            (
                // Peaks at four warps; negated distance so nearer is larger
                // under `max`, from both directions. See doc comment.
                -((c.candidate.num_warps as i64) - 4).abs(),
                c.candidate.cta_k,
                // Negated so that *smaller* is better under `max`.
                -(c.us() * 1000.0) as i64,
                c.candidate.cta_m,
                c.candidate.cta_n,
            )
        };
        if key(r) > key(&results[winner]) {
            winner = i;
        }
    }
    winner
}

pub fn describe(c: &AutotuneCandidate) -> String {
    format!(
        "{}x{}x{} {}x{} s{}",
        c.cta_m, c.cta_n, c.cta_k, c.warps_m, c.warps_n, c.num_stages
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(cta_m: u32, cta_n: u32, cta_k: u32, wm: u32, wn: u32, stages: u32) -> AutotuneCandidate {
        AutotuneCandidate {
            cta_m,
            cta_n,
            cta_k,
            warps_m: wm,
            warps_n: wn,
            num_stages: stages,
            num_warps: wm * wn,
        }
    }

    #[test]
    fn emittability_matches_the_emitters_own_contract() {
        // 128x128x32 over 2x2 warps: 64x64 per warp, both multiples of 16.
        assert!(is_emittable(&cand(128, 128, 32, 2, 2, 2), 4096));
        // `cta_k` need NOT divide K: the emitter has a real K tail (`k_tiles`
        // rounds up, partial tile zero-filled). This case used to be rejected,
        // which filtered out EVERY candidate at ragged shapes and silently
        // demoted them to the analytic heuristic - see `is_emittable`'s note.
        assert!(is_emittable(&cand(128, 128, 128, 2, 2, 2), 4096 + 32));
        // Ragged shapes must reach the measurement loop at all.
        assert!(is_emittable(&cand(128, 128, 32, 2, 2, 2), 1000));
        // K must still be a multiple of 8 - A's K tail is masked at 16-byte
        // (8-element) cp.async chunk granularity.
        assert!(!is_emittable(&cand(128, 128, 32, 2, 2, 2), 1007));
        // per-warp tile below one 16x16 MMA fragment.
        assert!(!is_emittable(&cand(16, 16, 32, 2, 2, 2), 4096));
        // cta_n not evenly splittable into 16-wide fragments per warp column.
        assert!(!is_emittable(&cand(64, 24, 32, 2, 2, 2), 4096));
    }

    #[test]
    fn launch_config_is_parsed_from_the_emitter_comment() {
        let ptx = "\
.version 8.4
    // [Y TENSOR CORE GEMM] M=1024 N=1024 K=1024 | CTA 64x128x32 | 2x2 warps | wmma.sync.m16n16k16.f16->f32
    // Autotuner selected 3 pipeline stages (3 used after k_tiles/smem clamping); cp.async multi-stage pipelined. Dynamic shared memory required: 41472 bytes.
";
        let cfg = parse_launch_config(ptx, 1024, 1024).expect("should parse");
        assert_eq!(cfg.grid_x, 8); // 1024 / 128
        assert_eq!(cfg.grid_y, 16); // 1024 / 64
        assert_eq!(cfg.threads, 128); // 2*2 warps
        assert_eq!(cfg.dyn_smem_bytes, 41472);
    }

    #[test]
    fn launch_config_reports_zero_smem_on_the_synchronous_fallback() {
        // The K-too-small fallback emits no "Dynamic shared memory required"
        // clause at all; 0 is the right answer, not a parse failure.
        let ptx = "    // [Y TENSOR CORE GEMM] M=64 N=64 K=32 | CTA 32x32x32 | 2x1 warps | wmma\n";
        let cfg = parse_launch_config(ptx, 64, 64).expect("should parse");
        assert_eq!(cfg.dyn_smem_bytes, 0);
        assert_eq!(cfg.threads, 64);
    }

    #[test]
    fn missing_gemm_comment_is_an_error_not_a_silent_default() {
        assert!(parse_launch_config(".version 8.4\n", 64, 64).is_err());
    }

    #[test]
    fn probe_source_is_shaped_like_the_shipping_gemm_tests() {
        let src = probe_source(1024, 2048, 512);
        assert!(src.contains("@tile(1024, 2048, 512)"));
        assert!(src.contains("A: GlobalMemory<F16>"));
        assert!(src.contains("C: GlobalMemory<F32>"));
    }

    #[test]
    fn weight_replication_defeats_l2_only_where_it_has_to() {
        const L2: usize = 50_331_648; // 48 MB, this project's dev GPU
        const PLENTY: usize = 14 << 30;

        // Decode shape: B is 33.55 MB, well inside L2, so it must be
        // replicated to ~4x L2 (the same convention as
        // tests/benchmark_y_decode_gemm.py's NUM_WEIGHT_BUFFERS = 6).
        let b = 4096 * 4096 * 2;
        let n = plan_weight_replicas(b, 1 * 4096 * 2, 1 * 4096 * 4, L2, PLENTY);
        assert!(n >= 6, "decode weights must oversubscribe L2, got {} replicas", n);
        assert!(b * n >= L2 * L2_OVERSUBSCRIBE);

        // M=N=K=16384: B alone is 512 MB, already 10x L2. Replicating it
        // would cost gigabytes and buy nothing.
        let big = 16384 * 16384 * 2;
        assert_eq!(plan_weight_replicas(big, big, 16384 * 16384 * 4, L2, PLENTY), 1);

        // Tight on memory: correctness of the measurement beats defeating
        // L2, so it must fall back to a single buffer rather than fail to
        // allocate.
        assert_eq!(plan_weight_replicas(b, 0, 0, L2, 40 << 20), 1);

        // No device L2 figure available - do not guess.
        assert_eq!(plan_weight_replicas(b, 0, 0, 0, PLENTY), 1);
    }

    /// `best_us` / `median_us` pairs below are real numbers copied out of
    /// forced remeasurements on this project's dev GPU, not invented ones -
    /// the tie band is derived from the best/median dispersion, so a
    /// synthetic `median == best` would test a resolution the harness never
    /// actually has.
    fn measured(c: AutotuneCandidate, best_us: f64, median_us: f64) -> MeasuredCandidate {
        MeasuredCandidate {
            candidate: c,
            timing: Timing { best_us, median_us, max_us: median_us * 1.05 },
            tflops: 0.0,
            finalist: true,
        }
    }

    #[test]
    fn tie_break_prefers_warps_then_k_depth_but_never_overrides_a_real_gap() {
        // Real M=1, N=K=4096 ranking (201 MB working set, DRAM-streaming).
        // A 32-thread CTA led by 1.0%, well inside the ~1.3% dispersion of
        // its own rounds - and across four runs that same candidate moved by
        // 4%. Structure breaks it.
        let results = vec![
            measured(cand(16, 32, 32, 1, 1, 4), 56.10, 56.81),
            measured(cand(32, 64, 64, 2, 2, 3), 56.64, 57.08),
            measured(cand(32, 64, 32, 2, 2, 4), 56.91, 59.59),
        ];
        let w = select_winner(&results);
        assert_eq!(
            results[w].candidate.num_warps, 4,
            "must not select a 1-warp CTA on a margin smaller than its own noise"
        );
        assert_eq!(results[w].candidate.cta_k, 64);

        // Real M=N=K=512 ranking, compute-bound and L2-resident: a 2.6% gap
        // against a 0.08us dispersion is a result, and structure must not
        // override it.
        let results = vec![
            measured(cand(64, 64, 64, 2, 2, 3), 9.72, 9.80),
            measured(cand(128, 128, 32, 4, 4, 2), 9.97, 9.98),
        ];
        assert_eq!(
            select_winner(&results), 0,
            "a gap several times the noise floor must not be overridden by structure"
        );

        // All else equal, the deeper K tile wins - the axis measurement has
        // already shown dominates in the DRAM-bound regime.
        let results = vec![
            measured(cand(32, 64, 32, 2, 2, 4), 56.53, 58.00),
            measured(cand(32, 64, 64, 2, 2, 3), 56.60, 58.10),
        ];
        assert_eq!(results[select_winner(&results)].candidate.cta_k, 64);
    }

    #[test]
    fn tie_break_is_deterministic_for_identical_structure() {
        // Real M=N=K=1024 top two: 0.46% apart, both 4-warp k=32.
        let build = || {
            vec![
                measured(cand(64, 128, 32, 2, 2, 3), 34.44, 35.13),
                measured(cand(128, 64, 32, 2, 2, 3), 34.60, 35.20),
            ]
        };
        let (a, b) = (build(), build());
        // Same input, same answer - a re-tune of an unchanged machine must
        // not flip the persisted tile.
        assert_eq!(a[select_winner(&a)].candidate, b[select_winner(&b)].candidate);
    }

    #[test]
    fn tie_band_tracks_the_regimes_noise_floor_not_a_fixed_percentage() {
        // Compute-bound: tiny dispersion, so a 2.6% gap is decisive.
        let quiet_a = Timing { best_us: 9.72, median_us: 9.80, max_us: 10.21 };
        let quiet_b = Timing { best_us: 9.97, median_us: 9.98, max_us: 10.47 };
        assert!(!quiet_a.tied_with(&quiet_b));

        // DRAM-bound: same 2.6% relative gap, but the dispersion is 20x
        // larger, so it is not separable.
        let noisy_a = Timing { best_us: 56.10, median_us: 57.71, max_us: 62.31 };
        let noisy_b = Timing { best_us: 57.56, median_us: 59.00, max_us: 63.76 };
        assert!(noisy_a.tied_with(&noisy_b));
    }

    #[test]
    fn median_is_order_independent() {
        let mut a = [3.0, 1.0, 2.0];
        let mut b = [2.0, 3.0, 1.0];
        assert_eq!(median(&mut a), median(&mut b));
        assert_eq!(median(&mut a), 2.0);
    }
}
