// ============================================================
//  Y — Autotuner Empirical Verification Harness
//  autotune_verify.rs
//
//  Autotuner::score_candidate (src/autotuner.rs) predicts occupancy and
//  ranks candidates analytically, without compiling anything. This harness
//  applies the same "verify, don't trust the model" discipline used to tune
//  tests/y_tensor_core_gemm.cu's 256x128 kernel: it actually compiles a
//  handful of the autotuner's top- and bottom-ranked candidates for a given
//  (M,N,K) via `nvcc -Xptxas -v` (real register/spill counts) and real,
//  warmed-up wall-clock timing on the GPU present in this machine, then
//  reports whether the predicted ranking matches what was measured.
//
//  This is NOT the hand-tuned tests/y_tensor_core_gemm.cu kernel - that
//  kernel's cp.async/ldmatrix/swizzle machinery is specific to one fixed tile
//  shape and isn't safely genericizable to arbitrary candidate parameters
//  without risking silent correctness bugs. Instead this uses a simpler,
//  correctness-first synchronous-pipelined WMMA GEMM, parameterized the same
//  way Autotuner's candidates are (cta_m/cta_n/cta_k/warps_m/warps_n/
//  num_stages). Its absolute register counts run lower than the hand-tuned
//  kernel's (less swizzle/index bookkeeping) - this checks relative
//  scaling/ranking trends against the model's predictions, not absolute
//  register-count parity with tests/y_tensor_core_gemm.cu.
//
//  It also runs an A/B: the same real compile-and-measure pipeline against
//  `old_score_candidate` below, a frozen copy of `score_candidate` as it
//  existed before the occupancy-based rework (hardcoded `stage_multiplier`
//  treating 4 pipeline stages as unconditionally better than 2, no
//  register/shared-memory occupancy model at all). This answers "does the
//  new scorer's #1 pick actually run faster than the old scorer's #1 pick"
//  with real, measured wall-clock numbers instead of re-asserting the model
//  is more correct.
//
//  Usage: cargo run --release --bin autotune_verify -- [M] [N] [K]
// ============================================================

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use y::autotuner::{AutotuneCandidate, Autotuner, Precision};
use y::sentinel::{check_or_probe_hardware, HardwareProfile};

const KERNEL_TEMPLATE: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace nvcuda;

#define BLOCK_M __BLOCK_M__
#define BLOCK_N __BLOCK_N__
#define BLOCK_K __BLOCK_K__
#define WARPS_M __WARPS_M__
#define WARPS_N __WARPS_N__
#define NUM_STAGES __NUM_STAGES__
#define THREADS (WARPS_M * WARPS_N * 32)
#define WM (BLOCK_M / WARPS_M)
#define WN (BLOCK_N / WARPS_N)
#define NUM_I (WM / 16)
#define NUM_J (WN / 16)
#define GM __GM__
#define GN __GN__
#define GK __GK__

extern "C" __global__ void gemm_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    extern __shared__ char smem_raw[];
    half (*smem_A)[BLOCK_M][BLOCK_K] = (half (*)[BLOCK_M][BLOCK_K]) smem_raw;
    half (*smem_B)[BLOCK_K][BLOCK_N] = (half (*)[BLOCK_K][BLOCK_N]) (smem_raw + (size_t)NUM_STAGES * BLOCK_M * BLOCK_K * sizeof(half));

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id % WARPS_M;
    int warp_n = warp_id / WARPS_M;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[NUM_I][NUM_J];
    #pragma unroll
    for (int i = 0; i < NUM_I; i++)
        #pragma unroll
        for (int j = 0; j < NUM_J; j++)
            wmma::fill_fragment(frag_c[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int stage = (k0 / BLOCK_K) % NUM_STAGES;
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += THREADS) {
            int r = idx / BLOCK_K, c = idx % BLOCK_K;
            smem_A[stage][r][c] = A[(size_t)(cta_m + r) * K + (k0 + c)];
        }
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += THREADS) {
            int r = idx / BLOCK_N, c = idx % BLOCK_N;
            smem_B[stage][r][c] = B[(size_t)(k0 + r) * N + (cta_n + c)];
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[NUM_I];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[NUM_J];
            #pragma unroll
            for (int i = 0; i < NUM_I; i++)
                wmma::load_matrix_sync(frag_a[i], &smem_A[stage][warp_m * WM + i * 16][kk], BLOCK_K);
            #pragma unroll
            for (int j = 0; j < NUM_J; j++)
                wmma::load_matrix_sync(frag_b[j], &smem_B[stage][kk][warp_n * WN + j * 16], BLOCK_N);
            #pragma unroll
            for (int i = 0; i < NUM_I; i++)
                #pragma unroll
                for (int j = 0; j < NUM_J; j++)
                    wmma::mma_sync(frag_c[i][j], frag_a[i], frag_b[j], frag_c[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NUM_I; i++)
        #pragma unroll
        for (int j = 0; j < NUM_J; j++) {
            int row = cta_m + warp_m * WM + i * 16;
            int col = cta_n + warp_n * WN + j * 16;
            wmma::store_matrix_sync(&C[(size_t)row * N + col], frag_c[i][j], N, wmma::mem_row_major);
        }
}

extern "C" __global__ void naive_ref_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += __half2float(A[(size_t)row * K + k]) * __half2float(B[(size_t)k * N + col]);
    }
    C[(size_t)row * N + col] = acc;
}

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA_ERROR=%s\n", cudaGetErrorString(e)); return 1; } } while (0)

int main() {
    int M = GM, N = GN, K = GK;
    size_t bytesA = (size_t)M * K * sizeof(half);
    size_t bytesB = (size_t)K * N * sizeof(half);
    size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<half> hA(M * K), hB(K * N);
    for (size_t i = 0; i < hA.size(); i++) hA[i] = __float2half(((int)(i % 7) - 3) * 0.1f);
    for (size_t i = 0; i < hB.size(); i++) hB[i] = __float2half(((int)(i % 5) - 2) * 0.1f);

    half *dA, *dB; float *dC, *dCref;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMalloc(&dCref, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    size_t smem_bytes = (size_t)NUM_STAGES * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(half);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(THREADS);

    // warmup
    for (int i = 0; i < 3; i++) gemm_kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int iters = 20;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) gemm_kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_ms = ms / iters;

    dim3 rblock(16, 16);
    dim3 rgrid((N + 15) / 16, (M + 15) / 16);
    naive_ref_kernel<<<rgrid, rblock>>>(dA, dB, dCref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hC(M * N), hCref(M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCref.data(), dCref, bytesC, cudaMemcpyDeviceToHost));

    double max_rel_err = 0.0;
    for (size_t i = 0; i < hC.size(); i += 37) {
        double ref = hCref[i];
        double got = hC[i];
        double denom = fabs(ref) > 1e-3 ? fabs(ref) : 1e-3;
        double rel_err = fabs(got - ref) / denom;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    printf("TIME_MS=%.5f\n", avg_ms);
    printf("MAX_REL_ERR=%.6f\n", max_rel_err);
    printf("CORRECT=%d\n", max_rel_err < 0.05 ? 1 : 0);
    return 0;
}
"#;

fn nvcc_arch_flag(hw: &HardwareProfile) -> String {
    let cc: String = hw
        .compute_capability
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect();
    if cc.len() >= 2 {
        format!("sm_{}", cc)
    } else {
        "sm_89".to_string()
    }
}

/// Frozen copy of `Autotuner::score_candidate` exactly as it existed before
/// this session's occupancy-based rework (verified against
/// `git show HEAD:src/autotuner.rs`, since the rework is uncommitted in this
/// working tree). No register/shared-memory/warp occupancy model at all -
/// `stage_multiplier` just hardcodes 4 stages > 3 stages > 2 stages
/// regardless of what that costs in occupancy, and `smem_penalty` is a flat
/// 0.85 no matter how far over budget a candidate's shared memory footprint
/// is (never rejects a candidate outright, even one that can't launch).
/// Lives here, not in src/autotuner.rs, purely so this harness can A/B it
/// against the current scorer on real hardware - must never be called from
/// production code.
fn old_score_candidate(candidate: &AutotuneCandidate, m: u32, n: u32, _k: u32, hw: &HardwareProfile) -> f64 {
    let num_tiles_m = (m + candidate.cta_m - 1) / candidate.cta_m;
    let num_tiles_n = (n + candidate.cta_n - 1) / candidate.cta_n;
    let total_ctas = num_tiles_m * num_tiles_n;

    let num_sms = if hw.sm_count > 0 { hw.sm_count } else { 66 };
    let ctas_per_sm = (total_ctas as f64 / num_sms as f64).ceil();

    // SM Wave Alignment Efficiency: penalize partial tail waves where many SMs stay idle
    let full_waves = total_ctas / num_sms;
    let remainder = total_ctas % num_sms;
    let wave_efficiency = if remainder == 0 {
        1.0
    } else if full_waves == 0 {
        (remainder as f64 / num_sms as f64).max(0.3)
    } else {
        (full_waves as f64 * num_sms as f64 + remainder as f64) / ((full_waves + 1) as f64 * num_sms as f64)
    };

    // Detect FP8 precision vs FP16 precision: FP8 loads 1 byte per element.
    // Deliberately NOT migrated to the new `candidate.precision` field - this
    // guess-from-cta_k heuristic (and its inaccuracy for any candidate where
    // cta_k>=64 wasn't actually FP8) is exactly the pre-fix behavior this
    // function exists to replicate for the A/B.
    let bytes_per_elem = if candidate.cta_k >= 64 { 1 } else { 2 };
    let load_bytes_a = candidate.cta_m * candidate.cta_k * bytes_per_elem;
    let load_bytes_b = candidate.cta_k * candidate.cta_n * bytes_per_elem;
    let total_load_bytes = (load_bytes_a + load_bytes_b) as f64;

    // Total shared memory required per CTA (in KB)
    let smem_kb = (total_load_bytes * candidate.num_stages as f64) / 1024.0;
    let smem_penalty = if smem_kb > 48.0 { 0.85 } else { 1.0 };

    // Pipeline stage latency hiding score
    let stage_multiplier = match candidate.num_stages {
        4 => 1.35, // 4-stage TMA / cp.async deep pipeline
        3 => 1.30, // 3-stage software pipeline hides L2/DRAM memory latency
        2 => 1.15, // 2-stage double buffering
        _ => 1.0,
    };

    // Grid occupancy factor: punish configurations that leave >30% of SMs idle
    let sm_occupancy = (total_ctas as f64 / (ctas_per_sm * num_sms as f64)).min(1.0);

    // Compute FLOPs per CTA tile iteration: 2 * M * N * K
    let flops_per_tile = 2.0 * (candidate.cta_m as f64) * (candidate.cta_n as f64) * (candidate.cta_k as f64);
    let compute_intensity = flops_per_tile / total_load_bytes;

    compute_intensity * stage_multiplier * sm_occupancy * wave_efficiency * smem_penalty * (1.0 / ctas_per_sm) * 1000.0
}

fn render_kernel(candidate: &AutotuneCandidate, m: u32, n: u32, k: u32) -> String {
    KERNEL_TEMPLATE
        .replace("__BLOCK_M__", &candidate.cta_m.to_string())
        .replace("__BLOCK_N__", &candidate.cta_n.to_string())
        .replace("__BLOCK_K__", &candidate.cta_k.to_string())
        .replace("__WARPS_M__", &candidate.warps_m.to_string())
        .replace("__WARPS_N__", &candidate.warps_n.to_string())
        .replace("__NUM_STAGES__", &candidate.num_stages.to_string())
        .replace("__GM__", &m.to_string())
        .replace("__GN__", &n.to_string())
        .replace("__GK__", &k.to_string())
}

/// Parses `nvcc -Xptxas -v` output for the real, compiled register and spill
/// counts - the ground truth `Autotuner::estimate_regs_per_thread` can only
/// approximate before compilation.
fn parse_ptxas_v(text: &str) -> (Option<u32>, Option<u32>) {
    let mut regs = None;
    let mut spill_bytes = None;
    for line in text.lines() {
        if let Some(idx) = line.find("Used ") {
            regs = line[idx + 5..]
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u32>().ok());
        }
        if line.contains("bytes spill stores") {
            let nums: Vec<u32> = line
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse::<u32>().ok())
                .collect();
            // Line shape: "N bytes stack frame, N bytes spill stores, N bytes spill loads"
            if nums.len() >= 3 {
                spill_bytes = Some(nums[1] + nums[2]);
            }
        }
    }
    (regs, spill_bytes)
}

#[derive(Debug, Default)]
struct MeasuredResult {
    compiled: bool,
    real_regs: Option<u32>,
    spill_bytes: Option<u32>,
    /// Median of `time_ms_samples` independent process launches - see
    /// `REPEAT_RUNS`'s doc comment for why a single launch isn't trustworthy.
    time_ms: Option<f64>,
    time_ms_min: Option<f64>,
    time_ms_max: Option<f64>,
    time_ms_samples: usize,
    correct: Option<bool>,
    error: Option<String>,
}

/// How many times to re-launch each compiled candidate's binary and take the
/// median TIME_MS. Not a knob for measurement precision within one process
/// (the kernel template already averages 20 warmed-up iterations internally)
/// - it exists because two back-to-back full harness runs of the IDENTICAL
/// 512x512x4096 candidate set, moments apart on the same GPU, produced
/// opposite A/B verdicts (one run: NEW's #1 pick measured 608% SLOWER than
/// OLD's; immediate re-run, same binaries: NEW's #1 pick measured 66%
/// FASTER) - almost certainly GPU clock/power-state transitions across
/// separate process launches, not anything about the kernels themselves.
/// A single sample per candidate cannot distinguish a real difference from
/// that noise floor.
const REPEAT_RUNS: usize = 5;

fn median(values: &mut Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn compile_and_run(
    candidate: &AutotuneCandidate,
    m: u32,
    n: u32,
    k: u32,
    arch: &str,
    workdir: &PathBuf,
    tag: usize,
) -> MeasuredResult {
    let src = render_kernel(candidate, m, n, k);
    let cu_path = workdir.join(format!("cand_{}.cu", tag));
    let exe_path = workdir.join(format!("cand_{}", tag));

    if let Err(e) = fs::write(&cu_path, &src) {
        return MeasuredResult {
            error: Some(format!("failed to write source: {}", e)),
            ..Default::default()
        };
    }

    let compile = Command::new("nvcc")
        .args(["-arch", arch, "-std=c++17", "-O3", "-Xptxas", "-v"])
        .arg(&cu_path)
        .arg("-o")
        .arg(&exe_path)
        .output();

    let compile = match compile {
        Ok(o) => o,
        Err(e) => {
            return MeasuredResult {
                error: Some(format!("failed to spawn nvcc: {}", e)),
                ..Default::default()
            }
        }
    };

    let compile_log = format!(
        "{}{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    if !compile.status.success() {
        return MeasuredResult {
            compiled: false,
            error: Some(compile_log),
            ..Default::default()
        };
    }

    let (real_regs, spill_bytes) = parse_ptxas_v(&compile_log);

    let mut time_samples: Vec<f64> = Vec::new();
    let mut saw_incorrect = false;
    let mut last_run_error: Option<String> = None;
    let mut any_run_succeeded = false;

    for _ in 0..REPEAT_RUNS {
        let run = match Command::new(&exe_path).output() {
            Ok(o) => o,
            Err(e) => {
                last_run_error = Some(format!("failed to run candidate binary: {}", e));
                continue;
            }
        };

        let stdout = String::from_utf8_lossy(&run.stdout);
        let mut this_time_ms = None;
        let mut this_correct = None;
        for line in stdout.lines() {
            if let Some(v) = line.strip_prefix("TIME_MS=") {
                this_time_ms = v.trim().parse::<f64>().ok();
            }
            if let Some(v) = line.strip_prefix("CORRECT=") {
                this_correct = v.trim().parse::<u32>().ok().map(|x| x == 1);
            }
        }

        if !run.status.success() {
            // The kernel template's CUDA_CHECK macro prints its error (e.g.
            // "CUDA_ERROR=invalid argument" from a too-large
            // cudaFuncSetAttribute smem request) to stdout via printf, not
            // stderr - capture both or this diagnostic is silently empty.
            last_run_error = Some(format!("{}{}", stdout, String::from_utf8_lossy(&run.stderr)));
            continue;
        }
        any_run_succeeded = true;
        if this_correct == Some(false) {
            saw_incorrect = true;
        }
        if let Some(t) = this_time_ms {
            time_samples.push(t);
        }
    }

    let time_ms_samples = time_samples.len();
    let time_ms_min = time_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let time_ms_max = time_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_ms = if time_samples.is_empty() {
        None
    } else {
        Some(median(&mut time_samples))
    };

    MeasuredResult {
        compiled: true,
        real_regs,
        spill_bytes,
        time_ms,
        time_ms_min: if time_ms_min.is_finite() { Some(time_ms_min) } else { None },
        time_ms_max: if time_ms_max.is_finite() { Some(time_ms_max) } else { None },
        time_ms_samples,
        // Correctness is checked with fixed, deterministic inputs each run -
        // it should never flip between runs. Require every successful run to
        // report correct=true; any run reporting incorrect makes the whole
        // candidate suspect rather than averaging it away.
        correct: if !any_run_succeeded {
            None
        } else {
            Some(!saw_incorrect)
        },
        error: if any_run_succeeded {
            None
        } else {
            last_run_error
        },
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let m: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let n: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let k: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2048);

    println!("=== Autotuner Empirical Verification Harness (OLD vs NEW scoring A/B) ===");
    println!("Probing real hardware (reusing the same sentinel probe the compiler uses)...");
    let hw = check_or_probe_hardware();
    let arch = nvcc_arch_flag(&hw);
    println!(
        "GPU: {} | arch={} | M={} N={} K={}\n",
        hw.gpu_name, arch, m, n, k
    );

    // This harness's kernel template always uses `half` (see KERNEL_TEMPLATE),
    // so every candidate is generated and scored as real FP16 - no more
    // guessing precision from cta_k (that heuristic is gone from
    // score_candidate; see Precision in src/autotuner.rs). This used to
    // require filtering out cta_k>=64 candidates to avoid an apples-to-oranges
    // FP8-vs-FP16 mismatch; with precision explicit and correct, the full
    // candidate pool is comparable now.
    let mut candidates = Autotuner::generate_candidates(m, n, k, Precision::F16);
    candidates.dedup();

    let mut scored: Vec<(f64, AutotuneCandidate)> = candidates
        .iter()
        .cloned()
        .map(|c| {
            let s = Autotuner::score_candidate(&c, m, n, k, &hw);
            (s, c)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut old_scored: Vec<(f64, AutotuneCandidate)> = candidates
        .into_iter()
        .map(|c| {
            let s = old_score_candidate(&c, m, n, k, &hw);
            (s, c)
        })
        .collect();
    old_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!(
        "Current (NEW, occupancy-based) predicted ranking ({} candidates, all FP16):",
        scored.len()
    );
    for (i, (score, cand)) in scored.iter().enumerate() {
        let occ = Autotuner::estimate_occupancy(cand, &hw);
        println!(
            "  #{:<2} score={:>9.2}  {}x{}x{} warps={}x{} stages={}  predicted: {:.0} regs/thread, {:.0} blocks/SM ({:.1}%)",
            i + 1,
            score,
            cand.cta_m,
            cand.cta_n,
            cand.cta_k,
            cand.warps_m,
            cand.warps_n,
            cand.num_stages,
            occ.est_regs_per_thread,
            occ.achieved_blocks_per_sm,
            occ.achieved_occupancy * 100.0
        );
    }

    println!(
        "\nPre-fix (OLD, hardcoded stage_multiplier) predicted ranking ({} candidates, all FP16):",
        old_scored.len()
    );
    for (i, (score, cand)) in old_scored.iter().enumerate() {
        println!(
            "  #{:<2} score={:>9.2}  {}x{}x{} warps={}x{} stages={}",
            i + 1,
            score,
            cand.cta_m,
            cand.cta_n,
            cand.cta_k,
            cand.warps_m,
            cand.warps_n,
            cand.num_stages
        );
    }

    // Compile+measure the union of: NEW's top 3, OLD's top 3 (this is the
    // actual A/B - what each scorer would have told you to launch), plus
    // NEW's bottom 2 (still hardware-fitting) for contrast.
    let mut selected: Vec<AutotuneCandidate> = Vec::new();
    for (_, c) in scored.iter().take(3) {
        if !selected.contains(c) {
            selected.push(c.clone());
        }
    }
    for (_, c) in old_scored.iter().take(3) {
        if !selected.contains(c) {
            selected.push(c.clone());
        }
    }
    let bottom_start = scored.len().saturating_sub(2);
    for (_, c) in scored[bottom_start..].iter() {
        if !selected.contains(c) {
            selected.push(c.clone());
        }
    }

    let workdir = env::temp_dir().join("autotune_verify_harness");
    if let Err(e) = fs::create_dir_all(&workdir) {
        eprintln!("failed to create work dir {:?}: {}", workdir, e);
        std::process::exit(1);
    }

    println!(
        "\nCompiling and measuring {} selected candidates on real hardware (this takes a bit)...",
        selected.len()
    );

    fn rank_of(ranking: &[(f64, AutotuneCandidate)], c: &AutotuneCandidate) -> (usize, f64) {
        ranking
            .iter()
            .position(|(_, rc)| rc == c)
            .map(|i| (i, ranking[i].0))
            .expect("selected candidate must be present in both rankings (same source list)")
    }

    let mut results: Vec<(AutotuneCandidate, usize, f64, usize, f64, MeasuredResult)> = Vec::new();
    for (tag, cand) in selected.iter().enumerate() {
        let (new_rank, new_score) = rank_of(&scored, cand);
        let (old_rank, old_score) = rank_of(&old_scored, cand);
        print!(
            "  [NEW #{:<2} / OLD #{:<2}] {}x{}x{}/warps{}x{}/stages{} ... ",
            new_rank + 1,
            old_rank + 1,
            cand.cta_m,
            cand.cta_n,
            cand.cta_k,
            cand.warps_m,
            cand.warps_n,
            cand.num_stages
        );
        std::io::stdout().flush().ok();
        let res = compile_and_run(cand, m, n, k, &arch, &workdir, tag);
        if !res.compiled {
            println!("COMPILE FAILED");
        } else if let Some(t) = res.time_ms {
            println!(
                "{:.4} ms median [{:.4}-{:.4} over {} runs], {} real regs, correct={:?}",
                t,
                res.time_ms_min.unwrap_or(t),
                res.time_ms_max.unwrap_or(t),
                res.time_ms_samples,
                res.real_regs.map(|r| r.to_string()).unwrap_or("?".into()),
                res.correct
            );
        } else {
            println!("compiled but did not report timing (run failure?)");
        }
        results.push((cand.clone(), new_rank, new_score, old_rank, old_score, res));
    }

    println!("\n=== Predicted (NEW vs OLD) vs Measured ===");
    println!(
        "{:<10} {:<10} {:<24} {:>10} {:>10} {:>9} {:>8} {:>10}",
        "NEWRank", "OLDRank", "Tile/warps/stages", "NEWScore", "OLDScore", "RealRegs", "Spills", "TimeMs"
    );
    for (cand, new_rank, new_score, old_rank, old_score, res) in &results {
        println!(
            "{:<10} {:<10} {:<24} {:>10.2} {:>10.2} {:>9} {:>8} {:>10}",
            format!("#{}", new_rank + 1),
            format!("#{}", old_rank + 1),
            format!(
                "{}x{}x{}/{}x{}/{}",
                cand.cta_m, cand.cta_n, cand.cta_k, cand.warps_m, cand.warps_n, cand.num_stages
            ),
            new_score,
            old_score,
            res.real_regs
                .map(|r| r.to_string())
                .unwrap_or_else(|| "-".into()),
            res.spill_bytes
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".into()),
            res.time_ms
                .map(|t| format!("{:.4}", t))
                .unwrap_or_else(|| "FAILED".into()),
        );
    }

    // ---- The actual A/B: what NEW's #1 pick measured vs what OLD's #1 pick measured ----
    let new_pick = scored[0].1.clone();
    let old_pick = old_scored[0].1.clone();
    let find_res = |c: &AutotuneCandidate| results.iter().find(|(rc, ..)| rc == c);

    println!("\n=== A/B: OLD scoring's #1 pick vs NEW scoring's #1 pick (real hardware) ===");
    println!(
        "OLD picks: {}x{}x{}/warps{}x{}/stages{}",
        old_pick.cta_m, old_pick.cta_n, old_pick.cta_k, old_pick.warps_m, old_pick.warps_n, old_pick.num_stages
    );
    println!(
        "NEW picks: {}x{}x{}/warps{}x{}/stages{}",
        new_pick.cta_m, new_pick.cta_n, new_pick.cta_k, new_pick.warps_m, new_pick.warps_n, new_pick.num_stages
    );
    if old_pick == new_pick {
        println!("OLD and NEW agree on the #1 pick for this (M,N,K) - no scoring-driven difference to measure here.");
    } else {
        match (find_res(&old_pick), find_res(&new_pick)) {
            (Some((_, _, _, _, _, old_res)), Some((_, _, _, _, _, new_res))) => {
                match (old_res.time_ms, new_res.time_ms, old_res.correct, new_res.correct) {
                    (Some(ot), Some(nt), Some(true), Some(true)) => {
                        let delta_pct = (ot - nt) / ot * 100.0;
                        println!(
                            "OLD pick measured: {:.4} ms median [{:.4}-{:.4} over {} runs]",
                            ot,
                            old_res.time_ms_min.unwrap_or(ot),
                            old_res.time_ms_max.unwrap_or(ot),
                            old_res.time_ms_samples
                        );
                        println!(
                            "NEW pick measured: {:.4} ms median [{:.4}-{:.4} over {} runs]",
                            nt,
                            new_res.time_ms_min.unwrap_or(nt),
                            new_res.time_ms_max.unwrap_or(nt),
                            new_res.time_ms_samples
                        );
                        let ranges_overlap = old_res.time_ms_min.unwrap_or(ot) <= new_res.time_ms_max.unwrap_or(nt)
                            && new_res.time_ms_min.unwrap_or(nt) <= old_res.time_ms_max.unwrap_or(ot);
                        if ranges_overlap {
                            println!(
                                "RESULT: INCONCLUSIVE - OLD's and NEW's measured ranges overlap ({:.1}% median delta \
                                 is within run-to-run noise; see REPEAT_RUNS doc comment). Not a confident win either way.",
                                delta_pct.abs()
                            );
                        } else if nt < ot {
                            println!(
                                "RESULT: NEW's pick is {:.1}% FASTER than OLD's pick on real hardware for M={} N={} K={} \
                                 (ranges do not overlap).",
                                delta_pct, m, n, k
                            );
                        } else {
                            println!(
                                "RESULT: NEW's pick is {:.1}% SLOWER than OLD's pick on real hardware for M={} N={} K={} \
                                 (ranges do not overlap) - the occupancy model did not win this comparison.",
                                -delta_pct, m, n, k
                            );
                        }
                    }
                    _ => println!(
                        "RESULT: inconclusive - one or both picks failed to compile/run/verify correctly \
                         (see diagnostics below)."
                    ),
                }
            }
            _ => println!("Internal error: OLD or NEW #1 pick was not in the compiled set (should be unreachable)."),
        }
    }

    let measured_best = results
        .iter()
        .filter(|(_, _, _, _, _, res)| res.time_ms.is_some() && res.correct == Some(true))
        .min_by(|a, b| a.5.time_ms.unwrap().partial_cmp(&b.5.time_ms.unwrap()).unwrap());

    match measured_best {
        Some((best_cand, new_rank, _, old_rank, _, best_res)) => {
            println!(
                "\nFastest measured candidate overall: NEW-rank #{} / OLD-rank #{} ({}x{}x{}/stages{}, {:.4}ms)",
                *new_rank + 1,
                *old_rank + 1,
                best_cand.cta_m,
                best_cand.cta_n,
                best_cand.cta_k,
                best_cand.num_stages,
                best_res.time_ms.unwrap()
            );
            if *new_rank == 0 {
                println!("MATCH: NEW's predicted #1 candidate was also the fastest one measured.");
            } else {
                println!(
                    "MISMATCH: NEW's predicted #1 was NOT the fastest measured candidate among those tested \
                     (this can mean the model needs more work, OR that this harness's simplified \
                     kernel doesn't capture an effect - e.g. real cp.async latency-hiding - that \
                     the prediction implicitly assumes; see the file header)."
                );
            }
        }
        None => println!(
            "\nNo selected candidate both compiled and passed the correctness check - \
             can't compare predicted vs measured ranking."
        ),
    }

    // res.error is only ever Some(..) on a genuine failure path (compile
    // failure, spawn failure, or every repeat run failing to report a clean
    // result) - no need to re-check compiled/correct here too.
    for (cand, _, _, _, _, res) in &results {
        if let Some(err) = &res.error {
            println!(
                "\n--- Diagnostic for {}x{}x{}/stages{} ---\n{}",
                cand.cta_m, cand.cta_n, cand.cta_k, cand.num_stages, err
            );
        }
    }
}

// ────────────────────────────────────────────────────────────
// Gated regression test - so this harness is discoverable and runnable in
// CI-with-GPU without anyone having to remember `cargo run --bin
// autotune_verify` exists, and so it gates on a real pass/fail instead of
// just printing a report for a human to read. `[[bin]]` targets support
// #[cfg(test)] like any other target - `cargo test` (or specifically
// `cargo test --bin autotune_verify`) builds and runs this like normal,
// just skipped by default because of #[ignore].
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod verify_tests {
    use super::*;

    /// Actually compiles and measures the autotuner's top few real
    /// candidates for a fixed, representative (M,N,K) on real hardware, and
    /// asserts the predicted #1 candidate's measured time is within a
    /// generous (2x) tolerance of the fastest candidate actually measured
    /// among them.
    ///
    /// Deliberately NOT "predicted #1 must be the single fastest measured" -
    /// this session's own investigation found real, documented gaps (the
    /// tile-orientation axis noted in score_candidate's doc comment; general
    /// model-vs-reality drift of up to ~1.4x observed at some sizes) that
    /// are known and out of scope to fully close here. The 2x bound is
    /// chosen from that real data: it comfortably covers every gap actually
    /// observed while still catching a genuine regression (e.g. reverting to
    /// the old backwards stage_multiplier would be expected to blow well past it).
    ///
    /// Needs a real CUDA GPU + nvcc, so it's marked #[ignore] and skips
    /// itself gracefully (not a failure) if either is unavailable. Run
    /// explicitly with:
    ///   cargo test --release --bin autotune_verify -- --ignored --nocapture
    #[test]
    #[ignore = "needs a real CUDA GPU + nvcc on PATH; run explicitly with -- --ignored"]
    fn test_predicted_ranking_roughly_matches_measured() {
        if Command::new("nvcc").arg("--version").output().is_err() {
            eprintln!("SKIPPED: nvcc not found on PATH.");
            return;
        }

        let hw = check_or_probe_hardware();
        if hw.gpu_vendor.eq_ignore_ascii_case("unknown") || hw.gpu_name.eq_ignore_ascii_case("unknown gpu") {
            eprintln!("SKIPPED: no real GPU detected by the sentinel probe.");
            return;
        }

        let (m, n, k) = (2048, 2048, 2048);
        let arch = nvcc_arch_flag(&hw);

        let mut candidates = Autotuner::generate_candidates(m, n, k, Precision::F16);
        candidates.dedup();
        let mut scored: Vec<(f64, AutotuneCandidate)> = candidates
            .into_iter()
            .map(|c| {
                let s = Autotuner::score_candidate(&c, m, n, k, &hw);
                (s, c)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        assert!(!scored.is_empty(), "generate_candidates produced no candidates for {}x{}x{}", m, n, k);

        let top_n = 4.min(scored.len());
        let workdir = env::temp_dir().join("autotune_verify_test_harness");
        fs::create_dir_all(&workdir).expect("failed to create workdir");

        let mut measured_times: Vec<f64> = Vec::new();
        for (tag, (_, cand)) in scored.iter().take(top_n).enumerate() {
            let res = compile_and_run(cand, m, n, k, &arch, &workdir, tag);
            assert!(
                res.compiled,
                "candidate {}x{}x{}/stages{} failed to compile:\n{}",
                cand.cta_m, cand.cta_n, cand.cta_k, cand.num_stages,
                res.error.as_deref().unwrap_or("(no diagnostic captured)")
            );
            assert_eq!(
                res.correct,
                Some(true),
                "candidate {}x{}x{}/stages{} failed its correctness check or every run failed:\n{}",
                cand.cta_m, cand.cta_n, cand.cta_k, cand.num_stages,
                res.error.as_deref().unwrap_or("(no diagnostic captured)")
            );
            measured_times.push(
                res.time_ms
                    .expect("a compiled, correct candidate must report a timing"),
            );
        }

        let predicted_best_time = measured_times[0];
        let actual_best_time = measured_times.iter().cloned().fold(f64::INFINITY, f64::min);

        assert!(
            predicted_best_time <= actual_best_time * 2.0,
            "predicted #1 candidate measured {:.4}ms, more than 2x slower than the fastest \
             of the top {} predicted candidates actually measured ({:.4}ms) at M={} N={} K={} - \
             the ranking has drifted further from reality than this session's own measurements \
             ever showed; see score_candidate's doc comments for the known, tolerated gaps.",
            predicted_best_time, top_n, actual_best_time, m, n, k
        );
    }
}
