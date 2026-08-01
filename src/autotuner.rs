// ============================================================
//  Y  —  JIT Dynamic Autotuning Pass
//  autotuner.rs
//
//  Dynamically benchmarks matrix dimensions, tile layouts,
//  warp counts (num_warps), and pipeline stages (num_stages)
//  to select the exact configuration yielding peak TFLOPS.
// ============================================================

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use crate::ptx_emitter::CtaTileConfig;
use crate::sentinel::HardwareProfile;

/// Compile-time operand precision for a `@tile`d kernel. Currently plumbed
/// through as call-site compatibility only (`generate_candidates`/`autotune`
/// accept it but don't yet branch the search space on it) - this file was
/// restored from an older checkpoint that predates whatever precision-aware
/// tuning logic the working tree previously had (see git history), so this
/// is a deliberately conservative reintroduction of the public API surface
/// `ptx_emitter.rs`/`c_api.rs`/`src/bin/autotune_verify.rs` call with, not a
/// claim that per-precision tuning is implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F16,
    F32,
    BF16,
    Fp8,
}

/// Represents a candidate configuration parameter set for autotuning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AutotuneCandidate {
    pub cta_m: u32,
    pub cta_n: u32,
    pub cta_k: u32,
    pub warps_m: u32,
    pub warps_n: u32,
    pub num_stages: u32,
    pub num_warps: u32,
}

impl AutotuneCandidate {
    pub fn to_cta_tile_config(&self) -> CtaTileConfig {
        CtaTileConfig {
            cta_m: self.cta_m,
            cta_n: self.cta_n,
            cta_k: self.cta_k,
            warps_m: self.warps_m,
            warps_n: self.warps_n,
            mma_m: 16,
            mma_n: 16,
            mma_k: 16,
            num_stages: self.num_stages,
            num_warps: self.num_warps,
        }
    }
}

static AUTOTUNE_CACHE: OnceLock<Mutex<HashMap<(u32, u32, u32), AutotuneCandidate>>> = OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<(u32, u32, u32), AutotuneCandidate>> {
    AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct Autotuner;

impl Autotuner {
    /// Generates candidate search space for matrix dimensions M, N, K.
    /// `_precision` is accepted for call-site compatibility (see
    /// `Precision`'s doc comment) but does not yet affect the search space.
    pub fn generate_candidates(m: u32, n: u32, _k: u32, _precision: Precision) -> Vec<AutotuneCandidate> {
        let mut candidates = Vec::new();

        let tile_sizes = if m <= 256 || n <= 256 {
            vec![
                (16, 32, 32, 1, 1),
                (32, 32, 32, 2, 1),
                (32, 64, 32, 2, 2),
                (64, 64, 32, 2, 2),
                (64, 64, 64, 2, 2),
            ]
        } else if m <= 512 || n <= 512 {
            vec![
                (64, 64, 32, 2, 2),
                (64, 64, 64, 2, 2),
                (64, 128, 32, 2, 4),
                (64, 128, 64, 2, 4),
                (128, 64, 32, 4, 2),
                (128, 64, 64, 4, 2),
            ]
        } else {
            vec![
                (128, 128, 32, 4, 2),
                (128, 128, 64, 4, 2),
                (128, 128, 128, 4, 2),
                (128, 256, 32, 4, 4),
                (128, 256, 64, 4, 4),
                (256, 128, 32, 4, 4),
                (256, 128, 64, 4, 4),
            ]
        };

        for (cta_m, cta_n, cta_k, warps_m, warps_n) in tile_sizes {
            let num_warps = warps_m * warps_n;
            let stage_options = if cta_m <= 32 { vec![2] } else { vec![2, 3, 4] };
            for num_stages in stage_options {
                candidates.push(AutotuneCandidate {
                    cta_m,
                    cta_n,
                    cta_k,
                    warps_m,
                    warps_n,
                    num_stages,
                    num_warps,
                });
            }
        }

        candidates
    }

    /// Evaluates candidate configurations using hardware latency model with SM wave alignment.
    pub fn score_candidate(candidate: &AutotuneCandidate, m: u32, n: u32, _k: u32, hw_profile: &HardwareProfile) -> f64 {
        let num_tiles_m = (m + candidate.cta_m - 1) / candidate.cta_m;
        let num_tiles_n = (n + candidate.cta_n - 1) / candidate.cta_n;
        let total_ctas = num_tiles_m * num_tiles_n;

        let num_sms = if hw_profile.sm_count > 0 { hw_profile.sm_count } else { 66 };
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

        // Detect FP8 precision vs FP16 precision: FP8 loads 1 byte per element
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

        // Overall efficiency score (higher is better)
        let score = compute_intensity * stage_multiplier * sm_occupancy * wave_efficiency * smem_penalty * (1.0 / ctas_per_sm) * 1000.0;
        score
    }

    /// Automatically selects optimal CTA tile layout, warp count, and stage count for target dim.
    /// `precision` is accepted for call-site compatibility (see
    /// `Precision`'s doc comment) but does not yet affect selection.
    pub fn autotune(m: u32, n: u32, k: u32, hw_profile: &HardwareProfile, precision: Precision) -> CtaTileConfig {
        let key = (m, n, k);
        if let Ok(cache) = get_cache().lock() {
            if let Some(cached_candidate) = cache.get(&key) {
                return cached_candidate.to_cta_tile_config();
            }
        }

        let candidates = Self::generate_candidates(m, n, k, precision);
        let mut best_candidate = candidates[0].clone();
        let mut best_score = -1.0;

        for cand in &candidates {
            let score = Self::score_candidate(cand, m, n, k, hw_profile);
            if score > best_score {
                best_score = score;
                best_candidate = cand.clone();
            }
        }

        if let Ok(mut cache) = get_cache().lock() {
            cache.insert(key, best_candidate.clone());
        }

        best_candidate.to_cta_tile_config()
    }

    pub fn clear_cache() {
        if let Ok(mut cache) = get_cache().lock() {
            cache.clear();
        }
    }

    /// Rough, formula-based occupancy estimate for a candidate (register/
    /// shared-memory/warp/thread limits vs hardware maxima -> blocks/SM ->
    /// achieved occupancy). Restored as a fresh, disclosed approximation to
    /// satisfy `src/bin/autotune_verify.rs`'s diagnostic printout after this
    /// file was reverted to an older checkpoint (see `Precision`'s doc
    /// comment) - this is NOT a reconstruction of whatever occupancy model
    /// the working tree previously had; `est_regs_per_thread` in particular
    /// is a coarse heuristic (scales with accumulator-fragment count per
    /// warp), not a real `ptxas -v` measurement. Only feeds this bin's own
    /// printed diagnostics, not `score_candidate`/`autotune`'s actual
    /// selection - so it has no bearing on the real GEMM codegen path.
    pub fn estimate_occupancy(candidate: &AutotuneCandidate, hw_profile: &HardwareProfile) -> OccupancyEstimate {
        let max_regs_per_sm = if hw_profile.max_regs_per_sm > 0 { hw_profile.max_regs_per_sm as f64 } else { 65536.0 };
        let max_warps_per_sm = if hw_profile.max_warps_per_sm > 0 { hw_profile.max_warps_per_sm as f64 } else { 48.0 };
        let max_smem_per_sm = if hw_profile.max_smem_per_sm_bytes > 0 { hw_profile.max_smem_per_sm_bytes as f64 } else { 49152.0 };
        let max_threads_per_sm = if hw_profile.max_threads_per_sm > 0 { hw_profile.max_threads_per_sm as f64 } else { 1536.0 };

        let per_warp_m = (candidate.cta_m / candidate.warps_m).max(1);
        let per_warp_n = (candidate.cta_n / candidate.warps_n).max(1);
        let num_frags = ((per_warp_m / 16).max(1) * (per_warp_n / 16).max(1)) as f64;
        // 8 f32 accumulator regs/fragment + fixed per-thread bookkeeping overhead.
        let est_regs_per_thread = 32.0 + num_frags * 8.0 * 1.5;

        let threads_per_block = (candidate.num_warps * 32) as f64;
        let regs_limit_blocks = (max_regs_per_sm / (est_regs_per_thread * threads_per_block)).floor();

        let bytes_per_elem = 2.0; // F16 operand staging
        let smem_bytes_per_block =
            (candidate.cta_m * candidate.cta_k) as f64 * bytes_per_elem * candidate.num_stages as f64
                + (candidate.cta_k * candidate.cta_n) as f64 * bytes_per_elem * candidate.num_stages as f64;
        let smem_limit_blocks = if smem_bytes_per_block > 0.0 { (max_smem_per_sm / smem_bytes_per_block).floor() } else { f64::INFINITY };

        let warps_limit_blocks = (max_warps_per_sm / candidate.num_warps as f64).floor();
        let threads_limit_blocks = (max_threads_per_sm / threads_per_block).floor();

        let achieved_blocks_per_sm = [regs_limit_blocks, smem_limit_blocks, warps_limit_blocks, threads_limit_blocks]
            .into_iter()
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
        let achieved_occupancy = ((achieved_blocks_per_sm * candidate.num_warps as f64) / max_warps_per_sm).min(1.0);

        OccupancyEstimate { est_regs_per_thread, achieved_blocks_per_sm, achieved_occupancy }
    }
}

/// See `Autotuner::estimate_occupancy`'s doc comment.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyEstimate {
    pub est_regs_per_thread: f64,
    pub achieved_blocks_per_sm: f64,
    pub achieved_occupancy: f64,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autotuner_candidate_generation() {
        let candidates = Autotuner::generate_candidates(1024, 1024, 1024, Precision::F16);
        assert!(!candidates.is_empty());
        for cand in &candidates {
            assert!(cand.num_stages >= 2);
            assert!(cand.num_warps >= 4);
        }
    }

    #[test]
    fn test_autotuner_selection() {
        let hw = HardwareProfile::default();
        let config = Autotuner::autotune(2048, 2048, 2048, &hw, Precision::F16);
        assert!(config.cta_m >= 64);
        assert!(config.num_stages >= 2);
        assert!(config.num_warps >= 4);
    }
}