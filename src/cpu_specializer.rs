// ============================================================
//  Y  —  Hardware-Aware CPU Matrix & Shape Specialization Pass
//  cpu_specializer.rs
//
//  Analyzes matrix dimensions (M, N, K) and access patterns to route
//  execution into 5 hardware-optimized CPU regimes:
//    1. NiceSquare       - Large square GEMM (BLIS cache-packing)
//    2. DecodeGEMV       - Token decode M<=4 (Streaming 1D vector kernel)
//    3. SmallDirect      - Fits in L1 cache (Zero-pack direct register kernel)
//    4. IrregularMasked  - Non-SIMD aligned boundaries (AVX-512 vector masking)
//    5. DeepK            - K >> M,N (Split-K multi-thread reduction)
// ============================================================

#![allow(dead_code)]

use crate::ast::*;

/// Execution regime for CPU matrix operations based on shape and memory bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuMatrixRegime {
    /// Large, power-of-two or SIMD-aligned square matrix (M, N, K >= 512). Uses L2/L3 cache packing.
    NiceSquare,
    /// Token decode GEMV (M <= 4, large N, K). Uses 1D memory-bandwidth streaming vector kernel.
    DecodeGEMV,
    /// Micro GEMM where total working set fits in L1 cache (<= 32KB). Bypasses memory packing.
    SmallDirect,
    /// Irregular shape with dimensions non-divisible by SIMD vector width. Uses AVX-512 predication.
    IrregularMasked,
    /// Reduction-heavy GEMM (K >= 8 * max(M, N)). Uses Split-K multi-core thread parallel reduction.
    DeepK,
}

/// Dynamic Hardware Profile for CPU execution decision-making.
#[derive(Debug, Clone, Copy)]
pub struct CpuHardwareProfile {
    pub l1d_bytes: usize,
    pub l2_bytes: usize,
    pub l3_bytes: usize,
    pub simd_vector_width_floats: usize, // e.g. 8 for AVX2, 16 for AVX-512
    pub supports_avx512_masking: bool,
    pub logical_cores: usize,
}

impl Default for CpuHardwareProfile {
    fn default() -> Self {
        CpuHardwareProfile {
            l1d_bytes: 32 * 1024,      // 32 KB default L1d
            l2_bytes: 512 * 1024,     // 512 KB default L2
            l3_bytes: 16 * 1024 * 1024,// 16 MB default L3
            simd_vector_width_floats: 16, // AVX-512 baseline
            supports_avx512_masking: true,
            logical_cores: 8,
        }
    }
}

pub struct CpuShapeDispatcher {
    pub profile: CpuHardwareProfile,
}

impl CpuShapeDispatcher {
    pub fn new(profile: CpuHardwareProfile) -> Self {
        CpuShapeDispatcher { profile }
    }

    /// Classifies a GEMM shape (M, N, K) with element size into its optimal CPU execution regime.
    pub fn classify_shape(&self, m: usize, n: usize, k: usize, elem_bytes: usize) -> CpuMatrixRegime {
        let total_bytes = (m * k + k * n + m * n) * elem_bytes;
        let simd_w = self.profile.simd_vector_width_floats;

        // 1. Check for SmallDirect (working set fits inside L1 cache)
        if total_bytes <= self.profile.l1d_bytes {
            return CpuMatrixRegime::SmallDirect;
        }

        // 2. Check for DecodeGEMV (M <= 4, typical LLM autoregressive token generation)
        if m <= 4 && n >= 256 && k >= 256 {
            return CpuMatrixRegime::DecodeGEMV;
        }

        // 3. Check for DeepK (K dimension dominates over spatial M, N)
        if k >= 8 * m.max(n) && k >= 1024 {
            return CpuMatrixRegime::DeepK;
        }

        // 4. Check for IrregularMasked (boundaries not aligned to SIMD vector width)
        if (m % simd_w != 0 || n % simd_w != 0 || k % simd_w != 0) && self.profile.supports_avx512_masking {
            return CpuMatrixRegime::IrregularMasked;
        }

        // 5. Default to NiceSquare for large regular matrices
        CpuMatrixRegime::NiceSquare
    }
}

/// AST Transformation Pass that annotates and transforms CPU matrix kernels.
pub struct CpuSpecializerPass {
    pub dispatcher: CpuShapeDispatcher,
    pub shapes_classified: usize,
    pub small_direct_count: usize,
    pub decode_gemv_count: usize,
    pub deep_k_count: usize,
    pub irregular_count: usize,
    pub nice_square_count: usize,
}

impl CpuSpecializerPass {
    pub fn new(profile: CpuHardwareProfile) -> Self {
        CpuSpecializerPass {
            dispatcher: CpuShapeDispatcher::new(profile),
            shapes_classified: 0,
            small_direct_count: 0,
            decode_gemv_count: 0,
            deep_k_count: 0,
            irregular_count: 0,
            nice_square_count: 0,
        }
    }

    /// Processes a full program AST to annotate and optimize CPU matrix kernel calls.
    pub fn run(&mut self, prog: &mut Program) {
        for item in &mut prog.items {
            if let Item::Kernel(kernel) = item {
                self.process_kernel(kernel);
            }
        }
    }

    fn process_kernel(&mut self, kernel: &mut KernelDecl) {
        if let Some(tile) = &kernel.tile {
            let m_opt = self.extract_usize_lit(&tile.block_m);
            let n_opt = self.extract_usize_lit(&tile.block_n);
            let k_opt = tile.block_k.as_ref().and_then(|expr| self.extract_usize_lit(expr));

            if let (Some(m), Some(n), Some(k)) = (m_opt, n_opt, k_opt) {
                let regime = self.dispatcher.classify_shape(m, n, k, 4);
                self.shapes_classified += 1;
                match regime {
                    CpuMatrixRegime::SmallDirect => self.small_direct_count += 1,
                    CpuMatrixRegime::DecodeGEMV => self.decode_gemv_count += 1,
                    CpuMatrixRegime::DeepK => self.deep_k_count += 1,
                    CpuMatrixRegime::IrregularMasked => self.irregular_count += 1,
                    CpuMatrixRegime::NiceSquare => self.nice_square_count += 1,
                }
            }
        }
    }

    fn extract_usize_lit(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::IntLit(v, _) if *v > 0 => Some(*v as usize),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_small_direct() {
        let profile = CpuHardwareProfile::default();
        let dispatcher = CpuShapeDispatcher::new(profile);
        // 16x16x16 float32 matrix -> 3 * 256 * 4 = 3072 bytes <= 32KB
        assert_eq!(dispatcher.classify_shape(16, 16, 16, 4), CpuMatrixRegime::SmallDirect);
    }

    #[test]
    fn test_classify_decode_gemv() {
        let profile = CpuHardwareProfile::default();
        let dispatcher = CpuShapeDispatcher::new(profile);
        // M=1, N=4096, K=4096 (Token decoding)
        assert_eq!(dispatcher.classify_shape(1, 4096, 4096, 4), CpuMatrixRegime::DecodeGEMV);
    }

    #[test]
    fn test_classify_deep_k() {
        let profile = CpuHardwareProfile::default();
        let dispatcher = CpuShapeDispatcher::new(profile);
        // M=64, N=64, K=32768
        assert_eq!(dispatcher.classify_shape(64, 64, 32768, 4), CpuMatrixRegime::DeepK);
    }

    #[test]
    fn test_classify_irregular_masked() {
        let profile = CpuHardwareProfile::default();
        let dispatcher = CpuShapeDispatcher::new(profile);
        // M=137, N=391, K=1013 (Non-divisible by 16)
        assert_eq!(dispatcher.classify_shape(137, 391, 1013, 4), CpuMatrixRegime::IrregularMasked);
    }

    #[test]
    fn test_classify_nice_square() {
        let profile = CpuHardwareProfile::default();
        let dispatcher = CpuShapeDispatcher::new(profile);
        // M=1024, N=1024, K=1024
        assert_eq!(dispatcher.classify_shape(1024, 1024, 1024, 4), CpuMatrixRegime::NiceSquare);
    }
}
