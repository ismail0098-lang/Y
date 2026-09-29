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
            // AVX2, NOT AVX-512. This defaulted to 16 floats and
            // `masking: true`, i.e. it assumed the WIDEST ISA when it knew
            // nothing - and an AVX-512 instruction on a CPU without it is
            // SIGILL, not a slowdown. Most consumer Intel since Alder Lake and
            // everything before Skylake-X lacks it. Guess DOWN: AVX2 runs
            // everywhere AVX-512 does, and the reverse is a crash.
            //
            // Callers that can probe should, rather than relying on this -
            // `sentinel::probe_cpu_hardware_profile()` reads CPUID.
            simd_vector_width_floats: 8,
            supports_avx512_masking: false,
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

    /// The masked regime is AVX-512-ONLY by construction (`classify_shape`
    /// gates it on `supports_avx512_masking`), so the profile has to say so.
    /// This used to rely on `Default`, which assumed AVX-512 - hiding both
    /// that the regime needs it and that the default was guessing UP at an
    /// ISA whose absence is a SIGILL rather than a slowdown.
    #[test]
    fn test_classify_irregular_masked() {
        let profile = CpuHardwareProfile {
            simd_vector_width_floats: 16,
            supports_avx512_masking: true,
            ..CpuHardwareProfile::default()
        };
        let dispatcher = CpuShapeDispatcher::new(profile);
        // M=137, N=391, K=1013 (Non-divisible by 16)
        assert_eq!(dispatcher.classify_shape(137, 391, 1013, 4), CpuMatrixRegime::IrregularMasked);
    }

    /// The control: on a CPU WITHOUT AVX-512 the same shape must not be
    /// classified into a regime that needs it. Otherwise the guess-down
    /// default would just be routed around.
    #[test]
    fn irregular_shapes_do_not_take_the_masked_path_without_avx512() {
        let profile = CpuHardwareProfile {
            simd_vector_width_floats: 8,
            supports_avx512_masking: false,
            ..CpuHardwareProfile::default()
        };
        let dispatcher = CpuShapeDispatcher::new(profile);
        assert_ne!(
            dispatcher.classify_shape(137, 391, 1013, 4),
            CpuMatrixRegime::IrregularMasked,
            "an AVX-512 masking regime was selected for a CPU without AVX-512; \
             the emitted kernel would SIGILL rather than run slowly"
        );
    }

    #[test]
    fn test_classify_nice_square() {
        let profile = CpuHardwareProfile::default();
        let dispatcher = CpuShapeDispatcher::new(profile);
        // M=1024, N=1024, K=1024
        assert_eq!(dispatcher.classify_shape(1024, 1024, 1024, 4), CpuMatrixRegime::NiceSquare);
    }
}
