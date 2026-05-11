// ============================================================
//  Y-Lang  —  CPU Backend: 256-bit Register Wrapper
//  Subagent A | avx_wrapper.rs
//
//  Wraps std::arch::x86_64 intrinsics into a safe, typed API
//  that mirrors Y-Lang's type system on the CPU side.
//
//  Register mapping:
//    Y-Lang  Fragment<_, _, F32>  →  YMM (__m256  / 8×f32)
//    Y-Lang  Fragment<_, _, F64>  →  YMM (__m256d / 4×f64)
//    Y-Lang  Fragment<_, _, I32>  →  YMM (__m256i / 8×i32)
//    Y-Lang  Fragment<_, _, F16>  →  YMM (__m256i / 16×f16 packed)
//
//  Safety model:
//    - All unsafe intrinsic calls are contained in this file.
//    - Public API is safe; panics on CPUs lacking AVX2.
//    - AVX-512 paths behind `avx512f` feature gate.
// ============================================================

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ────────────────────────────────────────────────────────────
//  Runtime feature detection
// ────────────────────────────────────────────────────────────

/// Call once at start-up. Panics if AVX2 is unavailable.
pub fn require_avx2() {
    assert!(
        is_x86_feature_detected!("avx2"),
        "Y-Lang CPU backend requires AVX2 support. \
         Compile with RUSTFLAGS=\"-C target-feature=+avx2\" or run on a capable CPU."
    );
}

/// Returns true if AVX-512F is available at runtime.
pub fn has_avx512f() -> bool {
    is_x86_feature_detected!("avx512f")
}

// ────────────────────────────────────────────────────────────
//  Y256f32  —  8 × f32  (maps to __m256 / YMM register)
// ────────────────────────────────────────────────────────────

/// A 256-bit register holding 8 single-precision floats.
/// Corresponds to Y-Lang's `Fragment<_, _, F32>` on the CPU backend.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Y256f32(#[cfg(target_arch = "x86_64")] __m256);

impl Y256f32 {
    /// Broadcast a single f32 across all 8 lanes.
    #[inline]
    pub fn splat(val: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_set1_ps(val))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!("Y256f32 requires x86_64")
    }

    /// Compatibility alias used by generated backend code.
    #[inline]
    pub fn from_scalar(val: f32) -> Self {
        Self::splat(val)
    }

    /// Load 8 f32 values from a 32-byte aligned slice.
    ///
    /// # Panics
    /// Panics if the slice has fewer than 8 elements or is unaligned.
    #[inline]
    pub fn load_aligned(src: &[f32]) -> Self {
        assert!(
            src.len() >= 8,
            "Y256f32::load_aligned requires at least 8 elements"
        );
        assert!(
            src.as_ptr() as usize % 32 == 0,
            "Y256f32::load_aligned: pointer must be 32-byte aligned"
        );
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_load_ps(src.as_ptr()))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Load 8 f32 values from a 32-byte aligned raw pointer.
    #[inline]
    pub fn load_aligned_ptr(src: *const f32) -> Self {
        assert!(
            !src.is_null(),
            "Y256f32::load_aligned_ptr requires a non-null pointer"
        );
        assert!(
            src as usize % 32 == 0,
            "Y256f32::load_aligned_ptr: pointer must be 32-byte aligned"
        );
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_load_ps(src))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Load 8 f32 values from an unaligned slice (slower, always safe).
    #[inline]
    pub fn load(src: &[f32]) -> Self {
        assert!(src.len() >= 8, "Y256f32::load requires at least 8 elements");
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_loadu_ps(src.as_ptr()))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Store 8 f32 values into a 32-byte aligned slice.
    #[inline]
    pub fn store_aligned(self, dst: &mut [f32]) {
        assert!(dst.len() >= 8);
        assert!(
            dst.as_ptr() as usize % 32 == 0,
            "store_aligned: pointer not 32-byte aligned"
        );
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_store_ps(dst.as_mut_ptr(), self.0)
        }
    }

    /// Store 8 f32 values into a 32-byte aligned raw pointer.
    #[inline]
    pub fn store_aligned_ptr(self, dst: *mut f32) {
        assert!(
            !dst.is_null(),
            "Y256f32::store_aligned_ptr requires a non-null pointer"
        );
        assert!(
            dst as usize % 32 == 0,
            "Y256f32::store_aligned_ptr: pointer must be 32-byte aligned"
        );
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_store_ps(dst, self.0)
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Store 8 f32 values into an unaligned slice.
    #[inline]
    pub fn store(self, dst: &mut [f32]) {
        assert!(dst.len() >= 8);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_storeu_ps(dst.as_mut_ptr(), self.0)
        }
    }

    /// Zero register (all 8 lanes = 0.0).
    #[inline]
    pub fn zero() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_setzero_ps())
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    // ── Arithmetic ────────────────────────────────────────────

    /// Lane-wise addition.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_add_ps(self.0, rhs.0))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Lane-wise subtraction.
    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_sub_ps(self.0, rhs.0))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Lane-wise multiplication.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_mul_ps(self.0, rhs.0))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Lane-wise division.
    #[inline]
    pub fn div(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_div_ps(self.0, rhs.0))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Fused multiply-add: (self * b) + c  — single rounding, no precision loss.
    /// Requires FMA feature. Falls back to mul+add if unavailable.
    #[inline]
    pub fn fmadd(self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("fma") {
                Self(_mm256_fmadd_ps(self.0, b.0, c.0))
            } else {
                Self(_mm256_add_ps(_mm256_mul_ps(self.0, b.0), c.0))
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Compatibility alias used by generated backend code.
    #[inline]
    pub fn ma(self, b: Self, c: Self) -> Self {
        self.fmadd(b, c)
    }

    /// Horizontal sum across all 8 lanes.  Returns a scalar f32.
    #[inline]
    pub fn horizontal_sum(self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // hadd reduces 8 → 4 → 2, then extract
            let h1 = _mm256_hadd_ps(self.0, self.0);
            let h2 = _mm256_hadd_ps(h1, h1);
            let lo = _mm256_castps256_ps128(h2);
            let hi = _mm256_extractf128_ps(h2, 1);
            let sum = _mm_add_ps(lo, hi);
            _mm_cvtss_f32(sum)
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }
}

// ────────────────────────────────────────────────────────────
//  Y256i32  —  8 × i32  (maps to __m256i / YMM register)
// ────────────────────────────────────────────────────────────

/// A 256-bit register holding 8 signed 32-bit integers.
/// Corresponds to Y-Lang's `Fragment<_, _, I32>` on the CPU backend.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Y256i32(#[cfg(target_arch = "x86_64")] __m256i);

impl Y256i32 {
    /// Broadcast a single i32 across all 8 lanes.
    #[inline]
    pub fn splat(val: i32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_set1_epi32(val))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Load 8 i32 values from an unaligned slice.
    #[inline]
    pub fn load(src: &[i32]) -> Self {
        assert!(src.len() >= 8);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_loadu_si256(src.as_ptr() as *const __m256i))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Store 8 i32 values into an unaligned slice.
    #[inline]
    pub fn store(self, dst: &mut [i32]) {
        assert!(dst.len() >= 8);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, self.0)
        }
    }

    /// Zero register.
    #[inline]
    pub fn zero() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_setzero_si256())
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Lane-wise addition (wrapping).
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_add_epi32(self.0, rhs.0))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Lane-wise subtraction (wrapping).
    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_sub_epi32(self.0, rhs.0))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Lane-wise multiplication (low 32 bits of 32×32 product).
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_mullo_epi32(self.0, rhs.0))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }
}

// ────────────────────────────────────────────────────────────
//  Y256f64  —  4 × f64  (maps to __m256d / YMM register)
// ────────────────────────────────────────────────────────────

/// A 256-bit register holding 4 double-precision floats.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Y256f64(#[cfg(target_arch = "x86_64")] __m256d);

impl Y256f64 {
    /// Broadcast a single f64 across all 4 lanes.
    #[inline]
    pub fn splat(val: f64) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_set1_pd(val))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Load 4 f64 values from an unaligned slice.
    #[inline]
    pub fn load(src: &[f64]) -> Self {
        assert!(src.len() >= 4);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_loadu_pd(src.as_ptr()))
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }

    /// Store 4 f64 values into an unaligned slice.
    #[inline]
    pub fn store(self, dst: &mut [f64]) {
        assert!(dst.len() >= 4);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_storeu_pd(dst.as_mut_ptr(), self.0)
        }
    }

    /// Lane-wise fused multiply-add.
    #[inline]
    pub fn fmadd(self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") {
                unsafe { Self(_mm256_fmadd_pd(self.0, b.0, c.0)) }
            } else {
                unsafe { Self(_mm256_add_pd(_mm256_mul_pd(self.0, b.0), c.0)) }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unimplemented!()
    }
}

// ────────────────────────────────────────────────────────────
//  Operator overloads — lets Y-Lang IR use  a + b  syntax
// ────────────────────────────────────────────────────────────

use std::ops::{Add, Div, Mul, Sub};

impl Add for Y256f32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Y256f32::add(self, rhs)
    }
}
impl Sub for Y256f32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Y256f32::sub(self, rhs)
    }
}
impl Mul for Y256f32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Y256f32::mul(self, rhs)
    }
}
impl Div for Y256f32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Y256f32::div(self, rhs)
    }
}

impl Add for Y256i32 {
    type Output = Self;
    #[cfg(target_arch = "x86_64")]
    fn add(self, rhs: Self) -> Self {
        unsafe { Y256i32(std::arch::x86_64::_mm256_add_epi32(self.0, rhs.0)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    fn add(self, _rhs: Self) -> Self {
        unimplemented!()
    }
}
impl Sub for Y256i32 {
    type Output = Self;
    #[cfg(target_arch = "x86_64")]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Y256i32(std::arch::x86_64::_mm256_sub_epi32(self.0, rhs.0)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    fn sub(self, _rhs: Self) -> Self {
        unimplemented!()
    }
}
impl Mul for Y256i32 {
    type Output = Self;
    #[cfg(target_arch = "x86_64")]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Y256i32(std::arch::x86_64::_mm256_mullo_epi32(self.0, rhs.0)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    fn mul(self, _rhs: Self) -> Self {
        unimplemented!()
    }
}

// ────────────────────────────────────────────────────────────
//  AVX-512 extensions (feature-gated, not always available)
// ────────────────────────────────────────────────────────────

/// 512-bit register holding 16 × f32.
/// Only constructed if `has_avx512f()` returns true.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Y512f32(__m512);

#[cfg(target_arch = "x86_64")]
impl Y512f32 {
    /// Broadcast a single f32 across all 16 lanes.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn splat(val: f32) -> Self {
        Self(_mm512_set1_ps(val))
    }

    /// Lane-wise fused multiply-add over 16 lanes.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn fmadd(self, b: Self, c: Self) -> Self {
        Self(_mm512_fmadd_ps(self.0, b.0, c.0))
    }

    /// Downcast: extract lower 256 bits as Y256f32 (zero-cost cast).
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn lower_half(self) -> Y256f32 {
        Y256f32(_mm512_castps512_ps256(self.0))
    }
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_y256f32_add() {
        let a = Y256f32::splat(1.0);
        let b = Y256f32::splat(2.0);
        let mut out = [0.0f32; 8];
        (a + b).store(&mut out);
        assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_y256f32_fmadd() {
        let a = Y256f32::splat(2.0);
        let b = Y256f32::splat(3.0);
        let c = Y256f32::splat(1.0);
        let mut out = [0.0f32; 8];
        a.fmadd(b, c).store(&mut out);
        // 2*3+1 = 7
        assert!(out.iter().all(|&x| (x - 7.0).abs() < 1e-6));
    }

    #[test]
    fn test_y256f32_hsum() {
        let mut out = [0.0f32; 8];
        Y256f32::splat(1.5).store(&mut out);
        let sum = Y256f32::load(&out).horizontal_sum();
        assert!((sum - 12.0).abs() < 1e-5, "hsum = {}", sum);
    }

    #[test]
    fn test_y256i32_mul() {
        let a = Y256i32::splat(4);
        let b = Y256i32::splat(5);
        let mut out = [0i32; 8];
        (a * b).store(&mut out);
        assert!(out.iter().all(|&x| x == 20));
    }
}
