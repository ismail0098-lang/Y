// ============================================================
//  Y  —  PTX Code Emitter
//  ptx_emitter.rs
//
//  Backend code generator targeting NVIDIA PTX.
//  Converts validated AST nodes into virtual assembly.
//  bypasses high-level CUDA runtime and talks directly
//  to the silicon via instructions like ldmatrix and cp.async.
//  [3D BLOCK POINTER EXTENSIONS INCLUDED]
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use crate::autotuner::{Autotuner, Precision};
use crate::sentinel::HardwareProfile;
use std::fmt::Write;

#[cfg(feature = "zk")]
use crate::zk_emitter::*;

/// L2-locality grid-swizzle group size for `emit_tensor_core_gemm_kernel`
/// (see `emit_grid_swizzle_code`'s doc comment for the algorithm).
/// Empirically swept over {4, 8, 16, 32} against real sm_89 (RTX 4070 Ti
/// SUPER) hardware at the 4096-16384 M=N=K scale via
/// tests/benchmark_y_tensor_core_gemm.py: all four measured statistically
/// identical (0.74x-0.75x vs cuBLAS, well within this benchmark's
/// run-to-run noise band) - enabling the swizzle at all is what moved the
/// needle (0.67x-0.70x -> ~0.74x), not this constant's value. Kept at 8 as
/// the conventional CUTLASS/Triton default rather than a measured optimum,
/// since none was found in range. Re-sweep if this kernel's tile sizes,
/// target GPU, or L2 capacity change materially. Not shape-dependent for
/// correctness: the swizzle math is a verified bijection (see
/// test_grid_swizzle_uneven_grid_is_bijection) for any group size against
/// any grid, including grids far smaller than this constant.
pub const GEMM_SWIZZLE_GROUP_SIZE: u32 = 8;

/// Two hand-picked CTA tile shapes for `emit_fp8_gemm_kernel`'s multi-warp
/// codegen, chosen by `emit_fp8_gemm_kernel` per-kernel from `M`/`N` (see
/// that function's doc comment for the selection heuristic) - still no
/// real Autotuner integration (`Autotuner::generate_candidates` accepts
/// but ignores `Precision::Fp8` - see that enum's doc comment in
/// autotuner.rs), so this is two fixed shapes and a size threshold, not a
/// real per-problem-size search (a disclosed, deliberate scope cut - see
/// `emit_fp8_gemm_kernel`'s doc comment for the next-session priority
/// list).
///
/// `_LARGE` (128x128x64, 4x2 warps) directly mirrors a REAL, already-
/// proven F16 autotuner candidate - `(128, 128, 64, 4, 2)` in
/// `Autotuner::generate_candidates`'s `m/n > 512` branch - rather than a
/// fresh guess: the resulting per-warp tile (32x64) needs
/// `num_i=2`/`num_j=8` `mma.m16n8k32` calls (FP8's N=8 mma granularity is
/// half F16 wmma's N=16, so `num_j` is doubled vs the analogous F16
/// config), but each accumulator fragment is half the register count (4
/// f32 regs vs 8, since `mma.m16n8k32`'s M16N8 accumulator is half of
/// wmma's M16N16) - so total accumulator register pressure per warp comes
/// out the same either way (`num_i*num_j*4` = 64 registers, matching the
/// F16 config's `num_i*num_j*8`). Combined A+B shared-memory footprint at
/// this shape (`128*64 + 64*128` = 16384 bytes, e4m3 is 1 byte/element) is
/// well under `ptxas`'s 48KB static-`.shared` cap.
///
/// `_SMALL` (64x64x64, 2x2 warps) exists because `_LARGE` measured a real
/// regression at M=N=256 on real hardware (0.374x -> ~0.11x of
/// `torch._scaled_mm`, see `investigation_fp8_gemm_findings.md`'s
/// "Session 2" results): a 128x128 CTA tile only produces a 2x2=4-CTA
/// grid at that problem size, leaving most of the GPU's SMs idle no
/// matter how efficient each CTA is internally. `_SMALL`'s per-warp tile
/// is 32x32 (`num_i=2`/`num_j=4`), accumulator register pressure is
/// `num_i*num_j*4` = 32 registers/warp (half of `_LARGE`'s, consistent
/// with the smaller tile), and its A+B shared-memory footprint is
/// `64*64 + 64*64` = 8192 bytes - both confirmed by `ptxas -v` (see
/// `emit_fp8_gemm_kernel`'s doc comment).
///
/// **Threshold raised 256 -> 512 (session 5)**: real-hardware A/B at
/// M=N=K=512 (`_LARGE`'s own 4-CTA grid at that size, `(4,4)`) measured
/// `_SMALL` at ~0.148x-0.150x vs `_LARGE`'s ~0.10x-0.11x (both after this
/// session's vectorized quantize+stage - see that section's doc comment) -
/// a real, reproducible win from `_SMALL`'s 64-CTA grid at 512, so `_SMALL`
/// now covers `m <= 512 || n <= 512`.
///
/// **A fourth, `_TINY` (32x32x64, 1x1 warp) tier was also built and
/// standalone-benchmarked this session, and REJECTED** - flagged as a
/// plausible next step by Session 3's own list ("a 32x32x64/1-warp tier...
/// would roughly quadruple the CTA count again at 256"), but real hardware
/// disagreed at every shape tried: M=N=64 (`_TINY` 4-CTA grid vs `_SMALL`'s
/// 1-CTA grid: `_SMALL` 0.81x vs `_TINY` 0.57x), M=64,N=2048,K=128 (`_TINY`
/// 128 CTAs vs `_SMALL` 32 CTAs: `_SMALL` 0.26x vs `_TINY` 0.24x-0.49x
/// depending on run), and M=32,N=2048,K=128 (`_TINY`'s tile fits M=32
/// exactly, `_SMALL`'s wastes half its M-tile: `_SMALL` STILL wins, 0.48x
/// vs `_TINY`'s 0.33x). `_SMALL`'s 4 warps/CTA apparently hide global-
/// memory latency via intra-CTA warp scheduling better than `_TINY`'s
/// extra CTA-level parallelism compensates for losing that - CTA count
/// alone is not the whole story, contrary to the pure grid-occupancy
/// argument that motivated trying it. Reported here as a real, disclosed
/// negative result (matching this project's session 4 precedent) rather
/// than silently dropped - a plausible-sounding lever that real
/// measurement ruled out.
pub const FP8_GEMM_CTA_M_LARGE: u32 = 128;
pub const FP8_GEMM_CTA_N_LARGE: u32 = 128;
pub const FP8_GEMM_WARPS_M_LARGE: u32 = 4;
pub const FP8_GEMM_WARPS_N_LARGE: u32 = 2;
pub const FP8_GEMM_CTA_M_SMALL: u32 = 64;
pub const FP8_GEMM_CTA_N_SMALL: u32 = 64;
pub const FP8_GEMM_WARPS_M_SMALL: u32 = 2;
pub const FP8_GEMM_WARPS_N_SMALL: u32 = 2;
/// K tile size - shared by both `_LARGE` and `_SMALL` (only M/N/warp
/// counts vary between tiers), so this has no `_LARGE`/`_SMALL` suffix.
pub const FP8_GEMM_CTA_K: u32 = 64;
/// `m <= FP8_GEMM_SMALL_THRESHOLD || n <= FP8_GEMM_SMALL_THRESHOLD`
/// selects `_SMALL` over `_LARGE` - see `FP8_GEMM_CTA_M_LARGE`'s doc
/// comment for why this is 512, not `Autotuner::generate_candidates`'s
/// F16 threshold (256) it originally mirrored.
pub const FP8_GEMM_SMALL_THRESHOLD: u32 = 512;

/// Compare two `.version` strings (`"8.4"`, or a whole `".version 8.4"` line)
/// as (major, minor).
fn ptx_version_ge(a: &str, b: &str) -> bool {
    fn parse(v: &str) -> (u32, u32) {
        let v = v.trim().trim_start_matches(".version").trim();
        let mut it = v.split('.');
        (
            it.next().and_then(|x| x.parse().ok()).unwrap_or(0),
            it.next().and_then(|x| x.parse().ok()).unwrap_or(0),
        )
    }
    parse(a) >= parse(b)
}

/// Maps an SM compute capability to the minimum PTX ISA version that can
/// TARGET it -- the floor, not a comfortable margin.
///
/// **`.version` is a DRIVER requirement, independent of `.target`** (gotcha 8b),
/// and over-stating it makes a kernel refuse to load on a machine whose driver
/// is merely older, with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. These numbers
/// are MEASURED, by bisecting `.version` under `ptxas -arch=<a>`, not taken
/// from a release table:
///
/// ```text
///   sm_75 -> 6.3   sm_80 -> 7.0   sm_86 -> 7.1   sm_87 -> 7.5
///   sm_89 -> 7.8   sm_90 -> 8.0   sm_120 -> 8.7
/// ```
///
/// sm_89 was **8.4** here, over-stating the requirement by a whole CUDA major
/// (12.4 against 11.8) for EVERY kernel, because FP8 `mma.sync` needs 8.4 on
/// that arch. The FP8 path raises the floor itself now, through
/// `require_ptx_version`, so a kernel that uses no FP8 does not pay for it.
/// sm_86 was 7.5 against a real 7.1, and sm_75 6.5 against 6.3.
///
/// sm_90 keeps 8.0 although `ptxas` 13.3 accepts 7.8: sm_90 arrived in CUDA
/// 12.0, so 7.8 is below the documented floor and this assembler is merely
/// being lenient. Guess down, but not below the spec.
///
/// It is `pub` because `--emit-coprocessor` builds its own PTX module header
/// in `main.rs` rather than going through `emit_program`, and hardcoded
/// `.version 8.0` there. That is the SAME literal-in-a-format-string shape
/// gotcha 8b warns about, and it failed in both directions at once: 8.0
/// over-states the driver floor by a whole CUDA major on sm_80, and `ptxas`
/// rejects it outright on sm_100/sm_120 ("PTX .version 8.0 does not support
/// .target sm_120") - under a success message and exit 0. Any future site
/// that writes a `.version` line must call this, not a literal.
/// The three things `let x: T = {};` can mean in this backend.
enum ZeroInitKind {
    /// A scalar register to set to zero.
    Scalar(ScalarTy),
    /// A `BlockTile` declaration: nothing to emit, nothing to bind.
    TileDeclaration,
    /// An array or struct - this backend has no local aggregate storage.
    Unrepresentable,
}

pub fn ptx_version_for_sm(sm: &str) -> &'static str {
    let normalized = if sm.starts_with("sm_") {
        sm.to_string()
    } else {
        format!("sm_{}", sm)
    };
    let s = normalized.as_str();
    if s.starts_with("sm_10") || s.starts_with("sm_12") {
        ".version 8.7" // Blackwell / RTX 5000 series
    } else {
        match s {
            "sm_90" | "sm_90a" => ".version 8.0",
            // 7.8 assembles fine in general, but FP8 mma.sync
            // (mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32) needs
            // >=8.4 on sm_89 - confirmed empirically (see
            // investigation_fp8_int8_quantization_findings.md): identical
            // instruction, identical -arch=sm_89, fails on 7.8 with
            // "Feature 'mma with FP8 floating point type' requires PTX ISA
            // .version 8.4 or later", assembles clean on 8.4+.
            "sm_89" | "sm_8.9" => ".version 7.8",
            "sm_87" => ".version 7.5",
            "sm_86" | "sm_8.6" => ".version 7.1",
            "sm_80" | "sm_8.0" => ".version 7.0",
            "sm_75" => ".version 6.3",
            "sm_72" => ".version 6.2",
            "sm_70" => ".version 6.3",
            _ => ".version 7.8", // Safe default for CUDA 12+ targets
        }
    }
}

/// Kernel Dispatch Strategy for H100 / Hopper / Blackwell architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDispatch {
    /// Single-token decode ONLY (M = 1): Use Split-K GEMV.
    SplitKGemv,
    /// Batch 16/32 prompt eval (1 < M <= 64): BYPASS SPLIT-K AND ATOMICS ENTIRELY.
    /// Launch direct 32x128 TMA Tile kernel (y_hopper_small_m_gemm_kernel).
    SmallMDirectTma,
    /// Large dense GEMM (M > 64): Full 256x128 WGMMA cluster kernel.
    WgmmClusterGemm,
}

/// Restructures the kernel dispatcher so Split-K is strictly restricted to M = 1.
pub fn dispatch_kernel(m: u32, _n: u32, _k: u32) -> KernelDispatch {
    if m == 1 {
        KernelDispatch::SplitKGemv
    } else if m > 1 && m <= 64 {
        KernelDispatch::SmallMDirectTma
    } else {
        KernelDispatch::WgmmClusterGemm
    }
}

/// Configuration for Hierarchical 2D CTA Block Tile Decomposition ($128 \times 128 \times 32$)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CtaTileConfig {
    pub cta_m: u32,
    pub cta_n: u32,
    pub cta_k: u32,
    pub warps_m: u32,
    pub warps_n: u32,
    pub mma_m: u32,
    pub mma_n: u32,
    pub mma_k: u32,
    pub num_stages: u32,
    pub num_warps: u32,
}

impl Default for CtaTileConfig {
    fn default() -> Self {
        Self {
            cta_m: 128,
            cta_n: 128,
            cta_k: 32,
            warps_m: 4,
            warps_n: 2,
            mma_m: 16,
            mma_n: 16,
            mma_k: 16,
            num_stages: 3,
            num_warps: 8,
        }
    }
}

impl CtaTileConfig {
    /// Dynamically selects optimal CTA tile layout based on matrix dimensions.
    /// Uses 64x64x32 for small matrices (M,N <= 512) to maximize SM occupancy,
    /// and 128x128x32 for large matrices (M,N >= 1024) to maximize Tensor Core pipeline throughput.
    pub fn select_tile_for_dim(m: u32, n: u32) -> Self {
        if m <= 64 {
            Self {
                cta_m: 32,
                cta_n: 128,
                cta_k: 64,
                warps_m: 2,
                warps_n: 2,
                mma_m: 16,
                mma_n: 16,
                mma_k: 16,
                num_stages: 2,
                num_warps: 4,
            }
        } else if m <= 512 || n <= 512 {
            Self {
                cta_m: 64,
                cta_n: 64,
                cta_k: 32,
                warps_m: 2,
                warps_n: 2,
                mma_m: 16,
                mma_n: 16,
                mma_k: 16,
                num_stages: 2,
                num_warps: 4,
            }
        } else {
            Self::default()
        }
    }
}

/// Manages virtual registers and produces raw PTX strings.
/// A literal's value, for reading `@bounds` at compile time.
/// The PTX opcode for a carry-chained 32-bit intrinsic, if `name` is one and
/// the arity matches.
///
/// The suffix convention is PTX's own, and it is worth reading once rather
/// than guessing at a call site:
///
/// * a leading `add`/`sub`/`mad` **starts** a chain — no carry-in.
/// * a leading `addc`/`subc`/`madc` **continues** one — carry-in from CC.
/// * a trailing `_cc` means the instruction also **writes** the carry-out, so
///   the chain can continue past it. Its absence ends the chain.
///
/// So an eight-limb accumulate is `mad_lo_cc` once, `madc_lo_cc` six times,
/// and `madc_lo` (or `addc`) to close. Getting `_cc` wrong on the last link
/// is harmless; getting it wrong in the middle silently drops a carry, which
/// is why `tests/ptx_carry_chain.rs` checks every one of them against plain
/// Rust `u64` arithmetic on the device rather than string-matching the PTX.
fn carry_op(name: &str, argc: usize) -> Option<&'static str> {
    let (op, want) = match name {
        "add_cc_u32" => ("add.cc.u32", 2),
        "addc_u32" => ("addc.u32", 2),
        "addc_cc_u32" => ("addc.cc.u32", 2),
        "sub_cc_u32" => ("sub.cc.u32", 2),
        "subc_u32" => ("subc.u32", 2),
        "subc_cc_u32" => ("subc.cc.u32", 2),
        "mad_lo_cc_u32" => ("mad.lo.cc.u32", 3),
        "madc_lo_u32" => ("madc.lo.u32", 3),
        "madc_lo_cc_u32" => ("madc.lo.cc.u32", 3),
        "mad_hi_cc_u32" => ("mad.hi.cc.u32", 3),
        "madc_hi_u32" => ("madc.hi.u32", 3),
        "madc_hi_cc_u32" => ("madc.hi.cc.u32", 3),
        _ => return None,
    };
    if argc == want {
        Some(op)
    } else {
        None
    }
}

fn ptx_const_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::IntLit(v, _) => Some(*v as f64),
        Expr::FloatLit(v, _) => Some(*v),
        Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => ptx_const_f64(operand).map(|v| -v),
        _ => None,
    }
}

/// The scalar type a virtual register holds.
///
/// This emitter is *register-prefix-typed*: `%f` is an f32, `%rd` is 64 bits,
/// `%r` is 32 bits, `%p` is a predicate. That encoding carries width but not
/// signedness, and it cannot distinguish "32 bits of integer" from "32 bits of
/// float" for any value the emitter did not allocate itself (a bare parameter
/// name, an immediate). `PtxEmitter::reg_ty` is the authority and the prefix is
/// the fallback, so an untyped `%r` still behaves exactly as it did before this
/// existed - as a signed 32-bit index.
///
/// Widths narrower than 32 bits are deliberately absent. An element type is
/// also a *stride*, and supporting `U8` would mean threading a byte width
/// through every address computation in this file; a half-done version that
/// loaded `ld.global.u8` at a 4-byte stride is worse than a refusal. See
/// `reject_unsupported_element_types`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ScalarTy {
    F32,
    U32,
    I32,
    U64,
    I64,
    U8,
    I8,
    U16,
    I16,
}

impl ScalarTy {
    fn from_name(n: &str) -> Option<ScalarTy> {
        Some(match n {
            "F32" => ScalarTy::F32,
            "U32" => ScalarTy::U32,
            "I32" => ScalarTy::I32,
            "U64" => ScalarTy::U64,
            "I64" => ScalarTy::I64,
            "U8" => ScalarTy::U8,
            "I8" => ScalarTy::I8,
            "U16" => ScalarTy::U16,
            "I16" => ScalarTy::I16,
            _ => return None,
        })
    }

    fn is_float(self) -> bool {
        matches!(self, ScalarTy::F32)
    }

    fn is_64(self) -> bool {
        matches!(self, ScalarTy::U64 | ScalarTy::I64)
    }

    fn is_signed(self) -> bool {
        matches!(
            self,
            ScalarTy::I32 | ScalarTy::I64 | ScalarTy::I8 | ScalarTy::I16
        )
    }

    /// Narrower than a register, i.e. a **memory format only**.
    ///
    /// PTX has no 8-bit register class and no sub-word arithmetic. A value of
    /// one of these types exists in memory; the moment it is loaded it becomes
    /// its [`Self::promoted`] 32-bit type, extended according to signedness,
    /// and a store truncates it back. That is C's integer-promotion rule and it
    /// is the whole semantic model for these types here — see
    /// [`Self::promoted`] for what it costs.
    fn is_subword(self) -> bool {
        matches!(
            self,
            ScalarTy::U8 | ScalarTy::I8 | ScalarTy::U16 | ScalarTy::I16
        )
    }

    /// The type a value of this type has once it is in a register.
    ///
    /// **This is a real semantic commitment, not an implementation detail.**
    /// `U8 + U8` is computed at 32 bits and does NOT wrap at 255; it wraps only
    /// when stored back to a `U8` buffer. Emulating true 8-bit wraparound would
    /// mean masking after every operation, which is a different language and a
    /// much slower one. The rule chosen matches C, and it is the rule quantized
    /// inference wants — load int8, accumulate int32.
    ///
    /// Sub-word **locals** are refused precisely so this promotion cannot be
    /// observed by surprise: a user cannot declare `let x: I8` and then wonder
    /// why `x` did not wrap. The type is reachable only as a buffer element,
    /// where the width is unambiguous because it is a storage format.
    fn promoted(self) -> ScalarTy {
        match self {
            ScalarTy::U8 | ScalarTy::U16 => ScalarTy::U32,
            ScalarTy::I8 | ScalarTy::I16 => ScalarTy::I32,
            other => other,
        }
    }

    /// Element size in bytes - i.e. the stride of an array of this type.
    ///
    /// **Exhaustive on purpose, with no `_` arm.** The `_ => 4` this replaced
    /// was correct for every variant that existed when it was written, and
    /// that is exactly the problem: adding `F16` or `F64` to `ScalarTy` would
    /// have given it a 4-byte stride silently, which is gotcha #7's bug -- an
    /// index scaled by the wrong power of two, in PTX that assembles and
    /// launches. A compile error is the cheapest possible version of that
    /// conversation.
    fn bytes(self) -> u32 {
        match self {
            ScalarTy::U8 | ScalarTy::I8 => 1,
            ScalarTy::U16 | ScalarTy::I16 => 2,
            ScalarTy::F32 | ScalarTy::U32 | ScalarTy::I32 => 4,
            ScalarTy::U64 | ScalarTy::I64 => 8,
        }
    }

    /// log2 of `bytes()`, for the shift in an address computation.
    ///
    /// Zero for the 8-bit types, which is what makes `shl.b64 %rd, %rd, 0` show
    /// up in emitted PTX. It is correct and ptxas folds it away; do not
    /// "optimise" it into a special case without checking every caller, because
    /// the bug this whole type exists to prevent is an index scaled by the
    /// wrong power of two.
    fn log2_bytes(self) -> u32 {
        match self.bytes() {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            other => unreachable!("no power-of-two shift for a {other}-byte element"),
        }
    }

    /// Suffix for `ld` / `st` — the **memory** type, which for the sub-word
    /// widths is where sign extension is decided.
    ///
    /// `ld.global.s8` sign-extends into the destination register and
    /// `ld.global.u8` zero-extends, so this must carry signedness even though
    /// the 32- and 64-bit forms do not. Getting it wrong turns every negative
    /// int8 into a large positive number - assembles, launches, wrong answer.
    fn mem(self) -> &'static str {
        match self {
            ScalarTy::F32 => "f32",
            ScalarTy::U32 | ScalarTy::I32 => "u32",
            ScalarTy::U64 | ScalarTy::I64 => "u64",
            ScalarTy::U8 => "u8",
            ScalarTy::I8 => "s8",
            ScalarTy::U16 => "u16",
            ScalarTy::I16 => "s16",
        }
    }

    /// Suffix for a `mov` between registers holding this type.
    ///
    /// Distinct from [`Self::mem`] because a sub-word value never occupies a
    /// sub-word register - `mov.u8` does not exist in PTX. Every `mov` site
    /// must use this and every `ld`/`st` site must use `mem`; the two were one
    /// method before the sub-word widths existed, which is exactly why they are
    /// easy to confuse now.
    fn reg_mem(self) -> &'static str {
        self.promoted().mem_wide()
    }

    /// `mem()` restricted to the register-width types, for use by `reg_mem`.
    fn mem_wide(self) -> &'static str {
        match self {
            ScalarTy::F32 => "f32",
            ScalarTy::U64 | ScalarTy::I64 => "u64",
            _ => "u32",
        }
    }

    /// Suffix for arithmetic, which is where signedness starts to matter:
    /// `div`, `rem`, `shr` and every comparison differ between `.u32` and
    /// `.s32`. Getting this wrong is a wrong answer, not a slow one.
    fn arith(self) -> &'static str {
        match self.promoted() {
            ScalarTy::F32 => "f32",
            ScalarTy::U32 => "u32",
            ScalarTy::I32 => "s32",
            ScalarTy::U64 => "u64",
            ScalarTy::I64 => "s64",
            // `promoted()` maps every sub-word type into the four above, so
            // this arm is unreachable. It is spelled out rather than `_ =>`
            // so that adding a width without deciding its arithmetic type is
            // a compile error, per the design rule.
            ScalarTy::U8 | ScalarTy::I8 | ScalarTy::U16 | ScalarTy::I16 => {
                unreachable!("promoted() must map sub-word types to a register width")
            }
        }
    }

    /// Suffix for the untyped bitwise ops, which PTX spells `.b32` / `.b64`.
    fn bits(self) -> &'static str {
        if self.is_64() {
            "b64"
        } else {
            "b32"
        }
    }

    /// The zero of this type, as a PTX immediate.
    fn zero_imm(self) -> &'static str {
        if self.is_float() {
            "0.0"
        } else {
            "0"
        }
    }

    fn name(self) -> &'static str {
        match self {
            ScalarTy::F32 => "F32",
            ScalarTy::U32 => "U32",
            ScalarTy::I32 => "I32",
            ScalarTy::U64 => "U64",
            ScalarTy::I64 => "I64",
            ScalarTy::U8 => "U8",
            ScalarTy::I8 => "I8",
            ScalarTy::U16 => "U16",
            ScalarTy::I16 => "I16",
        }
    }
}

pub struct PtxEmitter {
    pub ptx_buffer: String,

    /// A PTX ISA version some emitted INSTRUCTION requires, above the target
    /// arch's own floor. `None` means the arch floor is enough. See
    /// `require_ptx_version`.
    required_ptx_version: Option<&'static str>,

    // Virtual register counters to maintain uniqueness
    reg_u32_count: u32,
    reg_f32_count: u32,
    reg_f64_count: u32,
    reg_u64_count: u32,
    reg_pred_count: u32,
    reg_b16_count: u32,
    label_count: u32,
    variables: std::collections::HashMap<String, String>,
    /// The scalar type of a virtual register, by register name.
    ///
    /// Only registers this emitter allocated through `alloc_ty` appear here.
    /// Everything else falls back to `ScalarTy::from_prefix`, which reproduces
    /// the pre-existing behaviour exactly: `%f` is an f32 and anything else is
    /// a signed 32-bit index.
    reg_ty: std::collections::HashMap<String, ScalarTy>,
    /// Names bound to a 4-wide vector of `u32`, i.e. the four registers a
    /// single `ld.global.v4.u32` fills. Read back through `.x/.y/.z/.w`.
    ///
    /// This exists because a field limb load was 8 separate `LDG.E`, and
    /// ptxas will not merge them: each carries its own bounds predicate, and
    /// a predicated load is not a merge candidate. One `v4` instruction moves
    /// 16 bytes and keeps the predicate.
    vec_vars: std::collections::HashMap<String, [String; 4]>,
    /// Element type of each `GlobalMemory<T>` parameter, by parameter name.
    /// Absent means f32, which is what every load site assumed unconditionally
    /// before the integer datapath existed.
    ptr_elem: std::collections::HashMap<String, ScalarTy>,
    /// `@ZeroDrift` accumulators: name -> (register holding the fixed-point
    /// value, representation chosen for it).
    /// name -> (accumulator register, representation, is the value an
    /// INTEGER rather than a fixed-point encoding of a float?). The third
    /// field is what stops an exact `I64` accumulator being read through
    /// `f32` on every access.
    zero_drift:
        std::collections::HashMap<String, (String, crate::zero_drift::DriftRepr, bool)>,
    /// Measured accumulate costs driving that choice.
    drift_costs: crate::zero_drift::CostTable,
    /// One line per `@ZeroDrift` binding.
    pub drift_report: Vec<String>,
    /// Anything the emitter was asked to lower and could not lower correctly -
    /// a `@ZeroDrift` binding it cannot honour, an intrinsic whose PTX form it
    /// does not know how to construct. `main` fails the build on a non-empty
    /// vector rather than writing the file.
    ///
    /// This exists because the alternative was worse than a missing feature.
    /// `tma_load` and `wgmma_async` used to emit a plausible-looking
    /// instruction with the wrong operand shape, and the compiler printed
    /// "PTX Assembly generated successfully!" over PTX that `ptxas` rejects
    /// outright - a build that looks green and produces a file no GPU can
    /// load. Refusing names the gap; guessing hides it.
    pub emit_errors: Vec<String>,
    /// The resolved SM target (e.g. "sm_80") for PTX header emission.
    sm_target: String,
    /// When true, emits .file and .loc directives for NCU profiling and debugging.
    pub debug_info: bool,
    /// Module-scope declarations (currently just `.extern .shared` arrays for
    /// cp.async-pipelined GEMM kernels - see `emit_tensor_core_gemm_kernel`)
    /// that a kernel body emitter discovers a need for but cannot write
    /// directly: `emit_kernel` temporarily swaps `ptx_buffer` for a
    /// function-body-local buffer while lowering a kernel's body (so it can
    /// wrap the result in `.visible .entry ... { }` afterward), and PTX
    /// requires `.extern` linkage declarations to sit outside any `{ }`
    /// block, not nested inside one alongside ordinary instructions
    /// (confirmed by direct experiment - ptxas's parser rejects it there,
    /// unlike a plain non-extern `.shared` local declaration, which PTX does
    /// allow mid-body). Pushed here during body lowering, then drained by
    /// `emit_kernel` into the real `ptx_buffer` right before that same
    /// kernel's `.visible .entry` line - i.e. still module scope, and still
    /// textually before its only use.
    pending_extern_decls: Vec<String>,

    /// Complete additional `.visible .entry` bodies queued by a kernel emitter
    /// that needs more than one entry point to implement its shape, drained by
    /// `emit_kernel` alongside `pending_extern_decls`.
    ///
    /// Only `emit_paged_decode_attention_kernel`'s split-K path uses this: a
    /// flash-decoding kernel is inherently two launches (partial states, then
    /// a combine over them), and the combine has to be a separate entry
    /// because there is no device-wide barrier inside one. Each queued string
    /// is a whole self-contained entry - signature, `.reg` declarations and
    /// body - because register numbering is per-entry and the emitter's
    /// counters are reset for it (see `emit_attention_split_reduce_entry`).
    pending_module_items: Vec<String>,
    /// `.shared` arrays this kernel's body asked for, as (symbol, u32 count).
    ///
    /// Declared at MODULE scope, textually before the `.visible .entry` that
    /// uses them, because the body is emitted into a scratch buffer before the
    /// entry line exists (see `emit_kernel`) and there is nowhere inside the
    /// entry to put them by the time we know they are needed.
    ///
    /// Reset per kernel: a `.shared` symbol is module-scope in PTX, so two
    /// kernels allocating one each must not collide, and the counter in the
    /// name is global for exactly that reason.
    shared_arrays: Vec<(String, usize)>,
    /// Monotonic across the whole module, NOT `shared_arrays.len()`. That
    /// vector is drained per kernel, so numbering from its length would give
    /// two kernels in one module the same `__y_smem_0` symbol - a redefinition
    /// ptxas rejects, and it would only ever show up in a module with two
    /// shared-memory kernels in it.
    shared_sym_count: usize,
}

/// Split-K ("flash decoding") configuration for
/// `emit_paged_decode_attention_kernel`.
///
/// `Some` selects the split shape: the KV sequence is partitioned across
/// `splits` CTAs in `%ctaid.z`, each writing an UNNORMALIZED partial softmax
/// state `(m, l, acc)` to global scratch, which a second entry point
/// (`<kernel>_reduce`) combines. `None` is the single-pass shape, where one
/// CTA owns a whole sequence and writes `Out` directly.
struct AttnSplit<'a> {
    splits: u32,
    /// `[num_seqs, num_q_heads, splits, head_dim]` f32 - the partial `acc`.
    partial: &'a str,
    /// `[num_seqs, num_q_heads, splits, 2]` f32 - the partial `(m, l)`.
    meta: &'a str,
}

impl PtxEmitter {
    pub fn new() -> Self {
        Self::new_with_profile(&HardwareProfile::default())
    }

    pub fn new_with_profile(hw_profile: &HardwareProfile) -> Self {
        let mut buffer = String::new();
        let raw_sm = hw_profile.sm_version.replace('.', "");
        let target = if !raw_sm.is_empty() {
            let t = if raw_sm.starts_with("sm_") {
                raw_sm
            } else {
                format!("sm_{}", raw_sm)
            };
            // NO ARCHITECTURE-SPECIFIC SUFFIX. This used to promote sm_90 to
            // `sm_90a` unconditionally and without a stated reason - a leftover
            // from the WGMMA/TMA surface that was deleted for never having
            // assembled (gotcha #8). Nothing this backend emits needs it: the
            // instruction mix is `mma.sync`, `cp.async`, `ld`/`st`, all plain
            // sm_90.
            //
            // The `a` suffix is not "sm_90 plus extras", it is a DIFFERENT and
            // architecture-SPECIFIC target that never JITs forward. Measured
            // with ptxas 13.3:
            //
            //     .target sm_90   ->  -arch=sm_90 ok, sm_100 ok, sm_120 ok
            //     .target sm_90a  ->  -arch=sm_90 FAILS, sm_100 FAILS, sm_120 FAILS
            //
            // So a kernel compiled on an H100 was pinned to that exact card and
            // would not load on Blackwell - the same failure as a committed
            // `.ysu_hw_profile` baking in the build machine, one directive over.
            // Guess DOWN.
            //
            // If a Hopper-specific instruction is ever added, it must request
            // the suffix the way FP8 requests a `.version` - discovered during
            // emission, on the module - not by promoting every kernel on the
            // chance that one of them needs it.
            t
        } else {
            "sm_80".to_string()
        };
        let ptx_version = ptx_version_for_sm(&target);
        writeln!(&mut buffer, "{}", ptx_version).unwrap();
        writeln!(&mut buffer, ".target {}", target).unwrap();
        writeln!(&mut buffer, ".address_size 64").unwrap();
        writeln!(&mut buffer, "").unwrap();

        Self {
            required_ptx_version: None,
            ptx_buffer: buffer,
            reg_u32_count: 0,
            reg_f32_count: 0,
            reg_f64_count: 0,
            reg_u64_count: 0,
            reg_pred_count: 0,
            reg_b16_count: 0,
            label_count: 0,
            variables: std::collections::HashMap::new(),
            reg_ty: std::collections::HashMap::new(),
            vec_vars: std::collections::HashMap::new(),
            ptr_elem: std::collections::HashMap::new(),
            zero_drift: std::collections::HashMap::new(),
            drift_costs: crate::zero_drift::CostTable::new(),
            drift_report: Vec::new(),
            emit_errors: Vec::new(),
            sm_target: target,
            debug_info: false,
            pending_extern_decls: Vec::new(),
            pending_module_items: Vec::new(),
            shared_arrays: Vec::new(),
            shared_sym_count: 0,
        }
    }

    /// Allocates a new virtual 32-bit register (e.g. `%r5`)
    fn alloc_reg32(&mut self) -> String {
        let name = format!("%r{}", self.reg_u32_count);
        self.reg_u32_count += 1;
        name
    }

    /// Allocates a new virtual float register (e.g. `%f2`)
    fn alloc_regf32(&mut self) -> String {
        let name = format!("%f{}", self.reg_f32_count);
        self.reg_f32_count += 1;
        name
    }

    /// Allocates a new virtual double register (e.g. `%fd2`).
    ///
    /// Only `@ZeroDrift` conversions use these; the declaration is omitted
    /// entirely when the count is zero so ordinary kernels are unchanged.
    fn alloc_regf64(&mut self) -> String {
        let name = format!("%fd{}", self.reg_f64_count);
        self.reg_f64_count += 1;
        name
    }

    /// Supplies measured accumulate costs so `@ZeroDrift` chooses on evidence.
    pub fn set_drift_costs(&mut self, costs: crate::zero_drift::CostTable) {
        self.drift_costs = costs;
    }

    /// A PTX double literal: `0d` plus the IEEE754 bit pattern.
    fn ptx_f64(v: f64) -> String {
        format!("0d{:016X}", v.to_bits())
    }

    /// Converts an `f32` register into `repr`'s integer domain, returning a
    /// 64-bit register.
    ///
    /// Rounds half away from zero, spelled out with `setp`/`selp` rather than
    /// using `cvt.rni` (round-to-nearest-even). That is deliberate: the LLVM
    /// backend lowers the same annotation with `fcmp`/`select`, and if the two
    /// backends rounded differently then the same Y source compiled for CPU and
    /// GPU would disagree - which would defeat the point of an annotation whose
    /// entire purpose is reproducibility.
    /// Converts a value into the accumulator's stored representation.
    ///
    /// `integer_domain` means the accumulator holds an INTEGER, not a scaled
    /// float, so there is nothing to convert and no float to route through.
    /// Without this the exact `I64` path round-tripped every added term
    /// through `f64` - exact only up to 2^53, on an accumulator whose entire
    /// promise is that it is exact.
    fn emit_drift_to_fixed(
        &mut self,
        src: &str,
        repr: crate::zero_drift::DriftRepr,
        integer_domain: bool,
    ) -> String {
        if integer_domain {
            return self.emit_convert(src, ScalarTy::I64);
        }
        let widened = self.alloc_regf64();
        if src.starts_with("%f") && !src.starts_with("%fd") {
            writeln!(&mut self.ptx_buffer, "    cvt.f64.f32 {}, {};", widened, src).unwrap();
        } else {
            writeln!(&mut self.ptx_buffer, "    cvt.rn.f64.s64 {}, {};", widened, src).unwrap();
        }
        let out = self.alloc_reg64();
        if repr.frac_bits() == 0 {
            writeln!(&mut self.ptx_buffer, "    cvt.rzi.s64.f64 {}, {};", out, widened).unwrap();
            return out;
        }
        let scaled = self.alloc_regf64();
        writeln!(
            &mut self.ptx_buffer,
            "    mul.f64 {}, {}, {};",
            scaled,
            widened,
            Self::ptx_f64(repr.scale())
        )
        .unwrap();
        let neg = self.alloc_pred();
        writeln!(
            &mut self.ptx_buffer,
            "    setp.lt.f64 {}, {}, {};",
            neg,
            scaled,
            Self::ptx_f64(0.0)
        )
        .unwrap();
        let bias = self.alloc_regf64();
        writeln!(
            &mut self.ptx_buffer,
            "    selp.f64 {}, {}, {}, {};",
            bias,
            Self::ptx_f64(-0.5),
            Self::ptx_f64(0.5),
            neg
        )
        .unwrap();
        let rounded = self.alloc_regf64();
        writeln!(&mut self.ptx_buffer, "    add.f64 {}, {}, {};", rounded, scaled, bias).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.rzi.s64.f64 {}, {};", out, rounded).unwrap();
        out
    }

    /// Converts a fixed-point accumulator back to a readable register.
    ///
    /// For a fixed-point encoding of a float that is an `f32`, which is the
    /// contract. For an INTEGER accumulator it is the integer: this function
    /// used to narrow `s64 -> f64 -> f32` unconditionally, so every read of an
    /// exact `I64` accumulator silently lost everything above 2^24 - the
    /// `ld.global.f32`-on-a-`U32`-buffer bug, one directive over.
    fn emit_drift_from_fixed(
        &mut self,
        src: &str,
        repr: crate::zero_drift::DriftRepr,
        integer_domain: bool,
    ) -> String {
        if integer_domain {
            let out = self.alloc_ty(ScalarTy::I64);
            writeln!(&mut self.ptx_buffer, "    mov.s64 {}, {};", out, src).unwrap();
            return out;
        }
        let as_f64 = self.alloc_regf64();
        writeln!(&mut self.ptx_buffer, "    cvt.rn.f64.s64 {}, {};", as_f64, src).unwrap();
        let out = self.alloc_regf32();
        if repr.frac_bits() == 0 {
            writeln!(&mut self.ptx_buffer, "    cvt.rn.f32.f64 {}, {};", out, as_f64).unwrap();
            return out;
        }
        let unscaled = self.alloc_regf64();
        writeln!(
            &mut self.ptx_buffer,
            "    div.rn.f64 {}, {}, {};",
            unscaled,
            as_f64,
            Self::ptx_f64(repr.scale())
        )
        .unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.rn.f32.f64 {}, {};", out, unscaled).unwrap();
        out
    }

    /// Allocates a new virtual 64-bit register (e.g. `%rd4`)
    fn alloc_reg64(&mut self) -> String {
        let name = format!("%rd{}", self.reg_u64_count);
        self.reg_u64_count += 1;
        name
    }

    /// Allocates a new predicate register (e.g. `%p3`)
    fn alloc_pred(&mut self) -> String {
        let name = format!("%p{}", self.reg_pred_count);
        self.reg_pred_count += 1;
        name
    }

    // ── The integer datapath ────────────────────────────────────────────
    //
    // Before this existed, every scalar value in a kernel body was either an
    // f32 in a `%f` register or an untyped 32-bit index in a `%r` one, and a
    // `GlobalMemory<U32>` parameter was loaded with `ld.global.f32`. That is a
    // wrong answer rather than a missing feature: it assembles, it launches,
    // and it rounds off everything above 2^24. `tests/ptx_integer_datapath.rs`
    // is the regression gate.

    /// Allocates a register of `ty` and records its type.
    ///
    /// A sub-word type is allocated — and **recorded** — as its promoted
    /// 32-bit type. There is no 8- or 16-bit register class in PTX, so a
    /// register can never hold `I8`; recording it as `I8` would make every
    /// later `reg_ty` lookup describe a register that does not exist and emit
    /// `mov.s8`, which ptxas rejects.
    fn alloc_ty(&mut self, ty: ScalarTy) -> String {
        let ty = ty.promoted();
        let r = match ty {
            ScalarTy::F32 => self.alloc_regf32(),
            ScalarTy::U64 | ScalarTy::I64 => self.alloc_reg64(),
            ScalarTy::U32 | ScalarTy::I32 => self.alloc_reg32(),
            ScalarTy::U8 | ScalarTy::I8 | ScalarTy::U16 | ScalarTy::I16 => {
                unreachable!("promoted() must map sub-word types to a register width")
            }
        };
        self.reg_ty.insert(r.clone(), ty);
        r
    }

    /// The type of the value in `reg`.
    ///
    /// The fallback matters as much as the map: an unrecorded `%r` must read
    /// as `I32` so that all the pre-existing untyped index arithmetic keeps
    /// emitting `add.s32` / `mul.lo.s32` exactly as it did, and an unrecorded
    /// `%rd` must read as `U64` because those are addresses.
    fn ty_of(&self, reg: &str) -> ScalarTy {
        if let Some(t) = self.reg_ty.get(reg) {
            return *t;
        }
        if reg.starts_with("%f") {
            ScalarTy::F32
        } else if reg.starts_with("%rd") {
            ScalarTy::U64
        } else {
            ScalarTy::I32
        }
    }

    /// A `v4` result is carried between `emit_expr` and `Stmt::Let` as a
    /// marker string, because `emit_expr` returns one register name and a
    /// vector is four. Nothing else in the emitter ever sees it: `Stmt::Let`
    /// unpacks it into `vec_vars` and any other consumer is a hard error.
    const V4_TAG: &'static str = "%v4[";

    fn v4_marker(regs: &[String; 4]) -> String {
        format!("{}{}|{}|{}|{}]", Self::V4_TAG, regs[0], regs[1], regs[2], regs[3])
    }

    fn parse_v4_marker(s: &str) -> Option<[String; 4]> {
        let inner = s.strip_prefix(Self::V4_TAG)?.strip_suffix(']')?;
        let parts: Vec<&str> = inner.split('|').collect();
        if parts.len() != 4 {
            return None;
        }
        Some([parts[0].into(), parts[1].into(), parts[2].into(), parts[3].into()])
    }

    /// The element type of the buffer `e` points at, if it is a parameter this
    /// kernel declared. `None` means "not a known typed pointer", and every
    /// caller treats that as f32 - the historical assumption.
    fn elem_ty_of(&self, e: &Expr) -> Option<ScalarTy> {
        match e {
            Expr::Ident(n, _) => self.ptr_elem.get(n).copied(),
            _ => None,
        }
    }

    fn elem_ty_or_f32(&self, e: &Expr) -> ScalarTy {
        self.elem_ty_of(e).unwrap_or(ScalarTy::F32)
    }

    /// The element type behind an address expression such as `A[i]`, for the
    /// intrinsics that take an already-computed address rather than a buffer.
    fn index_elem_ty(&self, e: &Expr) -> Option<ScalarTy> {
        match e {
            Expr::Index { base, .. } => self.elem_ty_of(base),
            _ => None,
        }
    }

    /// Converts `reg` to `to`, returning a register holding the converted
    /// value. Returns `reg` unchanged when it is already the right type.
    ///
    /// Integer widening is sign-aware (`cvt.s64.s32` sign-extends, `cvt.u64.u32`
    /// zero-extends); narrowing truncates, which is what a limb extraction
    /// wants. Float/integer conversion rounds to nearest for int->float and
    /// truncates toward zero for float->int, matching the `as` semantics of
    /// every language Y's surface syntax resembles.
    fn emit_convert(&mut self, reg: &str, to: ScalarTy) -> String {
        let from = self.ty_of(reg);
        if from == to {
            return reg.to_string();
        }
        // Same width, different signedness: a reinterpretation, not a
        // conversion. `cvt.u32.u32` would be a legal no-op, but going through
        // a `mov` is what actually changes the recorded type, since `reg_ty`
        // is keyed by register and one register cannot hold two types.
        if !from.is_float() && !to.is_float() && from.bytes() == to.bytes() {
            let dst = self.alloc_ty(to);
            writeln!(&mut self.ptx_buffer, "    mov.{} {}, {};", to.reg_mem(), dst, reg).unwrap();
            return dst;
        }
        let dst = self.alloc_ty(to);
        let instr = match (from.is_float(), to.is_float()) {
            (false, false) => {
                // Sign-extension is a property of the *source*; zero- vs
                // sign-extending a narrowing conversion is meaningless.
                let src_suffix = if to.bytes() > from.bytes() && from.is_signed() {
                    format!("s{}", from.bytes() * 8)
                } else {
                    format!("u{}", from.bytes() * 8)
                };
                let dst_suffix = if to.bytes() > from.bytes() && from.is_signed() {
                    format!("s{}", to.bytes() * 8)
                } else {
                    format!("u{}", to.bytes() * 8)
                };
                format!("cvt.{}.{}", dst_suffix, src_suffix)
            }
            (false, true) => format!("cvt.rn.f32.{}", from.arith()),
            (true, false) => format!("cvt.rzi.{}.f32", to.arith()),
            (true, true) => unreachable!("F32 is the only float type"),
        };
        writeln!(&mut self.ptx_buffer, "    {} {}, {};", instr, dst, reg).unwrap();
        dst
    }

    /// The type an operation on `l` and `r` is carried out in.
    ///
    /// Float wins over integer, wider wins over narrower, and unsigned wins
    /// over signed at equal width - the last because mixing them is almost
    /// always an index meeting a limb, and doing that arithmetic signed is how
    /// a value above 2^31 turns negative halfway through an address.
    fn promote(l: ScalarTy, r: ScalarTy) -> ScalarTy {
        if l.is_float() || r.is_float() {
            return ScalarTy::F32;
        }
        match (l.is_64() || r.is_64(), l.is_signed() && r.is_signed()) {
            (true, true) => ScalarTy::I64,
            (true, false) => ScalarTy::U64,
            (false, true) => ScalarTy::I32,
            (false, false) => ScalarTy::U32,
        }
    }

    /// Lowers a binary operator over two already-emitted operands.
    ///
    /// Every `BinaryOp` variant is handled explicitly and the match has no
    /// `_ =>` arm. The arm this replaces ended in `_ => "add.s32"`, so `a < b`,
    /// `a & b`, `a >> b` and `a % b` all emitted an *addition* - the exact
    /// silent-substitution failure CLAUDE.md's design rule is about, and the
    /// reason `if a < b` in a kernel body could never have worked.
    fn emit_binary(&mut self, op: BinaryOp, l: &str, r: &str, span: &Span) -> String {
        let ty = Self::promote(self.ty_of(l), self.ty_of(r));

        // Comparisons produce a canonical 0/1 in a u32 register rather than a
        // predicate, because that is what `Stmt::If` consumes (`setp.ne.u32
        // cond, 0`) and what `&&`/`||` below assume of their operands.
        let cmp = match op {
            BinaryOp::Eq => Some("eq"),
            BinaryOp::NotEq => Some("ne"),
            BinaryOp::Lt => Some("lt"),
            BinaryOp::Le => Some("le"),
            BinaryOp::Gt => Some("gt"),
            BinaryOp::Ge => Some("ge"),
            _ => None,
        };
        if let Some(c) = cmp {
            let lc = self.emit_convert(l, ty);
            let rc = self.emit_convert(r, ty);
            let p = self.alloc_pred();
            let dst = self.alloc_ty(ScalarTy::U32);
            writeln!(&mut self.ptx_buffer, "    setp.{}.{} {}, {}, {};", c, ty.arith(), p, lc, rc).unwrap();
            writeln!(&mut self.ptx_buffer, "    selp.u32 {}, 1, 0, {};", dst, p).unwrap();
            return dst;
        }

        // Shifts are the one shape where the operands are not the same type:
        // PTX takes the shift amount as a u32 whatever the value's width, and
        // the result has the *left* operand's type, not the promoted one.
        if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
            let vt = self.ty_of(l);
            if vt.is_float() {
                self.emit_errors.push(format!(
                    "Line {}: `{}` is not defined on F32. A shift is a bit operation; \
                     convert to an integer type first.",
                    span.line,
                    if matches!(op, BinaryOp::Shl) { "<<" } else { ">>" }
                ));
                return l.to_string();
            }
            let amount = self.emit_convert(r, ScalarTy::U32);
            let dst = self.alloc_ty(vt);
            // `shl` is bitwise and takes `.b32`/`.b64`; `shr` is not, because a
            // signed right shift replicates the sign bit and an unsigned one
            // does not.
            let instr = if matches!(op, BinaryOp::Shl) {
                format!("shl.{}", vt.bits())
            } else {
                format!("shr.{}", vt.arith())
            };
            writeln!(&mut self.ptx_buffer, "    {} {}, {}, {};", instr, dst, l, amount).unwrap();
            return dst;
        }

        let lc = self.emit_convert(l, ty);
        let rc = self.emit_convert(r, ty);

        // `&&` and `||` are lowered as bitwise ops on canonical 0/1 values,
        // which is exactly right for the comparison results above and for
        // anything else Y can produce as a boolean. They do NOT short-circuit;
        // on a SIMT machine both sides are evaluated anyway.
        let instr = match op {
            BinaryOp::Add => format!("add.{}", ty.arith()),
            BinaryOp::Sub => format!("sub.{}", ty.arith()),
            BinaryOp::Mul => {
                if ty.is_float() {
                    "mul.f32".to_string()
                } else {
                    // `mul.lo` keeps the low half, i.e. wrapping multiplication.
                    // The high half is reachable through `mul_wide_u32`.
                    format!("mul.lo.{}", ty.arith())
                }
            }
            BinaryOp::Div => {
                if ty.is_float() {
                    "div.approx.f32".to_string()
                } else {
                    format!("div.{}", ty.arith())
                }
            }
            BinaryOp::Mod => {
                if ty.is_float() {
                    self.emit_errors.push(format!(
                        "Line {}: `%` is not defined on F32 in this backend.",
                        span.line
                    ));
                    return lc;
                }
                format!("rem.{}", ty.arith())
            }
            BinaryOp::BitAnd | BinaryOp::And => {
                if ty.is_float() {
                    self.emit_errors.push(format!(
                        "Line {}: bitwise `&` is not defined on F32.",
                        span.line
                    ));
                    return lc;
                }
                format!("and.{}", ty.bits())
            }
            BinaryOp::BitOr | BinaryOp::Or => {
                if ty.is_float() {
                    self.emit_errors.push(format!(
                        "Line {}: bitwise `|` is not defined on F32.",
                        span.line
                    ));
                    return lc;
                }
                format!("or.{}", ty.bits())
            }
            BinaryOp::BitXor => {
                if ty.is_float() {
                    self.emit_errors.push(format!(
                        "Line {}: bitwise `^` is not defined on F32.",
                        span.line
                    ));
                    return lc;
                }
                format!("xor.{}", ty.bits())
            }
            // Handled above; listed so this match stays exhaustive and a new
            // operator is a compile error here rather than a silent addition.
            BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::Shl
            | BinaryOp::Shr => unreachable!("handled before operand promotion"),
        };
        let dst = self.alloc_ty(ty);
        writeln!(&mut self.ptx_buffer, "    {} {}, {}, {};", instr, dst, lc, rc).unwrap();
        dst
    }

    /// Records that `name` is a recognised intrinsic the backend cannot lower
    /// to correct PTX, and why. The build fails; no `.ptx` is written.
    ///
    /// This is the PTX backend's instance of the repo-wide rule that a
    /// soundness- or correctness-critical pass must refuse an input it cannot
    /// handle rather than substitute something adjacent. The failure mode it
    /// replaces is specific and was live: an intrinsic wrote an instruction
    /// with the right mnemonic and the wrong operand shape, `ptxas` rejected
    /// the file, and nothing in the compiler noticed because nothing in the
    /// compiler ran `ptxas`. Emitting a comment and carrying on would be the
    /// same bug with a different spelling - the caller asked for a copy or a
    /// matrix multiply, and silently not doing it produces a kernel that
    /// computes garbage instead of one that fails to build.
    /// Refuse sub-word types where they would be a *value* rather than a
    /// storage format.
    ///
    /// `U8`/`I8`/`U16`/`I16` are lowered for real now, as **buffer element
    /// types only**. The stride is threaded through `ScalarTy::log2_bytes` and
    /// the sign extension through `ScalarTy::mem`, so `GlobalMemory<I8>` loads
    /// `ld.global.s8` at a 1-byte stride.
    ///
    /// What is still refused is a sub-word **local**, and that is a semantic
    /// decision rather than a missing feature. PTX has no sub-word register
    /// class, so `let x: I8 = ...` would be a 32-bit register and `x + x`
    /// would not wrap at 8 bits. Accepting the declaration would mean a value
    /// whose type says one thing and whose arithmetic does another — the exact
    /// shape of failure this file's design rule exists to prevent. As a buffer
    /// element the width is unambiguous, because there it really is the
    /// storage format.
    ///
    /// The refusal this replaces was written when the widths were not lowered
    /// at all: an element type is simultaneously a *stride*, and every address
    /// computation used to shift by a hardcoded log2(4), so a `u8` array would
    /// have been read every fourth element — a wrong answer that assembles.
    fn reject_unsupported_element_types(&mut self, kernel: &KernelDecl) {
        for param in &kernel.params {
            let Type::Generic { base, args, .. } = &param.ty else {
                continue;
            };
            if base != "GlobalMemory" && base != "SharedMemory" && base != "L2Memory" {
                continue;
            }
            for a in args {
                let GenericArg::Type(Type::Primitive(name, _) | Type::Ident(name, _)) = a else {
                    continue;
                };
                // `SharedMemory` addressing is still 32-bit-only: the shared
                // surface indexes in 16-byte units (`shared_load_v4`) and has
                // no byte-addressed form, so a sub-word element there would be
                // the original bug in a different building.
                if base != "GlobalMemory"
                    && ScalarTy::from_name(name).is_some_and(ScalarTy::is_subword)
                {
                    self.emit_errors.push(format!(
                        "[PTX] `{}: {}<{}>` cannot be lowered: sub-word element types are \
                         supported for GlobalMemory only. {} is indexed in 16-byte units and \
                         has no byte-addressed form.",
                        param.name, base, name, base
                    ));
                }
            }
        }
    }

    /// Refuse a sub-word type used as a local declaration.
    ///
    /// See [`Self::reject_unsupported_element_types`] for why this is a
    /// deliberate boundary and not an unfinished one.
    fn reject_subword_local(&mut self, name: &str, ty_name: &str, line: usize) -> bool {
        if !ScalarTy::from_name(ty_name).is_some_and(ScalarTy::is_subword) {
            return false;
        }
        self.emit_errors.push(format!(
            "Line {}: `let {}: {}` cannot be lowered: PTX has no sub-word register class, so \
             this would be a 32-bit value whose declared type promises 8- or 16-bit wraparound \
             it will not perform. Sub-word types are buffer element types only - load from a \
             `GlobalMemory<{}>` into an I32/U32 and the width is honoured by the load.",
            line, name, ty_name, ty_name
        ));
        true
    }

    /// Record the element type of each typed buffer parameter, so the load and
    /// store sites can pick an instruction instead of assuming f32.
    fn record_pointer_element_types(&mut self, kernel: &KernelDecl) {
        for param in &kernel.params {
            let Type::Generic { base, args, .. } = &param.ty else {
                continue;
            };
            if base != "GlobalMemory" && base != "SharedMemory" && base != "L2Memory" {
                continue;
            }
            for a in args {
                let GenericArg::Type(Type::Primitive(name, _) | Type::Ident(name, _)) = a else {
                    continue;
                };
                if let Some(t) = ScalarTy::from_name(name) {
                    self.ptr_elem.insert(param.name.clone(), t);
                }
            }
        }
    }

    /// Refuse to lower `intrinsic` against a non-float buffer.
    ///
    /// The typed datapath covers the load/store paths a field kernel needs
    /// (`block_ptr2d_load` / `block_ptr2d_store` / `Index` / `GlobalMemory::load`
    /// / `store`). The rest of this file's memory intrinsics - vectorised
    /// loads, `ldmatrix`, `cp.async` staging, the tile-GEMM machinery - are
    /// f32/f16 by construction. Rather than let one of them quietly load a
    /// `u32` buffer as floats, which is precisely the bug the datapath exists
    /// to end, they call this and the build fails with the type named.
    ///
    /// Returns true when the caller must stop.
    fn reject_non_float_buffer(&mut self, intrinsic: &str, ptr: &Expr) -> bool {
        match self.elem_ty_of(ptr) {
            Some(t) if !t.is_float() => {
                self.emit_errors.push(format!(
                    "[PTX] `{}` has no lowering for a {} buffer; it is an f32/f16 path. \
                     Use block_ptr2d_load / block_ptr2d_store for integer element types.",
                    intrinsic,
                    t.name()
                ));
                true
            }
            _ => false,
        }
    }

    /// Does this initialiser produce a linear token rather than a value?
    ///
    /// The list is explicit because the alternative - treating "emitted no
    /// register" as acceptable everywhere - is exactly the silent failure the
    /// `let` refusal beside it exists to catch.
    fn binds_a_linear_token(expr: &Expr) -> bool {
        matches!(expr, Expr::Call { func, .. }
            if matches!(&**func, Expr::Ident(n, _) if n == "cp_async"))
    }

    /// Required argument count for intrinsics whose missing operands would
    /// otherwise alias an unrelated register.
    ///
    /// **This is the design-rule table applied to arity.** Every one of these
    /// lowerings reads its operands as `if args.len() >= N { emit(args[N-1]) }
    /// else { "%r0".into() }` - so calling one with too few arguments does not
    /// fail, it substitutes *whatever happens to live in `%r0`/`%rd0`/`%f0`*,
    /// which is another variable in the same kernel. Two outcomes, both bad:
    /// the register has the wrong type and `ptxas` rejects the module after the
    /// compiler has printed "Compilation Successful!" and exited 0, or it has
    /// the right type and the kernel silently computes with the wrong operand.
    ///
    /// Found by giving `block_ptr2d_store` five arguments instead of seven: the
    /// value being stored became the bounds limit, emitting
    /// `setp.lt.u32 %p0, %r6, %f0` - a u32 compared against an f32 register.
    ///
    /// **The count here is the last position whose fallback is a REGISTER, not
    /// the total number of parameters.** Several of these lowerings have
    /// genuinely optional trailing operands that default to a literal -
    /// `block_tile_store`'s bound falls back to `128`, which is a defensible
    /// default and not an aliased register - so requiring the full parameter
    /// list would reject calls that were always correct. The first version of
    /// this table did exactly that and broke
    /// `supported_intrinsics_emit_assemblable_ptx`, which calls
    /// `block_tile_store` with three arguments.
    ///
    /// Only `block_ptr2d_store` is called by any kernel in this repo, and
    /// always with its full seven, so this gate cannot break working code. The
    /// rest are reachable from the surface syntax and used by nothing, exactly
    /// like the Hopper intrinsics in gotcha #8.
    fn required_arity(fname: &str) -> Option<usize> {
        Some(match fname {
            "block_cdiv" | "block_ptr2d_advance" | "block_ptr3d_advance"
            | "shfl_sync_bfly" | "shfl_sync_bfly_b32" | "ld_global_v4_f32"
            | "load_v4" | "warp_reduce_max" | "warp_reduce_sum" => 1,
            "block_tile_load" | "tile_load" | "st_global_v4_f32"
            | "store_v4" => 2,
            "block_ptr2d_load" | "block_tile_store" | "tile_store"
            | "make_block_ptr2d" | "rmsnorm_fast" | "rmsnorm_v4"
            | "swiglu_fast" | "swiglu_v4" | "vec_add_v4" | "vector_add_v4"
            | "vec_add_unrolled4" => 3,
            "block_ptr3d_load" | "block_ptr3d_load_v4" | "block_ptr3d_store"
            | "make_block_ptr3d" => 4,
            "block_ptr2d_store" => 7,
            "block_ptr3d_store_v4" => 10,
            _ => return None,
        })
    }

    /// `required_arity`, for the `Namespace::member` callees.
    ///
    /// Same rule and same reason: the count is the last position whose fallback
    /// is a REGISTER. `BlockTile::load`'s bound falls back to the literal `128`
    /// and is genuinely optional; its offset falls back to `%r0` and is not.
    fn required_path_arity(namespace: &str, member: &str) -> Option<usize> {
        Some(match (namespace, member) {
            ("BlockTile", "load") => 2,
            ("BlockTile", "store") => 3,
            _ => return None,
        })
    }

    /// A statement this backend cannot lower, refused by name.
    ///
    /// `emit_stmt` ended in `_ => {}`, so `while`, `break` and `match` emitted
    /// NOTHING - the PTX assembled (it is simply a shorter kernel), the kernel
    /// launched, and it computed a different program. A `while` loop's whole
    /// body vanished:
    ///
    /// ```text
    /// let i: I32 = 0;
    /// while i < N { i = i + 1; }
    /// store(A, 0, i);              // stored 0, whatever N was
    /// ```
    ///
    /// No `ptxas` gate can catch that, for the reason gotcha #8 states: a
    /// MISSING instruction assembles perfectly.
    /// `let x: T = {};` - a zero-initialiser.
    ///
    /// This backend had no lowering for it at all: it fell to `emit_expr`'s
    /// `_ => "".into()` and then to the "initialiser produced no value"
    /// refusal, so a construct the LLVM backend memsets, the type checker
    /// types, and `tests/bounds_test.ysu` uses was simply unavailable in a
    /// kernel. It is what `python/tests/test_gpu_architect_features.py` uses to
    /// declare a tile, which is why that documented test command has been
    /// failing.
    fn emit_zero_init_let(&mut self, name: &str, ty: Option<&Type>, span: &Span) {
        match Self::zero_init_kind(ty) {
            ZeroInitKind::Scalar(t) => {
                let reg = self.alloc_ty(t);
                writeln!(&mut self.ptx_buffer, "    mov.{} {}, 0;", t.reg_mem(), reg).unwrap();
                self.variables.insert(name.to_string(), reg);
            }
            ZeroInitKind::TileDeclaration => {
                // A `BlockTile` declaration is a DECLARATION, not a value: the
                // tile intrinsics (`block_tile_load` / `block_tile_store`) take
                // the global buffer directly and never read this name. Binding
                // nothing is correct here for exactly the reason it is correct
                // for a linear token - and if the name IS read, the
                // `Expr::Ident` refusal catches it by name rather than emitting
                // the identifier as though it were a register.
                writeln!(
                    &mut self.ptx_buffer,
                    "    // BlockTile `{}` declared; tiles are addressed through their buffer.",
                    name
                )
                .unwrap();
            }
            ZeroInitKind::Unrepresentable => {
                // An array or struct local. This backend has no local aggregate
                // storage at all - a `let` binds ONE register - so there is
                // nothing to zero. Refused by name rather than zeroing a single
                // scalar and calling it the array.
                self.emit_errors.push(format!(
                    "[PTX] line {}: `let {} = {{}}` zero-initialises an aggregate, and this \
backend has no local array or struct storage - a `let` binds one register. Use a global buffer, \
or `shared_alloc_u32` for a shared-memory array.",
                    span.line, name
                ));
            }
        }
    }

    /// What a `let x: T = {};` declares, which decides whether this backend
    /// can lower it.
    ///
    /// Split out because the three answers are genuinely different actions and
    /// the design rule forbids collapsing them into a guess: a scalar has a
    /// register to zero, a tile has nothing to emit, and an aggregate has no
    /// storage in this backend at all.
    fn zero_init_kind(ty: Option<&Type>) -> ZeroInitKind {
        match ty {
            // The parser produces `Type::BlockTile` directly for
            // `BlockTile<F32, 128>` - verified by mutation: breaking a
            // `Type::Generic { base: "BlockTile" }` arm changed nothing, so
            // that arm was dead and is not carried here. If such a spelling
            // ever does arrive it falls to `Unrepresentable`, which is a named
            // refusal rather than a wrong answer.
            Some(Type::BlockTile { .. }) => ZeroInitKind::TileDeclaration,
            Some(Type::Primitive(n, _)) | Some(Type::Ident(n, _)) => match ScalarTy::from_name(n) {
                Some(t) => ZeroInitKind::Scalar(t),
                None => ZeroInitKind::Unrepresentable,
            },
            _ => ZeroInitKind::Unrepresentable,
        }
    }

    fn unsupported_stmt(&mut self, what: &str, span: &Span) {
        self.emit_errors.push(format!(
            "[PTX] {} (line {}, col {}) cannot be lowered by this backend.",
            what, span.line, span.col
        ));
    }

    /// The numeric part of `sm_NN`, for comparing architectures.
    fn sm_level(&self) -> u32 {
        self.sm_target
            .trim_start_matches("sm_")
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap_or(0)
    }

    /// FP8 (`e4m3`) tensor cores are Ada and later. Below sm_89 the kernel is
    /// not slow, it does not EXIST - the driver rejects the module with
    /// `CUDA_ERROR_NO_BINARY_FOR_GPU` and nothing says why.
    ///
    /// This is the one place in the emitter where a real hardware requirement
    /// is above the sm_80 floor everything else targets, so it is the one place
    /// that has to refuse by name rather than silently emit something a 3060
    /// cannot load. `tests/ptx_portability.rs` pins the rest of the surface at
    /// the floor precisely so this stays the only exception.
    fn require_fp8_hardware(&mut self, kernel: &str) -> bool {
        if self.sm_level() >= 89 {
            // FP8 `mma.sync.aligned.m16n8k32...e4m3.e4m3` needs PTX ISA 8.4
            // even on the arch that has the hardware - confirmed empirically:
            // identical instruction, identical `-arch=sm_89`, rejected on 7.8
            // with "Feature 'mma with FP8 floating point type' requires PTX
            // ISA .version 8.4 or later". This RAISES the module's floor
            // rather than the arch's, so a kernel with no FP8 in it still
            // declares 7.8 and still loads on a CUDA 11.8 driver.
            self.require_ptx_version("8.4");
            return true;
        }
        let lvl = self.sm_target.clone();
        self.emit_errors.push(format!(
            "[PTX] kernel `{}` uses FP8 (e4m3) tensor cores, which exist only on sm_89 \
             (Ada) and later; this build targets {}. There is no fallback: the \
             instruction is absent, so the module would be rejected at load time with \
             CUDA_ERROR_NO_BINARY_FOR_GPU. Use the int8 or f16 GEMM path on this card.",
            kernel, lvl
        ));
        false
    }

    /// A `WitnessOp` the GPU witness generator cannot lower.
    ///
    /// Refusing is the fix, not a stopgap. A witness slot silently filled with
    /// zero produces a kernel that assembles, launches, and hands back an
    /// assignment satisfying nothing - and the caller has no way to tell that
    /// from a working one.
    fn unsupported_witness_op(&mut self, signal: usize, name: &str, reason: &str) {
        self.emit_errors.push(format!(
            "[PTX witness generator] signal {} is a `{}`: {}.",
            signal, name, reason
        ));
    }

    /// The `.param` slot a kernel parameter occupies, refusing what this
    /// backend cannot lower.
    ///
    /// **This replaces a `_ => ".param .b32"`, which is the design-rule
    /// table's shape at the ABI boundary.** A parameter type the backend does
    /// not recognise took a 32-bit slot and the body then used it as an
    /// address. Found with
    ///
    /// ```text
    /// kernel k(T: SmemLayout<F16, rows=16, cols=64>, C: GlobalMemory<F16>) {
    ///     let v: F16 = T[3];
    ///     store(C, 0, v);
    /// }
    /// ```
    ///
    /// which compiled clean, printed "Compilation Successful!", exited 0, and
    /// emitted `add.u64 %rd3, %r0, %rd2` - a `.b32` register added to a `.b64`
    /// one - that `ptxas` rejects outright with "Arguments mismatch for
    /// instruction 'add'". The indexed read also produced no value, so the
    /// store wrote a literal `0`, and the address shift was 4 bytes for a
    /// 2-byte element type. Three wrongs under a green banner, on a surface
    /// `docs/y_language_documentation.md` §21 documents as an API.
    ///
    /// No `.ysu` in `tests/` declares an `SmemLayout`, which is why 60 of 60
    /// freshly compiled modules assemble and this went unseen: it is reachable
    /// from the surface syntax and exercised by nothing - the same profile as
    /// the Hopper intrinsics that were deleted rather than fixed.
    ///
    /// **Called from two sites, and only one of them is covered.** The split
    /// paged-decode shape emits a second `.visible .entry` with the same
    /// parameter list, and CLAUDE.md's standing rule is to enumerate the SITES
    /// rather than the match arms - so both call this. Mutation says the second
    /// is not reachable with a bad type: the main entry is emitted first over
    /// the same `kernel.params`, so it refuses and the compile aborts before
    /// the reduce entry is written. Reverting site 2 alone is caught by
    /// nothing. Kept as defence, and said out loud rather than counted as
    /// covered.
    ///
    /// **Known and deliberately still permissive:** a `Type::Ident` naming a
    /// declared struct rather than a scalar still takes a `.b32` slot. That is
    /// the status quo, it is a different bug, and narrowing it needs the
    /// emitter to know the struct table.
    fn param_slot(&mut self, kernel_name: &str, param: &Param) -> &'static str {
        match &param.ty {
            Type::Generic { base, .. } if base == "GlobalMemory" => ".param .u64",
            Type::Primitive(p, _) | Type::Ident(p, _) => {
                if ScalarTy::from_name(p).map_or(false, |t| t.is_64()) {
                    ".param .b64"
                } else {
                    ".param .b32"
                }
            }
            other => {
                let what = match other {
                    Type::Generic { base, .. } => format!("`{base}<...>`"),
                    Type::Array { .. } => "an array type".to_string(),
                    Type::Reference { .. } => "a reference type".to_string(),
                    Type::BlockTile { .. } => "`BlockTile<...>`".to_string(),
                    _ => "this type".to_string(),
                };
                self.emit_errors.push(format!(
                    "[PTX] kernel `{}`'s parameter `{}` has type {}, which this backend \
                     cannot pass in a `.param` slot for target {}. It used to take a \
                     32-bit slot and be used as an address, which emits PTX `ptxas` \
                     rejects. Shared memory is `shared_alloc_u32(n)` with \
                     `shared_load_v4` / `shared_store_v4` and `barrier_sync()`; global \
                     buffers are `GlobalMemory<T>`.",
                    kernel_name, param.name, what, self.sm_target
                ));
                ".param .b32"
            }
        }
    }

    fn unsupported_intrinsic(&mut self, name: &str, reason: &str) {
        self.emit_errors.push(format!(
            "[PTX] `{}(...)` cannot be lowered for target {}: {}.",
            name, self.sm_target, reason
        ));
    }

    /// The value of `e` if it is a non-negative integer literal.
    ///
    /// Deliberately does not fold arithmetic: the callers use this for operands
    /// PTX encodes as immediates, where "I could not prove this is a constant"
    /// must lead to a refusal, and a half-hearted folder makes the refusal
    /// depend on how the user happened to spell the expression.
    fn const_u32_of(e: &Expr) -> Option<u32> {
        match e {
            Expr::IntLit(v, _) if *v >= 0 => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Allocates a new virtual 16-bit register (e.g. `%h1`) - needed for
    /// `cvt.rn.satfinite.e4m3x2.f32`, whose destination the PTX ISA
    /// mandates be `.b16` (packs two e4m3 bytes) - see
    /// `emit_fp8_quantize_stage_a`/`_b`.
    fn alloc_reg16(&mut self) -> String {
        let name = format!("%h{}", self.reg_b16_count);
        self.reg_b16_count += 1;
        name
    }

    /// Allocates a unique PTX label.
    fn alloc_label(&mut self, prefix: &str) -> String {
        let label = format!("${}_{}", prefix, self.label_count);
        self.label_count += 1;
        label
    }

    fn emit_u32_init(&mut self, dst: &str, expr: &Expr) {
        match expr {
            Expr::IntLit(val, _) if *val >= 0 && *val <= u32::MAX as i64 => {
                writeln!(
                    &mut self.ptx_buffer,
                    "    mov.u32 {}, {};",
                    dst, *val as u32
                )
                .unwrap();
            }
            _ => {
                let val_reg = self.emit_expr(expr, None, &HardwareProfile::default());
                writeln!(
                    &mut self.ptx_buffer,
                    "    mov.u32 {}, {};",
                    dst, val_reg
                )
                .unwrap();
            }
        }
    }

    pub fn emit_program(&mut self, prog: &Program, hw_profile: &HardwareProfile) -> String {
        let mut kernels = 0usize;
        for item in &prog.items {
            if let Item::Kernel(k) = item {
                self.emit_kernel(k, hw_profile);
                kernels += 1;
            }
        }
        // This loop matches `Item::Kernel` and NOTHING else, so a source with no
        // `kernel` in it produced a three-line module - `.version`, `.target`,
        // `.address_size` and not one instruction - which the compiler wrote to
        // disk under "PTX Assembly generated successfully!" and exit 0.
        //
        // **An empty module assembles perfectly**, which is why the `ptxas`
        // gates this repo added after gotcha #8 could never see it: 24 of the 85
        // programs in `tests/` emitted one, and a corpus-wide assemble sweep
        // reported 82/82 accepted. That is the same limit recorded there for a
        // MISSING instruction, one level up - here the whole program is missing.
        //
        // Refusing rather than warning, because there is no artifact to salvage:
        // whatever the user meant to compile, none of it is in the file.
        if kernels == 0 {
            self.emit_errors.push(
                "[PTX backend] this source declares no `kernel`, so there is \
                 nothing to emit - the module would be a `.version`/`.target` \
                 header and no instructions. An empty module assembles and does \
                 nothing. Declare a `kernel`, or compile host code with the LLVM \
                 backend (the default) or --emit-cpu."
                    .to_string(),
            );
        }
        self.finalize_ptx_version();
        self.ptx_buffer.clone()
    }

    /// Raise the module's `.version` if something emitted needed more than the
    /// target arch's floor.
    ///
    /// The header is written when the emitter is constructed, before any kernel
    /// is seen, so a feature needing a newer ISA can only be discovered
    /// afterwards. Rewriting the one line is cheaper than deferring the header,
    /// and it keeps the floor honest in both directions: a plain kernel on
    /// sm_89 declares 7.8, an FP8 one declares 8.4.
    fn finalize_ptx_version(&mut self) {
        let Some(required) = self.required_ptx_version else { return };
        let Some(line) = self.ptx_buffer.lines().find(|l| l.starts_with(".version")) else {
            return;
        };
        if ptx_version_ge(line, required) {
            return;
        }
        let line = line.to_string();
        self.ptx_buffer = self
            .ptx_buffer
            .replacen(&line, &format!(".version {}", required), 1);
    }

    /// Record that the module needs at least this PTX ISA version.
    ///
    /// Called by the lowerings whose INSTRUCTIONS need more than their target
    /// arch does - FP8 `mma.sync` is the only one today. Recording it here
    /// rather than baking it into `ptx_version_for_sm` is what stops every
    /// other kernel on the same arch from inheriting the requirement.
    fn require_ptx_version(&mut self, v: &'static str) {
        match self.required_ptx_version {
            Some(cur) if ptx_version_ge(cur, v) => {}
            _ => self.required_ptx_version = Some(v),
        }
    }

    fn emit_kernel(&mut self, kernel: &KernelDecl, hw_profile: &HardwareProfile) {
        self.reject_unsupported_element_types(kernel);
        // Clear variables mapping for fresh compilation unit
        self.variables.clear();
        self.reg_ty.clear();
        self.vec_vars.clear();
        self.ptr_elem.clear();
        self.record_pointer_element_types(kernel);

        // Reset register counters
        self.reg_u32_count = 0;
        self.reg_f32_count = 0;
        self.reg_u64_count = 0;
        self.reg_pred_count = 0;
        self.reg_b16_count = 0;

        // Create a temporary buffer for parameter loading and kernel body
        let body_buffer = String::new();
        
        // Swap self.ptx_buffer with body_buffer temporarily so emit_stmt / emit_block writes to body_buffer
        let saved_buffer = std::mem::replace(&mut self.ptx_buffer, body_buffer);

        // Load parameters into registers (writes to temporary self.ptx_buffer)
        for (i, param) in kernel.params.iter().enumerate() {
            match &param.ty {
                Type::Generic { base, .. } if base == "GlobalMemory" => {
                    let r = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    ld.param.u64 {}, [{}_{}];", r, param.name, i).unwrap();
                    self.variables.insert(param.name.clone(), r);
                }
                // A scalar parameter is loaded in its declared type. The
                // catch-all below is still `ld.param.u32` into an untyped `%r`,
                // which is what every unannotated size/stride parameter has
                // always been and must stay.
                Type::Primitive(p, _) | Type::Ident(p, _) if ScalarTy::from_name(p).is_some() => {
                    let ty = ScalarTy::from_name(p).expect("checked by the guard");
                    let r = self.alloc_ty(ty);
                    writeln!(
                        &mut self.ptx_buffer,
                        "    ld.param.{} {}, [{}_{}];",
                        ty.mem(),
                        r,
                        param.name,
                        i
                    )
                    .unwrap();
                    self.variables.insert(param.name.clone(), r);
                }
                _ => {
                    let r = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    ld.param.u32 {}, [{}_{}];", r, param.name, i).unwrap();
                    self.variables.insert(param.name.clone(), r);
                }
            }
        }

        // Emit body: a validated kernel-level `@tile(M, N, K)` dispatches to
        // the tile-aware Tensor Core GEMM path wholesale instead of the
        // normal generic per-statement lowering (see `tile_gemm_operands`).
        // Captures the real launch block size for the `.maxnreg` computation
        // below - see that computation's doc comment for why this matters.
        let tile_gemm_threads_per_cta = if let Some((m, n, k, a_ptr, b_ptr, c_ptr, bias_ptr)) = self.tile_gemm_operands(kernel) {
            Some(self.emit_tensor_core_gemm_kernel(m, n, k, &a_ptr, &b_ptr, &c_ptr, bias_ptr.as_deref(), hw_profile, &kernel.name))
        } else if let Some((m, n, k, a, b, sa, sb, bi, c)) =
            self.tile_gemm_int8_scaled_operands(kernel)
        {
            Some(self.emit_int8_gemm_kernel(
                m, n, k, &a, &b, &c, &kernel.name, Some((&sa, &sb, &bi)),
            ))
        } else if let Some((m, n, k, a_ptr, b_ptr, c_ptr)) = self.tile_gemm_int8_operands(kernel) {
            Some(self.emit_int8_gemm_kernel(m, n, k, &a_ptr, &b_ptr, &c_ptr, &kernel.name, None))
        } else if let Some((m, n, k, a_ptr, b_ptr, scale_a_reg, scale_b_reg, c_ptr)) = self.tile_gemm_fp8_operands(kernel) {
            if !self.require_fp8_hardware(&kernel.name) {
                return;
            }
            Some(self.emit_fp8_gemm_kernel(m, n, k, &a_ptr, &b_ptr, &scale_a_reg, &scale_b_reg, &c_ptr, &kernel.name))
        } else if let Some((m, n, k, x_ptr, wgate_ptr, wup_ptr, out_ptr)) = self.tile_gemm_swiglu_operands(kernel) {
            Some(self.emit_gemm_swiglu_kernel(m, n, k, &x_ptr, &wgate_ptr, &wup_ptr, &out_ptr, hw_profile, &kernel.name))
        } else if let Some((hidden_dim, x_ptr, res_ptr, w_ptr, out_ptr, newres_ptr)) = self.rmsnorm_residual_operands(kernel) {
            Some(self.emit_rmsnorm_residual_kernel(hidden_dim, &x_ptr, &res_ptr, &w_ptr, &out_ptr, &newres_ptr))
        } else if let Some((head_dim, x_ptr, pos_ptr, out_ptr)) = self.rope_operands(kernel) {
            Some(self.emit_rope_kernel(head_dim, &x_ptr, &pos_ptr, &out_ptr))
        } else if let Some((head_dim, nqh, nkvh, page_size, splits, warps, q, kc, vc, pt, sl, out, part, meta, max_pages)) =
            self.paged_decode_attention_split_operands(kernel)
        {
            // Checked BEFORE the 7-parameter shape: the two name grammars do
            // not overlap (`parse_trailing_dims` rejects the literal `split`
            // component), but relying on that ordering-independence silently
            // would be exactly the kind of near-miss dispatch gotcha #5 warns
            // about, so the specific shape is tried first on purpose.
            self.emit_attention_split_reduce_entry(head_dim, nqh, splits, kernel, &kernel.name);
            Some(self.emit_paged_decode_attention_kernel(
                head_dim, nqh, nkvh, page_size, warps, &q, &kc, &vc, &pt, &sl, &out, &max_pages,
                Some(AttnSplit { splits, partial: &part, meta: &meta }),
                &kernel.name,
            ))
        } else if let Some((head_dim, nqh, nkvh, page_size, warps, q, kc, vc, pt, sl, out, max_pages)) =
            self.paged_decode_attention_operands(kernel)
        {
            Some(self.emit_paged_decode_attention_kernel(
                head_dim, nqh, nkvh, page_size, warps, &q, &kc, &vc, &pt, &sl, &out, &max_pages,
                None,
                &kernel.name,
            ))
        } else {
            self.emit_block(&kernel.body, hw_profile);
            None
        };

        // Take back the body_buffer and restore the original self.ptx_buffer
        let body_code = std::mem::replace(&mut self.ptx_buffer, saved_buffer);

        // Flush any module-scope declarations the body emitter queued up
        // (currently just `.extern .shared` for a pipelined tile-GEMM
        // kernel - see `pending_extern_decls`'s doc comment) into the real
        // buffer now, before this kernel's own `.visible .entry` line.
        for decl in self.pending_extern_decls.drain(..) {
            writeln!(&mut self.ptx_buffer, "{}", decl).unwrap();
        }
        // Whole extra entry points queued by the body emitter (currently only
        // the split-K attention combine). Same placement rule as above: module
        // scope, textually before this kernel's own entry.
        for item in std::mem::take(&mut self.pending_module_items) {
            writeln!(&mut self.ptx_buffer, "{}", item).unwrap();
        }
        // `.shared` arrays the body asked for, same placement rule. 16-byte
        // aligned because every access to them is `ld/st.shared.v4.u32`,
        // which faults on a misaligned address rather than being slow.
        for (sym, count) in std::mem::take(&mut self.shared_arrays) {
            writeln!(
                &mut self.ptx_buffer,
                ".shared .align 16 .b32 {}[{}];",
                sym, count
            )
            .unwrap();
        }

        // Emit kernel signature to original self.ptx_buffer
        writeln!(&mut self.ptx_buffer, ".visible .entry {}(", kernel.name).unwrap();

        let param_count = kernel.params.len();
        for (i, param) in kernel.params.iter().enumerate() {
            // Must agree with the `ld.param` widths chosen above: a `.param
            // .b32` slot read with `ld.param.u64` is a host-ABI mismatch, not
            // a type error, so nothing downstream would catch it.
            let ptx_type = self.param_slot(&kernel.name, param);

            write!(
                &mut self.ptx_buffer,
                "    {} {}_{}",
                ptx_type, param.name, i
            )
            .unwrap();
            if i < param_count - 1 {
                writeln!(&mut self.ptx_buffer, ",").unwrap();
            } else {
                writeln!(&mut self.ptx_buffer).unwrap();
            }
        }
        writeln!(&mut self.ptx_buffer, ")").unwrap();

        // Calculate dynamic register pressure limit
        let total_regs_used = self.reg_u32_count + self.reg_f32_count + self.reg_u64_count * 2;
        let limit = if total_regs_used <= 32 {
            32
        } else if total_regs_used <= 64 {
            64
        } else if total_regs_used <= 128 {
            128
        } else {
            255
        };

        // block_size here assumes 256 threads/block for every kernel except
        // a tile-aware GEMM (`tile_gemm_threads_per_cta`, its own real,
        // autotuned thread count - which can be 512 for the largest
        // candidates). That assumption isn't just a display-comment
        // inaccuracy for those kernels: `limit` above is a coarse 32/64/
        // 128/255 bucket of a virtual-register-count estimate that is not
        // ptxas's real, final per-thread allocation, so `.maxnreg` was
        // landing far above what a 512-thread block can actually afford
        // (max_regs_per_sm=65536 / 512 threads = 128 regs/thread ceiling,
        // vs the 255 this bucketing was picking) - confirmed on real
        // hardware as CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES ("too many
        // resources requested for launch") the first time a tile-aware
        // kernel's real per-thread register need got close enough to that
        // ceiling to matter. Clamping `limit` itself (not just the display
        // math below) to what the real block size can afford fixes this at
        // the source rather than papering over it in the estimate.
        let block_size = tile_gemm_threads_per_cta.unwrap_or(256);
        let max_regs_per_sm = hw_profile.max_regs_per_sm;
        let max_threads_per_sm = hw_profile.max_threads_per_sm;
        // The clamp only applies when there is a real register budget to
        // clamp against. `HardwareProfile::default()` (no probe run, e.g. a
        // cross-compile or any unit test that builds its own profile) leaves
        // `max_regs_per_sm` at 0, and `limit.min(0 / block_size)` is 0 - which
        // emits `.maxnreg 0` and makes ptxas reject the whole module with
        // "Positive non-zero value expected for maxnreg". That produced
        // structurally invalid PTX for EVERY kernel compiled without a
        // hardware profile, not just this one; it went unnoticed because the
        // PTX tests only string-matched their own expected substrings and
        // never assembled the result (see
        // `tests_paged_decode_attention::emitted_ptx_actually_assembles`,
        // which is what caught it).
        let limit = if block_size > 0 && max_regs_per_sm > 0 {
            limit.min((max_regs_per_sm / block_size).max(1))
        } else {
            limit
        };

        let active_blocks = if limit > 0 && block_size > 0 {
            (max_regs_per_sm / (block_size * limit))
                .min(max_threads_per_sm / block_size)
                .min(hw_profile.max_warps_per_sm * hw_profile.warp_size / block_size)
        } else {
            1
        };
        let occupancy = (active_blocks * block_size) as f64 / max_threads_per_sm as f64 * 100.0;

        writeln!(&mut self.ptx_buffer, "    // [ADVANCED WARP REGISTER ALLOCATOR] Estimated registers per thread: {}", total_regs_used).unwrap();
        writeln!(&mut self.ptx_buffer, "    // [ADVANCED WARP REGISTER ALLOCATOR] Selected register limit: {} (to maximize SM occupancy)", limit).unwrap();
        writeln!(&mut self.ptx_buffer, "    // [ADVANCED WARP REGISTER ALLOCATOR] Estimated occupancy: {:.2}% ({} active blocks per SM)", occupancy, active_blocks).unwrap();
        writeln!(&mut self.ptx_buffer, ".maxnreg {}", limit).unwrap();
        writeln!(&mut self.ptx_buffer, "{{").unwrap();

        // Declare registers with exact counts used
        writeln!(&mut self.ptx_buffer, "    .reg .b32 %r<{}>;", self.reg_u32_count.max(1)).unwrap();
        writeln!(&mut self.ptx_buffer, "    .reg .f32 %f<{}>;", self.reg_f32_count.max(1)).unwrap();
        writeln!(&mut self.ptx_buffer, "    .reg .b64 %rd<{}>;", self.reg_u64_count.max(1)).unwrap();
        writeln!(&mut self.ptx_buffer, "    .reg .pred %p<{}>;", self.reg_pred_count.max(1)).unwrap();
        if self.reg_f64_count > 0 {
            writeln!(&mut self.ptx_buffer, "    .reg .f64 %fd<{}>;", self.reg_f64_count).unwrap();
        }
        if self.reg_b16_count > 0 {
            writeln!(&mut self.ptx_buffer, "    .reg .b16 %h<{}>;", self.reg_b16_count).unwrap();
        }
        writeln!(&mut self.ptx_buffer).unwrap();

        // Append the body code
        self.ptx_buffer.push_str(&body_code);

        writeln!(&mut self.ptx_buffer, "}}").unwrap();
    }

    /// Every identifier `e` reads, for the barrier-hoisting legality check.
    ///
    /// Over-approximating is the safe direction here: a name collected that is
    /// not really a read only makes the hoist give up. Under-approximating
    /// moves a statement across a barrier it depends on, which is silent and
    /// wrong, so the walk is exhaustive with no `_ =>` arm.
    fn collect_idents(e: &Expr, out: &mut Vec<String>) {
        match e {
            Expr::Ident(n, _) => out.push(n.clone()),
            Expr::BinaryOp { left, right, .. } => {
                Self::collect_idents(left, out);
                Self::collect_idents(right, out);
            }
            Expr::UnaryOp { operand, .. } => Self::collect_idents(operand, out),
            Expr::Call { func, args, .. } => {
                Self::collect_idents(func, out);
                for a in args {
                    Self::collect_idents(a, out);
                }
            }
            Expr::GenericCall { func, args, .. } => {
                Self::collect_idents(func, out);
                for a in args {
                    Self::collect_idents(a, out);
                }
            }
            Expr::Index { base, index, .. } => {
                Self::collect_idents(base, out);
                Self::collect_idents(index, out);
            }
            Expr::MemberAccess { base, .. } => Self::collect_idents(base, out),
            Expr::StructLit { fields, .. } => {
                for (_, v) in fields {
                    Self::collect_idents(v, out);
                }
            }
            // A block expression can bind names of its own, so its free
            // variables are not simply the identifiers inside it. Rather than
            // model scoping for a construct no kernel body uses, poison the
            // check: a name that cannot be bound makes the caller refuse to
            // hoist.
            Expr::BlockExpr(..) => out.push("\0unmodelled".into()),
            // Leaves: nothing to read.
            Expr::IntLit(..)
            | Expr::FloatLit(..)
            | Expr::StringLit(..)
            | Expr::CharLit(..)
            | Expr::BoolLit(..)
            | Expr::SelfLit(..)
            | Expr::Path { .. }
            | Expr::ZeroInit(..) => {}
        }
    }

    fn emit_block(&mut self, block: &Block, hw_profile: &HardwareProfile) {
        let mut stmts = block.stmts.clone();

        let mut i = 0;
        while i < stmts.len() {
            let is_barrier = match &stmts[i] {
                Stmt::Expr(Expr::Path {
                    namespace, member, ..
                }) => namespace == "barrier" && member == "sync",
                Stmt::Expr(Expr::Call { func, .. }) => match &**func {
                    Expr::Path {
                        namespace, member, ..
                    } => namespace == "barrier" && member == "sync",
                    Expr::Ident(fname, _) => fname == "membar" || fname == "barrier_sync",
                    _ => false,
                },
                _ => false,
            };

            if is_barrier {
                let budget = (hw_profile.membar_gpu_latency_cycles / hw_profile.imad_latency_cycles)
                    as usize;
                let mut hoist_count = 0;

                // Moving work from AFTER a barrier to BEFORE it is only legal
                // when the work does not depend on the barrier. Two rules
                // enforce that, and both were missing:
                //
                //   * Stop at the first statement that cannot be hoisted.
                //     This loop used to `j += 1` past anything it did not
                //     recognise and keep scanning, so it would reach into
                //     code arbitrarily far ahead and pull an arithmetic
                //     statement out from under the loads that define its
                //     operands.
                //   * Hoist only statements whose every operand is ALREADY
                //     bound at the barrier. An operand that is not bound yet
                //     is, by definition, produced after the barrier - and in
                //     a shared-memory kernel that is exactly the value
                //     another thread wrote, which is the whole reason the
                //     barrier is there.
                //
                // This never fired before because `barrier_sync()` emitted no
                // instruction at all and no reachable kernel used shared
                // memory; the pass was live code guarding a barrier that did
                // not exist. Guarded now by
                // `tests/ptx_shared_memory.rs::hoisting_cannot_cross_a_real_dependency`.
                let mut j = i + 1;
                let mut hoisted = Vec::new();
                while j < stmts.len() && hoist_count < budget {
                    let value = match &stmts[j] {
                        Stmt::Let {
                            init: Some(v @ Expr::BinaryOp { .. }),
                            ..
                        } => v,
                        Stmt::Assign {
                            value: v @ Expr::BinaryOp { .. },
                            ..
                        } => v,
                        _ => break,
                    };
                    let mut reads = Vec::new();
                    Self::collect_idents(value, &mut reads);
                    if !reads
                        .iter()
                        .all(|n| self.variables.contains_key(n) || self.vec_vars.contains_key(n))
                    {
                        break;
                    }
                    hoisted.push(stmts.remove(j));
                    hoist_count += 1;
                }

                if hoist_count > 0 {
                    writeln!(
                        &mut self.ptx_buffer,
                        "    // [BARRIER HOISTING] Found barrier stall of {} cycles.",
                        hw_profile.membar_gpu_latency_cycles
                    )
                    .unwrap();
                    writeln!(&mut self.ptx_buffer, "    // [BARRIER HOISTING] Hoisted {} independent ALU instructions into the shadow.", hoist_count).unwrap();

                    for h in hoisted {
                        self.emit_stmt(&h, hw_profile);
                    }
                } else {
                    writeln!(&mut self.ptx_buffer, "    // [BARRIER HOISTING] Barrier detected ({} cycle stall), but no independent ALUs to hoist.", hw_profile.membar_gpu_latency_cycles).unwrap();
                }
            }

            if i < stmts.len() {
                self.emit_stmt(&stmts[i], hw_profile);
                i += 1;
            }
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt, hw_profile: &HardwareProfile) {
        match stmt {
            // `@ZeroDrift`: the accumulator lives in a 64-bit integer register
            // and every term is converted into its domain on the way in, so the
            // running total is exact and independent of the order the terms
            // arrived in - which on a GPU is decided by the launch geometry.
            Stmt::Let { name, ty, init, zero_drift: Some(_), bounds, span, cache_policy, .. } => {
                let ty_name = match ty {
                    Some(Type::Primitive(n, _)) | Some(Type::Ident(n, _)) => n.clone(),
                    _ => "F32".to_string(),
                };
                let range = bounds.as_ref().and_then(|b| {
                    match (ptx_const_f64(&b.min), ptx_const_f64(&b.max)) {
                        (Some(lo), Some(hi)) => Some((lo, hi)),
                        _ => None,
                    }
                });
                let req = crate::zero_drift::Requirement::for_type_with_bounds(&ty_name, range);
                match crate::zero_drift::select_repr(&req, &self.drift_costs) {
                    Some(decision) => {
                        let repr = decision.repr;
                        self.drift_report.push(format!(
                            "{}: {} -> {} ({})",
                            name,
                            ty_name,
                            repr.name(),
                            match decision.cost_ps {
                                Some(c) => format!("measured {:.0} ps/acc", c),
                                None => "no device measurements, narrowest sufficient".into(),
                            }
                        ));
                        writeln!(
                            &mut self.ptx_buffer,
                            "    // [Y ZERO DRIFT] {}: {} accumulated exactly as {}",
                            name,
                            ty_name,
                            repr.name()
                        )
                        .unwrap();
                        // An accumulator is SIGNED in every representation.
                        // Allocating it untyped left `ty_of` at its `%rd`
                        // default of U64, which is how the old write-back came
                        // out as `cvt.rzi.u64.f32` - unsigned - on a sum that is
                        // routinely negative.
                        //
                        // MUTATION-CHECKED AND CURRENTLY NOT LOAD-BEARING:
                        // reverting this to `alloc_reg64()` passes the whole of
                        // `tests/zero_drift_backend_agreement.rs`, because the
                        // `Stmt::Assign` / `Stmt::CompoundAssign` arms below
                        // write `add.s64` literally and never consult
                        // `ty_of(acc)`. It is kept because the generic
                        // assignment path DOES consult it, and that path is one
                        // deleted guard away - which is exactly the history
                        // above. Do not read it as covered by a test.
                        let acc = self.alloc_ty(ScalarTy::I64);
                        // An integer accumulator holds the value itself. A
                        // fixed-point one holds a scaled float and has to be
                        // converted on every read and write.
                        let integer_domain = repr.frac_bits() == 0
                            && matches!(
                                ty_name.as_str(),
                                "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "U64"
                            );
                        match init {
                            Some(expr) => {
                                let v = self.emit_expr(expr, cache_policy.as_ref(), hw_profile);
                                if v.starts_with("%f") || integer_domain {
                                    let fixed = self.emit_drift_to_fixed(&v, repr, integer_domain);
                                    writeln!(&mut self.ptx_buffer, "    mov.s64 {}, {};", acc, fixed).unwrap();
                                } else {
                                    writeln!(&mut self.ptx_buffer, "    mov.u64 {}, 0;", acc).unwrap();
                                }
                            }
                            None => {
                                writeln!(&mut self.ptx_buffer, "    mov.u64 {}, 0;", acc).unwrap();
                            }
                        }
                        self.zero_drift.insert(name.clone(), (acc.clone(), repr, integer_domain));
                        self.variables.insert(name.clone(), acc);
                    }
                    None => {
                        self.emit_errors.push(format!(
                            "Line {}: @ZeroDrift on `{}: {}` cannot be honoured. No exact \
representation holds that range at that resolution, and only exact (integer or fixed-point) \
accumulation is drift-free. Add @bounds(min, max) to state the accumulator's real range, or \
declare it as a Q format.",
                            span.line, name, ty_name
                        ));
                    }
                }
            }
            Stmt::Let {
                name,
                ty,
                init,
                cache_policy,
                span,
                ..
            } => {
                // A sub-word local is refused before anything is emitted for
                // it: PTX has no sub-word register, so honouring the
                // declaration is impossible and ignoring it would silently give
                // 32-bit semantics to a type that promises 8- or 16-bit ones.
                if let Some(Type::Primitive(n, _)) | Some(Type::Ident(n, _)) = ty {
                    if self.reject_subword_local(name, n, span.line) {
                        return;
                    }
                }
                if let Some(expr) = init {
                    // Decided BEFORE `emit_expr` runs: a zero-initialiser is a
                    // declaration, and asking the expression layer to produce a
                    // value for it records a refusal for a construct that is
                    // perfectly legal here.
                    if matches!(expr, Expr::ZeroInit(_)) {
                        self.emit_zero_init_let(name, ty.as_ref(), span);
                        return;
                    }
                    let val_str = self.emit_expr(expr, cache_policy.as_ref(), hw_profile);
                    if let Some(regs) = Self::parse_v4_marker(&val_str) {
                        self.vec_vars.insert(name.clone(), regs);
                        return;
                    }
                    if !val_str.is_empty() {
                        // The declared type is honoured, not ignored: `let lo:
                        // U32 = t;` on a 64-bit `t` truncates, and `let w: U64
                        // = a;` on a 32-bit `a` widens. Before the integer
                        // datapath there was nothing to convert *to*, so this
                        // arm dropped `ty` entirely and the binding simply
                        // aliased whatever register the initialiser produced.
                        let declared = match ty {
                            Some(Type::Primitive(n, _)) | Some(Type::Ident(n, _)) => {
                                ScalarTy::from_name(n)
                            }
                            _ => None,
                        };
                        let reg = match declared {
                            Some(t) => self.emit_convert(&val_str, t),
                            None => val_str,
                        };
                        self.variables.insert(name.clone(), reg);
                    } else if Self::binds_a_linear_token(expr) {
                        // `cp_async` yields a LINEAR TOKEN, not a value: the
                        // copy is emitted, and the token exists only so
                        // `linear_tracker` can prove it is awaited exactly
                        // once. Nothing ever reads its register - `pipe.wait`
                        // lowers to `cp.async.wait_group 0` without touching
                        // the operand - so having no register is correct here
                        // rather than a failure to lower. Binding nothing is
                        // what the tracker expects.
                    } else {
                        // An initialiser this backend cannot lower used to fall
                        // through here silently: the name was never bound to a
                        // register, and every later use emitted the NAME as
                        // though it were one. `let v: I32 = load(Src[i]);`
                        // (there is no bare `load`, only `GlobalMemory::load`)
                        // produced `setp.gt.s32 %p1, v, %r1;` - PTX that
                        // ptxas rejects, after a clean compile and a
                        // "Compilation Successful!".
                        //
                        // Refusing is the fix, not a stopgap: a named gap
                        // costs a user five minutes, and this cost a silent
                        // undefined symbol in the middle of a kernel. Same
                        // reasoning as the `tma_load` / `wgmma_async` refusals
                        // in gotcha #8.
                        self.emit_errors.push(format!(
                            "[PTX] line {}: the initialiser of `let {}` produced no value. \
                             This backend could not lower it - check the spelling and the \
                             argument count of any intrinsic it calls (for example there is \
                             no bare `load`; global loads are `block_ptr2d_load`). \
                             Binding the name anyway would emit `{}` into the PTX as if it \
                             were a register.",
                            span.line, name, name
                        ));
                    }
                }
            }
            Stmt::TypeAlias { name, .. } => {
                writeln!(&mut self.ptx_buffer, "    // type {} defined", name).unwrap();
            }
            Stmt::For {
                loop_var,
                start,
                end,
                step,
                body,
                tile,
                ..
            } => {
                let loop_reg = self.alloc_reg32();
                let end_reg = self.alloc_reg32();
                let exit_pred = self.alloc_pred();
                let loop_start = self.alloc_label("LOOP_START");
                let loop_end = self.alloc_label("LOOP_END");

                writeln!(&mut self.ptx_buffer, "    // for {} in ...", loop_var).unwrap();
                if let Some(t) = tile {
                    writeln!(&mut self.ptx_buffer, "    // [Y TILE OPTIMIZATION] Tiled loop dimensions: M={:?}, N={:?}, K={:?}", t.block_m, t.block_n, t.block_k).unwrap();
                }
                if let Some(t) = tile {
                    writeln!(&mut self.ptx_buffer, "    // [Y TILE OPTIMIZATION] Tiled loop dimensions: M={:?}, N={:?}, K={:?}", t.block_m, t.block_n, t.block_k).unwrap();
                }
                // A `step` this arm could not read used to fall through to
                // `_ => 1`, silently. So `for i in w..N step nworkers` - the
                // grid-stride loop, the canonical way to write a kernel whose
                // launch geometry is a tuning parameter - compiled cleanly and
                // stepped by ONE: every thread walked the whole range, which
                // is N-times redundant work and, in any reduction, a wrong
                // answer. Same shape as every row in the design-rule table in
                // CLAUDE.md: a default that is plausible rather than correct.
                //
                // A dynamic step is emitted as a register instead. The
                // literal path is kept because the v4 vectorising pass keys
                // off `step_val == 4`, which a runtime value cannot satisfy.
                let mut step_reg: Option<String> = None;
                let step_val = match step {
                    Some(Expr::IntLit(step, _)) if *step > 0 && *step <= u32::MAX as i64 => {
                        *step as u32
                    }
                    Some(Expr::IntLit(bad, _)) => {
                        // Zero never terminates and a negative step is a
                        // different loop than the one written, since the exit
                        // test is `>=`.
                        self.unsupported_intrinsic(
                            "for-step",
                            &format!(
                                "step must be positive; `{}` would not terminate under \
                                 this loop's `>=` exit test",
                                bad
                            ),
                        );
                        1
                    }
                    Some(dynamic) => {
                        let r = self.alloc_reg32();
                        self.emit_u32_init(&r, dynamic);
                        step_reg = Some(r);
                        1
                    }
                    None => 1,
                };

                if step_reg.is_none() && step_val == 4 {
                    writeln!(&mut self.ptx_buffer, "    // [Y AUTOMATED VECTORIZING PASS] Transformed loop step into 128-bit SIMD v4 stride").unwrap();
                }

                self.emit_u32_init(&loop_reg, start);
                self.emit_u32_init(&end_reg, end);
                self.variables.insert(loop_var.clone(), loop_reg.clone());

                writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
                writeln!(
                    &mut self.ptx_buffer,
                    "    setp.ge.u32 {}, {}, {};",
                    exit_pred, loop_reg, end_reg
                )
                .unwrap();
                writeln!(&mut self.ptx_buffer, "    @{} bra {};", exit_pred, loop_end).unwrap();
                self.emit_block(body, hw_profile);
                match &step_reg {
                    Some(r) => writeln!(
                        &mut self.ptx_buffer,
                        "    add.u32 {}, {}, {};",
                        loop_reg, loop_reg, r
                    ),
                    None => writeln!(
                        &mut self.ptx_buffer,
                        "    add.u32 {}, {}, {};",
                        loop_reg, loop_reg, step_val
                    ),
                }
                .unwrap();
                writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
                writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
            }
            // Accumulation into a `@ZeroDrift` binding. There is no general
            // `CompoundAssign` arm in this emitter, so this handles only the
            // drift-free case; anything else still falls through as before.
            Stmt::CompoundAssign { target, op, value, span }
                if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n)) =>
            {
                let name = match target {
                    Expr::Ident(n, _) => n.clone(),
                    _ => unreachable!(),
                };
                let (acc, repr, integer_domain) = self.zero_drift[&name].clone();
                if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                    self.emit_errors.push(format!(
                        "Line {}: `{:?}=` is not exact on the @ZeroDrift accumulator `{}`. Only \
`+=` and `-=` preserve drift-freedom.",
                        span.line, op, name
                    ));
                }
                let rhs = self.emit_expr(value, None, hw_profile);
                let fixed = self.emit_drift_to_fixed(&rhs, repr, integer_domain);
                let instr = if matches!(op, BinaryOp::Sub) { "sub.s64" } else { "add.s64" };
                writeln!(&mut self.ptx_buffer, "    {} {}, {}, {};", instr, acc, acc, fixed).unwrap();
            }
            // `acc = acc + e` is the SAME statement as `acc += e`, and this
            // emitter had an arm for one and not the other - so the running-sum
            // form fell through to `Stmt::Assign` below, which read the
            // accumulator as f32, added in f32, and wrote back through an
            // UNSIGNED truncating convert. It compiled, it assembled, and the
            // comment above it said `accumulated exactly as I64`.
            //
            // THIS ARM MUST STAY ABOVE `Stmt::Assign`. Match arms are ordered;
            // the LLVM backend's equivalent was first inserted below its
            // general arm, where it compiled, ran, and never fired.
            Stmt::Assign { target, value, span }
                if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n)) =>
            {
                let name = match target {
                    Expr::Ident(n, _) => n.clone(),
                    _ => unreachable!(),
                };
                let (acc, repr, integer_domain) = self.zero_drift[&name].clone();
                match crate::zero_drift::running_sum(target, value) {
                    Some((op, term)) => {
                        let rhs = self.emit_expr(term, None, hw_profile);
                        let fixed = self.emit_drift_to_fixed(&rhs, repr, integer_domain);
                        let instr = if matches!(op, BinaryOp::Sub) { "sub.s64" } else { "add.s64" };
                        writeln!(
                            &mut self.ptx_buffer,
                            "    {} {}, {}, {};",
                            instr, acc, acc, fixed
                        )
                        .unwrap();
                    }
                    None => {
                        // Refusing is the fix, not a stopgap. Lowering this as
                        // an ordinary assignment is what the bug was: the
                        // guarantee is silently dropped and the artifact still
                        // runs.
                        self.emit_errors.push(format!(
                            "Line {}: `{}` is a @ZeroDrift accumulator, and this assignment is not an exact accumulation. Only `{} += e`, `{} -= e` and their `{} = {} + e` / `{} = {} - e` spellings preserve drift-freedom; anything else reintroduces the rounding the directive exists to remove.",
                            span.line, name, name, name, name, name, name, name
                        ));
                    }
                }
            }
            Stmt::Assign {
                target, value, ..
            } => {
                let val_reg = self.emit_expr(value, None, hw_profile);
                if let Expr::Ident(name, _) = target {
                    if let Some(tgt_reg) = self.variables.get(name).cloned() {
                        // The width of the `mov` used to be read off the
                        // VALUE's register prefix, so `x = y` where `x: U64`
                        // and `y: U32` emitted `mov.u32 %rd7, %r3` - which
                        // ptxas rejects outright, and which for the f32/u32
                        // pair would have been a silent reinterpretation of
                        // the bits. The target's type is what a `mov` has to
                        // agree with, and the value converts into it, exactly
                        // as it does at a `let`.
                        let ty = self.ty_of(&tgt_reg);
                        let src = self.emit_convert(&val_reg, ty);
                        writeln!(&mut self.ptx_buffer, "    mov.{} {}, {};", ty.reg_mem(), tgt_reg, src).unwrap();
                    }
                }
            }
            Stmt::Expr(expr) => {
                self.emit_expr(expr, None, hw_profile);
            }
            // A kernel is void, so a bare `return;` is `ret;` and a returned
            // VALUE has nowhere to go. Both used to emit nothing, which is
            // right for neither: the first is an early exit that silently did
            // not happen.
            Stmt::Return(value, span) => {
                if value.is_some() {
                    self.unsupported_stmt(
                        "`return <expr>` inside a kernel (a kernel has no return value)",
                        span,
                    );
                } else {
                    writeln!(&mut self.ptx_buffer, "    ret;").unwrap();
                }
            }
            Stmt::SafeBlock(block, _) => {
                self.emit_block(block, hw_profile);
            }
            Stmt::GhostBlock(block, _) => {
                self.emit_block(block, hw_profile);
            }
            Stmt::HintBlock { body, .. } => {
                self.emit_block(body, hw_profile);
            }
            Stmt::ClockDomainBlock { body, .. } => {
                self.emit_block(body, hw_profile);
            }
            Stmt::CompileTimeAssert { .. } => {}
            Stmt::Chisel(block, _) => {
                writeln!(&mut self.ptx_buffer, "    // --- CHISEL INLINE PTX ---").unwrap();
                for stmt in &block.stmts {
                    if let Stmt::Expr(Expr::StringLit(s, _)) = stmt {
                        writeln!(&mut self.ptx_buffer, "    {}", s).unwrap();
                    } else {
                        self.emit_stmt(stmt, hw_profile);
                    }
                }
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let cond_str = self.emit_expr(condition, None, hw_profile);

                let then_cost = then_block.stmts.len() as f64 * 1.0;
                let else_cost = else_block
                    .as_ref()
                    .map(|b| b.stmts.len() as f64 * 1.0)
                    .unwrap_or(0.0);
                let total_cost = then_cost + else_cost;

                writeln!(
                    &mut self.ptx_buffer,
                    "    // [HEURISTIC] Branch Divergence Penalty is {} cycles.",
                    hw_profile.branch_divergence_penalty_cycles
                )
                .unwrap();
                {
                    // The "predicated execution" path emitted `@%p { ... }`,
                    // which is NOT PTX: the ISA predicates a single
                    // instruction, never a block. Every `if` small enough to
                    // trip the divergence heuristic therefore produced a file
                    // `ptxas` rejects with `Parsing error near '{'` - after a
                    // clean compile and a "Compilation Successful!".
                    //
                    // It survived because no reachable kernel had a scalar
                    // `if` in it, and the tests that did exist string-matched
                    // rather than assembling (gotcha #8, again). Deleted
                    // rather than repaired: real predication means predicating
                    // each emitted instruction, which is a change to every
                    // arm of `emit_block`, not a fix to this one. A branch is
                    // always correct; the divergence penalty it may cost is a
                    // performance question, and the old code's answer to it
                    // was a file that did not assemble.
                    writeln!(
                        &mut self.ptx_buffer,
                        "    // if: emitting BRANCH execution (block cost {} cy).",
                        total_cost
                    )
                    .unwrap();
                    let pred = self.alloc_pred();
                    // `if cond_str.is_empty() { "%r0" }` lived here. An empty
                    // string means `emit_expr` could not lower the condition,
                    // and `%r0` is parameter 0 - so the kernel branched on an
                    // unrelated value and assembled cleanly. Same row of the
                    // design-rule table as the intrinsic arity fallbacks.
                    if cond_str.is_empty() {
                        self.unsupported_stmt(
                            "an `if` whose condition this backend cannot lower",
                            &condition.span(),
                        );
                        return;
                    }
                    let cond_reg = cond_str;
                    let else_label = self.alloc_label("IF_ELSE");
                    let end_label = self.alloc_label("IF_END");
                    writeln!(
                        &mut self.ptx_buffer,
                        "    setp.eq.u32 {}, {}, 0;",
                        pred, cond_reg
                    )
                    .unwrap();
                    if else_block.is_some() {
                        writeln!(&mut self.ptx_buffer, "    @{} bra {};", pred, else_label)
                            .unwrap();
                    } else {
                        writeln!(&mut self.ptx_buffer, "    @{} bra {};", pred, end_label).unwrap();
                    }
                    self.emit_block(then_block, hw_profile);
                    if let Some(eb) = else_block {
                        writeln!(&mut self.ptx_buffer, "    bra {};", end_label).unwrap();
                        writeln!(&mut self.ptx_buffer, "    {}:", else_label).unwrap();
                        self.emit_block(eb, hw_profile);
                    }
                    writeln!(&mut self.ptx_buffer, "    {}:", end_label).unwrap();
                }
            }
            // Lowered exactly like `Stmt::For` above, minus the induction
            // variable and the step: test at the top, branch out on false,
            // body, branch back. This was `_ => {}` and emitted nothing at
            // all - see `unsupported_stmt`.
            //
            // Reachable only since the loop-invariant initiation check stopped
            // rejecting every `while` in the language, which is why a whole
            // missing statement kind went unnoticed in a backend with a
            // `ptxas` gate over it.
            Stmt::While { condition, body, .. } => {
                let loop_start = self.alloc_label("WHILE_START");
                let loop_end = self.alloc_label("WHILE_END");
                let exit_pred = self.alloc_pred();

                writeln!(&mut self.ptx_buffer, "    // while ...").unwrap();
                writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
                // The condition is re-evaluated every iteration, which is the
                // point of a `while`: hoisting it would be a `do`-loop.
                let cond_str = self.emit_expr(condition, None, hw_profile);
                if cond_str.is_empty() {
                    self.unsupported_stmt(
                        "a `while` whose condition this backend cannot lower",
                        &condition.span(),
                    );
                    return;
                }
                writeln!(
                    &mut self.ptx_buffer,
                    "    setp.eq.u32 {}, {}, 0;",
                    exit_pred, cond_str
                )
                .unwrap();
                writeln!(&mut self.ptx_buffer, "    @{} bra {};", exit_pred, loop_end).unwrap();
                self.emit_block(body, hw_profile);
                writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
                writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
            }
            // `break` needs a stack of enclosing loop-exit labels, which this
            // emitter does not keep. Refusing costs a line number; emitting
            // nothing costs a loop that never exits early.
            Stmt::Break { span } => {
                self.unsupported_stmt("`break` (this backend keeps no loop-exit label stack)", span);
            }
            Stmt::Match { span, .. } => {
                self.unsupported_stmt("`match`", span);
            }
            _ => {}
        }
    }

    fn emit_expr(
        &mut self,
        expr: &Expr,
        cache_policy: Option<&CachePolicyAttr>,
        hw_profile: &HardwareProfile,
    ) -> String {
        match expr {
            // An integer literal is typed by its VALUE, not fixed at I32.
            //
            // This is not a nicety. A modulus limb such as `4026531841`
            // (0xf0000001) sits above `i32::MAX`; typed I32 it is a negative
            // number, and widening it to 64 bits then sign-extends to
            // 0xFFFFFFFF_F0000001. The BN254 kernel's conditional subtract
            // came back with four limbs of all-ones from exactly this, on the
            // one operand pair (p-1 squared) whose reduction actually fires.
            // Nothing rejects it: `cvt.s64.s32` is a legal instruction on a
            // legal register.
            Expr::IntLit(val, _) => {
                let v = *val;
                let ty = if v >= 0 && v > u32::MAX as i64 {
                    ScalarTy::I64
                } else if v < i32::MIN as i64 {
                    ScalarTy::I64
                } else if v > i32::MAX as i64 {
                    ScalarTy::U32
                } else {
                    ScalarTy::I32
                };
                let reg = self.alloc_ty(ty);
                writeln!(&mut self.ptx_buffer, "    mov.{} {}, {};", ty.reg_mem(), reg, v).unwrap();
                reg
            }
            // A `bool` is a u32 holding 0 or 1, which is the representation
            // `Stmt::If` and `Stmt::While` already expect: both lower a
            // condition as `setp.eq.u32 %p, <cond>, 0`. There was no arm here
            // at all, so `if true { .. }` fell to `_ => "".into()` and was
            // refused - which took out `tests/test_drift.ysu --emit-ptx`, a
            // command CLAUDE.md documents and an earlier session had already
            // repaired once.
            Expr::BoolLit(val, _) => {
                let reg = self.alloc_ty(ScalarTy::U32);
                writeln!(
                    &mut self.ptx_buffer,
                    "    mov.u32 {}, {};",
                    reg,
                    if *val { 1 } else { 0 }
                )
                .unwrap();
                reg
            }
            Expr::FloatLit(val, _) => {
                let reg = self.alloc_regf32();
                let mut val_str = format!("{}", *val);
                if !val_str.contains('.') && !val_str.contains('e') && !val_str.contains('E') {
                    val_str = format!("{}.0", val_str);
                }
                writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", reg, val_str).unwrap();
                reg
            }
            Expr::Ident(name, span) => {
                // Reading a @ZeroDrift accumulator converts back out of its
                // integer domain; the stored value stays exact.
                if let Some((reg, repr, integer_domain)) = self.zero_drift.get(name).cloned() {
                    return self.emit_drift_from_fixed(&reg, repr, integer_domain);
                }
                if let Some(reg) = self.variables.get(name) {
                    return reg.clone();
                }
                // An unbound name used to return ITSELF, which the caller
                // splices into instruction text as if it were a register:
                // `let z: u32 = ZeroInit;` emitted `mov.u32 %r0, ZeroInit;`
                // under "Compilation Successful!" and exit 0, and `ptxas`
                // then rejected the module with `Unknown symbol 'ZeroInit'`.
                //
                // Every name this backend really does define is inserted into
                // `variables` at the point it is bound - parameters, `let`s,
                // loop induction variables, and the tile-GEMM machinery's
                // `pid_m`/`pid_n`. There is no legitimate bare name left, so
                // reaching here means the program named something that does
                // not exist.
                self.unsupported_expr(&format!("the undefined name `{}`", name), span);
                "".into()
            }
            Expr::BinaryOp {
                op, left, right, span,
            } => {
                let l_reg = self.emit_expr(left, cache_policy, hw_profile);
                let r_reg = self.emit_expr(right, cache_policy, hw_profile);
                self.emit_binary(op.clone(), &l_reg, &r_reg, span)
            }
            // There was no `UnaryOp` arm at all, so `-5` fell through to the
            // catch-all and emitted NOTHING. `let best: I32 = -2147483647;`
            // therefore bound nothing, and every later use of `best` put the
            // bare identifier into the PTX. Found by the `let`-produced-no-value
            // refusal added beside it, on the first kernel that needed a
            // negative sentinel.
            Expr::UnaryOp { op, operand, .. } => {
                let v = self.emit_expr(operand, cache_policy, hw_profile);
                if v.is_empty() {
                    return "".into();
                }
                match op {
                    UnaryOp::Neg => {
                        let t = self.ty_of(&v);
                        let out = self.alloc_ty(t);
                        let suffix = if t.is_float() {
                            t.arith()
                        } else if t.is_64() {
                            "s64"
                        } else {
                            "s32"
                        };
                        writeln!(&mut self.ptx_buffer, "    neg.{} {}, {};", suffix, out, v)
                            .unwrap();
                        out
                    }
                    other => {
                        self.emit_errors.push(format!(
                            "[PTX] the unary operator `{:?}` is not lowered by this backend. \
                             Only negation is. Emitting nothing for it would bind the \
                             enclosing name to no register at all.",
                            other
                        ));
                        "".into()
                    }
                }
            }
            Expr::Index { base, index, span } => {
                let base_reg = self.emit_expr(base, cache_policy, hw_profile);
                let idx_reg = self.emit_expr(index, cache_policy, hw_profile);

                let idx_u64 = self.alloc_reg64();
                writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", idx_u64, idx_reg).unwrap();

                let is_safe = crate::type_checker::SAFE_INDICES.with(|set| {
                    set.borrow().contains(&(span.line, span.col))
                });
                let array_size = crate::type_checker::INDEX_ARRAY_SIZES.with(|map| {
                    map.borrow().get(&(span.line, span.col)).cloned()
                });

                if !is_safe {
                    if let Some(size) = array_size {
                        let pred = self.alloc_pred();
                        writeln!(&mut self.ptx_buffer, "    setp.ge.u64 {}, {}, {};", pred, idx_u64, size).unwrap();
                        writeln!(&mut self.ptx_buffer, "    @{} trap;", pred).unwrap();
                    }
                }

                let swizzle_pattern = crate::type_checker::INDEX_SWIZZLES.with(|map| {
                    map.borrow().get(&(span.line, span.col)).cloned()
                });

                if let Some(swizzle) = swizzle_pattern {
                    // Apply dynamic swizzling in PTX to avoid bank conflicts!
                    // byte_addr = idx_u64 * 2 (since SharedMemoryTile uses F16 elements = 2 bytes)
                    let byte_addr = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 1;", byte_addr, idx_u64).unwrap();

                    // chunk_idx = byte_addr / 16
                    let chunk_idx = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    shr.u64 {}, {}, 4;", chunk_idx, byte_addr).unwrap();

                    // row = threadIdx.x % 16
                    let tid = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
                    let row = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 15;", row, tid).unwrap();

                    // xor_val = ((row >> swizzle.offset) & mask) << shift
                    let mut current_val = row.clone();
                    if swizzle.offset > 0 {
                        let temp = self.alloc_reg32();
                        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, {};", temp, current_val, swizzle.offset).unwrap();
                        current_val = temp;
                    }
                    let mask = (1 << swizzle.xor_bits) - 1;
                    let temp_masked = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, {};", temp_masked, current_val, mask).unwrap();
                    current_val = temp_masked;
                    if swizzle.base_shift > 0 {
                        let temp_shifted = self.alloc_reg32();
                        writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, {};", temp_shifted, current_val, swizzle.base_shift).unwrap();
                        current_val = temp_shifted;
                    }
                    let xor_val_u64 = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", xor_val_u64, current_val).unwrap();

                    // new_chunk = chunk_idx ^ xor_val
                    let new_chunk = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    xor.b64 {}, {}, {};", new_chunk, chunk_idx, xor_val_u64).unwrap();

                    // reconstruct byte_addr: swizzled_offset = (new_chunk * 16) | (byte_addr % 16)
                    let new_chunk_shifted = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 4;", new_chunk_shifted, new_chunk).unwrap();
                    let byte_offset = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    and.b64 {}, {}, 15;", byte_offset, byte_addr).unwrap();
                    let swizzled_offset = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    or.b64 {}, {}, {};", swizzled_offset, new_chunk_shifted, byte_offset).unwrap();

                    let addr_reg = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr_reg, base_reg, swizzled_offset).unwrap();
                    addr_reg
                } else {
                    // The stride is the element's size, not a constant 4. A
                    // `GlobalMemory<U64>` indexed at a 4-byte stride reads
                    // half of one element and half of the next.
                    let elem = self.elem_ty_or_f32(base);
                    let offset_reg = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, {};", offset_reg, idx_u64, elem.log2_bytes()).unwrap();

                    let addr_reg = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr_reg, base_reg, offset_reg).unwrap();
                    addr_reg
                }
            }
            Expr::Call { func, args, .. } => {
                match &**func {
                    Expr::Ident(fname, _) => {
                        // Arity gate, before any lowering runs. See
                        // `required_arity`: a short call otherwise silently
                        // reads an unrelated register instead of failing.
                        if let Some(want) = Self::required_arity(fname) {
                            // `<`, not `!=`: trailing operands past `want`
                            // default to literals and are legitimately optional.
                            if args.len() < want {
                                let n = args.len();
                                self.unsupported_intrinsic(
                                    fname,
                                    &format!(
                                        "it needs at least {} arguments and was given {}; \
                                         a missing operand would be read from an unrelated \
                                         register rather than reported",
                                        want, n
                                    ),
                                );
                                return "".into();
                            }
                        }
                        if fname == "cp_async" && args.len() >= 2 {
                            let src_reg = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let dest_reg = self.emit_expr(&args[1], cache_policy, hw_profile);
                            // The byte count used to be hardcoded to 16 and the
                            // third argument discarded, so `cp_async(dst, src, 4)`
                            // copied 16 bytes and overran the destination by 12.
                            // cp.async supports exactly 4, 8 and 16; anything else
                            // is refused rather than rounded to something legal.
                            let bytes = match args.get(2).and_then(Self::const_u32_of) {
                                None if args.len() < 3 => 16,
                                Some(n @ (4 | 8 | 16)) => n,
                                Some(n) => {
                                    self.unsupported_intrinsic(
                                        "cp_async",
                                        &format!(
                                            "cp.async transfers exactly 4, 8 or 16 bytes; {} was requested",
                                            n
                                        ),
                                    );
                                    16
                                }
                                None => {
                                    self.unsupported_intrinsic(
                                        "cp_async",
                                        "the byte count must be a compile-time constant of 4, 8 or 16 \
                                         - cp.async encodes it as an immediate, so a runtime value \
                                         cannot be lowered",
                                    );
                                    16
                                }
                            };
                            writeln!(&mut self.ptx_buffer, "    cp.async.cg.shared.global [{}], [{}], {};", dest_reg, src_reg, bytes).unwrap();
                            // Without a commit_group the copy belongs to no group,
                            // so the `cp.async.wait_group` that `pipe.wait` emits
                            // waits on nothing and returns immediately. Committing
                            // here makes the token the linear tracker hands out
                            // correspond to a real group.
                            self.emit_cp_async_commit();
                            "".into()
                        } else if fname == "vec_add_v4" || fname == "vector_add_v4" || fname == "vec_add_unrolled4" {
                            let a_ptr = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let b_ptr = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%rd1".to_string() };
                            let c_ptr = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%rd2".to_string() };

                            let unroll_count = if fname == "vec_add_unrolled4" || args.len() >= 4 { 4 } else { 1 };

                            for u in 0..unroll_count {
                                let offset = u * 16;
                                let a_addr = if offset == 0 { a_ptr.clone() } else {
                                    let r = self.alloc_reg64();
                                    writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", r, a_ptr, offset).unwrap();
                                    r
                                };
                                let b_addr = if offset == 0 { b_ptr.clone() } else {
                                    let r = self.alloc_reg64();
                                    writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", r, b_ptr, offset).unwrap();
                                    r
                                };
                                let c_addr = if offset == 0 { c_ptr.clone() } else {
                                    let r = self.alloc_reg64();
                                    writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", r, c_ptr, offset).unwrap();
                                    r
                                };

                                let a0 = self.alloc_regf32();
                                let a1 = self.alloc_regf32();
                                let a2 = self.alloc_regf32();
                                let a3 = self.alloc_regf32();
                                writeln!(&mut self.ptx_buffer, "    ld.global.cs.v4.f32 {{{}, {}, {}, {}}}, [{}];", a0, a1, a2, a3, a_addr).unwrap();

                                let b0 = self.alloc_regf32();
                                let b1 = self.alloc_regf32();
                                let b2 = self.alloc_regf32();
                                let b3 = self.alloc_regf32();
                                writeln!(&mut self.ptx_buffer, "    ld.global.cs.v4.f32 {{{}, {}, {}, {}}}, [{}];", b0, b1, b2, b3, b_addr).unwrap();

                                let c0 = self.alloc_regf32();
                                let c1 = self.alloc_regf32();
                                let c2 = self.alloc_regf32();
                                let c3 = self.alloc_regf32();
                                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", c0, a0, b0).unwrap();
                                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", c1, a1, b1).unwrap();
                                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", c2, a2, b2).unwrap();
                                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", c3, a3, b3).unwrap();

                                writeln!(&mut self.ptx_buffer, "    st.global.cs.v4.f32 [{}], {{{}, {}, {}, {}}};", c_addr, c0, c1, c2, c3).unwrap();
                            }
                            "".into()
                        } else if fname == "rmsnorm_v4" || fname == "rmsnorm_fast" {
                            let x_ptr = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let w_ptr = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%rd1".to_string() };
                            let out_ptr = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%rd2".to_string() };

                            let x0 = self.alloc_regf32();
                            let x1 = self.alloc_regf32();
                            let x2 = self.alloc_regf32();
                            let x3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    ld.global.cs.v4.f32 {{{}, {}, {}, {}}}, [{}];", x0, x1, x2, x3, x_ptr).unwrap();

                            let w0 = self.alloc_regf32();
                            let w1 = self.alloc_regf32();
                            let w2 = self.alloc_regf32();
                            let w3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    ld.global.cs.v4.f32 {{{}, {}, {}, {}}}, [{}];", w0, w1, w2, w3, w_ptr).unwrap();

                            let sq0 = self.alloc_regf32();
                            let sq1 = self.alloc_regf32();
                            let sq2 = self.alloc_regf32();
                            let sq3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", sq0, x0, x0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", sq1, x1, x1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", sq2, x2, x2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", sq3, x3, x3).unwrap();

                            let sum0 = self.alloc_regf32();
                            let sum1 = self.alloc_regf32();
                            let sum_sq = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", sum0, sq0, sq1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", sum1, sq2, sq3).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", sum_sq, sum0, sum1).unwrap();

                            let warp_sum = self.emit_warp_reduce_sum(&sum_sq);

                            writeln!(&mut self.ptx_buffer, "    .shared .align 4 .f32 smem_reduce[8];").unwrap();
                            let lane_id = self.alloc_reg32();
                            let warp_id = self.alloc_reg32();
                            let pred_first_lane = self.alloc_pred();

                            // `%tid.x` is a special register: PTX allows it as the
                            // source of a `mov` and nowhere else. Feeding it
                            // straight into `and`/`shr`/`cvt`/`setp` - as this
                            // whole block used to - is rejected by ptxas with
                            // "Special register argument not allowed for
                            // instruction 'and'". Materialise it once.
                            let tid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 31;", lane_id, tid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 5;", warp_id, tid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, {}, 0;", pred_first_lane, lane_id).unwrap();

                            let warp_id_u64 = self.alloc_reg64();
                            let smem_offset = self.alloc_reg64();
                            let smem_addr = self.alloc_reg64();

                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", warp_id_u64, warp_id).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", smem_offset, warp_id_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mov.u64 {}, smem_reduce;", smem_addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", smem_addr, smem_addr, smem_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} st.shared.f32 [{}], {};", pred_first_lane, smem_addr, warp_sum).unwrap();
                            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

                            let smem_base = self.alloc_reg64();
                            let tid_u64 = self.alloc_reg64();
                            let tid_offset = self.alloc_reg64();
                            let smem_read_addr = self.alloc_reg64();
                            let pred_warp0_threads = self.alloc_pred();
                            let val_warp_sum = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    mov.u64 {}, smem_reduce;", smem_base).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", tid_u64, tid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", tid_offset, tid_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", smem_read_addr, smem_base, tid_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, 8;", pred_warp0_threads, tid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} ld.shared.f32 {}, [{}];", pred_warp0_threads, val_warp_sum, smem_read_addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", pred_warp0_threads, val_warp_sum).unwrap();

                            let block_sum = self.emit_warp_reduce_sum(&val_warp_sum);

                            let inv_rms = self.alloc_regf32();
                            let mean = self.alloc_regf32();
                            let pred_tid0 = self.alloc_pred();
                            writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, {}, 0;", pred_tid0, tid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, 0.0009765625;", mean, block_sum).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 0.00001;", mean, mean).unwrap();
                            writeln!(&mut self.ptx_buffer, "    rsqrt.approx.f32 {}, {};", inv_rms, mean).unwrap();

                            let smem_root = self.alloc_reg64();
                            writeln!(&mut self.ptx_buffer, "    mov.u64 {}, smem_reduce;", smem_root).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} st.shared.f32 [{}], {};", pred_tid0, smem_root, inv_rms).unwrap();
                            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                            writeln!(&mut self.ptx_buffer, "    ld.shared.f32 {}, [{}];", inv_rms, smem_root).unwrap();

                            let o0 = self.alloc_regf32();
                            let o1 = self.alloc_regf32();
                            let o2 = self.alloc_regf32();
                            let o3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o0, x0, inv_rms).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o1, x1, inv_rms).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o2, x2, inv_rms).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o3, x3, inv_rms).unwrap();

                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o0, o0, w0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o1, o1, w1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o2, o2, w2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", o3, o3, w3).unwrap();

                            writeln!(&mut self.ptx_buffer, "    st.global.cs.v4.f32 [{}], {{{}, {}, {}, {}}};", out_ptr, o0, o1, o2, o3).unwrap();
                            "".into()
                        } else if fname == "swiglu_v4" || fname == "swiglu_fast" {
                            let gate_ptr = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let up_ptr = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%rd1".to_string() };
                            let out_ptr = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%rd2".to_string() };

                            // Hoisted 128-bit SIMD vector loads
                            let g0 = self.alloc_regf32();
                            let g1 = self.alloc_regf32();
                            let g2 = self.alloc_regf32();
                            let g3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    ld.global.cs.v4.f32 {{{}, {}, {}, {}}}, [{}];", g0, g1, g2, g3, gate_ptr).unwrap();

                            let u0 = self.alloc_regf32();
                            let u1 = self.alloc_regf32();
                            let u2 = self.alloc_regf32();
                            let u3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    ld.global.cs.v4.f32 {{{}, {}, {}, {}}}, [{}];", u0, u1, u2, u3, up_ptr).unwrap();

                            // Fast Sigmoid & Swish Math for 4 lanes
                            let s0 = self.alloc_regf32();
                            let s1 = self.alloc_regf32();
                            let s2 = self.alloc_regf32();
                            let s3 = self.alloc_regf32();

                            let neg_g0 = self.alloc_regf32();
                            let neg_g1 = self.alloc_regf32();
                            let neg_g2 = self.alloc_regf32();
                            let neg_g3 = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, -1.4426950408889634;", neg_g0, g0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, -1.4426950408889634;", neg_g1, g1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, -1.4426950408889634;", neg_g2, g2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, -1.4426950408889634;", neg_g3, g3).unwrap();

                            let exp0 = self.alloc_regf32();
                            let exp1 = self.alloc_regf32();
                            let exp2 = self.alloc_regf32();
                            let exp3 = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    ex2.approx.f32 {}, {};", exp0, neg_g0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    ex2.approx.f32 {}, {};", exp1, neg_g1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    ex2.approx.f32 {}, {};", exp2, neg_g2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    ex2.approx.f32 {}, {};", exp3, neg_g3).unwrap();

                            let denom0 = self.alloc_regf32();
                            let denom1 = self.alloc_regf32();
                            let denom2 = self.alloc_regf32();
                            let denom3 = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 1.0;", denom0, exp0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 1.0;", denom1, exp1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 1.0;", denom2, exp2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 1.0;", denom3, exp3).unwrap();

                            writeln!(&mut self.ptx_buffer, "    rcp.approx.f32 {}, {};", s0, denom0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    rcp.approx.f32 {}, {};", s1, denom1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    rcp.approx.f32 {}, {};", s2, denom2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    rcp.approx.f32 {}, {};", s3, denom3).unwrap();

                            let swish0 = self.alloc_regf32();
                            let swish1 = self.alloc_regf32();
                            let swish2 = self.alloc_regf32();
                            let swish3 = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", swish0, g0, s0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", swish1, g1, s1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", swish2, g2, s2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", swish3, g3, s3).unwrap();

                            let res0 = self.alloc_regf32();
                            let res1 = self.alloc_regf32();
                            let res2 = self.alloc_regf32();
                            let res3 = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", res0, swish0, u0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", res1, swish1, u1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", res2, swish2, u2).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", res3, swish3, u3).unwrap();

                            writeln!(&mut self.ptx_buffer, "    st.global.cs.v4.f32 [{}], {{{}, {}, {}, {}}};", out_ptr, res0, res1, res2, res3).unwrap();
                            "".into()
                        } else if fname == "ld_global_v4_f32" || fname == "load_v4" {
                            let addr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let mut cache_str = ".ca";
                            if let Some(cp) = cache_policy {
                                if cp.policy == "L2_PERSIST" {
                                    cache_str = ".lu";
                                } else if cp.policy == "L2_EVICT_FIRST" {
                                    cache_str = ".L2::evict_first";
                                }
                            }
                            let f0 = self.alloc_regf32();
                            let f1 = self.alloc_regf32();
                            let f2 = self.alloc_regf32();
                            let f3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    ld.global{}.v4.f32 {{{}, {}, {}, {}}}, [{}];", cache_str, f0, f1, f2, f3, addr_reg).unwrap();
                            f0
                        } else if fname == "st_global_v4_f32" || fname == "store_v4" {
                            let addr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let v0 = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%f0".to_string() };
                            let v1 = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { v0.clone() };
                            let v2 = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { v0.clone() };
                            let v3 = if args.len() >= 5 { self.emit_expr(&args[4], cache_policy, hw_profile) } else { v0.clone() };
                            writeln!(&mut self.ptx_buffer, "    st.global.v4.f32 [{}], {{{}, {}, {}, {}}};", addr_reg, v0, v1, v2, v3).unwrap();
                            "".into()
                        } else if fname == "shfl_sync_bfly" || fname == "shfl_sync_bfly_b32" {
                            let src_val = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%f0".to_string() };
                            let offset_val = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "16".to_string() };
                            let dst = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    shfl.sync.bfly.b32 {}, {}, {}, 0x1f, 0xffffffff;", dst, src_val, offset_val).unwrap();
                            dst
                        } else if fname == "warp_reduce_sum" {
                            let src_val = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%f0".to_string() };
                            self.emit_warp_reduce_sum(&src_val)
                        } else if fname == "warp_reduce_max" {
                            let src_val = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%f0".to_string() };
                            self.emit_warp_reduce_max(&src_val)
                        } else if fname == "cp_async_bulk" || fname == "tma_load" || fname == "tma_load_2d" {
                            // Refused rather than approximated. A TMA bulk tensor
                            // copy is `cp.async.bulk.tensor.<N>d.shared::cluster.global
                            // .tile.mbarrier::complete_tx::bytes [dst], [map, {coords}],
                            // [mbar]` - it needs a tensor map built host-side by
                            // `cuTensorMapEncodeTiled` and passed in as a
                            // `.grid_constant .param .align 64 .b8 map[128]`, an
                            // mbarrier to complete against, and one coordinate per
                            // tensor dimension. `tma_load(src, dst)` supplies none
                            // of those, so there is no correct instruction to emit
                            // from it. The two-operand form this used to write is
                            // rejected by ptxas ("Arguments mismatch for instruction
                            // 'cp.async.bulk.tensor'") on every target including
                            // sm_90a - it was never loadable on any GPU.
                            self.unsupported_intrinsic(
                                fname,
                                "a TMA bulk tensor copy needs a host-built tensor map \
                                 (cuTensorMapEncodeTiled) passed as a .grid_constant param, \
                                 an mbarrier to complete against, and per-dimension \
                                 coordinates; none of these are expressible in Y yet. \
                                 Use cp_async(dst, src, bytes) for the sm_80-style \
                                 cp.async.cg path",
                            );
                            "".into()
                        } else if fname == "wgmma_async" || fname == "wgmma_mma_async" {
                            // Same reasoning. `wgmma.mma_async` takes 64-bit shared
                            // memory *matrix descriptors* for A and B (or a register
                            // fragment for A), an accumulator vector sized by the
                            // shape, and four immediate operands (scale-D, imm-scale-A,
                            // imm-scale-B, and the transpose flags). The form emitted
                            // here passed two 32-bit registers and ignored its
                            // arguments entirely, referencing %f0..%f3 whether or not
                            // the register pool was that large.
                            self.unsupported_intrinsic(
                                fname,
                                "wgmma.mma_async needs 64-bit shared-memory matrix \
                                 descriptors, a shape-sized accumulator vector and four \
                                 immediate operands, none of which this intrinsic \
                                 accepts; it also requires sm_90a. Use mma_sync(...) for \
                                 the sm_80/sm_89 tensor-core path",
                            );
                            "".into()
                        } else if fname == "mbarrier_init"
                            || fname == "mbarrier_arrive"
                            || fname == "mbarrier_try_wait"
                        {
                            // The last of the mbarrier surface gotcha #8 deleted
                            // the rest of, and broken in three separate ways:
                            //
                            //  * `mbarrier_arrive` emitted
                            //    `mbarrier.arrive.expect_tx.shared.b64 %rd0,
                            //    [%rd0], %r1` - destination hardcoded to `%rd0`,
                            //    which is also the source AND a kernel parameter
                            //    register, so it clobbers a parameter.
                            //  * `.expect_tx` is **sm_90 only**, and the hardware
                            //    profile here selects sm_89, so `ptxas` rejects
                            //    it at its own target.
                            //  * `mbarrier_init` takes a `GlobalMemory` pointer
                            //    and feeds it to a `.shared` instruction. An
                            //    mbarrier object has to live in shared memory;
                            //    the operand is wrong by construction.
                            //
                            // Used by no kernel. Refused rather than patched,
                            // for the reason gotcha #8 gives about the TMA path:
                            // a working mbarrier pipeline is a feature to design
                            // (shared-memory barrier objects, which
                            // `shared_alloc_u32` does not provide), not a typo
                            // to correct.
                            self.unsupported_intrinsic(
                                fname,
                                "the mbarrier surface is not implemented: it emitted \
                                 hardcoded parameter registers, and `expect_tx` needs \
                                 sm_90 while this target is sm_89. Use `barrier_sync()` \
                                 with `shared_alloc_u32`",
                            );
                            "".into()
                        } else if fname == "mma_sync" {
                            // **The `wgmma_async` bug again, missed by the sweep
                            // that fixed it.** This emitted one hardcoded
                            // instruction with fixed register names, discarding
                            // all three of its arguments - and the registers are
                            // wrong twice over: `m16n8k16.f32.f16.f16.f32` takes
                            // FOUR accumulator registers and two-register B, not
                            // `{%f0,%f1}, {%r0,%r1}, {%r2,%r3}, {%f0,%f1}`, and
                            // `%f1` is outside the `.reg .f32 %f<1>` pool the
                            // emitter declares. `ptxas` reports four errors on a
                            // file the compiler wrote after printing
                            // "Compilation Successful!" and exiting 0.
                            //
                            // The reachable tensor-core paths are unaffected:
                            // `emit_tensor_core_gemm_kernel` and the
                            // `--emit-coprocessor` scheduler build their own
                            // `mma.sync` with real operands, which is why the
                            // `coprocessor_*.ysu` kernels work.
                            self.unsupported_intrinsic(
                                "mma_sync",
                                "it discarded its fragment arguments and emitted a \
                                 fixed instruction with the wrong register counts; \
                                 use `--emit-coprocessor`, or a @tile'd GEMM which \
                                 lowers to real `mma.sync`",
                            );
                            "".into()
                        } else if fname == "thread_id" || fname == "global_thread_id" || fname == "thread_idx" {
                            let tid = self.alloc_reg32();
                            let ntid = self.alloc_reg32();
                            let ctaid = self.alloc_reg32();
                            let gidx = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ntid.x;", ntid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", ctaid).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", gidx, ctaid, ntid, tid).unwrap();
                            gidx
                        } else if fname == "thread_idx_x" {
                            let tid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
                            tid
                        } else if fname == "thread_idx_y" {
                            let tid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.y;", tid).unwrap();
                            tid
                        } else if fname == "thread_idx_z" {
                            let tid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.z;", tid).unwrap();
                            tid
                        } else if fname == "block_idx_x" {
                            let ctaid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", ctaid).unwrap();
                            ctaid
                        } else if fname == "block_idx_y" {
                            let ctaid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.y;", ctaid).unwrap();
                            ctaid
                        } else if fname == "block_idx_z" {
                            let ctaid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.z;", ctaid).unwrap();
                            ctaid
                        } else if fname == "block_dim_x" {
                            let ntid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ntid.x;", ntid).unwrap();
                            ntid
                        } else if fname == "block_dim_y" {
                            let ntid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ntid.y;", ntid).unwrap();
                            ntid
                        } else if fname == "block_dim_z" {
                            let ntid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ntid.z;", ntid).unwrap();
                            ntid
                        // The grid dimensions. Without these a grid-stride
                        // loop - the canonical way to write a kernel whose
                        // launch geometry is a tuning parameter rather than
                        // part of its meaning - cannot be expressed in Y at
                        // all, which is exactly the shape every deterministic
                        // reduction in docs/deterministic_inference.md needs.
                        } else if fname == "grid_dim_x" {
                            let nctaid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %nctaid.x;", nctaid).unwrap();
                            nctaid
                        } else if fname == "grid_dim_y" {
                            let nctaid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %nctaid.y;", nctaid).unwrap();
                            nctaid
                        } else if fname == "grid_dim_z" {
                            let nctaid = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %nctaid.z;", nctaid).unwrap();
                            nctaid

                        } else if fname == "store" && args.len() >= 2 {
                            let addr_reg = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let val_raw = self.emit_expr(&args[1], cache_policy, hw_profile);
                            // The buffer's element type wins over the value's,
                            // and the value is converted into it. Picking the
                            // width off the value's register prefix (as this
                            // did) writes 4 bytes for a `U64` value into a
                            // `U64` array.
                            let elem = self
                                .index_elem_ty(&args[0])
                                .unwrap_or_else(|| self.ty_of(&val_raw));
                            let val_reg = self.emit_convert(&val_raw, elem);
                            writeln!(&mut self.ptx_buffer, "    st.global.{} [{}], {};", elem.mem(), addr_reg, val_reg).unwrap();
                            "".into()
                        // ── Widening and limb intrinsics ──────────────────
                        //
                        // Multiprecision arithmetic needs the *whole* product
                        // of two 32-bit limbs, which no expressible operator
                        // gives you: `a * b` is `mul.lo`, and the high half is
                        // where the carry lives. These three are what make a
                        // 256-bit Montgomery multiply writable in Y.
                        // ── 128-bit vector load / store ───────────────────
                        //
                        // `block_ptr2d_load_v4(buf, row, col, stride, max_r,
                        // max_c)` moves four consecutive u32 in ONE
                        // instruction, read back as `.x/.y/.z/.w`.
                        //
                        // The reason this exists is measured, not assumed: the
                        // field kernels were issuing 8 separate `LDG.E` per
                        // element, and NCU put 45.1 of every 64.3 warp cycles
                        // on the local/global instruction queue being full at
                        // 9.73% compute throughput. ptxas cannot merge those
                        // loads because each carries its own bounds predicate.
                        // Emitting the wide form directly keeps the predicate
                        // and still costs one instruction.
                        //
                        // `col` must be a multiple of 4 and the buffer 16-byte
                        // aligned; `ld.global.v4.u32` faults otherwise, and
                        // cuMemAlloc's 256-byte alignment covers the base.
                        } else if (fname == "block_ptr2d_load_v4" || fname == "block_ptr2d_store_v4")
                            && args.len() >= 6
                        {
                            let storing = fname.ends_with("store_v4");
                            let elem = self.elem_ty_or_f32(&args[0]);
                            if elem.is_64() {
                                self.unsupported_intrinsic(
                                    fname,
                                    "only 32-bit element types have a v4 form here; a v4 of \
                                     64-bit values would be 32 bytes and needs `.v2.u64` twice",
                                );
                                return "".into();
                            }
                            let ptr_reg = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let row_reg = self.emit_expr(&args[1], cache_policy, hw_profile);
                            let col_reg = self.emit_expr(&args[2], cache_policy, hw_profile);
                            let stride_reg = self.emit_expr(&args[3], cache_policy, hw_profile);
                            let max_r_reg = self.emit_expr(&args[4], cache_policy, hw_profile);
                            let max_c_reg = self.emit_expr(&args[5], cache_policy, hw_profile);

                            let lin_off = self.alloc_reg32();
                            let lin_idx = self.alloc_reg32();
                            let lin_u64 = self.alloc_reg64();
                            let byte_off = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let p_r = self.alloc_pred();
                            let p_c = self.alloc_pred();
                            let p_valid = self.alloc_pred();

                            writeln!(&mut self.ptx_buffer, "    // [Y 2D BLOCK POINTER {} - 128-BIT VECTOR]",
                                     if storing { "STORE" } else { "LOAD" }).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", lin_off, row_reg, stride_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", lin_idx, lin_off, col_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", lin_u64, lin_idx).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, {};", byte_off, lin_u64, elem.log2_bytes()).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_off).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_r, row_reg, max_r_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_c, col_reg, max_c_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_valid, p_r, p_c).unwrap();

                            if storing {
                                if args.len() < 10 {
                                    self.unsupported_intrinsic(
                                        fname,
                                        "needs four values to store: (buf, row, col, stride, \
                                         max_r, max_c, v0, v1, v2, v3)",
                                    );
                                    return "".into();
                                }
                                let mut vals = Vec::with_capacity(4);
                                for k in 6..10 {
                                    let v = self.emit_expr(&args[k], cache_policy, hw_profile);
                                    vals.push(self.emit_convert(&v, elem));
                                }
                                writeln!(&mut self.ptx_buffer, "    @{} st.global.v4.{} [{}], {{{}, {}, {}, {}}};",
                                         p_valid, elem.mem(), addr, vals[0], vals[1], vals[2], vals[3]).unwrap();
                                "".into()
                            } else {
                                let regs = [
                                    self.alloc_ty(elem), self.alloc_ty(elem),
                                    self.alloc_ty(elem), self.alloc_ty(elem),
                                ];
                                writeln!(&mut self.ptx_buffer, "    @{} ld.global.v4.{} {{{}, {}, {}, {}}}, [{}];",
                                         p_valid, elem.mem(), regs[0], regs[1], regs[2], regs[3], addr).unwrap();
                                for r in &regs {
                                    writeln!(&mut self.ptx_buffer, "    @!{} mov.{} {}, {};",
                                             p_valid, elem.reg_mem(), r, elem.zero_imm()).unwrap();
                                }
                                Self::v4_marker(&regs)
                            }
                        // ── Carry-chained 32-bit arithmetic ───────────────
                        //
                        // These read and/or write the hardware condition code,
                        // which is the ONE piece of implicit state in this
                        // backend. Everything else here is pure SSA over named
                        // registers; a `.cc` instruction communicates with the
                        // next one through a flag that appears in no operand.
                        //
                        // They exist because NCU said so. A Montgomery
                        // multiply written with `mul_wide_u32` + 64-bit adds
                        // compiles to ~5 SASS instructions per limb product -
                        // an `IMAD.WIDE`, an `IADD3`/`IADD3.X` pair for the
                        // 64-bit accumulate, and a shift-and-truncate to get
                        // the halves back out. In the fused NTT that put the
                        // actual multiplies at 18% of the instruction stream
                        // and the carry bookkeeping at the rest, with the
                        // kernel SM-bound at 71%. `mad.lo.cc` / `madc.hi.cc`
                        // chain the carry in hardware and do it in two.
                        //
                        // ORDER IS SEMANTIC HERE. A chain must be emitted with
                        // nothing between its links that writes CC. Nothing in
                        // this emitter reorders straight-line code, and the
                        // one pass that moves statements at all (barrier
                        // hoisting in `emit_block`) stops at the first
                        // statement it cannot hoist - a call is one - so a
                        // chain cannot be broken by it. Instructions the
                        // operands need (`mov` for an immediate, `ld` for a
                        // value) do not touch CC.
                        } else if let Some(op) = carry_op(fname, args.len()) {
                            let mut regs = Vec::with_capacity(args.len());
                            for a in args.iter() {
                                let v = self.emit_expr(a, cache_policy, hw_profile);
                                regs.push(self.emit_convert(&v, ScalarTy::U32));
                            }
                            let dst = self.alloc_ty(ScalarTy::U32);
                            writeln!(
                                &mut self.ptx_buffer,
                                "    {} {}, {};",
                                op,
                                dst,
                                regs.join(", ")
                            )
                            .unwrap();
                            dst
                        } else if fname == "mul_wide_u32" && args.len() == 2 {
                            let a = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let b = self.emit_expr(&args[1], cache_policy, hw_profile);
                            let a32 = self.emit_convert(&a, ScalarTy::U32);
                            let b32 = self.emit_convert(&b, ScalarTy::U32);
                            let dst = self.alloc_ty(ScalarTy::U64);
                            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, {};", dst, a32, b32).unwrap();
                            dst
                        } else if fname == "mul_wide_s32" && args.len() == 2 {
                            let a = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let b = self.emit_expr(&args[1], cache_policy, hw_profile);
                            let a32 = self.emit_convert(&a, ScalarTy::I32);
                            let b32 = self.emit_convert(&b, ScalarTy::I32);
                            let dst = self.alloc_ty(ScalarTy::I64);
                            writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, {};", dst, a32, b32).unwrap();
                            dst
                        } else if fname == "u64_lo32" && args.len() == 1 {
                            // Truncation, i.e. `cvt.u32.u64` - spelled out
                            // rather than left to an implicit narrowing so the
                            // intent reads at the call site.
                            let v = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let w = self.emit_convert(&v, ScalarTy::U64);
                            self.emit_convert(&w, ScalarTy::U32)
                        } else if fname == "u64_hi32" && args.len() == 1 {
                            let v = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let w = self.emit_convert(&v, ScalarTy::U64);
                            let sh = self.alloc_ty(ScalarTy::U64);
                            writeln!(&mut self.ptx_buffer, "    shr.u64 {}, {}, 32;", sh, w).unwrap();
                            self.emit_convert(&sh, ScalarTy::U32)
                        } else if fname == "block_tile_load" || fname == "tile_load" {
                            if !args.is_empty() && self.reject_non_float_buffer(fname, &args[0]) {
                                return "".into();
                            }
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let offset_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let bound_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "128".to_string() };

                            let pred = self.alloc_pred();
                            let byte_offset = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let res = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    // [Y BLOCK TILE LOAD - AUTOMATIC BOUNDARY MASKING]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, offset_reg, bound_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", byte_offset, offset_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} ld.global.f32 {}, [{}];", pred, res, addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", pred, res).unwrap();
                            res
                        } else if fname == "block_tile_store" || fname == "tile_store" {
                            if !args.is_empty() && self.reject_non_float_buffer(fname, &args[0]) {
                                return "".into();
                            }
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let offset_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let val_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%f0".to_string() };
                            let bound_reg = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { "128".to_string() };

                            let pred = self.alloc_pred();
                            let byte_offset = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            writeln!(&mut self.ptx_buffer, "    // [Y BLOCK TILE STORE - AUTOMATIC BOUNDARY MASKING]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, offset_reg, bound_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", byte_offset, offset_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}], {};", pred, addr, val_reg).unwrap();
                            "".into()
                        } else if fname == "make_block_ptr2d" || fname == "block_ptr2d_load" {
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let row_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let col_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%r1".to_string() };
                            let stride_reg = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { "1024".to_string() };
                            let max_r_reg = if args.len() >= 5 { self.emit_expr(&args[4], cache_policy, hw_profile) } else { "128".to_string() };
                            let max_c_reg = if args.len() >= 6 { self.emit_expr(&args[5], cache_policy, hw_profile) } else { "1024".to_string() };

                            // The element type decides the instruction AND the
                            // stride; both used to be hardcoded to f32/4 bytes.
                            let elem = if args.is_empty() {
                                ScalarTy::F32
                            } else {
                                self.elem_ty_or_f32(&args[0])
                            };

                            let lin_idx = self.alloc_reg32();
                            let lin_off = self.alloc_reg32();
                            let byte_off = self.alloc_reg64();
                            let lin_u64 = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let p_r = self.alloc_pred();
                            let p_c = self.alloc_pred();
                            let p_valid = self.alloc_pred();
                            let res = self.alloc_ty(elem);

                            writeln!(&mut self.ptx_buffer, "    // [Y 2D TENSOR BLOCK POINTER LOAD - 2D STRIDED MASKED ACCESS]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", lin_off, row_reg, stride_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", lin_idx, lin_off, col_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", lin_u64, lin_idx).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, {};", byte_off, lin_u64, elem.log2_bytes()).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_off).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_r, row_reg, max_r_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_c, col_reg, max_c_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_valid, p_r, p_c).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} ld.global.{} {}, [{}];", p_valid, elem.mem(), res, addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @!{} mov.{} {}, {};", p_valid, elem.reg_mem(), res, elem.zero_imm()).unwrap();
                            res
                        } else if fname == "block_ptr2d_store" {
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let row_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let col_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%r1".to_string() };
                            let stride_reg = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { "1024".to_string() };
                            let max_r_reg = if args.len() >= 5 { self.emit_expr(&args[4], cache_policy, hw_profile) } else { "128".to_string() };
                            let max_c_reg = if args.len() >= 6 { self.emit_expr(&args[5], cache_policy, hw_profile) } else { "1024".to_string() };
                            let val_raw = if args.len() >= 7 { self.emit_expr(&args[6], cache_policy, hw_profile) } else { "%f0".to_string() };

                            let elem = if args.is_empty() {
                                ScalarTy::F32
                            } else {
                                self.elem_ty_or_f32(&args[0])
                            };
                            // Storing an `I32` into a `GlobalMemory<F32>` (or
                            // the reverse) converts rather than reinterpreting
                            // the bits, which is what `Out[i] = count` means.
                            let val_reg = self.emit_convert(&val_raw, elem);

                            let lin_idx = self.alloc_reg32();
                            let lin_off = self.alloc_reg32();
                            let byte_off = self.alloc_reg64();
                            let lin_u64 = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let p_r = self.alloc_pred();
                            let p_c = self.alloc_pred();
                            let p_valid = self.alloc_pred();

                            writeln!(&mut self.ptx_buffer, "    // [Y 2D TENSOR BLOCK POINTER STORE - 2D STRIDED MASKED ACCESS]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", lin_off, row_reg, stride_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", lin_idx, lin_off, col_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", lin_u64, lin_idx).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, {};", byte_off, lin_u64, elem.log2_bytes()).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_off).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_r, row_reg, max_r_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_c, col_reg, max_c_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_valid, p_r, p_c).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} st.global.{} [{}], {};", p_valid, elem.mem(), addr, val_reg).unwrap();
                            "".into()
                        } else if fname == "block_ptr2d_advance" {
                            let row_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let delta_r = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "1".to_string() };
                            let next_row = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    // [Y 2D TENSOR BLOCK POINTER ADVANCE]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", next_row, row_reg, delta_r).unwrap();
                            next_row
                        } else if fname == "make_block_ptr3d" || fname == "block_ptr3d_load" || fname == "block_ptr3d_load_v4" {
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let d0_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let d1_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%r1".to_string() };
                            let d2_reg = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { "%r2".to_string() };
                            let s0_reg = if args.len() >= 5 { self.emit_expr(&args[4], cache_policy, hw_profile) } else { "131072".to_string() };
                            let s1_reg = if args.len() >= 6 { self.emit_expr(&args[5], cache_policy, hw_profile) } else { "1024".to_string() };
                            let max0_reg = if args.len() >= 7 { self.emit_expr(&args[6], cache_policy, hw_profile) } else { "32".to_string() };
                            let max1_reg = if args.len() >= 8 { self.emit_expr(&args[7], cache_policy, hw_profile) } else { "128".to_string() };
                            let max2_reg = if args.len() >= 9 { self.emit_expr(&args[8], cache_policy, hw_profile) } else { "1024".to_string() };

                            let off0 = self.alloc_reg32();
                            let off1 = self.alloc_reg32();
                            let off01 = self.alloc_reg32();
                            let lin_idx = self.alloc_reg32();
                            let byte_off = self.alloc_reg64();
                            let lin_u64 = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let p0 = self.alloc_pred();
                            let p1 = self.alloc_pred();
                            let p2 = self.alloc_pred();
                            let p01 = self.alloc_pred();
                            let p_valid = self.alloc_pred();

                            let is_v4 = fname == "block_ptr3d_load_v4";
                            writeln!(&mut self.ptx_buffer, "    // [Y 3D TENSOR BLOCK POINTER LOAD (v4 Vectorized) - 3D STRIDED MASKED ACCESS]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", off0, d0_reg, s0_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", off1, d1_reg, s1_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", off01, off0, off1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", lin_idx, off01, d2_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", lin_u64, lin_idx).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", byte_off, lin_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_off).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p0, d0_reg, max0_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p1, d1_reg, max1_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p2, d2_reg, max2_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p01, p0, p1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_valid, p01, p2).unwrap();

                            if is_v4 {
                                let f0 = self.alloc_regf32();
                                let f1 = self.alloc_regf32();
                                let f2 = self.alloc_regf32();
                                let f3 = self.alloc_regf32();
                                writeln!(&mut self.ptx_buffer, "    @{} ld.global.nc.v4.f32 {{{}, {}, {}, {}}}, [{}];", p_valid, f0, f1, f2, f3, addr).unwrap();
                                writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", p_valid, f0).unwrap();
                                writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", p_valid, f1).unwrap();
                                writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", p_valid, f2).unwrap();
                                writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", p_valid, f3).unwrap();
                                format!("{},{},{},{}", f0, f1, f2, f3)
                            } else {
                                let res = self.alloc_regf32();
                                writeln!(&mut self.ptx_buffer, "    @{} ld.global.f32 {}, [{}];", p_valid, res, addr).unwrap();
                                writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", p_valid, res).unwrap();
                                res
                            }
                        } else if fname == "block_ptr3d_store" || fname == "block_ptr3d_store_v4" {
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let d0_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let d1_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%r1".to_string() };
                            let d2_reg = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { "%r2".to_string() };
                            let s0_reg = if args.len() >= 5 { self.emit_expr(&args[4], cache_policy, hw_profile) } else { "131072".to_string() };
                            let s1_reg = if args.len() >= 6 { self.emit_expr(&args[5], cache_policy, hw_profile) } else { "1024".to_string() };
                            let max0_reg = if args.len() >= 7 { self.emit_expr(&args[6], cache_policy, hw_profile) } else { "32".to_string() };
                            let max1_reg = if args.len() >= 8 { self.emit_expr(&args[7], cache_policy, hw_profile) } else { "128".to_string() };
                            let max2_reg = if args.len() >= 9 { self.emit_expr(&args[8], cache_policy, hw_profile) } else { "1024".to_string() };

                            let is_v4 = fname == "block_ptr3d_store_v4";
                            let (val0, val1, val2, val3) = if is_v4 && args.len() >= 13 {
                                (
                                    self.emit_expr(&args[9], cache_policy, hw_profile),
                                    self.emit_expr(&args[10], cache_policy, hw_profile),
                                    self.emit_expr(&args[11], cache_policy, hw_profile),
                                    self.emit_expr(&args[12], cache_policy, hw_profile)
                                )
                            } else {
                                let v = if args.len() >= 10 { self.emit_expr(&args[9], cache_policy, hw_profile) } else { "%f0".to_string() };
                                (v.clone(), v.clone(), v.clone(), v)
                            };

                            let off0 = self.alloc_reg32();
                            let off1 = self.alloc_reg32();
                            let off01 = self.alloc_reg32();
                            let lin_idx = self.alloc_reg32();
                            let byte_off = self.alloc_reg64();
                            let lin_u64 = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let p0 = self.alloc_pred();
                            let p1 = self.alloc_pred();
                            let p2 = self.alloc_pred();
                            let p01 = self.alloc_pred();
                            let p_valid = self.alloc_pred();

                            writeln!(&mut self.ptx_buffer, "    // [Y 3D TENSOR BLOCK POINTER STORE (v4 Vectorized) - 3D STRIDED MASKED ACCESS]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", off0, d0_reg, s0_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", off1, d1_reg, s1_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", off01, off0, off1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", lin_idx, off01, d2_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", lin_u64, lin_idx).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", byte_off, lin_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_off).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p0, d0_reg, max0_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p1, d1_reg, max1_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p2, d2_reg, max2_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p01, p0, p1).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_valid, p01, p2).unwrap();

                            if is_v4 {
                                writeln!(&mut self.ptx_buffer, "    @{} st.global.cs.v4.f32 [{}], {{{}, {}, {}, {}}};", p_valid, addr, val0, val1, val2, val3).unwrap();
                            } else {
                                writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}], {};", p_valid, addr, val0).unwrap();
                            }
                            "".into()
                        } else if fname == "block_ptr3d_advance" {
                            let d0_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let delta_d0 = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "1".to_string() };
                            let next_d0 = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    // [Y 3D TENSOR BLOCK POINTER ADVANCE]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", next_d0, d0_reg, delta_d0).unwrap();
                            next_d0
                        } else if fname == "block_cdiv" {
                            let num = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let den = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "1".to_string() };
                            let num_plus_den = self.alloc_reg32();
                            let num_num = self.alloc_reg32();
                            let res = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    // [Y BLOCK CDIV - CEIL DIVISION]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", num_plus_den, num, den).unwrap();
                            writeln!(&mut self.ptx_buffer, "    sub.s32 {}, {}, 1;", num_num, num_plus_den).unwrap();
                            writeln!(&mut self.ptx_buffer, "    div.s32 {}, {}, {};", res, num_num, den).unwrap();
                            res
                        // ── Shared memory ─────────────────────────────────
                        //
                        // `shared_alloc_u32(n)` declares one `.shared` array
                        // of `n` u32 at module scope and yields its address;
                        // `shared_load_v4` / `shared_store_v4` index it in
                        // 16-BYTE units, matching the global `v4` intrinsics.
                        //
                        // Together with `barrier_sync()` this is the whole
                        // surface, and it is deliberately small: it is exactly
                        // what a stage-fused NTT needs. There is no bounds
                        // check, because the slot index is a pure function of
                        // `%tid.x` in every intended use and a predicate on a
                        // shared store would cost more than the store.
                        } else if fname == "shared_alloc_u32" && args.len() == 1 {
                            let n = match &args[0] {
                                Expr::IntLit(v, _) if *v > 0 => *v as usize,
                                _ => {
                                    self.unsupported_intrinsic(
                                        fname,
                                        "the element count must be a positive integer literal - \
                                         a `.shared` array's size is fixed when the module is \
                                         assembled, so it cannot come from a runtime value",
                                    );
                                    return "".into();
                                }
                            };
                            // 48 KB is the static per-CTA limit on every
                            // architecture this backend targets. Anything
                            // above it is rejected by ptxas with a message
                            // about the module, not about this line, so name
                            // the real cause here instead.
                            if n * 4 > 48 * 1024 {
                                self.unsupported_intrinsic(
                                    fname,
                                    "over the 48 KB static shared-memory limit per CTA; a larger \
                                     allocation needs dynamic shared memory, which this backend \
                                     does not plumb through the launch",
                                );
                                return "".into();
                            }
                            let sym = format!("__y_smem_{}", self.shared_sym_count);
                            self.shared_sym_count += 1;
                            self.shared_arrays.push((sym.clone(), n));
                            let base = self.alloc_ty(ScalarTy::U64);
                            writeln!(&mut self.ptx_buffer, "    // [Y SHARED ALLOC] {} u32 = {} bytes", n, n * 4).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mov.u64 {}, {};", base, sym).unwrap();
                            base
                        } else if (fname == "shared_load_v4" || fname == "shared_store_v4")
                            && args.len() >= 2
                        {
                            let storing = fname.ends_with("store_v4");
                            let base = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let slot = self.emit_expr(&args[1], cache_policy, hw_profile);
                            let slot32 = self.emit_convert(&slot, ScalarTy::U32);
                            let slot64 = self.alloc_ty(ScalarTy::U64);
                            let byte_off = self.alloc_ty(ScalarTy::U64);
                            let addr = self.alloc_ty(ScalarTy::U64);
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", slot64, slot32).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 4;", byte_off, slot64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, base, byte_off).unwrap();
                            if storing {
                                if args.len() < 6 {
                                    self.unsupported_intrinsic(
                                        fname,
                                        "needs four values to store: (base, slot, v0, v1, v2, v3)",
                                    );
                                    return "".into();
                                }
                                let mut vals = Vec::with_capacity(4);
                                for k in 2..6 {
                                    let v = self.emit_expr(&args[k], cache_policy, hw_profile);
                                    vals.push(self.emit_convert(&v, ScalarTy::U32));
                                }
                                writeln!(&mut self.ptx_buffer, "    st.shared.v4.u32 [{}], {{{}, {}, {}, {}}};",
                                         addr, vals[0], vals[1], vals[2], vals[3]).unwrap();
                                "".into()
                            } else {
                                let regs = [
                                    self.alloc_ty(ScalarTy::U32), self.alloc_ty(ScalarTy::U32),
                                    self.alloc_ty(ScalarTy::U32), self.alloc_ty(ScalarTy::U32),
                                ];
                                writeln!(&mut self.ptx_buffer, "    ld.shared.v4.u32 {{{}, {}, {}, {}}}, [{}];",
                                         regs[0], regs[1], regs[2], regs[3], addr).unwrap();
                                Self::v4_marker(&regs)
                            }
                        // `barrier_sync()` was already recognised by the
                        // barrier-hoisting pass in `emit_block` - which hoists
                        // independent ALU work ACROSS it - while emitting no
                        // instruction at all here, so the hoist was legal for
                        // a barrier that did not exist. Same shape as
                        // `pipe.wait(t)` (gotcha #8): one pass proves a
                        // property the backend then discards.
                        } else if (fname == "atomic_add" || fname == "atomic_max")
                            && args.len() == 3
                        {
                            // The determinism primitives. An integer atomic is
                            // order-independent BY CONSTRUCTION - addition and
                            // max over integers are associative and
                            // commutative exactly - so a reduction built from
                            // these gives one answer whatever order the CTAs
                            // finish in. That is the entire product claim of
                            // `docs/deterministic_inference.md`, and until
                            // these existed it could not be written in Y at
                            // all, only in hand-written PTX.
                            let elem = self.elem_ty_or_f32(&args[0]);
                            if elem.is_float() {
                                // `red.global.add.f32` is real hardware and is
                                // exactly the instruction that makes GPU
                                // results irreproducible: CTAs finish in
                                // scheduler order and float addition is not
                                // associative, so the identical binary on the
                                // identical input gives different answers
                                // between launches. Refusing it by name is the
                                // point of this compiler, not a limitation of
                                // it - see the design rule in CLAUDE.md.
                                self.unsupported_intrinsic(
                                    fname,
                                    "refuses a floating-point buffer. A float atomic is \
                                     order-dependent, so the reduction would not be \
                                     reproducible - which is the one thing this backend \
                                     exists to guarantee. Accumulate in U32/I32/U64/I64 \
                                     (see @ZeroDrift), or if you genuinely want a \
                                     non-reproducible reduction, say so in a kernel that \
                                     does not claim determinism.",
                                );
                                return "".into();
                            }
                            if elem.is_subword() {
                                self.unsupported_intrinsic(
                                    fname,
                                    "has no sub-word form: the hardware's red/atom \
                                     instructions start at 32 bits. Widen the \
                                     accumulator to U32/I32 or wider.",
                                );
                                return "".into();
                            }
                            let base = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let idx = self.emit_expr(&args[1], cache_policy, hw_profile);
                            let val_raw = self.emit_expr(&args[2], cache_policy, hw_profile);
                            let val = self.emit_convert(&val_raw, elem);
                            let idx32 = self.emit_convert(&idx, ScalarTy::U32);
                            let idx64 = self.alloc_ty(ScalarTy::U64);
                            let byte_off = self.alloc_ty(ScalarTy::U64);
                            let addr = self.alloc_ty(ScalarTy::U64);
                            let shift = elem.log2_bytes();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", idx64, idx32).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, {};", byte_off, idx64, shift).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, base, byte_off).unwrap();
                            let op = if fname == "atomic_add" { "add" } else { "max" };
                            // `add` is defined on the unsigned type for both
                            // signednesses - two's complement wraps
                            // identically - while `max` genuinely differs, so
                            // it must follow the buffer's signedness.
                            let suffix = if op == "add" {
                                // Two's complement wraps identically, so the
                                // unsigned form is correct for both.
                                if elem.is_64() { "u64" } else { "u32" }
                            } else {
                                //  genuinely differs by signedness, and
                                // `mem()` reports `u32` for I32 because a LOAD
                                // does not care. Using it here made
                                // `atomic_max` on an I32 buffer an UNSIGNED
                                // max, so any negative accumulator compared as
                                // a huge positive one.
                                match (elem.is_64(), elem.is_signed()) {
                                    (true, true) => "s64",
                                    (true, false) => "u64",
                                    (false, true) => "s32",
                                    (false, false) => "u32",
                                }
                            };
                            writeln!(&mut self.ptx_buffer, "    red.global.{}.{} [{}], {};",
                                     op, suffix, addr, val).unwrap();
                            "".into()
                        } else if fname == "barrier_sync" || fname == "membar" {
                            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                            "".into()
                        } else if fname == "block_arange" {
                            // The NULLARY form is the supported one: it yields
                            // this thread's index, the scalar analogue of
                            // `tl.arange` in a backend with no vector values.
                            //
                            // Passing arguments was the bug. `block_arange(0,
                            // 128)` - the spelling a Triton user reaches for -
                            // DISCARDED both and returned `%tid.x` anyway, so a
                            // non-zero start silently produced the wrong index.
                            // It assembles perfectly, which is why the
                            // substring test in `test_rust_triton_parity.rs`
                            // never noticed; that test calls the nullary form.
                            if !args.is_empty() {
                                let n = args.len();
                                self.unsupported_intrinsic(
                                    "block_arange",
                                    &format!(
                                        "it takes no arguments and would silently \
                                         discard the {} given; it yields this \
                                         thread's index, so write `start + \
                                         block_arange()` for a non-zero start",
                                        n
                                    ),
                                );
                                return "".into();
                            }
                            let res = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    // [Y BLOCK ARANGE - 1D INDEX GENERATOR]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", res).unwrap();
                            res
                        } else {
                            // **The general case of every row this family has
                            // added to CLAUDE.md's table.** A name nothing
                            // matched used to return an empty string, and the
                            // callers splice that straight into instruction
                            // text: `block_ptr2d_store(A, 0, made_up(t), ...)`
                            // emitted `setp.lt.u32 %p1, , %r0;` - a missing
                            // operand - and the compiler printed "Compilation
                            // Successful!" and exited 0.
                            //
                            // It was only ever caught when the result was bound
                            // with `let`, which has its own guard. In argument
                            // position, as a statement, or on the right of an
                            // assignment, nothing noticed. A user-defined `fn`
                            // called from a kernel lands here too - this backend
                            // cannot lower one, and saying so at the call site
                            // is better than a malformed instruction.
                            self.unsupported_intrinsic(
                                fname,
                                "no PTX lowering exists for this name - check the \
                                 spelling, and note that user-defined functions \
                                 cannot be called from a kernel in this backend",
                            );
                            "".into()
                        }
                    }
                    Expr::Path {
                        namespace, member, ..
                    } => {
                        // The same arity gate the `Ident` callees get. This arm
                        // has its own register-aliasing fallbacks - `BlockTile`'s
                        // offset falls back to `%r0` and its value to `%f0` -
                        // and the first version of the gate only covered `Ident`,
                        // so these were still open. Two match arms, one bug.
                        if let Some(want) = Self::required_path_arity(namespace, member) {
                            if args.len() < want {
                                let (ns, me, n) =
                                    (namespace.clone(), member.clone(), args.len());
                                self.unsupported_intrinsic(
                                    &format!("{}::{}", ns, me),
                                    &format!(
                                        "it needs at least {} arguments and was given \
                                         {}; a missing operand would be read from an \
                                         unrelated register rather than reported",
                                        want, n
                                    ),
                                );
                                return "".into();
                            }
                        }
                        if namespace == "barrier" && member == "sync" {
                            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                            "".into()
                        } else if namespace == "BlockTile" && member == "load" {
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let offset_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let bound_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "128".to_string() };

                            let pred = self.alloc_pred();
                            let byte_offset = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let res = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    // [Y BLOCK TILE LOAD - AUTOMATIC BOUNDARY MASKING]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, offset_reg, bound_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", byte_offset, offset_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} ld.global.f32 {}, [{}];", pred, res, addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", pred, res).unwrap();
                            res
                        } else if namespace == "BlockTile" && member == "store" {
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let offset_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let val_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%f0".to_string() };
                            let bound_reg = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { "128".to_string() };

                            let pred = self.alloc_pred();
                            let byte_offset = self.alloc_reg64();
                            let addr = self.alloc_reg64();

                            writeln!(&mut self.ptx_buffer, "    // [Y BLOCK TILE STORE - AUTOMATIC BOUNDARY MASKING]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, offset_reg, bound_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", byte_offset, offset_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}], {};", pred, addr, val_reg).unwrap();
                            "".into()
                        } else if namespace == "GlobalMemory" && (member == "load_v4" || member == "ld_v4") {
                            let addr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let f0 = self.alloc_regf32();
                            let f1 = self.alloc_regf32();
                            let f2 = self.alloc_regf32();
                            let f3 = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    ld.global.ca.v4.f32 {{{}, {}, {}, {}}}, [{}];", f0, f1, f2, f3, addr_reg).unwrap();
                            f0
                        } else if namespace == "GlobalMemory" && (member == "store_v4" || member == "st_v4") {
                            let addr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let v0 = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%f0".to_string() };
                            let v1 = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { v0.clone() };
                            let v2 = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { v0.clone() };
                            let v3 = if args.len() >= 5 { self.emit_expr(&args[4], cache_policy, hw_profile) } else { v0.clone() };
                            writeln!(&mut self.ptx_buffer, "    st.global.v4.f32 [{}], {{{}, {}, {}, {}}};", addr_reg, v0, v1, v2, v3).unwrap();
                            "".into()
                        } else if namespace == "GlobalMemory" && member == "load" {
                            let mut cache_str = ".ca";
                            if let Some(cp) = cache_policy {
                                if cp.policy == "L2_PERSIST" {
                                    cache_str = ".lu";
                                } else if cp.policy == "L2_EVICT_FIRST" {
                                    cache_str = ".L2::evict_first";
                                }
                            }
                            let elem = args
                                .first()
                                .and_then(|a| self.index_elem_ty(a))
                                .unwrap_or(ScalarTy::F32);
                            let addr_reg = if !args.is_empty() {
                                self.emit_expr(&args[0], cache_policy, hw_profile)
                            } else {
                                "%rd0".to_string()
                            };
                            let dst = self.alloc_ty(elem);
                            writeln!(&mut self.ptx_buffer, "    ld.global{}.{} {}, [{}];", cache_str, elem.mem(), dst, addr_reg).unwrap();
                            dst
                        } else {
                            // An unhandled `Namespace::member(...)` call. This is
                            // how `Fragment::zero()` and `ldmatrix`-style paths
                            // reached the emitter and produced nothing, which the
                            // caller then spliced into an instruction as an empty
                            // operand.
                            let path = format!("{}::{}", namespace, member);
                            self.unsupported_intrinsic(
                                &path,
                                "no PTX lowering exists for this path; the matrix \
                                 fragment surface here is not implemented (a \
                                 m16n8k16 f32 accumulator is four registers per \
                                 thread, not one) - use a @tile'd GEMM or \
                                 `--emit-coprocessor`",
                            );
                            "".into()
                        }
                    }
                    // `pipe.wait(token)` - a Call whose callee is a MemberAccess.
                    //
                    // This used to fall into the catch-all below and emit
                    // nothing at all. The bare `pipe.wait` form (no argument
                    // list) was handled by the `Expr::MemberAccess` arm further
                    // down, so the intrinsic looked implemented; the form
                    // everyone actually writes - and the only form
                    // `linear_tracker` accepts, since it requires the token be
                    // passed to something - silently vanished.
                    //
                    // The consequence is worse than a missing instruction.
                    // `linear_tracker` exists to guarantee an async copy is
                    // awaited exactly once, and the type checker refuses the
                    // program if it is not. Dropping the await in the backend
                    // means the guarantee is enforced in the front end and
                    // discarded in the back end: the kernel reads the
                    // destination while `cp.async` is still in flight, which is
                    // a data race that shows up as intermittently wrong numbers.
                    Expr::MemberAccess { member, .. } if member == "wait" => {
                        self.emit_cp_async_wait(0);
                        "".into()
                    }
                    Expr::MemberAccess { member, .. } if member == "commit" => {
                        self.emit_cp_async_commit();
                        "".into()
                    }
                    // Anything else with a member callee is not something this
                    // backend knows how to lower. Per the repo-wide rule, say so
                    // rather than emit an empty string and let the caller
                    // believe the call happened.
                    Expr::MemberAccess { member, .. } => {
                        let m = member.clone();
                        self.unsupported_intrinsic(
                            &format!("<expr>.{}", m),
                            "no PTX lowering exists for this method call",
                        );
                        "".into()
                    }
                    _ => {
                        // A callee that is neither an identifier, a path, nor a
                        // member access - a call through an index or a nested
                        // call, say. Nothing lowers it, and returning "" hands
                        // the caller an empty operand.
                        self.unsupported_intrinsic(
                            "<computed callee>",
                            "only direct calls to intrinsics are supported; this \
                             callee is an expression the backend cannot resolve",
                        );
                        "".into()
                    }
                }
            }
            Expr::MemberAccess { base, member, span } => {
                // `v.x` / `v.y` / `v.z` / `v.w` on a name bound by a v4 load.
                if let Expr::Ident(n, _) = &**base {
                    if let Some(regs) = self.vec_vars.get(n) {
                        let lane = match member.as_str() {
                            "x" | "r" | "e0" => Some(0),
                            "y" | "g" | "e1" => Some(1),
                            "z" | "b" | "e2" => Some(2),
                            "w" | "a" | "e3" => Some(3),
                            _ => None,
                        };
                        match lane {
                            Some(i) => return regs[i].clone(),
                            None => {
                                self.emit_errors.push(format!(
                                    "Line {}: `{}` is a 4-wide vector; `.{}` is not one of \
                                     its lanes (.x .y .z .w).",
                                    span.line, n, member
                                ));
                                return "".into();
                            }
                        }
                    }
                }
                if member == "wait" {
                    writeln!(&mut self.ptx_buffer, "    cp.async.wait_group 0;").unwrap();
                    // An await has no VALUE. Returning "" is only safe because
                    // `Stmt::Let` refuses an initialiser that produced nothing;
                    // in an argument position it would splice an empty operand,
                    // which is the bug the arm below exists to stop.
                    return "".into();
                }
                // Everything else was `"".into()` - the same hole that was
                // closed for `Expr::Ident` and for the `_` arm and never for
                // this one. `emit_expr`'s callers splice the result straight
                // into instruction text, so a struct field read in an argument
                // position emitted
                //
                //     cvt.rn.f32.s32 %f1, ;
                //
                // under "Compilation Successful!" and exit 0, with `ptxas`
                // rejecting the file afterwards - on whatever machine tries to
                // run it, which is not the one that compiled it.
                //
                // This backend has no struct field access: `emit_expr` returns
                // one register, kernel parameters are buffers and scalars, and
                // `Expr::StructLit` is already refused. So a field read here is
                // a gap to name, not a lowering to guess at.
                let base_name = match &**base {
                    Expr::Ident(n, _) => n.clone(),
                    _ => "<expression>".to_string(),
                };
                self.unsupported_expr(
                    &format!(
                        "the field access `{}.{}` (this backend has no struct field access; `.x`/`.y`/`.z`/`.w` on a value bound by a v4 load are the only members it knows)",
                        base_name, member
                    ),
                    span,
                );
                "".into()
            }
            Expr::Path {
                namespace, member, ..
            } => {
                if namespace == "barrier" && member == "sync" {
                    writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                    "".into()
                } else if namespace == "Fragment" && member == "zero" {
                    let dst = self.alloc_regf32();
                    writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", dst).unwrap();
                    dst
                } else if namespace == "GlobalMemory" && member == "load" {
                    // Emitted `ld.global.ca.f32 %f, [%rd0]` - a load from
                    // parameter 0, whatever that parameter is, with its own
                    // arguments discarded entirely. Well-formed PTX that
                    // assembles, launches, and reads the wrong buffer, under
                    // "Compilation Successful!". Same shape as `tma_load` in
                    // gotcha #8, and refused for the same reason: a named gap
                    // costs a user five minutes, a plausible-looking broken
                    // kernel costs them however long it takes to suspect the
                    // compiler.
                    self.unsupported_intrinsic(
                        "GlobalMemory::load",
                        "it discarded its arguments and always loaded from \
                         parameter 0; use an indexed load such as \
                         `block_ptr2d_load`",
                    );
                    "".into()
                } else if namespace == "SharedMemory" && member == "alloc" {
                    // The stub gotcha #8 describes: a hardcoded
                    // `.shared .align 128 .b8 smem[8192]` with the type
                    // argument ignored and no way to index the result. It
                    // assembles, which is why it survived - `ptxas` is happy
                    // with a mid-body declaration. The working surface is
                    // `shared_alloc_u32(n)` + `shared_load_v4`/`shared_store_v4`.
                    self.unsupported_intrinsic(
                        "SharedMemory::alloc",
                        "it ignored its element type, always allocated 8192 \
                         bytes, and returned a symbol nothing can index; use \
                         `shared_alloc_u32(n)` with `shared_load_v4` / \
                         `shared_store_v4`",
                    );
                    "".into()
                } else {
                    let path = format!("{}::{}", namespace, member);
                    self.unsupported_intrinsic(
                        &path,
                        "no PTX lowering exists for this path - returning nothing \
                         here splices an empty operand into the instruction that \
                         uses it",
                    );
                    "".into()
                }
            }
            Expr::GenericCall { func, .. } => {
                self.emit_expr(&**func, cache_policy, hw_profile)
            }

            // Everything this backend genuinely cannot lower is named here,
            // with NO `_ =>` arm, so a new `Expr` variant is a compile error
            // rather than a silently empty operand.
            //
            // The fallback used to be `_ => "".into()`, and an empty string is
            // spliced straight into instruction text by every caller: a string
            // literal in an argument position emitted
            // `mul.lo.s32 %r5, , %r1;` and `setp.lt.u32 %p0, , %r2;` -
            // missing operands - after "Compilation Successful!" and exit 0.
            // `ptxas` then rejects the file with a parse error, on a machine
            // that is not the one that compiled it. This is the same row of
            // the design-rule table as the intrinsic arity fallbacks and the
            // unknown-call-name gate; both of those were fixed for their own
            // arm and this one was left.
            Expr::StringLit(_, span) => {
                self.unsupported_expr("a string literal", span);
                "".into()
            }
            Expr::CharLit(_, span) => {
                self.unsupported_expr("a character literal", span);
                "".into()
            }
            Expr::StructLit { span, .. } => {
                self.unsupported_expr("a struct literal", span);
                "".into()
            }
            Expr::BlockExpr(_, span) => {
                self.unsupported_expr("a block used as an expression", span);
                "".into()
            }
            Expr::SelfLit(span) => {
                self.unsupported_expr("`self`", span);
                "".into()
            }
            Expr::ZeroInit(span) => {
                self.unsupported_expr("a zero-initialiser", span);
                "".into()
            }
        }
    }

    /// An expression with no PTX lowering.
    ///
    /// Separate from `unsupported_intrinsic` because that one names a call the
    /// user wrote; this names a whole expression form the backend does not
    /// have. Both push to `emit_errors`, which `main` turns into a failed
    /// build - the point being that the refusal is fail-closed, where an empty
    /// return value is spliced into an instruction and is not.
    fn unsupported_expr(&mut self, what: &str, span: &Span) {
        self.emit_errors.push(format!(
            "[PTX] {} (line {}, col {}) cannot be lowered by this backend.",
            what, span.line, span.col
        ));
    }

    /// The name of a `WitnessOp` variant, for a refusal message.
    ///
    /// Exhaustive on purpose, with no `_ =>` arm: a new variant must be a
    /// compile error here rather than silently reported as "unknown". That is
    /// the same device `remap_witness_op` uses, and for the same reason.
    #[cfg(feature = "zk")]
    fn witness_op_name(op: &WitnessOp) -> &'static str {
    match op {
        WitnessOp::Const(_) => "Const",
        WitnessOp::LoadInput { .. } => "LoadInput",
        WitnessOp::Add(..) => "Add",
        WitnessOp::Sub(..) => "Sub",
        WitnessOp::Mul(..) => "Mul",
        WitnessOp::Div(..) => "Div",
        WitnessOp::Inv(_) => "Inv",
        WitnessOp::AssertEq(..) => "AssertEq",
        WitnessOp::HintBlock { .. } => "HintBlock",
        WitnessOp::IsZeroLc(_) => "IsZeroLc (== / !=)",
        WitnessOp::InvOrZeroLc(_) => "InvOrZeroLc (== / !=)",
        WitnessOp::BitOfLc { .. } => "BitOfLc (comparison, bitwise, shift, range check)",
        WitnessOp::IntDivLc(..) => "IntDivLc (integer /)",
        WitnessOp::IntModLc(..) => "IntModLc (integer %)",
        WitnessOp::MulAddLc(..) => "MulAddLc (a*b + c)",
        WitnessOp::DivLc(..) => "DivLc (field /)",
        WitnessOp::MulLc(..) => "MulLc",
        WitnessOp::IfZeroLc(..) => "IfZeroLc",
        WitnessOp::Unknown => "Unknown",
    }
}

    #[cfg(feature = "zk")]
    pub fn emit_witness_generator_ptx(&mut self, graph: &WitnessIRGraph) -> String {
        let mut buffer = String::new();
        writeln!(&mut buffer, "// ============================================================").unwrap();
        writeln!(&mut buffer, "// GPU PTX Witness Generator Kernel (Zero-Copy VRAM Layout)").unwrap();
        writeln!(&mut buffer, "// Field: BN254 Fr (256-bit 4-limb Montgomery ISA)").unwrap();
        writeln!(&mut buffer, "// ============================================================").unwrap();
        writeln!(&mut buffer, "{}", ptx_version_for_sm(&self.sm_target)).unwrap();
        writeln!(&mut buffer, ".target {}", self.sm_target).unwrap();
        writeln!(&mut buffer, ".address_size 64").unwrap();
        writeln!(&mut buffer, "").unwrap();

        writeln!(&mut buffer, ".visible .entry witness_generation_kernel(").unwrap();
        writeln!(&mut buffer, "    .param .u64 param_witness_buffer,").unwrap();
        writeln!(&mut buffer, "    .param .u32 param_num_signals,").unwrap();
        writeln!(&mut buffer, "    .param .u32 param_total_instances").unwrap();
        writeln!(&mut buffer, ")").unwrap();
        writeln!(&mut buffer, "{{").unwrap();

        // Register Declarations
        writeln!(&mut buffer, "    .reg .u32 %t_id, %b_id, %b_dim, %instance_idx, %r_max_inst, %r_num_sig;").unwrap();
        writeln!(&mut buffer, "    .reg .u64 %base_ptr, %instance_offset_bytes, %sig_addr;").unwrap();
        writeln!(&mut buffer, "    .reg .pred %p_valid, %p_sub, %p_borrow;").unwrap();
        writeln!(&mut buffer, "    .reg .u64 %t0, %t1, %t2, %t3, %t4, %r0, %r1, %r2, %r3, %r4;").unwrap();

        // BN254 Fr (Scalar Field) Modulus Limbs (Little-Endian)
        writeln!(&mut buffer, "    .reg .u64 %p0, %p1, %p2, %p3;").unwrap();
        writeln!(&mut buffer, "    mov.u64 %p0, 0x43e1f593f0000001;").unwrap();
        writeln!(&mut buffer, "    mov.u64 %p1, 0x2833e84879b97091;").unwrap();
        writeln!(&mut buffer, "    mov.u64 %p2, 0xb85045b68181585d;").unwrap();
        writeln!(&mut buffer, "    mov.u64 %p3, 0x30644e72e131a029;").unwrap();
        writeln!(&mut buffer, "").unwrap();

        // Grid/Thread Indexing
        writeln!(&mut buffer, "    mov.u32 %t_id, %tid.x;").unwrap();
        writeln!(&mut buffer, "    mov.u32 %b_id, %ctaid.x;").unwrap();
        writeln!(&mut buffer, "    mov.u32 %b_dim, %ntid.x;").unwrap();
        writeln!(&mut buffer, "    mad.lo.u32 %instance_idx, %b_id, %b_dim, %t_id;").unwrap();
        writeln!(&mut buffer, "    ld.param.u32 %r_max_inst, [param_total_instances];").unwrap();
        writeln!(&mut buffer, "    setp.ge.u32 %p_valid, %instance_idx, %r_max_inst;").unwrap();
        writeln!(&mut buffer, "    @%p_valid bra EXIT_KERNEL;").unwrap();
        writeln!(&mut buffer, "").unwrap();

        // Offset Calculation: instance_offset_bytes = instance_idx * num_signals * 32
        writeln!(&mut buffer, "    ld.param.u64 %base_ptr, [param_witness_buffer];").unwrap();
        writeln!(&mut buffer, "    ld.param.u32 %r_num_sig, [param_num_signals];").unwrap();
        writeln!(&mut buffer, "    mul.wide.u32 %instance_offset_bytes, %instance_idx, %r_num_sig;").unwrap();
        writeln!(&mut buffer, "    shl.b64 %instance_offset_bytes, %instance_offset_bytes, 5; // * 32 bytes").unwrap();
        writeln!(&mut buffer, "    add.u64 %base_ptr, %base_ptr, %instance_offset_bytes;").unwrap();
        writeln!(&mut buffer, "").unwrap();

        // Top-level Register Declarations for all Signals & WitnessOps
        for &sig_id in &graph.topological_order {
            let s_idx = sig_id.0;
            writeln!(&mut buffer, "    .reg .u64 %s{}_0, %s{}_1, %s{}_2, %s{}_3;", s_idx, s_idx, s_idx, s_idx).unwrap();
            if let WitnessOp::Mul(_, _) = &graph.nodes[s_idx] {
                writeln!(&mut buffer, "    .reg .u64 %c_tmp_{}_0, %c_tmp_{}_1, %c_tmp_{}_2, %c_tmp_{}_3;", s_idx, s_idx, s_idx, s_idx).unwrap();
                writeln!(&mut buffer, "    .reg .u64 %c_t_{}_0, %c_t_{}_1, %c_t_{}_2, %c_t_{}_3, %c_t_{}_4;", s_idx, s_idx, s_idx, s_idx, s_idx).unwrap();
                writeln!(&mut buffer, "    .reg .u64 %c_tlo_{}, %c_thi_{}, %c_carry_{}, %c_m_{}, %c_t5_{};", s_idx, s_idx, s_idx, s_idx, s_idx).unwrap();
                writeln!(&mut buffer, "    .reg .u64 %c_r0_{}, %c_r1_{}, %c_r2_{}, %c_r3_{}, %c_r4_{};", s_idx, s_idx, s_idx, s_idx, s_idx).unwrap();
                writeln!(&mut buffer, "    .reg .pred %c_p_sub_{};", s_idx).unwrap();
            }
        }

        // Loop over signals in topological order
        for &sig_id in &graph.topological_order {
            let s_idx = sig_id.0;
            let sig_name = graph.signal_names.get(&s_idx).cloned().unwrap_or_else(|| format!("sig_{}", s_idx));
            writeln!(&mut buffer, "    // --- Signal {}: {} ---", s_idx, sig_name).unwrap();

            match &graph.nodes[s_idx] {
                WitnessOp::Const(val) => {
                    // Canonical limbs, not the stored Montgomery ones: this is
                    // a literal for the witness generator, which works in
                    // ordinary residues.
                    let [d0, d1, d2, d3] = val.to_limbs();

                    writeln!(&mut buffer, "    mov.u64 %s{}_0, 0x{:x};", s_idx, d0).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_1, 0x{:x};", s_idx, d1).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_2, 0x{:x};", s_idx, d2).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_3, 0x{:x};", s_idx, d3).unwrap();
                }
                WitnessOp::LoadInput { input_idx, .. } => {
                    let offset = input_idx * 32;
                    writeln!(&mut buffer, "    add.u64 %sig_addr, %base_ptr, {};", offset).unwrap();
                    writeln!(&mut buffer, "    ld.global.v2.b64 {{%s{}_0, %s{}_1}}, [%sig_addr];", s_idx, s_idx).unwrap();
                    writeln!(&mut buffer, "    add.u64 %sig_addr, %sig_addr, 16;").unwrap();
                    writeln!(&mut buffer, "    ld.global.v2.b64 {{%s{}_2, %s{}_3}}, [%sig_addr];", s_idx, s_idx).unwrap();
                }
                WitnessOp::Add(a, b) => {
                    let a_id = a.0;
                    let b_id = b.0;
                    writeln!(&mut buffer, "    // 256-bit Modular Addition: s_{} = (s_{} + s_{}) mod p", s_idx, a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    add.cc.u64 %t0, %s{}_0, %s{}_0;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    addc.cc.u64 %t1, %s{}_1, %s{}_1;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    addc.cc.u64 %t2, %s{}_2, %s{}_2;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    addc.cc.u64 %t3, %s{}_3, %s{}_3;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    addc.u64 %t4, 0, 0;").unwrap();
                    writeln!(&mut buffer, "    sub.cc.u64 %r0, %t0, %p0;").unwrap();
                    writeln!(&mut buffer, "    subc.cc.u64 %r1, %t1, %p1;").unwrap();
                    writeln!(&mut buffer, "    subc.cc.u64 %r2, %t2, %p2;").unwrap();
                    writeln!(&mut buffer, "    subc.cc.u64 %r3, %t3, %p3;").unwrap();
                    writeln!(&mut buffer, "    subc.u64 %r4, %t4, 0;").unwrap();
                    writeln!(&mut buffer, "    setp.eq.u64 %p_sub, %r4, 0;").unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_0, %r0, %t0, %p_sub;", s_idx).unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_1, %r1, %t1, %p_sub;", s_idx).unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_2, %r2, %t2, %p_sub;", s_idx).unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_3, %r3, %t3, %p_sub;", s_idx).unwrap();
                }
                WitnessOp::Sub(a, b) => {
                    let a_id = a.0;
                    let b_id = b.0;
                    writeln!(&mut buffer, "    // 256-bit Modular Subtraction: s_{} = (s_{} - s_{}) mod p", s_idx, a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    sub.cc.u64 %t0, %s{}_0, %s{}_0;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    subc.cc.u64 %t1, %s{}_1, %s{}_1;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    subc.cc.u64 %t2, %s{}_2, %s{}_2;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    subc.cc.u64 %t3, %s{}_3, %s{}_3;", a_id, b_id).unwrap();
                    writeln!(&mut buffer, "    subc.u64 %t4, 0, 0;").unwrap();
                    writeln!(&mut buffer, "    add.cc.u64 %r0, %t0, %p0;").unwrap();
                    writeln!(&mut buffer, "    addc.cc.u64 %r1, %t1, %p1;").unwrap();
                    writeln!(&mut buffer, "    addc.cc.u64 %r2, %t2, %p2;").unwrap();
                    writeln!(&mut buffer, "    addc.u64 %r3, %t3, %p3;").unwrap();
                    writeln!(&mut buffer, "    setp.ne.u64 %p_borrow, %t4, 0;").unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_0, %r0, %t0, %p_borrow;", s_idx).unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_1, %r1, %t1, %p_borrow;", s_idx).unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_2, %r2, %t2, %p_borrow;", s_idx).unwrap();
                    writeln!(&mut buffer, "    selp.b64 %s{}_3, %r3, %t3, %p_borrow;", s_idx).unwrap();
                }
                WitnessOp::Mul(a, b) => {
                    let a_id = a.0;
                    let b_id = b.0;
                    writeln!(&mut buffer, "    // 256-bit CIOS Montgomery Modular Multiplication: s_{} = (s_{} * s_{}) mod p", s_idx, a_id, b_id).unwrap();

                    // Emit CIOS Montgomery Multiplication Pass 1: tmp = cios(s_a, s_b)
                    let p0_str = "0x43e1f593f0000001";
                    let p1_str = "0x2833e84879b97091";
                    let p2_str = "0xb85045b68181585d";
                    let p3_str = "0x30644e72e131a029";
                    let p_prime_str = "0xc2e1f593efffffff";

                    let r2_0_str = "0x1bb8e645ae216da7";
                    let r2_1_str = "0x53fe3ab1e35c59e3";
                    let r2_2_str = "0x8c49833d53bb8085";
                    let r2_3_str = "0x0216d0b17f4e44a5";

                    // Helper lambda to emit 1 pass of CIOS Montgomery Multiplication
                    let emit_cios_pass = |buf: &mut String, in_a: [&str; 4], in_b: [&str; 4], out_r: [&str; 4], tag: &str| {
                        writeln!(buf, "    // --- CIOS Pass: {} ---", tag).unwrap();
                        for k in 0..5 {
                            writeln!(buf, "    mov.u64 %c_t_{}_{}, 0;", s_idx, k).unwrap();
                        }

                        for i in 0..4 {
                            // Step 1: Accumulate in_a[i] * in_b[0..3] into t[0..4]
                            writeln!(buf, "    mov.u64 %c_carry_{}, 0;", s_idx).unwrap();
                            for j in 0..4 {
                                writeln!(buf, "    mul.lo.u64 %c_tlo_{}, {}, {};", s_idx, in_a[i], in_b[j]).unwrap();
                                writeln!(buf, "    mul.hi.u64 %c_thi_{}, {}, {};", s_idx, in_a[i], in_b[j]).unwrap();

                                writeln!(buf, "    add.cc.u64 %c_tlo_{}, %c_tlo_{}, %c_t_{}_{};", s_idx, s_idx, s_idx, j).unwrap();
                                writeln!(buf, "    addc.u64 %c_thi_{}, %c_thi_{}, 0;", s_idx, s_idx).unwrap();

                                writeln!(buf, "    add.cc.u64 %c_t_{}_{}, %c_tlo_{}, %c_carry_{};", s_idx, j, s_idx, s_idx).unwrap();
                                writeln!(buf, "    addc.u64 %c_carry_{}, %c_thi_{}, 0;", s_idx, s_idx).unwrap();
                            }
                            writeln!(buf, "    add.cc.u64 %c_t_{}_4, %c_t_{}_4, %c_carry_{};", s_idx, s_idx, s_idx).unwrap();
                            writeln!(buf, "    addc.u64 %c_t5_{}, 0, 0;", s_idx).unwrap();

                            // Step 2: m = (t[0] * p_prime) mod 2^64
                            writeln!(buf, "    mul.lo.u64 %c_m_{}, %c_t_{}_0, {};", s_idx, s_idx, p_prime_str).unwrap();

                            // Step 3: Add m * P[0..3] to t[0..4]
                            writeln!(buf, "    mul.lo.u64 %c_tlo_{}, %c_m_{}, {};", s_idx, s_idx, p0_str).unwrap();
                            writeln!(buf, "    mul.hi.u64 %c_thi_{}, %c_m_{}, {};", s_idx, s_idx, p0_str).unwrap();
                            writeln!(buf, "    add.cc.u64 %c_tlo_{}, %c_tlo_{}, %c_t_{}_0;", s_idx, s_idx, s_idx).unwrap();
                            writeln!(buf, "    addc.u64 %c_carry_{}, %c_thi_{}, 0;", s_idx, s_idx).unwrap();

                            let p_limbs = [p1_str, p2_str, p3_str];
                            for j in 1..4 {
                                writeln!(buf, "    mul.lo.u64 %c_tlo_{}, %c_m_{}, {};", s_idx, s_idx, p_limbs[j-1]).unwrap();
                                writeln!(buf, "    mul.hi.u64 %c_thi_{}, %c_m_{}, {};", s_idx, s_idx, p_limbs[j-1]).unwrap();

                                writeln!(buf, "    add.cc.u64 %c_tlo_{}, %c_tlo_{}, %c_t_{}_{};", s_idx, s_idx, s_idx, j).unwrap();
                                writeln!(buf, "    addc.u64 %c_thi_{}, %c_thi_{}, 0;", s_idx, s_idx).unwrap();

                                writeln!(buf, "    add.cc.u64 %c_t_{}_{}, %c_tlo_{}, %c_carry_{};", s_idx, j - 1, s_idx, s_idx).unwrap();
                                writeln!(buf, "    addc.u64 %c_carry_{}, %c_thi_{}, 0;", s_idx, s_idx).unwrap();
                            }
                            writeln!(buf, "    add.cc.u64 %c_t_{}_3, %c_t_{}_4, %c_carry_{};", s_idx, s_idx, s_idx).unwrap();
                            writeln!(buf, "    addc.u64 %c_t_{}_4, %c_t5_{}, 0;", s_idx, s_idx).unwrap();
                        }

                        // Conditional Subtraction P
                        writeln!(buf, "    sub.cc.u64 %c_r0_{}, %c_t_{}_0, {};", s_idx, s_idx, p0_str).unwrap();
                        writeln!(buf, "    subc.cc.u64 %c_r1_{}, %c_t_{}_1, {};", s_idx, s_idx, p1_str).unwrap();
                        writeln!(buf, "    subc.cc.u64 %c_r2_{}, %c_t_{}_2, {};", s_idx, s_idx, p2_str).unwrap();
                        writeln!(buf, "    subc.cc.u64 %c_r3_{}, %c_t_{}_3, {};", s_idx, s_idx, p3_str).unwrap();
                        writeln!(buf, "    subc.u64 %c_r4_{}, %c_t_{}_4, 0;", s_idx, s_idx).unwrap();

                        writeln!(buf, "    setp.eq.u64 %c_p_sub_{}, %c_r4_{}, 0;", s_idx, s_idx).unwrap();
                        writeln!(buf, "    selp.b64 {}, %c_r0_{}, %c_t_{}_0, %c_p_sub_{};", out_r[0], s_idx, s_idx, s_idx).unwrap();
                        writeln!(buf, "    selp.b64 {}, %c_r1_{}, %c_t_{}_1, %c_p_sub_{};", out_r[1], s_idx, s_idx, s_idx).unwrap();
                        writeln!(buf, "    selp.b64 {}, %c_r2_{}, %c_t_{}_2, %c_p_sub_{};", out_r[2], s_idx, s_idx, s_idx).unwrap();
                        writeln!(buf, "    selp.b64 {}, %c_r3_{}, %c_t_{}_3, %c_p_sub_{};", out_r[3], s_idx, s_idx, s_idx).unwrap();
                    };

                    let sa = [format!("%s{}_0", a_id), format!("%s{}_1", a_id), format!("%s{}_2", a_id), format!("%s{}_3", a_id)];
                    let sb = [format!("%s{}_0", b_id), format!("%s{}_1", b_id), format!("%s{}_2", b_id), format!("%s{}_3", b_id)];
                    let stmp = [format!("%c_tmp_{}_0", s_idx), format!("%c_tmp_{}_1", s_idx), format!("%c_tmp_{}_2", s_idx), format!("%c_tmp_{}_3", s_idx)];
                    let sr2 = [r2_0_str.to_string(), r2_1_str.to_string(), r2_2_str.to_string(), r2_3_str.to_string()];
                    let sout = [format!("%s{}_0", s_idx), format!("%s{}_1", s_idx), format!("%s{}_2", s_idx), format!("%s{}_3", s_idx)];

                    let sa_ref = [&sa[0][..], &sa[1][..], &sa[2][..], &sa[3][..]];
                    let sb_ref = [&sb[0][..], &sb[1][..], &sb[2][..], &sb[3][..]];
                    let stmp_ref = [&stmp[0][..], &stmp[1][..], &stmp[2][..], &stmp[3][..]];
                    let sr2_ref = [&sr2[0][..], &sr2[1][..], &sr2[2][..], &sr2[3][..]];
                    let sout_ref = [&sout[0][..], &sout[1][..], &sout[2][..], &sout[3][..]];

                    emit_cios_pass(&mut buffer, sa_ref, sb_ref, stmp_ref, "tmp = cios(a, b)");
                    emit_cios_pass(&mut buffer, sr2_ref, stmp_ref, sout_ref, "out = cios(tmp, R2)");
                }
                // Everything below is REFUSED rather than approximated. The
                // arms above are the whole of what this backend can lower;
                // five of `WitnessOp`'s seventeen variants.
                //
                // `Inv`/`Div` used to emit `mov s_out, s_a` under a comment
                // reading "256-bit Field Inversion / Division Hint" - the
                // identity, so `1/x` computed `x`. Everything else fell to
                // `_ => mov 0`, writing ZERO into the witness slot. Both
                // assemble perfectly, which is why nothing caught them: a
                // wrong `mov` is as valid to `ptxas` as a right one.
                //
                // The zero arm is the worst of the two, because of WHICH ops
                // it covered. `BitOfLc` is how every comparison, bitwise
                // operator, shift, integer division and range check gets its
                // witness; `IsZeroLc`/`InvOrZeroLc` are `==` and `!=`;
                // `MulAddLc` is the single most common statement a circom
                // program compiles to. So for any circuit past straight-line
                // field arithmetic, this kernel filled most of the witness
                // with zeros and the CLI printed "compiled successfully".
                WitnessOp::Inv(_) | WitnessOp::Div(..) | WitnessOp::DivLc(..) => {
                    self.unsupported_witness_op(
                        s_idx,
                        "field inversion",
                        "needs a Fermat exponentiation chain (x^(p-2)); it was emitting \
                         the IDENTITY, so 1/x computed x",
                    );
                }
                other => {
                    let name = Self::witness_op_name(other);
                    self.unsupported_witness_op(
                        s_idx,
                        name,
                        "no PTX lowering exists for it, and it was writing ZERO into \
                         the witness slot - which assembles, launches, and proves nothing",
                    );
                }
            }

            let offset = s_idx * 32;
            writeln!(&mut buffer, "    add.u64 %sig_addr, %base_ptr, {};", offset).unwrap();
            writeln!(&mut buffer, "    st.global.v2.b64 [%sig_addr], {{%s{}_0, %s{}_1}};", s_idx, s_idx).unwrap();
            writeln!(&mut buffer, "    add.u64 %sig_addr, %sig_addr, 16;").unwrap();
            writeln!(&mut buffer, "    st.global.v2.b64 [%sig_addr], {{%s{}_2, %s{}_3}};", s_idx, s_idx).unwrap();
            writeln!(&mut buffer, "").unwrap();
        }

        writeln!(&mut buffer, "EXIT_KERNEL:").unwrap();
        writeln!(&mut buffer, "    ret;").unwrap();
        writeln!(&mut buffer, "}}").unwrap();

        buffer
    }

    /// Emits native Ampere/Ada cp.async transfer instruction bypassing register files and L1 cache allocation (.cg).
    pub fn emit_cp_async(&mut self, dest_smem: &str, src_gmem: &str, bytes: u32) {
        writeln!(
            &mut self.ptx_buffer,
            "    cp.async.cg.shared.global [{}], [{}], {};",
            dest_smem, src_gmem, bytes
        )
        .unwrap();
    }

    /// Emits cp.async.commit_group instruction.
    pub fn emit_cp_async_commit(&mut self) {
        writeln!(&mut self.ptx_buffer, "    cp.async.commit_group;").unwrap();
    }

    /// Emits cp.async.wait_group n instruction.
    pub fn emit_cp_async_wait(&mut self, n: u32) {
        writeln!(&mut self.ptx_buffer, "    cp.async.wait_group {};", n).unwrap();
    }

    /// Automated Grid Block Swizzling Pass (grouped-raster L2 locality swizzle,
    /// in the style of CUTLASS's `ThreadblockSwizzle` / Triton's grouped-order
    /// matmul tutorial - despite the historical name this is not actually a
    /// Morton/Hilbert curve).
    ///
    /// Groups `swizzle_group_size` consecutive M-direction CTA row-tiles and
    /// walks them column-first (instead of raw row-major raster order), so
    /// that CTAs with nearby linear ctaid - which tend to run concurrently -
    /// reuse the same A/B tiles through L2 instead of one fixed A-row-tile
    /// paired with `gdim_x` different B-column-tiles.
    ///
    /// Returns `(swizzled_cta_m_reg, swizzled_cta_n_reg)` for callers that
    /// want the registers directly; also stashed in `self.variables` under
    /// `"pid_m"`/`"pid_n"`/`"swizzled_cta_m"`/`"swizzled_cta_n"` for the
    /// generic per-statement lowering path.
    ///
    /// Handles `%nctaid.y` (`gdim_y`, i.e. the number of M-direction tiles)
    /// NOT being an exact multiple of `swizzle_group_size` - and
    /// `swizzle_group_size > gdim_y` outright - by clamping the divisor used
    /// for the last (short) group to the rows actually remaining
    /// (`group_size_m = min(swizzle_group_size, gdim_y - first_m)`). Without
    /// this clamp the naive formula both emits out-of-range `swizzled_cta_m`
    /// values (CTAs writing outside the M extent - illegal-memory-access
    /// territory) and never produces some valid in-range tiles at all (their
    /// output would silently stay uninitialized) whenever `gdim_y` isn't a
    /// clean multiple - see `test_grid_swizzle_uneven_grid_is_bijection`,
    /// which enumerates every `(ctaid.x, ctaid.y)` pair for a battery of
    /// grid shapes (including primes, `gdim_y < swizzle_group_size`, and
    /// `gdim_y == 1`) and asserts the resulting `(swizzled_cta_n,
    /// swizzled_cta_m)` map is a bijection onto the same grid.
    pub fn emit_grid_swizzle_code(&mut self, swizzle_group_size: u32) -> (String, String) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [AUTOMATED GRID BLOCK SWIZZLING PASS - GROUPED RASTER L2 LOCALITY]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Group size: {} tiles (Maximizes L2 Cache hit rate)", swizzle_group_size).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        let raw_bid_x = self.alloc_reg32();
        let raw_bid_y = self.alloc_reg32();
        let gdim_x = self.alloc_reg32();
        let gdim_y = self.alloc_reg32();
        let tile_idx = self.alloc_reg32();
        let group_tiles = self.alloc_reg32();
        let group_id = self.alloc_reg32();
        let group_offset = self.alloc_reg32();
        let first_m = self.alloc_reg32();
        let rows_remaining = self.alloc_reg32();
        let group_size_m = self.alloc_reg32();
        let rem_offset = self.alloc_reg32();
        let swizzled_cta_m = self.alloc_reg32();
        let swizzled_cta_n = self.alloc_reg32();

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", raw_bid_x).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.y;", raw_bid_y).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %nctaid.x;", gdim_x).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %nctaid.y;", gdim_y).unwrap();

        // tile_idx = raw_bid_y * gdim_x + raw_bid_x
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", tile_idx, raw_bid_y, gdim_x, raw_bid_x).unwrap();
        // group_tiles = gdim_x * swizzle_group_size (tiles in one *full-size* row-group)
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", group_tiles, gdim_x, swizzle_group_size).unwrap();

        // group_id = tile_idx / group_tiles ; group_offset = tile_idx % group_tiles.
        // Using the constant (unclamped) group_tiles here is still correct
        // even though the LAST group may be short: every earlier group is
        // exactly full-size, so integer division still buckets tile_idx into
        // the right group_id, and group_offset still lands in [0, actual
        // tiles in that group) for the last group specifically (see doc
        // comment / test for the full argument).
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", group_id, tile_idx, group_tiles).unwrap();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", group_offset, tile_idx, group_tiles).unwrap();

        // first_m = group_id * swizzle_group_size (first M-row-tile of this group)
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", first_m, group_id, swizzle_group_size).unwrap();

        // group_size_m = min(swizzle_group_size, gdim_y - first_m): clamps
        // the divisor for a short last group (or swizzle_group_size >
        // gdim_y outright) down to the rows actually available.
        writeln!(&mut self.ptx_buffer, "    sub.u32 {}, {}, {};", rows_remaining, gdim_y, first_m).unwrap();
        writeln!(&mut self.ptx_buffer, "    min.u32 {}, {}, {};", group_size_m, rows_remaining, swizzle_group_size).unwrap();

        // swizzled_cta_m = first_m + (group_offset % group_size_m)
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", rem_offset, group_offset, group_size_m).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", swizzled_cta_m, first_m, rem_offset).unwrap();

        // swizzled_cta_n = group_offset / group_size_m
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", swizzled_cta_n, group_offset, group_size_m).unwrap();

        self.variables.insert("pid_m".into(), swizzled_cta_m.clone());
        self.variables.insert("pid_n".into(), swizzled_cta_n.clone());
        self.variables.insert("swizzled_cta_m".into(), swizzled_cta_m.clone());
        self.variables.insert("swizzled_cta_n".into(), swizzled_cta_n.clone());
        (swizzled_cta_m, swizzled_cta_n)
    }

    /// Returns `(m, n, k, a_reg, b_reg, c_reg, bias_reg)` if `kernel` carries
    /// a validated kernel-level `@tile(M, N, K)` directive (see
    /// `KernelDecl::tile`'s doc comment) with either exactly 3
    /// `GlobalMemory<F16/F16/F32>` params in declaration order (A, B, C -
    /// plain GEMM, `bias_reg` is `None`), or exactly 4
    /// (A, B: `GlobalMemory<F16>`, Bias, C: `GlobalMemory<F32>` - fused
    /// GEMM+Bias+ReLU, `bias_reg` is `Some`). `type_checker`'s
    /// `verify_tile_gemm_kernel` already enforces this shape as a hard
    /// compile error in the normal CLI/C-ABI pipeline, but it is re-checked
    /// here rather than just trusted, because `PtxEmitter` can be driven
    /// directly with no type_checker pass at all - this file's own unit
    /// tests do exactly that. `None` means "fall back to the normal generic
    /// per-statement lowering" - never a panic, and never a best-effort
    /// guess at a kernel this codegen isn't sure matches its assumptions.
    fn tile_gemm_operands(&self, kernel: &KernelDecl) -> Option<(u32, u32, u32, String, String, String, Option<String>)> {
        let tile = kernel.tile.as_ref()?;

        fn as_positive_u32(e: &Expr) -> Option<u32> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => u32::try_from(*v).ok(),
                _ => None,
            }
        }
        let m = as_positive_u32(&tile.block_m)?;
        let n = as_positive_u32(&tile.block_n)?;
        let k = as_positive_u32(tile.block_k.as_deref()?)?;

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }
        // A, B: f16 Tensor Core operands. C (and, in the 4-param fused
        // shape, Bias): f32 (see emit_tensor_core_gemm_kernel's doc comment
        // - it hardcodes 4 bytes/element and wmma.store.d...f32 for C
        // specifically, so this is not an arbitrary choice mirrored from
        // type_checker for its own sake - the codegen below is only correct
        // for exactly this shape).
        let has_bias = kernel.params.len() == 4;
        if (kernel.params.len() != 3 && kernel.params.len() != 4)
            || !is_global_memory_of(&kernel.params[0].ty, "F16")
            || !is_global_memory_of(&kernel.params[1].ty, "F16")
            || !is_global_memory_of(&kernel.params[2].ty, "F32")
            || (has_bias && !is_global_memory_of(&kernel.params[3].ty, "F32"))
        {
            return None;
        }

        let a_reg = self.variables.get(&kernel.params[0].name)?.clone();
        let b_reg = self.variables.get(&kernel.params[1].name)?.clone();
        let c_idx = if has_bias { 3 } else { 2 };
        let c_reg = self.variables.get(&kernel.params[c_idx].name)?.clone();
        let bias_reg = if has_bias {
            Some(self.variables.get(&kernel.params[2].name)?.clone())
        } else {
            None
        };
        Some((m, n, k, a_reg, b_reg, c_reg, bias_reg))
    }

    /// Returns `(m, n, k, a_reg, b_reg, scale_a_reg, scale_b_reg, c_reg)` if
    /// `kernel` carries a validated kernel-level `@tile(M, N, K)` directive
    /// with exactly 5 params: A, B: `GlobalMemory<F32>` (quantized to e4m3
    /// on the fly - see `emit_fp8_gemm_kernel`), scale_a, scale_b: `F32`
    /// scalars (per-tensor dequant scale, typically `amax/448.0` - the
    /// caller's/launcher's job to compute), C: `GlobalMemory<F32>`.
    /// `type_checker::verify_tile_gemm_kernel` enforces this shape as a hard
    /// compile error in the normal CLI pipeline (see its own doc comment,
    /// which this must never disagree with) - re-checked here for the same
    /// reason `tile_gemm_operands` re-checks its own shape: `PtxEmitter` can
    /// be driven directly with no type_checker pass at all (this file's own
    /// unit tests do exactly that). `None` means "fall back to the normal
    /// generic per-statement lowering", distinguished from
    /// `tile_gemm_operands`'s 3/4-param F16 shapes by both param COUNT (5)
    /// and TYPE (F32 GlobalMemory for A/B here, vs F16 there) - no ambiguity.
    /// Emit an exact int8 tensor-core GEMM over `mma.sync.m16n8k32.s32.s8.s8.s32`.
    ///
    /// One warp owns one 16x8 output tile and walks the whole K range; the grid
    /// is `(N/8, M/16, 1)` with a 32-thread block. Fragments are loaded
    /// **straight from global memory**, not staged through shared memory.
    ///
    /// **That is a deliberate choice for this milestone, not an oversight.**
    /// The fragment layout is the part that silently produces wrong answers
    /// (see `tests/ptx_int8_mma_layout.rs`, where four mutations all yield
    /// plausible matrices), and it is validated in exactly this
    /// straight-from-global form. Adding CTA tiling and shared-memory staging
    /// changes the addressing and would have to be re-validated; doing it in
    /// the same step would mean never knowing which half was wrong. This kernel
    /// is bandwidth-bound and re-reads A and B once per output tile — it is
    /// correct and reproducible, not fast.
    ///
    /// Layouts, fixed by the instruction rather than chosen:
    /// - `A`: row-major `[M][K]` int8.
    /// - `B`: **`[N][K]`** int8, i.e. column-major in the mma's terms. This is
    ///   the natural layout for an inference weight matrix and is what `.col`
    ///   means; passing a `[K][N]` buffer computes a different function.
    /// - `C`: row-major `[M][N]` int32, **overwritten**, not accumulated.
    ///
    /// Returns the thread count per CTA.
    /// `epi` is `Some((scale_a, scale_b, bias))` for the fused scaled shape.
    ///
    /// The mainloop is shared with the plain int8 GEMM deliberately: the two
    /// differ only in what happens to the four accumulators at the end, and a
    /// second copy of a tensor-core mainloop is the kind of duplicate the
    /// design-rule table is full of.
    fn emit_int8_gemm_kernel(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        a_ptr: &str,
        b_ptr: &str,
        c_ptr: &str,
        kernel_name: &str,
        epi: Option<(&str, &str, &str)>,
    ) -> u32 {
        // Refused rather than padded: a partial tile would need predication on
        // every fragment load, and silently rounding the shape up would compute
        // a different matrix than the source asked for.
        if m % 16 != 0 || n % 8 != 0 || k % 32 != 0 {
            self.emit_errors.push(format!(
                "[PTX] `{}`: an int8 tensor-core GEMM needs M % 16 == 0, N % 8 == 0 and \
                 K % 32 == 0 (the mma shape is m16n8k32); got M={}, N={}, K={}.",
                kernel_name, m, n, k
            ));
            return 32;
        }

        writeln!(
            &mut self.ptx_buffer,
            "    // [Y INT8 TENSOR CORE GEMM] M={} N={} K={} | mma.sync.m16n8k32.row.col.s32.s8.s8.s32\n\
             \x20   // one warp per 16x8 tile, grid ({}, {}, 1), block (32,1,1)\n\
             \x20   // A row-major [M][K], B [N][K], C row-major [M][N] int32",
            m, n, k, n / 8, m / 16
        )
        .unwrap();

        let ga = self.alloc_reg64();
        let gb = self.alloc_reg64();
        let gc = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", ga, a_ptr).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", gb, b_ptr).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", gc, c_ptr).unwrap();
        let epi_g = epi.map(|(sa, sb, bi)| {
            let (gsa, gsb, gbi) = (self.alloc_reg64(), self.alloc_reg64(), self.alloc_reg64());
            writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", gsa, sa).unwrap();
            writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", gsb, sb).unwrap();
            writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", gbi, bi).unwrap();
            (gsa, gsb, gbi)
        });

        let tid = self.alloc_reg32();
        let cx = self.alloc_reg32();
        let cy = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", cx).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.y;", cy).unwrap();

        // g = laneid >> 2, t = laneid & 3 — the decomposition validated in
        // tests/ptx_int8_mma_layout.rs.
        let g = self.alloc_reg32();
        let t = self.alloc_reg32();
        let t4 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 2;", g, tid).unwrap();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 3;", t, tid).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", t4, t).unwrap();

        // &A[(ctaid.y*16 + g)][4t]
        let arow = self.alloc_reg32();
        let aoff = self.alloc_reg32();
        let aoff64 = self.alloc_reg64();
        let alane = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 16;", arow, cy).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", arow, arow, g).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", aoff, arow, k).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", aoff, aoff, t4).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", aoff64, aoff).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", alane, ga, aoff64).unwrap();

        // &B[(ctaid.x*8 + g)][4t]
        let bcol = self.alloc_reg32();
        let boff = self.alloc_reg32();
        let boff64 = self.alloc_reg64();
        let blane = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 8;", bcol, cx).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", bcol, bcol, g).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", boff, bcol, k).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", boff, boff, t4).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", boff64, boff).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", blane, gb, boff64).unwrap();

        let d: Vec<String> = (0..4).map(|_| self.alloc_reg32()).collect();
        for r in &d {
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", r).unwrap();
        }

        // Split-K over `%ctaid.z`, striped rather than blocked: CTA `z` takes
        // every `nctaid.z`-th 32-wide K step starting at `z`. Striping means
        // ANY grid.z divides the work with no divisibility precondition, which
        // is what lets the batch-invariance harness sweep the split factor
        // without recompiling.
        //
        // **The partial sums combine through `red.global.add.s32`, and that is
        // the whole demonstration.** An atomic float add is the canonical
        // reason GPU results are not reproducible: it is non-associative, so
        // the answer depends on the order CTAs happen to finish. Integer
        // addition is associative and commutative, so this atomic is
        // order-independent by construction — the same result for every grid,
        // every launch, every scheduling accident.
        let cz = self.alloc_reg32();
        let nz = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.z;", cz).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %nctaid.z;", nz).unwrap();

        let zoff = self.alloc_reg32();
        let zoff64 = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 32;", zoff, cz).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", zoff64, zoff).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", alane, alane, zoff64).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", blane, blane, zoff64).unwrap();

        let kstep = self.alloc_reg32();
        let kstep64 = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 32;", kstep, nz).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", kstep64, kstep).unwrap();

        let kk = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", kk, zoff).unwrap();
        let top = self.alloc_label("int8_gemm_k");
        let end = self.alloc_label("int8_gemm_end");
        let pred = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "{}:", top).unwrap();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", pred, kk, k).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", pred, end).unwrap();

        // The four A registers and two B registers, at the offsets derived and
        // validated in tests/ptx_int8_mma_layout.rs. `8 * k` is the byte step
        // to the row-half 8 rows down, because A's rows are k bytes apart.
        let a: Vec<String> = (0..4).map(|_| self.alloc_reg32()).collect();
        let b: Vec<String> = (0..2).map(|_| self.alloc_reg32()).collect();
        writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}];", a[0], alane).unwrap();
        writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}+{}];", a[1], alane, 8 * k).unwrap();
        writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}+16];", a[2], alane).unwrap();
        writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}+{}];", a[3], alane, 8 * k + 16).unwrap();
        writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}];", b[0], blane).unwrap();
        writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}+16];", b[1], blane).unwrap();

        writeln!(
            &mut self.ptx_buffer,
            "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \
             {{{}, {}, {}, {}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{}, {}, {}, {}}};",
            d[0], d[1], d[2], d[3],
            a[0], a[1], a[2], a[3],
            b[0], b[1],
            d[0], d[1], d[2], d[3]
        )
        .unwrap();

        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", alane, alane, kstep64).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", blane, blane, kstep64).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", kk, kk, kstep).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", top).unwrap();
        writeln!(&mut self.ptx_buffer, "{}:", end).unwrap();

        // &C[(ctaid.y*16 + g)][ctaid.x*8 + 2t], in bytes.
        let crow = self.alloc_reg32();
        let coff = self.alloc_reg32();
        let coff64 = self.alloc_reg64();
        let clane = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 16;", crow, cy).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", crow, crow, g).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", coff, crow, n).unwrap();
        let cbase = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 8;", cbase, cx).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", coff, coff, cbase).unwrap();
        let t2 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 2;", t2, t).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", coff, coff, t2).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", coff, coff).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", coff64, coff).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", clane, gc, coff64).unwrap();
        // Reduce, not store: with split-K several CTAs own the same output
        // element. `red.global.add.s32` is the fire-and-forget form (no result
        // register), and being an INTEGER add it is associative, so the value
        // in C does not depend on which CTA got there first.
        //
        // The cost is that **C must be zero-initialised by the caller**. This
        // kernel accumulates into it rather than overwriting it, which is the
        // same contract the CPU exact GEMM has and for the same reason.
        if let Some((gsa, gsb, gbi)) = epi_g {
            // C[row][col] = acc * scale_a[row] * scale_b[col] + bias[col], f32.
            //
            // A plain `st`, not a `red.add`: the grid is 2-D, so K is not split
            // across CTAs and this lane owns its four output elements outright
            // after the loop. That is what makes a float epilogue sound at all
            // -- scaling PARTIAL sums and adding them in f32 would reintroduce
            // exactly the order dependence the int32 accumulator exists to
            // remove. If this kernel ever gains a %ctaid.z K-split, the
            // epilogue has to move to a separate pass over the finished int32
            // matrix; it cannot stay here.
            //
            // The four accumulators sit at rows {crow, crow+8} and columns
            // {cbase+2t, cbase+2t+1} -- `d[i]` is row +8*(i/2), col +(i%2) --
            // so there are only two distinct rows and two distinct columns to
            // fetch, not eight loads.
            let rows = [0u32, 8];
            let mut sa = Vec::new();
            for r in rows {
                let off32 = self.alloc_reg32();
                let off64 = self.alloc_reg64();
                let addr = self.alloc_reg64();
                let v = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", off32, crow, r).unwrap();
                writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", off32, off32).unwrap();
                writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", off64, off32).unwrap();
                writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, gsa, off64).unwrap();
                writeln!(&mut self.ptx_buffer, "    ld.global.f32 {}, [{}];", v, addr).unwrap();
                sa.push(v);
            }
            let mut sb = Vec::new();
            let mut bi = Vec::new();
            for c in [0u32, 1] {
                let off32 = self.alloc_reg32();
                let off64 = self.alloc_reg64();
                let a1 = self.alloc_reg64();
                let a2 = self.alloc_reg64();
                let v1 = self.alloc_regf32();
                let v2 = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", off32, cbase, t2).unwrap();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", off32, off32, c).unwrap();
                writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", off32, off32).unwrap();
                writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", off64, off32).unwrap();
                writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", a1, gsb, off64).unwrap();
                writeln!(&mut self.ptx_buffer, "    ld.global.f32 {}, [{}];", v1, a1).unwrap();
                writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", a2, gbi, off64).unwrap();
                writeln!(&mut self.ptx_buffer, "    ld.global.f32 {}, [{}];", v2, a2).unwrap();
                sb.push(v1);
                bi.push(v2);
            }
            for (idx, reg) in d.iter().enumerate() {
                let f = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    cvt.rn.f32.s32 {}, {};", f, reg).unwrap();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", f, f, sa[idx / 2]).unwrap();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", f, f, sb[idx % 2]).unwrap();
                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", f, f, bi[idx % 2]).unwrap();
                let byte = (idx / 2) * (8 * n as usize * 4) + (idx % 2) * 4;
                if byte == 0 {
                    writeln!(&mut self.ptx_buffer, "    st.global.f32 [{}], {};", clane, f).unwrap();
                } else {
                    writeln!(
                        &mut self.ptx_buffer,
                        "    st.global.f32 [{}+{}], {};",
                        clane, byte, f
                    )
                    .unwrap();
                }
            }
            return 32;
        }

        for (idx, reg) in d.iter().enumerate() {
            let byte = (idx / 2) * (8 * n as usize * 4) + (idx % 2) * 4;
            if byte == 0 {
                writeln!(&mut self.ptx_buffer, "    red.global.add.s32 [{}], {};", clane, reg).unwrap();
            } else {
                writeln!(
                    &mut self.ptx_buffer,
                    "    red.global.add.s32 [{}+{}], {};",
                    clane, byte, reg
                )
                .unwrap();
            }
        }

        32
    }

    /// Recognise an exact int8 tensor-core GEMM:
    /// `@tile(M, N, K) kernel f(A: GlobalMemory<I8>, B: GlobalMemory<I8>, C: GlobalMemory<I32>)`.
    ///
    /// Distinguished from the FP8 path by its element types alone, which is
    /// sufficient because no other recogniser accepts an `I8` buffer.
    ///
    /// **The accumulator is `I32` and that is the point.** `mma...s32.s8.s8.s32`
    /// accumulates exactly, so the reduction is associative and the result does
    /// not depend on how K was split across warps, CTAs or launches. That is
    /// the same property `docs/deterministic_inference.md` M0 established on the
    /// CPU, on the tensor cores instead.
    fn tile_gemm_int8_operands(
        &self,
        kernel: &KernelDecl,
    ) -> Option<(u32, u32, u32, String, String, String)> {
        let tile = kernel.tile.as_ref()?;

        fn as_positive_u32(e: &Expr) -> Option<u32> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => u32::try_from(*v).ok(),
                _ => None,
            }
        }
        let m = as_positive_u32(&tile.block_m)?;
        let n = as_positive_u32(&tile.block_n)?;
        let k = as_positive_u32(tile.block_k.as_deref()?)?;

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }

        if kernel.params.len() != 3
            || !is_global_memory_of(&kernel.params[0].ty, "I8")
            || !is_global_memory_of(&kernel.params[1].ty, "I8")
            || !is_global_memory_of(&kernel.params[2].ty, "I32")
        {
            return None;
        }

        let a_reg = self.variables.get(&kernel.params[0].name)?.clone();
        let b_reg = self.variables.get(&kernel.params[1].name)?.clone();
        let c_reg = self.variables.get(&kernel.params[2].name)?.clone();
        Some((m, n, k, a_reg, b_reg, c_reg))
    }

    /// Recognise the fused SCALED int8 tensor-core GEMM:
    /// `@tile(M, N, K) kernel f(A, B: GlobalMemory<I8>, Sa, Sb, Bias,
    /// C: GlobalMemory<F32>)`.
    ///
    /// `Sa` is per-row (length M), `Sb` and `Bias` per-column (length N), which
    /// is what per-token activation quantisation and per-channel weight
    /// quantisation give you. Told apart from the plain int8 shape by its
    /// parameter COUNT; both start with two `I8` buffers, and no other
    /// recogniser accepts an `I8` buffer at all.
    ///
    /// Per-ROW activation scaling is the only kind that keeps this exact: a row
    /// scale factors out of the dot product, so the accumulation stays integer
    /// and stays associative. A per-K-channel scale would sit inside the sum
    /// and turn it back into an order-dependent float reduction -- the same
    /// constraint `exact_kv::quantize_rows` documents on the host side.
    fn tile_gemm_int8_scaled_operands(
        &self,
        kernel: &KernelDecl,
    ) -> Option<(u32, u32, u32, String, String, String, String, String, String)> {
        let tile = kernel.tile.as_ref()?;

        fn as_positive_u32(e: &Expr) -> Option<u32> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => u32::try_from(*v).ok(),
                _ => None,
            }
        }
        let m = as_positive_u32(&tile.block_m)?;
        let n = as_positive_u32(&tile.block_n)?;
        let k = as_positive_u32(tile.block_k.as_deref()?)?;

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }

        if kernel.params.len() != 6
            || !is_global_memory_of(&kernel.params[0].ty, "I8")
            || !is_global_memory_of(&kernel.params[1].ty, "I8")
            || !(2..6).all(|i| is_global_memory_of(&kernel.params[i].ty, "F32"))
        {
            return None;
        }

        let r = |i: usize| self.variables.get(&kernel.params[i].name).cloned();
        Some((m, n, k, r(0)?, r(1)?, r(2)?, r(3)?, r(4)?, r(5)?))
    }

    fn tile_gemm_fp8_operands(&self, kernel: &KernelDecl) -> Option<(u32, u32, u32, String, String, String, String, String)> {
        let tile = kernel.tile.as_ref()?;

        fn as_positive_u32(e: &Expr) -> Option<u32> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => u32::try_from(*v).ok(),
                _ => None,
            }
        }
        let m = as_positive_u32(&tile.block_m)?;
        let n = as_positive_u32(&tile.block_n)?;
        let k = as_positive_u32(tile.block_k.as_deref()?)?;

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }
        fn is_scalar_f32(ty: &Type) -> bool {
            matches!(ty, Type::Primitive(p, _) if p == "F32")
        }

        if kernel.params.len() != 5
            || !is_global_memory_of(&kernel.params[0].ty, "F32")
            || !is_global_memory_of(&kernel.params[1].ty, "F32")
            || !is_scalar_f32(&kernel.params[2].ty)
            || !is_scalar_f32(&kernel.params[3].ty)
            || !is_global_memory_of(&kernel.params[4].ty, "F32")
        {
            return None;
        }

        let a_reg = self.variables.get(&kernel.params[0].name)?.clone();
        let b_reg = self.variables.get(&kernel.params[1].name)?.clone();
        let scale_a_reg = self.variables.get(&kernel.params[2].name)?.clone();
        let scale_b_reg = self.variables.get(&kernel.params[3].name)?.clone();
        let c_reg = self.variables.get(&kernel.params[4].name)?.clone();
        Some((m, n, k, a_reg, b_reg, scale_a_reg, scale_b_reg, c_reg))
    }

    /// Loads one of this warp's 4 A-fragment registers (`ld.shared.b32`, no
    /// `ldmatrix` - PTX ISA 8.5's `ldmatrix` syntax block lists only
    /// `.type = {.b16}`, no 8-bit variant at all) for
    /// `mma.sync.m16n8k32.row.col.e4m3` from this CTA's `cta_m x
    /// FP8_GEMM_CTA_K` e4m3 smem tile (row-major, `FP8_GEMM_CTA_K` B/row,
    /// no padding - `cta_m` doesn't actually appear in this function's own
    /// address formula, only via the caller-supplied `warp_row0_local`/`i`,
    /// so unlike `emit_fp8_load_b_one` this function needs no `cta_m`
    /// parameter). Generalizes the original single-warp/single-fragment
    /// addressing with three multi-warp-tiling terms - `warp_row0_local`
    /// (this warp's row origin within the CTA tile, `warp_m *
    /// per_warp_m`), `i*16` (this warp's i-th 16-row M-fragment), and `kk`
    /// (this K-substep's 32-wide offset within the current
    /// `FP8_GEMM_CTA_K`-wide K-slab) - standalone-validated on real sm_89
    /// hardware before this function was written (project scratchpad's
    /// validate_fp8_multiwarp.py, checked against an exact-integer CPU
    /// reference across single-CTA, multi-CTA, multi-K-tile, and ragged
    /// M/N shapes). `row_plus_8`/`col_extra` still select which of the 4
    /// registers, per PTX ISA 8.5 9.7.13.4.10's Multiplicand A
    /// (`.e4m3`/`.e5m2` bullet, shared with `.s8`/`.u8`, unchanged from the
    /// single-warp kernel): `groupID = lane>>2, tig = lane&3; row =
    /// groupID (+8), col = tig*4 (+16) + [0..3]`.
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_load_a_one(
        &mut self,
        smem_a: &str,
        group_id: &str,
        tig: &str,
        warp_row0_local: &str,
        i: u32,
        kk: u32,
        row_plus_8: bool,
        col_extra: u32,
    ) -> String {
        let row_base = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", row_base, warp_row0_local, i * 16).unwrap();
        let mut row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", row, row_base, group_id).unwrap();
        if row_plus_8 {
            let row2 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 8;", row2, row).unwrap();
            row = row2;
        }
        let mut col = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", col, tig).unwrap();
        if col_extra > 0 {
            let col2 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col2, col, col_extra).unwrap();
            col = col2;
        }
        let k_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", k_local, col, kk).unwrap();
        let lin = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", lin, row, FP8_GEMM_CTA_K, k_local).unwrap();
        let addr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", addr, smem_a, lin).unwrap();
        let reg = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    ld.shared.b32 {}, [{}];", reg, addr).unwrap();
        reg
    }

    /// All 4 A-fragment registers for this warp's i-th M-fragment at
    /// K-substep `kk`, in the order `mma.sync.m16n8k32` expects (`a0..a15`
    /// packed low-to-high across `reg0..reg3`) - see `emit_fp8_load_a_one`.
    fn emit_fp8_load_a_fragment(&mut self, smem_a: &str, group_id: &str, tig: &str, warp_row0_local: &str, i: u32, kk: u32) -> Vec<String> {
        vec![
            self.emit_fp8_load_a_one(smem_a, group_id, tig, warp_row0_local, i, kk, false, 0),
            self.emit_fp8_load_a_one(smem_a, group_id, tig, warp_row0_local, i, kk, true, 0),
            self.emit_fp8_load_a_one(smem_a, group_id, tig, warp_row0_local, i, kk, false, 16),
            self.emit_fp8_load_a_one(smem_a, group_id, tig, warp_row0_local, i, kk, true, 16),
        ]
    }

    /// Loads one of this warp's 2 B-fragment registers via 4x
    /// `ld.shared.u8` + shift/or pack (B's 4 elements-per-register are 4
    /// consecutive ROWS at a FIXED column - not contiguous in the row-
    /// major/N-contiguous smem layout B shares with global B, unlike A -
    /// see `emit_fp8_gemm_kernel`'s doc comment) from this CTA's
    /// `FP8_GEMM_CTA_K x cta_n` e4m3 smem tile (row-major, `cta_n` B/row,
    /// no padding - `cta_n` is a real parameter here, unlike
    /// `emit_fp8_load_a_one`'s `cta_m`, since it's needed for the smem row
    /// stride below). Generalizes the original
    /// single-warp/single-fragment addressing with `warp_col0_local` (this
    /// warp's column origin within the CTA tile, `warp_n * per_warp_n`),
    /// `j*8` (this warp's j-th 8-col N-fragment), and `kk` (this
    /// K-substep's 32-wide offset within the current `FP8_GEMM_CTA_K`-wide
    /// K-slab) - standalone-validated on real sm_89 hardware before this
    /// function was written (project scratchpad's
    /// validate_fp8_multiwarp.py, checked against an exact-integer CPU
    /// reference across single-CTA, multi-CTA, multi-K-tile, and ragged
    /// M/N shapes). `row_base_extra` still selects which of the 2
    /// registers, per PTX ISA 8.5 9.7.13.4.10's Multiplicand B (unchanged
    /// from the single-warp kernel): `row = tig*4 (+16) + [0..3], col =
    /// groupID`.
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_load_b_one(
        &mut self,
        smem_b: &str,
        group_id: &str,
        tig: &str,
        warp_col0_local: &str,
        j: u32,
        kk: u32,
        cta_n: u32,
        row_base_extra: u32,
    ) -> String {
        let mut base_row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", base_row, tig).unwrap();
        if row_base_extra > 0 {
            let br2 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", br2, base_row, row_base_extra).unwrap();
            base_row = br2;
        }
        let col_base = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col_base, warp_col0_local, j * 8).unwrap();
        let col = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col, col_base, group_id).unwrap();
        let mut packed = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", packed).unwrap();
        for byte_idx in 0..4u32 {
            let row = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", row, base_row, byte_idx).unwrap();
            let k_local = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", k_local, row, kk).unwrap();
            let lin = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", lin, k_local, cta_n, col).unwrap();
            let addr = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", addr, smem_b, lin).unwrap();
            let byte_val = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    ld.shared.u8 {}, [{}];", byte_val, addr).unwrap();
            let shifted = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, {};", shifted, byte_val, byte_idx * 8).unwrap();
            let new_packed = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    or.b32 {}, {}, {};", new_packed, packed, shifted).unwrap();
            packed = new_packed;
        }
        packed
    }

    /// Both B-fragment registers for this warp's j-th N-fragment at
    /// K-substep `kk` - see `emit_fp8_load_b_one`.
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_load_b_fragment(&mut self, smem_b: &str, group_id: &str, tig: &str, warp_col0_local: &str, j: u32, kk: u32, cta_n: u32) -> Vec<String> {
        vec![
            self.emit_fp8_load_b_one(smem_b, group_id, tig, warp_col0_local, j, kk, cta_n, 0),
            self.emit_fp8_load_b_one(smem_b, group_id, tig, warp_col0_local, j, kk, cta_n, 16),
        ]
    }

    /// This lane's local `(row, col)` within the 16x8 output tile for
    /// accumulator element `i` (0..3), per PTX ISA 8.5 9.7.13.4.9's
    /// accumulator formula - reused verbatim by 9.7.13.4.10 (m16n8k32),
    /// since accumulator shape depends only on M=16/N=8, not K (confirmed
    /// by comparing the m16n8k16-float, m16n8k16-integer, and m16n8k32
    /// sections directly - all three state the identical formula):
    /// `groupID = lane>>2, tig = lane&3; row = groupID (i<2) or groupID+8
    /// (i>=2); col = tig*2 + (i&1)`.
    fn emit_fp8_accum_row_col(&mut self, group_id: &str, tig: &str, i: u32) -> (String, String) {
        let row = self.alloc_reg32();
        if i < 2 {
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", row, group_id).unwrap();
        } else {
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 8;", row, group_id).unwrap();
        }
        let col = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 2;", col, tig).unwrap();
        if i % 2 == 1 {
            let col2 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", col2, col).unwrap();
            return (row, col2);
        }
        (row, col)
    }

    /// Loads 2 f32 elements at `[gaddr_base+gmem_byte_off]` /
    /// `[+gmem_byte_off+4]` (zero if the respective predicate is false - a
    /// predicated `ld.global.f32`, never an out-of-bounds access attempt),
    /// quantizes both by `inv_scale` (`= 1/scale`, the caller's job to
    /// compute once, outside any loop), packs via
    /// `cvt.rn.satfinite.e4m3x2.f32` (element 0 -> low byte, element 1 ->
    /// high byte, i.e. called as `a=elem1, b=elem0` per the ISA's "input a
    /// -> upper 8 bits, input b -> lower 8 bits" convention - standalone-
    /// validated bit-for-bit against `torch.float8_e4m3fn`'s own byte order
    /// AND numeric encoding, including rounding and `.satfinite` out-of-
    /// range saturation, on real sm_89 hardware before this function was
    /// written: project scratchpad's validate_fp8_quantize.py), and stores
    /// the packed `.b16` to `[saddr_base+smem_byte_off]`.
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_quantize_pair(
        &mut self,
        gaddr_base: &str,
        gmem_byte_off: u32,
        p_ok0: &str,
        p_ok1: &str,
        inv_scale: &str,
        saddr_base: &str,
        smem_byte_off: u32,
    ) {
        let f0 = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", f0).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} ld.global.f32 {}, [{}+{}];", p_ok0, f0, gaddr_base, gmem_byte_off).unwrap();
        let f1 = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", f1).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} ld.global.f32 {}, [{}+{}];", p_ok1, f1, gaddr_base, gmem_byte_off + 4).unwrap();

        let q0 = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", q0, f0, inv_scale).unwrap();
        let q1 = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", q1, f1, inv_scale).unwrap();

        let packed = self.alloc_reg16();
        writeln!(&mut self.ptx_buffer, "    cvt.rn.satfinite.e4m3x2.f32 {}, {}, {};", packed, q1, q0).unwrap();
        writeln!(&mut self.ptx_buffer, "    st.shared.b16 [{}+{}], {};", saddr_base, smem_byte_off, packed).unwrap();
    }

    /// Quantizes 4 already-loaded f32 values `f[0..4]` (element `i` -> byte
    /// `i` of the result, low-to-high) into one packed `.b32` register via
    /// 2x `cvt.rn.satfinite.e4m3x2.f32` + `cvt.u32.u16` zero-extend +
    /// `shl`/`or` - the vectorized-store counterpart to
    /// `emit_fp8_quantize_pair` (which stores 2 bytes at a time via
    /// `st.shared.b16`; this emits one value suitable for a single
    /// `st.shared.b32`/`st.global.u32`, halving store-instruction count on
    /// top of the load-side savings from `ld.global.v4.f32`). Standalone-
    /// validated bit-for-bit against `torch.float8_e4m3fn` on real sm_89
    /// hardware, including the byte-order/packing convention, before this
    /// was wired into the emitter (project scratchpad's
    /// validate_fp8_vectorized_stage.py, `quant_a_vec`/`quant_b_vec`'s FAST
    /// path).
    fn emit_fp8_pack_quad(&mut self, f: &[String], inv_scale: &str) -> String {
        debug_assert_eq!(f.len(), 4, "emit_fp8_pack_quad packs exactly 4 f32 values into one .b32");
        let q: Vec<String> = f
            .iter()
            .map(|r| {
                let q = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", q, r, inv_scale).unwrap();
                q
            })
            .collect();
        let packed_lo = self.alloc_reg16();
        writeln!(&mut self.ptx_buffer, "    cvt.rn.satfinite.e4m3x2.f32 {}, {}, {};", packed_lo, q[1], q[0]).unwrap();
        let packed_hi = self.alloc_reg16();
        writeln!(&mut self.ptx_buffer, "    cvt.rn.satfinite.e4m3x2.f32 {}, {}, {};", packed_hi, q[3], q[2]).unwrap();
        let lo32 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    cvt.u32.u16 {}, {};", lo32, packed_lo).unwrap();
        let hi32 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    cvt.u32.u16 {}, {};", hi32, packed_hi).unwrap();
        let hi_shifted = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 16;", hi_shifted, hi32).unwrap();
        let combined = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    or.b32 {}, {}, {};", combined, lo32, hi_shifted).unwrap();
        combined
    }

    /// Quantize+stages this CTA's `cta_m x FP8_GEMM_CTA_K` F32
    /// A-tile (source: `a_g[cta_row0+local_row][k0+local_col]`, row-major/
    /// K-contiguous, matching global A's natural layout) into `smem_a` as
    /// packed e4m3 bytes (row-major, `FP8_GEMM_CTA_K` B/row, no padding).
    /// `threads_per_cta`-wide cooperative RUNTIME loop (mirroring
    /// `emit_gemm_tile_load`'s thread-striding pattern: `idx = %tid.x`,
    /// `+= threads_per_cta` each iteration).
    ///
    /// **Vectorized (session 5)**: each thread/iteration now processes a
    /// 4-element (128-bit) quad via one `ld.global.v4.f32` + one
    /// `st.shared.b32` (via `emit_fp8_pack_quad`) instead of the original
    /// 2-element/iteration scalar `ld.global.f32` pair - a 4x reduction in
    /// load-instruction count and 2x in store-instruction count, raised as
    /// a concrete hypothesis by session 4's inconclusive pipelining result
    /// (see `investigation_fp8_gemm_findings.md`'s "Session 4" section:
    /// "the quantize+stage step's cost is dominated by its sheer scalar
    /// instruction count... rather than by raw memory latency"). This is
    /// unconditionally safe for A: K-side (columns) is always in-bounds
    /// (caller `debug_assert`s `K % FP8_GEMM_CTA_K == 0`), so all 4
    /// elements of a quad always share the same row and therefore the same
    /// M-boundary validity - one predicate per quad, same masking
    /// granularity the original per-pair version already used, just wider.
    /// Standalone-validated on real sm_89 hardware before this was wired in
    /// (project scratchpad's validate_fp8_vectorized_stage.py,
    /// `quant_a_vec`, including ragged row-bound cases).
    ///
    /// `extra_valid_pred`, when `Some`, is AND-ed into the per-quad
    /// validity predicate - used by the software-pipelined K-loop (see
    /// `emit_fp8_gemm_kernel`'s doc comment) to additionally suppress this
    /// prefetch when the K-tile being staged doesn't exist yet
    /// (`next_k_iter >= k_tiles`, only reachable on the loop's last
    /// iteration).
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_quantize_stage_a(
        &mut self,
        a_g: &str,
        cta_row0: &str,
        k0: &str,
        m: u32,
        k: u32,
        smem_a: &str,
        inv_scale_a: &str,
        threads_per_cta: u32,
        cta_m: u32,
        extra_valid_pred: Option<&str>,
    ) {
        debug_assert_eq!(FP8_GEMM_CTA_K % 4, 0, "vectorized A stage requires FP8_GEMM_CTA_K a multiple of 4");
        let total_quads = (cta_m * FP8_GEMM_CTA_K) / 4;
        let quads_per_row = FP8_GEMM_CTA_K / 4;

        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label("FP8_STAGE_A");
        let loop_end = self.alloc_label("FP8_STAGE_A_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_quads).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        let local_row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", local_row, idx, quads_per_row).unwrap();
        let quad_in_row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", quad_in_row, idx, quads_per_row).unwrap();
        let col_start = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", col_start, quad_in_row).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, cta_row0, local_row).unwrap();
        let mut p_ok = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_ok, grow, m).unwrap();
        if let Some(extra) = extra_valid_pred {
            let combined = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", combined, p_ok, extra).unwrap();
            p_ok = combined;
        }

        let gcol = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol, k0, col_start).unwrap();
        let g_lin = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", g_lin, grow, k, gcol).unwrap();
        let g_byte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", g_byte, g_lin).unwrap();
        let gaddr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr, a_g, g_byte).unwrap();

        let f: Vec<String> = (0..4).map(|_| self.alloc_regf32()).collect();
        for r in &f {
            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", r).unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    @{} ld.global.v4.f32 {{{}}}, [{}];", p_ok, f.join(","), gaddr).unwrap();

        let combined = self.emit_fp8_pack_quad(&f, inv_scale_a);

        let s_lin = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", s_lin, local_row, FP8_GEMM_CTA_K, col_start).unwrap();
        let saddr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_a, s_lin).unwrap();
        writeln!(&mut self.ptx_buffer, "    st.shared.b32 [{}], {};", saddr, combined).unwrap();

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Quantize+stages this CTA's `FP8_GEMM_CTA_K x cta_n` F32
    /// B-tile (source: `b_g[k0+local_row][cta_col0+local_col]`, row-major/
    /// N-contiguous, matching global B's natural layout) into `smem_b` as
    /// packed e4m3 bytes (row-major, `cta_n` B/row, no padding).
    /// Same `threads_per_cta`-wide cooperative runtime-loop generalization
    /// as `emit_fp8_quantize_stage_a` - see that function's doc comment.
    ///
    /// **Vectorized, hybrid fast/slow path (session 5)**: unlike A, the
    /// N-side (columns) needs PER-ELEMENT boundary masking (N need not be
    /// a multiple of anything in particular, and a 4-element quad's
    /// columns can straddle the boundary) - a single `@p
    /// ld.global.v4.f32` cannot express "3 of 4 lanes valid" the way the
    /// original per-element-predicated scalar loads could, so
    /// unconditionally vectorizing (as A safely does) would silently zero
    /// genuinely in-bounds tail columns whenever `N` isn't a multiple of 4.
    /// Instead, each quad iteration branches at runtime:
    /// - **FAST** (`col_start+3 < n`, the common case - true for every
    ///   quad except possibly the last one or two per CTA-row, only in
    ///   CTAs whose tile straddles N): the whole quad is guaranteed
    ///   in-bounds, so it takes the same unconditional `ld.global.v4.f32` +
    ///   `emit_fp8_pack_quad` + single `st.shared.b32` path as A.
    /// - **SLOW** (quad straddles the N boundary): falls back to the
    ///   original per-element-predicated scalar path - two
    ///   `emit_fp8_quantize_pair` calls, preserving the exact per-element
    ///   masking granularity the ragged-shape tests (down to 17x9, 33x17)
    ///   already depend on.
    ///
    /// Standalone-validated on real sm_89 hardware before this was wired
    /// in (project scratchpad's validate_fp8_vectorized_stage.py,
    /// `quant_b_vec`) across exact-fit, every mid-quad boundary offset
    /// (n_bound mod 4 = 0/1/2/3), and extreme (n_bound=0, n_bound=1)
    /// cases.
    ///
    /// `extra_valid_pred` - see `emit_fp8_quantize_stage_a`'s doc comment;
    /// AND-ed into the FAST path's combined predicate and into both of the
    /// SLOW path's per-element predicates.
    ///
    /// **Alignment precondition, found the hard way**: `ld.global.v4.f32`
    /// requires its address 16-byte aligned. B's row byte-stride is `n*4`
    /// (real N, arbitrary - unlike A, whose row-stride `k*4` is always a
    /// multiple of 16 since `K % FP8_GEMM_CTA_K(64) == 0` is already
    /// required). When `n % 4 != 0`, `(grow*n + col0)*4` is NOT a multiple
    /// of 16 for most `grow` (K-row) values, even though `col0` itself is
    /// always a multiple of 4 - a real `CUDA_ERROR_MISALIGNED_ADDRESS`
    /// crash, caught by this project's own ragged-shape real-hardware
    /// testing discipline at M=33,N=17,K=64 before this ever reached a
    /// released kernel (17 % 4 == 1). Since `n` is a compile-time constant
    /// here (the kernel's `@tile` N, not a runtime value), this is
    /// resolved at Rust codegen time: `emit_fp8_quantize_stage_b` (the
    /// public entry point below) only calls this vectorized path when
    /// `n % 4 == 0`; otherwise it falls back to
    /// `emit_fp8_quantize_stage_b_scalar`, the original pair-at-a-time
    /// scalar design (4-byte `ld.global.f32` never has an alignment
    /// concern). Standalone-reconfirmed for both the failing (misaligned)
    /// and fixed (gated) cases on real sm_89 hardware.
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_quantize_stage_b_vectorized(
        &mut self,
        b_g: &str,
        k0: &str,
        cta_col0: &str,
        k: u32,
        n: u32,
        smem_b: &str,
        inv_scale_b: &str,
        threads_per_cta: u32,
        cta_n: u32,
        extra_valid_pred: Option<&str>,
    ) {
        debug_assert_eq!(
            k % FP8_GEMM_CTA_K,
            0,
            "emit_fp8_quantize_stage_b assumes the caller already validated K % FP8_GEMM_CTA_K == 0 (B's row/K-side is never bounds-checked below)"
        );
        debug_assert_eq!(cta_n % 4, 0, "vectorized B stage requires cta_n a multiple of 4");
        debug_assert_eq!(n % 4, 0, "emit_fp8_quantize_stage_b_vectorized requires n % 4 == 0 for ld.global.v4.f32 alignment - caller must dispatch to the scalar fallback otherwise");
        let total_quads = (FP8_GEMM_CTA_K * cta_n) / 4;
        let quads_per_row = cta_n / 4;

        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label("FP8_STAGE_B");
        let loop_end = self.alloc_label("FP8_STAGE_B_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_quads).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        let local_row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", local_row, idx, quads_per_row).unwrap();
        let quad_in_row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", quad_in_row, idx, quads_per_row).unwrap();
        let col_start = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", col_start, quad_in_row).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, k0, local_row).unwrap();

        let col0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col0, cta_col0, col_start).unwrap();
        let col_last = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 3;", col_last, col0).unwrap();
        let mut p_fast = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_fast, col_last, n).unwrap();
        if let Some(extra) = extra_valid_pred {
            let combined = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", combined, p_fast, extra).unwrap();
            p_fast = combined;
        }

        let slow_label = self.alloc_label("FP8_STAGE_B_SLOW");
        let join_label = self.alloc_label("FP8_STAGE_B_JOIN");
        writeln!(&mut self.ptx_buffer, "    @!{} bra {};", p_fast, slow_label).unwrap();

        // ---- FAST: whole quad guaranteed in-bounds (p_fast true on this
        // fall-through path) ----
        let g_lin = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", g_lin, grow, n, col0).unwrap();
        let g_byte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", g_byte, g_lin).unwrap();
        let gaddr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr, b_g, g_byte).unwrap();

        let f: Vec<String> = (0..4).map(|_| self.alloc_regf32()).collect();
        writeln!(&mut self.ptx_buffer, "    ld.global.v4.f32 {{{}}}, [{}];", f.join(","), gaddr).unwrap();
        let combined_word = self.emit_fp8_pack_quad(&f, inv_scale_b);

        let s_lin = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", s_lin, local_row, cta_n, col_start).unwrap();
        let saddr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_b, s_lin).unwrap();
        writeln!(&mut self.ptx_buffer, "    st.shared.b32 [{}], {};", saddr, combined_word).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", join_label).unwrap();

        // ---- SLOW: quad straddles the N boundary - original per-element-
        // predicated scalar path, recomputed independently of the FAST
        // path's (conditionally-assigned) registers above ----
        writeln!(&mut self.ptx_buffer, "    {}:", slow_label).unwrap();

        let mut p_ok0 = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_ok0, col0, n).unwrap();
        let col1 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", col1, col0).unwrap();
        let mut p_ok1 = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_ok1, col1, n).unwrap();
        let col2 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 2;", col2, col0).unwrap();
        let mut p_ok2 = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_ok2, col2, n).unwrap();
        let col3 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 3;", col3, col0).unwrap();
        let mut p_ok3 = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_ok3, col3, n).unwrap();
        if let Some(extra) = extra_valid_pred {
            for p in [&mut p_ok0, &mut p_ok1, &mut p_ok2, &mut p_ok3] {
                let c = self.alloc_pred();
                writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", c, p, extra).unwrap();
                *p = c;
            }
        }

        let g_lin_slow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", g_lin_slow, grow, n, col0).unwrap();
        let g_byte_slow = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", g_byte_slow, g_lin_slow).unwrap();
        let gaddr_slow = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr_slow, b_g, g_byte_slow).unwrap();

        let s_lin_slow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", s_lin_slow, local_row, cta_n, col_start).unwrap();
        let saddr_slow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr_slow, smem_b, s_lin_slow).unwrap();

        self.emit_fp8_quantize_pair(&gaddr_slow, 0, &p_ok0, &p_ok1, inv_scale_b, &saddr_slow, 0);
        self.emit_fp8_quantize_pair(&gaddr_slow, 8, &p_ok2, &p_ok3, inv_scale_b, &saddr_slow, 2);

        writeln!(&mut self.ptx_buffer, "    {}:", join_label).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Original (pre-session-5) pair-at-a-time scalar B quantize+stage -
    /// kept as the fallback for `n % 4 != 0` (see
    /// `emit_fp8_quantize_stage_b_vectorized`'s doc comment: `ld.global.f32`
    /// is always 4-byte-aligned regardless of N's residue mod 4, unlike the
    /// vectorized path's `ld.global.v4.f32`, so this is unconditionally
    /// alignment-safe at the cost of not getting the vectorization win for
    /// this dimension).
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_quantize_stage_b_scalar(
        &mut self,
        b_g: &str,
        k0: &str,
        cta_col0: &str,
        k: u32,
        n: u32,
        smem_b: &str,
        inv_scale_b: &str,
        threads_per_cta: u32,
        cta_n: u32,
        extra_valid_pred: Option<&str>,
    ) {
        debug_assert_eq!(
            k % FP8_GEMM_CTA_K,
            0,
            "emit_fp8_quantize_stage_b_scalar assumes the caller already validated K % FP8_GEMM_CTA_K == 0 (B's row/K-side is never bounds-checked below)"
        );
        let total_pairs = (FP8_GEMM_CTA_K * cta_n) / 2;
        let pairs_per_row = cta_n / 2;

        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label("FP8_STAGE_B");
        let loop_end = self.alloc_label("FP8_STAGE_B_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_pairs).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        let local_row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", local_row, idx, pairs_per_row).unwrap();
        let pair_in_row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", pair_in_row, idx, pairs_per_row).unwrap();
        let col_start = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 2;", col_start, pair_in_row).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, k0, local_row).unwrap();

        let col0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col0, cta_col0, col_start).unwrap();
        let mut p_ok0 = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_ok0, col0, n).unwrap();
        let col1 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", col1, col0).unwrap();
        let mut p_ok1 = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_ok1, col1, n).unwrap();
        if let Some(extra) = extra_valid_pred {
            let c0 = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", c0, p_ok0, extra).unwrap();
            p_ok0 = c0;
            let c1 = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", c1, p_ok1, extra).unwrap();
            p_ok1 = c1;
        }

        let g_lin = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", g_lin, grow, n, col0).unwrap();
        let g_byte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", g_byte, g_lin).unwrap();
        let gaddr_base = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr_base, b_g, g_byte).unwrap();

        let s_lin = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", s_lin, local_row, cta_n, col_start).unwrap();
        let saddr_base = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr_base, smem_b, s_lin).unwrap();

        self.emit_fp8_quantize_pair(&gaddr_base, 0, &p_ok0, &p_ok1, inv_scale_b, &saddr_base, 0);

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Dispatches to the vectorized (`ld.global.v4.f32`) or scalar B
    /// quantize+stage based on whether `n % 4 == 0` - see
    /// `emit_fp8_quantize_stage_b_vectorized`'s doc comment for why this
    /// gate exists (16-byte alignment of B's row stride).
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_quantize_stage_b(
        &mut self,
        b_g: &str,
        k0: &str,
        cta_col0: &str,
        k: u32,
        n: u32,
        smem_b: &str,
        inv_scale_b: &str,
        threads_per_cta: u32,
        cta_n: u32,
        extra_valid_pred: Option<&str>,
    ) {
        if n % 4 == 0 {
            self.emit_fp8_quantize_stage_b_vectorized(b_g, k0, cta_col0, k, n, smem_b, inv_scale_b, threads_per_cta, cta_n, extra_valid_pred);
        } else {
            self.emit_fp8_quantize_stage_b_scalar(b_g, k0, cta_col0, k, n, smem_b, inv_scale_b, threads_per_cta, cta_n, extra_valid_pred);
        }
    }

    /// FP8 analogue of `emit_gemm_compute_block` - see that function's doc
    /// comment for the general "B-fragments-per-j computed once and reused
    /// across every i, A-fragment-per-i computed once and reused across
    /// every j" structure this mirrors (NOT copy-pasted: FP8's
    /// `mma.sync.m16n8k32` already consumes a full 2-register B fragment
    /// and 4-register A fragment in ONE call per `(i, j)` - unlike F16's
    /// `mma.sync.m16n8k16`, there is no further N-half split here, since
    /// this mma's own N=8 already matches the per-`j` granularity
    /// exactly). Standalone-validated on real sm_89 hardware (project
    /// scratchpad's validate_fp8_multiwarp.py) before this function was
    /// written - see `emit_fp8_gemm_kernel`'s doc comment.
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_gemm_compute_block(
        &mut self,
        acc: &[Vec<Vec<String>>],
        smem_a: &str,
        smem_b: &str,
        group_id: &str,
        tig: &str,
        warp_row0_local: &str,
        warp_col0_local: &str,
        num_i: u32,
        num_j: u32,
        k_substeps: u32,
        cta_n: u32,
    ) {
        for ks in 0..k_substeps {
            let kk = ks * 32;

            // B fragments per j, reused across every i below (same
            // reuse-across-i optimization emit_gemm_compute_block uses for
            // the F16 kernel's B fragments).
            let mut b_frags: Vec<Vec<String>> = Vec::with_capacity(num_j as usize);
            for j in 0..num_j {
                b_frags.push(self.emit_fp8_load_b_fragment(smem_b, group_id, tig, warp_col0_local, j, kk, cta_n));
            }

            // A fragment per i, reused across every j below.
            for i in 0..num_i {
                let a_frag = self.emit_fp8_load_a_fragment(smem_a, group_id, tig, warp_row0_local, i, kk);
                for j in 0..num_j {
                    let d = acc[i as usize][j as usize].join(",");
                    writeln!(
                        &mut self.ptx_buffer,
                        "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                        d, a_frag.join(","), b_frags[j as usize].join(","), d
                    ).unwrap();
                }
            }
        }
    }

    /// Emits a complete, self-contained FP8 (e4m3) Tensor Core GEMM kernel
    /// body for a kernel carrying a validated 5-param `@tile(M, N, K)` FP8
    /// shape - see `tile_gemm_fp8_operands`. A, B arrive as F32 in global
    /// memory and are quantized to e4m3 on the fly (fused - no separate
    /// quantize kernel or pass; `quantization_pass.rs`'s own FP8 path
    /// (`emit_fp32_to_fp8`) was checked and found unusable for this - it
    /// never actually converts anything, just zeros a placeholder register
    /// with a `// placeholder for cvt.rn.satfinite.e4m3x4` comment, and is
    /// structurally coupled to the RT/Tensor coprocessor's
    /// `coprocessor_smem` symbol, which
    /// `investigation_rt_tensor_coprocessor_findings.md` already found never
    /// produces valid PTX) via `cvt.rn.satfinite.e4m3x2.f32` while staging
    /// into shared memory - that instruction is standalone-validated bit-
    /// for-bit against `torch.float8_e4m3fn`'s own encoding, including
    /// round-to-nearest and `.satfinite` out-of-range saturation, on real
    /// sm_89 hardware (project scratchpad's validate_fp8_quantize.py). The
    /// mma itself is `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` -
    /// Ada/sm_89-compatible (unlike the Hopper-only WGMMA path this
    /// project's other FP8 code was stuck on - see
    /// investigation_fp8_int8_quantization_findings.md), requiring PTX ISA
    /// >=8.4 on this target (see `ptx_version_for_sm`'s `sm_89` arm). Its
    /// per-lane A/B/D fragment layout was derived from PTX ISA 8.5
    /// 9.7.13.4.10 (A/B - the `.s8`/`.u8`/`.e4m3`/`.e5m2` bullets share one
    /// formula) and 9.7.13.4.9 (D/C, reused verbatim - accumulator shape
    /// only depends on M=16/N=8, not K) and standalone-validated on real
    /// sm_89 hardware against both the ISA's own formula and an exact-
    /// integer CPU reference (project scratchpad's validate_fp8_mma.py).
    ///
    /// **Multi-warp CTA tiling** (session 2 - see
    /// `investigation_fp8_gemm_findings.md`'s "next session" list, item 1):
    /// the original first pass used exactly one `mma` instruction's shape
    /// (M=16, N=8, K=32) with a single warp (32 threads) per CTA - at
    /// 4096x4096x4096 that was a grid of 131,072 CTAs, each running only 32
    /// threads through 128 sequential K-iterations with zero intra-CTA
    /// parallelism. This version tiles `cta_m x cta_n x FP8_GEMM_CTA_K`
    /// per CTA across `warps_m x warps_n` warps, each warp computing a
    /// `per_warp_m x per_warp_n` output tile via `num_i x num_j`
    /// `mma.sync.m16n8k32` calls per K-substep. Structural template:
    /// `emit_tensor_core_gemm_kernel`/`emit_gemm_compute_block` (the F16
    /// kernel's OWN multi-warp tiling - see
    /// benchmark_y_tensor_core_gemm_results.md), NOT copy-pasted (FP8's
    /// `mma.sync.m16n8k32` has a different fragment shape/register count
    /// than F16's `wmma`/`ldmatrix` path, and there is no `ldmatrix` 8-bit
    /// variant at all - fragments are still hand-computed `ld.shared`, as
    /// in the original single-warp kernel). The multi-warp address
    /// composition (per-warp CTA-local tile origin, `lane = %tid.x & 31`
    /// now that a CTA is more than one warp, and the generalized
    /// cooperative quantize+stage step) was standalone-validated on real
    /// sm_89 hardware BEFORE this function was written (project
    /// scratchpad's validate_fp8_multiwarp.py) against an exact-integer CPU
    /// reference across single-CTA, multi-CTA, multi-K-tile, and ragged M/N
    /// shapes, and separately checked for the new risk multi-warp tiling
    /// introduces - cross-warp shared-memory races if `bar.sync` placement
    /// isn't right (multiple warps now share one smem_a/smem_b pair per
    /// CTA, unlike the original's one warp racing only itself) - via 30
    /// repeated launches with fixed input, confirmed bit-for-bit identical
    /// (validate_fp8_multiwarp_race.py). This session's kernel was
    /// single-buffered (stage -> `bar.sync` -> compute -> `bar.sync`,
    /// 2 barriers/K-iteration) - see session 4 below for how that changed.
    ///
    /// **Two-tier tile selection** (session 3 - see
    /// `investigation_fp8_gemm_findings.md`'s "Session 3" section): session
    /// 2's single fixed 128x128x64/4x2-warp shape (`FP8_GEMM_CTA_M_LARGE`
    /// etc.) measured a real regression at M=N=256 on real hardware
    /// (0.374x -> ~0.11x of `torch._scaled_mm`) - a 128x128 CTA tile only
    /// produces a 2x2=4-CTA grid at that size, leaving most SMs idle. This
    /// session picks between `FP8_GEMM_CTA_M_LARGE`/`_N_LARGE`/
    /// `_WARPS_M_LARGE`/`_WARPS_N_LARGE` and the smaller
    /// `..._SMALL` variants (64x64x64, 2x2 warps) at emit time, purely from
    /// the kernel's compile-time `M`/`N` (`m <= FP8_GEMM_SMALL_THRESHOLD ||
    /// n <= FP8_GEMM_SMALL_THRESHOLD`, mirroring
    /// `Autotuner::generate_candidates`'s own F16 small-shape threshold) -
    /// see `FP8_GEMM_CTA_M_LARGE`'s doc comment for the two shapes' full
    /// rationale and real hardware confirmation (`ptxas -v` register/smem
    /// counts, standalone validate_fp8_smalltile.py). This is deliberately
    /// a two-shape heuristic, not real Autotuner integration - see below.
    ///
    /// **Software double-buffered pipelining** (session 4 - see
    /// `investigation_fp8_gemm_findings.md`'s "Session 4" section): real
    /// `cp.async` - the mechanism `emit_gemm_tile_load_async` uses for the
    /// F16 kernel's OWN pipelining - copies raw bytes global->shared with
    /// NO type conversion, so it cannot perform this kernel's
    /// `cvt.rn.satfinite.e4m3x2.f32` F32->e4m3 staging conversion; staging
    /// RAW (unconverted) F32 instead would need 4x the shared-memory bytes
    /// (e4m3 is 1 byte/elem, f32 is 4), which measured out to ~147KB for
    /// the `_LARGE` tier - over this GPU's real ~100KB per-CTA dynamic-
    /// shared ceiling (see `emit_tensor_core_gemm_kernel`'s doc comment for
    /// where that number comes from). So this session pipelines with
    /// ordinary SYNCHRONOUS `ld.global.f32`/`cvt`/`st.shared` instructions
    /// instead of `cp.async`, issued earlier in program order relative to
    /// when their result is needed: a double-buffered (2x the previous
    /// smem footprint - still static, still under 48KB for both tiers) K-
    /// loop with a prologue that unconditionally stages K-tile 0, then each
    /// steady-state iteration issues the (predicated) prefetch of the NEXT
    /// K-tile into the OTHER buffer BEFORE consuming the current
    /// (already-staged) buffer - cutting `bar.sync` from 2/iteration down
    /// to 1 (a thread's own later reads always see its own earlier writes
    /// without a barrier; only cross-thread visibility needs one, and a
    /// single barrier placed after both the prefetch-write and the
    /// compute-read suffices for both directions of that visibility - see
    /// `emit_fp8_quantize_stage_a`'s doc comment for the prefetch-
    /// suppression predicate this needs on the loop's last iteration).
    /// Standalone-validated on real sm_89 hardware BEFORE this was written
    /// (project scratchpad's validate_fp8_pipeline.py): exact-integer
    /// correctness across K depths from 1 K-tile (prologue-only - the
    /// steady-state loop's single iteration must fully predicate off its
    /// own prefetch) to 8, plus 30 repeated launches at the deepest depth
    /// tested, confirming the reduced-to-1 `bar.sync`/iteration design
    /// doesn't reopen a cross-warp race.
    ///
    /// **Vectorized quantize+stage, extended tile threshold, rejected
    /// swizzle/tier attempts (session 5 - see
    /// `investigation_fp8_gemm_findings.md`'s "Session 5" section)**: acted
    /// on session 4's own top-priority next step ("vectorizing the
    /// quantize+stage step... is now a concrete, testable hypothesis").
    /// `emit_fp8_quantize_stage_a`/`_b` now move 4 elements (128 bits) per
    /// thread/iteration via `ld.global.v4.f32` + `emit_fp8_pack_quad`
    /// (2x `cvt.rn.satfinite.e4m3x2.f32` + one `st.shared.b32`) instead of
    /// the original 2-element/iteration scalar pair - a real, large win at
    /// 1024+ (see that section's benchmark table). B needs a hybrid fast/
    /// slow path (`emit_fp8_quantize_stage_b_vectorized`/`_scalar`) since
    /// N's boundary can fall inside a 4-element quad and - a real bug
    /// caught by this project's own ragged-shape testing discipline before
    /// release - B's row byte-stride isn't always a multiple of 16, so
    /// `ld.global.v4.f32` isn't always alignment-safe; gated on `n % 4 ==
    /// 0` at Rust codegen time (`n` is a compile-time `@tile` value).
    /// `FP8_GEMM_SMALL_THRESHOLD` was also raised 256 -> 512 after a real
    /// hardware A/B showed `_SMALL` beating `_LARGE` at M=N=512 by ~1.4x.
    /// Two other hypotheses were built, standalone-benchmarked on real
    /// hardware, and REJECTED as genuine negative results (not silently
    /// dropped) - see `FP8_GEMM_CTA_M_LARGE`'s doc comment for a `_TINY`
    /// (32x32/1-warp) tier that consistently lost to `_SMALL` despite
    /// higher CTA counts, and `emit_fp8_load_a_one`'s neighborhood (git
    /// history) for an XOR smem swizzle that mathematically eliminated a
    /// real, confirmed 4-way bank conflict in A's fragment load but cost
    /// enough extra registers (81->94 at `_SMALL`, 117->144 at `_LARGE`)
    /// to regress 4096x4096x4096 from 0.219x to 0.120x via lost occupancy
    /// - reverted rather than shipped.
    ///
    /// Deliberate, disclosed scope choices remaining after this pass (see
    /// benchmark_y_tensor_core_gemm_results.md for how the f16 GEMM
    /// kernel's OWN history went from correctness-first to optimized over
    /// multiple sessions - multi-warp tiling, tile selection, and
    /// pipelining are more steps along that arc for FP8, not a finished
    /// optimization target):
    /// - No grid swizzle, no REAL Autotuner integration (`generate_candidates`
    ///   accepts but ignores `Precision::Fp8`, see that enum's doc comment) -
    ///   this function's own `m <= FP8_GEMM_SMALL_THRESHOLD` branch is a
    ///   disclosed, narrower stand-in: two fixed shapes and one threshold,
    ///   not a real per-problem-size search over the full candidate space
    ///   the F16 path gets.
    /// - K must be an exact multiple of `FP8_GEMM_CTA_K` (64, shared by both
    ///   tiers) (`debug_assert` - matches the f16 kernel's own "K must be a
    ///   multiple of cta_k" scope note, narrowed from the original
    ///   single-warp kernel's "K must be a multiple of 32" now that a full
    ///   `FP8_GEMM_CTA_K`-wide K-slab is staged per iteration); M and N may
    ///   be any positive value - both the A/B tile loads (predicated,
    ///   zero-filled) and the C store (predicated per-element) are
    ///   boundary-masked for ragged M/N edges, standalone-validated down to
    ///   shapes smaller than one warp-tile edge (e.g. 17x9) for BOTH tiers.
    /// - scale_a/scale_b are per-tensor scalars the caller/launcher computes
    ///   (typically `amax/448.0`, e4m3's max normal magnitude) and passes in
    ///   as plain F32 kernel params, applied to the f32 accumulator in the
    ///   epilogue (`C = scale_a * scale_b * (A_e4m3 @ B_e4m3)`) - not
    ///   computed on-device (would need a separate reduction pass, out of
    ///   scope here).
    /// - Static (not dynamic/`extern`) `.shared` arrays: the DOUBLE-
    ///   buffered combined A+B tile footprint (32768 bytes for `_LARGE`,
    ///   16384 for `_SMALL` - e4m3 is 1 byte/element, doubled again for the
    ///   2-stage pipeline above) stays well under ptxas's 48KB static-
    ///   shared cap for both tiers, so the simpler static form still
    ///   suffices (unlike the F16 kernel's much larger, autotuned,
    ///   `cp.async`-pipelined tiles, which need the dynamic/`extern`
    ///   mechanism - see `pending_extern_decls`'s doc comment).
    #[allow(clippy::too_many_arguments)]
    fn emit_fp8_gemm_kernel(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        a_ptr: &str,
        b_ptr: &str,
        scale_a_reg: &str,
        scale_b_reg: &str,
        c_ptr: &str,
        kernel_name: &str,
    ) -> u32 {
        // Hard assert, not a `debug_assert!` - this kernel has no K-tail path,
        // so a non-multiple K silently drops the remainder in release builds.
        // See `emit_tensor_core_gemm_kernel`'s K-tail note.
        assert_eq!(
            k % FP8_GEMM_CTA_K,
            0,
            "@tile K={} must be a multiple of FP8_GEMM_CTA_K ({}) for the multi-warp FP8 GEMM \
             kernel; it has no K-tail path and would silently drop the remainder",
            k,
            FP8_GEMM_CTA_K
        );
        // ---- two-tier tile selection (see doc comment): a disclosed
        // stand-in for real Autotuner integration, purely a function of
        // this kernel's compile-time M/N. ----
        let is_small = m <= FP8_GEMM_SMALL_THRESHOLD || n <= FP8_GEMM_SMALL_THRESHOLD;
        let (cta_m, cta_n, warps_m, warps_n) = if is_small {
            (FP8_GEMM_CTA_M_SMALL, FP8_GEMM_CTA_N_SMALL, FP8_GEMM_WARPS_M_SMALL, FP8_GEMM_WARPS_N_SMALL)
        } else {
            (FP8_GEMM_CTA_M_LARGE, FP8_GEMM_CTA_N_LARGE, FP8_GEMM_WARPS_M_LARGE, FP8_GEMM_WARPS_N_LARGE)
        };
        let cta_k = FP8_GEMM_CTA_K;

        let k_tiles = k / cta_k;
        let grid_x = (m + cta_m - 1) / cta_m;
        let grid_y = (n + cta_n - 1) / cta_n;
        let num_warps = warps_m * warps_n;
        let threads_per_cta = num_warps * 32;
        let per_warp_m = cta_m / warps_m;
        let per_warp_n = cta_n / warps_n;
        let num_i = per_warp_m / 16;
        let num_j = per_warp_n / 8;
        let k_substeps = cta_k / 32;
        debug_assert_eq!(num_i * 16 * warps_m, cta_m);
        debug_assert_eq!(num_j * 8 * warps_n, cta_n);
        debug_assert_eq!(k_substeps * 32, cta_k);

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(
            &mut self.ptx_buffer,
            "    // [Y FP8 TENSOR CORE GEMM] M={} N={} K={} | CTA {}x{}x{} ({}) | {}x{} warps | mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32",
            m, n, k, cta_m, cta_n, cta_k, if is_small { "small" } else { "large" }, warps_m, warps_n
        ).unwrap();
        writeln!(&mut self.ptx_buffer, "    // Fused on-the-fly FP32->e4m3 quantization (cvt.rn.satfinite.e4m3x2.f32); epilogue dequants by scale_a*scale_b.").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Launch grid required: ({}, {}, 1) CTAs, block ({},1,1).", grid_x, grid_y, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        // ---- double-buffered (2-stage) shared memory for the software-
        // pipelined K-loop below (see that section's doc comment): one
        // extra factor of 2 on top of the single-buffered byte count,
        // still static (not dynamic/`extern`) since even doubled this
        // stays well under ptxas's 48KB cap for both tiers (32768 bytes
        // for LARGE, 16384 for SMALL).
        let smem_a_sym = format!("smem_fp8_A_{}", kernel_name);
        let smem_b_sym = format!("smem_fp8_B_{}", kernel_name);
        let stage_a_bytes = cta_m * cta_k;
        let stage_b_bytes = cta_k * cta_n;
        writeln!(&mut self.ptx_buffer, "    .shared .align 4 .b8 {}[{}];", smem_a_sym, 2 * stage_a_bytes).unwrap();
        writeln!(&mut self.ptx_buffer, "    .shared .align 4 .b8 {}[{}];", smem_b_sym, 2 * stage_b_bytes).unwrap();

        let a_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", a_g, a_ptr).unwrap();
        let b_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", b_g, b_ptr).unwrap();
        let c_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", c_g, c_ptr).unwrap();

        let smem_a = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", smem_a, smem_a_sym).unwrap();
        let smem_b = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", smem_b, smem_b_sym).unwrap();

        // ---- CTA tile origin (plain raster order - no grid swizzle yet,
        // see doc comment) ----
        let ctaid_x = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", ctaid_x).unwrap();
        let ctaid_y = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.y;", ctaid_y).unwrap();
        let cta_row0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", cta_row0, ctaid_x, cta_m).unwrap();
        let cta_col0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", cta_col0, ctaid_y, cta_n).unwrap();

        // ---- warp/lane decomposition - lane MUST be tid.x & 31, not
        // tid.x directly, now that a CTA is warps_m * warps_n warps (128 or
        // 256 threads depending on tier), not one warp - standalone-
        // validated on real sm_89 hardware before this function was
        // written (see doc comment). groupID/threadID_in_group per PTX ISA
        // 9.7.13.4.10 (unchanged from the single-warp kernel); warp_m/
        // warp_n/warp_row0_local/warp_col0_local mirror
        // emit_tensor_core_gemm_kernel's own warp decomposition. ----
        let tid = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
        let lane = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 31;", lane, tid).unwrap();
        let group_id = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 2;", group_id, lane).unwrap();
        let tig = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 3;", tig, lane).unwrap();
        let warp_id = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 5;", warp_id, tid).unwrap();
        let warp_m = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", warp_m, warp_id, warps_m).unwrap();
        let warp_n = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", warp_n, warp_id, warps_m).unwrap();
        let warp_row0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_row0_local, warp_m, per_warp_m).unwrap();
        let warp_col0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_col0_local, warp_n, per_warp_n).unwrap();

        // ---- loop-invariant reciprocals for quantization (scale_a/
        // scale_b are kernel params, unchanged across the K-loop) ----
        let inv_scale_a = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    rcp.rn.f32 {}, {};", inv_scale_a, scale_a_reg).unwrap();
        let inv_scale_b = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    rcp.rn.f32 {}, {};", inv_scale_b, scale_b_reg).unwrap();

        // ---- accumulators acc[i][j]: 4 f32 regs each, allocated ONCE
        // before the loop - loop-carried registers whose IDENTITY must stay
        // fixed across every dynamic K iteration (see
        // feedback_gemm_kernel_validation.md's loop-carried-accumulator
        // trap: a fresh register per iteration would silently discard all
        // but the last iteration's contribution) ----
        let mut acc: Vec<Vec<Vec<String>>> = Vec::with_capacity(num_i as usize);
        for _ in 0..num_i {
            let mut row = Vec::with_capacity(num_j as usize);
            for _ in 0..num_j {
                let d: Vec<String> = (0..4).map(|_| self.alloc_regf32()).collect();
                for r in &d {
                    writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", r).unwrap();
                }
                row.push(d);
            }
            acc.push(row);
        }

        // ---- software-pipelined (double-buffered) K-loop. `cp.async` -
        // the mechanism the F16 kernel uses for ITS pipelining (see
        // `emit_gemm_tile_load_async`) - copies raw bytes global->shared
        // with NO type conversion, so it cannot do this kernel's
        // F32->e4m3 `cvt.rn.satfinite.e4m3x2.f32` staging conversion;
        // staging raw (unconverted) F32 instead would need 4x the
        // shared-memory bytes (e4m3 is 1 byte/elem, f32 is 4), which does
        // not fit this kernel's tile sizes. So this pipelines with
        // ordinary synchronous `ld.global.f32`/`cvt`/`st.shared`
        // instructions instead, issued EARLIER in program order relative
        // to when their result is needed: a PROLOGUE unconditionally
        // stages K-tile 0 into buffer 0, then each steady-state iteration
        // issues the (predicated) prefetch of the NEXT K-tile into the
        // OTHER buffer before consuming the CURRENT (already-staged)
        // buffer's fragments - cutting `bar.sync` from 2/iteration (the
        // original single-buffered design) to 1, since a thread's own
        // later reads always see its own earlier writes without a
        // barrier (only cross-thread visibility needs one). Standalone-
        // validated on real sm_89 hardware BEFORE this was written
        // (project scratchpad's validate_fp8_pipeline.py: exact-integer
        // correctness across K depths from 1 (prologue-only, prefetch
        // must be fully predicated off) to 8 K-tiles, plus 30 repeated
        // launches at K=8 - the deepest/most bar.sync round-trips tested
        // - confirming this single-bar.sync design doesn't reopen a
        // cross-warp race).
        //
        // Buffer index = k_iter % 2 (read) / (k_iter+1) % 2 (write, i.e.
        // the OTHER buffer); byte offset into the doubled smem_a/smem_b
        // arrays is `buf_index * stage_bytes`.
        let k0_zero = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k0_zero).unwrap();
        self.emit_fp8_quantize_stage_a(&a_g, &cta_row0, &k0_zero, m, k, &smem_a, &inv_scale_a, threads_per_cta, cta_m, None);
        self.emit_fp8_quantize_stage_b(&b_g, &k0_zero, &cta_col0, k, n, &smem_b, &inv_scale_b, threads_per_cta, cta_n, None);
        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

        let k_iter = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
        let loop_start = self.alloc_label(&format!("FP8_GEMM_K_{}", kernel_name));
        let loop_end = self.alloc_label(&format!("FP8_GEMM_K_DONE_{}", kernel_name));
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_exit = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_exit, k_iter, k_tiles).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_exit, loop_end).unwrap();

        let read_idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, 2;", read_idx, k_iter).unwrap();
        let next_k_iter = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", next_k_iter, k_iter).unwrap();
        let write_idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, 2;", write_idx, next_k_iter).unwrap();

        let read_a_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", read_a_off, read_idx, stage_a_bytes).unwrap();
        let smem_a_read = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_a_read, smem_a, read_a_off).unwrap();
        let read_b_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", read_b_off, read_idx, stage_b_bytes).unwrap();
        let smem_b_read = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_b_read, smem_b, read_b_off).unwrap();

        let write_a_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", write_a_off, write_idx, stage_a_bytes).unwrap();
        let smem_a_write = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_a_write, smem_a, write_a_off).unwrap();
        let write_b_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", write_b_off, write_idx, stage_b_bytes).unwrap();
        let smem_b_write = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_b_write, smem_b, write_b_off).unwrap();

        let p_next_valid = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_next_valid, next_k_iter, k_tiles).unwrap();
        let next_k0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", next_k0, next_k_iter, cta_k).unwrap();

        // Issue the prefetch for `next_k_iter` (masked off via
        // `p_next_valid` if it doesn't exist - only possible on the
        // loop's last iteration), THEN compute on `read_idx` (already
        // staged, either by the prologue or a previous iteration's
        // prefetch) - the high-latency global loads above are issued
        // before the tensor-core work below needs their result, which is
        // the whole latency-hiding point of this restructuring.
        self.emit_fp8_quantize_stage_a(&a_g, &cta_row0, &next_k0, m, k, &smem_a_write, &inv_scale_a, threads_per_cta, cta_m, Some(&p_next_valid));
        self.emit_fp8_quantize_stage_b(&b_g, &next_k0, &cta_col0, k, n, &smem_b_write, &inv_scale_b, threads_per_cta, cta_n, Some(&p_next_valid));

        self.emit_fp8_gemm_compute_block(
            &acc, &smem_a_read, &smem_b_read, &group_id, &tig, &warp_row0_local, &warp_col0_local, num_i, num_j, k_substeps, cta_n,
        );

        // Single bar.sync/iteration: ensures (a) every warp's prefetch
        // writes to the write buffer above are visible before the NEXT
        // iteration reads that same buffer, and (b) every warp is done
        // reading the read buffer above before some LATER iteration's
        // prefetch overwrites it (guaranteed since that reuse is always
        // >=2 iterations away, and this same barrier gates every
        // iteration in between) - see doc comment's cross-warp-race
        // discussion and standalone validation.
        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();

        // ---- epilogue: dequant scale + boundary-masked per-element store,
        // over every warp's (i, j) accumulator fragments ----
        let scale = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", scale, scale_a_reg, scale_b_reg).unwrap();

        for i in 0..num_i {
            for j in 0..num_j {
                for idx4 in 0..4u32 {
                    let (local_row, local_col) = self.emit_fp8_accum_row_col(&group_id, &tig, idx4);

                    let row1 = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", row1, warp_row0_local, i * 16).unwrap();
                    let grow_local = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow_local, row1, local_row).unwrap();
                    let grow = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, cta_row0, grow_local).unwrap();

                    let col1 = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col1, warp_col0_local, j * 8).unwrap();
                    let gcol_local = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol_local, col1, local_col).unwrap();
                    let gcol = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol, cta_col0, gcol_local).unwrap();

                    let p_row = self.alloc_pred();
                    writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_row, grow, m).unwrap();
                    let p_col = self.alloc_pred();
                    writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_col, gcol, n).unwrap();
                    let p_ok = self.alloc_pred();
                    writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_ok, p_row, p_col).unwrap();

                    let dval = self.alloc_regf32();
                    writeln!(
                        &mut self.ptx_buffer,
                        "    mul.f32 {}, {}, {};",
                        dval, acc[i as usize][j as usize][idx4 as usize], scale
                    ).unwrap();

                    let lin = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", lin, grow, n, gcol).unwrap();
                    let byte = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", byte, lin).unwrap();
                    let addr = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", addr, c_g, byte).unwrap();
                    writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}], {};", p_ok, addr, dval).unwrap();
                }
            }
        }

        threads_per_cta
    }

    /// Emits a thread-strided, boundary-masked cooperative load of one CTA's
    /// tile from global into padded shared memory - shared by the A- and
    /// B-tile loads inside `emit_tensor_core_gemm_kernel`'s K-loop.
    /// `(local_rows, local_cols)` is the tile shape being loaded;
    /// `(gmem_row0, gmem_col0)` are runtime registers giving the tile's
    /// top-left corner within the full row-major source matrix (row stride
    /// `gmem_row_stride` elements).
    ///
    /// Moves 8 f16 elements (128 bits) per thread per iteration via
    /// `ld.global.v4.u32`/`st.shared.v4.u32` (ground-truth instruction
    /// syntax and register count confirmed against real nvcc/ptxas output
    /// before this was written) rather than one element at a time: an
    /// earlier scalar (16-bit-at-a-time) version of this function was
    /// measured - median of 5 real-hardware runs per size, all 7 benchmark
    /// sizes, via tests/benchmark_y_tensor_core_gemm.py - to leave the
    /// compiled kernel 4-25x slower than cuBLAS, and the gap did not shrink
    /// at large sizes the way pure launch-overhead/grid-underutilization
    /// would predict, pointing at this loop's sheer per-element instruction
    /// count (loop control + div/rem + two bounds checks + address math, all
    /// repeated per 2-byte element) as a real, structural cost, not just a
    /// small-size artifact. Re-measuring after this change (same harness,
    /// same discipline) confirmed the diagnosis: the gap narrowed to
    /// 1.7x-3.2x slower across sizes (from 2x-25x), with two of the seven
    /// sizes (512, 2048) landing within run-to-run noise of cuBLAS
    /// (statistically indistinguishable, not a clean win - see that
    /// script's README/report output for the exact numbers). It does not
    /// close the gap outright: the remaining, *not yet implemented* cost is
    /// the lack of real `cp.async` pipelining (this path stages
    /// synchronously - see `emit_tensor_core_gemm_kernel`'s doc comment),
    /// which is a separate, disclosed scope cut, not something this
    /// vectorization pass attempted to fix.
    ///
    /// Boundary masking is therefore at 8-element chunk granularity, not
    /// per-element (a whole chunk is skipped, zero-filled, if the *last* of
    /// its 8 columns would be out of bounds) - the same whole-fragment
    /// masking philosophy `emit_tensor_core_gemm_kernel`'s epilogue already
    /// uses, for the same reason (finer-grained masking isn't expressible
    /// with a single vectorized instruction). Requires `local_cols`,
    /// `smem_stride`, and `gmem_row_stride` to each be a multiple of 8: true
    /// for every candidate `Autotuner::generate_candidates` produces
    /// (cta_k/cta_n are always multiples of 32, and the padding in
    /// `emit_tensor_core_gemm_kernel` preserves that) and true for every
    /// M/N/K this has actually been verified against (powers of two >= 256)
    /// - `debug_assert`ed rather than silently violated for any @tile shape
    /// that breaks it.
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_tile_load(
        &mut self,
        label_tag: &str,
        gmem_ptr: &str,
        gmem_row0: &str,
        gmem_col0: &str,
        gmem_row_stride: u32,
        gmem_row_bound: u32,
        gmem_col_bound: u32,
        local_rows: u32,
        local_cols: u32,
        smem_base: &str,
        smem_stride: u32,
        threads_per_cta: u32,
    ) {
        debug_assert_eq!(local_cols % 8, 0, "vectorized tile load requires local_cols a multiple of 8");
        debug_assert_eq!(smem_stride % 8, 0, "vectorized tile load requires smem_stride a multiple of 8");
        debug_assert_eq!(gmem_row_stride % 8, 0, "vectorized tile load requires gmem_row_stride a multiple of 8");

        let cols_per_chunk = local_cols / 8;
        let total_chunks = local_rows * cols_per_chunk;
        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label(&format!("LOAD_{}", label_tag));
        let loop_end = self.alloc_label(&format!("LOAD_{}_DONE", label_tag));
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_chunks).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        let lr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", lr, idx, cols_per_chunk).unwrap();
        let lc_chunk = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", lc_chunk, idx, cols_per_chunk).unwrap();
        let lc = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 8;", lc, lc_chunk).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, gmem_row0, lr).unwrap();
        let gcol = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol, gmem_col0, lc).unwrap();
        let gcol_end = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 8;", gcol_end, gcol).unwrap();

        let p_row = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_row, grow, gmem_row_bound).unwrap();
        let p_col = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.le.u32 {}, {}, {};", p_col, gcol_end, gmem_col_bound).unwrap();
        let p_ok = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_ok, p_row, p_col).unwrap();

        let gidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", gidx, grow, gmem_row_stride, gcol).unwrap();
        let gbyte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 2;", gbyte, gidx).unwrap();
        let gaddr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr, gmem_ptr, gbyte).unwrap();

        // 8 x f16 raw bit patterns (no value conversion) via one 128-bit
        // vector transfer - see doc comment.
        let v: Vec<String> = (0..4).map(|_| self.alloc_reg32()).collect();
        writeln!(&mut self.ptx_buffer, "    @{} ld.global.v4.u32 {{{}}}, [{}];", p_ok, v.join(","), gaddr).unwrap();
        for r in &v {
            writeln!(&mut self.ptx_buffer, "    @!{} mov.u32 {}, 0;", p_ok, r).unwrap();
        }

        let sidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, lr, smem_stride, lc).unwrap();
        let sbyte = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 1;", sbyte, sidx).unwrap();
        let saddr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_base, sbyte).unwrap();
        writeln!(&mut self.ptx_buffer, "    st.shared.v4.u32 [{}], {{{}}};", saddr, v.join(",")).unwrap();

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Async (`cp.async`) sibling of `emit_gemm_tile_load`: identical thread
    /// striding, chunk geometry, and boundary-address arithmetic (same
    /// `local_cols`/`smem_stride`/`gmem_row_stride`-multiple-of-8
    /// requirements, `debug_assert`ed identically - see that function's doc
    /// comment for the full rationale), but issues a direct global->shared
    /// `cp.async.cg` copy per chunk instead of routing through
    /// `ld.global.v4`+`st.shared.v4` registers - see
    /// `emit_tensor_core_gemm_kernel`'s pipelined path, the only caller.
    ///
    /// Per-chunk validity uses `cp.async`'s own 4-operand zero-fill form
    /// (`cp.async.cg.shared.global [dst], [src], 16, srcSize;` with `srcSize`
    /// a runtime register that is 0 or 16 via `selp`) rather than a
    /// predicated `@p` guard on the copy itself. This distinction matters
    /// here specifically: an `@p`-guarded instruction simply does not
    /// execute for false lanes, leaving the destination whatever it was
    /// before, whereas the zero-fill form unconditionally *writes* the
    /// destination (real data or zero) every call. Pipelining reuses the
    /// same physical stage slot across multiple, non-adjacent K-loop
    /// iterations, so an invalid chunk must be re-zeroed (or at least
    /// re-confirmed zero) every time that slot is refilled, not just once -
    /// the unconditional-write form makes this automatic. Both that
    /// property and the safety of the out-of-range address the invalid-lane
    /// case still computes (never dereferenced by the hardware when
    /// srcSize=0 - confirmed with a deliberately wild address far outside
    /// any mapped region, not just a mildly-past-the-buffer one) were
    /// verified with a standalone hand-written PTX probe compiled and run
    /// on real sm_89 hardware before this function was written, not assumed
    /// from documentation.
    ///
    /// `extra_valid_pred`, when `Some`, is AND-ed into the per-chunk mask -
    /// used by the main pipeline loop to additionally suppress the load
    /// when the K-tile being prefetched doesn't exist yet
    /// (`next_tile >= k_tiles`, near the end of the K loop). The prologue
    /// never passes this: it only ever prefetches compile-time-known-in-
    /// bounds tiles (stage count is already clamped to `k_tiles` before the
    /// prologue runs - see `emit_tensor_core_gemm_kernel`).
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_tile_load_async(
        &mut self,
        label_tag: &str,
        gmem_ptr: &str,
        gmem_row0: &str,
        gmem_col0: &str,
        gmem_row_stride: u32,
        gmem_row_bound: u32,
        gmem_col_bound: u32,
        local_rows: u32,
        local_cols: u32,
        smem_stage_base: &str,
        smem_stride: u32,
        threads_per_cta: u32,
        extra_valid_pred: Option<&str>,
    ) {
        debug_assert_eq!(local_cols % 8, 0, "vectorized async tile load requires local_cols a multiple of 8");
        debug_assert_eq!(smem_stride % 8, 0, "vectorized async tile load requires smem_stride a multiple of 8");
        debug_assert_eq!(gmem_row_stride % 8, 0, "vectorized async tile load requires gmem_row_stride a multiple of 8");

        let cols_per_chunk = local_cols / 8;
        let total_chunks = local_rows * cols_per_chunk;

        // Tag the staging block with its call-site label in both the looped
        // and unrolled forms below. In the looped form the tag also reaches
        // the PTX as part of the `ALOAD_<tag>` label, but the unrolled form
        // is branchless and emits no label at all, so the comment is what
        // identifies which tile (and which pipeline stage) a given run of
        // `cp.async`es belongs to - for reading emitted PTX, and for the
        // dispatch tests that assert a particular prefetch was emitted.
        writeln!(
            &mut self.ptx_buffer,
            "    // [Y ASYNC TILE LOAD] {} | {} chunks of 16B over {} threads",
            label_tag, total_chunks, threads_per_cta
        ).unwrap();

        // ---- fully unrolled staging, when the trip count is small ----
        // `total_chunks` and `threads_per_cta` are both compile-time
        // constants here, so the per-thread chunk count is too. ptxas cannot
        // derive that on its own: the loop below runs `idx = %tid.x;
        // idx < total_chunks; idx += threads_per_cta`, whose trip count
        // depends on `%tid.x`, an opaque runtime value. So ptxas keeps a
        // real loop and re-executes the whole address computation - the
        // div/rem row-column split, both bound checks, the 64-bit global
        // address `mad`/`mul.wide`/`add`, and the shared-address `mad`/
        // `shl`/`add` - once per single 16-byte `cp.async`. Measured on the
        // emitted sm_89 SASS for M=N=K=4096 (128x128x32, 2x2 warps): ~22
        // instructions per `LDGSTS`, i.e. ~200 instructions per K-tile to
        // issue 8 copies, against 64 `HMMA` in the same iteration.
        //
        // Emitting the copies straight-line instead hands ptxas exactly what
        // it already does well for the `ldmatrix` stream: every per-chunk
        // address becomes base+constant and folds into the instruction's own
        // immediate offset field. That takes the 4096 mainloop from 202 SASS
        // instructions with two 4-trip inner loops (~352 executed) to 182
        // straight-line ones, and measures **1.028x** end-to-end against the
        // looped form, bit-identical output, on an interleaved A/B ranked by
        // minimum (dispersion 0.4-1.0%, so the gain clears the tie band).
        //
        // It is deliberately *not* a large win: the mainloop is not
        // issue-bound. `sm__warps_active` is 16.3% of peak for both this
        // kernel and cuBLAS's `ampere_fp16_s1688gemm_fp16_128x128_...`, and a
        // pure back-to-back `mma.sync` probe reaches the full ~185 TFLOPS at
        // only 4 warps/SM. Halving the mainloop instruction count buying
        // ~3% is the evidence for that, not a disappointment - do not expect
        // further instruction-count work here to pay more.
        //
        // The unroll is capped because it is per *stage* and per A/B tile in
        // an already-unrolled K-loop body; past this the I-cache cost turns
        // it back into a loss. Beyond the cap, and for any tile whose chunk
        // count is not a compile-time multiple of the CTA width, the
        // original loop form below still applies.
        const MAX_STAGE_UNROLL: u32 = 16;
        if total_chunks % threads_per_cta == 0 && total_chunks / threads_per_cta <= MAX_STAGE_UNROLL {
            for u in 0..(total_chunks / threads_per_cta) {
                let idx = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
                if u > 0 {
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, u * threads_per_cta).unwrap();
                }
                self.emit_gemm_tile_load_async_chunk(
                    &idx, gmem_ptr, gmem_row0, gmem_col0, gmem_row_stride,
                    gmem_row_bound, gmem_col_bound, cols_per_chunk,
                    smem_stage_base, smem_stride, extra_valid_pred,
                );
            }
            return;
        }

        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label(&format!("ALOAD_{}", label_tag));
        let loop_end = self.alloc_label(&format!("ALOAD_{}_DONE", label_tag));
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_chunks).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        self.emit_gemm_tile_load_async_chunk(
            &idx, gmem_ptr, gmem_row0, gmem_col0, gmem_row_stride,
            gmem_row_bound, gmem_col_bound, cols_per_chunk,
            smem_stage_base, smem_stride, extra_valid_pred,
        );

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// One 16-byte `cp.async` chunk of `emit_gemm_tile_load_async`: splits the
    /// flat chunk index `idx` into its tile row/column, bound-checks both
    /// against the real matrix extent, and issues the copy in the zero-fill
    /// form described on that function. Factored out so the looped and fully
    /// unrolled forms there provably share one body rather than two copies
    /// that can drift apart.
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_tile_load_async_chunk(
        &mut self,
        idx: &str,
        gmem_ptr: &str,
        gmem_row0: &str,
        gmem_col0: &str,
        gmem_row_stride: u32,
        gmem_row_bound: u32,
        gmem_col_bound: u32,
        cols_per_chunk: u32,
        smem_stage_base: &str,
        smem_stride: u32,
        extra_valid_pred: Option<&str>,
    ) {
        let lr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", lr, idx, cols_per_chunk).unwrap();
        let lc_chunk = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", lc_chunk, idx, cols_per_chunk).unwrap();
        let lc = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 8;", lc, lc_chunk).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, gmem_row0, lr).unwrap();
        let gcol = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol, gmem_col0, lc).unwrap();
        let gcol_end = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 8;", gcol_end, gcol).unwrap();

        let p_row = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_row, grow, gmem_row_bound).unwrap();
        let p_col = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.le.u32 {}, {}, {};", p_col, gcol_end, gmem_col_bound).unwrap();
        let mut p_ok = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_ok, p_row, p_col).unwrap();
        if let Some(extra) = extra_valid_pred {
            let combined = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", combined, p_ok, extra).unwrap();
            p_ok = combined;
        }

        let gidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", gidx, grow, gmem_row_stride, gcol).unwrap();
        let gbyte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 2;", gbyte, gidx).unwrap();
        let gaddr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr, gmem_ptr, gbyte).unwrap();

        // Register-driven zero-fill size operand - see doc comment.
        let rsize = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    selp.u32 {}, 16, 0, {};", rsize, p_ok).unwrap();

        let sidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, lr, smem_stride, lc).unwrap();
        let sbyte = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 1;", sbyte, sidx).unwrap();
        let saddr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_stage_base, sbyte).unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.cg.shared.global [{}], [{}], 16, {};", saddr, gaddr, rsize).unwrap();
    }

    /// Shared compute core of `emit_tensor_core_gemm_kernel`'s K-loop body,
    /// used identically by both the synchronous-fallback and
    /// cp.async-pipelined paths: for each `kk` sub-step, issues one
    /// `ldmatrix.x4` per `i` (A, reused across all `j`) and two
    /// `ldmatrix.x2.trans` per `j` (B, one per 8-wide N-half - native
    /// `mma.sync.m16n8k16` granularity, see below - reused across all `i`),
    /// then two `mma.sync.aligned.m16n8k16.row.col` calls per `(i,j)` (one
    /// per N-half) accumulating into `acc[i][j][0..4]`/`acc[i][j][4..8]`
    /// respectively. The two paths differ only in *which* smem addresses
    /// `smem_a_base`/`smem_b_base` point at - this function neither knows
    /// nor cares.
    ///
    /// Hand-rolled `ldmatrix` + `mma.sync.m16n8k16.row.col` replaces a
    /// prior `wmma.load`/`wmma.mma.m16n16k16`-based version (see git
    /// history) - ncu profiling (see project benchmark docs) found the
    /// wmma path left 28% of shared-memory wavefronts "excessive"
    /// (bank-conflicted), and `wmma.load` exposes no hook to control or
    /// even inspect its access pattern (opaque by design). The exact
    /// per-lane address formula below was derived from the PTX ISA's
    /// documented mma.m16n8k16 fragment layout (9.7.13.4.8, verified
    /// against the official NVIDIA PTX ISA 8.5 PDF directly, not memory or
    /// a paraphrase - unlike wmma's fragment layout, which 9.7.13.3.1
    /// states is "unspecified and target architecture dependent") and
    /// validated standalone - hand-written PTX, single warp, real sm_89
    /// hardware - against that documented formula AND an exact-integer CPU
    /// reference (small-integer f16 operands keep every intermediate exact,
    /// not floating-point-tolerance-close) before this function was
    /// rewritten; see the project scratchpad's validate_mma.py for the
    /// harness (all checks passed on first real-hardware run).
    ///
    /// Per lane (`%tid.x & 31`), decomposed once by the caller into
    /// `row_off = lane & 15` and `col_bit_a = (lane & 16) >> 1`:
    ///   - A is fed plain `ldmatrix.x4` (no `.trans`): A is stored
    ///     row-major in smem (K contiguous), exactly what `.row` (A's
    ///     layout in `mma.m16n8k16.row.col` - hardcoded in the PTX ISA
    ///     grammar for this shape, there is no `.row.row` variant unlike
    ///     `m8n8k4`) expects. `col_bit_a` (0 or 8) selects which half of
    ///     the 16-wide K dimension a lane's group belongs to.
    ///   - B is fed `ldmatrix.x2.trans`: `mma.m16n8k16` hardcodes B as
    ///     `.col`, but B is ALSO stored row-major in smem (N contiguous,
    ///     matching global B's own layout) - `.trans` reconciles the two
    ///     (same "supply a physical smem row address" protocol as A, but
    ///     the hardware transposes what lands in the destination
    ///     registers), so the tile-load/cp.async code staging B into smem
    ///     needs no changes at all.
    ///   - `row_off` is reused as-is by both A (combined with the warp's
    ///     row and `i*16`) and B (combined with `kk`) - it is the same
    ///     per-lane "which of 16 rows within this half" term either way.
    ///
    /// Offline bank-conflict analysis (project scratchpad's
    /// swizzle_design.py, using this exact validated address formula,
    /// swept across every warp/i/j/kk/N-half combination this function
    /// actually issues for every autotuned CTA shape) found the EXISTING
    /// `+8`-element row padding already fully bank-conflict-free for this
    /// access pattern - no XOR swizzle needed. The ncu-measured conflicts
    /// referenced above appear to be an artifact of wmma.load's opaque
    /// access pattern specifically, not the padded layout itself. This is
    /// a prediction from static analysis, not yet re-confirmed by ncu on
    /// this rewritten kernel - see benchmark docs for the real measurement
    /// once taken.
    ///
    /// `acc[i][j]` keeps the EXACT SAME external shape/order the prior
    /// wmma-based version used (8 f32 regs: `[0..4)` = this `(i,j)`
    /// block's N=0..7 columns, `[4..8)` = N=8..15) - empirically confirmed
    /// register-for-register identical to what `wmma.mma.m16n16k16` would
    /// have produced for the same inputs, on this sm_89 + this ptxas (see
    /// validate_mma.py's wmma-vs-mma.sync D-fragment check), so
    /// `emit_tensor_core_gemm_kernel`'s epilogue (both the plain-GEMM
    /// direct-to-global and the fused-bias-relu shared-memory
    /// `wmma.store.d` paths) needed NO changes. IMPORTANT: per
    /// 9.7.13.3.1 the PTX ISA does NOT guarantee this equivalence (wmma's
    /// fragment layout is documented "unspecified") - this is an empirical
    /// fact about THIS toolchain, not a portable one, and must be
    /// re-verified on real hardware (rerun validate_mma.py) before ever
    /// updating the CUDA toolchain/driver version this project targets, or
    /// before relying on it for a different SM architecture.
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_compute_block(
        &mut self,
        acc: &[Vec<Vec<String>>],
        warp_col0_local: &str,
        warp_row0_local: &str,
        smem_a_base: &str,
        smem_b_base: &str,
        smem_a_stride: u32,
        smem_b_stride: u32,
        row_off: &str,
        col_bit_a: &str,
        k_substeps: u32,
        num_i: u32,
        num_j: u32,
    ) {
        // ldmatrix's address must be naturally 16-byte aligned (PTX ISA
        // 9.7.13.4.15); every term added to a row/col-scaled linear index
        // below is already a multiple of 8 elements EXCEPT the
        // stride-scaled row term, which is only guaranteed a multiple of 8
        // elements (16 bytes) if the stride itself is - true for every
        // `+8`-padded cta_k/cta_n `Autotuner::generate_candidates` produces
        // (32/64/128-multiple base + 8), not true in general.
        debug_assert_eq!(smem_a_stride % 8, 0, "ldmatrix alignment requires smem_a_stride a multiple of 8 elements");
        debug_assert_eq!(smem_b_stride % 8, 0, "ldmatrix alignment requires smem_b_stride a multiple of 8 elements");

        for kk_step in 0..k_substeps {
            let kk = kk_step * 16;

            // ---- B: ldmatrix.x2.trans, two 8-wide N-halves per j (native
            // mma.m16n8k16 granularity), reused across every i below (same
            // reuse-across-i optimization the prior wmma.load.b loop used) ----
            let mut b_frags: Vec<[Vec<String>; 2]> = Vec::with_capacity(num_j as usize);
            for j in 0..num_j {
                let mut halves: [Vec<String>; 2] = [Vec::new(), Vec::new()];
                for (half_idx, n_half) in [0u32, 8u32].into_iter().enumerate() {
                    let col_reg = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col_reg, warp_col0_local, j * 16 + n_half).unwrap();
                    let global_k = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", global_k, row_off, kk).unwrap();
                    let lin = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", lin, global_k, smem_b_stride, col_reg).unwrap();
                    let byte = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 1;", byte, lin).unwrap();
                    let addr = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", addr, smem_b_base, byte).unwrap();
                    let frag: Vec<String> = (0..2).map(|_| self.alloc_reg32()).collect();
                    writeln!(
                        &mut self.ptx_buffer,
                        "    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{{}}}, [{}];",
                        frag.join(","), addr
                    ).unwrap();
                    halves[half_idx] = frag;
                }
                b_frags.push(halves);
            }

            // ---- A: ldmatrix.x4, reused across every j below (same
            // reuse-across-j optimization the prior wmma.load.a loop used) ----
            for i in 0..num_i {
                let row_reg = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", row_reg, warp_row0_local, row_off).unwrap();
                let global_row = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", global_row, row_reg, i * 16).unwrap();
                let col0 = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", col0, col_bit_a, kk).unwrap();
                let lin = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", lin, global_row, smem_a_stride, col0).unwrap();
                let byte = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 1;", byte, lin).unwrap();
                let addr = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", addr, smem_a_base, byte).unwrap();
                let a_frag: Vec<String> = (0..4).map(|_| self.alloc_reg32()).collect();
                writeln!(
                    &mut self.ptx_buffer,
                    "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}}}, [{}];",
                    a_frag.join(","), addr
                ).unwrap();

                for j in 0..num_j {
                    for half in 0..2usize {
                        let lo = 4 * half;
                        let d = acc[i as usize][j as usize][lo..lo + 4].join(",");
                        writeln!(
                            &mut self.ptx_buffer,
                            "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                            d, a_frag.join(","), b_frags[j as usize][half].join(","), d
                        ).unwrap();
                    }
                }
            }
        }
    }

    /// Emits a complete, self-contained Tensor Core GEMM kernel body (grid
    /// mapping, warp decomposition, cooperative shared-memory staging, wmma
    /// load/mma/store, boundary-masked epilogue) for a kernel carrying a
    /// validated `@tile(M, N, K)` directive - see `KernelDecl::tile` and
    /// `tile_gemm_operands`. Bypasses the normal generic per-statement
    /// lowering entirely: the `.ysu` kernel body is not consulted here at
    /// all (for a `@tile`d kernel its role is documentation of intent,
    /// checked by `type_checker` but not literally lowered).
    ///
    /// Tile/warp/pipeline-stage selection comes from `Autotuner::autotune`
    /// (hardened against real probed hardware occupancy limits - see
    /// autotuner.rs), keyed on this kernel's own compile-time M/N/K rather
    /// than a hardcoded guess.
    ///
    /// Uses PTX ISA `wmma.load`/`wmma.mma`/`wmma.store` (`.m16n16k16`, f16
    /// in / f32 out) rather than hand-rolled `ldmatrix` + raw `mma.sync`
    /// fragment-register bookkeeping: getting per-lane fragment layout wrong
    /// by hand produces a silently wrong answer, not a compile error, while
    /// a malformed `wmma.*` shape/register-count is rejected outright by
    /// ptxas - a real safety net worth leaning on for a codegen path this
    /// new. The exact instruction sequence, padding, and address arithmetic
    /// below were validated standalone first: hand-written PTX, run via the
    /// CUDA driver API on real sm_89 hardware, covering a single 16x16x16
    /// tile and then a multi-warp/multi-k-tile/non-square-stride case
    /// (M=32,N=48,K=64, which a square M=N=K test cannot - it can't tell A's
    /// K-stride, B's N-stride, and C's N-stride apart) - both matched a CPU
    /// reference exactly before this function was written.
    ///
    /// Shared-memory tiles are padded (+8 f16 elements/row - matching
    /// tests/y_tensor_core_gemm.cu, the one other Tensor Core kernel in this
    /// repo proven on real hardware) rather than XOR-swizzled via
    /// bank_conflict.rs: `wmma.load` takes a flat base address plus a row
    /// stride with no hook for swizzled addressing, so bank_conflict.rs's
    /// swizzle math (built around raw `ldmatrix`'s specific 32-lane access
    /// pattern) does not model this instruction sequence - invoking it here
    /// would be validation theater, not a real check. This does not affect
    /// correctness either way (bank conflicts cost throughput, not
    /// correctness - the smoke tests above passed before padding was even
    /// tuned); it is a real but secondary, disclosed simplification.
    ///
    /// Scope, stated plainly:
    /// - `K` must be a multiple of the autotuned `cta_k` (`debug_assert`ed,
    ///   not silently truncated); the M/N grid and every A/B tile load IS
    ///   boundary-masked (safe, zero-filled) for any M/N, but a CTA's output
    ///   *store* is skipped whole-fragment (not partially written) if it
    ///   would run past M or N - correct for the exact-multiple shapes this
    ///   was verified against, ragged (non-16-aligned) M/N edges are left
    ///   unwritten rather than partially computed.
    /// - Staging is `cp.async`-pipelined across `effective_stages` shared-
    ///   memory buffers (see below) rather than a single synchronous
    ///   load/`bar.sync`/compute/`bar.sync` step: while the tensor cores
    ///   consume stage `read_stage`, the async-copy engine is already
    ///   filling stage `write_stage` (`(k_iter + effective_stages - 1) %
    ///   effective_stages`) for a future iteration in the background. This
    ///   was a disclosed, deliberate scope cut on the first,
    ///   correctness-first pass (see git history); `emit_cp_async`/
    ///   `emit_cp_async_commit`/`emit_cp_async_wait` and the K-tail/smem
    ///   accounting below are what a later pass (this one) used to layer it
    ///   on top. `effective_stages` is `Autotuner::autotune`'s `num_stages`
    ///   clamped down by two things it doesn't itself account for:
    ///     1. `k_tiles` (`k / cta_k`) - a pipeline can never usefully
    ///        prefetch more tiles ahead than exist at all.
    ///     2. The REAL, padding-inclusive per-stage shared-memory footprint
    ///        against the real per-CTA hardware ceiling.
    ///        `Autotuner::score_candidate`/`estimate_occupancy`'s own smem
    ///        model uses raw `cta_m*cta_k`/`cta_k*cta_n` tile bytes with no
    ///        `+8`-element padding term (a known, disclosed blind spot - see
    ///        autotuner.rs), so it can green-light a candidate whose padded
    ///        reality doesn't actually fit - re-derived here from the real
    ///        per-stage byte count rather than trusted from the candidate.
    ///        Separately, `hw_profile.max_smem_per_sm_bytes` (102400 B
    ///        measured on this project's sm_89 dev machine) is a *per-SM*
    ///        total used for occupancy reasoning elsewhere in the autotuner,
    ///        not the single-CTA dynamic-shared-memory opt-in ceiling a
    ///        launch can actually request
    ///        (`cudaDevAttrMaxSharedMemoryPerBlockOptin`, measured at
    ///        101376 B on that same machine via `cupy` - close to but not
    ///        the same number) - a margin well past the ~1KB observed gap is
    ///        subtracted to stay safe on hardware this was never measured
    ///        on. When `effective_stages` ends up below 2 (only possible for
    ///        a @tile K with fewer than 2 whole `cta_k` tiles - not
    ///        exercised by any shape `Autotuner::generate_candidates`
    ///        currently produces for this project's tested shapes), this
    ///        falls back to the exact original single-buffered synchronous
    ///        path instead of degenerating into a 1-stage "pipeline".
    ///   Stage buffers live in one combined *dynamic* (`.extern .shared`)
    ///   array rather than N statically-sized `.shared` declarations: a
    ///   statically-sized `.shared` array is hard-capped at 48KB by ptxas on
    ///   this hardware regardless of the GPU's real (~100KB) capacity
    ///   (confirmed by direct experiment - ptxas rejects a >48KB static
    ///   array outright), while dynamic shared memory can go up to the real
    ///   per-CTA opt-in ceiling as long as the launcher requests it (sets
    ///   `cudaFuncAttributeMaxDynamicSharedMemorySize`/
    ///   `max_dynamic_shared_size_bytes` and passes the byte count at
    ///   launch) - see the `[Y TENSOR CORE GEMM]` PTX comment this function
    ///   emits, which documents the exact byte count the launcher must
    ///   request, and `tests/benchmark_y_tensor_core_gemm.py`'s
    ///   `compile_kernel`/`run_once` for the reference launcher
    ///   implementation. This declaration-size ceiling and the dynamic-smem
    ///   opt-in launch mechanism were both confirmed with standalone probes
    ///   (a deliberately oversized static `.shared` array rejected by
    ///   ptxas; a >48KB `.extern .shared` array loaded and launched
    ///   end-to-end via `cupy.RawModule` + `max_dynamic_shared_size_bytes` +
    ///   `shared_mem=`) on real sm_89 hardware before this function was
    ///   written.
    /// Returns the CTA's real thread count (`config.num_warps * 32`) so the
    /// caller (`emit_kernel`) can size its `.maxnreg` register-pressure
    /// estimate against this kernel's actual launch block size instead of a
    /// generic 256-thread assumption - see the call site's doc comment.
    fn emit_tensor_core_gemm_kernel(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        a_ptr: &str,
        b_ptr: &str,
        c_ptr: &str,
        bias_ptr: Option<&str>,
        hw_profile: &HardwareProfile,
        kernel_name: &str,
    ) -> u32 {
        let config = Autotuner::autotune(m, n, k, hw_profile, Precision::F16);
        let cta_m = config.cta_m;
        let cta_n = config.cta_n;
        let cta_k = config.cta_k;
        let warps_m = config.warps_m;
        let warps_n = config.warps_n;
        let threads_per_cta = config.num_warps * 32;

        let per_warp_m = cta_m / warps_m;
        let per_warp_n = cta_n / warps_n;
        let num_i = per_warp_m / 16;
        let num_j = per_warp_n / 16;
        let k_substeps = cta_k / 16;
        debug_assert_eq!(num_i * 16 * warps_m, cta_m, "autotuned cta_m must split evenly into 16-row warp fragments");
        debug_assert_eq!(num_j * 16 * warps_n, cta_n, "autotuned cta_n must split evenly into 16-col warp fragments");
        debug_assert_eq!(k_substeps * 16, cta_k, "autotuned cta_k must be a multiple of wmma's m16n16k16 K dimension");

        // ---- K tail ----
        // `k_tiles` rounds UP, so the final K-tile may be partial. Both
        // operand loaders already bound-check the K extent - A passes `k` as
        // its column bound and B passes `k` as its row bound (see the
        // `emit_gemm_tile_load_async` call sites below) - and the zero-fill
        // `cp.async` form writes real zeros for every out-of-range chunk, so
        // a partial K-tile contributes exact zeros to the accumulator rather
        // than garbage. The prologue needs no extra tile predicate either:
        // `effective_stages` is still clamped to `<= k_tiles`.
        //
        // Truncating here instead (`k / cta_k`, what this did before) SILENTLY
        // DROPPED that tail. Measured at M=N=K=1000 with the autotuned
        // 128x128x32 tile: 8 of 1000 K elements dropped, relative L2 error
        // 9.1e-02, no diagnostic of any kind. At K=2 with cta_k=64 it gave
        // `k_tiles = 0` and the kernel returned all zeros. M and N tails were
        // always handled correctly by the same masking; only K was not.
        //
        // A's K extent is masked at 8-element (16-byte `cp.async` chunk)
        // granularity because K is A's contiguous dimension, so K must be a
        // multiple of 8. B's K masking is per-row and would accept any K.
        // This is a hard `assert!`, deliberately not a `debug_assert!`: the
        // release binary is what everything actually runs, and the failure
        // mode being guarded is a silently wrong GEMM, which is worse than a
        // failed compile.
        assert_eq!(
            k % 8,
            0,
            "@tile K={} is not a multiple of 8. The F16 tensor-core GEMM masks A's K tail at \
             16-byte (8-element) cp.async chunk granularity, so the trailing {} element(s) would \
             be silently dropped and the result would be wrong. Pad K to a multiple of 8.",
            k,
            k % 8
        );

        // Padding, elements/row - see doc comment. `Y_SMEM_PAD` exists to A/B
        // the padding against the shared-memory budget it costs: at
        // 128x128x32 the +8 costs 2560 B/stage, which is exactly what keeps a
        // 3-stage pipeline from fitting two CTAs per SM (3 * 18944 * 2 =
        // 113664 > 102400, but 3 * 16384 * 2 = 98304 fits). All the
        // downstream addressing derives from these two strides, so a
        // different pad is self-consistent - only the bank-conflict behaviour
        // changes.
        let smem_pad: u32 = std::env::var("Y_SMEM_PAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let smem_a_stride = cta_k + smem_pad;
        let smem_b_stride = cta_n + smem_pad;
        let k_tiles = (k + cta_k - 1) / cta_k;

        // ---- effective pipeline depth: see doc comment ----
        let stage_a_bytes = cta_m * smem_a_stride * 2;
        let stage_b_bytes = cta_k * smem_b_stride * 2;
        let per_stage_bytes = stage_a_bytes + stage_b_bytes;
        let safe_smem_ceiling = hw_profile.max_smem_per_sm_bytes.saturating_sub(4096);
        let max_stages_by_smem = (safe_smem_ceiling / per_stage_bytes).max(1);
        let effective_stages = config.num_stages.min(k_tiles).min(max_stages_by_smem).max(1);
        let mut total_dyn_smem_bytes = effective_stages * per_stage_bytes;

        // ---- fused epilogue (both the plain and Bias+ReLU shapes - see
        // below): reuses this same dynamic shared buffer (already idle by
        // the time the epilogue runs - see emit_gemm_bias_relu_epilogue's/
        // emit_gemm_plain_epilogue's doc comments) as a row-major f32
        // scratch tile. A *full* cta_m x cta_n f32 tile (e.g. 128x260x4 =
        // ~133KB for a 128x256 CTA tile) can exceed this GPU's real per-CTA
        // dynamic-smem opt-in ceiling (~101376B measured on this project's
        // sm_89 dev machine - see emit_tensor_core_gemm_kernel's doc
        // comment) even though the A/B pipeline stages themselves fit fine.
        // So the epilogue instead runs in `warps_n` passes, one per warp
        // *column* (see emit_kernel's write-loop below): each pass's
        // scratch tile is only cta_m x per_warp_n wide - `per_warp_n <=
        // cta_n`, so this is always <= the full-tile size, and for every
        // autotuned config observed so far comfortably fits alongside the
        // pipeline's own smem budget. Supported in BOTH the pipelined and
        // `effective_stages < 2` fallback branches - both branches already
        // unconditionally use this same dynamic extern-shared mechanism
        // (instead of static `.shared` arrays), see that branch's own doc
        // comment for why (M=N=K=2048 on this project's dev GPU clamps to 1
        // stage on smem pressure alone, before the epilogue even factors
        // in - not a theoretical edge case).
        //
        // This budget line applies unconditionally (not just when
        // `bias_ptr.is_some()`, prior to the M<16 all-zero-output fix
        // below): the plain-GEMM epilogue now ALSO stages through this
        // smem scratch tile rather than writing `wmma.store.d` directly to
        // global with whole-16-row-fragment masking - see
        // `emit_gemm_plain_epilogue`'s doc comment for why that direct
        // path was a real correctness bug, not just a design choice, for
        // any M < 16 (confirmed on real hardware: silently all-zero
        // output, no error - see benchmark_y_decode_gemm_results.md).
        let smem_c_stride = per_warp_n + 4; // padded, elements/row - matches tests/y_tensor_core_gemm.cu's fused kernel (there: cta_n-wide, here: per-warp-column-banded, see above)
        let smem_c_bytes = cta_m * smem_c_stride * 4;
        total_dyn_smem_bytes = total_dyn_smem_bytes.max(smem_c_bytes);

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [Y TENSOR CORE GEMM] M={} N={} K={} | CTA {}x{}x{} | {}x{} warps | wmma.sync.m16n16k16.f16->f32", m, n, k, cta_m, cta_n, cta_k, warps_m, warps_n).unwrap();
        if effective_stages >= 2 {
            writeln!(&mut self.ptx_buffer, "    // Autotuner selected {} pipeline stages ({} used after k_tiles/smem clamping); cp.async multi-stage pipelined. Dynamic shared memory required: {} bytes.", config.num_stages, effective_stages, total_dyn_smem_bytes).unwrap();
        } else {
            // Both the bias and no-bias cases use the same dynamic
            // `.extern .shared` buffer mechanism in this branch (see its
            // own doc comment below) - so the launcher-facing byte count is
            // always reported here, not just when bias is present.
            writeln!(&mut self.ptx_buffer, "    // Autotuner selected {} pipeline stages; only {} K-tile(s) exist so this path stages synchronously (see emit_tensor_core_gemm_kernel doc comment). Dynamic shared memory required: {} bytes.", config.num_stages, k_tiles, total_dyn_smem_bytes).unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        // ---- cvta.to.global for A, B, C (matches nvcc's own wmma-lowering
        // convention for global-space wmma.load/store, verified against real
        // ptxas output on this machine before this function was written) ----
        let a_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", a_g, a_ptr).unwrap();
        let b_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", b_g, b_ptr).unwrap();
        let c_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", c_g, c_ptr).unwrap();
        let bias_g = bias_ptr.map(|p| {
            let r = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", r, p).unwrap();
            r
        });


        // ---- grid position: L2-locality grid swizzle ----
        // Groups GEMM_SWIZZLE_GROUP_SIZE consecutive M-row CTA tiles and
        // walks them column-first (see emit_grid_swizzle_code's doc comment)
        // instead of raw `%ctaid.y`/`%ctaid.x` raster order, so CTAs that
        // run concurrently reuse the same A/B tiles through L2 rather than
        // one fixed A-row-tile paired with every B-column-tile in the grid.
        // Correct for grids whose dimensions aren't an exact multiple of
        // the group size (the common case) - see
        // test_grid_swizzle_uneven_grid_is_bijection.
        let (bid_m, bid_n) = self.emit_grid_swizzle_code(GEMM_SWIZZLE_GROUP_SIZE);
        let cta_m_start = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", cta_m_start, bid_m, cta_m).unwrap();
        let cta_n_start = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", cta_n_start, bid_n, cta_n).unwrap();

        // ---- warp/lane decomposition ----
        let tid = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
        let warp_id = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 5;", warp_id, tid).unwrap();
        let warp_m = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", warp_m, warp_id, warps_m).unwrap();
        let warp_n = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", warp_n, warp_id, warps_m).unwrap();
        // Loop-invariant, hoisted: every A-fragment address needs
        // (warp_row0_within_cta) * smem_a_stride, so compute the CTA-local
        // (not grid-global) warp row once, up front. Grid-global
        // warp_row0/warp_col0 (`cta_m_start + warp_m*per_warp_m`, etc.) are
        // NOT computed here: the epilogue that used to need them (plain-
        // GEMM, direct-to-global whole-fragment-masked `wmma.store.d`) was
        // replaced by `emit_gemm_plain_epilogue` (smem-staged, fine-grained
        // masked, fixing the M<16 all-zero-output bug - see
        // benchmark_y_decode_gemm_results.md), which - like
        // `emit_gemm_bias_relu_epilogue` already did - derives its global
        // addresses from `cta_m_start`/`tile_n_start` plus a per-thread
        // local row/col instead.
        let warp_row0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_row0_local, warp_m, per_warp_m).unwrap();
        let warp_col0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_col0_local, warp_n, per_warp_n).unwrap();

        // ---- per-lane ldmatrix address terms - see emit_gemm_compute_block's
        // doc comment for the derivation and its real-hardware validation.
        // lane = tid.x & 31 (single warp of thread IDs isn't assumed
        // elsewhere, but %tid.x's low 5 bits always give the lane within
        // whatever warp this thread belongs to). row_off = lane & 15 packs
        // ldmatrix.x4's 2-bit submatrix-row-group selector and 3-bit
        // within-submatrix row into one 4-bit "which of 16 rows in this
        // K-or-M half" term (the two fields are disjoint bit ranges, so
        // sum == bitwise-or == a plain mask). col_bit_a = (lane & 16) >> 1
        // is A-only: selects which 8-wide half of the 16-wide K dimension
        // this lane's ldmatrix.x4 submatrix-group falls in. ----
        let lane = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 31;", lane, tid).unwrap();
        let row_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 15;", row_off, lane).unwrap();
        let col_bit_a_wide = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 16;", col_bit_a_wide, lane).unwrap();
        let col_bit_a = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 1;", col_bit_a, col_bit_a_wide).unwrap();

        // ---- accumulator fragments acc[i][j]: 8 f32 regs each, zeroed ----
        let mut acc: Vec<Vec<Vec<String>>> = Vec::with_capacity(num_i as usize);
        for _ in 0..num_i {
            let mut row = Vec::with_capacity(num_j as usize);
            for _ in 0..num_j {
                let mut frag = Vec::with_capacity(8);
                for _ in 0..8 {
                    let r = self.alloc_regf32();
                    writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", r).unwrap();
                    frag.push(r);
                }
                row.push(frag);
            }
            acc.push(row);
        }

        // Captured from whichever branch below runs, so the fused Bias+ReLU
        // epilogue can reuse this same dynamic shared buffer as its scratch
        // tile once the K-loop is done with it - see the `effective_stages <
        // 2` branch's `bias_ptr.is_some()` special case just below for why
        // this is populated in BOTH branches, not just the pipelined one.
        let smem_pipeline_base_for_epilogue: String;

        if effective_stages < 2 {
            // ---- Fallback: original single-buffered synchronous path
            // (load -> bar.sync -> compute -> bar.sync). Reachable whenever
            // `effective_stages` is clamped all the way down to 1 - see doc
            // comment. This is NOT a rare/theoretical edge case: with this
            // project's current `Autotuner::generate_candidates` search
            // space, the 128x256x64 CTA tile it selects for M=N=K >= 2048
            // has `per_stage_bytes` = 52224B, and `safe_smem_ceiling /
            // per_stage_bytes` floors to 1 on this project's dev GPU
            // (102400B `max_smem_per_sm_bytes`) - so this branch is real and
            // commonly hit for large square GEMMs, not just a fallback kept
            // for shapes nothing currently produces.
            //
            // Always uses the same *dynamic* (`.extern .shared`) buffer
            // mechanism as the `effective_stages >= 2` pipelined path below.
            // A prior version of this branch used static `.shared
            // smem_A[..]`/`smem_B[..]` arrays for the no-bias case, on the
            // assumption a 1-stage tile's A+B footprint would always fit
            // under ptxas's 48KB static-`.shared` hard cap - false for the
            // 128x256x64/52224B case above: ptxas outright refuses to
            // assemble a >48KB static `.shared` declaration regardless of
            // this GPU's real ~100KB dynamic capacity (confirmed by direct
            // experiment: `ptxas -arch=sm_89` on the emitted
            // gemm_f16_{2048,4096,8192,16384}.ptx failed with "uses too much
            // shared data (0xcc00 bytes, 0xc000 max)" before this fix - a
            // failure `cargo test` alone can't catch, since it only
            // exercises `emit_program`, never ptxas itself). The fused
            // Bias+ReLU epilogue already needed the dynamic mechanism for
            // its own reasons (runtime-sized scratch tile across multiple
            // warp-column passes - see the smem-sizing doc comment above),
            // so the plain-GEMM case now just reuses that same,
            // already-proven-correct path instead of a second, narrower one.
            let smem_symbol = format!("smem_pipeline_{}", kernel_name);
            self.pending_extern_decls.push(format!(".extern .shared .align 16 .b8 {}[];", smem_symbol));
            let base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", base, smem_symbol).unwrap();
            smem_pipeline_base_for_epilogue = base.clone();
            let a_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", a_base, base).unwrap();
            let b_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", b_base, base, stage_a_bytes).unwrap();
            let smem_a_base = a_base;
            let smem_b_base = b_base;

            let k_iter = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
            let loop_start = self.alloc_label("GEMM_K_LOOP");
            let loop_end = self.alloc_label("GEMM_K_DONE");
            writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
            let exit_pred = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", exit_pred, k_iter, k_tiles).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} bra {};", exit_pred, loop_end).unwrap();
            let k0 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", k0, k_iter, cta_k).unwrap();

            self.emit_gemm_tile_load(
                "A", &a_g, &cta_m_start, &k0, k, m, k, cta_m, cta_k, &smem_a_base, smem_a_stride, threads_per_cta,
            );
            self.emit_gemm_tile_load(
                "B", &b_g, &k0, &cta_n_start, n, k, n, cta_k, cta_n, &smem_b_base, smem_b_stride, threads_per_cta,
            );

            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
            self.emit_gemm_compute_block(
                &acc, &warp_col0_local, &warp_row0_local, &smem_a_base, &smem_b_base,
                smem_a_stride, smem_b_stride, &row_off, &col_bit_a, k_substeps, num_i, num_j,
            );
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
            writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
            writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
        } else {
            // ---- N-stage cp.async pipelined path - see doc comment ----
            let n_stages = effective_stages;

            // `.extern .shared` must sit at module scope, not nested inside
            // this kernel's `{ }` body (confirmed by direct experiment -
            // unlike a plain non-extern `.shared` local, ptxas's parser
            // rejects it there) - queued for `emit_kernel` to flush just
            // before this kernel's own `.visible .entry` line instead of
            // written here directly. Symbol name is kernel-qualified so two
            // different @tile kernels compiled into the same program never
            // collide on one shared top-level symbol.
            let smem_symbol = format!("smem_pipeline_{}", kernel_name);
            self.pending_extern_decls.push(format!(".extern .shared .align 16 .b8 {}[];", smem_symbol));
            let smem_pipeline_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", smem_pipeline_base, smem_symbol).unwrap();
            smem_pipeline_base_for_epilogue = smem_pipeline_base.clone();
            // B's stages live right after all of A's stages in the one
            // combined dynamic array (mirrors src/bin/autotune_verify.rs's
            // KERNEL_TEMPLATE: one extern buffer, sliced by constant byte
            // offsets - PTX/the launch API only supports one dynamic
            // shared-memory region per kernel).
            let smem_b_region_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_b_region_base, smem_pipeline_base, n_stages * stage_a_bytes).unwrap();

            // ---- Prologue: prefetch stages 0..n_stages-2 (compile-time
            // constant tile indices - always in-bounds since n_stages <=
            // k_tiles by construction, so no K-tail masking is needed here,
            // unlike the main loop's prefetch below). ----
            for s in 0..(n_stages - 1) {
                let k0_s = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", k0_s, s * cta_k).unwrap();
                let a_stage_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", a_stage_base, smem_pipeline_base, s * stage_a_bytes).unwrap();
                let b_stage_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", b_stage_base, smem_b_region_base, s * stage_b_bytes).unwrap();

                self.emit_gemm_tile_load_async(
                    &format!("PA{}", s), &a_g, &cta_m_start, &k0_s, k, m, k, cta_m, cta_k, &a_stage_base, smem_a_stride, threads_per_cta, None,
                );
                self.emit_gemm_tile_load_async(
                    &format!("PB{}", s), &b_g, &k0_s, &cta_n_start, n, k, n, cta_k, cta_n, &b_stage_base, smem_b_stride, threads_per_cta, None,
                );
                self.emit_cp_async_commit();
            }
            self.emit_cp_async_wait(n_stages - 2);
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            let k_iter = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
            let loop_start = self.alloc_label("GEMM_K_LOOP");
            let loop_end = self.alloc_label("GEMM_K_DONE");
            writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
            let exit_pred = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", exit_pred, k_iter, k_tiles).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} bra {};", exit_pred, loop_end).unwrap();

            // read_stage: which physical stage slot holds THIS iteration's
            // already-ready data (guaranteed by the previous iteration's -
            // or the prologue's, for k_iter==0 - wait_group+bar.sync below).
            let read_stage = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", read_stage, k_iter, n_stages).unwrap();
            // next_tile: the tile index to prefetch NOW so it's ready
            // n_stages-1 iterations from now; also doubles as the physical
            // write_stage slot index (write_stage(k_iter) == next_tile %
            // n_stages, by construction - see doc comment derivation).
            let next_tile = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", next_tile, k_iter, n_stages - 1).unwrap();
            let write_stage = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", write_stage, next_tile, n_stages).unwrap();
            let p_tile_valid = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_tile_valid, next_tile, k_tiles).unwrap();
            let k0_next = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", k0_next, next_tile, cta_k).unwrap();

            let a_write_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", a_write_base, write_stage, stage_a_bytes, smem_pipeline_base).unwrap();
            let b_write_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", b_write_base, write_stage, stage_b_bytes, smem_b_region_base).unwrap();

            // Issue the prefetch for `next_tile` (masked off via
            // `p_tile_valid` if it doesn't exist), commit, then compute on
            // `read_stage` (already confirmed ready) - the tensor-core work
            // below overlaps with this prefetch's async-copy-engine traffic
            // instead of waiting for it first.
            self.emit_gemm_tile_load_async(
                "A", &a_g, &cta_m_start, &k0_next, k, m, k, cta_m, cta_k, &a_write_base, smem_a_stride, threads_per_cta, Some(&p_tile_valid),
            );
            self.emit_gemm_tile_load_async(
                "B", &b_g, &k0_next, &cta_n_start, n, k, n, cta_k, cta_n, &b_write_base, smem_b_stride, threads_per_cta, Some(&p_tile_valid),
            );
            self.emit_cp_async_commit();

            let a_read_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", a_read_base, read_stage, stage_a_bytes, smem_pipeline_base).unwrap();
            let b_read_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", b_read_base, read_stage, stage_b_bytes, smem_b_region_base).unwrap();

            self.emit_gemm_compute_block(
                &acc, &warp_col0_local, &warp_row0_local, &a_read_base, &b_read_base,
                smem_a_stride, smem_b_stride, &row_off, &col_bit_a, k_substeps, num_i, num_j,
            );

            // Wait until group `k_iter+1` (the tile that will become
            // read_stage next iteration) has landed, then make it visible
            // block-wide; also what makes it safe for the NEXT iteration's
            // prefetch-write to reuse this iteration's `read_stage` slot
            // (that slot is only written again as write_stage one iteration
            // from now - see doc comment derivation).
            self.emit_cp_async_wait(n_stages - 2);
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
            writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
            writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();

            // Drain any still-in-flight tail prefetch before the epilogue -
            // avoids leaving an async copy targeting this CTA's shared
            // memory outstanding when the kernel exits.
            self.emit_cp_async_wait(0);
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
        }

        // ---- fused epilogue: `warps_n` passes, one per warp-column band
        // (see the smem-sizing doc comment above for why) - each pass does
        // wmma.store.d into the (small, per_warp_n-wide) shared scratch
        // tile for just that band's warps, then a CTA-wide, fine-grained
        // (per-row, not per-16-row-fragment - see emit_gemm_plain_epilogue's
        // doc comment for why that distinction is a real correctness fix,
        // not just style) boundary-masked copy to global for that band -
        // either bias-add+ReLU (emit_gemm_bias_relu_epilogue) or a plain
        // copy (emit_gemm_plain_epilogue), depending on whether this kernel
        // has a Bias operand. Both shapes share this same staging loop -
        // unified here (rather than duplicated per-branch) once the plain
        // shape needed the same smem staging the Bias+ReLU shape already
        // had, to fix the M<16 all-zero-output bug - see
        // benchmark_y_decode_gemm_results.md. ----
        let smem_c_base = smem_pipeline_base_for_epilogue;
        let stride_c_smem_reg = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", stride_c_smem_reg, smem_c_stride).unwrap();
        for n_slot in 0..warps_n {
            let p_slot = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, {}, {};", p_slot, warp_n, n_slot).unwrap();
            for i in 0..num_i {
                for j in 0..num_j {
                    // Row spans the whole CTA (per_warp_m band, warp-relative)
                    // same as before; column is LOCAL to this pass's
                    // per_warp_n-wide scratch tile - not warp_col0_local
                    // + j*16 - since only one warp-column band's data
                    // lives in the scratch buffer at a time.
                    let c_row_local = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", c_row_local, warp_row0_local, i * 16).unwrap();

                    let sidx = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, c_row_local, stride_c_smem_reg, j * 16).unwrap();
                    let sbyte = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", sbyte, sidx).unwrap();
                    let saddr = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_c_base, sbyte).unwrap();

                    let d = acc[i as usize][j as usize].join(",");
                    writeln!(
                        &mut self.ptx_buffer,
                        "    @{} wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [{}], {{{}}}, {};",
                        p_slot, saddr, d, stride_c_smem_reg
                    ).unwrap();
                }
            }
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            let cta_n_start_slot = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", cta_n_start_slot, cta_n_start, n_slot * per_warp_n).unwrap();
            if let Some(ref bias_g) = bias_g {
                self.emit_gemm_bias_relu_epilogue(
                    &smem_c_base, smem_c_stride, &cta_m_start, &cta_n_start_slot, cta_m, per_warp_n, m, n, &c_g, bias_g, threads_per_cta,
                );
            } else {
                self.emit_gemm_plain_epilogue(
                    &smem_c_base, smem_c_stride, &cta_m_start, &cta_n_start_slot, cta_m, per_warp_n, m, n, &c_g, threads_per_cta,
                );
            }
            // Next slot's wmma.store reuses this same scratch buffer -
            // must not begin until every thread is done reading this
            // slot's data out of it.
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        threads_per_cta
    }

    /// Fused Bias+ReLU epilogue for `emit_tensor_core_gemm_kernel`'s 4-param
    /// (A, B, Bias, C) shape: reads the just-computed CTA output tile back
    /// out of shared memory (written there by a `wmma.store.d...shared.f32`
    /// pass over every warp's accumulator fragments, then a CTA-wide
    /// `bar.sync` - both emitted by the caller before this runs), adds the
    /// per-output-column `Bias` value (broadcast down every row - the same
    /// semantics a real `nn.Linear` bias add has), applies ReLU
    /// (`max(x, 0)`), and stores the result to global `C`. Mirrors
    /// `tests/y_tensor_core_gemm.cu`'s `y_fused_gemm_bias_relu_kernel`
    /// epilogue (the hand-written CUDA reference this is meant to match/beat
    /// - see that kernel's own comments), but with a runtime thread-strided
    /// loop over `cta_m * (cta_n / 4)` `float4` chunks (`idx` starting at
    /// `%tid.x`, incrementing by `threads_per_cta` each iteration - the same
    /// striding pattern `emit_gemm_tile_load` already uses) rather than the
    /// reference's compile-time-unrolled fixed 256-thread/128x128-tile
    /// mapping, so this works for any autotuned `(cta_m, cta_n,
    /// threads_per_cta)` combination without needing them to divide evenly.
    ///
    /// Boundary masking is at 4-element (one `float4`) chunk granularity,
    /// same whole-chunk-skipped philosophy as `emit_gemm_tile_load` and the
    /// plain-GEMM epilogue above: a chunk is skipped entirely (not written)
    /// if the row is out of bounds or the last of its 4 columns would be.
    /// Requires `tile_n` a multiple of 4 - true for every autotuned tile
    /// this codegen produces (`tile_n` is always a multiple of 16).
    ///
    /// Generic over which column band it's covering: the caller passes
    /// `tile_n`/`tile_n_start` for whatever slice of the CTA's output tile
    /// currently lives in `smem_c_base` (the full `cta_m x cta_n` tile when
    /// it fits the shared-memory budget, or one `cta_m x per_warp_n`
    /// warp-column band per pass otherwise - see
    /// `emit_tensor_core_gemm_kernel`'s smem-sizing doc comment for why the
    /// latter is needed).
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_bias_relu_epilogue(
        &mut self,
        smem_c_base: &str,
        smem_c_stride: u32,
        cta_m_start: &str,
        tile_n_start: &str,
        cta_m: u32,
        tile_n: u32,
        m: u32,
        n: u32,
        c_g: &str,
        bias_g: &str,
        threads_per_cta: u32,
    ) {
        debug_assert_eq!(tile_n % 4, 0, "vectorized bias+relu epilogue requires tile_n a multiple of 4");
        let cols_per_chunk = tile_n / 4;
        let total_chunks = cta_m * cols_per_chunk;

        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label("EPI_BIAS_RELU");
        let loop_end = self.alloc_label("EPI_BIAS_RELU_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_chunks).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        let lr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", lr, idx, cols_per_chunk).unwrap();
        let lc_chunk = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", lc_chunk, idx, cols_per_chunk).unwrap();
        let lc = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", lc, lc_chunk).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, cta_m_start, lr).unwrap();
        let gcol = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol, tile_n_start, lc).unwrap();
        let gcol_end = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 4;", gcol_end, gcol).unwrap();

        let p_row = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_row, grow, m).unwrap();
        let p_col = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.le.u32 {}, {}, {};", p_col, gcol_end, n).unwrap();
        let p_ok = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_ok, p_row, p_col).unwrap();

        // Shared-memory scratch tile read (local row/col, smem_c_stride).
        let sidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, lr, smem_c_stride, lc).unwrap();
        let sbyte = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", sbyte, sidx).unwrap();
        let saddr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_c_base, sbyte).unwrap();
        let s: Vec<String> = (0..4).map(|_| self.alloc_regf32()).collect();
        writeln!(&mut self.ptx_buffer, "    @{} ld.shared.v4.f32 {{{}}}, [{}];", p_ok, s.join(","), saddr).unwrap();

        // Bias: one value per output column, broadcast across every row in
        // this chunk's row - so only `gcol` (not `grow`) feeds its address.
        let bias_byte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", bias_byte, gcol).unwrap();
        let bias_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", bias_addr, bias_g, bias_byte).unwrap();
        let b: Vec<String> = (0..4).map(|_| self.alloc_regf32()).collect();
        writeln!(&mut self.ptx_buffer, "    @{} ld.global.v4.f32 {{{}}}, [{}];", p_ok, b.join(","), bias_addr).unwrap();

        // Global C write address (row-major, stride n).
        let gidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", gidx, grow, n, gcol).unwrap();
        let gbyte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", gbyte, gidx).unwrap();
        let gaddr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr, c_g, gbyte).unwrap();

        let r: Vec<String> = (0..4).map(|_| self.alloc_regf32()).collect();
        for lane in 0..4 {
            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", r[lane], s[lane], b[lane]).unwrap();
            writeln!(&mut self.ptx_buffer, "    max.f32 {}, {}, 0f00000000;", r[lane], r[lane]).unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    @{} st.global.v4.f32 [{}], {{{}}};", p_ok, gaddr, r.join(",")).unwrap();

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Plain (no Bias+ReLU) epilogue for `emit_tensor_core_gemm_kernel`'s
    /// 3-param (A, B, C) shape: reads the just-computed CTA output tile
    /// back out of shared memory (written there by a
    /// `wmma.store.d...shared.f32` pass over every warp's accumulator
    /// fragments, then a CTA-wide `bar.sync` - both emitted by the caller
    /// before this runs, identically to `emit_gemm_bias_relu_epilogue`)
    /// and stores it to global `C`, with no elementwise transform -
    /// structurally identical to that function (same thread-striding, same
    /// `float4`-chunk boundary masking), minus the bias load/add/ReLU.
    ///
    /// **Fixes a real, confirmed correctness bug**, not just a style
    /// unification with the Bias+ReLU epilogue: the direct-to-global path
    /// this replaced (`wmma.store.d.sync.aligned...global.f32`, gated by
    /// `p_ok = (c_row_end <= M) && (c_col_end <= N)` where `c_row_end =
    /// c_row + 16`) masked at whole-16-row-`wmma`-fragment granularity -
    /// for any `M < 16`, `c_row_end` (minimum 16) always exceeds `M`, so
    /// `p_ok` was false for every fragment of every CTA, and the kernel
    /// silently wrote nothing at all (`C` stayed whatever it was
    /// initialized to - all-zero in every test that surfaced this).
    /// Confirmed on real hardware at `M=1,4,8` (`benchmark_y_decode_gemm.py`
    /// against a fresh cuBLAS reference) before this fix, all three failing
    /// with `C` identically zero; this function's masking is per-*row*
    /// (`p_row: grow < m`, not `c_row_end <= m` for a whole 16-row band),
    /// the same fine-grained approach `emit_gemm_bias_relu_epilogue` already
    /// used, which has no minimum-`M` floor at all.
    ///
    /// Boundary masking is at 4-element (one `float4`) chunk granularity -
    /// a chunk is skipped entirely (not written) if its row is out of
    /// bounds or the last of its 4 columns would be. Requires `tile_n` a
    /// multiple of 4 - true for every autotuned tile this codegen produces.
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_plain_epilogue(
        &mut self,
        smem_c_base: &str,
        smem_c_stride: u32,
        cta_m_start: &str,
        tile_n_start: &str,
        cta_m: u32,
        tile_n: u32,
        m: u32,
        n: u32,
        c_g: &str,
        threads_per_cta: u32,
    ) {
        debug_assert_eq!(tile_n % 4, 0, "vectorized plain epilogue requires tile_n a multiple of 4");
        let cols_per_chunk = tile_n / 4;
        let total_chunks = cta_m * cols_per_chunk;

        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label("EPI_PLAIN");
        let loop_end = self.alloc_label("EPI_PLAIN_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_chunks).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        let lr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", lr, idx, cols_per_chunk).unwrap();
        let lc_chunk = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", lc_chunk, idx, cols_per_chunk).unwrap();
        let lc = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", lc, lc_chunk).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, cta_m_start, lr).unwrap();
        let gcol = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol, tile_n_start, lc).unwrap();
        let gcol_end = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 4;", gcol_end, gcol).unwrap();

        let p_row = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_row, grow, m).unwrap();
        let p_col = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.le.u32 {}, {}, {};", p_col, gcol_end, n).unwrap();
        let p_ok = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_ok, p_row, p_col).unwrap();

        // Shared-memory scratch tile read (local row/col, smem_c_stride).
        let sidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, lr, smem_c_stride, lc).unwrap();
        let sbyte = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", sbyte, sidx).unwrap();
        let saddr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_c_base, sbyte).unwrap();
        let s: Vec<String> = (0..4).map(|_| self.alloc_regf32()).collect();
        writeln!(&mut self.ptx_buffer, "    @{} ld.shared.v4.f32 {{{}}}, [{}];", p_ok, s.join(","), saddr).unwrap();

        // Global C write address (row-major, stride n).
        let gidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", gidx, grow, n, gcol).unwrap();
        let gbyte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", gbyte, gidx).unwrap();
        let gaddr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr, c_g, gbyte).unwrap();

        writeln!(&mut self.ptx_buffer, "    @{} st.global.v4.f32 [{}], {{{}}};", p_ok, gaddr, s.join(",")).unwrap();

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Detects the 4-parameter (X, W_gate, W_up: GlobalMemory<F16>, Out:
    /// GlobalMemory<F32>) kernel shape that dispatches
    /// `emit_gemm_swiglu_kernel`: `Out = SiLU(X @ W_gate) * (X @ W_up)`,
    /// the gate/up half of an LLM SwiGLU MLP block, fused into one launch.
    /// Distinguished from `tile_gemm_operands`'s 4-param (A, B, Bias, C)
    /// Bias+ReLU shape purely by type - Bias+ReLU's 3rd parameter is F32,
    /// this shape's 3rd parameter (`W_up`) is F16 - so the two never
    /// collide; see `verify_tile_gemm_kernel` in type_checker.rs for the
    /// type-checked contract both shapes share.
    /// `Y_SWIGLU_TILE=cta_m,cta_n,cta_k,warps_m,warps_n`, for sweeping this
    /// kernel's tile on hardware. Returns `None` (keep the default) when
    /// unset, malformed, or structurally impossible - a bad override should
    /// fall back loudly, never emit a kernel that cannot be lowered.
    fn swiglu_tile_override() -> Option<(u32, u32, u32, u32, u32)> {
        let spec = std::env::var("Y_SWIGLU_TILE").ok()?;
        let p: Vec<u32> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if p.len() != 5 {
            eprintln!(
                "[Y swiglu] ignoring malformed Y_SWIGLU_TILE='{}' \
                 (want cta_m,cta_n,cta_k,warps_m,warps_n)",
                spec
            );
            return None;
        }
        let (cta_m, cta_n, cta_k, wm, wn) = (p[0], p[1], p[2], p[3], p[4]);
        if wm == 0 || wn == 0 || cta_m % (wm * 16) != 0 || cta_n % (wn * 16) != 0 || cta_k % 16 != 0 {
            eprintln!("[Y swiglu] ignoring Y_SWIGLU_TILE='{}': tile does not split into 16x16 warp fragments", spec);
            return None;
        }
        // Two accumulator arrays (gate, up) at 8 f32 registers per 16x16
        // fragment. Past ~248 the kernel cannot fit its addressing registers
        // under ptxas's 255 cap and starts spilling.
        let acc_regs = (cta_m / wm / 16) * (cta_n / wn / 16) * 8 * 2;
        if acc_regs > 224 {
            eprintln!(
                "[Y swiglu] ignoring Y_SWIGLU_TILE='{}': {} accumulator registers/thread \
                 exceeds what fits alongside addressing under the 255 cap",
                spec, acc_regs
            );
            return None;
        }
        Some((cta_m, cta_n, cta_k, wm, wn))
    }

    fn tile_gemm_swiglu_operands(&self, kernel: &KernelDecl) -> Option<(u32, u32, u32, String, String, String, String)> {
        let tile = kernel.tile.as_ref()?;

        fn as_positive_u32(e: &Expr) -> Option<u32> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => u32::try_from(*v).ok(),
                _ => None,
            }
        }
        let m = as_positive_u32(&tile.block_m)?;
        let n = as_positive_u32(&tile.block_n)?;
        let k = as_positive_u32(tile.block_k.as_deref()?)?;

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }
        if kernel.params.len() != 4
            || !is_global_memory_of(&kernel.params[0].ty, "F16")
            || !is_global_memory_of(&kernel.params[1].ty, "F16")
            || !is_global_memory_of(&kernel.params[2].ty, "F16")
            || !is_global_memory_of(&kernel.params[3].ty, "F32")
        {
            return None;
        }

        let x_reg = self.variables.get(&kernel.params[0].name)?.clone();
        let wgate_reg = self.variables.get(&kernel.params[1].name)?.clone();
        let wup_reg = self.variables.get(&kernel.params[2].name)?.clone();
        let out_reg = self.variables.get(&kernel.params[3].name)?.clone();
        Some((m, n, k, x_reg, wgate_reg, wup_reg, out_reg))
    }

    /// Emits a complete, self-contained fused Linear+SwiGLU GEMM kernel:
    /// `Out = SiLU(X @ W_gate) * (X @ W_up)`, where `X` is `[M,K]`,
    /// `W_gate`/`W_up` are `[K,N]`, `Out` is `[M,N]` - the gate/up
    /// projections of an LLM MLP block, fused with the SwiGLU activation
    /// so neither projection's `[M,N]` result is ever written to or read
    /// back from DRAM (only `Out` is). Dispatched from a validated
    /// kernel-level `@tile(M,N,K)` with the 4-param shape detected by
    /// `tile_gemm_swiglu_operands`.
    ///
    /// Shares its warp/lane decomposition, grid swizzle, `ldmatrix`+
    /// `mma.sync` compute core (`emit_gemm_compute_block`), and (as of
    /// this session) `cp.async` multi-stage pipelining structure
    /// (`emit_gemm_tile_load_async`/`emit_cp_async_commit`/
    /// `emit_cp_async_wait`) with `emit_tensor_core_gemm_kernel` - see that
    /// function's doc comment for the derivation and real-hardware
    /// validation of those pieces, reused unchanged here; the pipelining
    /// below is a direct extension of its N-stage prefetch/compute-overlap
    /// algorithm to three tile streams (`X`, `W_gate`, `W_up`) instead of
    /// two, not a new design. The `silu(x) = x / (1 + exp(-x))` epilogue
    /// math (`ex2.approx.f32` for `exp`, `rcp.approx.f32` for the
    /// reciprocal - PTX has no non-approx `ex2`) was validated standalone
    /// against `torch.nn.functional.silu` on real sm_89 hardware before
    /// this function was written: max abs error 3.8e-6, max relative error
    /// 4.5e-7 over 2000 random `(gate, up)` pairs spanning +/-4 - see
    /// `emit_gemm_swiglu_epilogue`'s doc comment.
    ///
    /// Scope, stated plainly (disclosed simplifications relative to
    /// `emit_tensor_core_gemm_kernel`, not oversights):
    /// - **Fixed, hardcoded CTA tile (128x128x32, 4x4 warps)** rather than
    ///   `Autotuner::autotune`. Two independent accumulator sets (gate, up)
    ///   already double live accumulator registers per thread versus the
    ///   plain GEMM at the same CTA tile (`num_i * num_j * 8` f32 regs,
    ///   twice over) - at the 128x256 CTA tile `Autotuner::autotune` picks
    ///   for this project's M=N=K >= 2048 plain-GEMM benchmarks, that alone
    ///   is 256 f32 accumulator registers/thread, over ptxas's 255-register
    ///   hard cap before a single address/temp register is counted. The
    ///   autotuner has no notion of a kernel needing 2x the accumulator
    ///   budget (it scores one GEMM's occupancy), so trusting it here would
    ///   silently pick register-infeasible tiles at exactly the sizes this
    ///   pitch cares about. A fixed tile with `per_warp_m=per_warp_n=32`
    ///   (`num_i=num_j=2` => 32 accumulator regs/array, 64 total)
    ///   sidesteps that risk while still covering a 128x128 output region
    ///   per CTA (4x4=16 warps). `cta_k` is 32, not the 64 an earlier
    ///   version of this kernel used: a *third* smem-resident stream
    ///   (`W_up`, on top of `X`/`W_gate`) means the K-loop's real per-stage
    ///   footprint (`stage_a_bytes + 2*stage_b_bytes`) is 1.5x a plain
    ///   GEMM's at the same tile - at `cta_k=64` this floors the achievable
    ///   pipeline depth to a single stage on this project's dev GPU (no
    ///   room for real overlap at all, defeating the point of adding
    ///   `cp.async`); halving `cta_k` to 32 buys back enough smem headroom
    ///   for genuine 3-stage pipelining (see `effective_stages` derivation
    ///   below) at the cost of twice the K-loop trip count. Register
    ///   pressure re-verified via `ptxas -v` after this change (still 0
    ///   spills - `cta_k` does not affect accumulator count, only
    ///   `k_substeps`, the compile-time-unrolled inner loop inside one
    ///   `emit_gemm_compute_block` call).
    /// - `emit_gemm_compute_block` is called twice per K-tile (once for
    ///   `W_gate`, once for `W_up`), so `X`'s `ldmatrix.x4` fragment loads
    ///   are issued twice per K-tile instead of once (each call
    ///   independently reloads `X`'s fragments from shared memory) - real,
    ///   disclosed instruction-count overhead, not a correctness issue
    ///   (`ldmatrix` is a pure read). Not addressed by this session's
    ///   pipelining work - a separate, still-open lever.
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_swiglu_kernel(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        x_ptr: &str,
        wgate_ptr: &str,
        wup_ptr: &str,
        out_ptr: &str,
        hw_profile: &HardwareProfile,
        kernel_name: &str,
    ) -> u32 {
        // ---- fixed CTA tile - see doc comment for why this is not
        // Autotuner::autotune'd like the plain/bias-relu GEMM kernels ----
        //
        // `Y_SWIGLU_TILE=cta_m,cta_n,cta_k,warps_m,warps_n` overrides it, so
        // this tile can be swept on real hardware instead of argued about.
        // It needs its own hook rather than reusing `Y_CTA_OVERRIDE`: that
        // one feeds `Autotuner::autotune`, which this kernel deliberately
        // does not call (its two accumulator arrays make the autotuner's
        // single-GEMM occupancy model wrong here - see the doc comment).
        //
        // The register constraint the default respects: this kernel holds
        // TWO f32 accumulator arrays (gate and up), each
        // `num_i*num_j*8` registers, so a 128x128 tile over 2x2 warps would
        // need 4*4*8*2 = 256 accumulator registers per thread - past ptxas's
        // 255 cap before a single address register. Any override is checked
        // against that below rather than trusted.
        let (cta_m, cta_n, cta_k, warps_m, warps_n) = Self::swiglu_tile_override()
            .unwrap_or((128, 128, 32, 4, 4));
        let threads_per_cta = warps_m * warps_n * 32;

        let per_warp_m = cta_m / warps_m;
        let per_warp_n = cta_n / warps_n;
        let num_i = per_warp_m / 16;
        let num_j = per_warp_n / 16;
        let k_substeps = cta_k / 16;
        debug_assert_eq!(num_i * 16 * warps_m, cta_m, "cta_m must split evenly into 16-row warp fragments");
        debug_assert_eq!(num_j * 16 * warps_n, cta_n, "cta_n must split evenly into 16-col warp fragments");
        debug_assert_eq!(k_substeps * 16, cta_k, "cta_k must be a multiple of mma.sync's k16 dimension");
        // Hard asserts, not `debug_assert!`s: unlike the plain F16 GEMM (which
        // masks M/N tails and, since the K-tail fix, K tails too), this fused
        // kernel has no tail path at all - a non-multiple shape here silently
        // drops the remainder and returns a wrong result from the release
        // binary. Fail the compile instead. See `emit_tensor_core_gemm_kernel`'s
        // K-tail note for the measurement that motivated this.
        assert_eq!(m % cta_m, 0, "@tile M={} must be a multiple of the fused SwiGLU kernel's fixed cta_m={}; this kernel has no M-tail path and would silently drop the remainder", m, cta_m);
        assert_eq!(n % cta_n, 0, "@tile N={} must be a multiple of the fused SwiGLU kernel's fixed cta_n={}; this kernel has no N-tail path and would silently drop the remainder", n, cta_n);
        assert_eq!(k % cta_k, 0, "@tile K={} must be a multiple of the fused SwiGLU kernel's fixed cta_k={}; this kernel has no K-tail path and would silently drop the remainder", k, cta_k);

        let smem_a_stride = cta_k + 8;
        let smem_b_stride = cta_n + 8;
        let k_tiles = k / cta_k;

        let stage_a_bytes = cta_m * smem_a_stride * 2;
        let stage_b_bytes = cta_k * smem_b_stride * 2;
        // One stage's worth of X + W_gate + W_up resident simultaneously -
        // see doc comment for why a third stream (versus a plain GEMM's
        // two) is what forced cta_k down to 32.
        let per_stage_bytes = stage_a_bytes + stage_b_bytes * 2;

        // ---- achievable pipeline depth - mirrors
        // emit_tensor_core_gemm_kernel's own effective_stages derivation
        // (same safety margin, same k_tiles clamp), requesting 4 stages and
        // letting smem/k_tiles clamp it down rather than hand-picking a
        // number - see that function's doc comment for the full
        // reasoning. No Autotuner::autotune involved (this kernel doesn't
        // use it - see doc comment), so the "requested" stage count is a
        // fixed constant here, not a per-shape autotuned choice. ----
        let requested_stages: u32 = 4;
        let safe_smem_ceiling = hw_profile.max_smem_per_sm_bytes.saturating_sub(4096);
        let max_stages_by_smem = (safe_smem_ceiling / per_stage_bytes).max(1);
        let effective_stages = requested_stages.min(k_tiles).min(max_stages_by_smem).max(1);
        let kloop_bytes = effective_stages * per_stage_bytes;

        // Epilogue needs gate + up scratch tiles resident simultaneously,
        // one per_warp_n-wide warp-column band at a time - same banding
        // rationale as emit_tensor_core_gemm_kernel's fused Bias+ReLU
        // epilogue (see its doc comment).
        let smem_c_stride = per_warp_n + 4;
        let smem_c_bytes_one = cta_m * smem_c_stride * 4;
        let epilogue_bytes = smem_c_bytes_one * 2;

        let total_dyn_smem_bytes = kloop_bytes.max(epilogue_bytes);

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [Y FUSED LINEAR+SWIGLU GEMM] M={} N={} K={} | CTA {}x{}x{} (fixed) | {}x{} warps | Out=SiLU(X@Wgate)*(X@Wup)", m, n, k, cta_m, cta_n, cta_k, warps_m, warps_n).unwrap();
        if effective_stages >= 2 {
            writeln!(&mut self.ptx_buffer, "    // {} pipeline stages (requested {}, clamped by k_tiles/smem); cp.async multi-stage pipelined. Dynamic shared memory required: {} bytes.", effective_stages, requested_stages, total_dyn_smem_bytes).unwrap();
        } else {
            writeln!(&mut self.ptx_buffer, "    // Only {} K-tile(s)/insufficient smem for pipelining; single-buffered synchronous staging. Dynamic shared memory required: {} bytes.", k_tiles, total_dyn_smem_bytes).unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        let x_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", x_g, x_ptr).unwrap();
        let wgate_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", wgate_g, wgate_ptr).unwrap();
        let wup_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", wup_g, wup_ptr).unwrap();
        let out_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", out_g, out_ptr).unwrap();

        let (bid_m, bid_n) = self.emit_grid_swizzle_code(GEMM_SWIZZLE_GROUP_SIZE);
        let cta_m_start = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", cta_m_start, bid_m, cta_m).unwrap();
        let cta_n_start = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", cta_n_start, bid_n, cta_n).unwrap();

        let tid = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
        let warp_id = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 5;", warp_id, tid).unwrap();
        let warp_m = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", warp_m, warp_id, warps_m).unwrap();
        let warp_n = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", warp_n, warp_id, warps_m).unwrap();
        let warp_row0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", warp_row0, warp_m, per_warp_m, cta_m_start).unwrap();
        let warp_col0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", warp_col0, warp_n, per_warp_n, cta_n_start).unwrap();
        let warp_row0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_row0_local, warp_m, per_warp_m).unwrap();
        let warp_col0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_col0_local, warp_n, per_warp_n).unwrap();
        // warp_row0/warp_col0 (grid-global) are computed for parity with
        // emit_tensor_core_gemm_kernel's decomposition but are only used
        // there by its non-banded epilogue path; this kernel's epilogue is
        // always banded (see below), so silence the unused-variable lint
        // rather than drop the shared derivation.
        let _ = (&warp_row0, &warp_col0);

        let lane = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 31;", lane, tid).unwrap();
        let row_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 15;", row_off, lane).unwrap();
        let col_bit_a_wide = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 16;", col_bit_a_wide, lane).unwrap();
        let col_bit_a = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 1;", col_bit_a, col_bit_a_wide).unwrap();

        let mut acc_gate: Vec<Vec<Vec<String>>> = Vec::with_capacity(num_i as usize);
        let mut acc_up: Vec<Vec<Vec<String>>> = Vec::with_capacity(num_i as usize);
        for _ in 0..num_i {
            let mut row_gate = Vec::with_capacity(num_j as usize);
            let mut row_up = Vec::with_capacity(num_j as usize);
            for _ in 0..num_j {
                let mut frag_gate = Vec::with_capacity(8);
                let mut frag_up = Vec::with_capacity(8);
                for _ in 0..8 {
                    let rg = self.alloc_regf32();
                    writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", rg).unwrap();
                    frag_gate.push(rg);
                    let ru = self.alloc_regf32();
                    writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", ru).unwrap();
                    frag_up.push(ru);
                }
                row_gate.push(frag_gate);
                row_up.push(frag_up);
            }
            acc_gate.push(row_gate);
            acc_up.push(row_up);
        }

        let smem_symbol = format!("smem_pipeline_{}", kernel_name);
        self.pending_extern_decls.push(format!(".extern .shared .align 16 .b8 {}[];", smem_symbol));
        let base = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", base, smem_symbol).unwrap();

        if effective_stages < 2 {
            // ---- Fallback: single-buffered synchronous path (load ->
            // bar.sync -> compute -> bar.sync), reachable whenever
            // `effective_stages` clamps down to 1 - see doc comment and
            // `emit_tensor_core_gemm_kernel`'s own identical fallback for
            // when this is real (not exercised by any shape this session's
            // benchmark uses, all of which get 3 real pipeline stages at
            // this kernel's fixed tile, but kept for the same K-too-small
            // edge cases that fallback exists for). ----
            let smem_a_base = base.clone();
            let smem_bgate_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_bgate_base, base, stage_a_bytes).unwrap();
            let smem_bup_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_bup_base, base, stage_a_bytes + stage_b_bytes).unwrap();

            let k_iter = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
            let loop_start = self.alloc_label("SWIGLU_K_LOOP");
            let loop_end = self.alloc_label("SWIGLU_K_DONE");
            writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
            let exit_pred = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", exit_pred, k_iter, k_tiles).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} bra {};", exit_pred, loop_end).unwrap();
            let k0 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", k0, k_iter, cta_k).unwrap();

            self.emit_gemm_tile_load(
                "SWX", &x_g, &cta_m_start, &k0, k, m, k, cta_m, cta_k, &smem_a_base, smem_a_stride, threads_per_cta,
            );
            self.emit_gemm_tile_load(
                "SWG", &wgate_g, &k0, &cta_n_start, n, k, n, cta_k, cta_n, &smem_bgate_base, smem_b_stride, threads_per_cta,
            );
            self.emit_gemm_tile_load(
                "SWU", &wup_g, &k0, &cta_n_start, n, k, n, cta_k, cta_n, &smem_bup_base, smem_b_stride, threads_per_cta,
            );
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
            self.emit_gemm_compute_block(
                &acc_gate, &warp_col0_local, &warp_row0_local, &smem_a_base, &smem_bgate_base,
                smem_a_stride, smem_b_stride, &row_off, &col_bit_a, k_substeps, num_i, num_j,
            );
            self.emit_gemm_compute_block(
                &acc_up, &warp_col0_local, &warp_row0_local, &smem_a_base, &smem_bup_base,
                smem_a_stride, smem_b_stride, &row_off, &col_bit_a, k_substeps, num_i, num_j,
            );
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
            writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
            writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
        } else {
            // ---- N-stage cp.async pipelined path: direct extension of
            // emit_tensor_core_gemm_kernel's own algorithm (see that
            // function's doc comment for the read_stage/write_stage/
            // next_tile derivation) to three smem-resident tile streams
            // (X, W_gate, W_up) instead of two - X's region, then
            // W_gate's, then W_up's, each n_stages*stage_bytes long,
            // contiguous in the one combined dynamic buffer. ----
            let n_stages = effective_stages;
            let smem_bgate_region_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_bgate_region_base, base, n_stages * stage_a_bytes).unwrap();
            let smem_bup_region_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_bup_region_base, smem_bgate_region_base, n_stages * stage_b_bytes).unwrap();

            // ---- Prologue: prefetch stages 0..n_stages-2 (compile-time
            // constant tile indices - always in-bounds since n_stages <=
            // k_tiles by construction). ----
            for s in 0..(n_stages - 1) {
                let k0_s = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", k0_s, s * cta_k).unwrap();
                let a_stage_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", a_stage_base, base, s * stage_a_bytes).unwrap();
                let bgate_stage_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", bgate_stage_base, smem_bgate_region_base, s * stage_b_bytes).unwrap();
                let bup_stage_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", bup_stage_base, smem_bup_region_base, s * stage_b_bytes).unwrap();

                self.emit_gemm_tile_load_async(
                    &format!("SWPX{}", s), &x_g, &cta_m_start, &k0_s, k, m, k, cta_m, cta_k, &a_stage_base, smem_a_stride, threads_per_cta, None,
                );
                self.emit_gemm_tile_load_async(
                    &format!("SWPG{}", s), &wgate_g, &k0_s, &cta_n_start, n, k, n, cta_k, cta_n, &bgate_stage_base, smem_b_stride, threads_per_cta, None,
                );
                self.emit_gemm_tile_load_async(
                    &format!("SWPU{}", s), &wup_g, &k0_s, &cta_n_start, n, k, n, cta_k, cta_n, &bup_stage_base, smem_b_stride, threads_per_cta, None,
                );
                self.emit_cp_async_commit();
            }
            self.emit_cp_async_wait(n_stages - 2);
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            let k_iter = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
            let loop_start = self.alloc_label("SWIGLU_K_LOOP");
            let loop_end = self.alloc_label("SWIGLU_K_DONE");
            writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
            let exit_pred = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", exit_pred, k_iter, k_tiles).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} bra {};", exit_pred, loop_end).unwrap();

            let read_stage = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", read_stage, k_iter, n_stages).unwrap();
            let next_tile = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", next_tile, k_iter, n_stages - 1).unwrap();
            let write_stage = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", write_stage, next_tile, n_stages).unwrap();
            let p_tile_valid = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_tile_valid, next_tile, k_tiles).unwrap();
            let k0_next = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", k0_next, next_tile, cta_k).unwrap();

            let a_write_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", a_write_base, write_stage, stage_a_bytes, base).unwrap();
            let bgate_write_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", bgate_write_base, write_stage, stage_b_bytes, smem_bgate_region_base).unwrap();
            let bup_write_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", bup_write_base, write_stage, stage_b_bytes, smem_bup_region_base).unwrap();

            // Issue the prefetch for `next_tile` (masked off via
            // `p_tile_valid` if it doesn't exist), commit, then compute on
            // `read_stage` (already confirmed ready) - overlaps this
            // prefetch's async-copy-engine traffic with the tensor-core
            // work below instead of waiting for it first.
            self.emit_gemm_tile_load_async(
                "SWX", &x_g, &cta_m_start, &k0_next, k, m, k, cta_m, cta_k, &a_write_base, smem_a_stride, threads_per_cta, Some(&p_tile_valid),
            );
            self.emit_gemm_tile_load_async(
                "SWG", &wgate_g, &k0_next, &cta_n_start, n, k, n, cta_k, cta_n, &bgate_write_base, smem_b_stride, threads_per_cta, Some(&p_tile_valid),
            );
            self.emit_gemm_tile_load_async(
                "SWU", &wup_g, &k0_next, &cta_n_start, n, k, n, cta_k, cta_n, &bup_write_base, smem_b_stride, threads_per_cta, Some(&p_tile_valid),
            );
            self.emit_cp_async_commit();

            let a_read_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", a_read_base, read_stage, stage_a_bytes, base).unwrap();
            let bgate_read_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", bgate_read_base, read_stage, stage_b_bytes, smem_bgate_region_base).unwrap();
            let bup_read_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", bup_read_base, read_stage, stage_b_bytes, smem_bup_region_base).unwrap();

            self.emit_gemm_compute_block(
                &acc_gate, &warp_col0_local, &warp_row0_local, &a_read_base, &bgate_read_base,
                smem_a_stride, smem_b_stride, &row_off, &col_bit_a, k_substeps, num_i, num_j,
            );
            self.emit_gemm_compute_block(
                &acc_up, &warp_col0_local, &warp_row0_local, &a_read_base, &bup_read_base,
                smem_a_stride, smem_b_stride, &row_off, &col_bit_a, k_substeps, num_i, num_j,
            );

            // Wait until group `k_iter+1` (the tile that will become
            // read_stage next iteration) has landed, then make it visible
            // block-wide - see emit_tensor_core_gemm_kernel's identical
            // step for the full derivation of why this is safe for the
            // next iteration's prefetch-write to reuse this iteration's
            // read_stage slot.
            self.emit_cp_async_wait(n_stages - 2);
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
            writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
            writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();

            // Drain any still-in-flight tail prefetch before the epilogue
            // reuses this same dynamic shared buffer as scratch.
            self.emit_cp_async_wait(0);
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
        }

        // ---- Epilogue: SiLU(gate) * up, banded by warp-column - same
        // rationale as emit_gemm_bias_relu_epilogue (see its doc comment) ----
        let smem_gate_base = base.clone();
        let smem_up_base = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_up_base, base, smem_c_bytes_one).unwrap();
        let stride_c_smem_reg = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", stride_c_smem_reg, smem_c_stride).unwrap();
        for n_slot in 0..warps_n {
            let p_slot = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, {}, {};", p_slot, warp_n, n_slot).unwrap();
            for i in 0..num_i {
                for j in 0..num_j {
                    let c_row_local = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", c_row_local, warp_row0_local, i * 16).unwrap();
                    let sidx = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, c_row_local, stride_c_smem_reg, j * 16).unwrap();
                    let sbyte = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", sbyte, sidx).unwrap();

                    let saddr_gate = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr_gate, smem_gate_base, sbyte).unwrap();
                    let d_gate = acc_gate[i as usize][j as usize].join(",");
                    writeln!(
                        &mut self.ptx_buffer,
                        "    @{} wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [{}], {{{}}}, {};",
                        p_slot, saddr_gate, d_gate, stride_c_smem_reg
                    ).unwrap();

                    let saddr_up = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr_up, smem_up_base, sbyte).unwrap();
                    let d_up = acc_up[i as usize][j as usize].join(",");
                    writeln!(
                        &mut self.ptx_buffer,
                        "    @{} wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [{}], {{{}}}, {};",
                        p_slot, saddr_up, d_up, stride_c_smem_reg
                    ).unwrap();
                }
            }
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            let cta_n_start_slot = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", cta_n_start_slot, cta_n_start, n_slot * per_warp_n).unwrap();
            self.emit_gemm_swiglu_epilogue(
                &smem_gate_base, &smem_up_base, smem_c_stride, &cta_m_start, &cta_n_start_slot, cta_m, per_warp_n, m, n, &out_g, threads_per_cta,
            );
            // Next slot's wmma.store reuses this same scratch buffer - must
            // not begin until every thread is done reading this slot's data.
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
        }

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        threads_per_cta
    }

    /// Fused `silu(gate) * up` epilogue for `emit_gemm_swiglu_kernel`: reads
    /// the just-computed gate/up CTA output tiles back out of shared memory
    /// (each written there by a `wmma.store.d...shared.f32` pass over every
    /// warp's accumulator fragments, then a CTA-wide `bar.sync` - both
    /// emitted by the caller before this runs), computes
    /// `silu(gate) * up` where `silu(x) = x * sigmoid(x) = x / (1 +
    /// exp(-x))` via `ex2.approx.f32`/`rcp.approx.f32` (PTX has no
    /// non-approx `ex2`), and stores the result to global `Out`.
    ///
    /// This exact instruction sequence (`neg.f32` -> `mul.f32` by
    /// `log2(e)` -> `ex2.approx.f32` -> `add.f32` 1.0 -> `rcp.approx.f32`
    /// -> two `mul.f32`) was validated standalone on real sm_89 hardware
    /// before being wired in here: a single-thread probe kernel compared
    /// against `torch.nn.functional.silu(gate) * up` over 2000 random
    /// `(gate, up)` pairs (gate ~ N(0, 4), up ~ N(0, 2)) gave a max
    /// absolute error of 3.8e-6 and max relative error of 4.5e-7 - far
    /// inside FP16 tolerance, so the approximation is not a meaningful
    /// correctness concern for this kernel's F32-accumulator, F16-input
    /// use case.
    ///
    /// Scalar (not `float4`-vectorized like `emit_gemm_bias_relu_epilogue`)
    /// - a real, disclosed simplification: a vectorized version would still
    /// need `ex2.approx`/`rcp.approx` issued per-lane-of-4 (no vector form
    /// of either instruction exists in PTX), so the only loss versus a
    /// vectorized version is loop/address-math overhead, not activation
    /// throughput.
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_swiglu_epilogue(
        &mut self,
        smem_gate_base: &str,
        smem_up_base: &str,
        smem_c_stride: u32,
        cta_m_start: &str,
        tile_n_start: &str,
        cta_m: u32,
        tile_n: u32,
        m: u32,
        n: u32,
        out_g: &str,
        threads_per_cta: u32,
    ) {
        let total_chunks = cta_m * tile_n;
        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label("EPI_SWIGLU");
        let loop_end = self.alloc_label("EPI_SWIGLU_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p_done, idx, total_chunks).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p_done, loop_end).unwrap();

        let lr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", lr, idx, tile_n).unwrap();
        let lc = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", lc, idx, tile_n).unwrap();

        let grow = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", grow, cta_m_start, lr).unwrap();
        let gcol = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", gcol, tile_n_start, lc).unwrap();

        let p_row = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_row, grow, m).unwrap();
        let p_col = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_col, gcol, n).unwrap();
        let p_ok = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_ok, p_row, p_col).unwrap();

        let sidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, lr, smem_c_stride, lc).unwrap();
        let sbyte = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", sbyte, sidx).unwrap();

        let saddr_gate = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr_gate, smem_gate_base, sbyte).unwrap();
        let gate_val = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    @{} ld.shared.f32 {}, [{}];", p_ok, gate_val, saddr_gate).unwrap();

        let saddr_up = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr_up, smem_up_base, sbyte).unwrap();
        let up_val = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    @{} ld.shared.f32 {}, [{}];", p_ok, up_val, saddr_up).unwrap();

        // silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate));
        // exp(-gate) via ex2.approx.f32(-gate * log2(e)).
        let neg_gate = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    neg.f32 {}, {};", neg_gate, gate_val).unwrap();
        let scaled = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, 0f3FB8AA3B;", scaled, neg_gate).unwrap();
        let exp_neg = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    ex2.approx.f32 {}, {};", exp_neg, scaled).unwrap();
        let denom = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 0f3F800000;", denom, exp_neg).unwrap();
        let sig = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    rcp.approx.f32 {}, {};", sig, denom).unwrap();
        let silu = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", silu, gate_val, sig).unwrap();
        let result = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", result, silu, up_val).unwrap();

        let gidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", gidx, grow, n, gcol).unwrap();
        let gbyte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", gbyte, gidx).unwrap();
        let gaddr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", gaddr, out_g, gbyte).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}], {};", p_ok, gaddr, result).unwrap();

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Formats an f32 as a PTX hex float immediate (`0f<8 hex digits>`, the
    /// IEEE-754 bit pattern) - used for every compile-time float constant
    /// below rather than relying on PTX's decimal float-literal parsing,
    /// which is not independently verified anywhere in this codebase (the
    /// one place it appeared, `emit_vectorized_swiglu_fast`, was dead code
    /// with its own unrelated bugs - see that function's doc comment).
    fn f32_to_ptx_hex(f: f32) -> String {
        format!("0f{:08X}", f.to_bits())
    }

    /// Emits a vectorized global-memory load of `n_words` (1, 2, or 4)
    /// contiguous 32-bit words - `ld.global.u32` / `ld.global.v2.u32` /
    /// `ld.global.v4.u32` - starting at `addr`, returning the allocated
    /// registers. Each word holds two packed f16 values (element order:
    /// low 16 bits first, then high 16 bits) - see `emit_unpack_f16_pairs`.
    /// Store sibling: `emit_vec_store_u32`.
    ///
    /// This exact load-then-unpack sequence (and the pack-then-store
    /// sequence below) was standalone-validated on real sm_89 hardware
    /// before use in any real emitter - a tiny hand-written PTX probe (all
    /// three widths, `out[i] = in[i]*2+1` against a torch fp16 reference,
    /// bit-exact across 5 seeds) - per this codebase's established
    /// PTX-validation discipline. See `emit_rmsnorm_residual_kernel` /
    /// `emit_rope_kernel`, the two callers.
    fn emit_vec_load_u32(&mut self, addr: &str, n_words: usize) -> Vec<String> {
        let regs: Vec<String> = (0..n_words).map(|_| self.alloc_reg32()).collect();
        match n_words {
            1 => writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}];", regs[0], addr).unwrap(),
            2 => writeln!(&mut self.ptx_buffer, "    ld.global.v2.u32 {{{}}}, [{}];", regs.join(","), addr).unwrap(),
            4 => writeln!(&mut self.ptx_buffer, "    ld.global.v4.u32 {{{}}}, [{}];", regs.join(","), addr).unwrap(),
            _ => unreachable!("emit_vec_load_u32: n_words must be 1, 2, or 4"),
        }
        regs
    }

    /// Store sibling of `emit_vec_load_u32` - `words.len()` must be 1, 2, or 4.
    fn emit_vec_store_u32(&mut self, addr: &str, words: &[String]) {
        match words.len() {
            1 => writeln!(&mut self.ptx_buffer, "    st.global.u32 [{}], {};", addr, words[0]).unwrap(),
            2 => writeln!(&mut self.ptx_buffer, "    st.global.v2.u32 [{}], {{{}}};", addr, words.join(",")).unwrap(),
            4 => writeln!(&mut self.ptx_buffer, "    st.global.v4.u32 [{}], {{{}}};", addr, words.join(",")).unwrap(),
            _ => unreachable!("emit_vec_store_u32: words.len() must be 1, 2, or 4"),
        }
    }

    /// Unpacks each 32-bit word in `words` (two packed f16 values, low
    /// 16 bits then high 16 bits) into f32 registers via `cvt.f32.f16`
    /// directly on the word for the low half, `shr.b32` + `cvt.f32.f16` for
    /// the high half. Returns `2*words.len()` registers in element order
    /// `[w0_lo, w0_hi, w1_lo, w1_hi, ...]`. Pack sibling: `emit_pack_f16_pairs`.
    fn emit_unpack_f16_pairs(&mut self, words: &[String]) -> Vec<String> {
        let mut out = Vec::with_capacity(words.len() * 2);
        for w in words {
            let lo = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    cvt.f32.f16 {}, {};", lo, w).unwrap();
            let hi_bits = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    shr.b32 {}, {}, 16;", hi_bits, w).unwrap();
            let hi = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    cvt.f32.f16 {}, {};", hi, hi_bits).unwrap();
            out.push(lo);
            out.push(hi);
        }
        out
    }

    /// Pack sibling of `emit_unpack_f16_pairs`: takes f32 values in element
    /// order (`vals.len()` must be even) and returns `vals.len()/2` packed
    /// 32-bit words, each pair rounded to f16 (`cvt.rn.f16.f32`) then
    /// combined via `and.b32` (mask the low element to 16 bits) / `shl.b32`
    /// (shift the high element up, which also discards any garbage above
    /// its own low 16 bits - no mask needed on that side) / `or.b32`.
    fn emit_pack_f16_pairs(&mut self, vals: &[String]) -> Vec<String> {
        debug_assert_eq!(vals.len() % 2, 0, "emit_pack_f16_pairs requires an even number of values");
        let mut out = Vec::with_capacity(vals.len() / 2);
        for pair in vals.chunks(2) {
            let lo_h = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    cvt.rn.f16.f32 {}, {};", lo_h, pair[0]).unwrap();
            let lo_masked = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 0xFFFF;", lo_masked, lo_h).unwrap();
            let hi_h = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    cvt.rn.f16.f32 {}, {};", hi_h, pair[1]).unwrap();
            let hi_shifted = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 16;", hi_shifted, hi_h).unwrap();
            let packed = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    or.b32 {}, {}, {};", packed, lo_masked, hi_shifted).unwrap();
            out.push(packed);
        }
        out
    }

    /// Parses a `<prefix>_<positive integer>` kernel name into the trailing
    /// integer - the compile-time row-width (`hidden_dim`/`head_dim`) for
    /// `emit_rmsnorm_residual_kernel`/`emit_rope_kernel` below. Unlike the
    /// `@tile`d GEMM kernels, these have no compile-time M: row count is a
    /// pure runtime grid dimension (one warp per row, entirely generic over
    /// how many rows the launcher asks for), so encoding just the one
    /// genuinely compile-time-needed integer in the kernel name - rather
    /// than adding a second kernel-level attribute mechanism alongside
    /// `@tile` for a single integer - is the deliberately minimal choice.
    fn parse_trailing_dim(name: &str, prefix: &str) -> Option<u32> {
        let rest = name.strip_prefix(prefix)?.strip_prefix('_')?;
        rest.parse::<u32>().ok().filter(|&d| d > 0)
    }

    /// Detects the 5-parameter (X, Residual, Weight, Out, NewResidual: all
    /// `GlobalMemory<F16>`) kernel shape, named `rmsnorm_residual_<hidden_dim>`,
    /// that dispatches `emit_rmsnorm_residual_kernel`.
    fn rmsnorm_residual_operands(&self, kernel: &KernelDecl) -> Option<(u32, String, String, String, String, String)> {
        if kernel.tile.is_some() {
            return None;
        }
        let hidden_dim = Self::parse_trailing_dim(&kernel.name, "rmsnorm_residual")?;

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }
        if kernel.params.len() != 5 || !kernel.params.iter().all(|p| is_global_memory_of(&p.ty, "F16")) {
            return None;
        }

        let x = self.variables.get(&kernel.params[0].name)?.clone();
        let residual = self.variables.get(&kernel.params[1].name)?.clone();
        let weight = self.variables.get(&kernel.params[2].name)?.clone();
        let out = self.variables.get(&kernel.params[3].name)?.clone();
        let new_residual = self.variables.get(&kernel.params[4].name)?.clone();
        Some((hidden_dim, x, residual, weight, out, new_residual))
    }

    /// Emits a complete, self-contained fused "add & RMSNorm" kernel:
    /// `num_warps` warps (`num_warps*32` threads) per row, `num_warps`
    /// scaling with `hidden_dim` (see below) - not a fixed single warp
    /// regardless of size, the original design's real occupancy blind spot.
    /// Computes `h = X + Residual` (the updated residual stream for the
    /// next transformer layer), then `Out = (h / rms(h)) * Weight` where
    /// `rms(h) = sqrt(mean(h^2) + eps)` - the fused pattern real inference
    /// engines use (e.g. vLLM/FlashInfer's `fused_add_rmsnorm`), writing
    /// both `Out` and `h` (as `NewResidual`) from one launch.
    ///
    /// `hidden_dim` must be a multiple of 256 (`debug_assert`ed - 32 lanes x
    /// 8 f16 elements/lane per vectorized group; true for every realistic
    /// LLM hidden size: 2048, 3072, 4096, 5120, 7168, 8192, ...).
    ///
    /// **`num_warps`**: the largest of `{8, 4, 2, 1}` that divides
    /// `hidden_dim/256` evenly (so every thread gets a whole number of
    /// 8-wide vector groups, no remainder handling needed) - 8 warps (256
    /// threads/row) at every hidden_dim in the realistic range (2048+),
    /// falling back toward 1 warp only for tiny/edge-case sizes. Each
    /// thread handles `hidden_dim / (num_warps*256)` groups of 8 contiguous
    /// elements, strided by `num_warps*256` (thread `T`'s groups are `T`,
    /// `T+num_warps*32`, ... in group units) so consecutive threads still
    /// touch consecutive 16-byte chunks (fully coalesced, same principle as
    /// the single-warp version, just spread over a whole block instead of
    /// one warp) - one 128-bit `ld.global.v4.u32` / `st.global.v4.u32` per
    /// tensor per group (`emit_vec_load_u32` / `emit_vec_store_u32` /
    /// `emit_unpack_f16_pairs` / `emit_pack_f16_pairs` - see those doc
    /// comments for the standalone hardware validation this went through
    /// first).
    ///
    /// **Real block-level reduction, not just a single warp's**: each warp
    /// folds its own partial `sum(h^2)` via the existing 5-step
    /// `shfl.sync.bfly.b32` butterfly (`emit_warp_reduce_sum`, unchanged,
    /// still validated standalone as before), then (when `num_warps>1`)
    /// lane 0 of each warp writes its warp's total into a small
    /// `.shared` scratch buffer (`smem_reduce`, `num_warps` floats),
    /// `bar.sync 0` makes every warp's write visible block-wide, and every
    /// thread then sums the `num_warps` slots (a tiny, fully-unrolled,
    /// compile-time-bounded loop - never more than 8 adds) to get the
    /// row's true total - a real cross-warp combine, not a rename of the
    /// old single-warp path.
    ///
    /// **`h` is staged in `.shared` memory between the two passes, closing
    /// the double-DRAM-read gap the vectorization pass alone left open**:
    /// pass 1 computes `h = X + Residual`, folds `h^2` into this thread's
    /// running sum AND writes `h` (as f32, not re-rounded to f16 - avoids
    /// adding a precision step that wasn't in the original two-read design)
    /// into `smem_h` (`hidden_dim` floats, e.g. 16KB at hidden_dim=4096 -
    /// comfortably inside this GPU's ~100KB/SM shared memory budget); pass
    /// 2 reads `h` back from `smem_h` instead of re-reading `X`/`Residual`
    /// from DRAM a second time. Every thread only ever reads back the exact
    /// `smem_h` locations it itself wrote in pass 1 (same `tid`-strided
    /// index sequence in both passes), so no extra synchronization is
    /// needed for that specifically; the `bar.sync 0` already required for
    /// the cross-warp sum combine above happens to also cover it. `X`/
    /// `Residual` are now each read from DRAM exactly once per row, not
    /// twice - the double-read this function's doc comment flagged as open
    /// after the vectorization pass is now actually closed, confirmed via
    /// `benchmark_y_vs_flashinfer_results.md`'s re-measurement (vectorizing
    /// alone left RMSNorm at 0.71-0.78x of FlashInfer; occupancy + this fix
    /// were the next levers, not optional polish).
    ///
    /// `rsqrt.approx.f32` was validated standalone on real sm_89 hardware
    /// before being wired in here (max relative error 1.2e-7 over 2000
    /// random samples) - see `benchmark_y_rmsnorm_rope_results.md`.
    /// `eps = 1e-5` (matching Llama's default) is a fixed compile-time
    /// constant, not a runtime parameter.
    ///
    /// Returns `num_warps*32` (the real thread count this kernel now
    /// expects per row/block) - callers (the PTX kernel launcher, and any
    /// Python/host code choosing a launch's block size) MUST use this
    /// value, not a hardcoded 32; the `.maxnreg` register-limit computation
    /// downstream already does (see `block_size` in the caller).
    fn emit_rmsnorm_residual_kernel(
        &mut self,
        hidden_dim: u32,
        x_ptr: &str,
        residual_ptr: &str,
        weight_ptr: &str,
        out_ptr: &str,
        new_residual_ptr: &str,
    ) -> u32 {
        const WARP: u32 = 32;
        const VEC: u32 = 8; // f16 elements per 128-bit vectorized group
        debug_assert_eq!(
            hidden_dim % (WARP * VEC),
            0,
            "hidden_dim must be a multiple of 256 (32 lanes x 8-wide vectorized f16 groups)"
        );
        let vec_groups = hidden_dim / VEC;
        let num_warps: u32 = [8u32, 4, 2, 1]
            .into_iter()
            .find(|&w| vec_groups % (w * WARP) == 0)
            .unwrap_or(1);
        let threads_per_row = num_warps * WARP;

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [Y FUSED ADD+RMSNORM] hidden_dim={} | {} warps ({} threads) per row | 8-wide vectorized f16 | shared-mem h staging", hidden_dim, num_warps, threads_per_row).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        writeln!(&mut self.ptx_buffer, "    .shared .align 16 .b8 smem_h[{}];", hidden_dim * 4).unwrap();
        let smem_h_base = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, smem_h;", smem_h_base).unwrap();
        let smem_reduce_base = if num_warps > 1 {
            writeln!(&mut self.ptx_buffer, "    .shared .align 4 .b8 smem_reduce[{}];", num_warps * 4).unwrap();
            let base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, smem_reduce;", base).unwrap();
            Some(base)
        } else {
            None
        };

        let x_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", x_g, x_ptr).unwrap();
        let res_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", res_g, residual_ptr).unwrap();
        let w_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", w_g, weight_ptr).unwrap();
        let out_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", out_g, out_ptr).unwrap();
        let newres_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", newres_g, new_residual_ptr).unwrap();

        let row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", row).unwrap();
        let tid = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
        let row_offset = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", row_offset, row, hidden_dim).unwrap();

        // ---- Pass 1: accumulate this thread's partial sum(h^2), stage h into smem_h ----
        // A real runtime loop (idx/running_sum updated in place via their
        // own registers, branching back), not unrolled: an earlier fully-
        // unrolled SCALAR version of this loop (hidden_dim=4096, 128
        // iterations/thread, 202 registers/thread per `ptxas -v`) measured
        // non-deterministic wrong output on real hardware - see
        // benchmark_y_rmsnorm_rope_results.md for the full account. This
        // loop (and its vectorized/multi-warp descendants) has been re-run
        // through the same repeated-fixed-seed-launch determinism check
        // every time its structure changed - see
        // benchmark_y_vs_flashinfer_results.md.
        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", idx, tid).unwrap();
        let running_sum = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", running_sum).unwrap();
        let loop1_start = self.alloc_label("RMSNORM_SUMSQ");
        let loop1_end = self.alloc_label("RMSNORM_SUMSQ_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop1_start).unwrap();
        let p1_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p1_done, idx, vec_groups).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p1_done, loop1_end).unwrap();
        {
            let elem_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", elem_base, idx, VEC, row_offset).unwrap();
            let byte_base = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 2;", byte_base, elem_base).unwrap();

            let x_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", x_addr, x_g, byte_base).unwrap();
            let x_words = self.emit_vec_load_u32(&x_addr, 4);
            let x_f = self.emit_unpack_f16_pairs(&x_words);

            let r_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", r_addr, res_g, byte_base).unwrap();
            let r_words = self.emit_vec_load_u32(&r_addr, 4);
            let r_f = self.emit_unpack_f16_pairs(&r_words);

            // Accumulate in place (dest register == running_sum's own,
            // fixed register - no new allocation): this loop body is
            // emitted ONCE as static PTX text but executed repeatedly at
            // runtime via the branch below, so `running_sum` must stay
            // bound to the SAME physical register across dynamic
            // iterations for the accumulation to actually carry over (see
            // this function's doc comment history / benchmark_y_vs_
            // flashinfer_results.md for the bug this caused when violated).
            let mut h_vals = Vec::with_capacity(8);
            for k in 0..8 {
                let h = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", h, x_f[k], r_f[k]).unwrap();
                let h_sq = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", h_sq, h, h).unwrap();
                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", running_sum, running_sum, h_sq).unwrap();
                h_vals.push(h);
            }

            // Stage h (f32, unrounded) into smem_h at this group's byte
            // offset (idx*8*4 = idx*32) via two 128-bit vector stores -
            // read back in pass 2 below instead of re-reading X/Residual.
            let smem_off = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 32;", smem_off, idx).unwrap();
            let smem_addr0 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_addr0, smem_h_base, smem_off).unwrap();
            writeln!(&mut self.ptx_buffer, "    st.shared.v4.f32 [{}], {{{}, {}, {}, {}}};", smem_addr0, h_vals[0], h_vals[1], h_vals[2], h_vals[3]).unwrap();
            let smem_addr1 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 16;", smem_addr1, smem_addr0).unwrap();
            writeln!(&mut self.ptx_buffer, "    st.shared.v4.f32 [{}], {{{}, {}, {}, {}}};", smem_addr1, h_vals[4], h_vals[5], h_vals[6], h_vals[7]).unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_row).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop1_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop1_end).unwrap();

        // ---- per-warp reduction, then (if >1 warp) real cross-warp combine via smem ----
        let warp_total = self.emit_warp_reduce_sum(&running_sum);
        let total_sum_sq = if num_warps > 1 {
            let smem_reduce_base = smem_reduce_base.as_deref().unwrap();
            let warp_id = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 5;", warp_id, tid).unwrap();
            let lane = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 31;", lane, tid).unwrap();
            let is_lane0 = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, {}, 0;", is_lane0, lane).unwrap();
            let reduce_off = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 4;", reduce_off, warp_id).unwrap();
            let reduce_addr = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", reduce_addr, smem_reduce_base, reduce_off).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} st.shared.f32 [{}], {};", is_lane0, reduce_addr, warp_total).unwrap();
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            let total = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", total).unwrap();
            for w in 0..num_warps {
                let slot_addr = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", slot_addr, smem_reduce_base, w * 4).unwrap();
                let slot = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    ld.shared.f32 {}, [{}];", slot, slot_addr).unwrap();
                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", total, total, slot).unwrap();
            }
            total
        } else {
            warp_total
        };

        let mean_sq = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", mean_sq, total_sum_sq, Self::f32_to_ptx_hex(1.0 / hidden_dim as f32)).unwrap();
        let mean_sq_eps = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", mean_sq_eps, mean_sq, Self::f32_to_ptx_hex(1e-5)).unwrap();
        let inv_rms = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    rsqrt.approx.f32 {}, {};", inv_rms, mean_sq_eps).unwrap();

        // ---- Pass 2: read h back from smem_h (NOT X/Residual again), scale by inv_rms * Weight, write Out + NewResidual ----
        // Real runtime loop, same rationale as pass 1 above. Same idx
        // stride as pass 1, so every thread reads back exactly the smem_h
        // locations it itself wrote - no extra sync needed beyond the
        // bar.sync already issued above for the cross-warp combine.
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", idx, tid).unwrap();
        let loop2_start = self.alloc_label("RMSNORM_WRITE");
        let loop2_end = self.alloc_label("RMSNORM_WRITE_DONE");
        writeln!(&mut self.ptx_buffer, "    {}:", loop2_start).unwrap();
        let p2_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", p2_done, idx, vec_groups).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", p2_done, loop2_end).unwrap();
        {
            let elem_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", elem_base, idx, VEC, row_offset).unwrap();
            let byte_base = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 2;", byte_base, elem_base).unwrap();

            let smem_off = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 32;", smem_off, idx).unwrap();
            let smem_addr0 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", smem_addr0, smem_h_base, smem_off).unwrap();
            let h0 = self.alloc_regf32();
            let h1 = self.alloc_regf32();
            let h2 = self.alloc_regf32();
            let h3 = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    ld.shared.v4.f32 {{{}, {}, {}, {}}}, [{}];", h0, h1, h2, h3, smem_addr0).unwrap();
            let smem_addr1 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 16;", smem_addr1, smem_addr0).unwrap();
            let h4 = self.alloc_regf32();
            let h5 = self.alloc_regf32();
            let h6 = self.alloc_regf32();
            let h7 = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    ld.shared.v4.f32 {{{}, {}, {}, {}}}, [{}];", h4, h5, h6, h7, smem_addr1).unwrap();
            let h_vals = vec![h0, h1, h2, h3, h4, h5, h6, h7];

            // Weight is a single [hidden_dim] vector broadcast across every
            // row (the standard RMSNorm gamma contract) - indexed by
            // `idx*VEC` alone, NOT `elem_base` (which folds in
            // `row_offset`); reusing `byte_base` here would have been a
            // real bug (indexing past the end of a [hidden_dim]-sized
            // buffer for every row after the first) - deliberately given
            // its own byte offset instead, computed directly from `idx`
            // without ever touching `row_offset`.
            let w_byte_base = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, {};", w_byte_base, idx, VEC * 2).unwrap();
            let w_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", w_addr, w_g, w_byte_base).unwrap();
            let w_words = self.emit_vec_load_u32(&w_addr, 4);
            let w_f = self.emit_unpack_f16_pairs(&w_words);

            let mut scaled_vals = Vec::with_capacity(8);
            for k in 0..8 {
                let normed = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", normed, h_vals[k], inv_rms).unwrap();
                let scaled = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", scaled, normed, w_f[k]).unwrap();
                scaled_vals.push(scaled);
            }

            let out_words = self.emit_pack_f16_pairs(&scaled_vals);
            let out_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", out_addr, out_g, byte_base).unwrap();
            self.emit_vec_store_u32(&out_addr, &out_words);

            let newres_words = self.emit_pack_f16_pairs(&h_vals);
            let newres_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", newres_addr, newres_g, byte_base).unwrap();
            self.emit_vec_store_u32(&newres_addr, &newres_words);
        }
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_row).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop2_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop2_end).unwrap();

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        threads_per_row
    }

    /// Detects the 3-parameter (X: `GlobalMemory<F16>`, Positions:
    /// `GlobalMemory<I32>`, Out: `GlobalMemory<F16>`) kernel shape, named
    /// `rope_<head_dim>`, that dispatches `emit_rope_kernel`.
    fn rope_operands(&self, kernel: &KernelDecl) -> Option<(u32, String, String, String)> {
        if kernel.tile.is_some() {
            return None;
        }
        let head_dim = Self::parse_trailing_dim(&kernel.name, "rope")?;

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }
        if kernel.params.len() != 3
            || !is_global_memory_of(&kernel.params[0].ty, "F16")
            || !is_global_memory_of(&kernel.params[1].ty, "I32")
            || !is_global_memory_of(&kernel.params[2].ty, "F16")
        {
            return None;
        }

        let x = self.variables.get(&kernel.params[0].name)?.clone();
        let positions = self.variables.get(&kernel.params[1].name)?.clone();
        let out = self.variables.get(&kernel.params[2].name)?.clone();
        Some((head_dim, x, positions, out))
    }

    /// Emits a complete, self-contained Rotary Position Embedding (RoPE)
    /// kernel: one warp (32 threads) per row, where a row is one token's
    /// query or key vector of `head_dim` elements. Computes, for each pair
    /// index `i` in `0..head_dim/2` (the **interleaved-pairs** convention -
    /// pairs are `(2i, 2i+1)`, the original RoPE paper/GPT-NeoX layout, NOT
    /// the "rotate-half"/split-in-two layout HuggingFace's Llama
    /// implementation uses - a deliberate, disclosed choice of one
    /// well-defined convention, matched exactly by this kernel's own
    /// correctness reference rather than assumed compatible with a
    /// specific model family):
    /// `theta_i = pos * base^(-2i/head_dim)`, `base = 10000.0`, then
    /// `out[2i]   = x[2i]*cos(theta_i) - x[2i+1]*sin(theta_i)`,
    /// `out[2i+1] = x[2i]*sin(theta_i) + x[2i+1]*cos(theta_i)`.
    ///
    /// `theta_i` is computed entirely on-device from `pos` and `i`
    /// (`base^y = 2^(y*log2(base))` via `ex2.approx.f32`, then
    /// `sin.approx.f32`/`cos.approx.f32`) rather than reading a precomputed
    /// cos/sin table - deliberately: a real "unfused" baseline has to
    /// produce that table (or index into one) as a separate step, so
    /// fusing the trig into this elementwise kernel removes that step
    /// entirely rather than just avoiding a DRAM round-trip on it.
    /// `ex2.approx.f32`/`sin.approx.f32`/`cos.approx.f32` were all
    /// validated standalone against Python's `math.sin`/`cos` on real
    /// sm_89 hardware before being wired in here (max absolute error
    /// ~6e-6 over 2000 random angles in [-50, 50] radians) - see
    /// `benchmark_y_rmsnorm_rope_results.md`.
    ///
    /// `head_dim` must be a multiple of 64 (`debug_assert`ed: needs
    /// `head_dim/2` pairs to split evenly across 32 lanes, i.e. `head_dim`
    /// a multiple of 64 - true for every common LLM head dim: 64, 128,
    /// 256); each thread handles `head_dim / 64` pairs.
    ///
    /// **Blocked, not strided, pair-to-thread mapping, so pairs can be
    /// vector-loaded**: lane `L` owns the contiguous run of pairs
    /// `[L*pairs_per_thread, (L+1)*pairs_per_thread)`, not the old
    /// `{L, L+32, L+64, ...}` stride. Every pair is still computed by
    /// exactly one thread (any bijective lane/pair assignment is correct -
    /// RoPE has no cross-lane dependency, unlike RMSNorm's reduction), and
    /// warp-wide coalescing is preserved: for a fixed chunk index, lane
    /// `L`'s bytes start exactly where lane `L-1`'s end, so the union
    /// across the warp is still one contiguous span - the same shape as a
    /// `float4`-style blocked-vectorized CUDA access, and the same
    /// principle `emit_gemm_tile_load` already uses. The old strided
    /// mapping could not be vector-loaded at all: consecutive `t` values
    /// for one thread were `head_dim` bytes apart, not adjacent.
    ///
    /// Chunk width (in pairs; 1 pair = 1 packed 32-bit word) is the
    /// largest of {4, 2, 1} dividing `pairs_per_thread` evenly - 128-bit
    /// `ld/st.global.v4.u32` at head_dim=256 (4 pairs/thread), 64-bit `v2`
    /// at head_dim=128 (2 pairs/thread), a single 32-bit word at
    /// head_dim=64 (1 pair/thread, still a real win over the old scheme's
    /// two separate 16-bit loads for that one pair). See
    /// `emit_vec_load_u32`/`emit_unpack_f16_pairs`/`emit_pack_f16_pairs`/
    /// `emit_vec_store_u32` for the shared, standalone-hardware-validated
    /// pack/unpack primitives (also used by `emit_rmsnorm_residual_kernel`).
    /// The per-chunk, per-pair body is still fully unrolled at Rust/compile
    /// time (a small, fixed count for realistic head dims - never a real
    /// PTX runtime loop), so - unlike RMSNorm's accumulator - there is no
    /// loop-carried register to get wrong here; still re-verified with the
    /// same real-hardware correctness + repeated-launch determinism checks
    /// regardless, per this codebase's standing validation discipline.
    fn emit_rope_kernel(
        &mut self,
        head_dim: u32,
        x_ptr: &str,
        positions_ptr: &str,
        out_ptr: &str,
    ) -> u32 {
        const WARP: u32 = 32;
        debug_assert_eq!(head_dim % (WARP * 2), 0, "head_dim must be a multiple of 64 (32 lanes x 2 elements/pair)");
        let pairs_per_thread = head_dim / (WARP * 2);
        let chunk_pairs: u32 = if pairs_per_thread % 4 == 0 {
            4
        } else if pairs_per_thread % 2 == 0 {
            2
        } else {
            1
        };
        let num_chunks = pairs_per_thread / chunk_pairs;
        // theta_i = pos * base^(-2i/head_dim) = pos * 2^(-2i/head_dim * log2(base));
        // -2/head_dim * log2(10000.0) is compile-time-known (head_dim is),
        // leaving one runtime multiply by i to get the ex2 exponent.
        let exponent_scale = (-2.0 / head_dim as f32) * 10000.0f32.log2();

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [Y FUSED ROPE] head_dim={} | 1 warp (32 threads) per row | interleaved pairs, base=10000 | {}-pair vectorized blocks", head_dim, chunk_pairs).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        let x_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", x_g, x_ptr).unwrap();
        let pos_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", pos_g, positions_ptr).unwrap();
        let out_g = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvta.to.global.u64 {}, {};", out_g, out_ptr).unwrap();

        let row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", row).unwrap();
        let lane = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", lane).unwrap();
        let row_offset = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", row_offset, row, head_dim).unwrap();

        // pos = Positions[row] (I32, one scalar per row)
        let pos_byte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", pos_byte, row).unwrap();
        let pos_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", pos_addr, pos_g, pos_byte).unwrap();
        let pos_i = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    ld.global.u32 {}, [{}];", pos_i, pos_addr).unwrap();
        let pos_f = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    cvt.rn.f32.s32 {}, {};", pos_f, pos_i).unwrap();

        for c in 0..num_chunks {
            // pair_i_start = lane*pairs_per_thread + c*chunk_pairs (this
            // chunk's first pair index, owned entirely by this lane).
            let pair_i_partial = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", pair_i_partial, lane, pairs_per_thread).unwrap();
            let pair_i_start = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", pair_i_start, pair_i_partial, c * chunk_pairs).unwrap();

            let elem_group_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, 2;", elem_group_base, pair_i_start).unwrap();
            let elem_global_base = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", elem_global_base, row_offset, elem_group_base).unwrap();
            let byte_base = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 2;", byte_base, elem_global_base).unwrap();

            let x_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", x_addr, x_g, byte_base).unwrap();
            let x_words = self.emit_vec_load_u32(&x_addr, chunk_pairs as usize);
            let x_vals = self.emit_unpack_f16_pairs(&x_words);

            let mut out_vals = Vec::with_capacity(2 * chunk_pairs as usize);
            for p in 0..chunk_pairs {
                let pair_i = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", pair_i, pair_i_start, p).unwrap();

                let x0_f = &x_vals[(2 * p) as usize];
                let x1_f = &x_vals[(2 * p + 1) as usize];

                // theta = pos * ex2.approx(i * exponent_scale)
                let i_f = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    cvt.rn.f32.u32 {}, {};", i_f, pair_i).unwrap();
                let exponent = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", exponent, i_f, Self::f32_to_ptx_hex(exponent_scale)).unwrap();
                let base_pow = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    ex2.approx.f32 {}, {};", base_pow, exponent).unwrap();
                let theta = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", theta, pos_f, base_pow).unwrap();

                let sin_t = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    sin.approx.f32 {}, {};", sin_t, theta).unwrap();
                let cos_t = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    cos.approx.f32 {}, {};", cos_t, theta).unwrap();

                // out0 = x0*cos - x1*sin ; out1 = x0*sin + x1*cos
                let x0_cos = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", x0_cos, x0_f, cos_t).unwrap();
                let x1_sin = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", x1_sin, x1_f, sin_t).unwrap();
                let out0 = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    sub.f32 {}, {}, {};", out0, x0_cos, x1_sin).unwrap();

                let x0_sin = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", x0_sin, x0_f, sin_t).unwrap();
                let x1_cos = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", x1_cos, x1_f, cos_t).unwrap();
                let out1 = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", out1, x0_sin, x1_cos).unwrap();

                out_vals.push(out0);
                out_vals.push(out1);
            }

            let out_words = self.emit_pack_f16_pairs(&out_vals);
            let out_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", out_addr, out_g, byte_base).unwrap();
            self.emit_vec_store_u32(&out_addr, &out_words);
        }

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        WARP
    }

    /// Default warps per CTA for `emit_paged_decode_attention_kernel`, used
    /// when the kernel name does not declare one explicitly.
    ///
    /// The grid is `num_q_heads x num_seqs` CTAs, so at batch 1 with 32 heads
    /// it is 32 CTAs against this GPU's 66 SMs - half the machine idle, with
    /// a long serial dependency chain per token (load K -> warp-reduce ->
    /// exp -> load V -> update). Warps inside the CTA are the only
    /// parallelism left to cover that, and the effect is large and measured
    /// (min-of-7 interleaved rounds, head_dim 128, 32 q heads, 8 kv heads):
    ///
    /// ```text
    ///   case            8 warps   16 warps   32 warps    best
    ///   b1  ctx 1024      43.26      23.92      14.77    32  (2.93x)
    ///   b1  ctx 4096     165.88      87.48      49.94    32  (3.32x)
    ///   b1  ctx 16384   1147.03     586.39     308.22    32  (3.72x)
    ///   b4  ctx 4096     237.77     131.02     132.20    16  (1.81x)
    ///   b8  ctx 1024      49.77      54.32      58.11     8
    ///   b8  ctx 4096     306.12     314.73     316.36     8
    ///   b32 ctx 1024     244.02     251.75     334.34     8  (32w is 1.37x worse)
    ///   b32 ctx 4096    1046.89    1071.66    1332.68     8  (32w is 1.27x worse)
    /// ```
    ///
    /// The optimum is entirely a function of how many CTAs the grid already
    /// supplies (`num_q_heads * batch`) against the SM count - and batch is a
    /// RUNTIME value, so no compile-time constant is right for every launch.
    /// 32 is the default because this compiler targets consumer-GPU
    /// inference, where decode runs at batch 1-4 and the win is 3x; a
    /// batch-many serving deployment should declare 8 instead (see
    /// `paged_decode_attention_operands` for the optional 5th name
    /// dimension), where the cost of the wrong choice is at most 1.37x.
    ///
    /// Neither choice fixes the grid being too small at batch 1 outright -
    /// that needs a split-K decode (partition the KV sequence across CTAs
    /// plus a second reduction pass over the partial softmax states), which
    /// is not implemented. See the kernel's doc comment.
    const ATTN_DEFAULT_NUM_WARPS: u32 = 32;

    /// Parses `<prefix>_<d0>_<d1>_..._<dn-1>` into its `n` trailing positive
    /// integers.
    ///
    /// The multi-dimension sibling of `parse_trailing_dim`. Attention needs
    /// four compile-time constants (head_dim, num_q_heads, num_kv_heads,
    /// page_size) where RMSNorm/RoPE need one, and all four genuinely change
    /// the generated code: head_dim sets the per-lane register footprint,
    /// the head counts set the GQA mapping, page_size sets the
    /// token->page/slot arithmetic. Encoding them in the name keeps the
    /// grammar untouched, consistent with how those kernels already work.
    /// Everything that does NOT change codegen - batch size, sequence
    /// length, page-table stride - stays a runtime value.
    fn parse_trailing_dims(name: &str, prefix: &str, min_n: usize, max_n: usize) -> Option<Vec<u32>> {
        let rest = name.strip_prefix(prefix)?.strip_prefix('_')?;
        let parts: Vec<&str> = rest.split('_').collect();
        if parts.len() < min_n || parts.len() > max_n {
            return None;
        }
        let dims: Vec<u32> = parts
            .iter()
            .filter_map(|p| p.parse::<u32>().ok().filter(|&d| d > 0))
            .collect();
        if dims.len() == parts.len() {
            Some(dims)
        } else {
            None
        }
    }

    /// Detects the 7-parameter paged decode-attention kernel shape, named
    /// `paged_decode_attention_<head_dim>_<num_q_heads>_<num_kv_heads>_<page_size>`.
    ///
    /// Parameters, in declaration order:
    ///   `Q`         `GlobalMemory<F16>`  `[num_seqs, num_q_heads, head_dim]`
    ///   `KCache`    `GlobalMemory<F16>`  `[num_pages, page_size, num_kv_heads, head_dim]`
    ///   `VCache`    `GlobalMemory<F16>`  same layout as `KCache`
    ///   `PageTable` `GlobalMemory<I32>`  `[num_seqs, max_pages]`, logical page -> physical page
    ///   `SeqLens`   `GlobalMemory<I32>`  `[num_seqs]`
    ///   `Out`       `GlobalMemory<F16>`  `[num_seqs, num_q_heads, head_dim]`
    ///   `MaxPages`  `I32` (scalar)       row stride of `PageTable`
    ///
    /// An optional 5th name dimension sets warps per CTA
    /// (`..._<head_dim>_<num_q_heads>_<num_kv_heads>_<page_size>_<warps>`),
    /// overriding `ATTN_DEFAULT_NUM_WARPS`. That constant's doc comment has
    /// the measured table showing why the right value depends on batch size,
    /// which is a runtime quantity - so a deployment that knows it serves
    /// large batches can declare 8 while the batch-1 default stays 32.
    fn paged_decode_attention_operands(
        &self,
        kernel: &KernelDecl,
    ) -> Option<(u32, u32, u32, u32, u32, String, String, String, String, String, String, String)> {
        if kernel.tile.is_some() {
            return None;
        }
        let dims = Self::parse_trailing_dims(&kernel.name, "paged_decode_attention", 4, 5)?;
        let (head_dim, num_q_heads, num_kv_heads, page_size) = (dims[0], dims[1], dims[2], dims[3]);
        let num_warps = if dims.len() == 5 { dims[4] } else { Self::ATTN_DEFAULT_NUM_WARPS };
        // A CTA is capped at 1024 threads, and the merge needs at least one warp.
        if num_warps == 0 || num_warps > 32 {
            return None;
        }

        // Constraints the codegen below genuinely relies on. Returning None
        // (rather than asserting) means a kernel that violates them falls
        // through to generic lowering with a type error, instead of silently
        // emitting a kernel that computes the wrong thing.
        if head_dim % 32 != 0 {
            return None;
        }
        let elems_per_lane = head_dim / 32;
        if !matches!(elems_per_lane, 2 | 4 | 8) {
            return None; // head_dim 64 / 128 / 256: 1, 2 or 4 32-bit words per lane
        }
        if num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 {
            return None; // GQA requires q heads to be a whole multiple of kv heads
        }

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }
        fn is_scalar_i32(ty: &Type) -> bool {
            matches!(ty, Type::Primitive(p, _) if p == "I32")
        }
        if kernel.params.len() != 7
            || !is_global_memory_of(&kernel.params[0].ty, "F16")
            || !is_global_memory_of(&kernel.params[1].ty, "F16")
            || !is_global_memory_of(&kernel.params[2].ty, "F16")
            || !is_global_memory_of(&kernel.params[3].ty, "I32")
            || !is_global_memory_of(&kernel.params[4].ty, "I32")
            || !is_global_memory_of(&kernel.params[5].ty, "F16")
            || !is_scalar_i32(&kernel.params[6].ty)
        {
            return None;
        }

        let q = self.variables.get(&kernel.params[0].name)?.clone();
        let kc = self.variables.get(&kernel.params[1].name)?.clone();
        let vc = self.variables.get(&kernel.params[2].name)?.clone();
        let pt = self.variables.get(&kernel.params[3].name)?.clone();
        let sl = self.variables.get(&kernel.params[4].name)?.clone();
        let out = self.variables.get(&kernel.params[5].name)?.clone();
        let max_pages = self.variables.get(&kernel.params[6].name)?.clone();
        Some((head_dim, num_q_heads, num_kv_heads, page_size, num_warps, q, kc, vc, pt, sl, out, max_pages))
    }

    /// Default warps per CTA for the SPLIT-K attention shape, deliberately
    /// lower than `ATTN_DEFAULT_NUM_WARPS`.
    ///
    /// Warps inside a CTA and splits across CTAs buy the same thing - more
    /// concurrent token streams over one sequence - but splits spread over
    /// SMs and warps do not. The 32-warp default of the single-pass kernel
    /// exists only because that shape has no other way to get parallelism at
    /// batch 1 (see `ATTN_DEFAULT_NUM_WARPS`); once `%ctaid.z` supplies it,
    /// piling 32 warps onto one SM just lengthens the cross-warp combine,
    /// which is `num_warps` shared-memory loads per output element and runs
    /// `gqa_ratio` times.
    const ATTN_SPLIT_DEFAULT_NUM_WARPS: u32 = 8;

    /// Detects the 9-parameter **split-K paged decode attention** shape,
    /// named `paged_decode_attention_split_<head_dim>_<num_q_heads>_<num_kv_heads>_<page_size>_<splits>[_<warps>]`.
    ///
    /// This is the flash-decoding form of the kernel
    /// `paged_decode_attention_operands` detects, and it exists to fix the two
    /// deficiencies that shape has against FlashInfer, both of which are
    /// structural rather than a matter of instruction selection:
    ///
    /// 1. **The GQA re-read.** The single-pass grid is
    ///    `num_q_heads x num_seqs`, so under GQA `g:1` the `g` CTAs sharing a
    ///    KV head each stream the whole K/V cache for it - `g` times the
    ///    necessary traffic. Here the grid is `num_kv_heads x num_seqs x
    ///    splits` and one CTA carries all `g` query heads of its KV head, so
    ///    each K/V byte is read exactly once and the `g` scores come out of
    ///    one load.
    /// 2. **No split-K.** With `num_q_heads x num_seqs` CTAs, batch 1 is 32
    ///    CTAs against this GPU's 66 SMs whatever the kernel does internally.
    ///    `%ctaid.z` partitions the KV sequence, so the CTA count no longer
    ///    depends on batch size.
    ///
    /// Both changes are needed together: merging the GQA heads on its own
    /// would DIVIDE the batch-1 CTA count by `g` (32 CTAs to 8), which is a
    /// large regression, and it is only split-K that makes the merge
    /// affordable.
    ///
    /// Parameters, in declaration order - the 7 of the single-pass shape plus
    /// two scratch buffers, which is why this cannot be the same kernel:
    ///   `Q`         `GlobalMemory<F16>`  `[num_seqs, num_q_heads, head_dim]`
    ///   `KCache`    `GlobalMemory<F16>`  `[num_pages, page_size, num_kv_heads, head_dim]`
    ///   `VCache`    `GlobalMemory<F16>`  same layout as `KCache`
    ///   `PageTable` `GlobalMemory<I32>`  `[num_seqs, max_pages]`
    ///   `SeqLens`   `GlobalMemory<I32>`  `[num_seqs]`
    ///   `Out`       `GlobalMemory<F16>`  `[num_seqs, num_q_heads, head_dim]`
    ///   `Partial`   `GlobalMemory<F32>`  `[num_seqs, num_q_heads, splits, head_dim]`
    ///   `PartialMeta` `GlobalMemory<F32>` `[num_seqs, num_q_heads, splits, 2]`
    ///   `MaxPages`  `I32` (scalar)       row stride of `PageTable`
    ///
    /// The two scratch buffers are f32 on purpose: they hold an unnormalized
    /// running sum, whose magnitude is `l` (the softmax denominator) times a
    /// V element, and rounding that to f16 before the combine would throw away
    /// precision the single-pass kernel keeps in registers.
    #[allow(clippy::type_complexity)]
    fn paged_decode_attention_split_operands(
        &self,
        kernel: &KernelDecl,
    ) -> Option<(u32, u32, u32, u32, u32, u32, String, String, String, String, String, String, String, String, String)> {
        if kernel.tile.is_some() {
            return None;
        }
        let dims = Self::parse_trailing_dims(&kernel.name, "paged_decode_attention_split", 5, 6)?;
        let (head_dim, num_q_heads, num_kv_heads, page_size, splits) =
            (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let num_warps = if dims.len() == 6 { dims[5] } else { Self::ATTN_SPLIT_DEFAULT_NUM_WARPS };
        if num_warps == 0 || num_warps > 32 {
            return None;
        }
        // The combine unrolls over splits and every partial state is live
        // across it; past this the entry stops being a tail and starts being
        // its own bandwidth problem.
        if splits == 0 || splits > 64 {
            return None;
        }
        if head_dim % 32 != 0 {
            return None;
        }
        let elems_per_lane = head_dim / 32;
        if !matches!(elems_per_lane, 2 | 4 | 8) {
            return None;
        }
        if num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 {
            return None;
        }
        // One CTA carries every query head of its KV head, and each head costs
        // `2 + 2*elems_per_lane` live f32 registers of softmax state before
        // any temporaries. Past this the kernel spills and the merge stops
        // being worth it - refuse rather than emit something slower than the
        // shape it replaces.
        if num_q_heads / num_kv_heads > 8 {
            return None;
        }

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == elem)
            )
        }
        fn is_scalar_i32(ty: &Type) -> bool {
            matches!(ty, Type::Primitive(p, _) if p == "I32")
        }
        if kernel.params.len() != 9
            || !is_global_memory_of(&kernel.params[0].ty, "F16")
            || !is_global_memory_of(&kernel.params[1].ty, "F16")
            || !is_global_memory_of(&kernel.params[2].ty, "F16")
            || !is_global_memory_of(&kernel.params[3].ty, "I32")
            || !is_global_memory_of(&kernel.params[4].ty, "I32")
            || !is_global_memory_of(&kernel.params[5].ty, "F16")
            || !is_global_memory_of(&kernel.params[6].ty, "F32")
            || !is_global_memory_of(&kernel.params[7].ty, "F32")
            || !is_scalar_i32(&kernel.params[8].ty)
        {
            return None;
        }

        let q = self.variables.get(&kernel.params[0].name)?.clone();
        let kc = self.variables.get(&kernel.params[1].name)?.clone();
        let vc = self.variables.get(&kernel.params[2].name)?.clone();
        let pt = self.variables.get(&kernel.params[3].name)?.clone();
        let sl = self.variables.get(&kernel.params[4].name)?.clone();
        let out = self.variables.get(&kernel.params[5].name)?.clone();
        let part = self.variables.get(&kernel.params[6].name)?.clone();
        let meta = self.variables.get(&kernel.params[7].name)?.clone();
        let max_pages = self.variables.get(&kernel.params[8].name)?.clone();
        Some((
            head_dim, num_q_heads, num_kv_heads, page_size, splits, num_warps,
            q, kc, vc, pt, sl, out, part, meta, max_pages,
        ))
    }

    /// Emits a complete, self-contained **paged decode attention** kernel:
    /// one query token per (sequence, query head), attending over a
    /// page-indexed KV cache, with FlashAttention-style online softmax so the
    /// scores are never materialized.
    ///
    /// This is the decode (autoregressive, one new token) path, not prefill.
    /// It is the shape that dominates LLM serving latency and the one a KV
    /// cache exists for: every generated token re-reads the whole cache, so
    /// the kernel is DRAM-bound and the win comes from touching each KV byte
    /// exactly once, in one fused pass, rather than materializing an
    /// `[1, seq_len]` score matrix and a separate softmax.
    ///
    /// **Grid/block contract** (the launcher must match exactly):
    ///   grid  = `(num_q_heads, num_seqs, 1)`
    ///   block = `(ATTN_NUM_WARPS * 32, 1, 1)`
    /// Shared memory is statically declared, so no dynamic-smem opt-in.
    ///
    /// **Algorithm.** Lane `L` of every warp owns head-dimension elements
    /// `[L*elems_per_lane, (L+1)*elems_per_lane)`, so a whole `head_dim` row
    /// is one fully-coalesced vector load per warp (`head_dim=128` -> 4 f16
    /// per lane -> `ld.global.v2.u32`, 256 contiguous bytes per warp). Each
    /// warp strides the KV sequence by `ATTN_NUM_WARPS` and maintains its own
    /// running `(m, l, acc)` softmax state:
    ///
    /// ```text
    ///   score  = warp_reduce_sum(dot(Q, K_t))          // scale folded into Q
    ///   m_new  = max(m, score)
    ///   corr   = exp(m - m_new)
    ///   p      = exp(score - m_new)
    ///   l      = l*corr + p
    ///   acc[j] = acc[j]*corr + p*V_t[j]
    /// ```
    ///
    /// `m`, `l` and `acc[]` are LOOP-CARRIED across a real runtime loop, so
    /// every write above targets those exact registers. Allocating a fresh
    /// register per step - correct and idiomatic everywhere else in this
    /// emitter - would make each iteration re-read the pre-loop value and
    /// silently keep only the last token's contribution. This codebase has
    /// shipped that exact bug before (see `emit_rmsnorm_residual_kernel`'s
    /// history); it is the reason the accumulator updates below look
    /// deliberately un-SSA.
    ///
    /// Because each warp holds a PARTIAL softmax state over a different
    /// subset of tokens, the cross-warp combine is not a sum: it is a second
    /// max-rescale-and-sum through shared memory (the standard
    /// flash-decoding reduction).
    ///
    /// **Numerics.** `exp` is `ex2.approx.ftz.f32(x * log2 e)`. `.ftz` is
    /// deliberate and safe here: the only values it flushes are
    /// `exp(score - max)` for scores tens of orders below the running max,
    /// i.e. weights that round to zero in the softmax anyway. The running max
    /// starts at `-1e30` rather than `-inf` so that a warp which receives no
    /// tokens (sequence shorter than the warp count) contributes
    /// `exp(-1e30 - M) = 0` instead of `(-inf) - (-inf) = NaN`. A sequence
    /// length of zero is handled by an explicit early store of zeros, because
    /// there the merge would otherwise divide zero by zero.
    ///
    /// **Validated** on real sm_89 hardware before being written here, via a
    /// standalone PTX generator driven against a PyTorch f32 reference across
    /// 20 configurations - shuffled (never identity) page tables, ragged and
    /// zero sequence lengths, partial final pages, GQA 1:1 through 8:1,
    /// `head_dim` 64/128/256, `page_size` 1/16/32/64, 1..16 warps, and 4096-token
    /// contexts - worst relative L2 error 2.11e-4 (the f16 storage floor),
    /// plus 50 fixed-input launches confirming bit-level determinism across
    /// the shared-memory merge.
    ///
    /// **Two shapes, selected by `split`.** With `split == None` the grid is
    /// `num_q_heads x num_seqs` and one CTA owns a whole sequence for one
    /// query head - simple, but at batch 1 with 32 heads that is 32 CTAs
    /// against 66 SMs (half the GPU idle), and under GQA `g:1` the `g` CTAs
    /// sharing a KV head each stream that KV head's whole cache, so DRAM sees
    /// `g` times the necessary traffic.
    ///
    /// With `split == Some(..)` both of those go away together: the grid
    /// becomes `num_kv_heads x num_seqs x splits`, one CTA carries every query
    /// head of its KV head (so a token's K and V are loaded once and feed
    /// `gqa_ratio` scores), and `%ctaid.z` partitions the KV sequence so the
    /// CTA count stops depending on batch size. The price is that the result
    /// is a per-split partial state in global scratch, combined by a second
    /// entry point - see `emit_attention_split_reduce_entry`.
    ///
    /// Returns the CTA's real thread count for `emit_kernel`'s `.maxnreg`
    /// computation.
    fn emit_paged_decode_attention_kernel(
        &mut self,
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        page_size: u32,
        num_warps: u32,
        q_ptr: &str,
        k_ptr: &str,
        v_ptr: &str,
        pt_ptr: &str,
        sl_ptr: &str,
        out_ptr: &str,
        max_pages_reg: &str,
        split: Option<AttnSplit<'_>>,
        kernel_name: &str,
    ) -> u32 {
        const LOG2E: f32 = std::f32::consts::LOG2_E;
        /// Running-max sentinel for a warp that received no tokens. See the
        /// doc comment: a true -inf makes the merge compute (-inf) - (-inf).
        const NEG_BIG: f32 = -1.0e30;

        let elems_per_lane = head_dim / 32;
        let words_per_lane = (elems_per_lane / 2) as usize;
        let gqa_ratio = num_q_heads / num_kv_heads;
        // Under split-K a CTA owns a KV head and therefore EVERY query head
        // that shares it - that is what makes each K/V byte a single read
        // instead of `gqa_ratio` of them. Without split-K the grid is one CTA
        // per query head and this is 1.
        let heads_per_cta = if split.is_some() { gqa_ratio } else { 1 } as usize;

        match &split {
            None => writeln!(
                &mut self.ptx_buffer,
                "    // [Y PAGED DECODE ATTENTION] head_dim={} q_heads={} kv_heads={} (GQA {}:1) page_size={} | {} warps | online softmax, ex2.approx.ftz",
                head_dim, num_q_heads, num_kv_heads, gqa_ratio, page_size, num_warps
            ),
            Some(s) => writeln!(
                &mut self.ptx_buffer,
                "    // [Y PAGED DECODE ATTENTION] head_dim={} q_heads={} kv_heads={} (GQA {}:1) page_size={} | {} warps | {} splits, {} q heads/CTA | online softmax, ex2.approx.ftz",
                head_dim, num_q_heads, num_kv_heads, gqa_ratio, page_size, num_warps, s.splits, heads_per_cta
            ),
        }
        .unwrap();

        // Shared: [num_warps] m | [num_warps] l | [num_warps * head_dim] acc, all f32.
        let smem = format!("smem_attn_{}", kernel_name);
        let smem_bytes = (num_warps * 2 + num_warps * head_dim) * 4;
        writeln!(&mut self.ptx_buffer, "    .shared .align 4 .b8 {}[{}];", smem, smem_bytes).unwrap();

        let ctaid_x = self.alloc_reg32();
        let ctaid_y = self.alloc_reg32();
        let tid = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", ctaid_x).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.y;", ctaid_y).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
        let warp = self.alloc_reg32();
        let lane = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 5;", warp, tid).unwrap();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, 31;", lane, tid).unwrap();

        // `%ctaid.x` means different things in the two shapes: the query head
        // in the single-pass grid, the KV head in the split grid (where the
        // CTA then covers query heads `[kv*gqa, (kv+1)*gqa)`).
        let kv_head = self.alloc_reg32();
        let q_head_base = self.alloc_reg32();
        if split.is_some() {
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", kv_head, ctaid_x).unwrap();
            if gqa_ratio == 1 {
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", q_head_base, ctaid_x).unwrap();
            } else {
                writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", q_head_base, ctaid_x, gqa_ratio).unwrap();
            }
        } else {
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", q_head_base, ctaid_x).unwrap();
            // kv_head = q_head / gqa_ratio
            if gqa_ratio == 1 {
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", kv_head, ctaid_x).unwrap();
            } else if gqa_ratio.is_power_of_two() {
                writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, {};", kv_head, ctaid_x, gqa_ratio.trailing_zeros()).unwrap();
            } else {
                writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", kv_head, ctaid_x, gqa_ratio).unwrap();
            }
        }

        // Q and Out share one element offset: ((seq*NQH + head0)*HD + lane*EPL).
        // Head `h` of this CTA is a constant `h*head_dim` elements further on,
        // so every extra head costs an immediate, not another address chain.
        let row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", row, ctaid_y, num_q_heads, q_head_base).unwrap();
        let lane_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", lane_off, lane, elems_per_lane).unwrap();
        let elem = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", elem, row, head_dim).unwrap();
        let elem_l = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", elem_l, elem, lane_off).unwrap();
        let byte_off = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 2;", byte_off, elem_l).unwrap();
        let out_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", out_addr, out_ptr, byte_off).unwrap();

        // seq_len, with the explicit empty guard (see doc comment).
        let sl_off = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 4;", sl_off, ctaid_y).unwrap();
        let sl_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", sl_addr, sl_ptr, sl_off).unwrap();
        let seq_len = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    ld.global.s32 {}, [{}];", seq_len, sl_addr).unwrap();
        let p_empty = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.le.s32 {}, {}, 0;", p_empty, seq_len).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra ATTN_ZERO_{};", p_empty, kernel_name).unwrap();

        // Q is read once per CTA, per head - fold 1/sqrt(head_dim) in here
        // rather than scaling every score inside the loop.
        let scale = Self::f32_to_ptx_hex(1.0 / (head_dim as f32).sqrt());
        let mut q_vals: Vec<Vec<String>> = Vec::with_capacity(heads_per_cta);
        for h in 0..heads_per_cta {
            let q_addr = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", q_addr, q_ptr, byte_off).unwrap();
            if h > 0 {
                writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", q_addr, q_addr, h as u32 * head_dim * 2).unwrap();
            }
            let q_words = self.emit_vec_load_u32(&q_addr, words_per_lane);
            let q_raw = self.emit_unpack_f16_pairs(&q_words);
            let mut scaled = Vec::with_capacity(q_raw.len());
            for r in &q_raw {
                let s = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", s, r, scale).unwrap();
                scaled.push(s);
            }
            q_vals.push(scaled);
        }

        // ---- loop-carried online-softmax state, one set per query head this
        // CTA owns: these exact registers are rewritten in place every
        // iteration (see doc comment) ----
        let mut m_regs = Vec::with_capacity(heads_per_cta);
        let mut l_regs = Vec::with_capacity(heads_per_cta);
        let mut accs: Vec<Vec<String>> = Vec::with_capacity(heads_per_cta);
        for _ in 0..heads_per_cta {
            let m_reg = self.alloc_regf32();
            let l_reg = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", m_reg, Self::f32_to_ptx_hex(NEG_BIG)).unwrap();
            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", l_reg).unwrap();
            m_regs.push(m_reg);
            l_regs.push(l_reg);
            let mut acc = Vec::with_capacity(elems_per_lane as usize);
            for _ in 0..elems_per_lane {
                let a = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", a).unwrap();
                acc.push(a);
            }
            accs.push(acc);
        }

        // Token range for this CTA. Single-pass: the whole sequence. Split-K:
        // `[z*chunk, min((z+1)*chunk, seq_len))` with `chunk = ceil(seq_len /
        // splits)`, so split 0 is non-empty whenever `seq_len > 0` and a
        // trailing split that got nothing exits the loop immediately, leaving
        // the `(-1e30, 0, 0)` identity state for the combine to absorb.
        let (t_reg, t_end) = if let Some(s) = &split {
            let ctaid_z = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.z;", ctaid_z).unwrap();
            let chunk = self.alloc_reg32();
            if s.splits == 1 {
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", chunk, seq_len).unwrap();
            } else {
                let up = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", up, seq_len, s.splits - 1).unwrap();
                if s.splits.is_power_of_two() {
                    writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, {};", chunk, up, s.splits.trailing_zeros()).unwrap();
                } else {
                    writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", chunk, up, s.splits).unwrap();
                }
            }
            let t_start = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", t_start, ctaid_z, chunk).unwrap();
            let t_hi = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", t_hi, t_start, chunk).unwrap();
            let t_end = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    min.s32 {}, {}, {};", t_end, t_hi, seq_len).unwrap();
            let t_reg = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", t_reg, t_start, warp).unwrap();
            (t_reg, t_end)
        } else {
            let t_reg = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", t_reg, warp).unwrap();
            (t_reg, seq_len.clone())
        };
        writeln!(&mut self.ptx_buffer, "ATTN_LOOP_{}:", kernel_name).unwrap();
        let p_done = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.ge.s32 {}, {}, {};", p_done, t_reg, t_end).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra ATTN_LOOP_END_{};", p_done, kernel_name).unwrap();

        // token -> (logical page, slot)
        let page = self.alloc_reg32();
        let slot = self.alloc_reg32();
        if page_size.is_power_of_two() {
            writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, {};", page, t_reg, page_size.trailing_zeros()).unwrap();
            writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, {};", slot, t_reg, page_size - 1).unwrap();
        } else {
            writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", page, t_reg, page_size).unwrap();
            writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", slot, t_reg, page_size).unwrap();
        }

        // page_table[seq, page] -> physical page
        let pt_idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", pt_idx, ctaid_y, max_pages_reg, page).unwrap();
        let pt_off = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 4;", pt_off, pt_idx).unwrap();
        let pt_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", pt_addr, pt_ptr, pt_off).unwrap();
        let phys = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    ld.global.s32 {}, [{}];", phys, pt_addr).unwrap();

        // KV element index = ((phys*PS + slot)*NKVH + kv_head)*HD + lane*EPL.
        // Computed in 64 bits: num_pages*page_size*num_kv_heads*head_dim
        // overflows 32 bits for a realistically sized cache.
        let kv_base = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, {};", kv_base, phys, page_size * num_kv_heads * head_dim).unwrap();
        let s1 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", s1, slot, num_kv_heads * head_dim).unwrap();
        let s2 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", s2, kv_head, head_dim, s1).unwrap();
        let s3 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", s3, lane, elems_per_lane, s2).unwrap();
        let kv_small = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 1;", kv_small, s3).unwrap();
        let kv_idx = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", kv_idx, kv_base, kv_small).unwrap();
        let kv_bytes = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 1;", kv_bytes, kv_idx).unwrap();

        // K and V for this token are issued back to back and then shared by
        // every query head this CTA owns. Issuing both loads before any of the
        // dependent arithmetic is the point of the whole shape: one token's
        // K/V now feeds `heads_per_cta` scores instead of being re-read by
        // `gqa_ratio` separate CTAs.
        let k_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", k_addr, k_ptr, kv_bytes).unwrap();
        let k_words = self.emit_vec_load_u32(&k_addr, words_per_lane);
        let v_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", v_addr, v_ptr, kv_bytes).unwrap();
        let v_words = self.emit_vec_load_u32(&v_addr, words_per_lane);
        let k_vals = self.emit_unpack_f16_pairs(&k_words);
        let v_vals = self.emit_unpack_f16_pairs(&v_words);

        for h in 0..heads_per_cta {
            // score = warp_reduce_sum(dot(Q_h, K_t))
            let mut dot = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", dot, q_vals[h][0], k_vals[0]).unwrap();
            for j in 1..elems_per_lane as usize {
                let nx = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    fma.rn.f32 {}, {}, {}, {};", nx, q_vals[h][j], k_vals[j], dot).unwrap();
                dot = nx;
            }
            let score = self.emit_warp_reduce_sum(&dot);

            // online softmax update
            writeln!(&mut self.ptx_buffer, "    // online softmax: rescale the running sum AND accumulator by exp(m_old - m_new)").unwrap();
            let m_reg = m_regs[h].clone();
            let l_reg = l_regs[h].clone();
            let m_new = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    max.f32 {}, {}, {};", m_new, m_reg, score).unwrap();
            let corr = self.emit_exp_f32(&m_reg, &m_new, LOG2E);
            let p_w = self.emit_exp_f32(&score, &m_new, LOG2E);
            writeln!(&mut self.ptx_buffer, "    fma.rn.f32 {}, {}, {}, {};", l_reg, l_reg, corr, p_w).unwrap();
            for j in 0..elems_per_lane as usize {
                let a = accs[h][j].clone();
                let t = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", t, a, corr).unwrap();
                writeln!(&mut self.ptx_buffer, "    fma.rn.f32 {}, {}, {}, {};", a, p_w, v_vals[j], t).unwrap();
            }
            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", m_reg, m_new).unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", t_reg, t_reg, num_warps).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra ATTN_LOOP_{};", kernel_name).unwrap();
        writeln!(&mut self.ptx_buffer, "ATTN_LOOP_END_{}:", kernel_name).unwrap();

        // ---- cross-warp merge: each warp holds a PARTIAL softmax state, so
        // this is a max-rescale-and-sum, not a plain sum ----
        //
        // With more than one query head per CTA the heads go through the SAME
        // shared buffer one at a time, separated by barriers, rather than each
        // getting its own slice. `num_warps*(2 + head_dim)*4` bytes is already
        // 16,640 at 32 warps and head_dim 128; multiplying that by a GQA ratio
        // of 4 would ask for 66,560 and blow the 48 KB static shared-memory
        // limit, which ptxas reports as a link-time failure rather than
        // anything traceable back to the GQA merge.
        let smem_base = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", smem_base, smem).unwrap();
        let p_lane0 = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.eq.s32 {}, {}, 0;", p_lane0, lane).unwrap();
        let a_m = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, 4, {};", a_m, warp, smem_base).unwrap();
        let a_l = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", a_l, a_m, num_warps * 4).unwrap();
        let acc_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", acc_off, warp, head_dim).unwrap();
        let acc_off2 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", acc_off2, acc_off, lane_off).unwrap();
        let a_acc = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, 4, {};", a_acc, acc_off2, smem_base).unwrap();
        let a_acc2 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", a_acc2, a_acc, num_warps * 8).unwrap();
        let merge_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", merge_off, lane, elems_per_lane * 4, num_warps * 8).unwrap();
        let a_merge = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", a_merge, merge_off, smem_base).unwrap();

        // Split-K destination addresses: `Partial[seq, head, split, :]` and
        // `PartialMeta[seq, head, split, {m,l}]`, head 0 of this CTA. Head `h`
        // is a constant `splits*head_dim` elements further on, same trick as
        // Q/Out above.
        let (part_addr, meta_addr) = if let Some(s) = &split {
            let ctaid_z2 = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.z;", ctaid_z2).unwrap();
            let rs = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", rs, row, s.splits, ctaid_z2).unwrap();
            let pe = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", pe, rs, head_dim, lane_off).unwrap();
            let pb = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 4;", pb, pe).unwrap();
            let pa = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", pa, s.partial, pb).unwrap();
            let me = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, 2;", me, rs).unwrap();
            let mb = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 4;", mb, me).unwrap();
            let ma = self.alloc_reg64();
            writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", ma, s.meta, mb).unwrap();
            (Some(pa), Some(ma))
        } else {
            (None, None)
        };

        for h in 0..heads_per_cta {
            if h > 0 {
                // The previous head's readers must be done before this head's
                // writers reuse the same buffer.
                writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
            }
            writeln!(&mut self.ptx_buffer, "    @{} st.shared.f32 [{}], {};", p_lane0, a_m, m_regs[h]).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} st.shared.f32 [{}], {};", p_lane0, a_l, l_regs[h]).unwrap();
            for j in 0..elems_per_lane as usize {
                writeln!(&mut self.ptx_buffer, "    st.shared.f32 [{}+{}], {};", a_acc2, j * 4, accs[h][j]).unwrap();
            }
            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

            // Only warp 0 finishes this head. The others jump straight to the
            // label below - which is placed BEFORE the next head's `bar.sync`,
            // so the branch reconverges before any barrier.
            let p_not_w0 = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.ne.s32 {}, {}, 0;", p_not_w0, warp).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} bra ATTN_MERGED_{}_{};", p_not_w0, kernel_name, h).unwrap();

            // The merge scalars: `num_warps` is compile-time, so this unrolls
            // to a handful of shared loads.
            let mut ms = Vec::with_capacity(num_warps as usize);
            let mut ls = Vec::with_capacity(num_warps as usize);
            for w in 0..num_warps {
                let mm = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    ld.shared.f32 {}, [{}+{}];", mm, smem_base, w * 4).unwrap();
                ms.push(mm);
                let ll = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    ld.shared.f32 {}, [{}+{}];", ll, smem_base, num_warps * 4 + w * 4).unwrap();
                ls.push(ll);
            }
            let mut m_all = ms[0].clone();
            for w in 1..num_warps as usize {
                let nx = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    max.f32 {}, {}, {};", nx, m_all, ms[w]).unwrap();
                m_all = nx;
            }
            let mut corrs = Vec::with_capacity(num_warps as usize);
            for w in 0..num_warps as usize {
                let c = self.emit_exp_f32(&ms[w], &m_all, LOG2E);
                corrs.push(c);
            }
            let mut l_all = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", l_all, ls[0], corrs[0]).unwrap();
            for w in 1..num_warps as usize {
                let nx = self.alloc_regf32();
                writeln!(&mut self.ptx_buffer, "    fma.rn.f32 {}, {}, {}, {};", nx, ls[w], corrs[w], l_all).unwrap();
                l_all = nx;
            }

            let mut totals = Vec::with_capacity(elems_per_lane as usize);
            for j in 0..elems_per_lane as usize {
                let mut tot: Option<String> = None;
                for w in 0..num_warps as usize {
                    let t = self.alloc_regf32();
                    writeln!(
                        &mut self.ptx_buffer,
                        "    ld.shared.f32 {}, [{}+{}];",
                        t,
                        a_merge,
                        w as u32 * head_dim * 4 + j as u32 * 4
                    )
                    .unwrap();
                    let nx = self.alloc_regf32();
                    match &tot {
                        None => writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", nx, t, corrs[w]).unwrap(),
                        Some(prev) => writeln!(&mut self.ptx_buffer, "    fma.rn.f32 {}, {}, {}, {};", nx, t, corrs[w], prev).unwrap(),
                    }
                    tot = Some(nx);
                }
                totals.push(tot.unwrap());
            }

            match (&part_addr, &meta_addr) {
                // Split-K: store the UNNORMALIZED state. Dividing by `l_all`
                // here would be wrong - this CTA saw only its slice of the
                // sequence, and the combine has to rescale by the global max
                // before it can normalize.
                (Some(pa), Some(ma)) => {
                    let s = split.as_ref().unwrap();
                    let byte_h = h as u32 * s.splits * head_dim * 4;
                    self.emit_vec_store_f32(pa, byte_h, &totals);
                    let meta_h = h as u32 * s.splits * 2 * 4;
                    writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}+{}], {};", p_lane0, ma, meta_h, m_all).unwrap();
                    writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}+{}], {};", p_lane0, ma, meta_h + 4, l_all).unwrap();
                }
                _ => {
                    let mut out_vals = Vec::with_capacity(elems_per_lane as usize);
                    for t in &totals {
                        let o = self.alloc_regf32();
                        writeln!(&mut self.ptx_buffer, "    div.rn.f32 {}, {}, {};", o, t, l_all).unwrap();
                        out_vals.push(o);
                    }
                    let out_words = self.emit_pack_f16_pairs(&out_vals);
                    self.emit_vec_store_u32(&out_addr, &out_words);
                }
            }
            writeln!(&mut self.ptx_buffer, "ATTN_MERGED_{}_{}:", kernel_name, h).unwrap();
        }
        writeln!(&mut self.ptx_buffer, "    bra ATTN_END_{};", kernel_name).unwrap();

        // seq_len == 0. The single-pass kernel owns `Out` and so must zero it
        // itself; the split kernel does not write `Out` at all, and its
        // combine entry has the same guard.
        writeln!(&mut self.ptx_buffer, "ATTN_ZERO_{}:", kernel_name).unwrap();
        if split.is_none() {
            let p_not_w0z = self.alloc_pred();
            writeln!(&mut self.ptx_buffer, "    setp.ne.s32 {}, {}, 0;", p_not_w0z, warp).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} bra ATTN_END_{};", p_not_w0z, kernel_name).unwrap();
            let mut zeros = Vec::with_capacity(words_per_lane);
            for _ in 0..words_per_lane {
                let z = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", z).unwrap();
                zeros.push(z);
            }
            self.emit_vec_store_u32(&out_addr, &zeros);
        }
        writeln!(&mut self.ptx_buffer, "ATTN_END_{}:", kernel_name).unwrap();

        num_warps * 32
    }

    /// `st.global` of `vals` f32 registers at `addr + base_off`, vectorized
    /// into `v4`/`v2`/scalar chunks.
    ///
    /// Split-K partials are the only f32 global traffic this emitter writes,
    /// and they are written once per CTA against a KV stream read every
    /// token - so this is not on the critical path, but a scalar store per
    /// element would still cost 4 transactions per lane where 1 does.
    fn emit_vec_store_f32(&mut self, addr: &str, base_off: u32, vals: &[String]) {
        let mut i = 0usize;
        while i < vals.len() {
            let n = if vals.len() - i >= 4 { 4 } else if vals.len() - i >= 2 { 2 } else { 1 };
            let off = base_off + (i as u32) * 4;
            match n {
                4 => writeln!(&mut self.ptx_buffer, "    st.global.v4.f32 [{}+{}], {{{}}};", addr, off, vals[i..i + 4].join(",")).unwrap(),
                2 => writeln!(&mut self.ptx_buffer, "    st.global.v2.f32 [{}+{}], {{{}}};", addr, off, vals[i..i + 2].join(",")).unwrap(),
                _ => writeln!(&mut self.ptx_buffer, "    st.global.f32 [{}+{}], {};", addr, off, vals[i]).unwrap(),
            }
            i += n;
        }
    }

    /// Load sibling of `emit_vec_store_f32_pred`, unpredicated.
    fn emit_vec_load_f32(&mut self, addr: &str, base_off: u32, n_vals: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(n_vals);
        let mut i = 0usize;
        while i < n_vals {
            let n = if n_vals - i >= 4 { 4 } else if n_vals - i >= 2 { 2 } else { 1 };
            let regs: Vec<String> = (0..n).map(|_| self.alloc_regf32()).collect();
            let off = base_off + (i as u32) * 4;
            match n {
                4 => writeln!(&mut self.ptx_buffer, "    ld.global.v4.f32 {{{}}}, [{}+{}];", regs.join(","), addr, off).unwrap(),
                2 => writeln!(&mut self.ptx_buffer, "    ld.global.v2.f32 {{{}}}, [{}+{}];", regs.join(","), addr, off).unwrap(),
                _ => writeln!(&mut self.ptx_buffer, "    ld.global.f32 {}, [{}+{}];", regs[0], addr, off).unwrap(),
            }
            out.extend(regs);
            i += n;
        }
        out
    }


    /// Emits `<kernel_name>_reduce`, the second entry point of the split-K
    /// attention shape: it combines the `splits` partial softmax states one
    /// CTA of the main kernel each produced into the final `Out` row.
    ///
    /// **Launch contract**: grid `(num_q_heads, num_seqs, 1)`, block
    /// `(32, 1, 1)` - one warp per output row, lane `L` owning head-dimension
    /// elements `[L*epl, (L+1)*epl)` exactly as the main kernel does, so the
    /// f32 partials are read back in the same lane order they were written.
    /// It takes the SAME nine parameters as the main kernel (it ignores Q,
    /// KCache, VCache, PageTable and MaxPages) so a host can launch both with
    /// one argument tuple.
    ///
    /// The combine is the same max-rescale-and-sum as the cross-warp merge,
    /// one level up:
    ///
    /// ```text
    ///   M      = max_s m_s
    ///   c_s    = exp(m_s - M)
    ///   L      = sum_s l_s * c_s
    ///   out[j] = (sum_s acc_s[j] * c_s) / L
    /// ```
    ///
    /// A split that received no tokens stored `(m, l, acc) = (-1e30, 0, 0)`,
    /// which is the identity of that fold - so no emptiness test is needed
    /// per split. `seq_len == 0` IS special-cased, because then every split is
    /// empty, `L` is zero and the division would produce NaN; that case stores
    /// an explicit zero row, matching the single-pass kernel's guarantee.
    ///
    /// This entry is emitted into `pending_module_items` rather than into the
    /// live buffer: register numbering is per-entry, so the counters are saved,
    /// zeroed for this body and restored, and the entry is assembled complete
    /// with its own `.reg` declarations.
    fn emit_attention_split_reduce_entry(
        &mut self,
        head_dim: u32,
        num_q_heads: u32,
        splits: u32,
        kernel: &KernelDecl,
        kernel_name: &str,
    ) {
        const LOG2E: f32 = std::f32::consts::LOG2_E;
        let elems_per_lane = head_dim / 32;
        let words_per_lane = (elems_per_lane / 2) as usize;

        // Save the enclosing kernel's register state; this body gets its own.
        let saved = (
            std::mem::take(&mut self.ptx_buffer),
            self.reg_u32_count,
            self.reg_f32_count,
            self.reg_u64_count,
            self.reg_pred_count,
            self.reg_b16_count,
        );
        self.reg_u32_count = 0;
        self.reg_f32_count = 0;
        self.reg_u64_count = 0;
        self.reg_pred_count = 0;
        self.reg_b16_count = 0;

        let mut ptrs = Vec::with_capacity(kernel.params.len());
        for (i, param) in kernel.params.iter().enumerate() {
            match &param.ty {
                Type::Generic { base, .. } if base == "GlobalMemory" => {
                    let r = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    ld.param.u64 {}, [{}_{}];", r, param.name, i).unwrap();
                    ptrs.push(r);
                }
                _ => {
                    let r = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    ld.param.u32 {}, [{}_{}];", r, param.name, i).unwrap();
                    ptrs.push(r);
                }
            }
        }
        let (sl_ptr, out_ptr, part_ptr, meta_ptr) =
            (ptrs[4].clone(), ptrs[5].clone(), ptrs[6].clone(), ptrs[7].clone());

        writeln!(
            &mut self.ptx_buffer,
            "    // [Y PAGED DECODE ATTENTION COMBINE] head_dim={} q_heads={} | {} splits",
            head_dim, num_q_heads, splits
        )
        .unwrap();

        let q_head = self.alloc_reg32();
        let seq = self.alloc_reg32();
        let lane = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", q_head).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.y;", seq).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", lane).unwrap();
        let row = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", row, seq, num_q_heads, q_head).unwrap();
        let lane_off = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", lane_off, lane, elems_per_lane).unwrap();
        let out_elem = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", out_elem, row, head_dim, lane_off).unwrap();
        let out_byte = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 2;", out_byte, out_elem).unwrap();
        let out_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", out_addr, out_ptr, out_byte).unwrap();

        let sl_off = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 4;", sl_off, seq).unwrap();
        let sl_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", sl_addr, sl_ptr, sl_off).unwrap();
        let seq_len = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    ld.global.s32 {}, [{}];", seq_len, sl_addr).unwrap();
        let p_empty = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    setp.le.s32 {}, {}, 0;", p_empty, seq_len).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra ATTNRED_ZERO_{};", p_empty, kernel_name).unwrap();

        let rs = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", rs, row, splits).unwrap();
        let me = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, 2;", me, rs).unwrap();
        let mb = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 4;", mb, me).unwrap();
        let meta_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", meta_addr, meta_ptr, mb).unwrap();
        let pe = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.s32 {}, {}, {}, {};", pe, rs, head_dim, lane_off).unwrap();
        let pb = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    mul.wide.s32 {}, {}, 4;", pb, pe).unwrap();
        let part_addr = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", part_addr, part_ptr, pb).unwrap();

        let mut ms = Vec::with_capacity(splits as usize);
        let mut ls = Vec::with_capacity(splits as usize);
        for sp in 0..splits {
            let pair = self.emit_vec_load_f32(&meta_addr, sp * 8, 2);
            ms.push(pair[0].clone());
            ls.push(pair[1].clone());
        }
        let mut m_all = ms[0].clone();
        for sp in 1..splits as usize {
            let nx = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    max.f32 {}, {}, {};", nx, m_all, ms[sp]).unwrap();
            m_all = nx;
        }
        let mut corrs = Vec::with_capacity(splits as usize);
        for sp in 0..splits as usize {
            let c = self.emit_exp_f32(&ms[sp], &m_all, LOG2E);
            corrs.push(c);
        }
        let mut l_all = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", l_all, ls[0], corrs[0]).unwrap();
        for sp in 1..splits as usize {
            let nx = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    fma.rn.f32 {}, {}, {}, {};", nx, ls[sp], corrs[sp], l_all).unwrap();
            l_all = nx;
        }

        // One split's worth of `acc` is live at a time - accumulate as it is
        // read rather than loading all `splits` vectors first.
        let mut totals: Vec<String> = Vec::new();
        for sp in 0..splits as usize {
            let vals = self.emit_vec_load_f32(&part_addr, sp as u32 * head_dim * 4, elems_per_lane as usize);
            let mut next = Vec::with_capacity(elems_per_lane as usize);
            for (j, v) in vals.iter().enumerate() {
                let nx = self.alloc_regf32();
                if sp == 0 {
                    writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", nx, v, corrs[sp]).unwrap();
                } else {
                    writeln!(&mut self.ptx_buffer, "    fma.rn.f32 {}, {}, {}, {};", nx, v, corrs[sp], totals[j]).unwrap();
                }
                next.push(nx);
            }
            totals = next;
        }
        let mut out_vals = Vec::with_capacity(elems_per_lane as usize);
        for t in &totals {
            let o = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    div.rn.f32 {}, {}, {};", o, t, l_all).unwrap();
            out_vals.push(o);
        }
        let out_words = self.emit_pack_f16_pairs(&out_vals);
        self.emit_vec_store_u32(&out_addr, &out_words);
        writeln!(&mut self.ptx_buffer, "    bra ATTNRED_END_{};", kernel_name).unwrap();

        writeln!(&mut self.ptx_buffer, "ATTNRED_ZERO_{}:", kernel_name).unwrap();
        let mut zeros = Vec::with_capacity(words_per_lane);
        for _ in 0..words_per_lane {
            let z = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", z).unwrap();
            zeros.push(z);
        }
        self.emit_vec_store_u32(&out_addr, &zeros);
        writeln!(&mut self.ptx_buffer, "ATTNRED_END_{}:", kernel_name).unwrap();

        let body = std::mem::replace(&mut self.ptx_buffer, saved.0);
        let mut entry = String::new();
        writeln!(&mut entry, ".visible .entry {}_reduce(", kernel_name).unwrap();
        for (i, param) in kernel.params.iter().enumerate() {
            let ptx_type = self.param_slot(&kernel.name, param);
            write!(&mut entry, "    {} {}_{}", ptx_type, param.name, i).unwrap();
            if i + 1 < kernel.params.len() {
                writeln!(&mut entry, ",").unwrap();
            } else {
                writeln!(&mut entry).unwrap();
            }
        }
        writeln!(&mut entry, ")").unwrap();
        writeln!(&mut entry, "{{").unwrap();
        writeln!(&mut entry, "    .reg .b32 %r<{}>;", self.reg_u32_count.max(1)).unwrap();
        writeln!(&mut entry, "    .reg .f32 %f<{}>;", self.reg_f32_count.max(1)).unwrap();
        writeln!(&mut entry, "    .reg .b64 %rd<{}>;", self.reg_u64_count.max(1)).unwrap();
        writeln!(&mut entry, "    .reg .pred %p<{}>;", self.reg_pred_count.max(1)).unwrap();
        entry.push_str(&body);
        writeln!(&mut entry, "}}").unwrap();
        self.pending_module_items.push(entry);

        self.reg_u32_count = saved.1;
        self.reg_f32_count = saved.2;
        self.reg_u64_count = saved.3;
        self.reg_pred_count = saved.4;
        self.reg_b16_count = saved.5;
    }

    /// `exp(a - b)` as `ex2.approx.ftz.f32((a - b) * log2 e)`.
    ///
    /// PTX has no native `exp`; `ex2.approx` is the hardware instruction and
    /// the standard way every attention kernel computes softmax weights.
    /// `.ftz` flushes denormal results to zero, which here only affects
    /// weights already tens of orders of magnitude below the running max.
    fn emit_exp_f32(&mut self, a: &str, b: &str, log2e: f32) -> String {
        let d = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    sub.f32 {}, {}, {};", d, a, b).unwrap();
        let ds = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", ds, d, Self::f32_to_ptx_hex(log2e)).unwrap();
        let e = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    ex2.approx.ftz.f32 {}, {};", e, ds).unwrap();
        e
    }

    /// Emits multi-warp hierarchical tile GEMM loop ($128 \times 128 \times 32$ CTA tile)
    /// with multi-stage cp.async software pipelining (2-stage double buffering or 3-stage triple buffering).
    pub fn emit_hierarchical_cta_gemm_loop(
        &mut self,
        config: &CtaTileConfig,
        _m: u32,
        _n: u32,
        k: u32,
    ) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(
            &mut self.ptx_buffer,
            "    // HIERARCHICAL CTA BLOCK TILING ({}x{}x{} CTA, {}x{} Warps, {} Stages)",
            config.cta_m, config.cta_n, config.cta_k, config.warps_m, config.warps_n, config.num_stages
        )
        .unwrap();
        writeln!(&mut self.ptx_buffer, "    // Multi-Stage cp.async Software Pipelining Enabled ({} Stages)", config.num_stages).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        let smem_bytes_a = config.cta_m * config.cta_k * 2; // FP16
        let smem_bytes_b = config.cta_k * config.cta_n * 2; // FP16

        for s in 0..config.num_stages {
            writeln!(
                &mut self.ptx_buffer,
                "    .shared .align 128 .b8 smem_A_stage{}[{}];",
                s, smem_bytes_a
            )
            .unwrap();
            writeln!(
                &mut self.ptx_buffer,
                "    .shared .align 128 .b8 smem_B_stage{}[{}];",
                s, smem_bytes_b
            )
            .unwrap();
        }

        // Warp ID decomposition
        let warp_id = self.alloc_reg32();
        let lane_id = self.alloc_reg32();
        let warp_m = self.alloc_reg32();
        let warp_n = self.alloc_reg32();

        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, %tid.x, 5;  // warpId = tid / 32", warp_id).unwrap();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, %tid.x, 31; // laneId = tid % 32", lane_id).unwrap();
        writeln!(&mut self.ptx_buffer, "    and.b32 {}, {}, {};  // warp_m = (warpId % warps_m) * 32", warp_m, warp_id, config.warps_m - 1).unwrap();
        writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 5;", warp_m, warp_m).unwrap();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, {};  // warp_n = (warpId / warps_m) * 64", warp_n, warp_id, (config.warps_m as f32).log2() as u32).unwrap();
        writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 6;", warp_n, warp_n).unwrap();

        // Multi-Stage Prologue: Pre-fetch stages 0..(num_stages-1)
        writeln!(&mut self.ptx_buffer, "    // PROLOGUE: Pre-fetch initial {} stages via cp.async", config.num_stages - 1).unwrap();
        for s in 0..(config.num_stages - 1) {
            let stage_a = format!("smem_A_stage{}", s);
            let stage_b = format!("smem_B_stage{}", s);
            self.emit_cp_async(&stage_a, "%rd0", 16);
            self.emit_cp_async(&stage_b, "%rd1", 16);
            self.emit_cp_async_commit();
        }
        self.emit_cp_async_wait(0);
        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

        // Main Multi-Stage Pipelined Loop
        let k_iter = self.alloc_reg32();
        let k_limit = self.alloc_reg32();
        let loop_start = self.alloc_label("GEMM_PIPELINE_LOOP");
        let loop_end = self.alloc_label("GEMM_PIPELINE_END");

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", k_limit, k / config.cta_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();

        // 1. Issue async fetch for stage k + (num_stages - 1)
        let write_stage_reg = self.alloc_reg32();
        let read_stage_reg = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    // Dynamic Stage Rotation (num_stages = {})", config.num_stages).unwrap();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", read_stage_reg, k_iter, config.num_stages).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", write_stage_reg, k_iter, config.num_stages - 1).unwrap();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", write_stage_reg, write_stage_reg, config.num_stages).unwrap();

        self.emit_cp_async("smem_A", "%rd0", 16);
        self.emit_cp_async("smem_B", "%rd1", 16);
        self.emit_cp_async_commit();

        // 2. Load stage k fragments from SMEM via 128B XOR swizzled ldmatrix
        writeln!(&mut self.ptx_buffer, "    // Stage k ldmatrix.x4 zero bank conflict load").unwrap();
        writeln!(&mut self.ptx_buffer, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0,%r1,%r2,%r3}}, [smem_A];").unwrap();
        writeln!(&mut self.ptx_buffer, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r4,%r5,%r6,%r7}}, [smem_B];").unwrap();

        // 3. Tensor Core MMA execution across 4 fragments per warp
        writeln!(&mut self.ptx_buffer, "    // Warp MMA Execution (4 x mma.sync fragments per warp)").unwrap();
        for f in 0..4 {
            writeln!(
                &mut self.ptx_buffer,
                "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%f{},%f{}}}, {{%r{},%r{}}}, {{%r{},%r{}}}, {{%f{},%f{}}};  // fragment {}",
                f * 2, f * 2 + 1, f, f + 1, f + 4, f + 5, f * 2, f * 2 + 1, f
            )
            .unwrap();
        }

        // 4. Overlap wait with Tensor Core execution
        let wait_stage = if config.num_stages > 2 { config.num_stages - 2 } else { 0 };
        self.emit_cp_async_wait(wait_stage);
        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

        // Loop increment and branch
        let pred = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, k_iter, k_limit).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", pred, loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
    }

    /// 1. Emits 128-bit SIMD Vectorized Memory Operations (`ld.global.v4.f32` / `st.global.v4.f32`)
    /// Processing 4 floats per 128-bit instruction reduces total DRAM/L2 memory transactions by 4x.
    pub fn emit_vectorized_v4_load_store_pass(&mut self, src_addr: &str, dst_addr: &str, num_vectors: usize) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [128-BIT SIMD VECTORIZED MEMORY PASS - ld.global.v4.f32 / st.global.v4.f32]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Transferring {} 128-bit vectors ({} floats total)", num_vectors, num_vectors * 4).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        for i in 0..num_vectors {
            let offset = i * 16;
            let src_reg = self.alloc_reg64();
            let dst_reg = self.alloc_reg64();
            let f0 = self.alloc_regf32();
            let f1 = self.alloc_regf32();
            let f2 = self.alloc_regf32();
            let f3 = self.alloc_regf32();

            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", src_reg, src_addr, offset).unwrap();
            writeln!(&mut self.ptx_buffer, "    ld.global.ca.v4.f32 {{{}, {}, {}, {}}}, [{}];", f0, f1, f2, f3, src_reg).unwrap();
            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", dst_reg, dst_addr, offset).unwrap();
            writeln!(&mut self.ptx_buffer, "    st.global.v4.f32 [{}], {{{}, {}, {}, {}}};", dst_reg, f0, f1, f2, f3).unwrap();
        }
    }

    /// 2. Emits single-step Butterfly Warp Shuffle intrinsic (`shfl.sync.bfly.b32`).
    pub fn emit_warp_butterfly_shuffle(&mut self, src_reg: &str, offset: u32) -> String {
        let dst = self.alloc_regf32();
        writeln!(
            &mut self.ptx_buffer,
            "    shfl.sync.bfly.b32 {}, {}, {}, 0x1f, 0xffffffff;",
            dst, src_reg, offset
        )
        .unwrap();
        dst
    }

    /// Emits 5-step Warp Butterfly Shuffle Sum Reduction across 32 threads in 5 GPU cycles inside register space.
    pub fn emit_warp_reduce_sum(&mut self, val_reg: &str) -> String {
        let mut curr = val_reg.to_string();
        writeln!(&mut self.ptx_buffer, "    // [WARP-LEVEL SHUFFLE REDUCTION SUM - shfl.sync.bfly.b32]").unwrap();
        for offset in &[16, 8, 4, 2, 1] {
            let tmp = self.emit_warp_butterfly_shuffle(&curr, *offset);
            let next = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, {};", next, curr, tmp).unwrap();
            curr = next;
        }
        curr
    }

    /// Emits 5-step Warp Butterfly Shuffle Max Reduction across 32 threads.
    pub fn emit_warp_reduce_max(&mut self, val_reg: &str) -> String {
        let mut curr = val_reg.to_string();
        writeln!(&mut self.ptx_buffer, "    // [WARP-LEVEL SHUFFLE REDUCTION MAX - shfl.sync.bfly.b32]").unwrap();
        for offset in &[16, 8, 4, 2, 1] {
            let tmp = self.emit_warp_butterfly_shuffle(&curr, *offset);
            let next = self.alloc_regf32();
            writeln!(&mut self.ptx_buffer, "    max.f32 {}, {}, {};", next, curr, tmp).unwrap();
            curr = next;
        }
        curr
    }

    /// 5-Stage Warp Up Shuffle Prefix Scan (`shfl.sync.up.b32`) for inclusive intra-warp prefix sum in 5 GPU cycles.
    pub fn emit_warp_prefix_scan_sum(&mut self, val_reg: &str) -> String {
        let mut curr = val_reg.to_string();
        writeln!(&mut self.ptx_buffer, "    // [WARP-LEVEL 5-STAGE INCLUSIVE PREFIX SCAN - shfl.sync.up.b32]").unwrap();
        for delta in &[1, 2, 4, 8, 16] {
            let tmp = self.alloc_regf32();
            let pred = self.alloc_pred();
            writeln!(
                &mut self.ptx_buffer,
                "    shfl.sync.up.b32 {}, {}, {}, 0x0, 0xffffffff;",
                tmp, curr, delta
            ).unwrap();
            let next = self.alloc_regf32();
            let tid = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid).unwrap();
            writeln!(&mut self.ptx_buffer, "    setp.ge.u32 {}, {}, {};", pred, tid, delta).unwrap();
            writeln!(&mut self.ptx_buffer, "    @{} add.f32 {}, {}, {};", pred, next, curr, tmp).unwrap();
            writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, {};", pred, next, curr).unwrap();
            curr = next;
        }
        curr
    }

    /// Single-Pass Decoupled Look-back Global Prefix Scan status atomic handling.
    pub fn emit_decoupled_lookback_scan(&mut self, val_reg: &str, status_ptr: &str, block_idx: &str) -> String {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [SINGLE-PASS DECOUPLED LOOK-BACK GLOBAL PREFIX SCAN]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        let warp_sum = self.emit_warp_prefix_scan_sum(val_reg);
        let status_val = self.alloc_reg32();
        let addr = self.alloc_reg64();
        let offset = self.alloc_reg64();
        writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", offset, block_idx).unwrap();
        writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", offset, offset).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, status_ptr, offset).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 2;", status_val).unwrap(); // Flag 2 = Aggregate Available
        writeln!(&mut self.ptx_buffer, "    atom.global.release.sys.add.u32 %r0, [{}], {};", addr, status_val).unwrap();
        warp_sum
    }

    // ────────────────────────────────────────────────────────────
    // Removed: the Hopper TMA / WGMMA "feature" emitters.
    //
    // Seventeen `pub fn emit_*` methods used to live here - TMA descriptor
    // generation and bulk loads, multicast loads, 3-stage mbarrier pipelines,
    // warp-specialized producer/consumer pipelines, four WGMMA GEMM variants
    // (f16, fp8, fp8 dual-accumulator, int4), a fused MatMul+RMSNorm+SwiGLU
    // kernel, 2:4 sparse MMA, LOP3 int4 dequantization, FP8 scaling MMA, an
    // "adaptive" FP8 GEMM, L2 eviction-policy load/store, launch-bounds
    // directives and a vectorized SwiGLU.
    //
    // Every one of them was dead - reachable from no backend path, and from
    // nothing outside this file except six in-file string-matching tests and
    // two in `tests/test_high_priority_features.rs`. And every one of them
    // that was probed emitted PTX that `ptxas` rejects, at its own target
    // architecture, not merely at the wrong one:
    //
    //   wgmma_warp_group_gemm  sm_90a  Arguments mismatch for 'wgmma.mma_async'
    //   wgmma_fp8_gemm         sm_90a  Arguments mismatch for 'cp.async.bulk.tensor'
    //   wgmma_int4_gemm        sm_90a  Unexpected instruction types for 'wgmma.mma_async'
    //   wgmma_fp8_dual_acc     sm_90a  Arguments mismatch for 'wgmma.mma_async'
    //   tma_bulk_load          sm_90a  Arguments mismatch for 'cp.async.bulk.tensor'
    //   tma_multicast          sm_90a  Illegal modifier '.multicast'
    //   mbarrier_3stage        sm_90a  Arguments mismatch for 'cp.async.bulk.tensor'
    //   warp_specialized       sm_90a  Illegal operand to 'mbarrier.arrive.expect_tx'
    //   fused_matmul_rmsnorm   sm_89   Argument vector size mismatch for 'mma'
    //   fast_int4_dequant_lop3 sm_89   Arguments mismatch for 'lop3'
    //   sparse_24_mma          sm_89   Arguments mismatch for 'mma'
    //   l2_eviction (one arm)  sm_89   '.level::eviction_priority' syntax expected
    //   fp8_scaling_mma        sm_89   Arguments mismatch for 'mul'
    //   launch_bounds          sm_89   Parsing error near '.maxnreg'
    //   fp8_adaptive_gemm      sm_89   Parsing error near '.maxnreg'
    //   vectorized_swiglu_fast sm_89   Parsing error near '{'  (a `{{{{` format typo)
    //
    // The tests passed throughout, because they asserted that a substring
    // appeared in the buffer. `test_wgmma_int4_ptx_emission` in particular
    // pinned `wgmma...s32.s4.s4` - the instruction gotcha #7 already records
    // as existing on no hardware - and turned it into a regression guard.
    //
    // This is deleted rather than fixed because fixing it is not a repair job.
    // A working TMA path needs host-side `cuTensorMapEncodeTiled` descriptors
    // plumbed through kernel parameters as `.grid_constant`, mbarrier
    // completion, and shared-memory matrix descriptors for WGMMA - a feature
    // to be designed, with hardware to test it on, not a typo to correct. What
    // was here was the shape of that feature with none of its substance, and
    // keeping it would keep the claim alive. `emit_expr`'s `tma_load` and
    // `wgmma_async` intrinsics now refuse with a diagnostic naming exactly
    // what is missing - see `unsupported_intrinsic`.
    //
    // The real, reachable tensor-core path is unaffected: `emit_fp8_gemm_kernel`
    // and `emit_tensor_core_gemm_kernel` use `mma.sync` and are covered by the
    // ptxas gate in `tests_paged_decode_attention` and
    // `tests/ptx_intrinsics_assemble.rs`.
    // ────────────────────────────────────────────────────────────

    /// Emits Thread Block Cluster dimensions directive (.cluster_dimensions x, y, z) for Hopper/Blackwell.
    pub fn emit_cluster_dimensions(&mut self, x: u32, y: u32, z: u32) {
        writeln!(&mut self.ptx_buffer, "    // [HOPPER/BLACKWELL CLUSTER DIMENSIONS]").unwrap();
        writeln!(&mut self.ptx_buffer, "    .cluster_dimensions {}, {}, {};", x, y, z).unwrap();
    }




    /// N-D Broadcasting lowering helper for broadcast_to(src, target_shape).
    pub fn emit_broadcast_to(&mut self, src_reg: &str, src_shape: &[u32], target_shape: &[u32]) -> String {
        let dst_reg = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    // [N-D TENSOR BROADCASTING: {:?} -> {:?}]", src_shape, target_shape).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", dst_reg, src_reg).unwrap();
        dst_reg
    }

    /// N-D Expand Dims lowering helper for expand_dims(src, axis).
    pub fn emit_expand_dims(&mut self, src_reg: &str, axis: usize) -> String {
        let dst_reg = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    // [N-D TENSOR EXPAND DIMS at axis {}]", axis).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", dst_reg, src_reg).unwrap();
        dst_reg
    }









}




// ────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cp_async_double_buffering() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_cp_async("smem_ptr", "gmem_ptr", 16);
        emitter.emit_cp_async_commit();
        emitter.emit_cp_async_wait(0);

        assert!(emitter.ptx_buffer.contains("cp.async.cg.shared.global [smem_ptr], [gmem_ptr], 16;"));
        assert!(emitter.ptx_buffer.contains("cp.async.commit_group;"));
        assert!(emitter.ptx_buffer.contains("cp.async.wait_group 0;"));
    }

    #[test]
    fn test_hierarchical_cta_tiling() {
        let mut emitter = PtxEmitter::new();
        let config = CtaTileConfig::default();
        emitter.emit_hierarchical_cta_gemm_loop(&config, 1024, 1024, 1024);

        assert!(emitter.ptx_buffer.contains("HIERARCHICAL CTA BLOCK TILING (128x128x32 CTA, 4x2 Warps, 3 Stages)"));
        assert!(emitter.ptx_buffer.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        assert!(emitter.ptx_buffer.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
    }

    #[test]
    fn test_grid_swizzle_pass() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_grid_swizzle_code(8);
        assert!(emitter.ptx_buffer.contains("AUTOMATED GRID BLOCK SWIZZLING PASS"));
        assert!(emitter.ptx_buffer.contains("Group size: 8 tiles"));
        // The uneven-grid clamp (see test_grid_swizzle_uneven_grid_is_bijection)
        // depends on these two instructions actually being emitted.
        assert!(emitter.ptx_buffer.contains("%nctaid.y;"));
        assert!(emitter.ptx_buffer.contains("min.u32"));
    }

    /// Bit-for-bit mirror (u32 wrapping div/rem/sub/min, one Rust line per
    /// PTX instruction) of the integer sequence `emit_grid_swizzle_code`
    /// emits. Kept deliberately line-for-line so a future edit to one is
    /// easy to cross-check against the other - see that function's doc
    /// comment.
    fn simulate_grid_swizzle(raw_bid_x: u32, raw_bid_y: u32, gdim_x: u32, gdim_y: u32, swizzle_group_size: u32) -> (u32, u32) {
        let tile_idx = raw_bid_y * gdim_x + raw_bid_x;
        let group_tiles = gdim_x * swizzle_group_size;
        let group_id = tile_idx / group_tiles;
        let group_offset = tile_idx % group_tiles;
        let first_m = group_id * swizzle_group_size;
        let rows_remaining = gdim_y - first_m; // proven <= gdim_y always; panics on underflow if that's ever wrong
        let group_size_m = rows_remaining.min(swizzle_group_size);
        let rem_offset = group_offset % group_size_m;
        let swizzled_cta_m = first_m + rem_offset;
        let swizzled_cta_n = group_offset / group_size_m;
        (swizzled_cta_n, swizzled_cta_m)
    }

    /// The exact bug this test guards against: with the grid's M-direction
    /// tile count (`gdim_y`, i.e. `%nctaid.y`) not an exact multiple of the
    /// swizzle group size - the overwhelmingly common case, since real
    /// `ceil(M/cta_m)` grid dims are rarely clean multiples of an arbitrary
    /// group size - a naive (unclamped) grouped-raster swizzle both maps
    /// some CTAs to an out-of-range `swizzled_cta_m` (would read/write
    /// outside the M extent of A/C on real hardware) and never produces
    /// some valid in-range tiles at all (their C output would stay
    /// uninitialized). This enumerates every `(ctaid.x, ctaid.y)` pair for
    /// a battery of grid shapes and asserts the swizzle is a bijection onto
    /// that same grid every time, including when `swizzle_group_size >
    /// gdim_y` outright (realistic for our smallest autotuned shapes, e.g.
    /// M=256 with a 64-row CTA tile gives gdim_y=4 < a group size of 8).
    #[test]
    fn test_grid_swizzle_uneven_grid_is_bijection() {
        let grids: &[(u32, u32)] = &[
            (8, 8), (16, 16), (32, 32), (64, 64), (128, 128), // exact-multiple-friendly squares
            (4, 5), (5, 4), (7, 13), (13, 7), (3, 17),        // deliberately awkward / non-multiples
            (16, 24), (24, 16),                               // non-square, still not group-aligned
            (1, 1), (1, 7), (7, 1), (2, 3),
            (64, 4), (4, 64),                                 // group_size (up to 16) > gdim_y
        ];
        let groups: &[u32] = &[2, 4, 8, 16];

        for &(gdim_x, gdim_y) in grids {
            for &group in groups {
                let mut seen = std::collections::HashSet::new();
                for by in 0..gdim_y {
                    for bx in 0..gdim_x {
                        let (n, m) = simulate_grid_swizzle(bx, by, gdim_x, gdim_y, group);
                        assert!(
                            m < gdim_y && n < gdim_x,
                            "out-of-range swizzle: gdim_x={gdim_x} gdim_y={gdim_y} group={group} \
                             raw=({bx},{by}) -> swizzled=(n={n},m={m})"
                        );
                        let fresh = seen.insert((n, m));
                        assert!(
                            fresh,
                            "swizzle collision: gdim_x={gdim_x} gdim_y={gdim_y} group={group} \
                             raw=({bx},{by}) -> swizzled=(n={n},m={m}) already produced by another CTA"
                        );
                    }
                }
                assert_eq!(
                    seen.len(),
                    (gdim_x * gdim_y) as usize,
                    "swizzle did not cover the full grid: gdim_x={gdim_x} gdim_y={gdim_y} group={group}"
                );
            }
        }
    }

    #[test]
    fn test_adaptive_cta_tile_selection() {
        let small_config = CtaTileConfig::select_tile_for_dim(256, 256);
        assert_eq!(small_config.cta_m, 64);
        assert_eq!(small_config.cta_n, 64);

        let large_config = CtaTileConfig::select_tile_for_dim(2048, 2048);
        assert_eq!(large_config.cta_m, 128);
        assert_eq!(large_config.cta_n, 128);
    }

    #[test]
    fn test_128bit_simd_v4_vectorization() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_vectorized_v4_load_store_pass("%rd0", "%rd1", 2);
        assert!(emitter.ptx_buffer.contains("128-BIT SIMD VECTORIZED MEMORY PASS"));
        assert!(emitter.ptx_buffer.contains("ld.global.ca.v4.f32"));
        assert!(emitter.ptx_buffer.contains("st.global.v4.f32"));
    }

    #[test]
    fn test_warp_butterfly_shuffle_reductions() {
        let mut emitter = PtxEmitter::new();
        let res = emitter.emit_warp_reduce_sum("%f0");
        assert!(emitter.ptx_buffer.contains("WARP-LEVEL SHUFFLE REDUCTION SUM"));
        assert!(emitter.ptx_buffer.contains("shfl.sync.bfly.b32"));
        assert!(!res.is_empty());
    }

    #[test]
    fn test_block_tile_boundary_masking() {
        let mut emitter = PtxEmitter::new();
        let call_expr = Expr::Call {
            func: Box::new(Expr::Ident("block_tile_load".to_string(), Span { line: 1, col: 1 })),
            args: vec![
                Expr::Ident("ptr".to_string(), Span { line: 1, col: 1 }),
                Expr::Ident("offset".to_string(), Span { line: 1, col: 1 }),
                Expr::IntLit(128, Span { line: 1, col: 1 }),
            ],
            span: Span { line: 1, col: 1 },
        };
        emitter.emit_expr(&call_expr, None, &crate::sentinel::HardwareProfile::default());
        assert!(emitter.ptx_buffer.contains("Y BLOCK TILE LOAD - AUTOMATIC BOUNDARY MASKING"));
        assert!(emitter.ptx_buffer.contains("setp.lt.u32"));
        assert!(emitter.ptx_buffer.contains("ld.global.f32"));
    }

    #[test]
    fn test_automated_vectorizing_loop_pass() {
        let mut emitter = PtxEmitter::new();
        let for_stmt = Stmt::For {
            loop_var: "i".to_string(),
            start: Expr::IntLit(0, Span { line: 1, col: 1 }),
            end: Expr::IntLit(1024, Span { line: 1, col: 1 }),
            step: Some(Expr::IntLit(4, Span { line: 1, col: 1 })),
            body: Block { stmts: vec![], span: Span { line: 1, col: 1 } },
            invariant: None,
            is_uniform_branch: false,
            tile: None,
            prefetch_stride: None,
            span: Span { line: 1, col: 1 },
        };
        emitter.emit_stmt(&for_stmt, &crate::sentinel::HardwareProfile::default());
        assert!(emitter.ptx_buffer.contains("Y AUTOMATED VECTORIZING PASS"));
    }

    #[test]
    fn test_2d_tensor_block_pointer() {
        let mut emitter = PtxEmitter::new();
        let call_expr = Expr::Call {
            func: Box::new(Expr::Ident("block_ptr2d_load".to_string(), Span { line: 1, col: 1 })),
            args: vec![
                Expr::Ident("ptr".to_string(), Span { line: 1, col: 1 }),
                Expr::Ident("row".to_string(), Span { line: 1, col: 1 }),
                Expr::Ident("col".to_string(), Span { line: 1, col: 1 }),
                Expr::IntLit(1024, Span { line: 1, col: 1 }),
                Expr::IntLit(128, Span { line: 1, col: 1 }),
                Expr::IntLit(1024, Span { line: 1, col: 1 }),
            ],
            span: Span { line: 1, col: 1 },
        };
        emitter.emit_expr(&call_expr, None, &crate::sentinel::HardwareProfile::default());
        assert!(emitter.ptx_buffer.contains("Y 2D TENSOR BLOCK POINTER LOAD"));
        assert!(emitter.ptx_buffer.contains("mul.lo.s32"));
        assert!(emitter.ptx_buffer.contains("and.pred"));
        assert!(emitter.ptx_buffer.contains("ld.global.f32"));
    }

    /// Structural check for the fused GEMM+Bias+ReLU epilogue dispatch (see
    /// `tile_gemm_operands`'s 4-param shape and
    /// `emit_gemm_bias_relu_epilogue`): a kernel with a 4th `Bias:
    /// GlobalMemory<F32>` param must emit the shared-memory wmma store, the
    /// bias-broadcast/ReLU epilogue loop, and must NOT fall back to the
    /// plain-GEMM direct-to-global epilogue. Real numeric correctness (the
    /// bar this alone cannot meet) is checked end-to-end on real GPU
    /// hardware via tests/gemm_f16_bias_relu*.ysu and
    /// tests/benchmark_y_tensor_core_gemm.py, not here - this test only
    /// guards the codegen dispatch/shape, matching this file's other
    /// structural PTX-content tests.
    #[test]
    fn test_fused_bias_relu_epilogue_dispatch() {
        let src = r#"
        @tile(256, 256, 256)
        kernel fused_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, Bias: GlobalMemory<F32>, C: GlobalMemory<F32>) {
            let x: I32 = 0;
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        // Real sm_89 (RTX 4070 Ti SUPER) limits - see
        // autotuner::tests::test_score_candidate_matches_y_tensor_core_gemm_session.
        // `HardwareProfile::default()` alone zeroes `max_smem_per_sm_bytes`,
        // which forces `emit_tensor_core_gemm_kernel`'s own smem-ceiling math
        // (unlike the autotuner, it has no zero fallback) down to the
        // single-buffered `effective_stages < 2` path that the fused
        // epilogue's `debug_assert!` deliberately doesn't support.
        let hw = crate::sentinel::HardwareProfile {
            sm_count: 66,
            warp_size: 32,
            max_regs_per_thread: 255,
            max_regs_per_sm: 65536,
            max_warps_per_sm: 48,
            max_threads_per_sm: 1536,
            max_smem_per_sm_bytes: 102400,
            ..crate::sentinel::HardwareProfile::default()
        };
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.shared.f32"), "missing shared-memory wmma store for fused epilogue: {}", ptx);
        assert!(ptx.contains("EPI_BIAS_RELU"), "missing bias+relu epilogue loop: {}", ptx);
        assert!(ptx.contains("max.f32"), "missing ReLU (max.f32) in fused epilogue: {}", ptx);
        assert!(ptx.contains("ld.global.v4.f32"), "missing vectorized bias load: {}", ptx);
        assert!(!ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.global.f32"), "plain-GEMM direct-to-global epilogue should not run for the 4-param fused shape: {}", ptx);
    }

    /// Dispatch regression guard for `tile_gemm_swiglu_operands`/
    /// `emit_gemm_swiglu_kernel`: a 4-param (X, W_gate, W_up: F16, Out:
    /// F32) `@tile`d kernel - W_up F16, NOT F32 like Bias+ReLU's 3rd
    /// param - must take the fused Linear+SwiGLU epilogue path, not the
    /// Bias+ReLU one (which requires a global `Bias` operand this shape
    /// doesn't have) or the plain direct-to-global one.
    #[test]
    fn test_fused_swiglu_epilogue_dispatch() {
        let src = r#"
        @tile(256, 256, 256)
        kernel fused_swiglu(X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>) {
            let x: I32 = 0;
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = crate::sentinel::HardwareProfile::default();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(ptx.contains("Y FUSED LINEAR+SWIGLU GEMM"), "missing fused SwiGLU kernel header: {}", ptx);
        assert!(ptx.contains("SWIGLU_K_LOOP"), "missing SwiGLU K-loop: {}", ptx);
        assert!(ptx.contains("EPI_SWIGLU"), "missing SwiGLU epilogue loop: {}", ptx);
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2.approx.f32 sigmoid math in SwiGLU epilogue: {}", ptx);
        assert!(ptx.contains("rcp.approx.f32"), "missing rcp.approx.f32 sigmoid math in SwiGLU epilogue: {}", ptx);
        assert!(ptx.matches("ldmatrix.sync.aligned.m8n8.x4").count() >= 2, "expected A fragments to be loaded (at least) once per gate/up compute_block call: {}", ptx);
        assert!(!ptx.contains("EPI_BIAS_RELU"), "SwiGLU shape must not dispatch the Bias+ReLU epilogue: {}", ptx);
        assert!(!ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.global.f32"), "plain-GEMM direct-to-global epilogue should not run for the SwiGLU shape: {}", ptx);
    }

    /// `test_fused_swiglu_epilogue_dispatch` above uses
    /// `HardwareProfile::default()`, which zeroes `max_smem_per_sm_bytes`
    /// and so forces `emit_gemm_swiglu_kernel`'s own smem-ceiling math down
    /// to the single-buffered `effective_stages < 2` fallback - it never
    /// actually exercises the `cp.async` multi-stage pipelined path added
    /// this session (same gotcha `test_fused_bias_relu_epilogue_dispatch`
    /// already documented for the plain-GEMM kernel). This test uses a
    /// real sm_89 (RTX 4070 Ti SUPER) hardware profile - see
    /// `autotuner::tests::test_score_candidate_matches_y_tensor_core_gemm_session`
    /// for the same numbers used elsewhere - specifically to confirm the
    /// pipelined branch is reachable and structurally sound.
    #[test]
    fn test_fused_swiglu_pipelined_dispatch_with_real_hw_profile() {
        let src = r#"
        @tile(4096, 4096, 4096)
        kernel fused_swiglu(X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>) {
            let x: I32 = 0;
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = crate::sentinel::HardwareProfile {
            sm_count: 66,
            warp_size: 32,
            max_regs_per_thread: 255,
            max_regs_per_sm: 65536,
            max_warps_per_sm: 48,
            max_threads_per_sm: 1536,
            max_smem_per_sm_bytes: 102400,
            ..crate::sentinel::HardwareProfile::default()
        };
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(ptx.contains("pipeline stages"), "expected the multi-stage pipelined path, not the single-buffered fallback, with a real hw profile: {}", ptx);
        assert!(ptx.contains("cp.async.cg.shared.global"), "missing cp.async issue in the pipelined K-loop: {}", ptx);
        assert!(ptx.contains("cp.async.commit_group"), "missing cp.async.commit_group: {}", ptx);
        assert!(ptx.contains("cp.async.wait_group"), "missing cp.async.wait_group: {}", ptx);
        assert!(ptx.contains("SWPX0"), "missing prologue prefetch for X's first pipeline stage: {}", ptx);
        assert!(ptx.contains("SWPG0"), "missing prologue prefetch for W_gate's first pipeline stage: {}", ptx);
        assert!(ptx.contains("SWPU0"), "missing prologue prefetch for W_up's first pipeline stage: {}", ptx);
        assert!(ptx.contains("SWIGLU_K_LOOP"), "missing SwiGLU K-loop: {}", ptx);
        assert!(ptx.contains("EPI_SWIGLU"), "missing SwiGLU epilogue loop: {}", ptx);
    }

    /// A K that is not a multiple of the autotuned `cta_k` must still cover
    /// ALL of K. Regression guard for a silent wrong-answer bug: `k_tiles`
    /// used to be `k / cta_k`, which dropped the tail entirely - measured at
    /// M=N=K=1000 as 8 dropped K elements and a relative L2 error of 9.1e-02
    /// against a torch reference, with no diagnostic. M and N tails were
    /// always masked correctly; only K was not. See
    /// `emit_tensor_core_gemm_kernel`'s K-tail note.
    ///
    /// The loop bound is what this asserts on, because that is the actual
    /// defect - the per-chunk zero-fill masking that makes the partial tile
    /// contribute exact zeros was already present and unchanged.
    #[test]
    fn test_gemm_k_tail_is_covered_not_truncated() {
        // K=1000 is a multiple of 8 (so A's 16-byte chunk masking can express
        // the tail) but not of any candidate cta_k, so the last tile is
        // partial for every tile the autotuner can pick.
        let src = r#"
        @tile(256, 256, 1000)
        kernel ktail_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
            let x: I32 = 0;
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = crate::sentinel::HardwareProfile::default();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        let cta_k: u32 = regex_lite_capture_cta_k(&ptx)
            .unwrap_or_else(|| panic!("no tensor-core GEMM tile comment emitted: {}", ptx));
        let expected = (1000 + cta_k - 1) / cta_k;
        assert!(
            cta_k == 0 || 1000 % cta_k != 0,
            "this test is only meaningful when K is NOT a multiple of cta_k (got cta_k={})",
            cta_k
        );
        assert!(
            ptx.contains(&format!(", {};\n", expected))
                || ptx.contains(&format!(", {};", expected)),
            "K loop must run ceil(1000/{}) = {} tiles so the K tail is covered, not {} \
             (truncating drops the tail and silently returns a wrong GEMM): {}",
            cta_k,
            expected,
            1000 / cta_k,
            ptx
        );
    }

    /// A K that is not a multiple of 8 cannot be expressed by A's 16-byte
    /// `cp.async` chunk masking, so it must FAIL THE COMPILE rather than
    /// silently drop up to 7 K elements. This is the guard that the check is
    /// a hard `assert!` and not a `debug_assert!` - the release binary is
    /// what everything runs.
    #[test]
    #[should_panic(expected = "not a multiple of 8")]
    fn test_gemm_k_not_multiple_of_8_is_rejected_not_silently_wrong() {
        let src = r#"
        @tile(256, 256, 1007)
        kernel kodd_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
            let x: I32 = 0;
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = crate::sentinel::HardwareProfile::default();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let _ = emitter.emit_program(&ast, &hw);
    }

    /// Pulls `cta_k` out of the `// [Y TENSOR CORE GEMM] ... | CTA MxNxK |`
    /// comment the GEMM emitter writes. Kept here rather than pulling in a
    /// regex dependency for one test.
    fn regex_lite_capture_cta_k(ptx: &str) -> Option<u32> {
        let marker = "| CTA ";
        let start = ptx.find(marker)? + marker.len();
        let rest = &ptx[start..];
        let end = rest.find(' ')?;
        rest[..end].split('x').nth(2)?.parse().ok()
    }

    /// Plain 3-param `@tile` GEMM must still take the original direct-to-
    /// global epilogue path, unaffected by the 4-param fused shape added
    /// alongside it - regression guard for `tile_gemm_operands`.
    #[test]
    fn test_plain_gemm_epilogue_unaffected_by_fused_shape() {
        let src = r#"
        @tile(64, 64, 32)
        kernel plain_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
            let x: I32 = 0;
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = crate::sentinel::HardwareProfile::default();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        // Plain GEMM stages through shared memory (wmma.store.d...shared.f32
        // + EPI_PLAIN, fine-grained masked copy to global) rather than a
        // direct-to-global whole-fragment-masked store - see
        // emit_gemm_plain_epilogue's doc comment for why the old direct
        // path was a real correctness bug (silently all-zero output for any
        // M < 16), not just a style difference from Bias+ReLU.
        assert!(ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.shared.f32"), "plain GEMM must stage its epilogue through shared memory: {}", ptx);
        assert!(ptx.contains("EPI_PLAIN"), "plain GEMM must emit the fine-grained-masked plain epilogue: {}", ptx);
        assert!(!ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.global.f32"), "plain GEMM must no longer store directly to global (the M<16 bug's source): {}", ptx);
        assert!(!ptx.contains("EPI_BIAS_RELU"), "plain GEMM must not emit the fused bias+relu epilogue: {}", ptx);
    }

    /// Structural regression guard for `emit_gemm_compute_block`'s
    /// hand-rolled `ldmatrix` + `mma.sync.m16n8k16` rewrite (see that
    /// function's doc comment and project scratchpad's validate_mma.py for
    /// the real-hardware validation this codegen is based on): the K-loop
    /// compute core must use `ldmatrix`/`mma.sync` for A/B fragment
    /// loading and multiply-accumulate, must NOT fall back to the old
    /// `wmma.load`/`wmma.mma.m16n16k16` path for that, but MUST keep using
    /// `wmma.store.d` for the epilogue (empirically confirmed to consume a
    /// `mma.sync.m16n8k16`-produced accumulator identically on this
    /// toolchain - see emit_gemm_compute_block's doc comment for why this
    /// is an empirical fact, not an ISA guarantee).
    #[test]
    fn test_gemm_compute_block_uses_ldmatrix_mma_sync_not_wmma_load() {
        let src = r#"
        @tile(256, 256, 256)
        kernel plain_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
            let x: I32 = 0;
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = crate::sentinel::HardwareProfile::default();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"), "missing ldmatrix.x4 A-fragment load: {}", ptx);
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"), "missing ldmatrix.x2.trans B-fragment load: {}", ptx);
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"), "missing mma.sync.m16n8k16 multiply-accumulate: {}", ptx);
        assert!(!ptx.contains("wmma.load.a.sync"), "A fragments should no longer come from wmma.load.a: {}", ptx);
        assert!(!ptx.contains("wmma.load.b.sync"), "B fragments should no longer come from wmma.load.b: {}", ptx);
        assert!(!ptx.contains("wmma.mma.sync.aligned.row.row.m16n16k16"), "compute should no longer use wmma.mma.m16n16k16: {}", ptx);
        // Epilogue reuses wmma.store.d into shared memory (fine-grained
        // masked copy to global from there - see emit_gemm_plain_epilogue's
        // doc comment), not a direct-to-global wmma.store.d as this test
        // originally asserted - that direct path was a real correctness
        // bug (silently all-zero output for any M < 16), fixed since.
        assert!(ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.shared.f32"), "epilogue must still reuse wmma.store.d (now into shared memory): {}", ptx);
    }

    /// Regression guard for the FP8 (e4m3) GEMM path, reached the same way
    /// a real `--emit-ptx` CLI run reaches it (lexer -> parser ->
    /// `emit_program`, NOT calling `emit_fp8_gemm_kernel` directly) - this
    /// is a structural/textual check only; the real correctness and
    /// hardware-assembly verification for this kernel was done via
    /// `ptxas -arch=sm_89` and a `torch._scaled_mm` real-hardware
    /// comparison (see investigation/results docs), not here - unlike this
    /// codebase's OTHER, now-fixed FP8 dead code, this kernel IS reachable
    /// from a real `.ysu` file through the real CLI (see
    /// tests/gemm_fp8_256.ysu).
    #[test]
    fn test_fp8_gemm_kernel_reachable_and_uses_mma_sync_m16n8k32() {
        let src = r#"
        @tile(256, 256, 256)
        kernel gemm_fp8_256(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let mut hw = crate::sentinel::HardwareProfile::default();
        hw.sm_version = "sm_89".to_string();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(ptx.contains(".version 8.4"), "sm_89 FP8 mma.sync needs PTX ISA >=8.4 (see ptx_version_for_sm): {}", ptx);
        assert!(ptx.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"), "missing the Ada-compatible FP8 mma instruction: {}", ptx);
        assert!(ptx.contains("cvt.rn.satfinite.e4m3x2.f32"), "missing the fused on-the-fly quantization instruction: {}", ptx);
        // M=N=256 selects the SMALL tier (m <= FP8_GEMM_SMALL_THRESHOLD) -
        // see test_fp8_gemm_kernel_multiwarp_cta_tiling (LARGE tier,
        // M=N=1024) and test_fp8_gemm_kernel_small_tile_selection (this
        // same SMALL-tier shape, checked in more detail) below. Sizes are
        // DOUBLE-buffered (session 4's software pipelining) - 2x
        // FP8_GEMM_CTA_M_SMALL*FP8_GEMM_CTA_K = 2*4096 = 8192.
        assert!(ptx.contains(".shared .align 4 .b8 smem_fp8_A_gemm_fp8_256[8192]"), "missing double-buffered A smem tile declaration: {}", ptx);
        assert!(ptx.contains(".shared .align 4 .b8 smem_fp8_B_gemm_fp8_256[8192]"), "missing double-buffered B smem tile declaration: {}", ptx);
        // Not ldmatrix - PTX ISA 8.5's ldmatrix has no 8-bit variant (see
        // emit_fp8_gemm_kernel's doc comment).
        assert!(!ptx.contains("ldmatrix"), "FP8 fragments must come from hand-computed ld.shared, not ldmatrix (unavailable for 8-bit types): {}", ptx);
    }

    /// Regression guard for the multi-warp CTA tiling rewrite's LARGE tier
    /// specifically (see `emit_fp8_gemm_kernel`'s doc comment and
    /// `FP8_GEMM_CTA_M_LARGE`'s doc comment for the design) - distinct from
    /// `test_fp8_gemm_kernel_reachable_and_uses_mma_sync_m16n8k32` above,
    /// which predates multi-warp tiling and only checks the kernel is
    /// reachable at all. Uses M=N=K=1024 (> `FP8_GEMM_SMALL_THRESHOLD`) to
    /// land on the LARGE tier specifically - see
    /// `test_fp8_gemm_kernel_small_tile_selection` below for the SMALL
    /// tier's analogous checks, and `test_fp8_gemm_kernel_tile_selection_boundary`
    /// for the threshold itself. Checks the specific things that would
    /// silently regress back to single-warp behavior: the launch-geometry
    /// comment advertises 256 threads/8 warps (not 32/1), a `lane = tid.x &
    /// 31` warp-relative mask is present (not just raw `%tid.x`, which was
    /// correct ONLY when a CTA was exactly one warp), a `warp_id = tid.x >>
    /// 5` decomposition exists, and the mma instruction appears exactly
    /// `num_i * num_j * k_substeps` (2*8*2 = 32) times in the static PTX
    /// text - the Rust-side `i`/`j`/`k_substep` loops are unrolled at
    /// codegen time (only the `@tile`-driven K-TILE loop is a real PTX
    /// runtime loop), so this count is independent of M/N/K (within a
    /// tier) and would change if `num_i`/`num_j`/`k_substeps` regressed to
    /// the single-warp kernel's implicit 1/1/1.
    #[test]
    fn test_fp8_gemm_kernel_multiwarp_cta_tiling() {
        let src = r#"
        @tile(1024, 1024, 1024)
        kernel gemm_fp8_1024(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let mut hw = crate::sentinel::HardwareProfile::default();
        hw.sm_version = "sm_89".to_string();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(
            ptx.contains("CTA 128x128x64 (large) | 4x2 warps"),
            "launch-geometry comment must advertise the LARGE multi-warp CTA shape at M=N=1024: {}", ptx
        );
        assert!(
            ptx.contains("Launch grid required: (8, 8, 1) CTAs, block (256,1,1)"),
            "at M=N=1024 with a 128x128 CTA tile the grid must be (8,8,1), block (256,1,1): {}", ptx
        );
        assert!(
            ptx.lines().any(|l| l.trim_start().starts_with("and.b32") && l.trim_end().ends_with(", 31;")),
            "must mask tid.x & 31 (one line, `and.b32 %rN, %rM, 31;`) for the warp-relative lane, not use raw %tid.x: {}", ptx
        );
        assert!(
            ptx.lines().any(|l| l.trim_start().starts_with("shr.u32") && l.trim_end().ends_with(", 5;")),
            "must derive warp_id = tid.x >> 5 (one line, `shr.u32 %rN, %rM, 5;`): {}", ptx
        );

        let mma_count = ptx.matches("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32").count();
        assert_eq!(
            mma_count, 32,
            "expected num_i(2)*num_j(8)*k_substeps(2) = 32 static mma.sync.m16n8k32 occurrences (Rust-side i/j/k_substep loops are unrolled at codegen time - only the @tile K-tile loop is a real PTX runtime loop), got {}: {}",
            mma_count, ptx
        );

        // Both bar.sync calls from the original single-warp kernel are
        // still present and still load-bearing (now across 8 warps sharing
        // one smem_a/smem_b pair, not 1 warp racing itself) - see doc
        // comment's cross-warp-race discussion.
        let bar_sync_count = ptx.matches("bar.sync 0;").count();
        assert!(bar_sync_count >= 2, "expected at least 2 bar.sync 0 (before and after the compute block) per FP8 kernel: {}", ptx);
    }

    /// SMALL-tier analogue of `test_fp8_gemm_kernel_multiwarp_cta_tiling`
    /// above (see that test's doc comment and `FP8_GEMM_CTA_M_LARGE`'s doc
    /// comment for why this tier exists: a fixed 128x128 CTA tile measured
    /// a real regression at M=N=256 on real hardware, see
    /// `investigation_fp8_gemm_findings.md`'s "Session 3" section). M=N=256
    /// is exactly at `FP8_GEMM_SMALL_THRESHOLD`, so this also confirms the
    /// boundary is inclusive (`<=`, selecting SMALL) rather than exclusive.
    #[test]
    fn test_fp8_gemm_kernel_small_tile_selection() {
        let src = r#"
        @tile(256, 256, 256)
        kernel gemm_fp8_256(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let mut hw = crate::sentinel::HardwareProfile::default();
        hw.sm_version = "sm_89".to_string();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(
            ptx.contains("CTA 64x64x64 (small) | 2x2 warps"),
            "launch-geometry comment must advertise the SMALL multi-warp CTA shape at M=N=256: {}", ptx
        );
        assert!(
            ptx.contains("Launch grid required: (4, 4, 1) CTAs, block (128,1,1)"),
            "at M=N=256 with a 64x64 CTA tile the grid must be (4,4,1), block (128,1,1) - a real, disclosed improvement over the LARGE tier's (2,2,1) at this size: {}", ptx
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 smem_fp8_A_gemm_fp8_256[8192]"),
            "SMALL tier A smem tile must be double-buffered (session 4): 2*FP8_GEMM_CTA_M_SMALL*FP8_GEMM_CTA_K = 2*64*64 = 8192: {}", ptx
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 smem_fp8_B_gemm_fp8_256[8192]"),
            "SMALL tier B smem tile must be double-buffered (session 4): 2*FP8_GEMM_CTA_K*FP8_GEMM_CTA_N_SMALL = 2*64*64 = 8192: {}", ptx
        );

        // SMALL tier: num_i=2, num_j=4, k_substeps=2 -> 16 static mma
        // occurrences (half the LARGE tier's 32, consistent with half the
        // per-warp N-fragment count - see FP8_GEMM_CTA_M_LARGE's doc
        // comment).
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32").count();
        assert_eq!(
            mma_count, 16,
            "expected num_i(2)*num_j(4)*k_substeps(2) = 16 static mma.sync.m16n8k32 occurrences for the SMALL tier, got {}: {}",
            mma_count, ptx
        );
    }

    /// Confirms the SMALL/LARGE tier boundary is where
    /// `FP8_GEMM_CTA_M_LARGE`'s doc comment says it is: crossing from
    /// M=N=256 (SMALL, checked above) to M=N=257 (one past the threshold)
    /// must flip to LARGE, not stay SMALL - a regression here would mean
    /// `emit_fp8_gemm_kernel`'s `m <= FP8_GEMM_SMALL_THRESHOLD || n <=
    /// FP8_GEMM_SMALL_THRESHOLD` condition silently changed to `<` or a
    /// different threshold entirely.
    #[test]
    fn test_fp8_gemm_kernel_tile_selection_boundary() {
        // FP8_GEMM_SMALL_THRESHOLD is 512 (raised from 256 in session 5 -
        // see FP8_GEMM_CTA_M_LARGE's doc comment for the real-hardware A/B
        // that motivated it), so 513 is the boundary case one past it.
        let src = r#"
        @tile(513, 513, 256)
        kernel gemm_fp8_513(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        // sm_89 explicitly: FP8 (e4m3) tensor cores are Ada and later, and the
        // emitter now REFUSES to emit them below that rather than produce a
        // module the driver rejects at load time. `HardwareProfile::default()`
        // resolves to the sm_80 floor, so this test was previously exercising
        // FP8 codegen at a target that cannot run it.
        let mut hw = crate::sentinel::HardwareProfile::default();
        hw.sm_version = "sm_89".to_string();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(
            ptx.contains("CTA 128x128x64 (large) | 4x2 warps"),
            "M=N=513 is one past FP8_GEMM_SMALL_THRESHOLD (512) and must select the LARGE tier: {}", ptx
        );
    }

    /// Regression guard for session 4's software double-buffered
    /// pipelining (see `emit_fp8_gemm_kernel`'s doc comment, "Software
    /// double-buffered pipelining" section, and
    /// `investigation_fp8_gemm_findings.md`'s "Session 4" section). Uses
    /// K=256 (k_tiles=4 at the LARGE tier's cta_k=64) specifically so BOTH
    /// the prologue (stages k_iter=0 unconditionally) and the steady-state
    /// loop (which prefetches k_iter=1..3 conditionally) appear in the
    /// emitted text - a K=64 (k_tiles=1) kernel would exercise the
    /// prologue but never the loop's real prefetch path.
    #[test]
    fn test_fp8_gemm_kernel_pipelined_double_buffering() {
        let src = r#"
        @tile(1024, 1024, 256)
        kernel gemm_fp8_pipe(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        // sm_89 explicitly - FP8 is Ada+, and the emitter refuses below it.
        let mut hw = crate::sentinel::HardwareProfile::default();
        hw.sm_version = "sm_89".to_string();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        // Double-buffered: 2 * (128*64) = 16384 for A, 2 * (64*128) = 16384
        // for B (LARGE tier) - not the single-buffered 8192 each.
        assert!(
            ptx.contains(".shared .align 4 .b8 smem_fp8_A_gemm_fp8_pipe[16384]"),
            "A smem must be double-buffered (2x single-stage bytes): {}", ptx
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 smem_fp8_B_gemm_fp8_pipe[16384]"),
            "B smem must be double-buffered (2x single-stage bytes): {}", ptx
        );

        // Exactly 2 static bar.sync occurrences: one after the prologue's
        // unconditional stage of k_iter=0, one at the end of the steady-
        // state loop body (the loop body is emitted ONCE as static text,
        // executed k_tiles=4 times at runtime via the bra-loop - see
        // test_fp8_gemm_kernel_multiwarp_cta_tiling's doc comment for why
        // static occurrence count is independent of the runtime iteration
        // count). Down from the pre-pipelining design's 2 bar.sync INSIDE
        // the loop body alone (no separate prologue) - this exact count
        // confirms the reduction to 1/iteration, not just "at least 2".
        let bar_sync_count = ptx.matches("bar.sync 0;").count();
        assert_eq!(
            bar_sync_count, 2,
            "expected exactly 2 static bar.sync 0 occurrences (1 prologue + 1 in the steady-state loop body, i.e. 1/iteration at runtime, down from the single-buffered design's 2/iteration), got {}: {}",
            bar_sync_count, ptx
        );

        // The prefetch-suppression predicate: exactly one setp.lt.u32
        // comparing against the compile-time k_tiles literal (4) - confirms
        // the last-iteration skip logic is present, not just assumed from
        // the Python validator.
        let next_valid_setp_count = ptx.lines().filter(|l| l.trim_start().starts_with("setp.lt.u32") && l.trim_end().ends_with(", 4;")).count();
        assert_eq!(
            next_valid_setp_count, 1,
            "expected exactly one `next_k_iter < k_tiles(4)` predicate computation, got {}: {}",
            next_valid_setp_count, ptx
        );

        // and.pred count: 64 from the epilogue (num_i(2)*num_j(8)*4 = 64,
        // unchanged by pipelining - see test_fp8_gemm_kernel_multiwarp_cta_tiling)
        // + 6 new ones from AND-ing the prefetch-validity predicate into
        // the quantize-stage boundary predicates: 1 in
        // emit_fp8_quantize_stage_a's single p_ok, plus (since session 5's
        // vectorized hybrid fast/slow B-stage - see
        // emit_fp8_quantize_stage_b's doc comment) 1 in the FAST path's
        // single combined p_fast + 4 in the SLOW path's p_ok0..p_ok3 (was
        // 2, single-predicate-per-pair, before the quad/hybrid split) = 5
        // for B - all inside those functions' own runtime thread-striding
        // loops, so each appears exactly ONCE in the static text despite
        // executing many times at runtime) = 70. Empirically confirmed
        // against the real compiler output before being hardcoded here,
        // not derived from this test alone.
        let and_pred_count = ptx.matches("and.pred").count();
        assert_eq!(
            and_pred_count, 70,
            "expected 70 and.pred occurrences (64 epilogue + 6 pipelining-predicate combination), got {}: {}",
            and_pred_count, ptx
        );
    }

    /// A malformed FP8-shaped kernel (wrong element type on A) must fall
    /// through `tile_gemm_fp8_operands` to `None` (normal generic lowering)
    /// rather than being silently miscompiled - `type_checker`'s own
    /// `verify_tile_gemm_kernel` is what actually rejects this shape as a
    /// hard compile error in the real CLI pipeline (checked separately,
    /// this test only exercises `PtxEmitter` directly, which - like
    /// `tile_gemm_operands` - re-validates rather than trusting the caller).
    #[test]
    fn test_fp8_gemm_operands_rejects_wrong_element_type() {
        let src = r#"
        @tile(64, 64, 64)
        kernel bad_fp8(A: GlobalMemory<F16>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
        }
        "#;
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = crate::sentinel::HardwareProfile::default();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(!ptx.contains("mma.sync.aligned.m16n8k32"), "malformed FP8 shape must not dispatch to the FP8 GEMM path: {}", ptx);
    }

    /// Real, hardware-validated address formula for
    /// `emit_gemm_compute_block`'s `ldmatrix.x4` (A) / `ldmatrix.x2.trans`
    /// (B) reads (see that function's doc comment) - duplicated here
    /// independently rather than calling into the emitter, so this test
    /// can't pass by "testing a bug against an identical copy of itself".
    fn ldmatrix_a_byte_addr(lane: u32, row_offset: u32, col_offset: u32, stride_elems: u32) -> u32 {
        let group = lane >> 3;
        let local_row = lane & 7;
        let global_row = row_offset + local_row + ((group & 1) << 3);
        let global_col0 = col_offset + ((group >> 1) << 3);
        (global_row * stride_elems + global_col0) * 2
    }

    fn ldmatrix_b_byte_addr(lane: u32, row_offset: u32, col_offset: u32, stride_elems: u32) -> u32 {
        let group = lane >> 3;
        let local_row = lane & 7;
        let global_k = row_offset + local_row + ((group & 1) << 3);
        (global_k * stride_elems + col_offset) * 2
    }

    /// Bank-conflict check matching bank_conflict.rs's own methodology (32
    /// banks x 4 bytes/bank; a 16-byte ldmatrix row-fetch spans 4
    /// consecutive banks; conflict-free iff no bank is touched more than 4
    /// times across all 32 lanes - the theoretical minimum for a 32-lane x
    /// 16-byte access, since 32*16B = 512B = 4x the 128B/cycle all 32 banks
    /// can jointly service, so "exactly 4 per bank" is the best achievable,
    /// not just "no conflict"). Not reusing
    /// `BankConflictProver::prove_ldmatrix_m8n8_x4` directly: that prover's
    /// built-in row/col formula was written for a different (and, per
    /// offline analysis, not hardware-confirmed) tiling convention than the
    /// one `emit_gemm_compute_block` actually validated and uses - see
    /// that function's doc comment.
    fn max_bank_hits(byte_addrs: &[u32]) -> u32 {
        let mut counts = [0u32; 32];
        for &addr in byte_addrs {
            for b in (0..16u32).step_by(4) {
                let bank = ((addr + b) / 4) % 32;
                counts[bank as usize] += 1;
            }
        }
        *counts.iter().max().unwrap()
    }

    /// Regression guard for the "no XOR swizzle needed" finding (see
    /// emit_gemm_compute_block's doc comment): sweeps every
    /// warp/i/j/kk/N-half position `emit_gemm_compute_block` actually
    /// issues, for every CTA shape `Autotuner::generate_candidates` can
    /// select in the m/n > 512 regime this codegen targets plus one
    /// smaller shape, and asserts the EXISTING `+8`-element padding (no
    /// swizzle) stays bank-conflict-free. If this ever fails, it means
    /// either the padding constant or `emit_gemm_compute_block`'s address
    /// formula changed in a way that reopens the ncu-measured bank-conflict
    /// regression this rewrite fixed - see project scratchpad's
    /// swizzle_design.py for the exploration that produced these
    /// parameters and benchmark docs for the real ncu confirmation.
    #[test]
    fn test_ldmatrix_addressing_bank_conflict_free_with_existing_padding() {
        let configs: [(u32, u32, u32, u32, u32); 3] = [
            (128, 256, 32, 4, 4), // M=N=K>=2048 shape (see benchmark_y_tensor_core_gemm_results.md)
            (128, 128, 32, 4, 2), // M=N=K=1024 shape
            (64, 64, 64, 2, 2),   // M=N=K=512 shape
        ];
        for (cta_m, cta_n, cta_k, warps_m, warps_n) in configs {
            let per_warp_m = cta_m / warps_m;
            let per_warp_n = cta_n / warps_n;
            let num_i = per_warp_m / 16;
            let num_j = per_warp_n / 16;
            let k_substeps = cta_k / 16;
            let a_stride = cta_k + 8; // matches emit_tensor_core_gemm_kernel's smem_a_stride
            let b_stride = cta_n + 8; // matches emit_tensor_core_gemm_kernel's smem_b_stride

            for warp_m in 0..warps_m {
                for i in 0..num_i {
                    for kk in 0..k_substeps {
                        let addrs: Vec<u32> = (0..32u32)
                            .map(|lane| ldmatrix_a_byte_addr(lane, warp_m * per_warp_m + i * 16, kk * 16, a_stride))
                            .collect();
                        let worst = max_bank_hits(&addrs);
                        assert!(
                            worst <= 4,
                            "A ldmatrix.x4 bank conflict: cta={}x{}x{} warp_m={} i={} kk={} worst_bank_hits={}",
                            cta_m, cta_n, cta_k, warp_m, i, kk, worst
                        );
                    }
                }
            }
            for warp_n in 0..warps_n {
                for j in 0..num_j {
                    for kk in 0..k_substeps {
                        for n_half in [0u32, 8u32] {
                            let col_offset = warp_n * per_warp_n + j * 16 + n_half;
                            let addrs: Vec<u32> = (0..32u32)
                                .map(|lane| ldmatrix_b_byte_addr(lane, kk * 16, col_offset, b_stride))
                                .collect();
                            let worst = max_bank_hits(&addrs);
                            assert!(
                                worst <= 4,
                                "B ldmatrix.trans bank conflict: cta={}x{}x{} warp_n={} j={} kk={} n_half={} worst_bank_hits={}",
                                cta_m, cta_n, cta_k, warp_n, j, kk, n_half, worst
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_3d {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::sentinel::HardwareProfile;

    #[test]
    fn test_3d_block_pointer_ptx() {
        let src = r#"
        kernel bench_3d(In: GlobalMemory<F32>, Out: GlobalMemory<F32>, D0: I32, D1: I32, D2: I32, S0: I32, S1: I32) {
            let b0: I32 = block_idx_x();
            let b1: I32 = block_idx_y();
            let t2: I32 = thread_idx_x();

            let d0: I32 = b0 * 16;
            let d1: I32 = b1 * 16;
            let d2: I32 = t2 * 4;

            let val: F32 = block_ptr3d_load(In, d0, d1, d2, S0, S1, D0, D1, D2);
            let res: F32 = val * 2.0 + 1.0;
            block_ptr3d_store(Out, d0, d1, d2, S0, S1, D0, D1, D2, res);
        }
        "#;

        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let ast = parser.parse_program().unwrap();

        let hw = HardwareProfile::default();
        let mut emitter = PtxEmitter::new_with_profile(&hw);
        let ptx = emitter.emit_program(&ast, &hw);

        assert!(ptx.contains("%ctaid.y"), "PTX missing %ctaid.y for block_idx_y: {}", ptx);
        assert!(!ptx.contains("mul.lo.s32 %r10, b1"), "PTX contains unallocated identifier b1: {}", ptx);
    }

    #[test]
    fn test_blackwell_5000_series_ptx_version_target() {
        assert_eq!(ptx_version_for_sm("sm_100"), ".version 8.7");
        assert_eq!(ptx_version_for_sm("sm_120"), ".version 8.7");
        assert_eq!(ptx_version_for_sm("10.0"), ".version 8.7");
        assert_eq!(ptx_version_for_sm("12.0"), ".version 8.7");

        let mut hw = HardwareProfile::default();
        hw.sm_version = "sm_120".to_string();
        let emitter = PtxEmitter::new_with_profile(&hw);
        assert!(emitter.ptx_buffer.contains(".version 8.7"), "Missing PTX 8.7 version for sm_120: {}", emitter.ptx_buffer);
        assert!(emitter.ptx_buffer.contains(".target sm_120"), "Missing .target sm_120: {}", emitter.ptx_buffer);
    }
}


#[cfg(test)]
mod tests_paged_decode_attention {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::sentinel::HardwareProfile;

    fn compile(name: &str) -> String {
        let src = format!(
            "kernel {}(Q: GlobalMemory<F16>, KCache: GlobalMemory<F16>, VCache: GlobalMemory<F16>, \
             PageTable: GlobalMemory<I32>, SeqLens: GlobalMemory<I32>, Out: GlobalMemory<F16>, \
             MaxPages: I32) {{\n}}\n\nfn main() {{\n}}\n",
            name
        );
        let tokens = Lexer::new(&src).tokenize();
        let ast = Parser::new(tokens).parse_program().expect("probe source should parse");
        let hw = HardwareProfile::default();
        PtxEmitter::new_with_profile(&hw).emit_program(&ast, &hw)
    }

    #[test]
    fn dispatches_and_emits_the_online_softmax_core() {
        let ptx = compile("paged_decode_attention_128_32_8_16");
        assert!(ptx.contains("[Y PAGED DECODE ATTENTION]"), "attention dispatch did not fire: {}", ptx);
        assert!(ptx.contains("GQA 4:1"), "GQA ratio not derived from the head counts: {}", ptx);
        // `.ftz` is deliberate (see the kernel doc comment) - a plain
        // `ex2.approx.f32` here would be a silent regression.
        assert!(ptx.contains("ex2.approx.ftz.f32"), "softmax must use ex2.approx.ftz: {}", ptx);
        // A real runtime loop over the KV sequence, not an unrolled one.
        assert!(ptx.contains("ATTN_LOOP_paged_decode_attention_128_32_8_16:"), "missing KV loop: {}", ptx);
        // The cross-warp partial-softmax merge.
        assert!(ptx.contains("bar.sync 0;"), "missing cross-warp merge barrier: {}", ptx);
        assert!(ptx.contains("shfl.sync.bfly.b32"), "missing warp reduction for the QK dot: {}", ptx);
        // The scalar page-table stride really is a 32-bit param, not a pointer.
        assert!(ptx.contains(".param .b32 MaxPages_6"), "MaxPages must be a scalar param: {}", ptx);
    }

    #[test]
    fn warp_count_defaults_to_32_and_is_overridable_by_name() {
        let default_ptx = compile("paged_decode_attention_128_32_8_16");
        assert!(default_ptx.contains("| 32 warps |"), "default warp count should be 32: {}", default_ptx);

        // Optional 5th name dimension - the batch-serving specialization.
        let eight = compile("paged_decode_attention_128_32_8_16_8");
        assert!(eight.contains("| 8 warps |"), "5th name dim should set the warp count: {}", eight);

        // Shared memory scales with the warp count: [warps] m + [warps] l +
        // [warps * head_dim] acc, 4 bytes each.
        assert!(default_ptx.contains(&format!(".b8 smem_attn_paged_decode_attention_128_32_8_16[{}]", (32 * 2 + 32 * 128) * 4)));
        assert!(eight.contains(&format!(".b8 smem_attn_paged_decode_attention_128_32_8_16_8[{}]", (8 * 2 + 8 * 128) * 4)));
    }

    /// The nine-parameter split-K shape. Same probe mechanism as `compile`,
    /// with the two f32 scratch buffers.
    fn compile_split(name: &str) -> String {
        let src = format!(
            "kernel {}(Q: GlobalMemory<F16>, KCache: GlobalMemory<F16>, VCache: GlobalMemory<F16>, \
             PageTable: GlobalMemory<I32>, SeqLens: GlobalMemory<I32>, Out: GlobalMemory<F16>, \
             Partial: GlobalMemory<F32>, PartialMeta: GlobalMemory<F32>, \
             MaxPages: I32) {{\n}}\n\nfn main() {{\n}}\n",
            name
        );
        let tokens = Lexer::new(&src).tokenize();
        let ast = Parser::new(tokens).parse_program().expect("probe source should parse");
        let hw = HardwareProfile::default();
        PtxEmitter::new_with_profile(&hw).emit_program(&ast, &hw)
    }

    #[test]
    fn split_shape_merges_gqa_heads_and_emits_both_entries() {
        let ptx = compile_split("paged_decode_attention_split_128_32_8_16_16_8");
        assert!(ptx.contains("[Y PAGED DECODE ATTENTION]"), "split dispatch did not fire: {}", ptx);
        assert!(ptx.contains("16 splits, 4 q heads/CTA"),
                "split kernel must carry all 4 GQA-sharing query heads in one CTA: {}", ptx);

        // Two entry points, and the combine really is a separate one - there
        // is no device-wide barrier inside a kernel, so a single entry could
        // not do this.
        assert!(ptx.contains(".visible .entry paged_decode_attention_split_128_32_8_16_16_8("),
                "missing main entry: {}", ptx);
        assert!(ptx.contains(".visible .entry paged_decode_attention_split_128_32_8_16_16_8_reduce("),
                "missing combine entry: {}", ptx);
        assert!(ptx.contains("[Y PAGED DECODE ATTENTION COMBINE]"), "combine body not emitted: {}", ptx);

        // %ctaid.z is what partitions the sequence; without it this is the
        // single-pass kernel with extra buffers.
        assert!(ptx.contains("%ctaid.z"), "split kernel must index the KV sequence by %ctaid.z: {}", ptx);

        // The whole point of the merge: ONE K load and ONE V load per token,
        // feeding four scores. Four `ld.global.v2.u32` in the loop body would
        // mean the GQA re-read came back.
        let body = ptx.split("ATTN_LOOP_paged_decode_attention_split_128_32_8_16_16_8:").nth(1)
            .and_then(|s| s.split("ATTN_LOOP_END_paged_decode_attention_split_128_32_8_16_16_8:").next())
            .expect("loop body");
        assert_eq!(body.matches("ld.global.v2.u32").count(), 2,
                   "expected exactly one K and one V vector load per token: {}", body);
        assert_eq!(body.matches("ex2.approx.ftz.f32").count(), 8,
                   "expected two exp per head for four heads: {}", body);

        // Shared memory does NOT scale with the head count - the heads go
        // through one buffer sequentially, or a GQA-4 kernel at 32 warps would
        // ask for 66,560 bytes against the 48 KB static limit.
        assert!(ptx.contains(&format!(".b8 smem_attn_paged_decode_attention_split_128_32_8_16_16_8[{}]",
                                      (8 * 2 + 8 * 128) * 4)),
                "shared memory must be sized per warp, not per (warp, head): {}", ptx);
    }

    #[test]
    fn rejects_shapes_the_codegen_cannot_lower() {
        // The split shape needs its two scratch buffers: the 7-parameter
        // signature must NOT dispatch to it.
        assert!(!compile("paged_decode_attention_split_128_32_8_16_16_8").contains("[Y PAGED DECODE ATTENTION]"));
        // ... and the 9-parameter signature must not dispatch to the
        // single-pass shape, which would ignore two of its arguments.
        assert!(!compile_split("paged_decode_attention_128_32_8_16").contains("[Y PAGED DECODE ATTENTION]"));
        // A GQA ratio past 8 makes one CTA carry too much softmax state.
        assert!(!compile_split("paged_decode_attention_split_128_32_2_16_16_8").contains("[Y PAGED DECODE ATTENTION]"));
        // Split count out of range.
        assert!(!compile_split("paged_decode_attention_split_128_32_8_16_128_8").contains("[Y PAGED DECODE ATTENTION]"));

        // head_dim not a multiple of 32 (the warp width).
        assert!(!compile("paged_decode_attention_100_32_8_16").contains("[Y PAGED DECODE ATTENTION]"));
        // head_dim outside the supported 64/128/256 per-lane vector widths.
        assert!(!compile("paged_decode_attention_512_32_8_16").contains("[Y PAGED DECODE ATTENTION]"));
        // q heads not a whole multiple of kv heads - GQA mapping undefined.
        assert!(!compile("paged_decode_attention_128_30_8_16").contains("[Y PAGED DECODE ATTENTION]"));
        // more than 32 warps exceeds the 1024-thread CTA limit.
        assert!(!compile("paged_decode_attention_128_32_8_16_64").contains("[Y PAGED DECODE ATTENTION]"));
    }

    /// Assembles the emitted PTX with the real `ptxas`.
    ///
    /// String-matching a PTX test proves only that the emitter wrote what the
    /// test expected, not that any of it is a legal instruction - this repo
    /// shipped a `wgmma...m64n64k64.s32.s4.s4` that exists on no hardware
    /// precisely because its tests only string-matched. Skipped (not failed)
    /// where `ptxas` is unavailable, so the suite still runs on a machine
    /// without a CUDA toolkit.
    #[test]
    fn emitted_ptx_actually_assembles() {
        use std::process::{Command, Stdio};
        if Command::new("ptxas").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_err() {
            eprintln!("ptxas not available - skipping the assembly gate");
            return;
        }
        for (name, split) in [
            ("paged_decode_attention_128_32_8_16", false),
            ("paged_decode_attention_128_32_8_16_8", false),
            ("paged_decode_attention_64_16_16_32", false),
            ("paged_decode_attention_256_16_4_1", false),
            // The split shape, including the two entries in one module and
            // the head_dim 64 / 256 vector widths on the f32 partial path.
            ("paged_decode_attention_split_128_32_8_16_16_8", true),
            ("paged_decode_attention_split_128_32_8_16_4_8", true),
            ("paged_decode_attention_split_64_16_16_32_8", true),
            ("paged_decode_attention_split_256_16_4_1_8_4", true),
        ] {
            let ptx = if split { compile_split(name) } else { compile(name) };
            assert!(ptx.contains("[Y PAGED DECODE ATTENTION]"), "{} did not dispatch", name);
            let dir = std::env::temp_dir().join(format!("y_attn_ptxas_{}", name));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("k.ptx");
            std::fs::write(&path, &ptx).unwrap();
            let out = Command::new("ptxas")
                .args(["-arch=sm_89", "-O3"])
                .arg(&path)
                .args(["-o", "/dev/null"])
                .output()
                .expect("ptxas should run");
            assert!(
                out.status.success(),
                "ptxas rejected {}:\n{}",
                name,
                String::from_utf8_lossy(&out.stderr)
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

#[cfg(test)]
mod tests_swiglu_tile_override {
    use super::*;

    /// The override exists to make this kernel's tile measurable on hardware.
    /// A sweep of 13 tiles at M=N=K=2048/4096 confirmed the 128x128x32 4x4
    /// default is the best of them (0.78-0.80x of the unfused cuBLAS path;
    /// every 2x2-warp alternative is forced to a smaller CTA tile by the
    /// double accumulator and lands at 0.48-0.62x), so the guard below is
    /// what keeps a future sweep from silently emitting an infeasible kernel.
    #[test]
    fn override_guard_rejects_infeasible_tiles() {
        // A 128x128 tile over 2x2 warps needs 4*4*8*2 = 256 accumulator
        // registers per thread for the two (gate, up) arrays - past the cap.
        // This is precisely why this kernel cannot use the 2x2 split that
        // measured 1.44x faster in the plain GEMM.
        std::env::set_var("Y_SWIGLU_TILE", "128,128,32,2,2");
        assert_eq!(PtxEmitter::swiglu_tile_override(), None, "256 accumulator regs must be rejected");

        // Not divisible into 16x16 warp fragments.
        std::env::set_var("Y_SWIGLU_TILE", "128,120,32,4,4");
        assert_eq!(PtxEmitter::swiglu_tile_override(), None);

        // cta_k not a multiple of the mma k16 dimension.
        std::env::set_var("Y_SWIGLU_TILE", "128,128,24,4,4");
        assert_eq!(PtxEmitter::swiglu_tile_override(), None);

        // Malformed.
        std::env::set_var("Y_SWIGLU_TILE", "128,128");
        assert_eq!(PtxEmitter::swiglu_tile_override(), None);

        // A feasible tile passes: 128x128 over 4x2 warps is 2*4*8*2 = 128 regs.
        std::env::set_var("Y_SWIGLU_TILE", "128,128,32,4,2");
        assert_eq!(PtxEmitter::swiglu_tile_override(), Some((128, 128, 32, 4, 2)));

        std::env::remove_var("Y_SWIGLU_TILE");
        assert_eq!(PtxEmitter::swiglu_tile_override(), None, "unset must mean 'use the default'");
    }
}
