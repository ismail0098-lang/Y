// ============================================================
//  Y  —  Vectorized Quantization Pass
//  quantization_pass.rs
//
//  Solves the Precision & Fragment Transformation constraint:
//
//  RT Cores natively output FP32 attributes (intersection
//  distances, hit normals, barycentric coords). Tensor Cores
//  expect tightly packed low-precision matrix fragments:
//    - FP16 (half)  — standard mixed-precision GEMM
//    - BF16         — brain floating point for training
//    - FP8 (E4M3)   — Hopper+ inference
//    - FP4 / INT4   — extreme quantization
//
//  This module emits automated, vectorized quantization and
//  packing passes directly into the PTX register pipeline.
//  Key optimizations:
//    - half2 packing: two FP16 values in a single 32-bit register
//    - Vectorized cvt instructions (cvt.rn.f16x2.f32)
//    - In-register packing avoids SMEM round-trips where possible
// ============================================================

#![allow(dead_code)]

use crate::ir_grapher::Precision;
use crate::sentinel::HardwareProfile;
use std::fmt::Write;

/// Emits vectorized quantization/packing PTX code.
pub struct QuantizationPass {
    reg_f32: u32,
    reg_f16: u32,
    reg_u32: u32,
    reg_u64: u32,
    reg_pred: u32,
    label_count: u32,
}

impl QuantizationPass {
    pub fn new() -> Self {
        Self {
            reg_f32: 0,
            reg_f16: 0,
            reg_u32: 0,
            reg_u64: 0,
            reg_pred: 0,
            label_count: 0,
        }
    }

    fn alloc_f32(&mut self) -> String {
        let r = format!("%qf{}", self.reg_f32);
        self.reg_f32 += 1;
        r
    }

    fn alloc_f16(&mut self) -> String {
        let r = format!("%qh{}", self.reg_f16);
        self.reg_f16 += 1;
        r
    }

    fn alloc_u32(&mut self) -> String {
        let r = format!("%qr{}", self.reg_u32);
        self.reg_u32 += 1;
        r
    }

    fn alloc_u64(&mut self) -> String {
        let r = format!("%qrd{}", self.reg_u64);
        self.reg_u64 += 1;
        r
    }

    fn alloc_label(&mut self, prefix: &str) -> String {
        let l = format!("$QUANT_{}_{}", prefix, self.label_count);
        self.label_count += 1;
        l
    }

    /// Emit vectorized quantization from src_precision to dst_precision,
    /// operating on data in shared memory at [smem_offset..smem_offset+smem_bytes].
    pub fn emit_vectorized_quantization(
        &mut self,
        src: Precision,
        dst: Precision,
        smem_offset: u32,
        smem_bytes: u32,
        hw: &HardwareProfile,
    ) -> Result<String, String> {
        let mut out = String::new();

        // An identity conversion is a legitimate no-op, not an unsupported
        // pair: the bits already have the destination's meaning.
        if src == dst {
            writeln!(
                &mut out,
                "    // -- NO QUANTIZATION NEEDED: already {:?} --",
                src
            )
            .unwrap();
            return Ok(out);
        }

        writeln!(
            &mut out,
            "    // -- VECTORIZED QUANTIZATION PASS: {:?} -> {:?} --",
            src, dst
        )
        .unwrap();
        writeln!(
            &mut out,
            "    // Operating on {} bytes at SMEM[{}..{}]",
            smem_bytes,
            smem_offset,
            smem_offset + smem_bytes
        )
        .unwrap();

        match (src, dst) {
            (Precision::FP32, Precision::FP16) => {
                self.emit_fp32_to_fp16_half2(&mut out, smem_offset, smem_bytes, hw);
            }
            (Precision::FP32, Precision::BF16) => {
                self.emit_fp32_to_bf16(&mut out, smem_offset, smem_bytes, hw);
            }
            // Every other pair is REFUSED, not approximated. This arm used to
            // call `emit_scalar_fallback`, which emitted two PTX comments and
            // no instructions -- so the conversion silently did not happen and
            // the consumer read the source precision's bits as the destination
            // type. That is a reinterpretation, not a fallback, and it sits on
            // the cross-pipeline barrier path where the tensor core is the
            // consumer. 7 of the 64 ordered pairs are implemented; naming the
            // gap costs a line number, guessing costs a wrong kernel.
            _ => {
                return Err(format!(
                    "[Y COPROCESSOR] No quantization path from {:?} to {:?}. \
                     Implemented: FP32 -> FP16 and FP32 -> BF16, plus any identity conversion.",
                    src, dst
                ));
            }
        }

        Ok(out)
    }

    /// FP32 → FP16 with half2 packing.
    /// Processes two FP32 values at a time, packing them into a single
    /// 32-bit register as {half_lo, half_hi} = half2.
    fn emit_fp32_to_fp16_half2(
        &mut self,
        out: &mut String,
        _smem_offset: u32,
        smem_bytes: u32,
        hw: &HardwareProfile,
    ) {
        let num_fp32_elements = smem_bytes / 4;
        let num_half2_pairs = num_fp32_elements / 2;

        writeln!(
            out,
            "    // half2 packing: {} FP32 -> {} half2 (2x compression)",
            num_fp32_elements, num_half2_pairs
        )
        .unwrap();
        writeln!(
            out,
            "    // Estimated cost: {:.1} cycles per pair ({:.1} cy * {} pairs)",
            hw.f2h_latency_cycles,
            hw.f2h_latency_cycles,
            num_half2_pairs
        )
        .unwrap();

        let loop_start = self.alloc_label("HALF2_PACK");
        let loop_end = self.alloc_label("HALF2_DONE");
        let iter_reg = self.alloc_u32();
        let limit_reg = self.alloc_u32();

        writeln!(out, "    {{").unwrap();
        writeln!(out, "        .reg .b32 %tid_x;").unwrap();
        writeln!(out, "        .reg .b32 %ntid_x;").unwrap();
        writeln!(out, "        mov.u32 %tid_x, %tid.x;").unwrap();
        writeln!(out, "        mov.u32 %ntid_x, %ntid.x;").unwrap();
        writeln!(out, "        mov.u32 {}, %tid_x;", iter_reg).unwrap();
        writeln!(out, "        mov.u32 {}, {};", limit_reg, num_half2_pairs).unwrap();
        writeln!(out, "        {}:", loop_start).unwrap();

        // Load coprocessor_smem shared address
        let smem_gen = self.alloc_u64();
        writeln!(out, "        mov.u64 {}, coprocessor_smem;", smem_gen).unwrap();

        // Calculate source and destination SMEM addresses
        let src_addr = self.alloc_u64();
        let dst_addr = self.alloc_u64();
        let offset_bytes = self.alloc_u64();

        // src_offset = iter * 8 (two FP32 = 8 bytes per iteration)
        writeln!(out, "        cvt.u64.u32 {}, {};", offset_bytes, iter_reg).unwrap();
        writeln!(out, "        shl.b64 {}, {}, 3;", offset_bytes, offset_bytes).unwrap();
        writeln!(
            out,
            "        add.u64 {}, {}, {};",
            src_addr, smem_gen, offset_bytes
        )
        .unwrap();

        // dst_offset = iter * 4 (one half2 = 4 bytes per iteration)
        let dst_offset = self.alloc_u64();
        writeln!(out, "        cvt.u64.u32 {}, {};", dst_offset, iter_reg).unwrap();
        writeln!(out, "        shl.b64 {}, {}, 2;", dst_offset, dst_offset).unwrap();
        // Output goes after the input region
        writeln!(
            out,
            "        add.u64 {}, {}, {};",
            dst_addr, smem_gen, dst_offset
        )
        .unwrap();
        writeln!(
            out,
            "        add.u64 {}, {}, {};",
            dst_addr, dst_addr, smem_bytes as u64
        )
        .unwrap();

        // Load two FP32 values
        let val_lo = self.alloc_f32();
        let val_hi = self.alloc_f32();
        writeln!(
            out,
            "        ld.shared.f32 {}, [{}];     // val_lo = FP32[2*i]",
            val_lo, src_addr
        )
        .unwrap();
        writeln!(
            out,
            "        ld.shared.f32 {}, [{}+4];   // val_hi = FP32[2*i+1]",
            val_hi, src_addr
        )
        .unwrap();

        // Convert to FP16 and pack into half2
        let packed = self.alloc_u32();
        writeln!(
            out,
            "        // cvt.rn.f16x2.f32: vectorized FP32->FP16 + half2 pack"
        )
        .unwrap();
        writeln!(
            out,
            "        cvt.rn.f16x2.f32 {}, {}, {};  // packed = {{f16(lo), f16(hi)}}",
            packed, val_hi, val_lo
        )
        .unwrap();

        // Store packed half2 to destination SMEM region
        writeln!(
            out,
            "        st.shared.b32 [{}], {};  // Store half2 fragment",
            dst_addr, packed
        )
        .unwrap();

        let pred = format!("%qp{}", self.reg_pred);
        self.reg_pred += 1;
        writeln!(out, "        add.u32 {}, {}, %ntid_x;", iter_reg, iter_reg).unwrap();
        writeln!(
            out,
            "        setp.lt.u32 {}, {}, {};",
            pred, iter_reg, limit_reg
        )
        .unwrap();
        writeln!(out, "        @{} bra {};", pred, loop_start).unwrap();
        writeln!(out, "        {}:", loop_end).unwrap();
        writeln!(out, "    }}").unwrap();

        writeln!(
            out,
            "    // half2 pack complete: {} bytes FP32 -> {} bytes FP16",
            smem_bytes,
            smem_bytes / 2
        )
        .unwrap();
    }

    /// FP32 → BF16 conversion.
    /// BF16 = truncate FP32 mantissa (keep upper 16 bits of the 32-bit float).
    fn emit_fp32_to_bf16(
        &mut self,
        out: &mut String,
        _smem_offset: u32,
        smem_bytes: u32,
        _hw: &HardwareProfile,
    ) {
        let num_elements = smem_bytes / 4;

        writeln!(
            out,
            "    // FP32 -> BF16: truncate mantissa ({} elements)",
            num_elements
        )
        .unwrap();
        writeln!(
            out,
            "    // BF16 = upper 16 bits of FP32 (same exponent range, reduced precision)"
        )
        .unwrap();

        let loop_start = self.alloc_label("BF16_CVT");
        let loop_end = self.alloc_label("BF16_DONE");
        let iter = self.alloc_u32();
        let limit = self.alloc_u32();

        let smem_gen = self.alloc_u64();
        writeln!(out, "    mov.u64 {}, coprocessor_smem;", smem_gen).unwrap();

        writeln!(out, "    {{").unwrap();
        writeln!(out, "        .reg .b32 %tid_x;").unwrap();
        writeln!(out, "        .reg .b32 %ntid_x;").unwrap();
        writeln!(out, "        mov.u32 %tid_x, %tid.x;").unwrap();
        writeln!(out, "        mov.u32 %ntid_x, %ntid.x;").unwrap();
        writeln!(out, "        mov.u32 {}, %tid_x;", iter).unwrap();
        writeln!(out, "        mov.u32 {}, {};", limit, num_elements / 2).unwrap();
        writeln!(out, "        {}:", loop_start).unwrap();

        let val_lo = self.alloc_f32();
        let val_hi = self.alloc_f32();
        let src = self.alloc_u64();
        let offset = self.alloc_u64();
        writeln!(out, "        cvt.u64.u32 {}, {};", offset, iter).unwrap();
        writeln!(out, "        shl.b64 {}, {}, 3;", offset, offset).unwrap();
        writeln!(
            out,
            "        add.u64 {}, {}, {};",
            src, smem_gen, offset
        )
        .unwrap();
        writeln!(out, "        ld.shared.f32 {}, [{}];", val_lo, src).unwrap();
        writeln!(out, "        ld.shared.f32 {}, [{}+4];", val_hi, src).unwrap();

        // BF16 truncation: shift right by 16 to get upper 16 bits
        let bits_lo = self.alloc_u32();
        let bits_hi = self.alloc_u32();
        let packed = self.alloc_u32();

        writeln!(
            out,
            "        mov.b32 {}, {};  // reinterpret FP32 as U32",
            bits_lo, val_lo
        )
        .unwrap();
        writeln!(
            out,
            "        mov.b32 {}, {};",
            bits_hi, val_hi
        )
        .unwrap();
        // Round-to-nearest: add 0x7FFF + bit[16] for rounding
        writeln!(
            out,
            "        // Round-to-nearest-even BF16 truncation"
        )
        .unwrap();
        let rnd_bias = self.alloc_u32();
        let bit16 = self.alloc_u32();
        writeln!(out, "        bfe.u32 {}, {}, 16, 1;", bit16, bits_lo).unwrap();
        writeln!(out, "        add.u32 {}, {}, 0x7FFF;", rnd_bias, bit16).unwrap();
        writeln!(out, "        add.u32 {}, {}, {};", bits_lo, bits_lo, rnd_bias).unwrap();
        writeln!(out, "        shr.u32 {}, {}, 16;", bits_lo, bits_lo).unwrap();

        let bit16_hi = self.alloc_u32();
        let rnd_bias_hi = self.alloc_u32();
        writeln!(out, "        bfe.u32 {}, {}, 16, 1;", bit16_hi, bits_hi).unwrap();
        writeln!(out, "        add.u32 {}, {}, 0x7FFF;", rnd_bias_hi, bit16_hi).unwrap();
        writeln!(out, "        add.u32 {}, {}, {};", bits_hi, bits_hi, rnd_bias_hi).unwrap();
        // Pack: hi in upper 16, lo in lower 16
        writeln!(
            out,
            "        and.b32 {}, {}, 0xFFFF0000;  // hi BF16 in upper bits",
            bits_hi, bits_hi
        )
        .unwrap();
        writeln!(
            out,
            "        or.b32 {}, {}, {};  // packed = {{bf16_hi, bf16_lo}}",
            packed, bits_hi, bits_lo
        )
        .unwrap();

        let dst = self.alloc_u64();
        let dst_off = self.alloc_u64();
        writeln!(out, "        cvt.u64.u32 {}, {};", dst_off, iter).unwrap();
        writeln!(out, "        shl.b64 {}, {}, 2;", dst_off, dst_off).unwrap();
        writeln!(
            out,
            "        add.u64 {}, {}, {};",
            dst, smem_gen, dst_off
        )
        .unwrap();
        writeln!(
            out,
            "        add.u64 {}, {}, {};",
            dst, dst, smem_bytes as u64
        )
        .unwrap();
        writeln!(out, "        st.shared.b32 [{}], {};", dst, packed).unwrap();

        let pred = format!("%qp{}", self.reg_pred);
        self.reg_pred += 1;
        writeln!(out, "        add.u32 {}, {}, %ntid_x;", iter, iter).unwrap();
        writeln!(out, "        setp.lt.u32 {}, {}, {};", pred, iter, limit).unwrap();
        writeln!(out, "        @{} bra {};", pred, loop_start).unwrap();
        writeln!(out, "        {}:", loop_end).unwrap();
        writeln!(out, "    }}").unwrap();
    }

    /// Emits single-pass epilogue fusion directly on accumulator registers (%f0..%f3).
    /// Performs inline bias addition + activation (ReLU / GELU / SiLU) before store.
    pub fn emit_epilogue_fusion(
        &mut self,
        accumulators: &[&str],
        bias_ptr_reg: Option<&str>,
        activation: ActivationKind,
    ) -> String {
        let mut out = String::new();

        writeln!(
            &mut out,
            "    // -- IN-REGISTER EPILOGUE FUSION (Bias + {:?}) --",
            activation
        )
        .unwrap();

        for (i, acc) in accumulators.iter().enumerate() {
            // 1. Bias Addition
            if let Some(bias_ptr) = bias_ptr_reg {
                let bias_reg = self.alloc_f32();
                writeln!(
                    &mut out,
                    "    ld.global.f32 {}, [{}+{}];  // load bias for lane element {}",
                    bias_reg,
                    bias_ptr,
                    i * 4,
                    i
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "    add.f32 {}, {}, {};  // inline bias add",
                    acc, acc, bias_reg
                )
                .unwrap();
            }

            // 2. Activation Function
            match activation {
                ActivationKind::None => {}
                ActivationKind::ReLU => {
                    writeln!(
                        &mut out,
                        "    max.f32 {}, {}, 0f00000000;  // inline ReLU",
                        acc, acc
                    )
                    .unwrap();
                }
                ActivationKind::SiLU => {
                    let neg_x = self.alloc_f32();
                    let scaled_neg_x = self.alloc_f32();
                    let exp_val = self.alloc_f32();
                    let denom = self.alloc_f32();
                    let sig = self.alloc_f32();

                    writeln!(&mut out, "    neg.f32 {}, {};", neg_x, acc).unwrap();
                    writeln!(
                        &mut out,
                        "    mul.f32 {}, {}, 0f3fb8aa3b;  // -x * log2(e)",
                        scaled_neg_x, neg_x
                    )
                    .unwrap();
                    writeln!(
                        &mut out,
                        "    ex2.approx.f32 {}, {};",
                        exp_val, scaled_neg_x
                    )
                    .unwrap();
                    writeln!(
                        &mut out,
                        "    add.f32 {}, {}, 0f3f800000;  // 1.0 + exp(-x)",
                        denom, exp_val
                    )
                    .unwrap();
                    writeln!(&mut out, "    rcp.approx.f32 {}, {};", sig, denom).unwrap();
                    writeln!(
                        &mut out,
                        "    mul.f32 {}, {}, {};  // SiLU = x * sigmoid(x)",
                        acc, acc, sig
                    )
                    .unwrap();
                }
                ActivationKind::GELU => {
                    let x_sq = self.alloc_f32();
                    let poly = self.alloc_f32();
                    let g_in = self.alloc_f32();
                    let t_val = self.alloc_f32();
                    let t_one = self.alloc_f32();
                    let half_x = self.alloc_f32();

                    writeln!(&mut out, "    mul.f32 {}, {}, {};", x_sq, acc, acc).unwrap();
                    writeln!(
                        &mut out,
                        "    fma.rn.f32 {}, {}, 0f3d372713, 0f3f79b5c3;  // 0.044715*x^2 + 0.79788456",
                        poly, x_sq
                    )
                    .unwrap();
                    writeln!(&mut out, "    mul.f32 {}, {}, {};", g_in, acc, poly).unwrap();
                    writeln!(&mut out, "    tanh.approx.f32 {}, {};", t_val, g_in).unwrap();
                    writeln!(
                        &mut out,
                        "    add.f32 {}, {}, 0f3f800000;  // 1 + tanh(...)",
                        t_one, t_val
                    )
                    .unwrap();
                    writeln!(
                        &mut out,
                        "    mul.f32 {}, {}, 0f3f000000;  // 0.5 * x",
                        half_x, acc
                    )
                    .unwrap();
                    writeln!(
                        &mut out,
                        "    mul.f32 {}, {}, {};  // GELU",
                        acc, half_x, t_one
                    )
                    .unwrap();
                }
            }
        }

        out
    }
}

/// Activation functions supported for inline epilogue fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    None,
    ReLU,
    GELU,
    SiLU,
}

/// Pass 3: Fuses activation functions (SiLU, GELU, ReLU) and scale factors directly
/// into accumulation register writeback loops, avoiding DRAM round-trips.
pub struct EpilogueFusionPass {
    pub fused_epilogues: usize,
}

impl EpilogueFusionPass {
    pub fn new() -> Self {
        EpilogueFusionPass {
            fused_epilogues: 0,
        }
    }

    pub fn run_fusion(&mut self, activation: ActivationKind) -> usize {
        if activation != ActivationKind::None {
            self.fused_epilogues += 1;
        }
        self.fused_epilogues
    }
}

// ────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epilogue_fusion_relu_gelu_silu() {
        let mut pass = QuantizationPass::new();
        let accs = vec!["%f0", "%f1", "%f2", "%f3"];

        let relu_code = pass.emit_epilogue_fusion(&accs, Some("%rd_bias"), ActivationKind::ReLU);
        assert!(relu_code.contains("inline ReLU"));
        assert!(relu_code.contains("max.f32 %f0, %f0, 0f00000000;"));

        let silu_code = pass.emit_epilogue_fusion(&accs, None, ActivationKind::SiLU);
        assert!(silu_code.contains("SiLU = x * sigmoid(x)"));

        let gelu_code = pass.emit_epilogue_fusion(&accs, None, ActivationKind::GELU);
        assert!(gelu_code.contains("tanh.approx.f32"));
    }
}


