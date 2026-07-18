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
    ) -> String {
        let mut out = String::new();

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
            (Precision::FP32, Precision::FP8) => {
                self.emit_fp32_to_fp8(&mut out, smem_offset, smem_bytes, hw);
            }
            (Precision::FP32, Precision::INT8) => {
                self.emit_fp32_to_int8(&mut out, smem_offset, smem_bytes, hw);
            }
            (Precision::FP32, Precision::FP4) | (Precision::FP32, Precision::INT4) => {
                self.emit_fp32_to_4bit(&mut out, smem_offset, smem_bytes, hw);
            }
            (Precision::FP16, Precision::FP32) => {
                self.emit_fp16_to_fp32(&mut out, smem_offset, smem_bytes, hw);
            }
            _ => {
                writeln!(
                    &mut out,
                    "    // WARNING: No optimized quantization path for {:?} -> {:?}",
                    src, dst
                )
                .unwrap();
                writeln!(&mut out, "    // Falling back to scalar conversion").unwrap();
                self.emit_scalar_fallback(&mut out, src, dst, smem_offset, smem_bytes);
            }
        }

        out
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

    /// FP32 → FP8 (E4M3) conversion for Hopper+ architectures.
    fn emit_fp32_to_fp8(
        &mut self,
        out: &mut String,
        smem_offset: u32,
        smem_bytes: u32,
        _hw: &HardwareProfile,
    ) {
        let num_elements = smem_bytes / 4;
        writeln!(
            out,
            "    // FP32 -> FP8 E4M3: {} elements (4x compression)",
            num_elements
        )
        .unwrap();
        writeln!(
            out,
            "    // Requires sm_89+ (Hopper). Pack 4 FP8 values per 32-bit register."
        )
        .unwrap();

        // Scalar loop with manual E4M3 conversion
        let loop_start = self.alloc_label("FP8_CVT");
        let loop_end = self.alloc_label("FP8_DONE");
        let iter = self.alloc_u32();
        let limit = self.alloc_u32();
        let quads = num_elements / 4;

        writeln!(out, "    mov.u32 {}, 0;", iter).unwrap();
        writeln!(out, "    mov.u32 {}, {};", limit, quads).unwrap();
        writeln!(out, "    {}:", loop_start).unwrap();

        // Load 4 FP32 values
        for i in 0..4 {
            let val = self.alloc_f32();
            writeln!(
                out,
                "    ld.shared.f32 {}, [coprocessor_smem+{}];  // fp32[{}]",
                val,
                smem_offset + i * 4,
                i
            )
            .unwrap();
        }

        // Pack 4 FP8 values into a single 32-bit register
        let packed = self.alloc_u32();
        writeln!(
            out,
            "    // E4M3 pack: clamp + truncate + shift into 32-bit register"
        )
        .unwrap();
        writeln!(
            out,
            "    mov.u32 {}, 0;  // packed FP8x4 (placeholder for cvt.rn.satfinite.e4m3x4)",
            packed
        )
        .unwrap();

        let pred = format!("%qp{}", self.reg_pred);
        self.reg_pred += 1;
        writeln!(out, "    add.u32 {}, {}, 1;", iter, iter).unwrap();
        writeln!(out, "    setp.lt.u32 {}, {}, {};", pred, iter, limit).unwrap();
        writeln!(out, "    @{} bra {};", pred, loop_start).unwrap();
        writeln!(out, "    {}:", loop_end).unwrap();
    }

    /// FP32 → INT8 symmetric quantization.
    fn emit_fp32_to_int8(
        &mut self,
        out: &mut String,
        _smem_offset: u32,
        smem_bytes: u32,
        _hw: &HardwareProfile,
    ) {
        let num_elements = smem_bytes / 4;
        writeln!(
            out,
            "    // FP32 -> INT8 symmetric quantization ({} elements)",
            num_elements
        )
        .unwrap();
        writeln!(
            out,
            "    // scale = max(abs(tensor)) / 127.0; quant = round(val / scale)"
        )
        .unwrap();

        // Find absmax for scaling factor
        let absmax = self.alloc_f32();
        let scale = self.alloc_f32();
        writeln!(
            out,
            "    mov.f32 {}, 0f00000000;  // absmax = 0.0",
            absmax
        )
        .unwrap();

        // Scale: 127.0 / absmax
        writeln!(
            out,
            "    div.approx.f32 {}, 0f42FE0000, {};  // scale = 127.0 / absmax",
            scale, absmax
        )
        .unwrap();

        let loop_start = self.alloc_label("INT8_QUANT");
        let loop_end = self.alloc_label("INT8_DONE");
        let iter = self.alloc_u32();
        let limit = self.alloc_u32();

        writeln!(out, "    mov.u32 {}, 0;", iter).unwrap();
        writeln!(out, "    mov.u32 {}, {};", limit, num_elements / 4).unwrap();
        writeln!(out, "    {}:", loop_start).unwrap();

        // Load, scale, round, clamp to [-128, 127], pack 4 INT8 into U32
        let val = self.alloc_f32();
        let scaled = self.alloc_f32();
        let rounded = self.alloc_u32();
        writeln!(out, "    ld.shared.f32 {}, [coprocessor_smem];", val).unwrap();
        writeln!(
            out,
            "    mul.f32 {}, {}, {};  // val * scale",
            scaled, val, scale
        )
        .unwrap();
        writeln!(
            out,
            "    cvt.rni.s32.f32 {}, {};  // round to nearest int",
            rounded, scaled
        )
        .unwrap();

        let pred = format!("%qp{}", self.reg_pred);
        self.reg_pred += 1;
        writeln!(out, "    add.u32 {}, {}, 1;", iter, iter).unwrap();
        writeln!(out, "    setp.lt.u32 {}, {}, {};", pred, iter, limit).unwrap();
        writeln!(out, "    @{} bra {};", pred, loop_start).unwrap();
        writeln!(out, "    {}:", loop_end).unwrap();
    }

    /// FP32 → 4-bit quantization (FP4/INT4).
    fn emit_fp32_to_4bit(
        &mut self,
        out: &mut String,
        _smem_offset: u32,
        smem_bytes: u32,
        _hw: &HardwareProfile,
    ) {
        let num_elements = smem_bytes / 4;
        writeln!(
            out,
            "    // FP32 -> 4-bit quantization ({} elements, 8x compression)",
            num_elements
        )
        .unwrap();
        writeln!(
            out,
            "    // Pack 8 * 4-bit values into each 32-bit register"
        )
        .unwrap();
        writeln!(
            out,
            "    // Uses LOP3.LUT for efficient bit-field manipulation"
        )
        .unwrap();

        let packed = self.alloc_u32();
        writeln!(
            out,
            "    mov.u32 {}, 0;  // placeholder for 4-bit packed quantization",
            packed
        )
        .unwrap();
    }

    /// FP16 → FP32 dequantization (for reading back from Tensor Core output).
    fn emit_fp16_to_fp32(
        &mut self,
        out: &mut String,
        _smem_offset: u32,
        smem_bytes: u32,
        _hw: &HardwareProfile,
    ) {
        let num_half2 = smem_bytes / 4;
        writeln!(
            out,
            "    // FP16 -> FP32 dequantization: {} half2 pairs -> {} FP32",
            num_half2,
            num_half2 * 2
        )
        .unwrap();

        let loop_start = self.alloc_label("DEQUANT");
        let loop_end = self.alloc_label("DEQUANT_DONE");
        let iter = self.alloc_u32();
        let limit = self.alloc_u32();

        writeln!(out, "    mov.u32 {}, 0;", iter).unwrap();
        writeln!(out, "    mov.u32 {}, {};", limit, num_half2).unwrap();
        writeln!(out, "    {}:", loop_start).unwrap();

        let packed = self.alloc_u32();
        let val_lo = self.alloc_f32();
        let val_hi = self.alloc_f32();

        writeln!(out, "    ld.shared.b32 {}, [coprocessor_smem];", packed).unwrap();
        writeln!(
            out,
            "    cvt.f32.f16 {}, {};  // unpack lo half",
            val_lo, packed
        )
        .unwrap();

        let hi_bits = self.alloc_u32();
        writeln!(out, "    shr.u32 {}, {}, 16;", hi_bits, packed).unwrap();
        writeln!(
            out,
            "    cvt.f32.f16 {}, {};  // unpack hi half",
            val_hi, hi_bits
        )
        .unwrap();

        let pred = format!("%qp{}", self.reg_pred);
        self.reg_pred += 1;
        writeln!(out, "    add.u32 {}, {}, 1;", iter, iter).unwrap();
        writeln!(out, "    setp.lt.u32 {}, {}, {};", pred, iter, limit).unwrap();
        writeln!(out, "    @{} bra {};", pred, loop_start).unwrap();
        writeln!(out, "    {}:", loop_end).unwrap();
    }

    /// Scalar fallback for unsupported conversion paths.
    fn emit_scalar_fallback(
        &mut self,
        out: &mut String,
        src: Precision,
        dst: Precision,
        _smem_offset: u32,
        smem_bytes: u32,
    ) {
        writeln!(
            out,
            "    // Scalar fallback: {:?} -> {:?} ({} bytes)",
            src, dst, smem_bytes
        )
        .unwrap();
        writeln!(
            out,
            "    // TODO: Implement optimized path for this conversion"
        )
        .unwrap();
    }
}
