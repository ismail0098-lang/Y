// ============================================================
//  Y  —  Dual-Pipeline Co-Processor Scheduler
//  coprocessor_scheduler.rs
//
//  The central battleground module. Resolves the two critical
//  constraints from the Y backend design spec:
//
//  1. ASYMMETRIC EXECUTION & SYNCHRONIZATION
//     RT Core traversals are async & non-deterministic.
//     Tensor Core mma.sync is structured & synchronous.
//     This scheduler partitions data deps via intermediate
//     registers and shared memory to prevent SM stalling.
//
//  2. PRECISION & FRAGMENT TRANSFORMATION
//     RT Cores output FP32. Tensor Cores expect FP16/BF16/FP4
//     packed fragments. The scheduler inserts automated
//     quantization passes (half2 packing) into the register
//     pipeline.
//
//  The scheduler consumes the IrGraph from ir_grapher and
//  produces a fused PTX emission plan that overlaps RT and
//  Tensor Core execution within a single SM.
// ============================================================

#![allow(dead_code)]

use crate::ir_grapher::*;
use crate::rt_core_emitter::RtCoreEmitter;
use crate::quantization_pass::QuantizationPass;
use crate::sentinel::HardwareProfile;
use std::fmt::Write;

/// A scheduled execution slot for the dual pipeline.
#[derive(Debug, Clone)]
pub struct ScheduleSlot {
    pub node_id: NodeId,
    /// Which cycle this slot begins executing.
    pub start_cycle: f64,
    /// Which cycle this slot finishes.
    pub end_cycle: f64,
    pub pipeline: Pipeline,
}

/// A synchronization point injected between pipelines.
#[derive(Debug, Clone)]
pub struct SyncBarrier {
    /// Cycle at which the barrier fires.
    pub cycle: f64,
    /// Which node triggers the sync.
    pub trigger_node: NodeId,
    /// Shared memory address range for data handoff.
    pub smem_offset: u32,
    pub smem_bytes: u32,
    /// Whether a quantization pass is needed at this barrier.
    pub needs_quantization: bool,
    /// Source precision (from RT Core output).
    pub src_precision: Precision,
    /// Target precision (for Tensor Core input).
    pub dst_precision: Precision,
}

/// The complete co-processing schedule.
#[derive(Debug, Clone)]
pub struct CoprocessorSchedule {
    pub rt_slots: Vec<ScheduleSlot>,
    pub tensor_slots: Vec<ScheduleSlot>,
    pub scalar_slots: Vec<ScheduleSlot>,
    pub sync_barriers: Vec<SyncBarrier>,
    pub total_smem_bytes: u32,
    pub estimated_total_cycles: f64,
    /// How many cycles were saved by overlapping RT+Tensor.
    pub overlap_savings_cycles: f64,
}

/// The scheduler that produces overlapping RT+Tensor execution plans.
pub struct CoprocessorScheduler {
    pub schedule: CoprocessorSchedule,
}

impl CoprocessorScheduler {
    pub fn new() -> Self {
        Self {
            schedule: CoprocessorSchedule {
                rt_slots: Vec::new(),
                tensor_slots: Vec::new(),
                scalar_slots: Vec::new(),
                sync_barriers: Vec::new(),
                total_smem_bytes: 0,
                estimated_total_cycles: 0.0,
                overlap_savings_cycles: 0.0,
            },
        }
    }

    /// Build the co-processing schedule from the IR dependency graph.
    pub fn schedule(&mut self, graph: &IrGraph, hw: &HardwareProfile) {
        let mut rt_cursor: f64 = 0.0;
        let mut tensor_cursor: f64 = 0.0;
        let mut scalar_cursor: f64 = 0.0;
        let mut smem_offset: u32 = 0;

        // Phase 1: Assign slots to each node based on pipeline
        for node in &graph.nodes {
            match node.pipeline {
                Pipeline::RtCore => {
                    // RT ops are async — they can overlap with tensor ops
                    let start = rt_cursor;
                    let end = start + node.estimated_cycles;
                    self.schedule.rt_slots.push(ScheduleSlot {
                        node_id: node.id,
                        start_cycle: start,
                        end_cycle: end,
                        pipeline: Pipeline::RtCore,
                    });
                    rt_cursor = end;
                }
                Pipeline::TensorCore => {
                    let start = tensor_cursor;
                    let end = start + node.estimated_cycles;
                    self.schedule.tensor_slots.push(ScheduleSlot {
                        node_id: node.id,
                        start_cycle: start,
                        end_cycle: end,
                        pipeline: Pipeline::TensorCore,
                    });
                    tensor_cursor = end;
                }
                Pipeline::ScalarAlu => {
                    let start = scalar_cursor;
                    let end = start + node.estimated_cycles;
                    self.schedule.scalar_slots.push(ScheduleSlot {
                        node_id: node.id,
                        start_cycle: start,
                        end_cycle: end,
                        pipeline: Pipeline::ScalarAlu,
                    });
                    scalar_cursor = end;
                }
                Pipeline::SyncPoint => {
                    // Force all pipelines to align
                    let max_cursor = rt_cursor.max(tensor_cursor).max(scalar_cursor);
                    rt_cursor = max_cursor + hw.bar_sync_latency_cycles;
                    tensor_cursor = max_cursor + hw.bar_sync_latency_cycles;
                    scalar_cursor = max_cursor + hw.bar_sync_latency_cycles;
                }
            }
        }

        // Phase 2: Insert sync barriers at pipeline-crossing edges
        let mut scheduled_producers = std::collections::HashSet::new();
        for edge in graph.cross_pipeline_edges() {
            let from_node = &graph.nodes[edge.from];
            let to_node = &graph.nodes[edge.to];

            if !scheduled_producers.insert(edge.from) {
                continue;
            }

            let needs_quant = from_node.pipeline == Pipeline::RtCore
                && to_node.pipeline == Pipeline::TensorCore
                && from_node.output_precision != Precision::FP16;

            let barrier = SyncBarrier {
                cycle: self.find_slot_end(edge.from),
                trigger_node: edge.from,
                smem_offset: smem_offset,
                smem_bytes: edge.transfer_bytes.max(from_node.smem_bytes),
                needs_quantization: needs_quant,
                src_precision: from_node.output_precision,
                dst_precision: if to_node.pipeline == Pipeline::TensorCore {
                    Precision::FP16
                } else {
                    Precision::FP32
                },
            };
            smem_offset += barrier.smem_bytes;
            self.schedule.sync_barriers.push(barrier);
        }

        self.schedule.total_smem_bytes = smem_offset + graph.total_smem_bytes();

        // Phase 3: Calculate overlap savings
        let sequential_total = rt_cursor + tensor_cursor + scalar_cursor;
        let parallel_total = rt_cursor
            .max(tensor_cursor)
            .max(scalar_cursor)
            + (self.schedule.sync_barriers.len() as f64 * hw.bar_sync_latency_cycles);

        self.schedule.estimated_total_cycles = parallel_total;
        self.schedule.overlap_savings_cycles = sequential_total - parallel_total;
    }

    fn find_slot_end(&self, node_id: NodeId) -> f64 {
        for slot in &self.schedule.rt_slots {
            if slot.node_id == node_id {
                return slot.end_cycle;
            }
        }
        for slot in &self.schedule.tensor_slots {
            if slot.node_id == node_id {
                return slot.end_cycle;
            }
        }
        for slot in &self.schedule.scalar_slots {
            if slot.node_id == node_id {
                return slot.end_cycle;
            }
        }
        0.0
    }

    /// Emit the complete fused PTX for the co-processing pipeline.
    pub fn emit_fused_ptx(
        &self,
        graph: &IrGraph,
        hw: &HardwareProfile,
    ) -> String {
        let mut out = String::new();

        writeln!(&mut out, "    // +----------------------------------------------------------+").unwrap();
        writeln!(&mut out, "    // |  Y DUAL-ACCELERATOR CO-PROCESSING SCHEDULE              |").unwrap();
        writeln!(&mut out, "    // |  RT Core Pipeline:     {} nodes                          |", self.schedule.rt_slots.len()).unwrap();
        writeln!(&mut out, "    // |  Tensor Core Pipeline: {} nodes                          |", self.schedule.tensor_slots.len()).unwrap();
        writeln!(&mut out, "    // |  Sync Barriers:        {}                                |", self.schedule.sync_barriers.len()).unwrap();
        writeln!(&mut out, "    // |  SMEM Budget:          {} bytes                          |", self.schedule.total_smem_bytes).unwrap();
        writeln!(&mut out, "    // |  Est. Parallel Cycles: {:.0}                             |", self.schedule.estimated_total_cycles).unwrap();
        writeln!(&mut out, "    // |  Overlap Savings:      {:.0} cycles                      |", self.schedule.overlap_savings_cycles).unwrap();
        writeln!(&mut out, "    // +----------------------------------------------------------+").unwrap();
        writeln!(&mut out).unwrap();

        // Allocate shared memory for cross-pipeline data transfer
        if self.schedule.total_smem_bytes > 0 {
            writeln!(
                &mut out,
                "    .shared .align 128 .b8 coprocessor_smem[{}];",
                self.schedule.total_smem_bytes
            )
            .unwrap();
            writeln!(&mut out).unwrap();
        }

        // Emit RT Core pipeline
        if !self.schedule.rt_slots.is_empty() {
            let rt_nodes: Vec<&IrNode> = self
                .schedule
                .rt_slots
                .iter()
                .filter_map(|s| graph.nodes.get(s.node_id))
                .collect();

            let mut rt_emitter = RtCoreEmitter::new();
            let rt_code = rt_emitter.emit_rt_pipeline(&rt_nodes, &self.schedule.sync_barriers, hw);
            out.push_str(&rt_code);
            writeln!(&mut out).unwrap();
        }

        // Emit sync barriers with quantization passes
        let mut quant = QuantizationPass::new();
        for (i, barrier) in self.schedule.sync_barriers.iter().enumerate() {
            writeln!(&mut out, "    // -- CROSS-PIPELINE SYNC BARRIER {} --", i).unwrap();
            writeln!(
                &mut out,
                "    // RT -> Tensor handoff at cycle {:.0}, {} bytes via SMEM[{}..{}]",
                barrier.cycle,
                barrier.smem_bytes,
                barrier.smem_offset,
                barrier.smem_offset + barrier.smem_bytes
            )
            .unwrap();

            // Emit barrier
            writeln!(&mut out, "    bar.sync 0;").unwrap();

            // Emit quantization pass if needed
            if barrier.needs_quantization {
                writeln!(
                    &mut out,
                    "    // QUANTIZATION PASS: {:?} -> {:?}",
                    barrier.src_precision, barrier.dst_precision
                )
                .unwrap();

                let quant_code = quant.emit_vectorized_quantization(
                    barrier.src_precision,
                    barrier.dst_precision,
                    barrier.smem_offset,
                    barrier.smem_bytes,
                    hw,
                );
                out.push_str(&quant_code);
            }

            writeln!(&mut out, "    bar.sync 0;  // Post-quantization fence").unwrap();
            writeln!(&mut out).unwrap();
        }

        // Emit Tensor Core pipeline
        if !self.schedule.tensor_slots.is_empty() {
            writeln!(&mut out, "    // ===================================================").unwrap();
            writeln!(&mut out, "    // TENSOR CORE CO-PROCESSING PIPELINE  ({} nodes)", self.schedule.tensor_slots.len()).unwrap();
            writeln!(&mut out, "    // Consuming RT Core outputs via quantized SMEM fragments").unwrap();
            writeln!(&mut out, "    // ===================================================").unwrap();

            // Track SMEM read offset for successive ldmatrix loads
            let mut smem_load_offset: u32 = 0;
            if let Some(barrier) = self.schedule.sync_barriers.first() {
                if barrier.needs_quantization {
                    smem_load_offset = barrier.smem_offset + barrier.smem_bytes;
                }
            }
            let mut first_mma = true;
            let first_barrier = self.schedule.sync_barriers.first();

            for slot in &self.schedule.tensor_slots {
                if let Some(node) = graph.nodes.get(slot.node_id) {
                    if let Some(ref mapping) = node.tensor_mapping {
                        match mapping {
                            TensorCoreMapping::MmaSync {
                                m,
                                n,
                                k,
                                input_precision,
                                accumulator_precision,
                            } => {
                                self.emit_mma_sync(
                                    &mut out,
                                    *m,
                                    *n,
                                    *k,
                                    *input_precision,
                                    *accumulator_precision,
                                    hw,
                                    &mut smem_load_offset,
                                    &mut first_mma,
                                    first_barrier,
                                );
                            }
                            TensorCoreMapping::QuantizedGemm {
                                m,
                                n,
                                k,
                                quant_bits,
                            } => {
                                writeln!(
                                    &mut out,
                                    "    // Quantized GEMM: {}x{}x{} @ {} bits",
                                    m, n, k, quant_bits
                                )
                                .unwrap();
                                self.emit_mma_sync(
                                    &mut out,
                                    *m,
                                    *n,
                                    *k,
                                    Precision::FP16,
                                    Precision::FP32,
                                    hw,
                                    &mut smem_load_offset,
                                    &mut first_mma,
                                    first_barrier,
                                );
                            }
                        }
                    } else {
                        writeln!(
                            &mut out,
                            "    // Tensor node [{}] '{}': generic mma.sync",
                            node.id, node.label
                        )
                        .unwrap();
                        self.emit_mma_sync(
                            &mut out,
                            16, 8, 16,
                            Precision::FP16,
                            Precision::FP32,
                            hw,
                            &mut smem_load_offset,
                            &mut first_mma,
                            first_barrier,
                        );
                    }
                }
            }
        }

        out
    }

    fn emit_mma_sync(
        &self,
        out: &mut String,
        m: u32,
        n: u32,
        k: u32,
        input_prec: Precision,
        acc_prec: Precision,
        hw: &HardwareProfile,
        smem_offset: &mut u32,
        first_mma: &mut bool,
        barrier: Option<&SyncBarrier>,
    ) {
        let (in_type, acc_type) = match (input_prec, acc_prec) {
            (Precision::FP16, Precision::FP32) => ("f16", "f32"),
            (Precision::BF16, Precision::FP32) => ("bf16", "f32"),
            (Precision::TF32, Precision::FP32) => ("tf32", "f32"),
            (Precision::FP16, Precision::FP16) => ("f16", "f16"),
            _ => ("f16", "f32"),
        };

        let latency = match input_prec {
            Precision::FP16 => hw.hmma_f16_latency_cycles,
            Precision::TF32 => hw.tf32_latency_cycles,
            _ => hw.hmma_f16_latency_cycles,
        };

        // Zero-initialize accumulator on first MMA
        if *first_mma {
            writeln!(out, "    // Zero-initialize accumulator registers").unwrap();
            writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap();
            writeln!(out, "    mov.f32 %f1, 0f00000000;").unwrap();
            writeln!(out, "    mov.f32 %f2, 0f00000000;").unwrap();
            writeln!(out, "    mov.f32 %f3, 0f00000000;").unwrap();
            *first_mma = false;
        }

        // Compute byte offsets for matrix A and B in quantized SMEM
        // Matrix A: m*k FP16 values = m*k*2 bytes, packed into .x4 ldmatrix (4 regs)
        let a_bytes = m * k * 2;  // FP16 = 2 bytes each
        // Matrix B: k*n FP16 values = k*n*2 bytes, packed into .x2 ldmatrix (2 regs)
        let b_bytes = k * n * 2;

        let mut a_smem_offset = *smem_offset;
        let mut b_smem_offset = a_smem_offset + a_bytes;

        // Wrap offsets if they exceed the quantized buffer bounds
        if let Some(b) = barrier {
            if b.needs_quantization {
                let start = b.smem_offset + b.smem_bytes;
                let size = b.smem_bytes / 2;
                if size > 0 {
                    let a_rel = (a_smem_offset - start) % size;
                    a_smem_offset = start + a_rel;
                    let b_rel = (b_smem_offset - start) % size;
                    b_smem_offset = start + b_rel;
                }
            }
        }

        writeln!(out, "    // ldmatrix: load A fragment from quantized SMEM[{}..{}]",
            a_smem_offset, a_smem_offset + a_bytes).unwrap();
        writeln!(out, "    {{").unwrap();
        writeln!(out, "        .reg .b64 %mma_addr;").unwrap();
        writeln!(out, "        mov.u64 %mma_addr, coprocessor_smem;").unwrap();
        if a_smem_offset > 0 {
            writeln!(out, "        add.u64 %mma_addr, %mma_addr, {};", a_smem_offset).unwrap();
        }
        // Use ldmatrix.sync.aligned.x4 for matrix A (loads 4x .b32 regs = 8 FP16 per thread)
        writeln!(out, "        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0,%r1,%r2,%r3}}, [%mma_addr];").unwrap();
        writeln!(out, "    }}").unwrap();

        writeln!(out, "    // ldmatrix: load B fragment from quantized SMEM[{}..{}]",
            b_smem_offset, b_smem_offset + b_bytes).unwrap();
        writeln!(out, "    {{").unwrap();
        writeln!(out, "        .reg .b64 %mma_addr_b;").unwrap();
        writeln!(out, "        mov.u64 %mma_addr_b, coprocessor_smem;").unwrap();
        if b_smem_offset > 0 {
            writeln!(out, "        add.u64 %mma_addr_b, %mma_addr_b, {};", b_smem_offset).unwrap();
        }
        writeln!(out, "        ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%r4,%r5}}, [%mma_addr_b];").unwrap();
        writeln!(out, "    }}").unwrap();

        // Advance SMEM offset for next MMA tile
        *smem_offset += a_bytes + b_bytes;

        writeln!(out,
            "    // mma.sync.aligned.m{}n{}k{}.row.col.{}.{}.{}.{} - {:.0} cy on {}",
            m, n, k, acc_type, in_type, in_type, acc_type, latency, hw.gpu_name
        ).unwrap();
        writeln!(out,
            "    mma.sync.aligned.m{}n{}k{}.row.col.{}.{}.{}.{} {{%f0,%f1,%f2,%f3}}, {{%r0,%r1,%r2,%r3}}, {{%r4,%r5}}, {{%f0,%f1,%f2,%f3}};",
            m, n, k, acc_type, in_type, in_type, acc_type
        ).unwrap();
    }
}
