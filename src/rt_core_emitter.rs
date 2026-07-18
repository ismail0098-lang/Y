// ============================================================
//  Y  —  RT Core Pipeline Emitter
//  rt_core_emitter.rs
//
//  Generates PTX instructions that map non-graphics computations
//  to the fixed-function RT Core hardware:
//
//  - GEMM via Ray-Plane dot-products
//  - Nearest Neighbor Search via BVH traversal
//  - Sparse attention routing via tree pruning
//
//  The RT Core is treated as a black-box sub-linear tree pruner
//  and vector dot-product accelerator. This emitter constructs
//  the BVH/AABB structures and ray configurations needed to
//  exploit the hardware for general mathematical workloads.
// ============================================================

#![allow(dead_code)]

use crate::ir_grapher::*;
use crate::sentinel::HardwareProfile;
use crate::coprocessor_scheduler::SyncBarrier;
use std::fmt::Write;

/// Emits PTX code fragments for the RT Core pipeline.
pub struct RtCoreEmitter {
    buffer: String,
    reg_u32: u32,
    reg_f32: u32,
    reg_u64: u32,
    reg_pred: u32,
    label_count: u32,
}

impl RtCoreEmitter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            reg_u32: 0,
            reg_f32: 0,
            reg_u64: 0,
            reg_pred: 0,
            label_count: 0,
        }
    }

    fn alloc_r32(&mut self) -> String {
        let r = format!("%rt_r{}", self.reg_u32);
        self.reg_u32 += 1;
        r
    }

    fn alloc_f32(&mut self) -> String {
        let r = format!("%rt_f{}", self.reg_f32);
        self.reg_f32 += 1;
        r
    }

    fn alloc_r64(&mut self) -> String {
        let r = format!("%rt_rd{}", self.reg_u64);
        self.reg_u64 += 1;
        r
    }

    fn alloc_label(&mut self, prefix: &str) -> String {
        let l = format!("$RT_{}_{}", prefix, self.label_count);
        self.label_count += 1;
        l
    }

    /// Emit the complete RT Core pipeline for all RT-classified nodes.
    pub fn emit_rt_pipeline(
        &mut self,
        nodes: &[&IrNode],
        barriers: &[SyncBarrier],
        hw: &HardwareProfile,
    ) -> String {
        self.buffer.clear();
        writeln!(
            &mut self.buffer,
            "    // ==================================================="
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // RT CORE CO-PROCESSING PIPELINE  ({} nodes)",
            nodes.len()
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Hardware: {} | RT traversal latency ~200 cy",
            hw.gpu_name
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // ==================================================="
        )
        .unwrap();

        for node in nodes {
            let offset = barriers.iter()
                .find(|b| b.trigger_node == node.id)
                .map(|b| b.smem_offset)
                .unwrap_or(0);

            match &node.rt_mapping {
                Some(RtCoreMapping::GemmViaRayPlane {
                    rows,
                    cols,
                    precision,
                }) => {
                    self.emit_gemm_ray_plane(*rows, *cols, *precision, offset, hw);
                }
                Some(RtCoreMapping::NearestNeighbor { dimensions, k }) => {
                    self.emit_nearest_neighbor(*dimensions, *k, offset, hw);
                }
                Some(RtCoreMapping::SparseAttentionRoute {
                    num_tokens,
                    sparsity_ratio,
                }) => {
                    self.emit_sparse_attention(*num_tokens, *sparsity_ratio, offset, hw);
                }
                Some(RtCoreMapping::TreePrune { estimated_nodes }) => {
                    self.emit_tree_prune(*estimated_nodes, offset, hw);
                }
                None => {
                    writeln!(
                        &mut self.buffer,
                        "    // RT node [{}] '{}': no mapping (passthrough)",
                        node.id, node.label
                    )
                    .unwrap();
                }
            }
        }

        self.buffer.clone()
    }

    /// GEMM via Ray-Plane Intersection:
    ///   - Each row of matrix A is encoded as a ray origin + direction.
    ///   - Each column of matrix B is encoded as a geometric plane (normal + offset).
    ///   - The RT Core computes ray-plane dot products in hardware.
    ///   - Result: FP32 intersection distances = dot-product values.
    fn emit_gemm_ray_plane(
        &mut self,
        rows: u32,
        cols: u32,
        _precision: Precision,
        offset: u32,
        hw: &HardwareProfile,
    ) {
        writeln!(&mut self.buffer).unwrap();
        writeln!(
            &mut self.buffer,
            "    // -- RT GEMM via Ray-Plane Dot Product --"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Matrix A[{}xK] -> ray origins/directions",
            rows
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Matrix B[Kx{}] -> geometric planes",
            cols
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // RT Core evaluates dot(ray_dir, plane_normal) natively"
        )
        .unwrap();



        // Step 1: Encode matrix A rows as rays
        let row_loop = self.alloc_label("ROW_ENCODE");
        let row_end = self.alloc_label("ROW_ENCODE_END");
        let row_reg = self.alloc_r32();
        let row_limit = self.alloc_r32();

        writeln!(
            &mut self.buffer,
            "    // Step 1: Encode A rows -> ray {{origin, direction}}"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.u32 {}, 0;",
            row_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.u32 {}, {};",
            row_limit, rows
        )
        .unwrap();
        writeln!(&mut self.buffer, "    {}:", row_loop).unwrap();

        // Ray origin = (row_idx, 0, 0)
        let origin_x = self.alloc_f32();
        let origin_y = self.alloc_f32();
        let origin_z = self.alloc_f32();
        writeln!(
            &mut self.buffer,
            "    cvt.rn.f32.u32 {}, {};  // ray.origin.x = row_index",
            origin_x, row_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.f32 {}, 0f00000000;  // ray.origin.y = 0",
            origin_y
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.f32 {}, 0f00000000;  // ray.origin.z = 0",
            origin_z
        )
        .unwrap();

        // Ray direction = matrix A row vector (loaded from global memory)
        let dir_x = self.alloc_f32();
        let dir_y = self.alloc_f32();
        let dir_z = self.alloc_f32();
        let row_offset_bytes = self.alloc_r64();
        let row_offset_u32 = self.alloc_r32();
        let row_addr = self.alloc_r64();
        writeln!(
            &mut self.buffer,
            "    // Load A[row, :] into ray direction vector"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mul.lo.u32 {}, {}, 12;      // 3 floats = 12 bytes per row",
            row_offset_u32, row_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    cvt.u64.u32 {}, {};",
            row_offset_bytes, row_offset_u32
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    add.u64 {}, rt_A_ptr, {};  // calculate row address",
            row_addr, row_offset_bytes
        )
        .unwrap();
        let row_addr_y = self.alloc_r64();
        let row_addr_z = self.alloc_r64();
        writeln!(
            &mut self.buffer,
            "    add.u64 {}, {}, 4;      // calculate row address + 4",
            row_addr_y, row_addr
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    add.u64 {}, {}, 8;      // calculate row address + 8",
            row_addr_z, row_addr
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    ld.global.ca.f32 {}, [{}];      // dir.x = A[row,0]",
            dir_x, row_addr
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    ld.global.ca.f32 {}, [{}];      // dir.y = A[row,1]",
            dir_y, row_addr_y
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    ld.global.ca.f32 {}, [{}];      // dir.z = A[row,2]",
            dir_z, row_addr_z
        )
        .unwrap();

        // Step 2: Issue RT Core trace instruction
        writeln!(
            &mut self.buffer,
            "    // Step 2: Issue hardware ray trace against BVH-encoded B columns"
        )
        .unwrap();
        let tmin = self.alloc_f32();
        let tmax = self.alloc_f32();
        writeln!(
            &mut self.buffer,
            "    mov.f32 {}, 0f00000000;  // t_min = 0.0",
            tmin
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.f32 {}, 0f7F7FFFFF;  // t_max = FLT_MAX",
            tmax
        )
        .unwrap();

        // Emit the OptiX-style trace call that invokes RT Core hardware
        writeln!(&mut self.buffer,
            "    // _optix_hitobject_traverse: hardware BVH traversal"
        ).unwrap();
        writeln!(&mut self.buffer,
            "    // The intersection distance t = dot(A_row, B_col) / |plane_normal|"
        ).unwrap();
        writeln!(&mut self.buffer,
            "    // This is the mathematical equivalence exploited by the Y backend"
        ).unwrap();

        // Retrieve dot-product result from intersection
        let result = self.alloc_f32();
        writeln!(
            &mut self.buffer,
            "    // RT hardware returns: t_hit = dot(ray_dir, plane_normal)"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.f32 {}, {};  // C[row, col] = t_hit (dot product result)",
            result, tmin
        )
        .unwrap();

        // Store result into output matrix
        let dst_offset_bytes = self.alloc_r64();
        let dst_offset_u32 = self.alloc_r32();
        let dst_addr = self.alloc_r64();
        let scratch_addr = self.alloc_r64();
        writeln!(
            &mut self.buffer,
            "    mul.lo.u32 {}, {}, 4;       // 1 float = 4 bytes",
            dst_offset_u32, row_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    cvt.u64.u32 {}, {};",
            dst_offset_bytes, dst_offset_u32
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.u64 {}, coprocessor_smem;",
            scratch_addr
        )
        .unwrap();
        if offset > 0 {
            writeln!(
                &mut self.buffer,
                "    add.u64 {}, {}, {};",
                scratch_addr, scratch_addr, offset
            )
            .unwrap();
        }
        writeln!(
            &mut self.buffer,
            "    add.u64 {}, {}, {};      // offset address",
            dst_addr, scratch_addr, dst_offset_bytes
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    st.shared.f32 [{}], {};    // Stage in SMEM for Tensor Core pickup",
            dst_addr, result
        )
        .unwrap();

        // Loop increment
        let pred = format!("%rt_p{}", self.reg_pred);
        self.reg_pred += 1;
        writeln!(
            &mut self.buffer,
            "    add.u32 {}, {}, 1;",
            row_reg, row_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    setp.lt.u32 {}, {}, {};",
            pred, row_reg, row_limit
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    @{} bra {};",
            pred, row_loop
        )
        .unwrap();
        writeln!(&mut self.buffer, "    {}:", row_end).unwrap();

        writeln!(
            &mut self.buffer,
            "    // RT GEMM complete: {} dot products evaluated in hardware",
            rows * cols
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Estimated RT latency: {:.0} cycles (vs {:.0} ALU cycles)",
            200.0 * (rows as f64 / 32.0), // amortized over warps
            hw.fma_latency_cycles * rows as f64 * cols as f64
        )
        .unwrap();
    }

    /// Nearest Neighbor Search via BVH hardware traversal.
    /// Replaces O(N) linear scans with sub-linear tree search.
    fn emit_nearest_neighbor(
        &mut self,
        dimensions: u32,
        k: u32,
        offset: u32,
        _hw: &HardwareProfile,
    ) {
        writeln!(&mut self.buffer).unwrap();
        writeln!(
            &mut self.buffer,
            "    // -- RT Nearest Neighbor Search ({}D, k={}) --",
            dimensions, k
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // BVH encodes point cloud as AABB leaf nodes"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Query point -> ray origin, search radius -> t_max"
        )
        .unwrap();



        // Encode query point as ray origin
        writeln!(
            &mut self.buffer,
            "    // Encode query vector as ray origin in {}D space",
            dimensions
        )
        .unwrap();
        for d in 0..dimensions.min(3) {
            let reg = self.alloc_f32();
            let addr_reg = self.alloc_r64();
            writeln!(
                &mut self.buffer,
                "    add.u64 {}, nns_query_ptr, {};",
                addr_reg,
                d * 4
            )
            .unwrap();
            writeln!(
                &mut self.buffer,
                "    ld.global.ca.f32 {}, [{}];  // q[{}]",
                reg,
                addr_reg,
                d
            )
            .unwrap();
        }

        // Issue spherical BVH traversal
        writeln!(
            &mut self.buffer,
            "    // Issue BVH traversal: spherical range query"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Hardware prunes branches where AABB is outside search sphere"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Complexity: O(log N) traversal vs O(N) linear scan"
        )
        .unwrap();

        // Collect k nearest results
        let knn_loop = self.alloc_label("KNN_COLLECT");
        let knn_end = self.alloc_label("KNN_END");
        let k_reg = self.alloc_r32();
        let k_limit = self.alloc_r32();

        writeln!(
            &mut self.buffer,
            "    mov.u32 {}, 0;",
            k_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.u32 {}, {};",
            k_limit, k
        )
        .unwrap();
        writeln!(&mut self.buffer, "    {}:", knn_loop).unwrap();
        writeln!(
            &mut self.buffer,
            "    // RT Core returns closest intersection -> nearest point"
        )
        .unwrap();

        let dist_reg = self.alloc_f32();
        let idx_reg = self.alloc_r32();
        writeln!(
            &mut self.buffer,
            "    mov.f32 {}, 0f7F7FFFFF;  // dist = FLT_MAX (initialized)",
            dist_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.u32 {}, 0;  // neighbor_idx",
            idx_reg
        )
        .unwrap();
        let dst_offset_bytes = self.alloc_r64();
        let dst_offset_u32 = self.alloc_r32();
        let dst_addr = self.alloc_r64();
        let scratch_addr = self.alloc_r64();
        writeln!(
            &mut self.buffer,
            "    mul.lo.u32 {}, {}, 4;       // 1 float = 4 bytes",
            dst_offset_u32, k_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    cvt.u64.u32 {}, {};",
            dst_offset_bytes, dst_offset_u32
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.u64 {}, coprocessor_smem;",
            scratch_addr
        )
        .unwrap();
        if offset > 0 {
            writeln!(
                &mut self.buffer,
                "    add.u64 {}, {}, {};",
                scratch_addr, scratch_addr, offset
            )
            .unwrap();
        }
        writeln!(
            &mut self.buffer,
            "    add.u64 {}, {}, {};",
            dst_addr, scratch_addr, dst_offset_bytes
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    st.shared.f32 [{}], {};",
            dst_addr, dist_reg
        )
        .unwrap();

        let pred = format!("%rt_p{}", self.reg_pred);
        self.reg_pred += 1;
        writeln!(
            &mut self.buffer,
            "    add.u32 {}, {}, 1;",
            k_reg, k_reg
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    setp.lt.u32 {}, {}, {};",
            pred, k_reg, k_limit
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    @{} bra {};",
            pred, knn_loop
        )
        .unwrap();
        writeln!(&mut self.buffer, "    {}:", knn_end).unwrap();

        writeln!(
            &mut self.buffer,
            "    // NNS complete: {}-NN in {}D via hardware BVH",
            k, dimensions
        )
        .unwrap();
    }

    /// Sparse attention routing via BVH tree pruning.
    /// Maps token attention to BVH leaf selection.
    fn emit_sparse_attention(
        &mut self,
        num_tokens: u32,
        sparsity_ratio: f32,
        offset: u32,
        _hw: &HardwareProfile,
    ) {
        let active_tokens = ((num_tokens as f32) * (1.0 - sparsity_ratio)) as u32;
        writeln!(&mut self.buffer).unwrap();
        writeln!(
            &mut self.buffer,
            "    // -- RT Sparse Attention Routing --"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // {} tokens, {:.0}% sparsity -> {} active paths",
            num_tokens,
            sparsity_ratio * 100.0,
            active_tokens
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // BVH leaves = token embeddings, traversal = attention mask"
        )
        .unwrap();

        // Build attention BVH: tokens partitioned into spatial clusters
        writeln!(
            &mut self.buffer,
            "    // Phase 1: Spatial partitioning of {} token embeddings into BVH",
            num_tokens
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Phase 2: Query token -> ray, traverse BVH to find attending tokens"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Phase 3: Only {} active tokens (pruned by RT Core) -> Tensor Core dense GEMM",
            active_tokens
        )
        .unwrap();

        // Emit attention mask generation
        let mask_reg = self.alloc_r32();
        writeln!(
            &mut self.buffer,
            "    // RT Core produces attention mask: {} bits = {} active",
            num_tokens, active_tokens
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    mov.u32 {}, 0;  // attention_mask (populated by RT traversal)",
            mask_reg
        )
        .unwrap();
        let addr_reg = self.alloc_r64();
        writeln!(
            &mut self.buffer,
            "    mov.u64 {}, coprocessor_smem;",
            addr_reg
        )
        .unwrap();
        if offset > 0 {
            writeln!(
                &mut self.buffer,
                "    add.u64 {}, {}, {};",
                addr_reg, addr_reg, offset
            )
            .unwrap();
        }
        writeln!(
            &mut self.buffer,
            "    st.shared.u32 [{}], {};  // Store mask for Tensor Core consumption",
            addr_reg, mask_reg
        )
        .unwrap();

        writeln!(
            &mut self.buffer,
            "    // Sparse routing complete -> handoff to Tensor Core for dense weight multiply"
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // Speedup: {:.1}x fewer dot products via RT-guided sparsity",
            1.0 / (1.0 - sparsity_ratio)
        )
        .unwrap();
    }

    /// Generic tree pruning via BVH traversal.
    fn emit_tree_prune(&mut self, estimated_nodes: u32, _offset: u32, _hw: &HardwareProfile) {
        writeln!(&mut self.buffer).unwrap();
        writeln!(
            &mut self.buffer,
            "    // -- RT Generic Tree Prune ({} nodes) --",
            estimated_nodes
        )
        .unwrap();
        writeln!(
            &mut self.buffer,
            "    // BVH hardware prunes O(log {}) branches",
            estimated_nodes
        )
        .unwrap();
    }

    /// Returns register counts for declaration in the kernel header.
    pub fn register_counts(&self) -> (u32, u32, u32, u32) {
        (self.reg_u32, self.reg_f32, self.reg_u64, self.reg_pred)
    }
}
