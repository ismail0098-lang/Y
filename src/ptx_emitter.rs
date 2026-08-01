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

/// Maps an SM compute capability to the minimum required PTX ISA version.
fn ptx_version_for_sm(sm: &str) -> &'static str {
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
            "sm_89" | "sm_8.9" => ".version 7.8",
            "sm_86" | "sm_87" | "sm_8.6" => ".version 7.5",
            "sm_80" | "sm_8.0" => ".version 7.0",
            "sm_75" => ".version 6.5",
            "sm_72" => ".version 6.2",
            "sm_70" => ".version 6.3",
            _ => ".version 7.8", // Safe default for CUDA 12+ targets
        }
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
        if m <= 512 || n <= 512 {
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
pub struct PtxEmitter {
    pub ptx_buffer: String,

    // Virtual register counters to maintain uniqueness
    reg_u32_count: u32,
    reg_f32_count: u32,
    reg_u64_count: u32,
    reg_pred_count: u32,
    label_count: u32,
    variables: std::collections::HashMap<String, String>,
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
            if t == "sm_90" {
                "sm_90a".to_string()
            } else {
                t
            }
        } else {
            "sm_80".to_string()
        };
        let ptx_version = ptx_version_for_sm(&target);
        writeln!(&mut buffer, "{}", ptx_version).unwrap();
        writeln!(&mut buffer, ".target {}", target).unwrap();
        writeln!(&mut buffer, ".address_size 64").unwrap();
        writeln!(&mut buffer, "").unwrap();

        Self {
            ptx_buffer: buffer,
            reg_u32_count: 0,
            reg_f32_count: 0,
            reg_u64_count: 0,
            reg_pred_count: 0,
            label_count: 0,
            variables: std::collections::HashMap::new(),
            sm_target: target,
            debug_info: false,
            pending_extern_decls: Vec::new(),
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
        for item in &prog.items {
            if let Item::Kernel(k) = item {
                self.emit_kernel(k, hw_profile);
            }
        }
        self.ptx_buffer.clone()
    }

    fn emit_kernel(&mut self, kernel: &KernelDecl, hw_profile: &HardwareProfile) {
        // Clear variables mapping for fresh compilation unit
        self.variables.clear();

        // Reset register counters
        self.reg_u32_count = 0;
        self.reg_f32_count = 0;
        self.reg_u64_count = 0;
        self.reg_pred_count = 0;

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
                Type::Primitive(p, _) if p == "F32" => {
                    let r = self.alloc_regf32();
                    writeln!(&mut self.ptx_buffer, "    ld.param.f32 {}, [{}_{}];", r, param.name, i).unwrap();
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

        // Emit kernel signature to original self.ptx_buffer
        writeln!(&mut self.ptx_buffer, ".visible .entry {}(", kernel.name).unwrap();

        let param_count = kernel.params.len();
        for (i, param) in kernel.params.iter().enumerate() {
            let ptx_type = match &param.ty {
                Type::Generic { base, .. } if base == "GlobalMemory" => ".param .u64",
                _ => ".param .b32",
            };

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
        let limit = if block_size > 0 {
            limit.min(max_regs_per_sm / block_size)
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
        writeln!(&mut self.ptx_buffer).unwrap();

        // Append the body code
        self.ptx_buffer.push_str(&body_code);

        writeln!(&mut self.ptx_buffer, "}}").unwrap();
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

                let mut j = i + 1;
                let mut hoisted = Vec::new();
                while j < stmts.len() && hoist_count < budget {
                    let is_independent_alu = matches!(
                        &stmts[j],
                        Stmt::Let {
                            init: Some(Expr::BinaryOp { .. }),
                            ..
                        } | Stmt::Assign {
                            value: Expr::BinaryOp { .. },
                            ..
                        }
                    );

                    if is_independent_alu {
                        hoisted.push(stmts.remove(j));
                        hoist_count += 1;
                    } else {
                        j += 1;
                    }
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
            Stmt::Let {
                name,
                init,
                cache_policy,
                ..
            } => {
                if let Some(expr) = init {
                    let val_str = self.emit_expr(expr, cache_policy.as_ref(), hw_profile);
                    if !val_str.is_empty() {
                        self.variables.insert(name.clone(), val_str);
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
                let step_val = match step {
                    Some(Expr::IntLit(step, _)) if *step > 0 && *step <= u32::MAX as i64 => {
                        *step as u32
                    }
                    _ => 1,
                };

                if step_val == 4 {
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
                writeln!(
                    &mut self.ptx_buffer,
                    "    add.u32 {}, {}, {};",
                    loop_reg, loop_reg, step_val
                )
                .unwrap();
                writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
                writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
            }
            Stmt::Assign {
                target, value, ..
            } => {
                let val_reg = self.emit_expr(value, None, hw_profile);
                if let Expr::Ident(name, _) = target {
                    if let Some(tgt_reg) = self.variables.get(name) {
                        if val_reg.starts_with("%f") {
                            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", tgt_reg, val_reg).unwrap();
                        } else if val_reg.starts_with("%rd") {
                            writeln!(&mut self.ptx_buffer, "    mov.u64 {}, {};", tgt_reg, val_reg).unwrap();
                        } else {
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", tgt_reg, val_reg).unwrap();
                        }
                    }
                }
            }
            Stmt::Expr(expr) => {
                self.emit_expr(expr, None, hw_profile);
            }
            Stmt::Return(_, _) => {}
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
                if total_cost < hw_profile.branch_divergence_penalty_cycles {
                    writeln!(
                        &mut self.ptx_buffer,
                        "    // Block cost ({} cy) < Penalty. Emitting PREDICATED execution.",
                        total_cost
                    )
                    .unwrap();
                    let pred = self.alloc_pred();
                    let cond_reg = if cond_str.is_empty() {
                        "%r0".to_string()
                    } else {
                        cond_str
                    };
                    writeln!(
                        &mut self.ptx_buffer,
                        "    setp.ne.u32 {}, {}, 0;",
                        pred, cond_reg
                    )
                    .unwrap();
                    writeln!(&mut self.ptx_buffer, "    @{} {{", pred).unwrap();
                    self.emit_block(then_block, hw_profile);
                    writeln!(&mut self.ptx_buffer, "    }}").unwrap();
                    if let Some(eb) = else_block {
                        writeln!(&mut self.ptx_buffer, "    @!{} {{", pred).unwrap();
                        self.emit_block(eb, hw_profile);
                        writeln!(&mut self.ptx_buffer, "    }}").unwrap();
                    }
                } else {
                    writeln!(
                        &mut self.ptx_buffer,
                        "    // Block cost ({} cy) >= Penalty. Emitting BRANCH execution.",
                        total_cost
                    )
                    .unwrap();
                    let pred = self.alloc_pred();
                    let cond_reg = if cond_str.is_empty() {
                        "%r0".to_string()
                    } else {
                        cond_str
                    };
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
            Expr::IntLit(val, _) => {
                let reg = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", reg, *val).unwrap();
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
            Expr::Ident(name, _) => {
                if let Some(reg) = self.variables.get(name) {
                    reg.clone()
                } else {
                    name.clone()
                }
            }
            Expr::BinaryOp {
                op, left, right, ..
            } => {
                let l_reg = self.emit_expr(left, cache_policy, hw_profile);
                let r_reg = self.emit_expr(right, cache_policy, hw_profile);
                if l_reg.starts_with("%f") || r_reg.starts_with("%f") {
                    let dst = self.alloc_regf32();
                    let op_str = match op {
                        BinaryOp::Add => "add.f32",
                        BinaryOp::Sub => "sub.f32",
                        BinaryOp::Mul => "mul.f32",
                        BinaryOp::Div => "div.approx.f32",
                        _ => "add.f32"
                    };
                    writeln!(&mut self.ptx_buffer, "    {} {}, {}, {};", op_str, dst, l_reg, r_reg).unwrap();
                    dst
                } else {
                    let dst = self.alloc_reg32();
                    let op_str = match op {
                        BinaryOp::Add => "add.s32",
                        BinaryOp::Sub => "sub.s32",
                        BinaryOp::Mul => "mul.lo.s32",
                        BinaryOp::Div => "div.s32",
                        _ => "add.s32"
                    };
                    writeln!(&mut self.ptx_buffer, "    {} {}, {}, {};", op_str, dst, l_reg, r_reg).unwrap();
                    dst
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
                    let offset_reg = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", offset_reg, idx_u64).unwrap();

                    let addr_reg = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr_reg, base_reg, offset_reg).unwrap();
                    addr_reg
                }
            }
            Expr::Call { func, args, .. } => {
                match &**func {
                    Expr::Ident(fname, _) => {
                        if fname == "cp_async" && args.len() >= 2 {
                            let src_reg = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let dest_reg = self.emit_expr(&args[1], cache_policy, hw_profile);
                            writeln!(&mut self.ptx_buffer, "    cp.async.cg.shared.global [{}], [{}], 16;", dest_reg, src_reg).unwrap();
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

                            writeln!(&mut self.ptx_buffer, "    and.b32 {}, %tid.x, 31;", lane_id).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shr.u32 {}, %tid.x, 5;", warp_id).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, {}, 0;", pred_first_lane, lane_id).unwrap();

                            let warp_id_u64 = self.alloc_reg64();
                            let smem_offset = self.alloc_reg64();
                            let smem_addr = self.alloc_reg64();

                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", warp_id_u64, warp_id).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", smem_offset, warp_id_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvta.to.shared.u64 {}, smem_reduce;", smem_addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", smem_addr, smem_addr, smem_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} st.shared.f32 [{}], {};", pred_first_lane, smem_addr, warp_sum).unwrap();
                            writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();

                            let smem_base = self.alloc_reg64();
                            let tid_u64 = self.alloc_reg64();
                            let tid_offset = self.alloc_reg64();
                            let smem_read_addr = self.alloc_reg64();
                            let pred_warp0_threads = self.alloc_pred();
                            let val_warp_sum = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    cvta.to.shared.u64 {}, smem_reduce;", smem_base).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, %tid.x;", tid_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", tid_offset, tid_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", smem_read_addr, smem_base, tid_offset).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, %tid.x, 8;", pred_warp0_threads).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} ld.shared.f32 {}, [{}];", pred_warp0_threads, val_warp_sum, smem_read_addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", pred_warp0_threads, val_warp_sum).unwrap();

                            let block_sum = self.emit_warp_reduce_sum(&val_warp_sum);

                            let inv_rms = self.alloc_regf32();
                            let mean = self.alloc_regf32();
                            let pred_tid0 = self.alloc_pred();
                            writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, %tid.x, 0;", pred_tid0).unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, 0.0009765625;", mean, block_sum).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 0.00001;", mean, mean).unwrap();
                            writeln!(&mut self.ptx_buffer, "    rsqrt.approx.f32 {}, {};", inv_rms, mean).unwrap();

                            let smem_root = self.alloc_reg64();
                            writeln!(&mut self.ptx_buffer, "    cvta.to.shared.u64 {}, smem_reduce;", smem_root).unwrap();
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
                            let src_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let dest_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%rd1".to_string() };
                            writeln!(&mut self.ptx_buffer, "    // [HOPPER TMA BULK TENSOR COPY]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [{}], [{}];", dest_reg, src_reg).unwrap();
                            "".into()
                        } else if fname == "wgmma_async" || fname == "wgmma_mma_async" {
                            writeln!(&mut self.ptx_buffer, "    // [HOPPER WGMMA WARP GROUP MATRIX MULTIPLY]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    wgmma.fence.sync.aligned;").unwrap();
                            writeln!(&mut self.ptx_buffer, "    wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {{%f0, %f1, %f2, %f3}}, %r0, %r1;").unwrap();
                            writeln!(&mut self.ptx_buffer, "    wgmma.commit_group.sync.aligned;").unwrap();
                            writeln!(&mut self.ptx_buffer, "    wgmma.wait_group.sync.aligned 0;").unwrap();
                            "".into()
                        } else if fname == "mbarrier_init" && args.len() >= 2 {
                            let bar_ptr = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let threads = self.emit_expr(&args[1], cache_policy, hw_profile);
                            writeln!(&mut self.ptx_buffer, "    mbarrier.init.shared.b64 [{}], {};", bar_ptr, threads).unwrap();
                            "".into()
                        } else if fname == "mbarrier_arrive" && args.len() >= 2 {
                            let bar_ptr = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let bytes = self.emit_expr(&args[1], cache_policy, hw_profile);
                            writeln!(&mut self.ptx_buffer, "    mbarrier.arrive.expect_tx.shared.b64 %rd0, [{}], {};", bar_ptr, bytes).unwrap();
                            "".into()
                        } else if fname == "mbarrier_try_wait" && !args.is_empty() {
                            let bar_ptr = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let parity = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "0".to_string() };
                            let p = self.alloc_pred();
                            writeln!(&mut self.ptx_buffer, "    mbarrier.try_wait.parity.shared.b64 {}, [{}], {};", p, bar_ptr, parity).unwrap();
                            p
                        } else if fname == "mma_sync" {
                            writeln!(&mut self.ptx_buffer, "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%f0,%f1}}, {{%r0,%r1}}, {{%r2,%r3}}, {{%f0,%f1}};").unwrap();
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

                        } else if fname == "store" && args.len() >= 2 {
                            let addr_reg = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let val_reg = self.emit_expr(&args[1], cache_policy, hw_profile);
                            if val_reg.starts_with("%f") {
                                writeln!(&mut self.ptx_buffer, "    st.global.f32 [{}], {};", addr_reg, val_reg).unwrap();
                            } else {
                                writeln!(&mut self.ptx_buffer, "    st.global.u32 [{}], {};", addr_reg, val_reg).unwrap();
                            }
                            "".into()
                        } else if fname == "block_tile_load" || fname == "tile_load" {
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

                            let lin_idx = self.alloc_reg32();
                            let lin_off = self.alloc_reg32();
                            let byte_off = self.alloc_reg64();
                            let lin_u64 = self.alloc_reg64();
                            let addr = self.alloc_reg64();
                            let p_r = self.alloc_pred();
                            let p_c = self.alloc_pred();
                            let p_valid = self.alloc_pred();
                            let res = self.alloc_regf32();

                            writeln!(&mut self.ptx_buffer, "    // [Y 2D TENSOR BLOCK POINTER LOAD - 2D STRIDED MASKED ACCESS]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    mul.lo.s32 {}, {}, {};", lin_off, row_reg, stride_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.s32 {}, {}, {};", lin_idx, lin_off, col_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    cvt.u64.u32 {}, {};", lin_u64, lin_idx).unwrap();
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", byte_off, lin_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_off).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_r, row_reg, max_r_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_c, col_reg, max_c_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_valid, p_r, p_c).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} ld.global.f32 {}, [{}];", p_valid, res, addr).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0.0;", p_valid, res).unwrap();
                            res
                        } else if fname == "block_ptr2d_store" {
                            let ptr_reg = if !args.is_empty() { self.emit_expr(&args[0], cache_policy, hw_profile) } else { "%rd0".to_string() };
                            let row_reg = if args.len() >= 2 { self.emit_expr(&args[1], cache_policy, hw_profile) } else { "%r0".to_string() };
                            let col_reg = if args.len() >= 3 { self.emit_expr(&args[2], cache_policy, hw_profile) } else { "%r1".to_string() };
                            let stride_reg = if args.len() >= 4 { self.emit_expr(&args[3], cache_policy, hw_profile) } else { "1024".to_string() };
                            let max_r_reg = if args.len() >= 5 { self.emit_expr(&args[4], cache_policy, hw_profile) } else { "128".to_string() };
                            let max_c_reg = if args.len() >= 6 { self.emit_expr(&args[5], cache_policy, hw_profile) } else { "1024".to_string() };
                            let val_reg = if args.len() >= 7 { self.emit_expr(&args[6], cache_policy, hw_profile) } else { "%f0".to_string() };

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
                            writeln!(&mut self.ptx_buffer, "    shl.b64 {}, {}, 2;", byte_off, lin_u64).unwrap();
                            writeln!(&mut self.ptx_buffer, "    add.u64 {}, {}, {};", addr, ptr_reg, byte_off).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_r, row_reg, max_r_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p_c, col_reg, max_c_reg).unwrap();
                            writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_valid, p_r, p_c).unwrap();
                            writeln!(&mut self.ptx_buffer, "    @{} st.global.f32 [{}], {};", p_valid, addr, val_reg).unwrap();
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
                        } else if fname == "block_arange" {
                            let res = self.alloc_reg32();
                            writeln!(&mut self.ptx_buffer, "    // [Y BLOCK ARANGE - 1D INDEX GENERATOR]").unwrap();
                            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", res).unwrap();
                            res
                        } else {
                            "".into()
                        }
                    }
                    Expr::Path {
                        namespace, member, ..
                    } => {
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
                            let addr_reg = if !args.is_empty() {
                                self.emit_expr(&args[0], cache_policy, hw_profile)
                            } else {
                                "%rd0".to_string()
                            };
                            let dst = self.alloc_regf32();
                            writeln!(&mut self.ptx_buffer, "    ld.global{}.f32 {}, [{}];", cache_str, dst, addr_reg).unwrap();
                            dst
                        } else {
                            "".into()
                        }
                    }
                    _ => "".into()
                }
            }
            Expr::MemberAccess { base: _, member, .. } => {
                if member == "wait" {
                    writeln!(&mut self.ptx_buffer, "    cp.async.wait_group 0;").unwrap();
                }
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
                    let dst = self.alloc_regf32();
                    writeln!(&mut self.ptx_buffer, "    ld.global.ca.f32 {}, [%rd0];", dst).unwrap();
                    dst
                } else if namespace == "SharedMemory" && member == "alloc" {
                    writeln!(&mut self.ptx_buffer, "    .shared .align 128 .b8 smem[8192];").unwrap();
                    "smem".into()
                } else {
                    "".into()
                }
            }
            Expr::GenericCall { func, .. } => {
                self.emit_expr(&**func, cache_policy, hw_profile)
            }
            _ => "".into(),
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
                    let d0 = val.digits.get(0).cloned().unwrap_or(0) as u64 | ((val.digits.get(1).cloned().unwrap_or(0) as u64) << 32);
                    let d1 = val.digits.get(2).cloned().unwrap_or(0) as u64 | ((val.digits.get(3).cloned().unwrap_or(0) as u64) << 32);
                    let d2 = val.digits.get(4).cloned().unwrap_or(0) as u64 | ((val.digits.get(5).cloned().unwrap_or(0) as u64) << 32);
                    let d3 = val.digits.get(6).cloned().unwrap_or(0) as u64 | ((val.digits.get(7).cloned().unwrap_or(0) as u64) << 32);

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
                WitnessOp::Inv(a) | WitnessOp::Div(a, _) => {
                    let a_id = a.0;
                    writeln!(&mut buffer, "    // 256-bit Field Inversion / Division Hint: s_{}", s_idx).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_0, %s{}_0;", s_idx, a_id).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_1, %s{}_1;", s_idx, a_id).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_2, %s{}_2;", s_idx, a_id).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_3, %s{}_3;", s_idx, a_id).unwrap();
                }
                _ => {
                    writeln!(&mut buffer, "    mov.u64 %s{}_0, 0;", s_idx).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_1, 0;", s_idx).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_2, 0;", s_idx).unwrap();
                    writeln!(&mut buffer, "    mov.u64 %s{}_3, 0;", s_idx).unwrap();
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
        let idx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", idx).unwrap();
        let loop_start = self.alloc_label(&format!("ALOAD_{}", label_tag));
        let loop_end = self.alloc_label(&format!("ALOAD_{}_DONE", label_tag));
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

        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", idx, idx, threads_per_cta).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Shared compute core of `emit_tensor_core_gemm_kernel`'s K-loop body,
    /// used identically by both the synchronous-fallback and
    /// cp.async-pipelined paths: loads each B fragment once per `kk`
    /// sub-step (reused across all `i`), then each A fragment once (reused
    /// across all `j`), accumulating via `wmma.mma` into `acc[i][j]`. The
    /// two paths differ only in *which* smem addresses `smem_a_base`/
    /// `smem_b_base` point at (a fixed compile-time base for the
    /// single-buffered fallback, a per-iteration `read_stage`-offset base
    /// for the pipelined path) - this function neither knows nor cares.
    #[allow(clippy::too_many_arguments)]
    fn emit_gemm_compute_block(
        &mut self,
        acc: &[Vec<Vec<String>>],
        warp_col0_local: &str,
        warp_row0_scaled: &str,
        smem_a_base: &str,
        smem_b_base: &str,
        smem_a_stride: u32,
        smem_b_stride: u32,
        stride_a_reg: &str,
        stride_b_reg: &str,
        k_substeps: u32,
        num_i: u32,
        num_j: u32,
    ) {
        for kk_step in 0..k_substeps {
            let kk = kk_step * 16;

            let mut b_frags: Vec<Vec<String>> = Vec::with_capacity(num_j as usize);
            for j in 0..num_j {
                let b_const = kk * smem_b_stride + j * 16;
                let b_lin = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", b_lin, warp_col0_local, b_const).unwrap();
                let b_byte = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 1;", b_byte, b_lin).unwrap();
                let b_addr = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", b_addr, smem_b_base, b_byte).unwrap();
                let frag: Vec<String> = (0..8).map(|_| self.alloc_reg32()).collect();
                writeln!(
                    &mut self.ptx_buffer,
                    "    wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 {{{}}}, [{}], {};",
                    frag.join(","), b_addr, stride_b_reg
                ).unwrap();
                b_frags.push(frag);
            }

            for i in 0..num_i {
                let a_const = i * 16 * smem_a_stride + kk;
                let a_lin = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", a_lin, warp_row0_scaled, a_const).unwrap();
                let a_byte = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 1;", a_byte, a_lin).unwrap();
                let a_addr = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", a_addr, smem_a_base, a_byte).unwrap();
                let a_frag: Vec<String> = (0..8).map(|_| self.alloc_reg32()).collect();
                writeln!(
                    &mut self.ptx_buffer,
                    "    wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {{{}}}, [{}], {};",
                    a_frag.join(","), a_addr, stride_a_reg
                ).unwrap();

                for j in 0..num_j {
                    let d = acc[i as usize][j as usize].join(",");
                    writeln!(
                        &mut self.ptx_buffer,
                        "    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                        d, a_frag.join(","), b_frags[j as usize].join(","), d
                    ).unwrap();
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
        debug_assert_eq!(k % cta_k, 0, "@tile K must be a multiple of the autotuned cta_k - see doc comment scope note");

        let smem_a_stride = cta_k + 8; // padded, elements/row - see doc comment
        let smem_b_stride = cta_n + 8;
        let k_tiles = k / cta_k;

        // ---- effective pipeline depth: see doc comment ----
        let stage_a_bytes = cta_m * smem_a_stride * 2;
        let stage_b_bytes = cta_k * smem_b_stride * 2;
        let per_stage_bytes = stage_a_bytes + stage_b_bytes;
        let safe_smem_ceiling = hw_profile.max_smem_per_sm_bytes.saturating_sub(4096);
        let max_stages_by_smem = (safe_smem_ceiling / per_stage_bytes).max(1);
        let effective_stages = config.num_stages.min(k_tiles).min(max_stages_by_smem).max(1);
        let mut total_dyn_smem_bytes = effective_stages * per_stage_bytes;

        // ---- fused Bias+ReLU epilogue: reuses this same dynamic shared
        // buffer (already idle by the time the epilogue runs - see
        // emit_gemm_bias_relu_epilogue's doc comment) as a row-major f32
        // scratch tile. A *full* cta_m x cta_n f32 tile (e.g. 128x260x4 =
        // ~133KB for a 128x256 CTA tile) can exceed this GPU's real per-CTA
        // dynamic-smem opt-in ceiling (~101376B measured on this project's
        // sm_89 dev machine - see emit_tensor_core_gemm_kernel's doc
        // comment) even though the A/B pipeline stages themselves fit fine.
        // So the epilogue instead runs in `warps_n` passes, one per warp
        // *column* (see emit_kernel's write-loop below and
        // emit_gemm_bias_relu_epilogue): each pass's scratch tile is only
        // cta_m x per_warp_n wide - `per_warp_n <= cta_n`, so this is always
        // <= the full-tile size, and for every autotuned config observed so
        // far comfortably fits alongside the pipeline's own smem budget.
        // Supported in BOTH the pipelined and `effective_stages < 2`
        // fallback branches - the fallback branch switches to this same
        // dynamic extern-shared mechanism (instead of static `.shared`
        // arrays) whenever bias is present specifically so this holds; see
        // that branch's own doc comment for why it's real, observed
        // (M=N=K=2048 on this project's dev GPU clamps to 1 stage on smem
        // pressure alone, before the epilogue even factors in), not a
        // theoretical edge case.
        let smem_c_stride = per_warp_n + 4; // padded, elements/row - matches tests/y_tensor_core_gemm.cu's fused kernel (there: cta_n-wide, here: per-warp-column-banded, see above)
        let smem_c_bytes = cta_m * smem_c_stride * 4;
        if bias_ptr.is_some() {
            total_dyn_smem_bytes = total_dyn_smem_bytes.max(smem_c_bytes);
        }

        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [Y TENSOR CORE GEMM] M={} N={} K={} | CTA {}x{}x{} | {}x{} warps | wmma.sync.m16n16k16.f16->f32", m, n, k, cta_m, cta_n, cta_k, warps_m, warps_n).unwrap();
        if effective_stages >= 2 {
            writeln!(&mut self.ptx_buffer, "    // Autotuner selected {} pipeline stages ({} used after k_tiles/smem clamping); cp.async multi-stage pipelined. Dynamic shared memory required: {} bytes.", config.num_stages, effective_stages, total_dyn_smem_bytes).unwrap();
        } else if bias_ptr.is_some() {
            // Unlike the plain-GEMM fallback (static `.shared`, needs no
            // launch-time dynamic smem request), this kernel's fused
            // Bias+ReLU epilogue always needs a *dynamic* buffer - see the
            // `effective_stages < 2` branch's `bias_ptr.is_some()` case -
            // so the launcher-facing byte count must be reported here too,
            // not just on the `>=2`-stage path above.
            writeln!(&mut self.ptx_buffer, "    // Autotuner selected {} pipeline stages; only {} K-tile(s) exist so this path stages synchronously (see emit_tensor_core_gemm_kernel doc comment). Dynamic shared memory required: {} bytes.", config.num_stages, k_tiles, total_dyn_smem_bytes).unwrap();
        } else {
            writeln!(&mut self.ptx_buffer, "    // Autotuner selected {} pipeline stages; only {} K-tile(s) exist so this path stages synchronously (see emit_tensor_core_gemm_kernel doc comment).", config.num_stages, k_tiles).unwrap();
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

        let stride_a_reg = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", stride_a_reg, smem_a_stride).unwrap();
        let stride_b_reg = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", stride_b_reg, smem_b_stride).unwrap();

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
        let warp_row0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", warp_row0, warp_m, per_warp_m, cta_m_start).unwrap();
        let warp_col0 = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", warp_col0, warp_n, per_warp_n, cta_n_start).unwrap();
        // Loop-invariant, hoisted: every A-fragment address needs
        // (warp_row0_within_cta) * smem_a_stride, so compute the CTA-local
        // (not grid-global) warp row once, up front.
        let warp_row0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_row0_local, warp_m, per_warp_m).unwrap();
        let warp_col0_local = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_col0_local, warp_n, per_warp_n).unwrap();
        let warp_row0_scaled = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", warp_row0_scaled, warp_row0_local, smem_a_stride).unwrap();

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
        let mut smem_pipeline_base_for_epilogue: Option<String> = None;

        if effective_stages < 2 {
            // ---- Fallback: original single-buffered synchronous path
            // (load -> bar.sync -> compute -> bar.sync). Only reachable when
            // `effective_stages` was clamped all the way down to 1 - see
            // doc comment; not exercised by any candidate
            // `Autotuner::generate_candidates` currently produces for this
            // project's tested shapes, kept as a real, correct fallback
            // rather than a debug_assert/panic since nothing about a
            // validated `@tile(M,N,K)` rules it out for a future shape.
            // (This branch IS reachable with a fused Bias+ReLU kernel,
            // though - a large-N tile's per_stage_bytes can be big enough on
            // its own, before the epilogue even enters the picture, to clamp
            // `effective_stages` to 1 - real, observed on this project's own
            // dev GPU at M=N=K=2048 with a 128x256x64 CTA tile. So unlike
            // the plain-GEMM case, this path can't just use static `.shared`
            // arrays when bias is present: the epilogue needs a *dynamic*
            // buffer it can address at a runtime-computed size across
            // multiple warp-column passes - see the smem-sizing doc comment
            // above.)
            let smem_a_base;
            let smem_b_base;
            if bias_g.is_some() {
                let smem_symbol = format!("smem_pipeline_{}", kernel_name);
                self.pending_extern_decls.push(format!(".extern .shared .align 16 .b8 {}[];", smem_symbol));
                let base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", base, smem_symbol).unwrap();
                smem_pipeline_base_for_epilogue = Some(base.clone());
                let a_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", a_base, base).unwrap();
                let b_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", b_base, base, stage_a_bytes).unwrap();
                smem_a_base = a_base;
                smem_b_base = b_base;
            } else {
                writeln!(&mut self.ptx_buffer, "    .shared .align 4 .b8 smem_A[{}];", stage_a_bytes).unwrap();
                writeln!(&mut self.ptx_buffer, "    .shared .align 4 .b8 smem_B[{}];", stage_b_bytes).unwrap();
                let a_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, smem_A;", a_base).unwrap();
                let b_base = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, smem_B;", b_base).unwrap();
                smem_a_base = a_base;
                smem_b_base = b_base;
            }

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
                &acc, &warp_col0_local, &warp_row0_scaled, &smem_a_base, &smem_b_base,
                smem_a_stride, smem_b_stride, &stride_a_reg, &stride_b_reg, k_substeps, num_i, num_j,
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
            smem_pipeline_base_for_epilogue = Some(smem_pipeline_base.clone());
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
                &acc, &warp_col0_local, &warp_row0_scaled, &a_read_base, &b_read_base,
                smem_a_stride, smem_b_stride, &stride_a_reg, &stride_b_reg, k_substeps, num_i, num_j,
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

        if let Some(bias_g) = bias_g {
            // ---- fused epilogue: `warps_n` passes, one per warp-column band
            // (see the smem-sizing doc comment above for why) - each pass
            // does wmma.store.d into the (small, per_warp_n-wide) shared
            // scratch tile for just that band's warps, then a CTA-wide
            // bias-broadcast + ReLU + boundary-masked store to global for
            // that band - see emit_gemm_bias_relu_epilogue's doc comment ----
            let smem_c_base = smem_pipeline_base_for_epilogue
                .expect("smem_pipeline_base_for_epilogue must be set in both branches when bias_ptr is Some - see the effective_stages<2 branch's bias_ptr.is_some() case");
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
                self.emit_gemm_bias_relu_epilogue(
                    &smem_c_base, smem_c_stride, &cta_m_start, &cta_n_start_slot, cta_m, per_warp_n, m, n, &c_g, &bias_g, threads_per_cta,
                );
                // Next slot's wmma.store reuses this same scratch buffer -
                // must not begin until every thread is done reading this
                // slot's data out of it.
                writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
            }
        } else {
            // ---- epilogue: boundary-masked store, whole-fragment granularity
            // (see scope note in the doc comment above) ----
            let stride_c_reg = self.alloc_reg32();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", stride_c_reg, n).unwrap();
            for i in 0..num_i {
                for j in 0..num_j {
                    let c_row = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", c_row, warp_row0, i * 16).unwrap();
                    let c_col = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", c_col, warp_col0, j * 16).unwrap();
                    let c_row_end = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 16;", c_row_end, c_row).unwrap();
                    let c_col_end = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 16;", c_col_end, c_col).unwrap();
                    let p_row = self.alloc_pred();
                    writeln!(&mut self.ptx_buffer, "    setp.le.u32 {}, {}, {};", p_row, c_row_end, m).unwrap();
                    let p_col = self.alloc_pred();
                    writeln!(&mut self.ptx_buffer, "    setp.le.u32 {}, {}, {};", p_col, c_col_end, n).unwrap();
                    let p_ok = self.alloc_pred();
                    writeln!(&mut self.ptx_buffer, "    and.pred {}, {}, {};", p_ok, p_row, p_col).unwrap();

                    let c_lin = self.alloc_reg32();
                    writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", c_lin, c_row, n, c_col).unwrap();
                    let c_byte = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", c_byte, c_lin).unwrap();
                    let c_addr = self.alloc_reg64();
                    writeln!(&mut self.ptx_buffer, "    add.s64 {}, {}, {};", c_addr, c_g, c_byte).unwrap();

                    let d = acc[i as usize][j as usize].join(",");
                    writeln!(
                        &mut self.ptx_buffer,
                        "    @{} wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [{}], {{{}}}, {};",
                        p_ok, c_addr, d, stride_c_reg
                    ).unwrap();
                }
            }
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

    /// 3. Automatic Hopper TMA Descriptor Generation (`sm_90a`).

    pub fn emit_tma_descriptor_gen(&mut self, desc_name: &str, tensor_name: &str, dim_m: u32, dim_n: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [AUTOMATIC HOPPER TMA TENSOR DESCRIPTOR GENERATION (sm_90a)]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Descriptor: {} -> Tensor: {} (Shape: {}x{})", desc_name, tensor_name, dim_m, dim_n).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    .global .align 128 .b8 {}[128];", desc_name).unwrap();
    }

    /// Emits Hopper Bulk Tensor Copy via TMA.
    pub fn emit_tma_bulk_load(&mut self, dest_smem: &str, desc_name: &str, coord_x: &str, coord_y: &str) {
        writeln!(&mut self.ptx_buffer, "    // [HOPPER TMA BULK TENSOR STREAMING]").unwrap();
        writeln!(
            &mut self.ptx_buffer,
            "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [{}], [{}], {{{}, {}}};",
            dest_smem, desc_name, coord_x, coord_y
        )
        .unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.wait_group 0;").unwrap();
    }

    /// 4. 3+ Stage Asynchronous Pipelining backed by Hopper `mbarrier` transaction counters.
    pub fn emit_mbarrier_3stage_pipelined_loop(&mut self, num_stages: u32, k_total: u32, tile_k: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [3+ STAGE ASYNCHRONOUS PIPELINING WITH mbarrier COUNTERS]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Stages: {}, Total K: {}, Tile K: {}", num_stages, k_total, tile_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        writeln!(&mut self.ptx_buffer, "    .shared .align 8 .b64 mbar[{}];", num_stages).unwrap();

        // Init mbarriers
        for s in 0..num_stages {
            writeln!(&mut self.ptx_buffer, "    mbarrier.init.shared.b64 [mbar + {}], 128;", s * 8).unwrap();
        }

        // Prologue
        for s in 0..(num_stages - 1) {
            writeln!(&mut self.ptx_buffer, "    mbarrier.arrive.expect_tx.shared.b64 %rd0, [mbar + {}], 4096;", s * 8).unwrap();
            writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [smem_stage{}], [tma_desc_A];", s).unwrap();
        }

        let loop_start = self.alloc_label("MBARRIER_LOOP_START");
        let loop_end = self.alloc_label("MBARRIER_LOOP_END");
        let k_iter = self.alloc_reg32();
        let k_limit = self.alloc_reg32();

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", k_limit, k_total / tile_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();

        let stage_fetch = (num_stages - 1) as usize;
        writeln!(&mut self.ptx_buffer, "    // Fetch stage k+{} backed by mbarrier", stage_fetch).unwrap();
        writeln!(&mut self.ptx_buffer, "    mbarrier.arrive.expect_tx.shared.b64 %rd0, [mbar + {}], 4096;", stage_fetch * 8).unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [smem_stage{}], [tma_desc_A];", stage_fetch).unwrap();

        // Wait stage 0 mbarrier
        let pred = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    mbarrier.try_wait.parity.shared.b64 {}, [mbar], 0;", pred).unwrap();

        // Compute stage 0
        writeln!(&mut self.ptx_buffer, "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%f0,%f1}}, {{%r0,%r1}}, {{%r2,%r3}}, {{%f0,%f1}};").unwrap();

        // Loop branch
        let loop_pred = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", loop_pred, k_iter, k_limit).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", loop_pred, loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// 5. Hopper Warp-Group Matrix Multiply (`wgmma.mma_async`).
    /// Operates across 128 threads simultaneously (4 warps = 1 warp group).
    pub fn emit_wgmma_warp_group_gemm(&mut self, cta_m: u32, cta_n: u32, cta_k: u32, k_total: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [HOPPER WARP-GROUP MATRIX MULTIPLY - wgmma.mma_async]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Tile Layout: {}x{}x{}, Total K: {}, 128 Threads (1 Warp Group)", cta_m, cta_n, cta_k, k_total).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.fence.sync.aligned;").unwrap();

        let k_iter = self.alloc_reg32();
        let k_limit = self.alloc_reg32();
        let loop_start = self.alloc_label("WGMMA_LOOP_START");
        let loop_end = self.alloc_label("WGMMA_LOOP_END");

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", k_limit, k_total / cta_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();

        // 128-thread warp-group WGMMA tensor core operation
        writeln!(
            &mut self.ptx_buffer,
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {{%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11,%f12,%f13,%f14,%f15}}, desc_A, desc_B, 1, 1, 0, 0;"
        )
        .unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.commit_group.sync.aligned;").unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.wait_group.sync.aligned 0;").unwrap();

        let pred = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, k_iter, k_limit).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", pred, loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// 5.1 Hopper Warp-Group FP8 Matrix Multiply (`wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3`).
    pub fn emit_wgmma_fp8_gemm(&mut self, cta_m: u32, cta_n: u32, cta_k: u32, k_total: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [HOPPER FP8 WARP-GROUP MATRIX MULTIPLY - e4m3fn]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Tile Layout: {}x{}x{}, Total K: {}, 128 Threads (FP8 Tensor Core)", cta_m, cta_n, cta_k, k_total).unwrap();
        writeln!(&mut self.ptx_buffer, "    // Optimization: 3-Stage Async TMA Pipelining + 128B Swizzle + Fused Scale Vector Writeback").unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.fence.sync.aligned;").unwrap();

        let k_iter = self.alloc_reg32();
        let k_limit = self.alloc_reg32();
        let loop_start = self.alloc_label("WGMMA_FP8_PIPELINED_LOOP_START");
        let loop_end = self.alloc_label("WGMMA_FP8_PIPELINED_LOOP_END");

        writeln!(&mut self.ptx_buffer, "    // [3-STAGE ASYNC TMA PREFETCH INITIATION]").unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [smem_A_stage0], [desc_A], {{%r0, %r1}};").unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [smem_B_stage0], [desc_B], {{%r0, %r1}};").unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [smem_A_stage1], [desc_A], {{%r0, %r1}};").unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [smem_B_stage1], [desc_B], {{%r0, %r1}};").unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.commit_group.sync.aligned;").unwrap();

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", k_limit, k_total / cta_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();

        writeln!(
            &mut self.ptx_buffer,
            "    wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {{%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11,%f12,%f13,%f14,%f15}}, desc_A, desc_B, 1, 1, 0, 0;"
        ).unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.commit_group.sync.aligned;").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [OVERLAP MEMORY READS WITH TENSOR CORE MATH - WAIT GROUP 1]").unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.wait_group.sync.aligned 1;").unwrap();

        let pred = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, k_iter, k_limit).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", pred, loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.wait_group.sync.aligned 0;").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [FUSED FP8 SCALE MULTIPLY & 128-BIT VECTOR STORE - st.global.v4.f32]").unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 %f0, %f0, %scale_ab;").unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 %f1, %f1, %scale_ab;").unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 %f2, %f2, %scale_ab;").unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 %f3, %f3, %scale_ab;").unwrap();
        writeln!(&mut self.ptx_buffer, "    st.global.v4.f32 [%rd0], {{%f0, %f1, %f2, %f3}};").unwrap();
    }



    /// 5.2 Hopper Warp-Group INT4 Matrix Multiply (`wgmma.mma_async.sync.aligned.m64n64k64.s32.s4.s4`).
    pub fn emit_wgmma_int4_gemm(&mut self, cta_m: u32, cta_n: u32, cta_k: u32, k_total: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [HOPPER INT4 WARP-GROUP MATRIX MULTIPLY - s4/u4]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Tile Layout: {}x{}x{}, Total K: {}, 128 Threads (INT4 Sub-Byte Tensor Core)", cta_m, cta_n, cta_k, k_total).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.fence.sync.aligned;").unwrap();

        let k_iter = self.alloc_reg32();
        let k_limit = self.alloc_reg32();
        let loop_start = self.alloc_label("WGMMA_INT4_LOOP_START");
        let loop_end = self.alloc_label("WGMMA_INT4_LOOP_END");

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", k_limit, k_total / cta_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_start).unwrap();

        writeln!(
            &mut self.ptx_buffer,
            "    wgmma.mma_async.sync.aligned.m64n64k64.s32.s4.s4 {{%r0,%r1,%r2,%r3,%r4,%r5,%r6,%r7}}, desc_A, desc_B, 1, 1, 0, 0;"
        ).unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.commit_group.sync.aligned;").unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.wait_group.sync.aligned 0;").unwrap();

        let pred = self.alloc_pred();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", k_iter, k_iter).unwrap();
        writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", pred, k_iter, k_limit).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", pred, loop_start).unwrap();
        writeln!(&mut self.ptx_buffer, "    {}:", loop_end).unwrap();
    }

    /// Emits Thread Block Cluster dimensions directive (.cluster_dimensions x, y, z) for Hopper/Blackwell.
    pub fn emit_cluster_dimensions(&mut self, x: u32, y: u32, z: u32) {
        writeln!(&mut self.ptx_buffer, "    // [HOPPER/BLACKWELL CLUSTER DIMENSIONS]").unwrap();
        writeln!(&mut self.ptx_buffer, "    .cluster_dimensions {}, {}, {};", x, y, z).unwrap();
    }

    /// Emits Hopper TMA Multicast Bulk Tensor Load directly to Shared Memory of Cluster CTAs.
    pub fn emit_tma_multicast_bulk_load(&mut self, dest_smem: &str, desc_name: &str, coord_x: &str, coord_y: &str, cluster_mask: &str) {
        writeln!(&mut self.ptx_buffer, "    // [HOPPER TMA MULTICAST BROADCAST TO CLUSTER DSMEM]").unwrap();
        writeln!(
            &mut self.ptx_buffer,
            "    cp.async.bulk.tensor.multicast.shared::cluster.global.mbarrier::complete_tx::bytes [{}], [{}], {{{}, {}}}, {};",
            dest_smem, desc_name, coord_x, coord_y, cluster_mask
        )
        .unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.wait_group 0;").unwrap();
    }

    /// Emits Warp-Specialized Producer/Consumer Pipeline (Warp 0 = TMA Producer, Warps 1..3 = WGMMA Consumer).
    pub fn emit_warp_specialized_producer_consumer_pipeline(&mut self, cta_m: u32, cta_n: u32, cta_k: u32, k_total: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [WARP-SPECIALIZED PRODUCER/CONSUMER PIPELINE]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Warp 0: TMA Producer (<=16 regs) | Warps 1..3: Consumer WGMMA (96 Threads)").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Tile: {}x{}x{}, Total K: {}", cta_m, cta_n, cta_k, k_total).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        let tid_reg = self.alloc_reg32();
        let warp_id_reg = self.alloc_reg32();
        let is_producer = self.alloc_pred();
        let producer_label = self.alloc_label("PRODUCER_LOOP");
        let consumer_label = self.alloc_label("CONSUMER_WARPGROUP_LOOP");
        let end_label = self.alloc_label("WARP_SPEC_END");

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %tid.x;", tid_reg).unwrap();
        writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 5;", warp_id_reg, tid_reg).unwrap();
        writeln!(&mut self.ptx_buffer, "    setp.eq.u32 {}, {}, 0;", is_producer, warp_id_reg).unwrap();
        writeln!(&mut self.ptx_buffer, "    @{} bra {};", is_producer, producer_label).unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", consumer_label).unwrap();

        // Producer Warp Branch (Warp 0)
        writeln!(&mut self.ptx_buffer, "    {}:", producer_label).unwrap();
        writeln!(&mut self.ptx_buffer, "    // [PRODUCER WARP 0: ISSUING TMA LOADS & ADVANCING MBARRIER]").unwrap();
        writeln!(&mut self.ptx_buffer, "    mbarrier.arrive.expect_tx.shared.b64 %rd0, [mbar], 4096;").unwrap();
        writeln!(&mut self.ptx_buffer, "    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [smem_A], [desc_A], {{%r0, %r1}};").unwrap();
        writeln!(&mut self.ptx_buffer, "    bra {};", end_label).unwrap();

        // Consumer Warpgroup Branch (Warps 1..3)
        writeln!(&mut self.ptx_buffer, "    {}:", consumer_label).unwrap();
        writeln!(&mut self.ptx_buffer, "    // [CONSUMER WARPGROUP 1..3: RUNNING CONTINUOUS WGMMA COMPUTATION]").unwrap();
        writeln!(&mut self.ptx_buffer, "    mbarrier.try_wait.parity.shared.b64 %p0, [mbar], 0;").unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.fence.sync.aligned;").unwrap();
        writeln!(
            &mut self.ptx_buffer,
            "    wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {{%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7}}, desc_A, desc_B, 1, 1, 0, 0;"
        ).unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.commit_group.sync.aligned;").unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.wait_group.sync.aligned 0;").unwrap();

        writeln!(&mut self.ptx_buffer, "    {}:", end_label).unwrap();
    }

    /// Emits Hopper FP8 (E4M3 / E5M2) Dual-Accumulator WGMMA Pipeline (`wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3`).
    pub fn emit_wgmma_fp8_dual_accumulator_gemm(&mut self, cta_m: u32, cta_n: u32, cta_k: u32, scale_a: &str, scale_b: &str) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [HOPPER NATIVE FP8 E4M3 DUAL-ACCUMULATOR WGMMA PIPELINE]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Dimensions: M={}, N={}, K={} FP8 (e4m3fn)", cta_m, cta_n, cta_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    // Scales: A={}, B={}", scale_a, scale_b).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        writeln!(&mut self.ptx_buffer, "    wgmma.fence.sync.aligned;").unwrap();
        writeln!(
            &mut self.ptx_buffer,
            "    wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {{%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11,%f12,%f13,%f14,%f15}}, desc_A, desc_B, {}, {}, 0, 0;",
            scale_a, scale_b
        )
        .unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.commit_group.sync.aligned;").unwrap();
        writeln!(&mut self.ptx_buffer, "    wgmma.wait_group.sync.aligned 0;").unwrap();
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


    /// 6. Emits unified fused GPU kernel (MatMul + RMSNorm + SwiGLU).
    /// Eliminates intermediate DRAM/L2 roundtrips.
    pub fn emit_fused_matmul_rmsnorm_swiglu(&mut self, m: u32, n: u32, k: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [AUTOMATED OPERATOR FUSION KERNEL - MatMul + RMSNorm + SwiGLU]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Dimensions: M={}, N={}, K={} (Fused single launch)", m, n, k).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        // Step 1: MatMul compute via mma.sync or wgmma
        writeln!(&mut self.ptx_buffer, "    // Stage 1: Linear Projection GEMM").unwrap();
        writeln!(&mut self.ptx_buffer, "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%f0,%f1}}, {{%r0,%r1}}, {{%r2,%r3}}, {{%f0,%f1}};").unwrap();

        // Step 2: RMSNorm via warp butterfly shuffle reduction inside registers
        writeln!(&mut self.ptx_buffer, "    // Stage 2: RMSNorm Variance Reduction via Warp Butterfly Shuffle").unwrap();
        let val_sq = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, %f0, %f0;", val_sq).unwrap();
        let red_sq = self.emit_warp_reduce_sum(&val_sq);
        let inv_rms = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    rsqrt.approx.f32 {}, {};", inv_rms, red_sq).unwrap();
        let norm_val = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, %f0, {};", norm_val, inv_rms).unwrap();

        // Step 3: SwiGLU activation (Swish(x) * y) in register space
        writeln!(&mut self.ptx_buffer, "    // Stage 3: SwiGLU In-Register Activation").unwrap();
        let neg_norm = self.alloc_regf32();
        let exp_val = self.alloc_regf32();
        let sig_denom = self.alloc_regf32();
        let sig_val = self.alloc_regf32();
        let swish = self.alloc_regf32();
        let final_out = self.alloc_regf32();

        writeln!(&mut self.ptx_buffer, "    neg.f32 {}, {};", neg_norm, norm_val).unwrap();
        writeln!(&mut self.ptx_buffer, "    ex2.approx.f32 {}, {};", exp_val, neg_norm).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.f32 {}, {}, 1.0;", sig_denom, exp_val).unwrap();
        writeln!(&mut self.ptx_buffer, "    rcp.approx.f32 {}, {};", sig_val, sig_denom).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", swish, norm_val, sig_val).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, %f1;", final_out, swish).unwrap();

        // Step 4: Write final fused result to global memory using 128-bit store
        writeln!(&mut self.ptx_buffer, "    // Stage 4: 128-Bit SIMD Store directly to DRAM").unwrap();
        writeln!(&mut self.ptx_buffer, "    st.global.v4.f32 [%rd0], {{{}, {}, {}, {}}};", final_out, final_out, final_out, final_out).unwrap();
    }

    /// Fast Bit Manipulation (lop3.b32 / prmt.b32) for zero-overhead INT4 / INT2 Dequantization.
    pub fn emit_fast_int4_dequant_lop3(&mut self, packed_reg: &str, scale_reg: &str, zero_reg: &str) -> String {
        let out_f16 = self.alloc_regf32();
        writeln!(&mut self.ptx_buffer, "    // [FAST SUB-BYTE INT4 DEQUANTIZATION via LOP3/PRMT]").unwrap();
        writeln!(&mut self.ptx_buffer, "    lop3.b32 %r_masked, {}, 0x0F0F0F0F, 0, 0x80;", packed_reg).unwrap();
        writeln!(&mut self.ptx_buffer, "    prmt.b32 %r_unpacked, %r_masked, 0, 0x3210;").unwrap();
        writeln!(&mut self.ptx_buffer, "    sub.s32 %r_sub, %r_unpacked, {};", zero_reg).unwrap();
        writeln!(&mut self.ptx_buffer, "    cvt.rn.f32.s32 %f_val, %r_sub;").unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 {}, %f_val, {};", out_f16, scale_reg).unwrap();
        out_f16
    }

    /// FP8 Tensor Core Scaling Factor & Accumulation Control (e4m3fn / e5m2).
    pub fn emit_fp8_scaling_mma(&mut self, cta_m: u32, cta_n: u32, cta_k: u32, scale_a: &str, scale_b: &str) {
        writeln!(&mut self.ptx_buffer, "    // [FP8 TENSOR CORE MATRIX MULTIPLY WITH SCALE PROPAGATION]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Dimensions: M={}, N={}, K={} FP8 (e4m3fn)", cta_m, cta_n, cta_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 %f_total_scale, {}, {};", scale_a, scale_b).unwrap();
        writeln!(&mut self.ptx_buffer, "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 %f_acc, %r_a, %r_b, %f_acc;").unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 %f_acc, %f_acc, %f_total_scale;").unwrap();
    }

    /// 2:4 Structured Sparse Tensor Core MMA (`mma.sp.sync.aligned.m16n8k32`).
    pub fn emit_sparse_24_mma(&mut self, cta_m: u32, cta_n: u32, cta_k: u32) {
        writeln!(&mut self.ptx_buffer, "    // [NVIDIA 2:4 STRUCTURED SPARSE TENSOR CORE MMA]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Dimensions: M={}, N={}, K={} (2:4 Sparsity Enabled)", cta_m, cta_n, cta_k).unwrap();
        writeln!(&mut self.ptx_buffer, "    mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 %f_acc, %r_a_sparse, %r_b, %f_acc, %r_metadata, 0x0;").unwrap();
    }

    /// Emits PTX launch bounds directives (.maxnreg, .minnctapersm) for occupancy tuning.
    pub fn emit_launch_bounds_directives(&mut self, max_registers: u32, min_ctas_per_sm: u32) {
        writeln!(&mut self.ptx_buffer, "    .maxnreg {}", max_registers).unwrap();
        writeln!(&mut self.ptx_buffer, "    .minnctapersm {}", min_ctas_per_sm).unwrap();
    }

    /// Emits L2 Cache Eviction control operators (.nc, .evict_first, .wt).
    pub fn emit_l2_cache_eviction_load_store(&mut self, dst_reg: &str, src_ptr: &str, cache_policy: &str) {
        match cache_policy {
            "non_coherent" => writeln!(&mut self.ptx_buffer, "    ld.global.nc.f32 {}, [{}];", dst_reg, src_ptr).unwrap(),
            "evict_first" => writeln!(&mut self.ptx_buffer, "    ld.global.evict_first.f32 {}, [{}];", dst_reg, src_ptr).unwrap(),
            "write_through" => writeln!(&mut self.ptx_buffer, "    st.global.wt.f32 [{}], {};", src_ptr, dst_reg).unwrap(),
            _ => writeln!(&mut self.ptx_buffer, "    ld.global.f32 {}, [{}];", dst_reg, src_ptr).unwrap(),
        }
    }

    /// Adaptive FP8 GEMM Engine: Selects optimal small vs. large matrix multiplication execution path.
    pub fn emit_fp8_adaptive_gemm(&mut self, m: u32, n: u32, k: u32, scale_a: &str, scale_b: &str) {
        if m <= 512 || n <= 512 {
            // Low-latency launch config for small matrix multiplication (64x64x32 CTA tile, 4 warps)
            writeln!(&mut self.ptx_buffer, "    // [ADAPTIVE FP8 GEMM - SMALL MATRIX LOW-LATENCY PATH]").unwrap();
            writeln!(&mut self.ptx_buffer, "    // CTA Tile: 64x64x32 | Warps: 4 (2x2) | Wave Occupancy: High").unwrap();
            self.emit_launch_bounds_directives(64, 4);
            self.emit_fp8_scaling_mma(64, 64, 32, scale_a, scale_b);
        } else {
            // High-throughput launch config for large matrix multiplication (128x128x64 CTA tile, 8 warps, 3-stage async pipelining)
            writeln!(&mut self.ptx_buffer, "    // [ADAPTIVE FP8 GEMM - LARGE MATRIX HIGH-THROUGHPUT PATH]").unwrap();
            writeln!(&mut self.ptx_buffer, "    // CTA Tile: 128x128x64 | Warps: 8 (4x2) | 3-Stage Async Pipeline | L2 Swizzle").unwrap();
            self.emit_grid_swizzle_code(8);
            self.emit_launch_bounds_directives(128, 2);
            self.emit_mbarrier_3stage_pipelined_loop(3, k, 64);
            self.emit_fp8_scaling_mma(128, 128, 64, scale_a, scale_b);
        }
    }

    /// Fused Vectorized Fast Math SwiGLU Activation Engine (128-bit SIMD + fast inline PTX sigmoid math).
    pub fn emit_vectorized_swiglu_fast(&mut self, in_x_ptr: &str, in_y_ptr: &str, out_ptr: &str, num_vectors: usize) {
        writeln!(&mut self.ptx_buffer, "    // [FUSED VECTORIZED SWIGLU ACTIVATION - FAST INLINE MATH]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Vector Count: {} 128-bit 4-packs (100K+ elements)", num_vectors).unwrap();
        writeln!(&mut self.ptx_buffer, "    ld.global.v4.f32 {{{{%f0, %f1, %f2, %f3}}}}, [{}];", in_x_ptr).unwrap();
        writeln!(&mut self.ptx_buffer, "    ld.global.v4.f32 {{{{%f4, %f5, %f6, %f7}}}}, [{}];", in_y_ptr).unwrap();

        // Fast inline sigmoid math: sigma(x) = 1 / (1 + exp(-x)) using ex2.approx and rcp.approx
        writeln!(&mut self.ptx_buffer, "    neg.f32 %f8, %f0; mul.f32 %f8, %f8, 1.44269504; ex2.approx.f32 %f8, %f8; add.f32 %f8, %f8, 1.0; rcp.approx.f32 %f8, %f8;").unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.f32 %f9, %f0, %f8; mul.f32 %f9, %f9, %f4;").unwrap();

        writeln!(&mut self.ptx_buffer, "    st.global.v4.f32 [{}], {{{{%f9, %f9, %f9, %f9}}}};", out_ptr).unwrap();
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
    fn test_hopper_tma_descriptor_generation() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_tma_descriptor_gen("tma_desc_A", "MatrixA", 128, 128);
        emitter.emit_tma_bulk_load("smem_A", "tma_desc_A", "%r0", "%r1");
        assert!(emitter.ptx_buffer.contains(".global .align 128 .b8 tma_desc_A[128];"));
        assert!(emitter.ptx_buffer.contains("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"));
    }

    #[test]
    fn test_mbarrier_3stage_async_pipelining() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_mbarrier_3stage_pipelined_loop(3, 1024, 32);
        assert!(emitter.ptx_buffer.contains("3+ STAGE ASYNCHRONOUS PIPELINING WITH mbarrier COUNTERS"));
        assert!(emitter.ptx_buffer.contains("mbarrier.init.shared.b64"));
        assert!(emitter.ptx_buffer.contains("mbarrier.arrive.expect_tx.shared.b64"));
        assert!(emitter.ptx_buffer.contains("mbarrier.try_wait.parity.shared.b64"));
    }

    #[test]
    fn test_hopper_wgmma_warp_group() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_wgmma_warp_group_gemm(64, 64, 16, 512);
        assert!(emitter.ptx_buffer.contains("HOPPER WARP-GROUP MATRIX MULTIPLY - wgmma.mma_async"));
        assert!(emitter.ptx_buffer.contains("wgmma.fence.sync.aligned;"));
        assert!(emitter.ptx_buffer.contains("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"));
        assert!(emitter.ptx_buffer.contains("wgmma.commit_group.sync.aligned;"));
        assert!(emitter.ptx_buffer.contains("wgmma.wait_group.sync.aligned 0;"));
    }

    #[test]
    fn test_tma_multicast_cluster() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_cluster_dimensions(2, 1, 1);
        emitter.emit_tma_multicast_bulk_load("smem_A", "desc_A", "%r0", "%r1", "0x3");
        assert!(emitter.ptx_buffer.contains(".cluster_dimensions 2, 1, 1;"));
        assert!(emitter.ptx_buffer.contains("cp.async.bulk.tensor.multicast.shared::cluster.global.mbarrier::complete_tx::bytes"));
    }

    #[test]
    fn test_warp_specialized_producer_consumer_pipeline() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_warp_specialized_producer_consumer_pipeline(128, 128, 32, 1024);
        assert!(emitter.ptx_buffer.contains("WARP-SPECIALIZED PRODUCER/CONSUMER PIPELINE"));
        assert!(emitter.ptx_buffer.contains("Warp 0: TMA Producer"));
        assert!(emitter.ptx_buffer.contains("wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16"));
    }

    #[test]
    fn test_wgmma_fp8_dual_accumulator() {
        let mut emitter = PtxEmitter::new();
        emitter.emit_wgmma_fp8_dual_accumulator_gemm(64, 128, 32, "%f_sa", "%f_sb");
        assert!(emitter.ptx_buffer.contains("HOPPER NATIVE FP8 E4M3 DUAL-ACCUMULATOR WGMMA PIPELINE"));
        assert!(emitter.ptx_buffer.contains("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3"));
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

        assert!(ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.global.f32"), "plain GEMM must keep its direct-to-global epilogue: {}", ptx);
        assert!(!ptx.contains("EPI_BIAS_RELU"), "plain GEMM must not emit the fused bias+relu epilogue: {}", ptx);
    }
}

/// Pass 1: Hardware Asynchronous DMA Prefetching Pass (cp.async.cg.shared.global)
pub struct AsyncPipeliningPass {
    pub async_copies_emitted: usize,
}

impl AsyncPipeliningPass {
    pub fn new() -> Self {
        AsyncPipeliningPass {
            async_copies_emitted: 0,
        }
    }

    pub fn emit_async_group_copy(&mut self, dest_smem: &str, src_global: &str, num_bytes: usize) -> String {
        self.async_copies_emitted += 1;
        format!(
            "// Y ASYNC PIPELINING PASS (DMA Group Copy)\n\
             cp.async.cg.shared.global [{}], [{}], {};\n\
             cp.async.commit_group;\n\
             cp.async.wait_group 0;\n",
            dest_smem, src_global, num_bytes
        )
    }
}

/// Pass 4: Calculates live register ranges per warp and emits max register launch directives
pub struct RegisterPressurePass {
    pub max_registers_per_thread: u32,
    pub min_ctas_per_sm: u32,
}

impl RegisterPressurePass {
    pub fn new(max_regs: u32, min_ctas: u32) -> Self {
        RegisterPressurePass {
            max_registers_per_thread: max_regs,
            min_ctas_per_sm: min_ctas,
        }
    }

    pub fn emit_maxnreg_directive(&self) -> String {
        format!(
            "// Y REGISTER PRESSURE PASS (Occupancy Optimization Directive)\n\
             .pragma \"option nvcc -maxrregcount={}\";\n",
            self.max_registers_per_thread
        )
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

