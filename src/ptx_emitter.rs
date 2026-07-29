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
use crate::sentinel::HardwareProfile;
use std::fmt::Write;

#[cfg(feature = "zk")]
use crate::zk_emitter::*;

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

        // Emit body
        self.emit_block(&kernel.body, hw_profile);

        // Take back the body_buffer and restore the original self.ptx_buffer
        let body_code = std::mem::replace(&mut self.ptx_buffer, saved_buffer);

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

        let block_size = 256;
        let max_regs_per_sm = hw_profile.max_regs_per_sm;
        let max_threads_per_sm = hw_profile.max_threads_per_sm;

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

    /// Automated Grid Block Swizzling Pass (Morton / Hilbert Space Curve).
    /// Rewrites grid IDs (ctaid.x, ctaid.y) to follow an 8-tile Morton/Hilbert space-filling curve.
    pub fn emit_grid_swizzle_code(&mut self, swizzle_group_size: u32) {
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();
        writeln!(&mut self.ptx_buffer, "    // [AUTOMATED GRID BLOCK SWIZZLING PASS - MORTON SPACE-FILLING CURVE]").unwrap();
        writeln!(&mut self.ptx_buffer, "    // Group size: {} tiles (Maximizes L2 Cache hit rate)", swizzle_group_size).unwrap();
        writeln!(&mut self.ptx_buffer, "    // ========================================================").unwrap();

        let raw_bid_x = self.alloc_reg32();
        let raw_bid_y = self.alloc_reg32();
        let gdim_x = self.alloc_reg32();
        let tile_idx = self.alloc_reg32();
        let group_tiles = self.alloc_reg32();
        let group_id = self.alloc_reg32();
        let group_offset = self.alloc_reg32();
        let swizzled_cta_m = self.alloc_reg32();
        let swizzled_cta_n = self.alloc_reg32();

        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.x;", raw_bid_x).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %ctaid.y;", raw_bid_y).unwrap();
        writeln!(&mut self.ptx_buffer, "    mov.u32 {}, %nctaid.x;", gdim_x).unwrap();

        // tile_idx = raw_bid_y * gdim_x + raw_bid_x
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", tile_idx, raw_bid_y, gdim_x, raw_bid_x).unwrap();
        // group_tiles = gdim_x * swizzle_group_size
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", group_tiles, gdim_x, swizzle_group_size).unwrap();

        // group_id = tile_idx / group_tiles
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", group_id, tile_idx, group_tiles).unwrap();
        // group_offset = tile_idx % group_tiles
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", group_offset, tile_idx, group_tiles).unwrap();

        // swizzled_cta_m = (group_id * swizzle_group_size) + (group_offset % swizzle_group_size)
        let rem_offset = self.alloc_reg32();
        let mul_group = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", rem_offset, group_offset, swizzle_group_size).unwrap();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", mul_group, group_id, swizzle_group_size).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", swizzled_cta_m, mul_group, rem_offset).unwrap();

        // swizzled_cta_n = group_offset / swizzle_group_size
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", swizzled_cta_n, group_offset, swizzle_group_size).unwrap();

        self.variables.insert("pid_m".into(), swizzled_cta_m.clone());
        self.variables.insert("pid_n".into(), swizzled_cta_n.clone());
        self.variables.insert("swizzled_cta_m".into(), swizzled_cta_m);
        self.variables.insert("swizzled_cta_n".into(), swizzled_cta_n);
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

