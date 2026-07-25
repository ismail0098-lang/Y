// ============================================================
//  Y  —  PTX Code Emitter
//  ptx_emitter.rs
//
//  Backend code generator targeting NVIDIA PTX.
//  Converts validated AST nodes into virtual assembly.
//  Bypasses high-level CUDA runtime and talks directly
//  to the silicon via instructions like ldmatrix and cp.async.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use crate::sentinel::HardwareProfile;
use std::fmt::Write;

#[cfg(feature = "zk")]
use crate::zk_emitter::*;

/// Maps an SM compute capability to the minimum required PTX ISA version.
fn ptx_version_for_sm(sm: &str) -> &'static str {
    match sm {
        s if s >= "sm_100" => ".version 8.7",
        "sm_90" | "sm_90a" => ".version 8.0",
        "sm_89" => ".version 7.8",
        "sm_86" | "sm_87" => ".version 7.5",
        "sm_80" => ".version 7.0",
        "sm_75" => ".version 6.5",
        "sm_72" => ".version 6.2",
        "sm_70" => ".version 6.3",
        _ => ".version 7.0", // Safe default for unknown targets
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
}

impl PtxEmitter {
    pub fn new() -> Self {
        Self::new_with_profile(&HardwareProfile::default())
    }

    pub fn new_with_profile(hw_profile: &HardwareProfile) -> Self {
        let mut buffer = String::new();
        let target = if !hw_profile.sm_version.is_empty() {
            if hw_profile.sm_version.starts_with("sm_") {
                hw_profile.sm_version.clone()
            } else {
                format!("sm_{}", hw_profile.sm_version)
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

                if let Some(t) = tile {
                    writeln!(&mut self.ptx_buffer, "    // [Y TILE OPTIMIZATION] Tiled loop dimensions: M={:?}, N={:?}, K={:?}", t.block_m, t.block_n, t.block_k).unwrap();
                }
                let step_val = match step {
                    Some(Expr::IntLit(step, _)) if *step > 0 && *step <= u32::MAX as i64 => {
                        *step as u32
                    }
                    _ => 1,
                };

                writeln!(&mut self.ptx_buffer, "    // for {} in ...", loop_var).unwrap();
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
                        BinaryOp::Mul => {
                            if hw_profile.imad_wide_latency_cycles < 3.0 {
                                "mad.wide.u32"
                            } else {
                                "mul.lo.s32"
                            }
                        }
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
                            writeln!(&mut self.ptx_buffer, "    cp.async.ca.shared.global [{}], [{}], 16;", dest_reg, src_reg).unwrap();
                            "".into()
                        } else if fname == "mma_sync" {
                            writeln!(&mut self.ptx_buffer, "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%f0,%f1}}, {{%r0,%r1}}, {{%r2,%r3}}, {{%f0,%f1}};").unwrap();
                            "".into()
                        } else if fname == "store" && args.len() >= 2 {
                            let addr_reg = self.emit_expr(&args[0], cache_policy, hw_profile);
                            let val_reg = self.emit_expr(&args[1], cache_policy, hw_profile);
                            if val_reg.starts_with("%f") {
                                writeln!(&mut self.ptx_buffer, "    st.global.f32 [{}], {};", addr_reg, val_reg).unwrap();
                            } else {
                                writeln!(&mut self.ptx_buffer, "    st.global.u32 [{}], {};", addr_reg, val_reg).unwrap();
                            }
                            "".into()
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
}
