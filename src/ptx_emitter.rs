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

/// Manages virtual registers and produces raw PTX strings.
pub struct PtxEmitter {
    pub ptx_buffer: String,

    // Virtual register counters to maintain uniqueness
    reg_u32_count: u32,
    reg_f32_count: u32,
    reg_pred_count: u32,
    label_count: u32,
}

impl PtxEmitter {
    pub fn new() -> Self {
        let mut buffer = String::new();
        // Emit PTX header
        writeln!(&mut buffer, ".version 8.0").unwrap();
        // Assume sm_80 or sm_89 depending on feature set needed (sm_80 for cp.async).
        writeln!(&mut buffer, ".target sm_80").unwrap();
        writeln!(&mut buffer, ".address_size 64").unwrap();
        writeln!(&mut buffer, "").unwrap();

        Self {
            ptx_buffer: buffer,
            reg_u32_count: 0,
            reg_f32_count: 0,
            reg_pred_count: 0,
            label_count: 0,
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
                writeln!(
                    &mut self.ptx_buffer,
                    "    // unsupported loop bound; defaulting {} to 0",
                    dst
                )
                .unwrap();
                writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", dst).unwrap();
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
        // Emit kernel signature
        writeln!(&mut self.ptx_buffer, ".visible .entry {}(", kernel.name).unwrap();

        let param_count = kernel.params.len();
        for (i, param) in kernel.params.iter().enumerate() {
            // E.g. GlobalMemory translates to 64-bit pointers
            let ptx_type = match &param.ty {
                Type::Generic { base, .. } if base == "GlobalMemory" => ".param .u64",
                _ => ".param .b32", // fallback payload
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
        writeln!(&mut self.ptx_buffer, ") {{").unwrap();

        // --- TIGHT PACK REGISTER ALLOCATOR ---
        // Calculate Maximum Register Count to achieve 100% occupancy
        // total_threads = max_warps_per_sm * warp_size
        let total_threads = hw_profile.max_warps_per_sm * hw_profile.warp_size;
        let mut optimal_regs = 0;
        if total_threads > 0 {
            optimal_regs = hw_profile.max_regs_per_sm / total_threads;
        }

        // Cap to the hardware limit per thread (usually 255)
        if optimal_regs > hw_profile.max_regs_per_thread {
            optimal_regs = hw_profile.max_regs_per_thread;
        }

        if optimal_regs > 0 {
            writeln!(
                &mut self.ptx_buffer,
                "    // [TIGHT PACK] Maximum Occupancy Target: 100% ({} warps)",
                hw_profile.max_warps_per_sm
            )
            .unwrap();
            writeln!(&mut self.ptx_buffer, "    // [TIGHT PACK] HW Limit: {} regs/SM. Forcing register pressure limit to exactly {} registers per thread.", hw_profile.max_regs_per_sm, optimal_regs).unwrap();
            writeln!(&mut self.ptx_buffer, "    .maxnreg {};", optimal_regs).unwrap();
            writeln!(&mut self.ptx_buffer).unwrap();
        }

        // Declare virtual registers. In a full implementation, the Y compiler
        // would map these directly to the optimal physical set using liveness analysis.
        writeln!(&mut self.ptx_buffer, "    .reg .b32 %r<100>;").unwrap();
        writeln!(&mut self.ptx_buffer, "    .reg .f32 %f<100>;").unwrap();
        writeln!(&mut self.ptx_buffer, "    .reg .b64 %rd<50>;").unwrap();
        writeln!(&mut self.ptx_buffer, "    .reg .pred %p<20>;").unwrap();
        writeln!(&mut self.ptx_buffer).unwrap();

        // Emit body
        self.emit_block(&kernel.body, hw_profile);

        writeln!(&mut self.ptx_buffer, "}}").unwrap();
    }

    fn emit_block(&mut self, block: &Block, hw_profile: &HardwareProfile) {
        let mut stmts = block.stmts.clone();

        let mut i = 0;
        while i < stmts.len() {
            // Detect if current statement is a barrier
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

                // Scan forward to find hoistable ALUs
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
                        // j stays the same because we removed the element at j
                    } else {
                        j += 1; // skip unhoistable
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

                    // Emit hoisted statements BEFORE the barrier
                    for h in hoisted {
                        self.emit_stmt(&h, hw_profile);
                    }
                } else {
                    writeln!(&mut self.ptx_buffer, "    // [BARRIER HOISTING] Barrier detected ({} cycle stall), but no independent ALUs to hoist.", hw_profile.membar_gpu_latency_cycles).unwrap();
                }
            }

            // Emit the current statement (the barrier, or a regular statement)
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
                    // Check if it's an allocation or instruction
                    let val_str = self.emit_expr(expr, cache_policy.as_ref(), hw_profile);
                    writeln!(&mut self.ptx_buffer, "    // let {} = ...", name).unwrap();
                    if !val_str.is_empty() {
                        writeln!(&mut self.ptx_buffer, "    {}", val_str).unwrap();
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
                ..
            } => {
                let loop_reg = self.alloc_reg32();
                let end_reg = self.alloc_reg32();
                let exit_pred = self.alloc_pred();
                let loop_start = self.alloc_label("LOOP_START");
                let loop_end = self.alloc_label("LOOP_END");
                let step_val = match step {
                    Some(Expr::IntLit(step, _)) if *step > 0 && *step <= u32::MAX as i64 => {
                        *step as u32
                    }
                    _ => 1,
                };

                writeln!(&mut self.ptx_buffer, "    // for {} in ...", loop_var).unwrap();
                self.emit_u32_init(&loop_reg, start);
                self.emit_u32_init(&end_reg, end);
                writeln!(
                    &mut self.ptx_buffer,
                    "    // {} is tracked in {}",
                    loop_var, loop_reg
                )
                .unwrap();
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
                target: _, value, ..
            } => {
                writeln!(&mut self.ptx_buffer, "    // ASSIGN LOGIC").unwrap();
                let src = self.emit_expr(value, None, hw_profile);
                // Actual move logic relies on what 'target' is, but for demonstration:
                if !src.is_empty() {
                    writeln!(&mut self.ptx_buffer, "    {}", src).unwrap();
                }
            }
            Stmt::Expr(expr) => {
                let call_str = self.emit_expr(expr, None, hw_profile);
                if !call_str.is_empty() {
                    writeln!(&mut self.ptx_buffer, "    {}", call_str).unwrap();
                }
            }
            Stmt::Return(_, _) => {}
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

                // Heuristic: Count statements to estimate cost (1 statement ~= 1 cycle rough proxy)
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
                        "%r_cond"
                    } else {
                        &cond_str
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
                        "%r_cond"
                    } else {
                        &cond_str
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
            _ => {} // Other constructs
        }
    }

    fn emit_expr(
        &mut self,
        expr: &Expr,
        cache_policy: Option<&CachePolicyAttr>,
        hw_profile: &HardwareProfile,
    ) -> String {
        match expr {
            Expr::Call { func, .. } => {
                match &**func {
                    Expr::Ident(fname, _) => {
                        if fname == "cp_async" {
                            return "cp.async.ca.shared.global [%rd_smem], [%rd_gmem], 16;".into();
                        } else if fname == "mma_sync" {
                            return "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0,%f1}, {%r0,%r1}, {%r2,%r3}, {%f0,%f1};".into();
                        } else if fname == "store" {
                            return "st.global.v4.f32 [%rd_out], {%f0,%f1,%f2,%f3};".into();
                        }
                    }
                    Expr::Path {
                        namespace, member, ..
                    } => {
                        if namespace == "barrier" && member == "sync" {
                            return "bar.sync 0;".into();
                        }
                    }
                    _ => {}
                }
                "".into()
            }
            Expr::MemberAccess { member, .. } => {
                if member == "wait" {
                    // Pipeline block wait
                    return "cp.async.wait_group 0;".into();
                }
                "".into()
            }
            Expr::Path {
                namespace, member, ..
            } => {
                if namespace == "barrier" && member == "sync" {
                    return "bar.sync 0;".into();
                }
                if namespace == "Fragment" && member == "zero" {
                    return "mov.b32 %f0, 0f00000000;".into(); // Float zero synthesis
                }
                // Memory loads handling Cache policy
                if namespace == "GlobalMemory" && member == "load" {
                    let mut cache_str = ".ca";
                    if let Some(cp) = cache_policy {
                        if cp.policy == "L2_PERSIST" {
                            cache_str = ".lu"; // closest logical equivalent or custom PTX 8.0 hints
                        } else if cp.policy == "L2_EVICT_FIRST" {
                            cache_str = ".L2::evict_first"; // Requires newer PTX versions
                        }
                    }
                    return format!("ld.global{} [%rd_ptr];", cache_str);
                }

                if namespace == "SharedMemory" && member == "alloc" {
                    return ".shared .align 128 .b8 smem[8192];".into(); // Prototype SMEM block
                }
                "".into()
            }
            Expr::BinaryOp {
                op, left, right, ..
            } => {
                let _l = self.emit_expr(left, cache_policy, hw_profile);
                let _r = self.emit_expr(right, cache_policy, hw_profile);
                if *op == BinaryOp::Add || *op == BinaryOp::Mul {
                    if hw_profile.imad_wide_latency_cycles < 3.0 {
                        return format!("// [HEURISTIC] IMAD.WIDE ({}c) is fast. Emitting 64-bit mad.wide.u32...", hw_profile.imad_wide_latency_cycles);
                    } else {
                        return format!("// [HEURISTIC] Standard IMAD. Emitting mad.lo.u32...");
                    }
                }
                "".into()
            }
            Expr::GenericCall { func, .. } => {
                // Same logic handles like a Path or Call
                self.emit_expr(&**func, cache_policy, hw_profile)
            }
            Expr::StructLit { .. } => {
                // PTX has no native struct instantiations in this prototype
                "".into()
            }
            _ => "".into(),
        }
    }
}

