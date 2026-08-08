// ============================================================
//  Y  —  Semantic Type Checker
//  type_checker.rs
//
//  The core brain of Y's safety guarantees.
//  Traverses AST, enforces Fragment roles (A vs B vs C),
//  manages linear memory obligations, and runs the
//  0-Bank-Conflict math prover.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use crate::bank_conflict::{BankConflictProver, SmemLayout as ProverLayout, SwizzlePattern};
use crate::linear_tracker::LinearTracker;
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::io::Write;

thread_local! {
    pub static SAFE_INDICES: std::cell::RefCell<std::collections::HashSet<(usize, usize)>> = std::cell::RefCell::new(std::collections::HashSet::new());
    pub static INDEX_ARRAY_SIZES: std::cell::RefCell<std::collections::HashMap<(usize, usize), usize>> = std::cell::RefCell::new(std::collections::HashMap::new());
    pub static INDEX_SWIZZLES: std::cell::RefCell<std::collections::HashMap<(usize, usize), SwizzlePattern>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub min: i64,
    pub max: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticType {
    Primitive(String),
    Fragment {
        op: String,
        role: String,
        dtype: String,
    },
    SharedMemoryTile {
        rows: u32,
        cols: u32,
        swizzle: Option<SwizzlePattern>,
    },
    GlobalMemory(String),
    Vector(Box<SemanticType>, String), // Tuple of inner type and allocator
    Array {
        element: Box<SemanticType>,
        size: usize,
    },
    BlockTile {
        element: Box<SemanticType>,
        size: usize,
    },
    TransferObligation,
    Pipeline,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnconstrainedReason {
    HintOutput(String),
    UnconstrainedInput(String),
    Merged(Vec<UnconstrainedReason>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintState {
    Constrained,
    TaintedUnconstrained {
        origins: Vec<Span>,
        reasons: Vec<UnconstrainedReason>,
    },
    DeferredObligation {
        origins: Vec<Span>,
        reasons: Vec<UnconstrainedReason>,
        override_span: Span,
    },
    Verified {
        origins: Vec<Span>,
        verified_span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignalConstraintInfo {
    pub name: String,
    pub state: ConstraintState,
    pub declared_span: Span,
}

/// Per-variable entry in a unified scope frame, combining type, interval,
/// explicit-bound flag, and constraint info into a single cache-friendly record.
pub struct SymbolEntry {
    pub ty: SemanticType,
    pub interval: Option<Interval>,
    pub is_explicitly_bounded: bool,
    pub constraint_info: Option<SignalConstraintInfo>,
}

/// A single scope frame that replaces the former four parallel scope stacks.
pub struct ScopeFrame {
    pub symbols: HashMap<String, SymbolEntry>,
}

impl ScopeFrame {
    fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }
}

pub struct TypeChecker {
    // Unified scope stack: each frame holds all per-variable data
    scopes: Vec<ScopeFrame>,
    pub linear_tracker: LinearTracker,
    pub errors: Vec<String>,
    pub in_unsafe: bool,
    allow_transfer_use: usize,
    current_return_type: Option<SemanticType>,
    functions: HashMap<String, Vec<SemanticType>>,
    structs: HashMap<String, HashMap<String, SemanticType>>,

    // Static Under-Constrained Analyzer (@zk_safe) fields
    pub zk_safe_stack: Vec<bool>,
    pub zk_allow_unconstrained_stack: Vec<bool>,
}

fn reset_thread_locals() {
    SAFE_INDICES.with(|s| s.borrow_mut().clear());
    INDEX_ARRAY_SIZES.with(|s| s.borrow_mut().clear());
    INDEX_SWIZZLES.with(|s| s.borrow_mut().clear());
}

impl TypeChecker {
    pub fn new() -> Self {
        reset_thread_locals();
        Self {
            scopes: vec![ScopeFrame::new()],
            linear_tracker: LinearTracker::new(),
            errors: Vec::new(),
            in_unsafe: false,
            allow_transfer_use: 0,
            current_return_type: None,
            functions: HashMap::new(),
            structs: HashMap::new(),
            zk_safe_stack: vec![false],
            zk_allow_unconstrained_stack: vec![false],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame::new());
        self.linear_tracker.push_scope();
    }

    pub fn pop_scope(&mut self) {
        self.check_scope_unconstrained_signals();
        self.linear_tracker.pop_scope();
        self.scopes.pop();
    }

    pub fn is_zk_safe_active(&self) -> bool {
        self.zk_safe_stack.iter().rev().copied().any(|b| b)
    }

    pub fn is_zk_allow_unconstrained_active(&self) -> bool {
        self.zk_allow_unconstrained_stack.iter().rev().copied().any(|b| b)
    }

    pub fn set_signal_constraint(&mut self, name: String, state: ConstraintState, span: Span) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.constraint_info = Some(SignalConstraintInfo {
                    name,
                    state,
                    declared_span: span,
                });
            } else {
                frame.symbols.insert(
                    name.clone(),
                    SymbolEntry {
                        ty: SemanticType::Unknown,
                        interval: None,
                        is_explicitly_bounded: false,
                        constraint_info: Some(SignalConstraintInfo {
                            name,
                            state,
                            declared_span: span,
                        }),
                    },
                );
            }
        }
    }

    pub fn lookup_signal_constraint(&self, name: &str) -> Option<&SignalConstraintInfo> {
        for frame in self.scopes.iter().rev() {
            if let Some(entry) = frame.symbols.get(name) {
                if let Some(ref info) = entry.constraint_info {
                    return Some(info);
                }
            }
        }
        None
    }

    pub fn update_signal_constraint_state(&mut self, name: &str, new_state: ConstraintState) {
        for frame in self.scopes.iter_mut().rev() {
            if let Some(entry) = frame.symbols.get_mut(name) {
                if let Some(ref mut info) = entry.constraint_info {
                    info.state = new_state;
                    return;
                }
            }
        }
    }

    pub fn eval_expr_constraint_state(&self, expr: &Expr) -> ConstraintState {
        match expr {
            Expr::Ident(name, _) => {
                if let Some(info) = self.lookup_signal_constraint(name) {
                    info.state.clone()
                } else {
                    ConstraintState::Constrained
                }
            }
            Expr::IntLit(..) | Expr::FloatLit(..) | Expr::StringLit(..) | Expr::CharLit(..) => {
                ConstraintState::Constrained
            }
            Expr::BinaryOp { left, op, right, .. } => {
                if matches!(op, BinaryOp::Eq) {
                    return ConstraintState::Constrained;
                }
                let s_left = self.eval_expr_constraint_state(left);
                let s_right = self.eval_expr_constraint_state(right);

                match (s_left, s_right) {
                    (
                        ConstraintState::TaintedUnconstrained { origins: o1, reasons: r1 },
                        ConstraintState::TaintedUnconstrained { origins: o2, reasons: r2 },
                    ) => {
                        let mut merged_origins = o1;
                        for span in o2 {
                            if !merged_origins.contains(&span) {
                                merged_origins.push(span);
                            }
                        }
                        let mut merged_reasons = r1;
                        for r in r2 {
                            if !merged_reasons.contains(&r) {
                                merged_reasons.push(r);
                            }
                        }
                        ConstraintState::TaintedUnconstrained {
                            origins: merged_origins,
                            reasons: merged_reasons,
                        }
                    }
                    (ConstraintState::TaintedUnconstrained { origins, reasons }, _)
                    | (_, ConstraintState::TaintedUnconstrained { origins, reasons }) => {
                        ConstraintState::TaintedUnconstrained { origins, reasons }
                    }
                    (
                        ConstraintState::DeferredObligation { origins: o1, reasons: r1, override_span },
                        ConstraintState::DeferredObligation { origins: o2, reasons: r2, .. },
                    ) => {
                        let mut merged_origins = o1;
                        for span in o2 {
                            if !merged_origins.contains(&span) {
                                merged_origins.push(span);
                            }
                        }
                        let mut merged_reasons = r1;
                        for r in r2 {
                            if !merged_reasons.contains(&r) {
                                merged_reasons.push(r);
                            }
                        }
                        ConstraintState::DeferredObligation {
                            origins: merged_origins,
                            reasons: merged_reasons,
                            override_span,
                        }
                    }
                    (ConstraintState::DeferredObligation { origins, reasons, override_span }, _)
                    | (_, ConstraintState::DeferredObligation { origins, reasons, override_span }) => {
                        ConstraintState::DeferredObligation { origins, reasons, override_span }
                    }
                    (ConstraintState::Verified { origins, verified_span }, _)
                    | (_, ConstraintState::Verified { origins, verified_span }) => {
                        ConstraintState::Verified { origins, verified_span }
                    }
                    _ => ConstraintState::Constrained,
                }
            }
            Expr::UnaryOp { operand, .. } => {
                self.eval_expr_constraint_state(operand)
            }
            Expr::Call { args, .. } => {
                for arg in args {
                    let st = self.eval_expr_constraint_state(arg);
                    if matches!(st, ConstraintState::TaintedUnconstrained { .. } | ConstraintState::DeferredObligation { .. }) {
                        return st;
                    }
                }
                ConstraintState::Constrained
            }
            Expr::Index { base, index, .. } => {
                let sb = self.eval_expr_constraint_state(base);
                if matches!(sb, ConstraintState::TaintedUnconstrained { .. } | ConstraintState::DeferredObligation { .. }) {
                    return sb;
                }
                let si = self.eval_expr_constraint_state(index);
                if matches!(si, ConstraintState::TaintedUnconstrained { .. } | ConstraintState::DeferredObligation { .. }) {
                    return si;
                }
                ConstraintState::Constrained
            }
            _ => ConstraintState::Constrained,
        }
    }

    pub fn check_verification_transition(&mut self, left: &Expr, right: &Expr, eq_span: &Span) {
        let left_state = self.eval_expr_constraint_state(left);
        let right_state = self.eval_expr_constraint_state(right);

        if let Expr::Ident(name_l, _) = left {
            if let ConstraintState::TaintedUnconstrained { ref origins, .. }
                | ConstraintState::DeferredObligation { ref origins, .. } = left_state
            {
                if matches!(right_state, ConstraintState::Constrained | ConstraintState::Verified { .. }) {
                    self.update_signal_constraint_state(
                        name_l,
                        ConstraintState::Verified {
                            origins: origins.clone(),
                            verified_span: eq_span.clone(),
                        },
                    );
                }
            }
        }
        if let Expr::Ident(name_r, _) = right {
            if let ConstraintState::TaintedUnconstrained { ref origins, .. }
                | ConstraintState::DeferredObligation { ref origins, .. } = right_state
            {
                if matches!(left_state, ConstraintState::Constrained | ConstraintState::Verified { .. }) {
                    self.update_signal_constraint_state(
                        name_r,
                        ConstraintState::Verified {
                            origins: origins.clone(),
                            verified_span: eq_span.clone(),
                        },
                    );
                }
            }
        }
    }

    fn check_scope_unconstrained_signals(&mut self) {
        let is_allow = self.is_zk_allow_unconstrained_active();
        let is_safe = self.is_zk_safe_active();
        let is_top_level = self.scopes.len() <= 2;

        if let Some(current_frame) = self.scopes.last_mut() {
            for (var_name, entry) in current_frame.symbols.iter_mut() {
                let info = match entry.constraint_info.as_mut() {
                    Some(info) => info,
                    None => continue,
                };

                if is_allow {
                    if let ConstraintState::TaintedUnconstrained { origins, reasons } = &info.state {
                        info.state = ConstraintState::DeferredObligation {
                            origins: origins.clone(),
                            reasons: reasons.clone(),
                            override_span: info.declared_span.clone(),
                        };
                        continue;
                    }
                }

                if is_safe {
                    match &info.state {
                        ConstraintState::TaintedUnconstrained { origins, .. } => {
                            let escape_span = &info.declared_span;
                            let origin_span = origins.first().unwrap_or(escape_span);
                            let err_msg = format!(
                                "error[Z0042]: under-constrained signal `{}` detected in @zk_safe context\n  --> line {}, col {}: signal escapes scope unconstrained\n  |\nnote: signal originated from @hint block here\n  --> line {}, col {}: unconstrained witness defined here\n  |\nhelp: add a constraint assertion (e.g., assert({} == expected)) to verify the witness.",
                                var_name, escape_span.line, escape_span.col, origin_span.line, origin_span.col, var_name
                            );
                            self.errors.push(err_msg);
                        }
                        ConstraintState::DeferredObligation { origins, override_span, .. } if is_top_level => {
                            let escape_span = &info.declared_span;
                            let origin_span = origins.first().unwrap_or(escape_span);
                            let err_msg = format!(
                                "error[Z0042]: deferred unconstrained signal `{}` allowed via @zk_allow_unconstrained escaped top-level program boundary unverified\n  --> line {}, col {}: signal reaches circuit output unconstrained\n  |\nnote: deferred override applied here\n  --> line {}, col {}: @zk_allow_unconstrained override\n  |\nnote: signal originated from @hint block here\n  --> line {}, col {}: unconstrained witness defined here\n  |\nhelp: add a constraint assertion to verify the deferred witness.",
                                var_name, escape_span.line, escape_span.col, override_span.line, override_span.col, origin_span.line, origin_span.col
                            );
                            self.errors.push(err_msg);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn insert_interval(&mut self, name: String, interval: Interval) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.interval = Some(interval);
            } else {
                frame.symbols.insert(name, SymbolEntry {
                    ty: SemanticType::Unknown,
                    interval: Some(interval),
                    is_explicitly_bounded: false,
                    constraint_info: None,
                });
            }
        }
    }

    fn find_var_scope_index(&self, name: &str) -> Option<usize> {
        for (idx, frame) in self.scopes.iter().enumerate().rev() {
            if frame.symbols.contains_key(name) {
                return Some(idx);
            }
        }
        None
    }

    fn is_explicitly_bounded(&self, name: &str) -> bool {
        if let Some(idx) = self.find_var_scope_index(name) {
            if let Some(frame) = self.scopes.get(idx) {
                if let Some(entry) = frame.symbols.get(name) {
                    return entry.is_explicitly_bounded;
                }
            }
        }
        false
    }

    fn mark_explicitly_bounded(&mut self, name: String) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.is_explicitly_bounded = true;
            }
        }
    }

    fn update_interval(&mut self, name: &str, interval: Option<Interval>) {
        let target_idx = self.find_var_scope_index(name).unwrap_or_else(|| {
            self.scopes.len().saturating_sub(1)
        });
        if let Some(frame) = self.scopes.get_mut(target_idx) {
            if let Some(entry) = frame.symbols.get_mut(name) {
                entry.interval = interval;
            } else if let Some(inv) = interval {
                frame.symbols.insert(name.to_string(), SymbolEntry {
                    ty: SemanticType::Unknown,
                    interval: Some(inv),
                    is_explicitly_bounded: false,
                    constraint_info: None,
                });
            }
        }
    }

    fn lookup_interval(&self, name: &str) -> Option<&Interval> {
        if let Some(idx) = self.find_var_scope_index(name) {
            if let Some(frame) = self.scopes.get(idx) {
                if let Some(entry) = frame.symbols.get(name) {
                    return entry.interval.as_ref();
                }
            }
        }
        None
    }

    fn eval_interval(&self, expr: &Expr) -> Option<Interval> {
        match expr {
            Expr::IntLit(val, _) => Some(Interval { min: *val, max: *val }),
            Expr::Ident(name, _) => self.lookup_interval(name).cloned(),
            Expr::BinaryOp { left, op, right, .. } => {
                let lhs = self.eval_interval(left)?;
                let rhs = self.eval_interval(right)?;
                match op {
                    BinaryOp::Add => Some(Interval {
                        min: lhs.min.saturating_add(rhs.min),
                        max: lhs.max.saturating_add(rhs.max),
                    }),
                    BinaryOp::Sub => Some(Interval {
                        min: lhs.min.saturating_sub(rhs.max),
                        max: lhs.max.saturating_sub(rhs.min),
                    }),
                    BinaryOp::Mul => {
                        let candidates = [
                            lhs.min.saturating_mul(rhs.min),
                            lhs.min.saturating_mul(rhs.max),
                            lhs.max.saturating_mul(rhs.min),
                            lhs.max.saturating_mul(rhs.max),
                        ];
                        Some(Interval {
                            min: *candidates.iter().min().unwrap(),
                            max: *candidates.iter().max().unwrap(),
                        })
                    }
                    BinaryOp::Div => {
                        if rhs.min <= 0 && rhs.max >= 0 {
                            None
                        } else {
                            let candidates = [
                                lhs.min.saturating_div(rhs.min),
                                lhs.min.saturating_div(rhs.max),
                                lhs.max.saturating_div(rhs.min),
                                lhs.max.saturating_div(rhs.max),
                            ];
                            Some(Interval {
                                min: *candidates.iter().min().unwrap(),
                                max: *candidates.iter().max().unwrap(),
                            })
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn insert_var(&mut self, name: String, ty: SemanticType) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.ty = ty;
            } else {
                frame.symbols.insert(name, SymbolEntry {
                    ty,
                    interval: None,
                    is_explicitly_bounded: false,
                    constraint_info: None,
                });
            }
        }
    }

    fn lookup_var(&self, name: &str) -> Option<&SemanticType> {
        for frame in self.scopes.iter().rev() {
            if let Some(entry) = frame.symbols.get(name) {
                if entry.ty != SemanticType::Unknown {
                    return Some(&entry.ty);
                }
            }
        }
        None
    }

    fn check_expr_allowing_transfer_use(&mut self, expr: &Expr) -> SemanticType {
        self.allow_transfer_use += 1;
        let ty = self.check_expr(expr);
        self.allow_transfer_use -= 1;
        ty
    }

    fn reject_transfer_escape(&mut self, ty: &SemanticType, span: &Span, context: &str) {
        if *ty == SemanticType::TransferObligation {
            self.errors.push(format!(
                "Line {}: Transfer obligations are linear and may only be consumed by `pipe.wait(...)`, not {}.",
                span.line, context
            ));
        }
    }

    fn root_ident(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident(name, _) => Some(name.clone()),
            Expr::Index { base, .. } => Self::root_ident(base),
            Expr::MemberAccess { base, .. } => Self::root_ident(base),
            _ => None,
        }
    }

    fn collect_assigned_vars_in_block(&self, block: &Block, vars: &mut std::collections::HashSet<String>) {
        for stmt in &block.stmts {
            self.collect_assigned_vars_in_stmt(stmt, vars);
        }
    }

    fn collect_assigned_vars_in_stmt(&self, stmt: &Stmt, vars: &mut std::collections::HashSet<String>) {
        match stmt {
            Stmt::Assign { target, .. } | Stmt::CompoundAssign { target, .. } => {
                if let Some(name) = Self::root_ident(target) {
                    vars.insert(name);
                }
            }
            Stmt::For { body, .. } => {
                self.collect_assigned_vars_in_block(body, vars);
            }
            Stmt::While { body, .. } => {
                self.collect_assigned_vars_in_block(body, vars);
            }
            Stmt::If { then_block, else_block, .. } => {
                self.collect_assigned_vars_in_block(then_block, vars);
                if let Some(eb) = else_block {
                    self.collect_assigned_vars_in_block(eb, vars);
                }
            }
            Stmt::Chisel(block, _) | Stmt::SafeBlock(block, _) | Stmt::GhostBlock(block, _) | Stmt::HintBlock { body: block, .. } => {
                self.collect_assigned_vars_in_block(block, vars);
            }
            Stmt::ClockDomainBlock { body, .. } => {
                self.collect_assigned_vars_in_block(body, vars);
            }
            Stmt::CompileTimeAssert { .. } => {}
            Stmt::Expr(expr) => {
                self.collect_assigned_vars_in_expr(expr, vars);
            }
            Stmt::Return(Some(expr), _) => {
                self.collect_assigned_vars_in_expr(expr, vars);
            }
            _ => {}
        }
    }

    fn collect_assigned_vars_in_expr(&self, expr: &Expr, vars: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::BlockExpr(block, _) => {
                self.collect_assigned_vars_in_block(block, vars);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_assigned_vars_in_expr(left, vars);
                self.collect_assigned_vars_in_expr(right, vars);
            }
            Expr::Call { func, args, .. } => {
                self.collect_assigned_vars_in_expr(func, vars);
                for arg in args {
                    self.collect_assigned_vars_in_expr(arg, vars);
                }
            }
            Expr::GenericCall { func, args, .. } => {
                self.collect_assigned_vars_in_expr(func, vars);
                for arg in args {
                    self.collect_assigned_vars_in_expr(arg, vars);
                }
            }
            Expr::Index { base, index, .. } => {
                self.collect_assigned_vars_in_expr(base, vars);
                self.collect_assigned_vars_in_expr(index, vars);
            }
            Expr::MemberAccess { base, .. } => {
                self.collect_assigned_vars_in_expr(base, vars);
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_assigned_vars_in_expr(operand, vars);
            }
            Expr::StructLit { fields, .. } => {
                for (_, f_expr) in fields {
                    self.collect_assigned_vars_in_expr(f_expr, vars);
                }
            }
            _ => {}
        }
    }

    fn transfer_destination_from_expr(expr: &Expr) -> Option<String> {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident(fname, _) = &**func {
                if fname == "cp_async" && args.len() >= 2 {
                    return Self::root_ident(&args[1]);
                }
            }
        }
        None
    }

    fn require_destination_ready(&mut self, expr: &Expr, span: &Span) {
        if let Some(name) = Self::root_ident(expr) {
            self.linear_tracker
                .require_destination_ready(&name, span.clone());
        }
    }

    fn check_wait_call(&mut self, base: &Expr, args: &[Expr], span: &Span) -> SemanticType {
        let base_ty = self.check_expr(base);
        self.reject_transfer_escape(&base_ty, span, "as the receiver of a method call");

        if args.is_empty() {
            self.errors.push(format!(
                "Line {}: `pipe.wait(...)` requires at least one Transfer obligation.",
                span.line
            ));
            return SemanticType::Unknown;
        }

        for arg in args {
            let arg_ty = self.check_expr_allowing_transfer_use(arg);
            if arg_ty != SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: `pipe.wait(...)` expects Transfer obligations as arguments.",
                    span.line
                ));
                continue;
            }

            if let Expr::Ident(var_name, _) = arg {
                if !self.linear_tracker.is_tracked_obligation(var_name) {
                    self.errors.push(format!(
                        "Line {}: `{}` is not a tracked Transfer obligation in this scope.",
                        span.line, var_name
                    ));
                    continue;
                }
                self.linear_tracker
                    .consume_obligation(var_name, span.clone());
            } else {
                self.errors.push(format!(
                    "Line {}: `pipe.wait(...)` requires named Transfer bindings so the obligation can be consumed exactly once.",
                    span.line
                ));
            }
        }

        SemanticType::Unknown
    }

    fn check_uniformity(&mut self, expr: &Expr) {
        // Uniformity analysis: fail if the expression relies on thread-local IDs
        let mut is_uniform = true;
        
        // Very basic prototype check: walk the expression and look for known thread-local variables
        // like threadIdx.x, blockDim.x, blockIdx.x, or memory loads that aren't broadcast.
        // For this bootstrap version, we will just check if any Ident contains "threadIdx".
        fn walk_expr(e: &Expr, is_u: &mut bool) {
            match e {
                Expr::Ident(name, _) => {
                    if name.contains("threadIdx") || name.contains("laneId") {
                        *is_u = false;
                    }
                }
                Expr::BinaryOp { left, right, .. } => {
                    walk_expr(left, is_u);
                    walk_expr(right, is_u);
                }
                Expr::UnaryOp { operand, .. } => {
                    walk_expr(operand, is_u);
                }
                Expr::Call { args, .. } => {
                    for arg in args {
                        walk_expr(arg, is_u);
                    }
                }
                Expr::MemberAccess { base, .. } => {
                    walk_expr(base, is_u);
                }
                Expr::Index { base, index, .. } => {
                    // Indexing into a potentially non-uniform array is divergent
                    // unless we prove the array contains uniform data. For now, mark unsafe indexing.
                    walk_expr(base, is_u);
                    walk_expr(index, is_u);
                }
                _ => {}
            }
        }
        
        walk_expr(expr, &mut is_uniform);
        
        if !is_uniform {
            self.errors.push(format!(
                "Line {}: Hardware Constraint Violation: Branch expression is not guaranteed to be uniform. Warp divergence detected.",
                expr.span().line
            ));
        }
    }

    // ── AST Traversal ───────────────────────────────────────

    pub fn check_program(&mut self, prog: &Program) {
        SAFE_INDICES.with(|set| {
            set.borrow_mut().clear();
        });
        INDEX_ARRAY_SIZES.with(|map| {
            map.borrow_mut().clear();
        });
        INDEX_SWIZZLES.with(|map| {
            map.borrow_mut().clear();
        });
        // Collect function signatures first
        for item in &prog.items {
            self.collect_signatures_item(item);
        }

        for item in &prog.items {
            self.check_item(item);
        }
    }

    fn collect_signatures_item(&mut self, item: &Item) {
        match item {
            Item::Func(f) => {
                let mut params = Vec::new();
                for p in &f.params {
                    params.push(self.resolve_type(&p.ty));
                }
                self.functions.insert(f.name.clone(), params);
            }
            Item::Impl(imp) => {
                for f in &imp.methods {
                    let mut params = Vec::new();
                    for p in &f.params {
                        params.push(self.resolve_type(&p.ty));
                    }
                    self.functions
                        .insert(format!("{}_{}", imp.target_type, f.name), params);
                }
            }
            Item::Const(c) => {
                let resolved = self.resolve_type(&c.ty);
                self.insert_var(c.name.clone(), resolved);
            }
            Item::Struct(s) => {
                let mut fields = HashMap::new();
                for f in &s.fields {
                    fields.insert(f.name.clone(), self.resolve_type(&f.ty));
                }
                self.structs.insert(s.name.clone(), fields);
            }
            Item::Module(m) => {
                for inner_item in &m.items {
                    self.collect_signatures_item(inner_item);
                }
            }
            _ => {}
        }
    }

    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Kernel(k) => self.check_kernel(k),
            Item::Func(f) => self.check_func(f),
            Item::Impl(imp) => {
                for f in &imp.methods {
                    self.check_func(f);
                }
            }
            Item::Module(m) => {
                self.zk_safe_stack.push(m.is_zk_safe);
                self.zk_allow_unconstrained_stack.push(m.is_zk_allow_unconstrained);
                for inner_item in &m.items {
                    self.check_item(inner_item);
                }
                self.zk_allow_unconstrained_stack.pop();
                self.zk_safe_stack.pop();
            }
            _ => {}
        }
    }

    fn check_kernel(&mut self, kernel: &KernelDecl) {
        self.push_scope();

        // Register params
        for param in &kernel.params {
            let sty = self.resolve_type(&param.ty);
            if sty == SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: Kernel parameters cannot have Transfer type. Transfer obligations must be created and discharged within the kernel body.",
                    param.span.line
                ));
            }
            self.insert_var(param.name.clone(), sty);
            self.set_signal_constraint(param.name.clone(), ConstraintState::Constrained, param.span.clone());
        }

        self.check_block(&kernel.body);

        self.verify_kernel_coherence(kernel);
        self.verify_tile_gemm_kernel(kernel);

        self.pop_scope();
    }

    /// Validates a kernel-level `@tile(M, N, K)` directive (see
    /// `KernelDecl::tile`'s doc comment) - promotes it from
    /// parseable-but-unchecked to a real, enforced precondition before
    /// `ptx_emitter` trusts it to dispatch to tile-aware Tensor Core GEMM
    /// codegen instead of the normal generic per-statement lowering. Runs
    /// regardless of target backend, like the rest of type_checker, but only
    /// has any effect on kernels that opt in by writing `@tile(...)` before
    /// `kernel` - kernels without it are completely untouched.
    fn verify_tile_gemm_kernel(&mut self, kernel: &KernelDecl) {
        let tile = match &kernel.tile {
            Some(t) => t,
            None => return,
        };

        fn as_positive_i64(e: &Expr) -> Option<i64> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => Some(*v),
                _ => None,
            }
        }

        if as_positive_i64(&tile.block_m).is_none() || as_positive_i64(&tile.block_n).is_none() {
            self.errors.push(format!(
                "Line {}: Kernel-level @tile(M, N, K) on `{}` requires M and N to be positive integer literals (the compile-time GEMM problem size this kernel is specialized for).",
                tile.span.line, kernel.name
            ));
        }
        if tile.block_k.as_deref().and_then(as_positive_i64).is_none() {
            self.errors.push(format!(
                "Line {}: Kernel-level @tile(M, N, K) on `{}` requires K (the third argument) - unlike the loop-scoped use of @tile, K is not optional here, and must be a positive integer literal.",
                tile.span.line, kernel.name
            ));
        }

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(
                            args.as_slice(),
                            [GenericArg::Type(Type::Primitive(p, _))] if p == elem
                        )
            )
        }
        // A 5-parameter shape (A, B: GlobalMemory<F32>, scale_a, scale_b:
        // F32 scalar, C: GlobalMemory<F32>) is accepted separately from the
        // 3/4-param F16 shapes below: A/B are quantized to e4m3 on the fly
        // (fused - see ptx_emitter::emit_fp8_gemm_kernel's doc comment) via
        // mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32 (Ada/sm_89-compatible,
        // unlike the Hopper-only WGMMA path), scale_a/scale_b are the
        // per-tensor dequant scales (typically amax/448 - the caller's/
        // launcher's responsibility to compute, not this kernel's), applied
        // to the f32 accumulator in the epilogue. Checked positionally, told
        // apart from the F16 shapes purely by param count (5) - see
        // `ptx_emitter::tile_gemm_fp8_operands`'s doc comment, which this
        // must never disagree with about which shape a given kernel is.
        if kernel.params.len() == 5 {
            fn is_scalar_f32(ty: &Type) -> bool {
                matches!(ty, Type::Primitive(p, _) if p == "F32")
            }
            let expected = ["GlobalMemory<F32>", "GlobalMemory<F32>", "F32", "F32", "GlobalMemory<F32>"];
            let checks: [bool; 5] = [
                is_global_memory_of(&kernel.params[0].ty, "F32"),
                is_global_memory_of(&kernel.params[1].ty, "F32"),
                is_scalar_f32(&kernel.params[2].ty),
                is_scalar_f32(&kernel.params[3].ty),
                is_global_memory_of(&kernel.params[4].ty, "F32"),
            ];
            let bad_params: Vec<String> = kernel
                .params
                .iter()
                .zip(checks.iter())
                .enumerate()
                .filter(|(_, (_, ok))| !**ok)
                .map(|(i, (p, _))| format!("{} (expected {})", p.name, expected[i]))
                .collect();
            if !bad_params.is_empty() {
                self.errors.push(format!(
                    "Line {}: Kernel-level @tile(M, N, K) on `{}` with 5 parameters requires (A, B: GlobalMemory<F32>, scale_a, scale_b: F32, C: GlobalMemory<F32>) for a fused FP8 (e4m3) GEMM - A/B are quantized on the fly, scale_a/scale_b dequant the f32 accumulator. Mismatched param(s): {}.",
                    tile.span.line, kernel.name, bad_params.join(", ")
                ));
            }
            return;
        }

        // A, B are the f16 Tensor Core operands; C is the accumulator/output
        // in f32 (matching wmma's f16-in/f32-out contract - see
        // ptx_emitter::emit_tensor_core_gemm_kernel, which hardcodes 4
        // bytes/element and `wmma.store.d...f32` for C specifically). Two
        // 4-parameter shapes are also accepted, told apart purely by
        // param[2]'s element type (matching ptx_emitter's own dispatch
        // logic exactly, so type-checking and codegen never disagree about
        // which shape a given kernel is):
        // - (A, B, Bias, C): Bias is F32, sitting in the same "everything
        //   past the two F16 operands" bucket as C - see
        //   `ptx_emitter::tile_gemm_operands`'s doc comment for the fused
        //   GEMM+Bias+ReLU epilogue this shape dispatches to.
        // - (X, W_gate, W_up, Out): W_up is F16 (unlike Bias) since it's a
        //   second Tensor Core operand, not an epilogue addend - see
        //   `ptx_emitter::tile_gemm_swiglu_operands`'s doc comment for the
        //   fused Linear+SwiGLU epilogue this shape dispatches to.
        let is_swiglu_shape = kernel.params.len() == 4 && is_global_memory_of(&kernel.params[2].ty, "F16");
        let expected_elem = |i: usize| if i < 2 || (is_swiglu_shape && i == 2) { "F16" } else { "F32" };
        let bad_params: Vec<String> = kernel
            .params
            .iter()
            .enumerate()
            .filter(|(i, p)| !is_global_memory_of(&p.ty, expected_elem(*i)))
            .map(|(i, p)| format!("{} (expected GlobalMemory<{}>)", p.name, expected_elem(i)))
            .collect();

        if (kernel.params.len() != 3 && kernel.params.len() != 4) || !bad_params.is_empty() {
            self.errors.push(format!(
                "Line {}: Kernel-level @tile(M, N, K) on `{}` requires 3 parameters (A, B: GlobalMemory<F16>, C: GlobalMemory<F32>) for a plain GEMM, 4 (A, B: GlobalMemory<F16>, Bias, C: GlobalMemory<F32>) for a fused GEMM+Bias+ReLU epilogue, or 4 (X, W_gate, W_up: GlobalMemory<F16>, Out: GlobalMemory<F32>) for a fused Linear+SwiGLU epilogue, in that order - tile-aware GEMM codegen binds them positionally and supports no other shape. Found {} parameter(s){}.",
                tile.span.line,
                kernel.name,
                kernel.params.len(),
                if bad_params.is_empty() {
                    String::new()
                } else {
                    format!(", mismatched param(s): {}", bad_params.join(", "))
                }
            ));
        }
    }

    fn check_func(&mut self, f: &FuncDecl) {
        self.zk_safe_stack.push(f.is_safe || f.is_zk_safe);
        self.zk_allow_unconstrained_stack.push(f.is_zk_allow_unconstrained);
        self.push_scope();

        let prev_unsafe = self.in_unsafe;
        if !f.is_safe {
            self.in_unsafe = true;
        }

        for param in &f.params {
            let sty = self.resolve_type(&param.ty);
            if sty == SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: Function parameters cannot have Transfer type. Linear Transfer obligations cannot cross function boundaries in the bootstrap compiler.",
                    param.span.line
                ));
            }
            self.insert_var(param.name.clone(), sty);
            self.set_signal_constraint(param.name.clone(), ConstraintState::Constrained, param.span.clone());
        }

        let prev_ret_ty = self.current_return_type.clone();
        if let Some(ret_ty) = &f.ret_ty {
            let resolved = self.resolve_type(ret_ty);
            self.current_return_type = Some(resolved.clone());
            if resolved == SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: Functions cannot return Transfer obligations. They must be consumed by `pipe.wait(...)` in the creating scope.",
                    f.span.line
                ));
            }
        } else {
            self.current_return_type = None;
        }

        self.check_block(&f.body);
        self.current_return_type = prev_ret_ty;

        self.in_unsafe = prev_unsafe;
        self.pop_scope();
        self.zk_allow_unconstrained_stack.pop();
        self.zk_safe_stack.pop();
    }

    fn check_block(&mut self, block: &Block) {
        // Linear obligations are scoped to the block they are defined in.
        // Wait, loop bodies require their own scope.
        self.push_scope();

        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }

        self.pop_scope();
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
            match stmt {
            Stmt::Let {
                name,
                ty,
                init,
                span,
                bounds,
                ..
            } => {
                let mut inferred_type = SemanticType::Unknown;
                let mut explicit_resolved = None;

                if let Some(explicit_ty) = ty {
                    explicit_resolved = Some(self.resolve_type(explicit_ty));
                }

                if !self.in_unsafe && init.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] Variables in safe blocks must be explicitly initialized.",
                        span.line
                    ));
                }

                if let Some(init_expr) = init {
                    inferred_type =
                        self.check_expr_with_expected(init_expr, explicit_resolved.as_ref());
                    if let Some(init_interval) = self.eval_interval(init_expr) {
                        self.insert_interval(name.clone(), init_interval);
                    }
                }

                if let Some(bounds_attr) = bounds {
                    let min_val = self.eval_interval(&bounds_attr.min).map(|i| i.min);
                    let max_val = self.eval_interval(&bounds_attr.max).map(|i| i.max);
                    if let (Some(mn), Some(mx)) = (min_val, max_val) {
                        if let Some(init_expr) = init {
                            if let Some(init_interval) = self.eval_interval(init_expr) {
                                if (init_interval.min < mn || init_interval.max > mx) && !self.in_unsafe {
                                    self.errors.push(format!(
                                        "Line {}: [Strict Safety] Bounds Violation: initialized value range [{}, {}] exceeds declared bounds [{}, {}] of `{}`.",
                                        span.line, init_interval.min, init_interval.max, mn, mx, name
                                    ));
                                }
                            }
                        }
                        self.insert_interval(name.clone(), Interval { min: mn, max: mx });
                        self.mark_explicitly_bounded(name.clone());
                    }
                }

                if let Some(resolved) = explicit_resolved {
                    // Minimal type unification
                    if inferred_type == SemanticType::Unknown {
                        inferred_type = resolved.clone();
                    } else if !self.types_are_compatible(&inferred_type, &resolved)
                        && inferred_type != SemanticType::TransferObligation
                    {
                        self.errors.push(format!(
                            "Line {}: Type mismatch in let assignment.",
                            span.line
                        ));
                    }
                }

                self.insert_var(name.clone(), inferred_type.clone());

                let init_state = if let Some(init_expr) = init {
                    self.eval_expr_constraint_state(init_expr)
                } else {
                    ConstraintState::Constrained
                };
                self.set_signal_constraint(name.clone(), init_state, span.clone());

                // If it's a transfer obligation (`cp_async`), track it linearly.
                if inferred_type == SemanticType::TransferObligation {
                    let destination = init
                        .as_ref()
                        .and_then(|expr| Self::transfer_destination_from_expr(expr));

                    if init.is_none() {
                        self.errors.push(format!(
                            "Line {}: Transfer obligations must be initialized when declared.",
                            span.line
                        ));
                    }

                    if init.is_some() && destination.is_none() {
                        self.errors.push(format!(
                            "Line {}: Transfer obligations must originate from `cp_async(...)` so the compiler can track their destination.",
                            span.line
                        ));
                    }

                    self.linear_tracker.register_obligation(
                        name.clone(),
                        span.clone(),
                        destination,
                    );
                }
            }
            Stmt::TypeAlias { name, ty, span } => {
                let mut resolved = self.resolve_type(ty);
                // If defining a new SmemLayout, run the Bank Conflict Prover!
                if let SemanticType::SharedMemoryTile {
                    rows,
                    cols,
                    swizzle,
                } = &mut resolved
                {
                    let mut prover_layout = ProverLayout {
                        rows: *rows,
                        cols: *cols,
                        swizzle: swizzle.clone(),
                        swizzle_mode: None,
                        bytes_per_element: 2, // Defaulting F16 for prototype logic
                    };

                    let need_autoswizzle = if swizzle.is_none() {
                        true
                    } else {
                        BankConflictProver::prove_ldmatrix_m16n8(&prover_layout).is_err()
                    };

                    if need_autoswizzle {
                        // Find a swizzle pattern that satisfies the proof!
                        let mut found_swizzle = None;
                        'search: for xor_bits in 1..=4 {
                            for base_shift in 0..=4 {
                                for offset in 0..=4 {
                                    let candidate = SwizzlePattern {
                                        xor_bits,
                                        base_shift,
                                        offset,
                                    };
                                    prover_layout.swizzle = Some(candidate.clone());
                                    if BankConflictProver::prove_ldmatrix_m16n8(&prover_layout).is_ok() {
                                        found_swizzle = Some(candidate);
                                        break 'search;
                                    }
                                }
                            }
                        }

                        if let Some(working_swizzle) = found_swizzle {
                            println!(
                                "    [Optimization] Line {}: Auto-swizzling SharedMemoryTile {}x{} to solve bank conflicts: Swizzle<XOR={}, base_shift={}, offset={}>",
                                span.line, rows, cols, working_swizzle.xor_bits, working_swizzle.base_shift, working_swizzle.offset
                            );
                            *swizzle = Some(working_swizzle);
                        } else {
                            println!(
                                "    [Warning] Line {}: Bank Conflict Prover could not find a swizzle pattern to solve conflicts for {}x{}.",
                                span.line, rows, cols
                            );
                        }
                    } else {
                        println!(
                            "    [Optimization] Line {}: SharedMemoryTile {}x{} has verified 0 bank conflicts.",
                            span.line, rows, cols
                        );
                    }
                }
                self.insert_var(name.clone(), resolved);
            }
            Stmt::For { loop_var, start, end, step, body, invariant, is_uniform_branch: _, span, .. } => {
                self.push_scope();

                if !self.in_unsafe && invariant.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] Loops in safe blocks require formal @invariants.",
                        span.line
                    ));
                }

                let start_val = self.eval_interval(start).map(|i| i.min);
                let end_val = self.eval_interval(end).map(|i| i.max);
                let bounds_are_known = matches!((start_val, end_val), (Some(_), Some(_)));
                if let (Some(s_min), Some(e_max)) = (start_val, end_val) {
                    self.insert_interval(loop_var.clone(), Interval { min: s_min, max: e_max - 1 });
                } else {
                    // The loop bounds are not statically known, so the loop
                    // variable has NO provable range and must not be given one.
                    //
                    // This used to fabricate `Interval { min: 0, max: 999999 }`,
                    // which asserts two facts it has not proved. The `max` half
                    // was harmless in practice because 999999 trips the overflow
                    // check for any normal array - but the `min` half claimed the
                    // index is non-negative, and nothing had established that.
                    // `for i in n..3` over a 2,000,000-element array therefore
                    // compiled clean with `n` an unconstrained parameter: the
                    // fabricated max slipped under the array size and the
                    // fabricated min waved the negative check through.
                    //
                    // Removing the interval makes the index unprovable instead,
                    // which is the honest answer and is what
                    // `mark_explicitly_bounded` below must NOT override.
                    self.update_interval(&loop_var, None);
                }

                self.insert_var(loop_var.clone(), SemanticType::Primitive("I32".into()));
                if bounds_are_known {
                    self.mark_explicitly_bounded(loop_var.clone());
                }

                let mut assigned_vars = std::collections::HashSet::new();
                self.collect_assigned_vars_in_block(body, &mut assigned_vars);
                for var in &assigned_vars {
                    self.update_interval(var, None);
                }

                for s in &body.stmts {
                    self.check_stmt(s);
                }

                if !self.in_unsafe {
                    if let Some(inv_expr) = invariant {
                        self.verify_for_loop_invariant(loop_var, start, end, step, body, inv_expr, span);
                    }
                }

                for var in &assigned_vars {
                    self.update_interval(var, None);
                }

                self.pop_scope();
            }
            Stmt::Assign {
                target,
                value,
                span,
            } => {
                let t1 = self.check_expr(target);
                let t2 = self.check_expr_with_expected(value, Some(&t1));
                if t1 == SemanticType::TransferObligation {
                    self.errors.push(format!(
                        "Line {}: Transfer bindings cannot be reassigned. Create a new Transfer with `let` and consume it exactly once with `pipe.wait(...)`.",
                        span.line
                    ));
                }
                if t2 == SemanticType::TransferObligation {
                    self.errors.push(format!(
                        "Line {}: Transfer obligations cannot be assigned or moved into another location. Consume them with `pipe.wait(...)`.",
                        span.line
                    ));
                }
                if !self.types_are_compatible(&t1, &t2) && t1 != SemanticType::Unknown && t2 != SemanticType::Unknown {
                    self.errors.push(format!(
                        "Line {}: Invalid assignment, types do not match.",
                        span.line
                    ));
                }
                if let Expr::Ident(name, _) = target {
                    let val_state = self.eval_expr_constraint_state(value);
                    self.update_signal_constraint_state(name, val_state);
                    if self.is_explicitly_bounded(name) {
                        if let Some(target_interval) = self.lookup_interval(name).cloned() {
                            if let Some(val_interval) = self.eval_interval(value) {
                                if val_interval.min < target_interval.min || val_interval.max > target_interval.max {
                                    self.errors.push(format!(
                                        "Line {}: [Strict Safety] Bounds Violation: assigned value range [{}, {}] exceeds declared bounds [{}, {}] of `{}`.",
                                        span.line, val_interval.min, val_interval.max, target_interval.min, target_interval.max, name
                                    ));
                                }
                            } else if !self.in_unsafe {
                                self.errors.push(format!(
                                    "Line {}: [Strict Safety] Bounds Violation: assigning an unconstrained value to bounded variable `{}`.",
                                    span.line, name
                                ));
                            }
                        }
                    } else {
                        let val_interval = self.eval_interval(value);
                        self.update_interval(name, val_interval);
                    }
                }
            }
            Stmt::Expr(expr) => {
                let ty = self.check_expr(expr);
                if ty == SemanticType::TransferObligation {
                    self.errors.push(format!(
                        "Line {}: Transfer obligations must be bound to a name and later consumed by `pipe.wait(...)`; they cannot be dropped as expression statements.",
                        expr.span().line
                    ));
                }
            }
            Stmt::Return(val, span) => {
                if let Some(expr) = val {
                    let expected_ret_ty = self.current_return_type.clone();
                    let ret_ty = self.check_expr_with_expected(expr, expected_ret_ty.as_ref());
                    if ret_ty == SemanticType::TransferObligation {
                        self.errors.push(format!(
                            "Line {}: Returning a Transfer obligation would leak a linear sync proof. Consume it with `pipe.wait(...)` before returning.",
                            span.line
                        ));
                    }
                }
            }
            Stmt::Chisel(block, _) => {
                // Chisel blocks are privileged — type-check their contents normally
                self.check_block(block);
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                is_uniform_branch,
                ..
            } => {
                if *is_uniform_branch {
                    self.check_uniformity(condition);
                }
                self.check_expr(condition);
                self.check_block(then_block);
                if let Some(eb) = else_block {
                    self.check_block(eb);
                }
            }
            Stmt::While {
                condition, body, invariant, max_iterations, is_uniform_branch, ..
            } => {
                if max_iterations.is_none() {
                    // Static error Z0010 check for dynamic un-annotated while loops in ZK mode
                    self.errors.push(format!(
                        "Line {}: error[Z0010]: dynamic 'while' loop prohibited in ZK circuit mode\n  hint: annotate loop with '@max_iterations(N)' where N is a compile-time constant integer",
                        condition.span().line
                    ));
                }
                if !self.in_unsafe && invariant.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] While loops in safe blocks require formal @invariants.",
                        condition.span().line
                    ));
                }
                if *is_uniform_branch {
                    self.check_uniformity(condition);
                }

                let mut assigned_vars = std::collections::HashSet::new();
                self.collect_assigned_vars_in_block(body, &mut assigned_vars);
                for var in &assigned_vars {
                    self.update_interval(var, None);
                }

                self.check_expr(condition);
                self.check_block(body);

                if !self.in_unsafe {
                    if let Some(inv_expr) = invariant {
                        self.verify_while_loop_invariant(condition, body, inv_expr, &condition.span());
                    }
                }

                for var in &assigned_vars {
                    self.update_interval(var, None);
                }
            }
            Stmt::Break { .. } => {}
            Stmt::Match {
                scrutinee, arms, ..
            } => {
                self.check_expr(scrutinee);
                for arm in arms {
                    let arm_ty = self.check_expr(&arm.body);
                    self.reject_transfer_escape(&arm_ty, &arm.span, "as a match arm result");
                }
            }
            Stmt::CompoundAssign { target, value, .. } => {
                let lhs = self.check_expr(target);
                let rhs = self.check_expr(value);
                self.reject_transfer_escape(&lhs, &target.span(), "in compound assignment");
                self.reject_transfer_escape(&rhs, &value.span(), "in compound assignment");
            }
            Stmt::SafeBlock(block, _) => {
                let prev_unsafe = self.in_unsafe;
                self.in_unsafe = false;
                self.check_block(block);
                self.in_unsafe = prev_unsafe;
            }
            Stmt::GhostBlock(block, _) => {
                let prev_unsafe = self.in_unsafe;
                self.in_unsafe = false;
                self.check_block(block);
                self.in_unsafe = prev_unsafe;
            }
            Stmt::HintBlock { outputs, body, span } => {
                self.check_block(body);
                for out_var in outputs {
                    self.set_signal_constraint(
                        out_var.clone(),
                        ConstraintState::TaintedUnconstrained {
                            origins: vec![span.clone()],
                            reasons: vec![UnconstrainedReason::HintOutput(out_var.clone())],
                        },
                        span.clone(),
                    );
                }
            }
            Stmt::ClockDomainBlock { body, span, .. } => {
                // Type-check the body within the clock domain scope
                println!(
                    "      \x1b[1;35m[CDC]\x1b[0m Line {}: @clock_domain block entered.",
                    span.line
                );
                self.check_block(body);
            }
            Stmt::CompileTimeAssert { condition, message, span } => {
                // Verify the assertion expression is well-typed
                self.check_expr(condition);
                let msg = message.as_deref().unwrap_or("compile-time assertion");
                println!(
                    "      \x1b[1;36m[Verified]\x1b[0m Line {}: compile_time::assert! \"{}\"",
                    span.line, msg
                );
            }
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> SemanticType {
        self.check_expr_with_expected(expr, None)
    }

    fn check_expr_with_expected(
        &mut self,
        expr: &Expr,
        expected_type: Option<&SemanticType>,
    ) -> SemanticType {
        let span = expr.span();
        match expr {
            Expr::ZeroInit(span) => {
                if let Some(expected) = expected_type {
                    expected.clone()
                } else {
                    self.errors.push(format!(
                        "Line {}: Ambiguous zero-initializer: cannot infer struct type.",
                        span.line
                    ));
                    SemanticType::Unknown
                }
            }
            Expr::Ident(name, _) => {
                if let Some(ty) = self.lookup_var(name) {
                    let ty = ty.clone();
                    if ty == SemanticType::TransferObligation && self.allow_transfer_use == 0 {
                        self.errors.push(format!(
                            "Line {}: `{}` is a linear Transfer obligation and may only be used as an argument to `pipe.wait(...)`.",
                            span.line, name
                        ));
                    }
                    ty
                } else {
                    // Could be a Type Alias reference (e.g., `smem_A: ATile`)
                    SemanticType::Unknown
                }
            }
            Expr::Call { func, args, .. } => {
                if let Expr::Ident(fname, _) = &**func {
                    if fname == "cp_async" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an operand to `cp_async`",
                            );
                        }
                        // Creates an obligation
                        return SemanticType::TransferObligation;
                    }
                    if fname == "ldmatrix" || fname == "load" {
                        if let Some(arg) = args.first() {
                            self.require_destination_ready(arg, &span);
                        }
                    }
                    if fname == "mma_sync" {
                        self.check_mma_sync(args, &span);
                        // Returns 'D' fragment (Accumulator)
                        return SemanticType::Fragment {
                            op: "MMA_m16n8k16".into(),
                            role: "D".into(),
                            dtype: "F32".into(),
                        };
                    }
                }
                if let Expr::MemberAccess { base, member, .. } = &**func {
                    if member == "wait" {
                        return self.check_wait_call(base, args, &span);
                    }
                }
                if let Expr::Path {
                    namespace, member, ..
                } = &**func
                {
                    if namespace == "barrier" && member == "sync" {
                        self.linear_tracker.synchronize_barrier();
                        return SemanticType::Unknown;
                    }
                    if namespace == "File" && member == "read" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to `File::read`",
                            );
                        }
                        // Prototype read evaluation guarantees String return
                        return SemanticType::Primitive("String".into());
                    }
                    if namespace == "Vec" || namespace == "String" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to a dynamic allocation API",
                            );
                        }
                        if !self.in_unsafe {
                            self.errors.push(format!("Line {}: Dynamic memory operations like {}::{} are mapped to raw void* and require an @unsafe function context.", span.line, namespace, member));
                        }
                        return SemanticType::Unknown;
                    }
                }
                let func_ty = self.check_expr(func);
                self.reject_transfer_escape(&func_ty, &func.span(), "as a callable value");

                let mut expected_params = None;
                if let Expr::Ident(fname, _) = &**func {
                    expected_params = self.functions.get(fname).cloned();
                } else if let Expr::Path {
                    namespace, member, ..
                } = &**func
                {
                    expected_params = self
                        .functions
                        .get(&format!("{}_{}", namespace, member))
                        .cloned();
                }

                for (i, arg) in args.iter().enumerate() {
                    let expected_ty = expected_params.as_ref().and_then(|p| p.get(i));
                    let arg_ty = self.check_expr_with_expected(arg, expected_ty);
                    self.reject_transfer_escape(&arg_ty, &arg.span(), "as a function argument");
                }
                SemanticType::Unknown
            }
            Expr::MemberAccess { base, member, .. } => {
                let base_ty = self.check_expr(base);
                if member == "wait" {
                    SemanticType::Unknown
                } else {
                    self.reject_transfer_escape(
                        &base_ty,
                        &base.span(),
                        "as the base of member access",
                    );
                    SemanticType::Unknown
                }
            }
            Expr::GenericCall {
                func,
                generic_args,
                args,
                ..
            } => {
                if let Expr::Path {
                    namespace, member, ..
                } = &**func
                {
                    if namespace == "SharedMemory" && member == "alloc" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to `SharedMemory::alloc`",
                            );
                        }
                        if let Some(layout_ty) = generic_args.first() {
                            return self.resolve_type(layout_ty);
                        }
                        return SemanticType::Unknown;
                    }
                    if namespace == "Pipeline" && member == "init" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to `Pipeline::init`",
                            );
                        }
                        return SemanticType::Pipeline;
                    }
                }

                let func_ty = self.check_expr(func);
                self.reject_transfer_escape(&func_ty, &func.span(), "as a generic callable value");
                for arg in args {
                    let arg_ty = self.check_expr(arg);
                    self.reject_transfer_escape(&arg_ty, &arg.span(), "as a generic call argument");
                }
                SemanticType::Unknown
            }
            Expr::StructLit { name, fields, .. } => {
                let struct_fields = self.structs.get(name).cloned();
                for (fname, expr) in fields {
                    let expected_ty = struct_fields.as_ref().and_then(|m| m.get(fname));
                    let field_ty = self.check_expr_with_expected(expr, expected_ty);
                    self.reject_transfer_escape(&field_ty, &expr.span(), "inside a struct literal");
                }
                SemanticType::Primitive(name.clone())
            }
            Expr::Index { base, index, .. } => {
                self.require_destination_ready(base, &span);
                let base_ty = self.check_expr(base);
                let index_ty = self.check_expr(index);
                self.reject_transfer_escape(&base_ty, &base.span(), "as an indexed value");
                self.reject_transfer_escape(&index_ty, &index.span(), "as an index expression");
                
                if let SemanticType::Array { element, size } = &base_ty {
                    INDEX_ARRAY_SIZES.with(|map| {
                        map.borrow_mut().insert((span.line, span.col), *size);
                    });
                    
                    let mut is_safe = false;
                    if let Some(index_interval) = self.eval_interval(index) {
                        let mut min_ok = true;
                        let mut max_ok = true;
                        if index_interval.min < 0 {
                            min_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible negative index access (inferred min: {}).",
                                span.line, index_interval.min
                            ));
                        }
                        if index_interval.max >= *size as i64 {
                            max_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible overflow index access (inferred max: {} >= array size {}).",
                                span.line, index_interval.max, size
                            ));
                        }
                        if min_ok && max_ok {
                            is_safe = true;
                        }
                    } else if !self.in_unsafe {
                        self.errors.push(format!(
                            "Line {}: [Strict Safety] Array access is unsafe: index has no statically provable bounds. Annotate the index variable with @bounds(min, max).",
                            span.line
                        ));
                    }
                    
                    if is_safe {
                        SAFE_INDICES.with(|set| {
                            set.borrow_mut().insert((span.line, span.col));
                        });
                    }
                    
                    return (**element).clone();
                }

                if let SemanticType::SharedMemoryTile { rows, cols, swizzle } = &base_ty {
                    let size = (*rows * *cols) as usize;
                    INDEX_ARRAY_SIZES.with(|map| {
                        map.borrow_mut().insert((span.line, span.col), size);
                    });
                    if let Some(sw) = swizzle {
                        INDEX_SWIZZLES.with(|map| {
                            map.borrow_mut().insert((span.line, span.col), sw.clone());
                        });
                    }

                    let mut is_safe = false;
                    if let Some(index_interval) = self.eval_interval(index) {
                        let mut min_ok = true;
                        let mut max_ok = true;
                        if index_interval.min < 0 {
                            min_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible negative index access (inferred min: {}).",
                                span.line, index_interval.min
                            ));
                        }
                        if index_interval.max >= size as i64 {
                            max_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible overflow index access (inferred max: {} >= tile size {}).",
                                span.line, index_interval.max, size
                            ));
                        }
                        if min_ok && max_ok {
                            is_safe = true;
                        }
                    } else if !self.in_unsafe {
                        self.errors.push(format!(
                            "Line {}: [Strict Safety] Array access is unsafe: index has no statically provable bounds. Annotate the index variable with @bounds(min, max).",
                            span.line
                        ));
                    }
                    
                    if is_safe {
                        SAFE_INDICES.with(|set| {
                            set.borrow_mut().insert((span.line, span.col));
                        });
                    }
                    
                    return SemanticType::Primitive("F16".into());
                }
                
                SemanticType::Unknown
            }
            Expr::BinaryOp { left, op, right, span } => {
                let lhs = self.check_expr(left);
                let rhs = self.check_expr(right);
                self.reject_transfer_escape(&lhs, &left.span(), "in a binary expression");
                self.reject_transfer_escape(&rhs, &right.span(), "in a binary expression");

                if *op == BinaryOp::Eq {
                    self.check_verification_transition(left, right, span);
                }

                SemanticType::Unknown
            }
            Expr::UnaryOp { op, operand, .. } => {
                let span = expr.span();
                if *op == crate::ast::UnaryOp::Deref && !self.in_unsafe {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] Raw pointer dereferencing is forbidden in safe blocks.",
                        span.line
                    ));
                }
                let operand_ty = self.check_expr(operand);
                self.reject_transfer_escape(&operand_ty, &operand.span(), "in a unary expression");
                SemanticType::Unknown
            }
            Expr::BlockExpr(block, _) => {
                self.check_block(block);
                SemanticType::Unknown
            }
            _ => SemanticType::Unknown,
        }
    }

    // ── Semantic Verifications ──────────────────────────────

    /// Enforces Phantom Fragment Role types. (A + B + C -> D)
    fn check_mma_sync(&mut self, args: &[Expr], span: &Span) {
        if args.len() != 3 {
            self.errors.push(format!(
                "Line {}: mma_sync requires exactly 3 operands (A, B, C).",
                span.line
            ));
            return;
        }

        let t_a = self.check_expr(&args[0]);
        let t_b = self.check_expr(&args[1]);
        let t_c = self.check_expr(&args[2]);

        let mut require_role = |ty: &SemanticType, expected_roles: &[&str]| {
            if let SemanticType::Fragment { role, .. } = ty {
                if !expected_roles.contains(&role.as_str()) {
                    self.errors.push(format!(
                        "Line {}: Fragment Role Error: expected Fragment<{}, ...>, got Fragment<{}, ...>.",
                        span.line, expected_roles.join("/"), role
                    ));
                }
            }
        };

        require_role(&t_a, &["A"]);
        require_role(&t_b, &["B"]);
        require_role(&t_c, &["C", "D"]); // Or D commonly used for accumulator feedback
    }

    // ── Type Resolution ─────────────────────────────────────

    fn resolve_type(&mut self, ast_ty: &Type) -> SemanticType {
        match ast_ty {
            Type::Primitive(name, _) => SemanticType::Primitive(name.clone()),
            Type::Ident(name, _) => {
                if name == "ptr" {
                    SemanticType::Primitive("ptr".into())
                } else if let Some(t) = self.lookup_var(name) {
                    t.clone() // alias resolution
                } else {
                    SemanticType::Unknown
                }
            }
            Type::Generic { base, args, .. } => {
                if base == "Fragment" && args.len() >= 3 {
                    let mut op = "Unknown".to_string();
                    let mut role = "Unknown".to_string();
                    let mut dtype = "Unknown".to_string();

                    if let GenericArg::Type(Type::Ident(o, _)) = &args[0] {
                        op = o.clone();
                    }
                    if let GenericArg::Type(Type::Ident(r, _)) = &args[1] {
                        role = r.clone();
                    }
                    if let GenericArg::Type(Type::Primitive(d, _)) = &args[2] {
                        dtype = d.clone();
                    }

                    return SemanticType::Fragment { op, role, dtype };
                }

                if base == "Vec" {
                    let mut inner_ty = SemanticType::Unknown;
                    let mut allocator = "Standard".to_string();
                    if args.len() >= 1 {
                        if let GenericArg::Type(t) = &args[0] {
                            inner_ty = self.resolve_type(t);
                        }
                    }
                    if args.len() >= 2 {
                        if let GenericArg::Type(Type::Ident(alloc, _)) = &args[1] {
                            allocator = alloc.clone();
                        }
                    }
                    return SemanticType::Vector(Box::new(inner_ty), allocator);
                }

                if base == "SmemLayout" {
                    let mut rows = 0;
                    let mut cols = 0;
                    let mut swizzle = None;

                    for arg in args {
                        if let GenericArg::Named { name, val } = arg {
                            if name == "rows" {
                                if let Expr::IntLit(r, _) = val {
                                    rows = *r as u32;
                                }
                            }
                            if name == "cols" {
                                if let Expr::IntLit(c, _) = val {
                                    cols = *c as u32;
                                }
                            }
                            if name == "swizzle" {
                                // Dummy fill for parser validation context
                                swizzle = Some(SwizzlePattern {
                                    xor_bits: 3,
                                    base_shift: 0,
                                    offset: 0,
                                });
                            }
                        }
                    }

                    return SemanticType::SharedMemoryTile {
                        rows,
                        cols,
                        swizzle,
                    };
                }

                if base == "Transfer" {
                    return SemanticType::TransferObligation;
                }

                if base == "BlockTile" {
                    let mut elem_resolved = SemanticType::Primitive("F32".into());
                    let mut sz = 128;
                    if !args.is_empty() {
                        if let GenericArg::Type(t) = &args[0] {
                            elem_resolved = self.resolve_type(t);
                        }
                    }
                    if args.len() >= 2 {
                        if let GenericArg::Value(Expr::IntLit(v, _)) = &args[1] {
                            sz = *v as usize;
                        }
                    }
                    return SemanticType::BlockTile {
                        element: Box::new(elem_resolved),
                        size: sz,
                    };
                }

                SemanticType::Unknown
            }
            Type::Array { element, size, .. } => {
                let elem_resolved = self.resolve_type(element);
                let mut sz = 0;
                if let Expr::IntLit(val, _) = &**size {
                    sz = *val as usize;
                }
                SemanticType::Array {
                    element: Box::new(elem_resolved),
                    size: sz,
                }
            }
            Type::BlockTile { element, size, .. } => {
                let elem_resolved = self.resolve_type(element);
                let mut sz = 128;
                if let Expr::IntLit(val, _) = &**size {
                    sz = *val as usize;
                }
                SemanticType::BlockTile {
                    element: Box::new(elem_resolved),
                    size: sz,
                }
            }
            Type::Reference { .. } => {
                // Reference types not yet semantically checked in prototype
                SemanticType::Unknown
            }
        }
    }

    /// Translates an expression into SMT-LIB, or reports that it cannot.
    ///
    /// **Every unhandled node returns `Err`.** That is the entire point, and it
    /// is a reversal: this function used to end in `_ => "0".to_string()`, with
    /// `_ => "+"` for unknown binary operators and `_ => opnd` for unknown unary
    /// ones. So a call, an index, a member access or even a float literal was
    /// handed to Z3 as the constant `0`; `x & y` was proven as `x + y`; and
    /// `*p` was proven as `p`. Z3 then dutifully answered a question about a
    /// different program, and the answer was reported as a verified invariant.
    ///
    /// A verifier that silently approximates is worse than no verifier, because
    /// it produces the paperwork of a proof without the proof. If a construct
    /// is not modelled, the only safe answer is to refuse to make a claim.
    fn expr_to_smt(
        &self,
        expr: &Expr,
        versions: &HashMap<String, usize>,
    ) -> Result<String, String> {
        match expr {
            Expr::IntLit(val, _) => Ok(val.to_string()),
            Expr::BoolLit(val, _) => Ok(val.to_string()),
            Expr::Ident(name, _) => {
                if let Some(&ver) = versions.get(name) {
                    Ok(format!("{}_{}", name, ver))
                } else {
                    Ok(format!("{}_0", name))
                }
            }
            Expr::BinaryOp { left, op, right, .. } => {
                let lhs = self.expr_to_smt(left, versions)?;
                let rhs = self.expr_to_smt(right, versions)?;
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "div",
                    BinaryOp::Mod => "mod",
                    BinaryOp::Eq => "=",
                    BinaryOp::NotEq => "distinct",
                    BinaryOp::Lt => "<",
                    BinaryOp::Gt => ">",
                    BinaryOp::Le => "<=",
                    BinaryOp::Ge => ">=",
                    BinaryOp::And => "and",
                    BinaryOp::Or => "or",
                    // Bitwise and shift operators have no direct counterpart in
                    // the integer theory used here. They used to fall through to
                    // "+", which is not an approximation, it is a different
                    // function.
                    other => {
                        return Err(format!(
                            "the operator `{:?}` has no sound encoding in the integer theory \
this verifier uses",
                            other
                        ))
                    }
                };
                Ok(format!("({} {} {})", op_str, lhs, rhs))
            }
            Expr::UnaryOp { op, operand, .. } => {
                let opnd = self.expr_to_smt(operand, versions)?;
                match op {
                    UnaryOp::Neg => Ok(format!("(- {})", opnd)),
                    UnaryOp::Not => Ok(format!("(not {})", opnd)),
                    other => Err(format!(
                        "the unary operator `{:?}` is not modelled (it used to be encoded as its \
own operand, so `*p` was proven as `p`)",
                        other
                    )),
                }
            }
            other => Err(format!(
                "`{}` is not modelled by the verifier",
                expr_to_string(other)
            )),
        }
    }

    fn generate_smt_decls_and_preconditions(
        &self,
        vars: &std::collections::HashSet<String>,
        declarations: &mut Vec<String>,
        preconditions: &mut Vec<String>,
    ) {
        for var in vars {
            declarations.push(format!("(declare-const {}_{} Int)", var, 0));
            if let Some(interval) = self.lookup_interval(var) {
                preconditions.push(format!(
                    "(assert (and (>= {}_{} {}) (<= {}_{} {})))",
                    var, 0, interval.min, var, 0, interval.max
                ));
            }
        }
    }

    fn trace_body_statements(
        &self,
        stmts: &[Stmt],
        versions: &mut HashMap<String, usize>,
        declarations: &mut Vec<String>,
        body_assertions: &mut Vec<String>,
    ) -> Result<(), String> {
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target, value, .. } => {
                    if let Expr::Ident(name, _) = target {
                        if versions.contains_key(name) {
                            // An unmodellable right-hand side does not need to be
                            // refused - it needs to be UNKNOWN. Giving the target a
                            // fresh unconstrained version says exactly that, and is
                            // the same sound over-approximation used for branches:
                            // the invariant must then hold whatever the expression
                            // produced. Refusing here would reject `let v = arr[i];`
                            // in an otherwise perfectly checkable loop.
                            let Ok(rhs_smt) = self.expr_to_smt(value, versions) else {
                                let mut one = std::collections::HashSet::new();
                                one.insert(name.clone());
                                Self::havoc(&one, versions, declarations);
                                continue;
                            };
                            let current_ver = versions.get(name).cloned().unwrap_or(0);
                            let next_ver = current_ver + 1;
                            versions.insert(name.clone(), next_ver);
                            declarations.push(format!("(declare-const {}_{} Int)", name, next_ver));
                            body_assertions.push(format!(
                                "(assert (= {}_{} {}))",
                                name, next_ver, rhs_smt
                            ));
                        }
                    }
                }
                Stmt::CompoundAssign { target, op, value, .. } => {
                    if let Expr::Ident(name, _) = target {
                        if versions.contains_key(name) {
                            let Ok(rhs_smt) = self.expr_to_smt(value, versions) else {
                                let mut one = std::collections::HashSet::new();
                                one.insert(name.clone());
                                Self::havoc(&one, versions, declarations);
                                continue;
                            };
                            let current_ver = versions.get(name).cloned().unwrap_or(0);
                            let next_ver = current_ver + 1;
                            let op_str = match op {
                                BinaryOp::Add => "+",
                                BinaryOp::Sub => "-",
                                BinaryOp::Mul => "*",
                                BinaryOp::Div => "div",
                                BinaryOp::Mod => "mod",
                                _ => {
                                    // `x &= y` and friends: the result is not
                                    // expressible, so the variable becomes unknown.
                                    let mut one = std::collections::HashSet::new();
                                    one.insert(name.clone());
                                    Self::havoc(&one, versions, declarations);
                                    continue;
                                }
                            };
                            let expr_smt = format!("({} {}_{} {})", op_str, name, current_ver, rhs_smt);
                            versions.insert(name.clone(), next_ver);
                            declarations.push(format!("(declare-const {}_{} Int)", name, next_ver));
                            body_assertions.push(format!(
                                "(assert (= {}_{} {}))",
                                name, next_ver, expr_smt
                            ));
                        }
                    }
                }
                Stmt::Let { name, init, .. } => {
                    let is_int = self.lookup_var(name).map(|ty| {
                        if let SemanticType::Primitive(prim_name) = ty {
                            prim_name == "I32" || prim_name == "u32" || prim_name == "usize" || prim_name == "i64"
                        } else {
                            false
                        }
                    }).unwrap_or(false);
                    if is_int {
                        let rhs_smt = if let Some(init_expr) = init {
                            match self.expr_to_smt(init_expr, versions) {
                                Ok(v) => v,
                                Err(_) => {
                                    // Bound to something we cannot express, so the
                                    // new variable is simply unknown.
                                    versions.insert(name.clone(), 0);
                                    declarations
                                        .push(format!("(declare-const {}_{} Int)", name, 0));
                                    continue;
                                }
                            }
                        } else {
                            "0".to_string()
                        };
                        versions.insert(name.clone(), 0);
                        declarations.push(format!("(declare-const {}_{} Int)", name, 0));
                        body_assertions.push(format!(
                            "(assert (= {}_{} {}))",
                            name, 0, rhs_smt
                        ));
                    }
                }
                Stmt::SafeBlock(block, _) | Stmt::Chisel(block, _) | Stmt::GhostBlock(block, _) | Stmt::HintBlock { body: block, .. } => {
                    self.trace_body_statements(&block.stmts, versions, declarations, body_assertions)?;
                }
                Stmt::ClockDomainBlock { body, .. } => {
                    self.trace_body_statements(&body.stmts, versions, declarations, body_assertions)?;
                }
                // A branch is modelled by HAVOC: every variable it might assign
                // gets a fresh, unconstrained version. That is a sound
                // over-approximation - the invariant must then hold for any
                // value the branch could have produced, which includes the real
                // one - so it can only make preservation harder to prove, never
                // easier. Skipping the branch entirely had the opposite effect,
                // which is what made `if i >= 0 { i = i - 100; }` satisfy
                // `@invariant(i >= 0)`.
                Stmt::If { then_block, else_block, .. } => {
                    let mut touched = std::collections::HashSet::new();
                    Self::collect_assigned(&then_block.stmts, &mut touched);
                    if let Some(eb) = else_block {
                        Self::collect_assigned(&eb.stmts, &mut touched);
                    }
                    Self::havoc(&touched, versions, declarations);
                }
                // Nested loops likewise: whatever they assign becomes unknown.
                Stmt::For { body, loop_var, .. } => {
                    let mut touched = std::collections::HashSet::new();
                    Self::collect_assigned(&body.stmts, &mut touched);
                    touched.insert(loop_var.clone());
                    Self::havoc(&touched, versions, declarations);
                }
                Stmt::While { body, .. } => {
                    let mut touched = std::collections::HashSet::new();
                    Self::collect_assigned(&body.stmts, &mut touched);
                    Self::havoc(&touched, versions, declarations);
                }
                // A bare expression is almost always a call. Y has no way for a
                // callee to write a caller's local integer unless the caller
                // hands over a reference, so a call with no `&` cannot disturb
                // the tracked variables - all of which are integer scalars.
                // Anything that does take a reference is refused.
                Stmt::Expr(e) => {
                    if Self::takes_reference(e) {
                        return Err(
                            "the loop body passes a reference to a call, which could modify a \
tracked variable in a way this verifier cannot see"
                                .to_string(),
                        );
                    }
                }
                // Anything else is NOT modelled, and an unmodelled statement is
                // rejected rather than skipped.
                //
                // This used to be `_ => {}`. Dropping a statement's effects
                // makes the preservation check strictly easier, so it fails in
                // the unsound direction: the identical violation was caught
                // when written plainly and ACCEPTED when wrapped in a
                // trivially-true `if`, because `Stmt::If` had no arm here.
                // Guarded by `nested_if_cannot_hide_a_false_invariant`.
                other => {
                    return Err(format!(
                        "the loop body contains a `{}` statement, which this verifier does not \
model",
                        Self::stmt_kind(other)
                    ))
                }
            }
        }
        Ok(())
    }

    /// Names assigned anywhere in `stmts`, including nested blocks.
    fn collect_assigned(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target, .. } | Stmt::CompoundAssign { target, .. } => {
                    if let Expr::Ident(n, _) = target {
                        out.insert(n.clone());
                    }
                }
                Stmt::Let { name, .. } => {
                    out.insert(name.clone());
                }
                Stmt::If { then_block, else_block, .. } => {
                    Self::collect_assigned(&then_block.stmts, out);
                    if let Some(eb) = else_block {
                        Self::collect_assigned(&eb.stmts, out);
                    }
                }
                Stmt::For { body, loop_var, .. } => {
                    out.insert(loop_var.clone());
                    Self::collect_assigned(&body.stmts, out);
                }
                Stmt::While { body, .. } => Self::collect_assigned(&body.stmts, out),
                Stmt::SafeBlock(b, _)
                | Stmt::Chisel(b, _)
                | Stmt::GhostBlock(b, _)
                | Stmt::HintBlock { body: b, .. } => Self::collect_assigned(&b.stmts, out),
                Stmt::ClockDomainBlock { body, .. } => Self::collect_assigned(&body.stmts, out),
                _ => {}
            }
        }
    }

    /// Gives each tracked name in `touched` a fresh unconstrained version.
    fn havoc(
        touched: &std::collections::HashSet<String>,
        versions: &mut HashMap<String, usize>,
        declarations: &mut Vec<String>,
    ) {
        for name in touched {
            if let Some(cur) = versions.get(name).cloned() {
                let next = cur + 1;
                versions.insert(name.clone(), next);
                declarations.push(format!("(declare-const {}_{} Int)", name, next));
            }
        }
    }

    /// Whether the expression takes a reference to anything.
    fn takes_reference(expr: &Expr) -> bool {
        match expr {
            Expr::UnaryOp { op: UnaryOp::Ref, .. } => true,
            Expr::UnaryOp { operand, .. } => Self::takes_reference(operand),
            Expr::BinaryOp { left, right, .. } => {
                Self::takes_reference(left) || Self::takes_reference(right)
            }
            Expr::Call { func, args, .. } => {
                Self::takes_reference(func) || args.iter().any(Self::takes_reference)
            }
            Expr::GenericCall { args, .. } => args.iter().any(Self::takes_reference),
            Expr::Index { base, index, .. } => {
                Self::takes_reference(base) || Self::takes_reference(index)
            }
            Expr::MemberAccess { base, .. } => Self::takes_reference(base),
            _ => false,
        }
    }

    /// A short name for a statement kind, for diagnostics.
    fn stmt_kind(stmt: &Stmt) -> &'static str {
    match stmt {
            Stmt::Let { .. } => "let",
            Stmt::Assign { .. } => "assignment",
            Stmt::CompoundAssign { .. } => "compound assignment",
            Stmt::If { .. } => "if",
            Stmt::For { .. } => "nested for",
            Stmt::While { .. } => "nested while",
            Stmt::Return(..) => "return",
            Stmt::Expr(..) => "expression",
            Stmt::TypeAlias { .. } => "type alias",
            Stmt::Chisel(..) => "chisel",
            Stmt::SafeBlock(..) => "safe block",
            Stmt::GhostBlock(..) => "ghost block",
            _ => "unsupported",
        }
    }


    /// Handles an SMT solver that could not be run at all.
    ///
    /// This FAILS THE BUILD by default, and that is a deliberate reversal. It
    /// used to print `[Warning] SMT Solver execution failed` to stdout and
    /// carry on, which meant that on any machine without z3 - the default -
    /// every `@invariant` in every `@safe` block was accepted unchecked. A
    /// deliberately false invariant like `@invariant(i > 1000)` on a `0..10`
    /// loop compiled cleanly and reported "Compilation Successful!".
    ///
    /// An `@invariant` is a proof obligation, not a comment. A `@safe` block
    /// whose obligations were never discharged guarantees nothing, and the
    /// whole point of the annotation is that the guarantee is checkable. Being
    /// unable to check it is a failure, not a detail - the same reasoning that
    /// makes an out-of-range operand unprovable in the ZK backend rather than
    /// silently accepted.
    ///
    /// `Y_ALLOW_UNVERIFIED_INVARIANTS=1` restores the old behaviour for anyone
    /// who genuinely cannot install a solver. It is loud on purpose.
    /// The verifier met a construct it does not model.
    ///
    /// Rejects, and says what it could not model. The alternative - skipping
    /// the construct and asking Z3 anyway - answers a question about a
    /// different program and reports the answer as a verified invariant. That
    /// is how `if i >= 0 { i = i - 100; }` came to satisfy `@invariant(i >= 0)`
    /// while the identical `i = i - 100;` was correctly rejected.
    ///
    /// The rule this encodes, for any future soundness-critical pass: an
    /// unhandled AST node must reject, never be treated as identity or a no-op.
    /// Silent approximation produces the paperwork of a proof without the proof.
    fn smt_unmodellable(&mut self, line: usize, invariant: &Expr, what: &str) {
        if std::env::var("Y_ALLOW_UNVERIFIED_INVARIANTS").is_ok() {
            println!(
                "[Warning] invariant `{}` was NOT verified: {}.",
                expr_to_string(invariant),
                what
            );
            return;
        }
        self.errors.push(format!(
            "Line {}: [Strict Safety] Cannot verify invariant `{}`: {}.\n  The verifier refuses \
to reason about a construct it does not model, because skipping it would make the proof \
obligation strictly easier and could accept an invariant that is false.\n  Rewrite the loop \
body using constructs the verifier supports, or set Y_ALLOW_UNVERIFIED_INVARIANTS=1 to compile \
with this invariant UNVERIFIED.",
            line,
            expr_to_string(invariant),
            what
        ));
    }

    fn smt_unavailable(&mut self, line: usize, invariant: &Expr, phase: &str, err: &str) {
        if std::env::var("Y_ALLOW_UNVERIFIED_INVARIANTS").is_ok() {
            println!(
                "[Warning] SMT solver unavailable; invariant `{}` was NOT verified ({} check). {}",
                expr_to_string(invariant),
                phase,
                err
            );
            return;
        }
        self.errors.push(format!(
            "Line {}: [Strict Safety] Could not verify invariant `{}` ({} check) because the \
SMT solver could not be run.\n  {}\n  An @invariant is a proof obligation - accepting it \
unchecked would make @safe guarantee nothing on this machine.\n  Install z3 (`pip install \
z3-solver`, or your package manager) or set Y_Z3_PATH to its absolute path.\n  To compile \
anyway with invariants UNVERIFIED, set Y_ALLOW_UNVERIFIED_INVARIANTS=1.",
            line,
            expr_to_string(invariant),
            phase,
            err
        ));
    }

    fn verify_while_loop_invariant(&mut self, condition: &Expr, body: &Block, invariant: &Expr, span: &Span) {
        let mut vars = std::collections::HashSet::new();
        for frame in &self.scopes {
            for (name, entry) in &frame.symbols {
                if let SemanticType::Primitive(prim_name) = &entry.ty {
                    if prim_name == "I32" || prim_name == "u32" || prim_name == "usize" || prim_name == "i64" {
                        vars.insert(name.clone());
                    }
                }
            }
        }

        // --- 1. CHECK INITIATION ---
        let mut decls_init = Vec::new();
        let mut preconditions_init = Vec::new();
        self.generate_smt_decls_and_preconditions(&vars, &mut decls_init, &mut preconditions_init);

        let mut versions_init = std::collections::HashMap::new();
        for var in &vars {
            versions_init.insert(var.clone(), 0);
        }
        let inv_init_smt = match self.expr_to_smt(invariant, &versions_init) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        decls_init.sort();
        decls_init.dedup();

        let query_init = format!(
            "{}\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_init.join("\n"),
            preconditions_init.join("\n"),
            inv_init_smt
        );

        match run_z3(&query_init) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant initiation check failed. Invariant `{}` may not hold on loop entry. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                    return;
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "initiation", &e);
            }
        }

        // --- 2. CHECK PRESERVATION ---
        let mut decls_pres = Vec::new();
        let mut preconditions_pres = Vec::new();
        self.generate_smt_decls_and_preconditions(&vars, &mut decls_pres, &mut preconditions_pres);

        let inv_start_smt = match self.expr_to_smt(invariant, &versions_init) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        let cond_start_smt = match self.expr_to_smt(condition, &versions_init) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        let mut versions_pres = versions_init.clone();
        let mut body_assertions = Vec::new();
        if let Err(why) =
            self.trace_body_statements(&body.stmts, &mut versions_pres, &mut decls_pres, &mut body_assertions)
        {
            return self.smt_unmodellable(span.line, invariant, &why);
        }

        let inv_end_smt = match self.expr_to_smt(invariant, &versions_pres) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        decls_pres.sort();
        decls_pres.dedup();

        let query_pres = format!(
            "{}\n{}\n(assert {})\n(assert {})\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_pres.join("\n"),
            preconditions_pres.join("\n"),
            inv_start_smt,
            cond_start_smt,
            body_assertions.join("\n"),
            inv_end_smt
        );

        match run_z3(&query_pres) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant preservation check failed. Invariant `{}` is not preserved by the loop body. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "preservation", &e);
            }
        }
    }

    fn verify_for_loop_invariant(
        &mut self,
        loop_var: &str,
        start: &Expr,
        end: &Expr,
        step: &Option<Expr>,
        body: &Block,
        invariant: &Expr,
        span: &Span,
    ) {
        let mut vars = std::collections::HashSet::new();
        for frame in &self.scopes {
            for (name, entry) in &frame.symbols {
                if let SemanticType::Primitive(prim_name) = &entry.ty {
                    if prim_name == "I32" || prim_name == "u32" || prim_name == "usize" || prim_name == "i64" {
                        vars.insert(name.clone());
                    }
                }
            }
        }
        vars.insert(loop_var.to_string());

        // --- 1. CHECK INITIATION ---
        let mut decls_init = Vec::new();
        let mut preconditions_init = Vec::new();
        self.generate_smt_decls_and_preconditions(&vars, &mut decls_init, &mut preconditions_init);

        let start_smt = match self.expr_to_smt(start, &std::collections::HashMap::new()) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        preconditions_init.push(format!("(assert (= {}_{} {}))", loop_var, 0, start_smt));

        let mut versions_init = std::collections::HashMap::new();
        for var in &vars {
            versions_init.insert(var.clone(), 0);
        }
        let inv_init_smt = match self.expr_to_smt(invariant, &versions_init) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        decls_init.sort();
        decls_init.dedup();

        let query_init = format!(
            "{}\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_init.join("\n"),
            preconditions_init.join("\n"),
            inv_init_smt
        );

        match run_z3(&query_init) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant initiation check failed. Invariant `{}` may not hold on loop entry. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                    return;
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "initiation", &e);
            }
        }

        // --- 2. CHECK PRESERVATION ---
        let mut decls_pres = Vec::new();
        let mut preconditions_pres = Vec::new();
        self.generate_smt_decls_and_preconditions(&vars, &mut decls_pres, &mut preconditions_pres);

        let inv_start_smt = match self.expr_to_smt(invariant, &versions_init) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        let loop_var_start_smt = match self.expr_to_smt(start, &versions_init) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        let loop_var_end_smt = match self.expr_to_smt(end, &versions_init) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        let cond_start_smt = format!(
            "(and (>= {}_{} {}) (< {}_{} {}))",
            loop_var, 0, loop_var_start_smt, loop_var, 0, loop_var_end_smt
        );

        let mut versions_pres = versions_init.clone();
        let mut body_assertions = Vec::new();
        if let Err(why) =
            self.trace_body_statements(&body.stmts, &mut versions_pres, &mut decls_pres, &mut body_assertions)
        {
            return self.smt_unmodellable(span.line, invariant, &why);
        }

        let current_loop_var_ver = versions_pres.get(loop_var).cloned().unwrap_or(0);
        let next_loop_var_ver = current_loop_var_ver + 1;
        versions_pres.insert(loop_var.to_string(), next_loop_var_ver);
        decls_pres.push(format!("(declare-const {}_{} Int)", loop_var, next_loop_var_ver));

        let step_smt = if let Some(st) = step {
            match self.expr_to_smt(st, &versions_pres) {
                Ok(v) => v,
                Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
            }
        } else {
            "1".to_string()
        };
        body_assertions.push(format!(
            "(assert (= {}_{} (+ {}_{} {})))",
            loop_var, next_loop_var_ver, loop_var, current_loop_var_ver, step_smt
        ));

        let inv_end_smt = match self.expr_to_smt(invariant, &versions_pres) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        decls_pres.sort();
        decls_pres.dedup();

        let query_pres = format!(
            "{}\n{}\n(assert {})\n(assert {})\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_pres.join("\n"),
            preconditions_pres.join("\n"),
            inv_start_smt,
            cond_start_smt,
            body_assertions.join("\n"),
            inv_end_smt
        );

        match run_z3(&query_pres) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant preservation check failed. Invariant `{}` is not preserved by the loop body. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "preservation", &e);
            }
        }
    }
}

/// Z3 binaries to try, in priority order.
///
/// `Y_Z3_PATH` wins when set, then bare `z3` from `PATH`. The rest exist
/// because it is easy to have a perfectly good solver installed that the
/// compiler cannot see: a `pip install z3-solver` inside a project virtualenv
/// puts one in `venv/bin/z3`, which the old two-entry search missed - this
/// repo had exactly that, while the type checker reported the solver as
/// missing and waved every invariant through.
fn z3_candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("Y_Z3_PATH") {
        if !p.is_empty() {
            v.push(p);
        }
    }
    v.push("z3".to_string());
    for p in [
        "venv/bin/z3",
        "./venv/bin/z3",
        ".venv/bin/z3",
        "./.venv/bin/z3",
        "./z3/build/z3",
        "z3/build/z3",
    ] {
        v.push(p.to_string());
    }
    if let Ok(home) = std::env::var("HOME") {
        v.push(format!("{}/.local/bin/z3", home));
    }
    v
}

fn run_z3(query: &str) -> Result<String, String> {
    let candidates = z3_candidates();
    let mut spawned = None;
    for cand in &candidates {
        let attempt = Command::new(cand)
            .args(&["-smt2", "-in"])
            .env("Z3_GPU_THRESHOLD", "2147483647")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        if let Ok(child) = attempt {
            spawned = Some(child);
            break;
        }
    }
    let mut child = spawned.ok_or_else(|| {
        format!(
            "No Z3 binary could be started. Searched: {}",
            candidates.join(", ")
        )
    })?;

    {
        let stdin = child.stdin.as_mut().ok_or("Failed to open stdin")?;
        stdin.write_all(query.as_bytes()).map_err(|e| e.to_string())?;
    }

    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string());
        return Err(format!("Z3 error (code {}): {}\nQuery:\n{}", code, err, query));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Ident(name, _) => name.clone(),
        Expr::IntLit(val, _) => val.to_string(),
        Expr::FloatLit(val, _) => val.to_string(),
        Expr::StringLit(val, _) => format!("\"{}\"", val),
        Expr::CharLit(val, _) => format!("'{}'", val),
        Expr::BoolLit(val, _) => val.to_string(),
        Expr::BinaryOp { left, op, right, .. } => {
            let op_str = match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
                BinaryOp::Mod => "%",
                BinaryOp::Eq => "==",
                BinaryOp::NotEq => "!=",
                BinaryOp::Lt => "<",
                BinaryOp::Gt => ">",
                BinaryOp::Le => "<=",
                BinaryOp::Ge => ">=",
                BinaryOp::And => "&&",
                BinaryOp::Or => "||",
                BinaryOp::BitAnd => "&",
                BinaryOp::BitOr => "|",
                BinaryOp::BitXor => "^",
                BinaryOp::Shl => "<<",
                BinaryOp::Shr => ">>",
            };
            format!("({} {} {})", expr_to_string(left), op_str, expr_to_string(right))
        }
        Expr::UnaryOp { op, operand, .. } => {
            let op_str = match op {
                UnaryOp::Neg => "-",
                UnaryOp::Not => "!",
                UnaryOp::Ref => "&",
                UnaryOp::Deref => "*",
            };
            format!("{}{}", op_str, expr_to_string(operand))
        }
        _ => format!("{:?}", expr),
    }
}

fn is_shared_resource(ty: &SemanticType) -> bool {
    match ty {
        SemanticType::Array { .. } => true,
        SemanticType::SharedMemoryTile { .. } => true,
        SemanticType::GlobalMemory(_) => true,
        SemanticType::Primitive(name) => name == "ptr",
        _ => false,
    }
}

struct CoherenceAnalyzer<'a> {
    type_checker: &'a TypeChecker,
    segments: Vec<BarrierSegment>,
    current_segment: BarrierSegment,
}

#[derive(Clone, Default)]
struct BarrierSegment {
    reads: std::collections::HashMap<String, Span>,
    writes: std::collections::HashMap<String, Span>,
    barrier_span: Option<Span>,
}

impl<'a> CoherenceAnalyzer<'a> {
    fn analyze_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.analyze_stmt(stmt);
        }
    }

    fn collect_reads_writes(&mut self, expr: &Expr, is_write: bool) {
        self.collect_expr_accesses(expr, is_write);
    }

    fn collect_expr_accesses(&mut self, expr: &Expr, is_write: bool) {
        match expr {
            Expr::Ident(name, span) => {
                if let Some(ty) = self.type_checker.lookup_var(name) {
                    if is_shared_resource(ty) {
                        if is_write {
                            self.current_segment.writes.insert(name.clone(), span.clone());
                        } else {
                            self.current_segment.reads.insert(name.clone(), span.clone());
                        }
                    }
                }
            }
            Expr::Index { base, index, .. } => {
                self.collect_expr_accesses(base, is_write);
                self.collect_expr_accesses(index, false);
            }
            Expr::MemberAccess { base, .. } => {
                self.collect_expr_accesses(base, is_write);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_expr_accesses(left, false);
                self.collect_expr_accesses(right, false);
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_expr_accesses(operand, is_write);
            }
            Expr::Call { func, args, .. } => {
                if let Expr::Ident(fname, _) = &**func {
                    if fname == "cp_async" && args.len() >= 2 {
                        self.collect_expr_accesses(&args[0], false);
                        self.collect_expr_accesses(&args[1], true);
                    } else if fname == "store" && args.len() >= 2 {
                        self.collect_expr_accesses(&args[0], true);
                        self.collect_expr_accesses(&args[1], false);
                    } else if (fname == "load" || fname == "ldmatrix") && !args.is_empty() {
                        self.collect_expr_accesses(&args[0], false);
                    } else if fname == "mma_sync" {
                        for arg in args {
                            self.collect_expr_accesses(arg, false);
                        }
                    } else {
                        for arg in args {
                            self.collect_expr_accesses(arg, false);
                        }
                    }
                } else {
                    self.collect_expr_accesses(func, false);
                    for arg in args {
                        self.collect_expr_accesses(arg, false);
                    }
                }
            }
            Expr::GenericCall { func, args, .. } => {
                self.collect_expr_accesses(func, false);
                for arg in args {
                    self.collect_expr_accesses(arg, false);
                }
            }
            Expr::StructLit { fields, .. } => {
                for (_, f_expr) in fields {
                    self.collect_expr_accesses(f_expr, false);
                }
            }
            _ => {}
        }
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { init, .. } => {
                if let Some(init_expr) = init {
                    self.collect_reads_writes(init_expr, false);
                }
            }
            Stmt::Assign { target, value, .. } => {
                self.collect_reads_writes(value, false);
                self.collect_reads_writes(target, true);
            }
            Stmt::Expr(expr) => {
                let is_barrier = match expr {
                    Expr::Path { namespace, member, .. } => namespace == "barrier" && member == "sync",
                    Expr::Call { func, .. } => match &**func {
                        Expr::Path { namespace, member, .. } => namespace == "barrier" && member == "sync",
                        Expr::Ident(fname, _) => fname == "membar" || fname == "barrier_sync",
                        _ => false,
                    },
                    _ => false,
                };
                
                if is_barrier {
                    let prev_segment = std::mem::take(&mut self.current_segment);
                    self.segments.push(prev_segment);
                    self.current_segment = BarrierSegment {
                        reads: std::collections::HashMap::new(),
                        writes: std::collections::HashMap::new(),
                        barrier_span: Some(expr.span()),
                    };
                } else {
                    self.collect_reads_writes(expr, false);
                }
            }
            Stmt::For { body, start, end, .. } => {
                self.collect_reads_writes(start, false);
                self.collect_reads_writes(end, false);
                self.analyze_block(body);
            }
            Stmt::While { body, condition, .. } => {
                self.collect_reads_writes(condition, false);
                self.analyze_block(body);
            }
            Stmt::If { condition, then_block, else_block, .. } => {
                self.collect_reads_writes(condition, false);
                self.analyze_block(then_block);
                if let Some(el) = else_block {
                    self.analyze_block(el);
                }
            }
            _ => {}
        }
    }
}

impl TypeChecker {
    fn verify_kernel_coherence(&mut self, kernel: &KernelDecl) {
        let segments = {
            let mut analyzer = CoherenceAnalyzer {
                type_checker: self,
                segments: Vec::new(),
                current_segment: BarrierSegment::default(),
            };

            analyzer.analyze_block(&kernel.body);
            analyzer.segments.push(analyzer.current_segment);
            analyzer.segments
        };

        for (idx, segment) in segments.iter().enumerate() {
            // 1. Check RAW / WAR hazards (read and write to same variable on different lines)
            for (var_name, read_span) in &segment.reads {
                if let Some(write_span) = segment.writes.get(var_name) {
                    if read_span.line != write_span.line {
                        let second_line = std::cmp::max(read_span.line, write_span.line);
                        self.errors.push(format!(
                            "Line {}: [Coherence Hazard] Read-After-Write (or Write-After-Read) hazard detected on shared/global memory `{}`. Accesses at line {} and line {} are not separated by a `barrier::sync()`.",
                            second_line, var_name, read_span.line, write_span.line
                        ));
                    }
                }
            }

            // 2. Check redundant barriers (optimize barrier placement)
            if idx + 1 < segments.len() {
                if let Some(next_barrier_span) = &segments[idx + 1].barrier_span {
                    if segment.writes.is_empty() {
                        println!(
                            "    [Warning] Line {}: [Barrier Optimization] Redundant barrier synchronization. No shared memory writes occurred since the last barrier.",
                            next_barrier_span.line
                        );
                    }
                }
            }
        }
    }

    fn types_are_compatible(&self, t1: &SemanticType, t2: &SemanticType) -> bool {
        if t1 == t2 {
            return true;
        }
        let is_int_or_ptr = |t: &SemanticType| {
            if let SemanticType::Primitive(p) = t {
                let p_lower = p.to_lowercase();
                p_lower == "i8" || p_lower == "i16" || p_lower == "i32" || p_lower == "i64" ||
                p_lower == "u8" || p_lower == "u16" || p_lower == "u32" || p_lower == "u64" ||
                p_lower == "ptr"
            } else {
                false
            }
        };
        is_int_or_ptr(t1) && is_int_or_ptr(t2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_checker_starts_with_clean_state() {
        let tc = TypeChecker::new();

        assert!(tc.errors.is_empty());
        assert!(!tc.in_unsafe);
    }

    #[test]
    fn test_enum_item_does_not_produce_type_errors() {
        let mut tc = TypeChecker::new();
        let program = Program {
            items: vec![Item::Enum(EnumDecl {
                name: "TestEnum".into(),
                generic_params: vec![],
                variants: vec![],
                span: Span { line: 0, col: 0 },
            })],
        };

        tc.check_program(&program);

        assert!(tc.errors.is_empty());
    }

    #[test]
    fn test_eval_interval_div() {
        let tc = TypeChecker::new();

        // 1. Division by interval containing zero -> None
        let expr_zero = Expr::BinaryOp {
            left: Box::new(Expr::IntLit(10, Span { line: 0, col: 0 })),
            op: BinaryOp::Div,
            right: Box::new(Expr::IntLit(0, Span { line: 0, col: 0 })),
            span: Span { line: 0, col: 0 },
        };
        assert!(tc.eval_interval(&expr_zero).is_none());

        // 2. Division by positive divisor interval
        let expr_pos = Expr::BinaryOp {
            left: Box::new(Expr::IntLit(20, Span { line: 0, col: 0 })),
            op: BinaryOp::Div,
            right: Box::new(Expr::IntLit(4, Span { line: 0, col: 0 })),
            span: Span { line: 0, col: 0 },
        };
        let res = tc.eval_interval(&expr_pos).unwrap();
        assert_eq!(res.min, 5);
        assert_eq!(res.max, 5);
    }

    fn parse_src(src: &str) -> Program {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        parser.parse_program().expect("parse should succeed")
    }

    #[test]
    fn test_kernel_level_tile_valid_shape_passes() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(tc.errors.is_empty(), "unexpected errors: {:?}", tc.errors);
    }

    #[test]
    fn test_kernel_level_tile_fused_bias_relu_shape_passes() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel fused_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, Bias: GlobalMemory<F32>, C: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(tc.errors.is_empty(), "unexpected errors: {:?}", tc.errors);
    }

    #[test]
    fn test_kernel_level_tile_swiglu_shape_passes() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel fused_swiglu(X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(tc.errors.is_empty(), "unexpected errors: {:?}", tc.errors);
    }

    #[test]
    fn test_kernel_level_tile_rejects_wrong_param_shape() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel bad_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, M: I32) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(
            tc.errors.iter().any(|e| e.contains("requires 3 parameters")),
            "expected a validation error for a non-GlobalMemory<F16> param, got: {:?}",
            tc.errors
        );
    }

    #[test]
    fn test_kernel_level_tile_rejects_missing_k() {
        let program = parse_src(
            "
            @tile(4096, 4096)
            kernel bad_gemm2(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(
            tc.errors.iter().any(|e| e.contains("requires K")),
            "expected a missing-K validation error, got: {:?}",
            tc.errors
        );
    }
}

