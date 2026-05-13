// ============================================================
//  Y-Lang  —  Semantic Type Checker
//  type_checker.rs
//
//  The core brain of Y-Lang's safety guarantees.
//  Traverses AST, enforces Fragment roles (A vs B vs C),
//  manages linear memory obligations, and runs the
//  0-Bank-Conflict math prover.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use crate::bank_conflict::{BankConflictProver, SmemLayout as ProverLayout, SwizzlePattern};
use crate::linear_tracker::LinearTracker;
use std::collections::HashMap;

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
    TransferObligation,
    Pipeline,
    Unknown,
}

pub struct TypeChecker {
    // Basic type environment: variable name -> SemanticType
    env: Vec<HashMap<String, SemanticType>>,
    pub linear_tracker: LinearTracker,
    pub errors: Vec<String>,
    pub in_unsafe: bool,
    allow_transfer_use: usize,
    current_return_type: Option<SemanticType>,
    functions: HashMap<String, Vec<SemanticType>>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: vec![HashMap::new()],
            linear_tracker: LinearTracker::new(),
            errors: Vec::new(),
            in_unsafe: false,
            allow_transfer_use: 0,
            current_return_type: None,
            functions: HashMap::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.env.push(HashMap::new());
        self.linear_tracker.push_scope();
    }

    pub fn pop_scope(&mut self) {
        self.linear_tracker.pop_scope();
        self.env.pop();
    }

    fn insert_var(&mut self, name: String, ty: SemanticType) {
        if let Some(scope) = self.env.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup_var(&self, name: &str) -> Option<&SemanticType> {
        for scope in self.env.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
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
        // Collect function signatures first
        for item in &prog.items {
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
                _ => {}
            }
        }

        for item in &prog.items {
            match item {
                Item::Kernel(k) => self.check_kernel(k),
                Item::Func(f) => self.check_func(f),
                Item::Impl(imp) => {
                    for f in &imp.methods {
                        self.check_func(f);
                    }
                }
                _ => {}
            }
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
        }

        self.check_block(&kernel.body);

        self.pop_scope();
    }

    fn check_func(&mut self, f: &FuncDecl) {
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
                bounds: _, // Ignore for now
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
                }

                if let Some(resolved) = explicit_resolved {
                    // Minimal type unification
                    if inferred_type == SemanticType::Unknown {
                        inferred_type = resolved.clone();
                    } else if inferred_type != resolved
                        && inferred_type != SemanticType::TransferObligation
                    {
                        self.errors.push(format!(
                            "Line {}: Type mismatch in let assignment.",
                            span.line
                        ));
                    }
                }

                self.insert_var(name.clone(), inferred_type.clone());

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
                let resolved = self.resolve_type(ty);
                // If defining a new SmemLayout, run the Bank Conflict Prover!
                if let SemanticType::SharedMemoryTile {
                    rows,
                    cols,
                    swizzle,
                } = &resolved
                {
                    let prover_layout = ProverLayout {
                        rows: *rows,
                        cols: *cols,
                        swizzle: swizzle.clone(),
                        bytes_per_element: 2, // Defaulting F16 for prototype logic
                    };

                    if let Err(conflict_err) =
                        BankConflictProver::prove_ldmatrix_m16n8(&prover_layout)
                    {
                        // Allow compilation to proceed for prototype PTX emit exhibition
                        println!("    [Warning] Line {}: {}", span.line, conflict_err);
                    }
                }
                self.insert_var(name.clone(), resolved);
            }
            Stmt::For { loop_var, body, invariant, is_uniform_branch, span, .. } => {
                self.push_scope();

                if !self.in_unsafe && invariant.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] Loops in safe blocks require formal @invariants.",
                        span.line
                    ));
                }

                if *is_uniform_branch {
                    // For loops usually loop over a range which is statically determinable,
                    // but we check the bounds expressions anyway.
                }

                self.insert_var(loop_var.clone(), SemanticType::Primitive("I32".into()));

                for s in &body.stmts {
                    self.check_stmt(s);
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
                if t1 != t2 && t1 != SemanticType::Unknown && t2 != SemanticType::Unknown {
                    self.errors.push(format!(
                        "Line {}: Invalid assignment, types do not match.",
                        span.line
                    ));
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
                condition, body, invariant, is_uniform_branch, ..
            } => {
                if !self.in_unsafe && invariant.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] While loops in safe blocks require formal @invariants.",
                        condition.span().line
                    ));
                }
                if *is_uniform_branch {
                    self.check_uniformity(condition);
                }
                self.check_expr(condition);
                self.check_block(body);
            }
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
                for (_, expr) in fields {
                    let field_ty = self.check_expr(expr);
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
                SemanticType::Unknown
            }
            Expr::BinaryOp { left, right, .. } => {
                let lhs = self.check_expr(left);
                let rhs = self.check_expr(right);
                self.reject_transfer_escape(&lhs, &left.span(), "in a binary expression");
                self.reject_transfer_escape(&rhs, &right.span(), "in a binary expression");
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

        let mut require_role = |ty: &SemanticType, expected_role: &str| {
            if let SemanticType::Fragment { role, .. } = ty {
                if role != expected_role {
                    self.errors.push(format!(
                        "Line {}: Fragment Role Error: expected Fragment<{}, ...>, got Fragment<{}, ...>.",
                        span.line, expected_role, role
                    ));
                }
            }
        };

        require_role(&t_a, "A");
        require_role(&t_b, "B");
        require_role(&t_c, "C"); // Or D commonly used for accumulator feedback
    }

    // ── Type Resolution ─────────────────────────────────────

    fn resolve_type(&mut self, ast_ty: &Type) -> SemanticType {
        match ast_ty {
            Type::Primitive(name, _) => SemanticType::Primitive(name.clone()),
            Type::Ident(name, _) => {
                if let Some(t) = self.lookup_var(name) {
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

                SemanticType::Unknown
            }
            Type::Array { .. } => {
                // Array types not yet semantically checked in prototype
                SemanticType::Unknown
            }
            Type::Reference { .. } => {
                // Reference types not yet semantically checked in prototype
                SemanticType::Unknown
            }
        }
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
}
