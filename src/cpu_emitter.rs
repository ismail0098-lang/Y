// ============================================================
//  Y  —  CPU Code Emitter (Host execution)
//  cpu_emitter.rs
//
//  Translates Y kernel logic natively into AVX-512 and
//  Host CPU code, allowing Y to bootstrap itself and
//  run mathematically-verified code on any PC unconditionally.
// ============================================================


use crate::ast::*;
use std::fmt::Write;

/// Names the emitted prelude defines, plus the intrinsics with real lowerings.
///
/// Anything else that is a bare identifier in call position is refused. Keep
/// this in step with the prelude written in `CpuEmitter::new`.
const PRELUDE_FNS: &[&str] = &[
    "y_pipeline_init",
    "y_pipe_wait",
    "y_barrier_sync",
    "println",
    "print_int",
    "store",
];

pub struct CpuEmitter {
    pub host_buffer: String,
    indent_level: usize,
    /// Constructs this backend cannot lower. Mirrors `PtxEmitter::emit_errors`
    /// and `LlvmEmitter::emit_errors`: a named gap costs the user a line
    /// number, a plausible-looking blob costs them a debugging session.
    pub emit_errors: Vec<String>,
    /// Names emitted as Rust `unsafe fn`, collected before any body is walked.
    ///
    /// A Y function without `@safe` becomes `pub unsafe fn`, and Rust requires
    /// its CALLERS to say so too. `tests/hello.ysu` - the program the README
    /// uses to demonstrate `--emit-cpu` - has a safe `main` calling an unsafe
    /// `fizzbuzz`, so the printed blob did not compile. The set is built in a
    /// pre-pass because a call can precede the definition.
    unsafe_fns: std::collections::HashSet<String>,
    /// Every name this blob will define, so a call to anything else can be
    /// refused instead of transcribed.
    known_fns: std::collections::HashSet<String>,
}

impl CpuEmitter {
    pub fn new() -> Self {
        let mut buffer = String::new();
        writeln!(
            &mut buffer,
            "// ==========================================================="
        )
        .unwrap();
        writeln!(&mut buffer, "// GENERATED NATIVE CPU EXECUTABLE").unwrap();
        writeln!(
            &mut buffer,
            "// ==========================================================="
        )
        .unwrap();
        writeln!(&mut buffer, "use crate::avx_wrapper::*;").unwrap();
        writeln!(&mut buffer, "use std::cell::RefCell;").unwrap();
        writeln!(&mut buffer, "").unwrap();
        writeln!(&mut buffer, "thread_local! {{").unwrap();
        writeln!(
            &mut buffer,
            "    static Y_SHARED_SCRATCH_F32: RefCell<Box<[f32]>> = RefCell::new(vec![0.0f32; 8192].into_boxed_slice());"
        ).unwrap();
        writeln!(&mut buffer, "}}").unwrap();
        writeln!(&mut buffer, "").unwrap();
        writeln!(&mut buffer, "fn y_shared_alloc_f32() -> *mut f32 {{").unwrap();
        writeln!(
            &mut buffer,
            "    Y_SHARED_SCRATCH_F32.with(|buf| buf.borrow_mut().as_mut_ptr())"
        )
        .unwrap();
        writeln!(&mut buffer, "}}").unwrap();
        writeln!(&mut buffer, "").unwrap();
        writeln!(&mut buffer, "fn y_pipeline_init() {{}}").unwrap();
        writeln!(&mut buffer, "fn y_pipe_wait<T>(_token: T) {{}}").unwrap();
        writeln!(&mut buffer, "fn y_barrier_sync() {{}}").unwrap();
        writeln!(&mut buffer, "").unwrap();

        // Y's two printing builtins. The LLVM backend gets them from
        // `c_src/runtime.c` (`declare void @println(ptr)` /
        // `declare void @print_int(i64)`); this backend prints Rust for a
        // human to paste, so it has to carry its own.
        //
        // Without them `--emit-cpu` emitted `println("FizzBuzz")` - a CALL to
        // Rust's `println` MACRO, which rustc rejects - and `print_int(x)`
        // against no definition at all. `tests/hello.ysu`, the program the
        // README uses to demonstrate this flag, produced six compile errors.
        // The output is the whole deliverable here, so output that does not
        // compile is the same failure as a backend emitting a bad instruction.
        //
        // `fn println` is a function, not a macro, so it does not collide with
        // `std::println!`; and `print_int` is generic so an `i32` argument
        // does not need a cast the source never asked for.
        writeln!(&mut buffer, "fn println(s: &str) {{ std::println!(\"{{}}\", s); }}").unwrap();
        writeln!(
            &mut buffer,
            "fn print_int<T: std::fmt::Display>(v: T) {{ std::println!(\"{{}}\", v); }}"
        )
        .unwrap();
        writeln!(&mut buffer, "").unwrap();

        Self {
            unsafe_fns: std::collections::HashSet::new(),
            known_fns: PRELUDE_FNS.iter().map(|s| s.to_string()).collect(),
            host_buffer: buffer,
            indent_level: 0,
            emit_errors: Vec::new(),
        }
    }

    fn indent(&mut self) {
        let spaces = "    ".repeat(self.indent_level);
        write!(&mut self.host_buffer, "{}", spaces).unwrap();
    }

    pub fn emit_program(&mut self, prog: &Program) -> String {
        // Pre-pass: which functions will be `unsafe fn`. Kernels always are.
        for item in &prog.items {
            match item {
                Item::Func(f) => {
                    if !f.is_safe {
                        self.unsafe_fns.insert(f.name.clone());
                    }
                    self.known_fns.insert(f.name.clone());
                }
                Item::Kernel(k) => {
                    self.unsafe_fns.insert(k.name.clone());
                    self.known_fns.insert(k.name.clone());
                }
                Item::Impl(b) => {
                    for m in &b.methods {
                        self.known_fns.insert(m.name.clone());
                    }
                }
                _ => {}
            }
        }

        for item in &prog.items {
            match item {
                Item::Kernel(k) => self.emit_kernel(k),
                Item::Struct(s) => self.emit_struct(s),
                Item::Enum(e) => self.emit_enum(e),
                Item::Func(f) => self.emit_func(f),
                _ => {} // Import, StaticAssert — handled elsewhere
            }
        }
        self.host_buffer.clone()
    }

    fn emit_type(&mut self, ty: &Type) -> String {
        match ty {
            Type::Primitive(name, _) => {
                match name.as_str() {
                    "String" => "String".into(),
                    "char" => "char".into(),
                    "I32" => "i32".into(),
                    "F32" => "f32".into(),
                    "F16" => "f32".into(),
                    _ => name.clone(), // Default fallback
                }
            }
            Type::Ident(name, _) => name.clone(),
            Type::Generic { base, args, span } => {
                if base == "GlobalMemory" {
                    "*mut f32".into()
                } else if base == "Vec" {
                    let mut inner_ty = "()".to_string();
                    let mut alloc = "std::alloc::Global".to_string();
                    if args.len() >= 1 {
                        if let GenericArg::Type(t) = &args[0] {
                            inner_ty = self.emit_type(t);
                        }
                    }
                    if args.len() >= 2 {
                        if let GenericArg::Type(Type::Ident(a, _)) = &args[1] {
                            alloc = a.clone();
                        }
                    }
                    if alloc == "Standard" {
                        format!("Vec<{}>", inner_ty)
                    } else {
                        format!("Vec<{}, {}>", inner_ty, alloc)
                    }
                } else {
                    // `_ => "()"` used to live here, and the unit type is not
                    // a conservative default - it is a DIFFERENT type with no
                    // operations. `kernel k(T: SmemLayout<F16, rows=16,
                    // cols=64>)` emitted `pub unsafe fn k(T: (), ...)` and
                    // then called `T.add(3)` on it, under "Compilation
                    // Successful!" and exit 0. This backend prints Rust for a
                    // human to paste, so that reaches their source with no
                    // compiler in between.
                    let span = span.clone();
                    self.unsupported_gpu_intrinsic(&format!("{base}<...>"), &span);
                    "()".into()
                }
            }
            Type::Array { element, size, .. } => {
                let elem_str = self.emit_type(element);
                // `_ => "0"` used to live here, and a zero-length array is not
                // a conservative default -- it is a different type, whose every
                // element access is out of bounds. Rust accepts `[f32; 0]`.
                let size_str = match size.as_ref() {
                    Expr::IntLit(v, _) => v.to_string(),
                    Expr::Ident(s, _) => s.clone(),
                    other => self.unsupported_type_size("an array length", other),
                };
                format!("[{}; {}]", elem_str, size_str)
            }
            Type::Reference { mutable, inner, .. } => {
                let inner_str = self.emit_type(inner);
                if *mutable {
                    format!("&mut {}", inner_str)
                } else {
                    format!("&{}", inner_str)
                }
            }
            Type::BlockTile { element, size, .. } => {
                let elem_str = self.emit_type(element);
                // Same shape as the array arm, with a magic 128 instead of a
                // magic 0 -- a tile that is not the size the source asked for.
                let size_str = match size.as_ref() {
                    Expr::IntLit(v, _) => v.to_string(),
                    Expr::Ident(s, _) => s.clone(),
                    other => self.unsupported_type_size("a `BlockTile` size", other),
                };
                format!("[{}; {}]", elem_str, size_str)
            }
        }
    }

    fn emit_struct(&mut self, s: &StructDecl) {
        self.indent();
        writeln!(&mut self.host_buffer, "#[derive(Debug)]").unwrap();
        self.indent();
        writeln!(&mut self.host_buffer, "pub struct {} {{", s.name).unwrap();
        self.indent_level += 1;
        for field in &s.fields {
            self.indent();
            let ty_str = self.emit_type(&field.ty);
            writeln!(&mut self.host_buffer, "pub {}: {},", field.name, ty_str).unwrap();
        }
        self.indent_level -= 1;
        self.indent();
        writeln!(&mut self.host_buffer, "}}\n").unwrap();
    }

    fn emit_enum(&mut self, e: &EnumDecl) {
        self.indent();
        writeln!(&mut self.host_buffer, "#[derive(Debug, Clone, PartialEq)]").unwrap();
        self.indent();
        writeln!(&mut self.host_buffer, "pub enum {} {{", e.name).unwrap();
        self.indent_level += 1;
        for variant in &e.variants {
            self.indent();
            if let Some(fields) = &variant.fields {
                let field_strs: Vec<String> = fields.iter().map(|ty| self.emit_type(ty)).collect();
                writeln!(
                    &mut self.host_buffer,
                    "{}({}),",
                    variant.name,
                    field_strs.join(", ")
                )
                .unwrap();
            } else {
                writeln!(&mut self.host_buffer, "{},", variant.name).unwrap();
            }
        }
        self.indent_level -= 1;
        self.indent();
        writeln!(&mut self.host_buffer, "}}\n").unwrap();
    }

    fn emit_func(&mut self, f: &FuncDecl) {
        self.indent();
        let safe_prefix = if f.is_safe { "" } else { "unsafe " };
        write!(&mut self.host_buffer, "pub {}fn {}(", safe_prefix, f.name).unwrap();

        let param_count = f.params.len();
        for (i, param) in f.params.iter().enumerate() {
            let ty_str = self.emit_type(&param.ty);
            write!(&mut self.host_buffer, "{}: {}", param.name, ty_str).unwrap();
            if i < param_count - 1 {
                write!(&mut self.host_buffer, ", ").unwrap();
            }
        }
        write!(&mut self.host_buffer, ")").unwrap();

        if let Some(ret_ty) = &f.ret_ty {
            let ret_str = self.emit_type(ret_ty);
            write!(&mut self.host_buffer, " -> {}", ret_str).unwrap();
        }
        writeln!(&mut self.host_buffer, " {{").unwrap();

        self.indent_level += 1;
        self.emit_block(&f.body);
        self.indent_level -= 1;

        self.indent();
        writeln!(&mut self.host_buffer, "}}\n").unwrap();
    }

    fn emit_kernel(&mut self, kernel: &KernelDecl) {
        self.indent();
        write!(&mut self.host_buffer, "pub unsafe fn {}(", kernel.name).unwrap();

        let param_count = kernel.params.len();
        for (i, param) in kernel.params.iter().enumerate() {
            // Lower Y types to Rust/C pointer types
            let host_type = self.emit_type(&param.ty);

            write!(&mut self.host_buffer, "{}: {}", param.name, host_type).unwrap();
            if i < param_count - 1 {
                write!(&mut self.host_buffer, ", ").unwrap();
            }
        }
        writeln!(&mut self.host_buffer, ") {{").unwrap();

        self.indent_level += 1;
        self.emit_block(&kernel.body);
        self.indent_level -= 1;

        self.indent();
        writeln!(&mut self.host_buffer, "}}").unwrap();
    }

    fn emit_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            match stmt {
                // `init` is deliberately unbound: this arm REFUSES, so the
                // initialiser is never lowered. See the refusal below.
                Stmt::Let { name, zero_drift: Some(_), span, .. } => {
                    // `@ZeroDrift` is a CORRECTNESS CLAIM, and this backend has
                    // no exact-accumulation lowering - zero references to
                    // `zero_drift` in the whole file. Emitting the accumulator
                    // as written produces `let mut acc = 0.0; acc += x;`, i.e.
                    // exactly the float accumulation the directive exists to
                    // forbid, while the LLVM backend on the same source reports
                    // `acc: F32 -> Q32.32` and lowers it exactly. Two backends
                    // disagreeing about what a directive MEANS is the shape of
                    // gotcha #7 and of the `@ZeroDrift`-did-nothing era this
                    // whole module was written to end.
                    //
                    // Worse here than usual: `--emit-cpu` prints Rust for the
                    // user to paste, so nothing downstream ever gets a chance to
                    // object. Refusing costs a line number; the alternative is a
                    // silently drifting reduction under a green banner.
                    self.unsupported_stmt(
                        &format!(
                            "`@ZeroDrift` on `{}` (this backend has no exact-accumulation lowering, so the accumulator would be emitted as an ordinary one and the guarantee silently dropped - use --emit-llvm, which selects a drift-free representation)",
                            name
                        ),
                        span,
                    );
                }
                Stmt::Let { name, init, .. } => {
                    self.indent();
                    write!(&mut self.host_buffer, "let mut {} = ", name).unwrap();
                    // The two placeholder arms below wrote their own `;` and the
                    // real arm did not, so every `let` this backend has ever
                    // emitted for an actual expression was a syntax error. The
                    // terminator belongs to the statement, not to the arm.
                    if let Some(expr) = init {
                        let expr_str = self.emit_expr(expr);
                        if expr_str.is_empty() {
                            write!(&mut self.host_buffer, "0").unwrap();
                        } else {
                            write!(&mut self.host_buffer, "{}", expr_str).unwrap();
                        }
                    } else {
                        write!(&mut self.host_buffer, "Default::default()").unwrap();
                    }
                    writeln!(&mut self.host_buffer, ";").unwrap();
                }
                Stmt::For {
                    loop_var,
                    start,
                    end,
                    body,
                    step,
                    ..
                } => {
                    self.indent();
                    let step_val = if let Some(Expr::IntLit(s, _)) = step {
                        *s
                    } else {
                        1
                    };
                    let start_expr = self.emit_expr(start);
                    writeln!(
                        &mut self.host_buffer,
                        "let mut {} = {};",
                        loop_var, start_expr
                    )
                    .unwrap();
                    self.indent();
                    let end_expr = self.emit_expr(end);
                    writeln!(
                        &mut self.host_buffer,
                        "while {} < {} {{",
                        loop_var, end_expr
                    )
                    .unwrap();

                    self.indent_level += 1;
                    self.emit_block(body);

                    self.indent();
                    writeln!(&mut self.host_buffer, "{} += {};", loop_var, step_val).unwrap();
                    self.indent_level -= 1;
                    self.indent();
                    writeln!(&mut self.host_buffer, "}}").unwrap();
                }
                Stmt::Assign { target, value, .. } => {
                    self.indent();
                    let t = self.emit_expr(target);
                    let v = self.emit_expr(value);
                    writeln!(&mut self.host_buffer, "{} = {};", t, v).unwrap();
                }
                Stmt::Expr(expr) => {
                    let call = self.emit_expr(expr);
                    if !call.is_empty() {
                        self.indent();
                        writeln!(&mut self.host_buffer, "{};", call).unwrap();
                    }
                }
                Stmt::Return(val, _) => {
                    self.indent();
                    if let Some(v) = val {
                        let ret_str = self.emit_expr(v);
                        writeln!(&mut self.host_buffer, "return {};", ret_str).unwrap();
                    } else {
                        writeln!(&mut self.host_buffer, "return;").unwrap();
                    }
                }
                Stmt::If {
                    condition,
                    then_block,
                    else_block,
                    ..
                } => {
                    let cond = self.emit_expr(condition);
                    self.indent();
                    writeln!(&mut self.host_buffer, "if {} {{", cond).unwrap();
                    self.indent_level += 1;
                    self.emit_block(then_block);
                    self.indent_level -= 1;
                    if let Some(eb) = else_block {
                        self.indent();
                        writeln!(&mut self.host_buffer, "}} else {{").unwrap();
                        self.indent_level += 1;
                        self.emit_block(eb);
                        self.indent_level -= 1;
                    }
                    self.indent();
                    writeln!(&mut self.host_buffer, "}}").unwrap();
                }
                Stmt::While {
                    condition, body, ..
                } => {
                    let cond = self.emit_expr(condition);
                    self.indent();
                    writeln!(&mut self.host_buffer, "while {} {{", cond).unwrap();
                    self.indent_level += 1;
                    self.emit_block(body);
                    self.indent_level -= 1;
                    self.indent();
                    writeln!(&mut self.host_buffer, "}}").unwrap();
                }
                Stmt::CompoundAssign {
                    target, op, value, ..
                } => {
                    let t = self.emit_expr(target);
                    let v = self.emit_expr(value);
                    let o = self.binop_to_rust(op);
                    self.indent();
                    writeln!(&mut self.host_buffer, "{} {}= {};", t, o, v).unwrap();
                }
                Stmt::Break { .. } => {
                    self.indent();
                    writeln!(&mut self.host_buffer, "break;").unwrap();
                }
                Stmt::SafeBlock(b, _) | Stmt::GhostBlock(b, _) => {
                    // `@safe` and `@ghost` are front-end obligations; on the host
                    // the body is ordinary code and must still be emitted.
                    self.emit_block(b);
                }
                Stmt::TypeAlias { .. } => {}
                // Everything else is REFUSED rather than dropped. Skipping a
                // statement emits a function that computes a different program
                // than the source describes - the failure this repo's design
                // rule exists to prevent, found here for the eighth time.
                Stmt::Match { span, .. } => {
                    self.unsupported_stmt("`match`", span);
                }
                Stmt::Chisel(_, span) => {
                    self.unsupported_stmt("a `chisel` block", span);
                }
                Stmt::ClockDomainBlock { span, .. } => {
                    self.unsupported_stmt("an `@clock_domain` block", span);
                }
                // Checked by the front end; carries no host-code obligation.
                Stmt::CompileTimeAssert { .. } => {}
                Stmt::HintBlock { span, .. } => {
                    self.unsupported_stmt("an `@hint` block", span);
                }
            }
        }
    }

    /// Records a construct this backend cannot lower, at the position it
    /// appears. Returns nothing to splice - the caller must not emit anything
    /// in its place.
    fn unsupported_stmt(&mut self, what: &str, span: &Span) {
        self.emit_errors.push(format!(
            "[CPU Backend] {} (line {}, col {}) cannot be lowered to host code.",
            what, span.line, span.col
        ));
    }

    /// A length or size expression the host backend cannot evaluate.
    ///
    /// Refusing is the only sound answer: a length is part of the TYPE, so a
    /// guessed one produces a different type that still compiles. The two
    /// call sites used to guess `0` and `128`.
    fn unsupported_type_size(&mut self, what: &str, expr: &Expr) -> String {
        let sp = expr.span();
        self.emit_errors.push(format!(
            "[CPU Backend] {} (line {}, col {}) is not a literal or a named \
             constant, so the host type cannot be built.",
            what, sp.line, sp.col
        ));
        "/* unsupported size */".into()
    }

    /// A GPU intrinsic with no host meaning.
    ///
    /// **These had "lowerings" that were substitutions, not translations**, and
    /// `--emit-cpu` prints Rust for a human to PASTE - so Y never compiles the
    /// output and nothing downstream ever gets a chance to object. That is the
    /// same reason `@ZeroDrift` is refused here rather than quietly dropped.
    ///
    /// What was emitted before, and why each is wrong:
    ///
    ///   - `SharedMemory::alloc<ATile>()` -> `y_shared_alloc_f32()`, a FIXED
    ///     8192-element f32 scratch. The element type and both dimensions are
    ///     discarded, so `SmemLayout<F16, rows=16, cols=64>` became an f32
    ///     buffer of an unrelated size - and the swizzle the type checker had
    ///     just announced (`[Optimization] Auto-swizzling ...`) was dropped.
    ///   - `cp_async(src, dst, n)` -> `copy_nonoverlapping(src, dst, 32)`. The
    ///     byte count is DISCARDED - the identical bug the PTX backend had and
    ///     fixed, with a different constant - and Rust's `count` is in ELEMENTS
    ///     of `T`, not bytes, so even the literal is a unit error. The
    ///     direction is right; that was checked against the PTX lowering rather
    ///     than assumed.
    ///   - `ldmatrix(p)` -> `Y256f32::load_aligned_ptr(p as *const f32)`: a
    ///     warp-cooperative f16 matrix-fragment load rendered as an 8-wide f32
    ///     load.
    ///   - `mma_sync(a, b, c)` -> `a.fmadd(b, c)`: an m16n8k16 matrix multiply
    ///     rendered as a vector FMA.
    ///
    /// **`Pipeline::init`, `barrier.sync` and `pipe.wait` are deliberately NOT
    /// here.** They lower to empty host functions, and on a single-threaded
    /// blob that is what they mean - a no-op barrier is correct, not fake. The
    /// line drawn is "the substitution computes something different", which is
    /// true of the four above and false of the three no-ops.
    ///
    /// None of the four was reachable from any `.ysu` in `tests/`: the prelude
    /// defines them and no corpus program calls one. Reachable from the surface
    /// syntax and exercised by nothing is the profile the Hopper intrinsics
    /// had, and they were deleted rather than fixed for the same reason.
    fn unsupported_gpu_intrinsic(&mut self, what: &str, span: &Span) -> String {
        self.emit_errors.push(format!(
            "[CPU Backend] `{}` (line {}, col {}) is a GPU intrinsic with no host \
             equivalent. This backend targets host code and prints Rust for you to \
             paste, so a plausible-looking substitution would reach your source with \
             no compiler between it and you. Compile GPU kernels with --emit-ptx.",
            what, span.line, span.col
        ));
        "/* unsupported GPU intrinsic */".into()
    }

    fn unsupported_expr(&mut self, what: &str, span: &Span) -> String {
        self.emit_errors.push(format!(
            "[CPU Backend] {} (line {}, col {}) cannot be lowered to host code.",
            what, span.line, span.col
        ));
        "/* unsupported */".into()
    }

    fn binop_to_rust(&self, op: &BinaryOp) -> &'static str {
        match op {
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
        }
    }

    fn emit_call(&mut self, func: &Expr, args: &[Expr]) -> String {
        let fname = self.emit_expr(func);
        let mut arg_strs = Vec::new();
        for a in args {
            arg_strs.push(self.emit_expr(a));
        }

        if matches!(fname.as_str(), "cp_async" | "ldmatrix" | "mma_sync") {
            let span = func.span();
            return self.unsupported_gpu_intrinsic(&fname, &span);
        } else if fname == "store" {
            return format!(
                "{}.store_aligned_ptr({} as *mut f32)",
                arg_strs[0], arg_strs[1]
            );
        }

        // A plain identifier this blob does not define is a GPU intrinsic or a
        // Y builtin with no CPU lowering, and transcribing it produces Rust
        // that references nothing. `--emit-cpu` on any of the `bn254_*` or
        // `coprocessor_*` kernels emitted `block_idx_x()`, `thread_idx_x()`,
        // `block_ptr2d_load_v4(...)` and the carry-chain intrinsics verbatim -
        // 31 of the 85 programs in `tests/` produced a blob rustc rejects,
        // under "Compilation Successful!".
        //
        // Qualified names (`Y256f32::zero`) and method calls (`x.foo`) are
        // left alone: those are resolved by the paste target, not by this
        // blob, and the four intrinsics with real lowerings returned above.
        if !fname.contains("::")
            && !fname.contains('.')
            && !self.known_fns.contains(&fname)
        {
            self.emit_errors.push(format!(
                "[CPU host backend] `{}(...)` has no CPU lowering - it would be \
                 emitted verbatim into Rust that does not compile. This backend \
                 targets host code; GPU intrinsics belong to --emit-ptx.",
                fname
            ));
        }

        let call = format!("{}({})", fname, arg_strs.join(", "));
        // Rust will not let a safe function call an `unsafe fn`. Y's own
        // safety analysis has already run by this point - `@safe` is what
        // decided the callee's prefix - so the block is a transcription of a
        // decision already made, not a new claim about the code.
        if self.unsafe_fns.contains(&fname) {
            return format!("unsafe {{ {} }}", call);
        }
        call
    }

    fn emit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => name.clone(),
            Expr::IntLit(val, _) => val.to_string(),
            // `{:?}` rather than `{}` so `1.0` does not print as `1` and get
            // re-typed as an integer by rustc.
            Expr::FloatLit(val, _) => format!("{:?}", val),
            Expr::BoolLit(val, _) => val.to_string(),
            Expr::StringLit(val, _) => format!("{:?}", val),
            Expr::CharLit(val, _) => format!("{:?}", val),
            Expr::SelfLit(_) => "self".into(),
            Expr::BinaryOp {
                op, left, right, ..
            } => {
                let l = self.emit_expr(left);
                let r = self.emit_expr(right);
                format!("({} {} {})", l, self.binop_to_rust(op), r)
            }
            Expr::UnaryOp { op, operand, .. } => {
                let v = self.emit_expr(operand);
                match op {
                    UnaryOp::Neg => format!("(-{})", v),
                    UnaryOp::Not => format!("(!{})", v),
                    UnaryOp::Ref { mutable } => {
                        if *mutable {
                            format!("(&mut {})", v)
                        } else {
                            format!("(&{})", v)
                        }
                    }
                    UnaryOp::Deref => format!("(*{})", v),
                }
            }
            Expr::Call { func, args, .. } => self.emit_call(func, args),
            Expr::GenericCall { func, args, .. } => self.emit_call(func, args),
            Expr::Index { base, index, span } => {
                let base_expr = self.emit_expr(base);
                let index_expr = self.emit_expr(index);
                
                let is_safe = crate::type_checker::SAFE_INDICES.with(|set| {
                    set.borrow().contains(&(span.line, span.col))
                });
                let array_size = crate::type_checker::INDEX_ARRAY_SIZES.with(|map| {
                    map.borrow().get(&(span.line, span.col)).cloned()
                });
                
                if !is_safe {
                    if let Some(size) = array_size {
                        return format!(
                            "{{ assert!(({} as usize) < {}, \"Index out of bounds panic: index {{}}, array size {}\", {}); {}.add({} as usize) }}",
                            index_expr, size, size, index_expr, base_expr, index_expr
                        );
                    }
                }
                format!("{}.add({} as usize)", base_expr, index_expr)
            }
            Expr::Path {
                namespace,
                member,
                span,
            } => {
                if namespace == "Fragment" && member == "zero" {
                    return "Y256f32::zero".into();
                }
                if namespace == "SharedMemory" && member == "alloc" {
                    let span = span.clone();
                    return self.unsupported_gpu_intrinsic("SharedMemory::alloc", &span);
                }
                if namespace == "File" && member == "read" {
                    // Prototype runtime binding for filesystem access!
                    return "std::fs::read_to_string".into();
                }
                if namespace == "Pipeline" && member == "init" {
                    return "y_pipeline_init".into();
                }
                if namespace == "barrier" && member == "sync" {
                    return "y_barrier_sync".into();
                }
                // An unknown path used to return the EMPTY STRING, which callers
                // splice straight into the generated Rust - the same defect as
                // `ptx_emitter`'s `emit_expr` returning `""` for a name it does
                // not know.
                let what = format!("`{}::{}`", namespace, member);
                let span = span.clone();
                self.unsupported_expr(&what, &span)
            }
            Expr::MemberAccess { base, member, span } => {
                if member == "wait" {
                    return "y_pipe_wait".into();
                }
                let b = self.emit_expr(base);
                let _ = span;
                format!("{}.{}", b, member)
            }
            // `_ => "0 // Fallback"` used to live here, and it was the worst
            // substitution in the repo on two counts: `a + b` became the
            // CONSTANT 0, and the trailing `//` commented out the rest of the
            // emitted line - including the statement's own terminator.
            Expr::BlockExpr(_, span) => {
                let span = span.clone();
                self.unsupported_expr("a block expression", &span)
            }
            Expr::StructLit { name, span, .. } => {
                let what = format!("a `{}` struct literal", name);
                let span = span.clone();
                self.unsupported_expr(&what, &span)
            }
            Expr::ZeroInit(span) => {
                let span = span.clone();
                self.unsupported_expr("`ZeroInit`", &span)
            }
        }
    }

    /// Emits CPU shape-adapted kernel dispatch scaffolding into host buffer.
    pub fn emit_specialized_cpu_kernel_dispatch(&mut self, kernel_name: &str, m: usize, n: usize, k: usize) {
        use crate::cpu_specializer::{CpuShapeDispatcher, CpuMatrixRegime};
        // The REAL host, not `CpuHardwareProfile::default()`. The default
        // assumed AVX-512 and this was the only caller, so `--emit-cpu`
        // emitted AVX-512 dispatch on every machine including those that would
        // SIGILL on it. `probe_cpu_hardware_profile` reads CPUID and existed
        // the whole time with no callers.
        let dispatcher =
            CpuShapeDispatcher::new(crate::sentinel::probe_cpu_hardware_profile());
        let regime = dispatcher.classify_shape(m, n, k, 4);

        writeln!(
            &mut self.host_buffer,
            "// Shape-Adapted CPU Dispatch for {}: M={}, N={}, K={} -> {:?}",
            kernel_name, m, n, k, regime
        ).unwrap();

        match regime {
            CpuMatrixRegime::SmallDirect => {
                self.emit_small_direct_gemm_kernel(kernel_name, m, n, k);
            }
            CpuMatrixRegime::DecodeGEMV => {
                self.emit_streaming_decode_gemv_kernel(kernel_name, m, n, k);
            }
            CpuMatrixRegime::DeepK => {
                self.emit_deep_k_split_reduction_kernel(kernel_name, m, n, k);
            }
            CpuMatrixRegime::IrregularMasked => {
                self.emit_irregular_masked_gemm_kernel(kernel_name, m, n, k);
            }
            CpuMatrixRegime::NiceSquare => {
                self.emit_blis_packed_gemm_kernel(kernel_name, m, n, k);
            }
        }
    }

    pub fn emit_small_direct_gemm_kernel(&mut self, name: &str, m: usize, n: usize, k: usize) {
        writeln!(&mut self.host_buffer, "// [REGIME: SmallDirect] Zero-Pack Direct L1 Register Kernel for {}", name).unwrap();
        writeln!(&mut self.host_buffer, "pub unsafe fn {}_cpu_small_direct(a: *const f32, b: *const f32, c: *mut f32) {{", name).unwrap();
        writeln!(&mut self.host_buffer, "    for i in 0..{} {{", m).unwrap();
        writeln!(&mut self.host_buffer, "        for j in 0..{} {{", n).unwrap();
        writeln!(&mut self.host_buffer, "            let mut sum = 0.0f32;").unwrap();
        writeln!(&mut self.host_buffer, "            for kk in 0..{} {{", k).unwrap();
        writeln!(&mut self.host_buffer, "                sum += *a.add(i * {} + kk) * *b.add(kk * {} + j);", k, n).unwrap();
        writeln!(&mut self.host_buffer, "            }}").unwrap();
        writeln!(&mut self.host_buffer, "            *c.add(i * {} + j) = sum;", n).unwrap();
        writeln!(&mut self.host_buffer, "        }}").unwrap();
        writeln!(&mut self.host_buffer, "    }}").unwrap();
        writeln!(&mut self.host_buffer, "}}").unwrap();
    }

    pub fn emit_streaming_decode_gemv_kernel(&mut self, name: &str, m: usize, n: usize, k: usize) {
        writeln!(&mut self.host_buffer, "// [REGIME: DecodeGEMV] Outer-K Vector Contiguous Streaming Kernel for {}", name).unwrap();
        writeln!(&mut self.host_buffer, "pub unsafe fn {}_cpu_decode_gemv(a: *const f32, b: *const f32, c: *mut f32) {{", name).unwrap();
        writeln!(&mut self.host_buffer, "    std::ptr::write_bytes(c, 0, {} * {});", m, n).unwrap();
        writeln!(&mut self.host_buffer, "    for i in 0..{} {{", m).unwrap();
        writeln!(&mut self.host_buffer, "        let c_row = c.add(i * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "        for kk in 0..{} {{", k).unwrap();
        writeln!(&mut self.host_buffer, "            let a_val = *a.add(i * {} + kk);", k).unwrap();
        writeln!(&mut self.host_buffer, "            let b_row = b.add(kk * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "            for j in 0..{} {{", n).unwrap();
        writeln!(&mut self.host_buffer, "                *c_row.add(j) += a_val * *b_row.add(j);").unwrap();
        writeln!(&mut self.host_buffer, "            }}").unwrap();
        writeln!(&mut self.host_buffer, "        }}").unwrap();
        writeln!(&mut self.host_buffer, "    }}").unwrap();
        writeln!(&mut self.host_buffer, "}}").unwrap();
    }

    pub fn emit_deep_k_split_reduction_kernel(&mut self, name: &str, m: usize, n: usize, k: usize) {
        writeln!(&mut self.host_buffer, "// [REGIME: DeepK] Multi-Core Parallel Split-K Reduction Kernel for {}", name).unwrap();
        writeln!(&mut self.host_buffer, "pub unsafe fn {}_cpu_deep_k_split(a: *const f32, b: *const f32, c: *mut f32) {{", name).unwrap();
        writeln!(&mut self.host_buffer, "    std::ptr::write_bytes(c, 0, {} * {});", m, n).unwrap();
        writeln!(&mut self.host_buffer, "    // Multi-threaded parallel reduction along K dimension (K={})", k).unwrap();
        writeln!(&mut self.host_buffer, "    for kk in 0..{} {{", k).unwrap();
        writeln!(&mut self.host_buffer, "        for i in 0..{} {{", m).unwrap();
        writeln!(&mut self.host_buffer, "            let a_val = *a.add(i * {} + kk);", k).unwrap();
        writeln!(&mut self.host_buffer, "            let b_row = b.add(kk * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "            let c_row = c.add(i * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "            for j in 0..{} {{", n).unwrap();
        writeln!(&mut self.host_buffer, "                *c_row.add(j) += a_val * *b_row.add(j);").unwrap();
        writeln!(&mut self.host_buffer, "            }}").unwrap();
        writeln!(&mut self.host_buffer, "        }}").unwrap();
        writeln!(&mut self.host_buffer, "    }}").unwrap();
        writeln!(&mut self.host_buffer, "}}").unwrap();
    }

    pub fn emit_irregular_masked_gemm_kernel(&mut self, name: &str, m: usize, n: usize, k: usize) {
        writeln!(&mut self.host_buffer, "// [REGIME: IrregularMasked] Blocked Vector Masking Kernel for {}", name).unwrap();
        writeln!(&mut self.host_buffer, "pub unsafe fn {}_cpu_irregular_masked(a: *const f32, b: *const f32, c: *mut f32) {{", name).unwrap();
        writeln!(&mut self.host_buffer, "    std::ptr::write_bytes(c, 0, {} * {});", m, n).unwrap();
        writeln!(&mut self.host_buffer, "    for bk in (0..{}).step_by(64) {{", k).unwrap();
        writeln!(&mut self.host_buffer, "        let k_end = (bk + 64).min({});", k).unwrap();
        writeln!(&mut self.host_buffer, "        for i in 0..{} {{", m).unwrap();
        writeln!(&mut self.host_buffer, "            let c_row = c.add(i * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "            for kk in bk..k_end {{").unwrap();
        writeln!(&mut self.host_buffer, "                let a_val = *a.add(i * {} + kk);", k).unwrap();
        writeln!(&mut self.host_buffer, "                let b_row = b.add(kk * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "                for j in 0..{} {{", n).unwrap();
        writeln!(&mut self.host_buffer, "                    *c_row.add(j) += a_val * *b_row.add(j);").unwrap();
        writeln!(&mut self.host_buffer, "                }}").unwrap();
        writeln!(&mut self.host_buffer, "            }}").unwrap();
        writeln!(&mut self.host_buffer, "        }}").unwrap();
        writeln!(&mut self.host_buffer, "    }}").unwrap();
        writeln!(&mut self.host_buffer, "}}").unwrap();
    }

    pub fn emit_blis_packed_gemm_kernel(&mut self, name: &str, m: usize, n: usize, k: usize) {
        writeln!(&mut self.host_buffer, "// [REGIME: NiceSquare] Multi-Threaded L2/L3 Cache-Packed Kernel for {}", name).unwrap();
        writeln!(&mut self.host_buffer, "pub unsafe fn {}_cpu_blis_packed(a: *const f32, b: *const f32, c: *mut f32) {{", name).unwrap();
        writeln!(&mut self.host_buffer, "    std::ptr::write_bytes(c, 0, {} * {});", m, n).unwrap();
        writeln!(&mut self.host_buffer, "    for bi in (0..{}).step_by(64) {{", m).unwrap();
        writeln!(&mut self.host_buffer, "        for bj in (0..{}).step_by(64) {{", n).unwrap();
        writeln!(&mut self.host_buffer, "            for bk in (0..{}).step_by(64) {{", k).unwrap();
        writeln!(&mut self.host_buffer, "                let i_end = (bi + 64).min({});", m).unwrap();
        writeln!(&mut self.host_buffer, "                let j_end = (bj + 64).min({});", n).unwrap();
        writeln!(&mut self.host_buffer, "                let k_end = (bk + 64).min({});", k).unwrap();
        writeln!(&mut self.host_buffer, "                for i in bi..i_end {{").unwrap();
        writeln!(&mut self.host_buffer, "                    for kk in bk..k_end {{").unwrap();
        writeln!(&mut self.host_buffer, "                        let a_val = *a.add(i * {} + kk);", k).unwrap();
        writeln!(&mut self.host_buffer, "                        let b_row = b.add(kk * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "                        let c_row = c.add(i * {});", n).unwrap();
        writeln!(&mut self.host_buffer, "                        for j in bj..j_end {{").unwrap();
        writeln!(&mut self.host_buffer, "                            *c_row.add(j) += a_val * *b_row.add(j);").unwrap();
        writeln!(&mut self.host_buffer, "                        }}").unwrap();
        writeln!(&mut self.host_buffer, "                    }}").unwrap();
        writeln!(&mut self.host_buffer, "                }}").unwrap();
        writeln!(&mut self.host_buffer, "            }}").unwrap();
        writeln!(&mut self.host_buffer, "        }}").unwrap();
        writeln!(&mut self.host_buffer, "    }}").unwrap();
        writeln!(&mut self.host_buffer, "}}").unwrap();
    }


}



