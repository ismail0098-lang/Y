// ============================================================
//  Y  —  CPU Code Emitter (Host execution)
//  cpu_emitter.rs
//
//  Translates Y kernel logic natively into AVX-512 and
//  Host CPU code, allowing Y to bootstrap itself and
//  run mathematically-verified code on any PC unconditionally.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use std::fmt::Write;

pub struct CpuEmitter {
    pub host_buffer: String,
    indent_level: usize,
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

        Self {
            host_buffer: buffer,
            indent_level: 0,
        }
    }

    fn indent(&mut self) {
        let spaces = "    ".repeat(self.indent_level);
        write!(&mut self.host_buffer, "{}", spaces).unwrap();
    }

    pub fn emit_program(&mut self, prog: &Program) -> String {
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

    fn emit_type(&self, ty: &Type) -> String {
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
            Type::Generic { base, args, .. } => {
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
                    "()".into()
                }
            }
            Type::Array { element, size, .. } => {
                let elem_str = self.emit_type(element);
                let size_str = match size.as_ref() {
                    Expr::IntLit(v, _) => v.to_string(),
                    Expr::Ident(s, _) => s.clone(),
                    _ => "0".into(),
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
                Stmt::Let { name, init, .. } => {
                    self.indent();
                    write!(&mut self.host_buffer, "let mut {} = ", name).unwrap();
                    if let Some(expr) = init {
                        let expr_str = self.emit_expr(expr);
                        if expr_str.is_empty() {
                            write!(&mut self.host_buffer, "0; // placeholder").unwrap();
                        } else {
                            write!(&mut self.host_buffer, "{}", expr_str).unwrap();
                        }
                    } else {
                        write!(&mut self.host_buffer, "0; // uninit").unwrap();
                    }
                    writeln!(&mut self.host_buffer, "").unwrap();
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
                _ => {}
            }
        }
    }

    fn emit_call(&mut self, func: &Expr, args: &[Expr]) -> String {
        let fname = self.emit_expr(func);
        let mut arg_strs = Vec::new();
        for a in args {
            arg_strs.push(self.emit_expr(a));
        }

        if fname == "cp_async" {
            return format!(
                "std::ptr::copy_nonoverlapping({}, {}, 32)",
                arg_strs[0], arg_strs[1]
            );
        } else if fname == "ldmatrix" {
            return format!("Y256f32::load_aligned_ptr({} as *const f32)", arg_strs[0]);
        } else if fname == "mma_sync" {
            return format!("{}.fmadd({}, {})", arg_strs[0], arg_strs[1], arg_strs[2]);
        } else if fname == "store" {
            return format!(
                "{}.store_aligned_ptr({} as *mut f32)",
                arg_strs[0], arg_strs[1]
            );
        }

        format!("{}({})", fname, arg_strs.join(", "))
    }

    fn emit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => name.clone(),
            Expr::IntLit(val, _) => val.to_string(),
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
                namespace, member, ..
            } => {
                if namespace == "Fragment" && member == "zero" {
                    return "Y256f32::zero".into();
                }
                if namespace == "SharedMemory" && member == "alloc" {
                    return "y_shared_alloc_f32".into();
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
                "".into()
            }
            Expr::MemberAccess { member, .. } => {
                if member == "wait" {
                    return "y_pipe_wait".into();
                }
                "".into()
            }
            _ => "0 // Fallback".into(),
        }
    }
}

