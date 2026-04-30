// ============================================================
//  Y-Lang — LLVM IR Backend Emitter
//  llvm_emitter.rs
//
//  Translates Y-Lang AST into LLVM IR textual representation.
//  The generated .ll file can be compiled by llc/clang to
//  produce native code for any LLVM-supported target.
//
//  Type mapping:
//    Y-Lang         LLVM IR
//    ------         -------
//    I32            i32
//    I64            i64
//    F32            float
//    F64            double
//    bool           i1
//    char           i8
//    usize          i64
//    String         %YStr (opaque ptr)
//    Vec<T>         %YVec (opaque ptr)
//    &T             ptr
//    &mut T         ptr
// ============================================================

#![allow(dead_code)]

use std::fmt::Write;
use std::collections::HashMap;
use crate::ast::*;

pub struct LlvmEmitter {
    pub output: String,
    /// String constants collected during emission, emitted at module scope
    string_constants: Vec<String>,
    string_counter: usize,
    tmp_counter: usize,
    label_counter: usize,
    current_impl_target: Option<String>,
    /// Track local variables and their LLVM IR types
    locals: HashMap<String, String>,
    /// Track function return types
    functions: HashMap<String, String>,
    /// Track struct fields: StructName -> Vec<(FieldName, IRType)>
    structs: HashMap<String, Vec<(String, String)>>,
    /// Track enums: EnumName -> has_data (true = tagged union, false = simple i32 tag)
    enums: HashMap<String, bool>,
    /// Track whether the current block already has a terminator
    block_terminated: bool,
    /// Store current cache policy during let bindings
    current_cache_policy: Option<String>,
    /// Hint for the load() intrinsic: the declared LHS type of the current let
    current_load_hint: Option<String>,
    /// Track all function names called during emission
    called_functions: Vec<String>,
    /// Track all function names defined in this module
    defined_functions: Vec<String>,
}

impl LlvmEmitter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            string_constants: Vec::new(),
            string_counter: 0,
            tmp_counter: 0,
            label_counter: 0,
            current_impl_target: None,
            locals: HashMap::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            block_terminated: false,
            current_cache_policy: None,
            current_load_hint: None,
            called_functions: Vec::new(),
            defined_functions: Vec::new(),
        }
    }

    fn fresh_tmp(&mut self) -> String {
        self.tmp_counter += 1;
        format!("%t{}", self.tmp_counter)
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        format!("{}.{}", prefix, self.label_counter)
    }

    fn emit_load(&mut self, ptr: &str, ty: &str) -> String {
        let tmp = self.fresh_tmp();
        writeln!(&mut self.output, "  {} = load {}, ptr {}", tmp, ty, ptr).unwrap();
        tmp
    }

    fn emit_store(&mut self, val: &str, ptr: &str, ty: &str) {
        writeln!(&mut self.output, "  store {} {}, ptr {}", ty, val, ptr).unwrap();
    }

    /// Insert an LLVM conversion instruction when src_ty != dst_ty.
    /// Returns the new SSA name holding the converted value, or the
    /// original `val` if no conversion is needed.
    fn emit_coerce(&mut self, val: &str, src_ty: &str, dst_ty: &str) -> String {
        if src_ty == dst_ty {
            return val.to_string();
        }

        // Named struct types (like %Token) cannot be converted via scalar instructions.
        // If either side is a named type, we pass through without conversion.
        let src_is_struct = src_ty.starts_with('%');
        let dst_is_struct = dst_ty.starts_with('%');
        if src_is_struct || dst_is_struct {
            // If both are structs but different, warn; otherwise just pass through
            writeln!(&mut self.output, "  ; NOTE: struct type coerce pass-through {} -> {}", src_ty, dst_ty).unwrap();
            return val.to_string();
        }

        let tmp = self.fresh_tmp();
        let src_float = src_ty == "float" || src_ty == "double" || src_ty == "half";
        let dst_float = dst_ty == "float" || dst_ty == "double" || dst_ty == "half";
        let src_ptr   = src_ty == "ptr";
        let dst_ptr   = dst_ty == "ptr";
        let src_int   = !src_float && !src_ptr;
        let dst_int   = !dst_float && !dst_ptr;

        if src_ptr && dst_int {
            // ptr -> integer
            writeln!(&mut self.output, "  {} = ptrtoint ptr {} to {}", tmp, val, dst_ty).unwrap();
        } else if src_int && dst_ptr {
            // integer -> ptr
            writeln!(&mut self.output, "  {} = inttoptr {} {} to ptr", tmp, src_ty, val).unwrap();
        } else if src_float && dst_int {
            // float -> integer (signed)
            writeln!(&mut self.output, "  {} = fptosi {} {} to {}", tmp, src_ty, val, dst_ty).unwrap();
        } else if src_int && dst_float {
            // integer -> float (signed)
            writeln!(&mut self.output, "  {} = sitofp {} {} to {}", tmp, src_ty, val, dst_ty).unwrap();
        } else if src_float && dst_float {
            // float <-> float (truncate or extend)
            let src_bits: u32 = if src_ty == "double" { 64 } else if src_ty == "float" { 32 } else { 16 };
            let dst_bits: u32 = if dst_ty == "double" { 64 } else if dst_ty == "float" { 32 } else { 16 };
            if src_bits > dst_bits {
                writeln!(&mut self.output, "  {} = fptrunc {} {} to {}", tmp, src_ty, val, dst_ty).unwrap();
            } else {
                writeln!(&mut self.output, "  {} = fpext {} {} to {}", tmp, src_ty, val, dst_ty).unwrap();
            }
        } else if src_int && dst_int {
            // integer <-> integer (different widths)
            let src_bits = Self::int_bits(src_ty);
            let dst_bits = Self::int_bits(dst_ty);
            if src_bits > dst_bits {
                writeln!(&mut self.output, "  {} = trunc {} {} to {}", tmp, src_ty, val, dst_ty).unwrap();
            } else {
                writeln!(&mut self.output, "  {} = sext {} {} to {}", tmp, src_ty, val, dst_ty).unwrap();
            }
        } else if src_ptr && dst_ptr {
            return val.to_string(); // ptr -> ptr, no conversion needed in opaque-ptr mode
        } else if src_float && dst_ptr {
            // float -> ptr via intermediate int
            let int_tmp = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = fptosi {} {} to i64", int_tmp, src_ty, val).unwrap();
            writeln!(&mut self.output, "  {} = inttoptr i64 {} to ptr", tmp, int_tmp).unwrap();
        } else if src_ptr && dst_float {
            // ptr -> float via intermediate int (PRESERVING BITS using bitcast)
            let int_tmp = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = ptrtoint ptr {} to i64", int_tmp, val).unwrap();
            
            if dst_ty == "double" {
                // 64-bit pointer fits perfectly into 64-bit double
                writeln!(&mut self.output, "  {} = bitcast i64 {} to double", tmp, int_tmp).unwrap();
            } else {
                // For 32-bit float, we must truncate the 64-bit pointer first
                let trunc_tmp = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = trunc i64 {} to i32", trunc_tmp, int_tmp).unwrap();
                writeln!(&mut self.output, "  {} = bitcast i32 {} to float", tmp, trunc_tmp).unwrap();
            }
        } else {
            // Unknown conversion — pass through without conversion
            writeln!(&mut self.output, "  ; WARN: unhandled coerce {} -> {}", src_ty, dst_ty).unwrap();
            return val.to_string();
        }
        tmp
    }

    /// Return bit width for an LLVM integer type string.
    fn int_bits(ty: &str) -> u32 {
        match ty {
            "i1" => 1,
            "i8" => 8,
            "i16" => 16,
            "i32" => 32,
            "i64" => 64,
            _ => 64, // conservative fallback
        }
    }

    /// Register a string constant and return its global name
    fn register_string(&mut self, s: &str) -> String {
        let id = self.string_counter;
        self.string_counter += 1;
        let escaped = s.replace('\\', "\\5C").replace('\n', "\\0A").replace('"', "\\22");
        let len = s.len() + 1; // +1 for null terminator
        let decl = format!("@.str.{} = private unnamed_addr constant [{} x i8] c\"{}\\00\"", id, len, escaped);
        self.string_constants.push(decl);
        format!("@.str.{}", id)
    }

    fn w(&mut self, s: &str) {
        write!(&mut self.output, "{}", s).unwrap();
    }

    fn wln(&mut self, s: &str) {
        writeln!(&mut self.output, "{}", s).unwrap();
    }

    // ── Type Mapping ────────────────────────────────────────

    fn emit_type(&self, ty: &Type) -> String {
        match ty {
            Type::Primitive(name, _) => match name.as_str() {
                "I32" | "u32" | "i32" => "i32".into(),
                "I64" | "usize" | "i64" => "i64".into(),
                "F16" | "f16" => "half".into(),
                "F32" | "f32" => "float".into(),
                "F64" | "f64" => "double".into(),
                "bool" => "i1".into(),
                "char" | "i8" | "u8" => "i8".into(),
                "I16" | "u16" | "i16" => "i16".into(),
                "String" => "ptr".into(),
                _ => "i32".into(),
            },
            Type::Ident(name, _) => match name.as_str() {
                "I32" | "u32" | "i32" => "i32".into(),
                "I64" | "usize" | "i64" => "i64".into(),
                "F32" | "f32" => "float".into(),
                "F64" | "f64" => "double".into(),
                "bool" => "i1".into(),
                "char" | "i8" | "u8" => "i8".into(),
                "I16" | "u16" | "i16" => "i16".into(),
                "String" => "ptr".into(),
                "Vec" => "ptr".into(),
                other => {
                    // Check if this is a known simple enum (maps to i32)
                    if let Some(has_data) = self.enums.get(other) {
                        if *has_data {
                            format!("%{}", other) // tagged union struct type
                        } else {
                            "i32".into() // simple enum = integer tag
                        }
                    } else {
                        format!("%{}", other) // Custom struct types passed by value
                    }
                }
            },
            Type::Reference { .. } => "ptr".into(),
            Type::Generic { base, .. } => match base.as_str() {
                "Vec" | "Option" | "Box" | "GlobalMemory" | "SharedMemory" => "ptr".into(),
                _ => "ptr".into(),
            },
            Type::Array { .. } => "ptr".into(),
        }
    }

    // ── Entry Point ─────────────────────────────────────────

    pub fn emit_program(&mut self, prog: &Program, profile: &crate::sentinel::HardwareProfile) -> String {
        // Phase 0: Collect struct layouts and function signatures
        for item in &prog.items {
            match item {
                Item::Struct(s) => {
                    let mut fields = Vec::new();
                    for f in &s.fields {
                        fields.push((f.name.clone(), self.emit_type(&f.ty)));
                    }
                    self.structs.insert(s.name.clone(), fields);
                }
                Item::Func(f) => {
                    let ret_ty = f.ret_ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "void".into());
                    self.functions.insert(f.name.clone(), ret_ty);
                }
                Item::Impl(imp) => {
                    for m in &imp.methods {
                        let ret_ty = m.ret_ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "void".into());
                        self.functions.insert(format!("{}_{}", imp.target_type, m.name), ret_ty);
                    }
                }
                Item::Kernel(k) => {
                    self.functions.insert(k.name.clone(), "void".into());
                }
                Item::Enum(e) => {
                    let has_data = e.variants.iter().any(|v| v.fields.is_some());
                    self.enums.insert(e.name.clone(), has_data);
                }
                _ => {}
            }
        }

        // Phase 1: emit all function bodies into a temporary buffer,
        // collecting string constants along the way
        let mut func_output = String::new();
        std::mem::swap(&mut self.output, &mut func_output);

        for item in &prog.items {
            match item {
                Item::Func(f) => self.emit_func(f),
                Item::Impl(imp) => self.emit_impl(imp),
                Item::Kernel(k) => self.emit_kernel(k),
                _ => {}
            }
        }

        std::mem::swap(&mut self.output, &mut func_output);

        // Phase 2: assemble final output with constants at module scope
        self.emit_prelude(profile);

        // Emit struct definitions
        self.wln("; --- Struct Definitions ---");
        for item in &prog.items {
            if let Item::Struct(s) = item {
                let mut field_tys = Vec::new();
                for f in &s.fields {
                    field_tys.push(self.emit_type(&f.ty));
                }
                self.wln(&format!("%{} = type {{ {} }}", s.name, field_tys.join(", ")));
            }
        }
        self.wln("");

        // Emit Enum definitions (tagged union layout)
        self.wln("; --- Enum Definitions ---");
        for item in &prog.items {
            if let Item::Enum(e) = item {
                let has_data = e.variants.iter().any(|v| v.fields.is_some());
                if has_data {
                    // LLVM represents tagged unions as { i32, [4 x i64] } (approximate placeholder size)
                    self.wln(&format!("%{} = type {{ i32, [4 x i64] }}", e.name));
                }
            }
        }
        self.wln("");

        self.wln("; --- External Runtime Declarations ---");
        self.wln("declare ptr @ystr_new(ptr)");
        self.wln("declare void @ystr_push(ptr, i8)");
        self.wln("declare void @ystr_push_str(ptr, ptr)");
        self.wln("declare i1 @ystr_eq_cstr(ptr, ptr)");
        self.wln("declare i64 @ystr_len(ptr)");
        self.wln("declare i8 @ystr_char_at(ptr, i64)");
        self.wln("declare ptr @ystr_clone(ptr)");
        self.wln("declare ptr @yvec_new(i64)");
        self.wln("declare void @yvec_push(ptr, ptr)");
        self.wln("declare ptr @yvec_get(ptr, i64)");
        self.wln("declare i64 @yvec_len(ptr)");
        self.wln("declare ptr @yfile_read_to_string(ptr)");
        self.wln("declare void @yfile_write(ptr, ptr)");
        self.wln("declare i32 @printf(ptr, ...)");
        self.wln("declare ptr @malloc(i64)");
        self.wln("declare void @free(ptr)");
        self.wln("declare void @exit(i32) noreturn");
        self.wln("declare void @println(ptr)");
        self.wln("declare void @print_int(i32)");
        self.wln("declare void @llvm.prefetch.p0(ptr nocapture readonly, i32, i32, i32)");
        self.wln("");

        // Emit all collected string constants at module scope
        if !self.string_constants.is_empty() {
            self.wln("; --- String Constants ---");
            for sc in &self.string_constants.clone() {
                self.wln(sc);
            }
            self.wln("");
        }

        // Emit format strings for printf
        self.wln("@.fmt.sn = private unnamed_addr constant [4 x i8] c\"%s\\0A\\00\"");
        self.wln("@.fmt.d = private unnamed_addr constant [4 x i8] c\"%ld\\00\"");
        self.wln("");

        // Append function bodies
        self.output.push_str(&func_output);

        // Auto-declare any called functions that are not defined or already declared
        let runtime_set: std::collections::HashSet<&str> = [
            "ystr_new", "ystr_push", "ystr_push_str", "ystr_eq_cstr", "ystr_len",
            "ystr_char_at", "ystr_clone", "yvec_new", "yvec_push", "yvec_get",
            "yvec_len", "yfile_read_to_string", "yfile_write", "printf", "malloc",
            "free", "exit", "println", "print_int", "llvm.prefetch.p0", "load",
        ].iter().cloned().collect();

        let defined_set: std::collections::HashSet<String> = self.defined_functions.iter().cloned().collect();
        let mut auto_declared: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut extern_decls = String::new();

        for fname in &self.called_functions {
            if !runtime_set.contains(fname.as_str())
                && !defined_set.contains(fname)
                && !auto_declared.contains(fname)
            {
                // Look up the return type from the functions table, default to i32
                let ret_ty = self.functions.get(fname).cloned().unwrap_or_else(|| "i32".into());
                writeln!(&mut extern_decls, "declare {} @{}(...)", ret_ty, fname).unwrap();
                auto_declared.insert(fname.clone());
            }
        }

        if !auto_declared.is_empty() {
            let marker = "; --- External Runtime Declarations ---\n";
            if let Some(pos) = self.output.find(marker) {
                let insert_at = pos + marker.len();
                self.output.insert_str(insert_at, &extern_decls);
            }
        }

        // Nontemporal metadata definition
        self.wln("!0 = !{i32 1}");

        self.output.clone()
    }

    fn emit_prelude(&mut self, profile: &crate::sentinel::HardwareProfile) {
        self.wln("; ================================================");
        self.wln(";  Generated by Y-Lang Compiler — LLVM IR Backend");
        self.wln(&format!(";  Hardware Profile: AVX={}, AVX512={}, L2 Line={}B", profile.has_avx, profile.has_avx512, profile.l2_line_size));
        self.wln("; ================================================");
        self.wln("");
        self.wln("target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"");
        self.wln("target triple = \"x86_64-pc-windows-msvc\"");
        self.wln("");
        
        // Dynamically inject LLVM function attributes based on Sentinel Probe
        if profile.has_avx512 {
            self.wln("attributes #0 = { \"target-cpu\"=\"skylake-avx512\" \"target-features\"=\"+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl\" }");
        } else if profile.has_avx {
            self.wln("attributes #0 = { \"target-cpu\"=\"haswell\" \"target-features\"=\"+avx2,+avx\" }");
        } else {
            self.wln("attributes #0 = { \"target-cpu\"=\"x86-64\" }");
        }
        self.wln("");
    }

    // ── Functions ───────────────────────────────────────────

    fn emit_func(&mut self, f: &FuncDecl) {
        self.tmp_counter = 0;
        self.locals.clear();
        self.block_terminated = false;

        let ret_type = match &f.ret_ty {
            Some(ty) => self.emit_type(ty),
            None => "void".into(),
        };

        let func_name = if let Some(ref target) = self.current_impl_target {
            format!("{}_{}", target, f.name)
        } else {
            f.name.clone()
        };
        self.defined_functions.push(func_name.clone());

        let params: Vec<String> = f.params.iter().map(|p| {
            let ty = self.emit_type(&p.ty);
            format!("{} %{}.arg", ty, p.name)
        }).collect();
        let params_str = params.join(", ");

        writeln!(&mut self.output, "define {} @{}({}) #0 {{", ret_type, func_name, params_str).unwrap();
        self.wln("entry:");

        // Alloca for all params so we can store/load them by name
        for p in &f.params {
            let ty = self.emit_type(&p.ty);
            self.locals.insert(p.name.clone(), ty.clone());
            writeln!(&mut self.output, "  %{} = alloca {}", p.name, ty).unwrap();
            self.emit_store(&format!("%{}.arg", p.name), &format!("%{}", p.name), &ty);
        }

        // Forward declare all lets in entry block to avoid loop stack growth
        self.emit_alloca_for_block(&f.body);

        self.emit_block_body(&f.body, &ret_type);

        // Add default return if the block didn't terminate
        if !self.block_terminated {
            if ret_type == "void" {
                self.wln("  ret void");
            } else if ret_type == "i32" {
                self.wln("  ret i32 0");
            } else {
                self.wln("  ret void");
            }
        }

        self.wln("}");
        self.wln("");
    }

    fn emit_alloca_for_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { name, ty, init, .. } => {
                    let ir_ty = match ty {
                        Some(t) => self.emit_type(t),
                        None => {
                            if let Some(init_expr) = init {
                                self.infer_type(init_expr)
                            } else {
                                "i32".into()
                            }
                        }
                    };
                    self.locals.insert(name.clone(), ir_ty.clone());
                    writeln!(&mut self.output, "  %{} = alloca {}", name, ir_ty).unwrap();
                }
                Stmt::For { loop_var, body, .. } => {
                    self.locals.insert(loop_var.clone(), "i32".into());
                    writeln!(&mut self.output, "  %{} = alloca i32", loop_var).unwrap();
                    self.emit_alloca_for_block(body);
                }
                Stmt::If { then_block, else_block, .. } => {
                    self.emit_alloca_for_block(then_block);
                    if let Some(eb) = else_block {
                        self.emit_alloca_for_block(eb);
                    }
                }
                Stmt::While { body, .. } => {
                    self.emit_alloca_for_block(body);
                }
                Stmt::Chisel(b, _) => {
                    self.emit_alloca_for_block(b);
                }
                _ => {}
            }
        }
    }

    fn emit_kernel(&mut self, k: &KernelDecl) {
        self.tmp_counter = 0;
        self.locals.clear();
        self.block_terminated = false;

        writeln!(&mut self.output, "; @kernel").unwrap();

        let params: Vec<String> = k.params.iter().map(|p| {
            let ty = self.emit_type(&p.ty);
            format!("{} %{}.arg", ty, p.name)
        }).collect();

        writeln!(&mut self.output, "define void @{}({}) #0 {{", k.name, params.join(", ")).unwrap();
        self.wln("entry:");
        self.defined_functions.push(k.name.clone());
        
        for p in &k.params {
            let ty = self.emit_type(&p.ty);
            self.locals.insert(p.name.clone(), ty.clone());
            writeln!(&mut self.output, "  %{} = alloca {}", p.name, ty).unwrap();
            self.emit_store(&format!("%{}.arg", p.name), &format!("%{}", p.name), &ty);
        }

        self.emit_alloca_for_block(&k.body);

        self.emit_block_body(&k.body, "void");
        if !self.block_terminated {
            self.wln("  ret void");
        }
        self.wln("}");
        self.wln("");
    }

    fn emit_impl(&mut self, imp: &ImplBlock) {
        writeln!(&mut self.output, "; impl {}", imp.target_type).unwrap();
        self.current_impl_target = Some(imp.target_type.clone());
        for method in &imp.methods {
            self.emit_func(method);
        }
        self.current_impl_target = None;
    }

    // ── Block / Statement Emission ──────────────────────────

    fn emit_block_body(&mut self, block: &Block, ret_type: &str) {
        for stmt in &block.stmts {
            if self.block_terminated {
                break; // Don't emit unreachable code after a terminator
            }
            self.emit_stmt(stmt, ret_type);
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt, ret_type: &str) {
        match stmt {
            Stmt::Let { name, init, cache_policy, .. } => {
                if let Some(cp) = cache_policy {
                    self.current_cache_policy = Some(cp.policy.clone());
                }

                // alloca is already done in entry
                if let Some(init_expr) = init {
                    // Set load hint so `load()` intrinsic uses the LHS type
                    let dst_ty = self.locals.get(name).cloned().unwrap_or_else(|| "i32".into());
                    self.current_load_hint = Some(dst_ty.clone());
                    let val = self.emit_expr(init_expr);
                    let val_ty = self.infer_type(init_expr);
                    self.current_load_hint = None;
                    let coerced = self.emit_coerce(&val, &val_ty, &dst_ty);
                    self.emit_store(&coerced, &format!("%{}", name), &dst_ty);
                }

                self.current_cache_policy = None;
            }
            Stmt::Assign { target, value, .. } => {
                let val = self.emit_expr(value);
                let val_ty = self.infer_type(value);
                let dst_ty = self.infer_type(target);
                // Get the address of the target (don't load it)
                let target_addr = self.emit_lvalue(target);
                let coerced = self.emit_coerce(&val, &val_ty, &dst_ty);
                self.emit_store(&coerced, &target_addr, &dst_ty);
            }
            Stmt::Return(expr, _) => {
                if let Some(e) = expr {
                    let val = self.emit_expr(e);
                    let val_ty = self.infer_type(e);
                    let coerced = self.emit_coerce(&val, &val_ty, ret_type);
                    writeln!(&mut self.output, "  ret {} {}", ret_type, coerced).unwrap();
                } else {
                    self.wln("  ret void");
                }
                self.block_terminated = true;
            }
            Stmt::Expr(e) => {
                self.emit_expr(e);
            }
            Stmt::If { condition, then_block, else_block, .. } => {
                let cond = self.emit_expr(condition);
                let then_lbl = self.fresh_label("then");
                let else_lbl = self.fresh_label("else");
                let merge_lbl = self.fresh_label("merge");

                writeln!(&mut self.output, "  br i1 {}, label %{}, label %{}",
                    cond, then_lbl, if else_block.is_some() { &else_lbl } else { &merge_lbl }).unwrap();

                // Then block
                writeln!(&mut self.output, "{}:", then_lbl).unwrap();
                self.block_terminated = false;
                self.emit_block_body(then_block, ret_type);
                let then_terminated = self.block_terminated;
                if !then_terminated {
                    writeln!(&mut self.output, "  br label %{}", merge_lbl).unwrap();
                }

                // Else block
                if let Some(eb) = else_block {
                    writeln!(&mut self.output, "{}:", else_lbl).unwrap();
                    self.block_terminated = false;
                    self.emit_block_body(eb, ret_type);
                    let else_terminated = self.block_terminated;
                    if !else_terminated {
                        writeln!(&mut self.output, "  br label %{}", merge_lbl).unwrap();
                    }
                }

                writeln!(&mut self.output, "{}:", merge_lbl).unwrap();
                self.block_terminated = false;
            }
            Stmt::While { condition, body, .. } => {
                let cond_lbl = self.fresh_label("while.cond");
                let body_lbl = self.fresh_label("while.body");
                let end_lbl = self.fresh_label("while.end");

                writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();
                writeln!(&mut self.output, "{}:", cond_lbl).unwrap();
                let cond = self.emit_expr(condition);
                writeln!(&mut self.output, "  br i1 {}, label %{}, label %{}", cond, body_lbl, end_lbl).unwrap();

                writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                self.block_terminated = false;
                self.emit_block_body(body, ret_type);
                if !self.block_terminated {
                    writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();
                }

                writeln!(&mut self.output, "{}:", end_lbl).unwrap();
                self.block_terminated = false;
            }
            Stmt::For { loop_var, start, end, step, body, .. } => {
                let s = self.emit_expr(start);
                let e = self.emit_expr(end);
                let cond_lbl = self.fresh_label("for.cond");
                let body_lbl = self.fresh_label("for.body");
                let end_lbl = self.fresh_label("for.end");

                // alloca is in entry
                self.emit_store(&s, &format!("%{}", loop_var), "i32");
                writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();

                writeln!(&mut self.output, "{}:", cond_lbl).unwrap();
                let cur = self.emit_load(&format!("%{}", loop_var), "i32");
                let cmp = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = icmp slt i32 {}, {}", cmp, cur, e).unwrap();
                writeln!(&mut self.output, "  br i1 {}, label %{}, label %{}", cmp, body_lbl, end_lbl).unwrap();

                writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                self.block_terminated = false;
                self.emit_block_body(body, ret_type);

                // Increment
                let step_val = if let Some(st) = step { self.emit_expr(st) } else { "1".into() };
                let loaded = self.emit_load(&format!("%{}", loop_var), "i32");
                let incremented = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = add i32 {}, {}", incremented, loaded, step_val).unwrap();
                self.emit_store(&incremented, &format!("%{}", loop_var), "i32");
                writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();

                writeln!(&mut self.output, "{}:", end_lbl).unwrap();
                self.block_terminated = false;
            }
            Stmt::CompoundAssign { target, op, value, .. } => {
                let addr = self.emit_lvalue(target);
                let rhs = self.emit_expr(value);
                let ty = self.infer_type(target);
                let loaded = self.emit_load(&addr, &ty);
                let result = self.fresh_tmp();
                let op_str = self.binop_to_llvm(op, &ty);
                writeln!(&mut self.output, "  {} = {} {} {}, {}", result, op_str, ty, loaded, rhs).unwrap();
                self.emit_store(&result, &addr, &ty);
            }
            Stmt::Chisel(block, _) => {
                self.wln("  ; --- CHISEL INLINE ASM ---");
                for stmt in &block.stmts {
                    if let Stmt::Expr(Expr::StringLit(s, _)) = stmt {
                        self.wln(&format!("  call void asm sideeffect \"{}\", \"~{{memory}},~{{dirflag}},~{{fpsr}},~{{flags}}\"()", s));
                    } else {
                        self.emit_stmt(stmt, ret_type);
                    }
                }
            }
            Stmt::Match { scrutinee, arms, .. } => {
                let scrut_val = self.emit_expr(scrutinee);
                let scrut_ty = self.infer_type(scrutinee);
                let merge_lbl = self.fresh_label("match.end");

                // Emit as cascading if-else (LLVM has switch but only for integer constants)
                let mut arm_labels: Vec<(String, String)> = Vec::new(); // (test_lbl, body_lbl)
                for _ in arms {
                    let test_lbl = self.fresh_label("match.test");
                    let body_lbl = self.fresh_label("match.arm");
                    arm_labels.push((test_lbl, body_lbl));
                }

                if !arms.is_empty() {
                    writeln!(&mut self.output, "  br label %{}", arm_labels[0].0).unwrap();
                }

                for (i, arm) in arms.iter().enumerate() {
                    let (test_lbl, body_lbl) = &arm_labels[i];
                    let next_test = if i + 1 < arms.len() {
                        arm_labels[i + 1].0.clone()
                    } else {
                        merge_lbl.clone()
                    };

                    writeln!(&mut self.output, "{}:", test_lbl).unwrap();
                    match &arm.pattern {
                        MatchPattern::Wildcard(_) => {
                            writeln!(&mut self.output, "  br label %{}", body_lbl).unwrap();
                        }
                        MatchPattern::Literal(lit) => {
                            let lit_val = self.emit_expr(lit);
                            let cmp = self.fresh_tmp();
                            let cmp_instr = if scrut_ty == "float" || scrut_ty == "double" { "fcmp oeq" } else { "icmp eq" };
                            writeln!(&mut self.output, "  {} = {} {} {}, {}", cmp, cmp_instr, scrut_ty, scrut_val, lit_val).unwrap();
                            writeln!(&mut self.output, "  br i1 {}, label %{}, label %{}", cmp, body_lbl, next_test).unwrap();
                        }
                        MatchPattern::Ident(name, _) => {
                            // Bind variable then always match
                            let cmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = icmp eq {} {}, {}", cmp, scrut_ty, scrut_val, name).unwrap();
                            writeln!(&mut self.output, "  br i1 {}, label %{}, label %{}", cmp, body_lbl, next_test).unwrap();
                        }
                        MatchPattern::EnumVariant { path, variant, .. } => {
                            // Compare tag value (simple enum = i32)
                            // Lookup variant index
                            let tag_name = if path.is_empty() {
                                variant.clone()
                            } else {
                                format!("{}_{}", path, variant)
                            };
                            let cmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = icmp eq {} {}, {} ; enum {}", cmp, scrut_ty, scrut_val, tag_name, variant).unwrap();
                            writeln!(&mut self.output, "  br i1 {}, label %{}, label %{}", cmp, body_lbl, next_test).unwrap();
                        }
                    }

                    writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                    self.block_terminated = false;
                    self.emit_expr(&arm.body);
                    if !self.block_terminated {
                        writeln!(&mut self.output, "  br label %{}", merge_lbl).unwrap();
                    }
                }

                writeln!(&mut self.output, "{}:", merge_lbl).unwrap();
                self.block_terminated = false;
            }
            Stmt::TypeAlias { .. } => {
                // Type aliases are resolved at compile time — no IR emission needed
            }
        }
    }

    // ── Expression Emission ─────────────────────────────────

    /// Emit an lvalue (address) for assignment targets — returns ptr
    fn emit_lvalue(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => format!("%{}", name),
            Expr::MemberAccess { base, member, .. } => {
                let base_val = self.emit_lvalue(base);
                let base_ty = self.infer_type(base);
                let tmp = self.fresh_tmp();
                
                let mut field_index = 0;
                if let Some(fields) = self.structs.get(base_ty.trim_start_matches('%')) {
                    for (i, (fname, _)) in fields.iter().enumerate() {
                        if fname == member {
                            field_index = i;
                            break;
                        }
                    }
                }
                
                writeln!(&mut self.output, "  ; lvalue .{}", member).unwrap();
                writeln!(&mut self.output, "  {} = getelementptr {}, ptr {}, i32 0, i32 {}", tmp, base_ty, base_val, field_index).unwrap();
                tmp
            }
            Expr::Index { base, index, .. } => {
                let base_val = self.emit_expr(base);
                let idx_val = self.emit_expr(index);
                let elem_ty = "i32"; // Fallback
                let tmp = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = getelementptr {}, ptr {}, i64 {}", tmp, elem_ty, base_val, idx_val).unwrap();
                tmp
            }
            Expr::UnaryOp { op: UnaryOp::Deref, operand, .. } => {
                self.emit_expr(operand)
            }
            _ => {
                self.emit_expr(expr)
            }
        }
    }

    fn emit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::IntLit(val, _) => format!("{}", val),
            Expr::FloatLit(val, _) => format!("{:.6e}", val),
            Expr::BoolLit(b, _) => if *b { "1".into() } else { "0".into() },
            Expr::CharLit(c, _) => format!("{}", *c as u32),
            Expr::Ident(name, _) => {
                let ty = self.locals.get(name).cloned().unwrap_or_else(|| "i32".into());
                self.emit_load(&format!("%{}", name), &ty)
            }
            Expr::StringLit(s, _) => {
                let global_name = self.register_string(s);
                let tmp = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = call ptr @ystr_new(ptr {})", tmp, global_name).unwrap();
                tmp
            }
            Expr::BinaryOp { left, op, right, .. } => {
                let l = self.emit_expr(left);
                let r = self.emit_expr(right);
                let ty = self.infer_type(left);
                let tmp = self.fresh_tmp();
                let instr = self.binop_to_llvm(op, &ty);
                writeln!(&mut self.output, "  {} = {} {} {}, {}", tmp, instr, ty, l, r).unwrap();
                tmp
            }
            Expr::UnaryOp { op, operand, .. } => {
                let val = self.emit_expr(operand);
                let tmp = self.fresh_tmp();
                let ty = self.infer_type(operand);
                match op {
                    UnaryOp::Neg => {
                        if ty == "float" || ty == "double" {
                            writeln!(&mut self.output, "  {} = fneg {} {}", tmp, ty, val).unwrap();
                        } else {
                            writeln!(&mut self.output, "  {} = sub {} 0, {}", tmp, ty, val).unwrap();
                        }
                    }
                    UnaryOp::Not => {
                        writeln!(&mut self.output, "  {} = xor {} {}, 1", tmp, ty, val).unwrap();
                    }
                    UnaryOp::Ref => {
                        return self.emit_lvalue(operand);
                    }
                    UnaryOp::Deref => {
                        writeln!(&mut self.output, "  {} = load i32, ptr {}", tmp, val).unwrap();
                    }
                }
                tmp
            }
            Expr::Call { func, args, .. } => {
                let func_name = self.emit_call_target(func);
                self.called_functions.push(func_name.clone());
                let mut arg_strs = Vec::new();
                for a in args {
                    let v = self.emit_expr(a);
                    let ty = self.infer_type(a);
                    arg_strs.push(format!("{} {}", ty, v));
                }

                match func_name.as_str() {
                    "load" => {
                        let ptr_val = self.emit_expr(&args[0]);
                        let tmp = self.fresh_tmp();
                        let mut metadata = String::new();
                        
                        if let Some(policy) = &self.current_cache_policy.clone() {
                            if policy == "L2_EVICT_FIRST" {
                                metadata = ", !nontemporal !0".to_string();
                            } else if policy == "L2_PERSIST" {
                                // 0 = Read, 3 = High temporal locality, 1 = Data cache
                                writeln!(&mut self.output, "  call void @llvm.prefetch.p0(ptr {}, i32 0, i32 3, i32 1)", ptr_val).unwrap();
                            }
                        }
                        
                        // Infer load type from the LHS variable's alloca type.
                        // The caller (emit_stmt for Let) will coerce if needed.
                        // We use the type annotation from `self.current_let_type` if
                        // available, otherwise fall back to the pointer element type.
                        let load_ty = self.current_load_hint.clone().unwrap_or_else(|| {
                            // Infer from args: if loading from a typed pointer, use that type
                            let arg_ty = self.infer_type(&args[0]);
                            if arg_ty == "ptr" { "double".into() } else { arg_ty }
                        });
                        writeln!(&mut self.output, "  {} = load {}, ptr {}{}", tmp, load_ty, ptr_val, metadata).unwrap();
                        return tmp;
                    }
                    "println" => {
                        if !args.is_empty() {
                            let tmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = call i32 (ptr, ...) @printf(ptr @.fmt.sn, {})", tmp, arg_strs[0]).unwrap();
                            return tmp;
                        }
                        let tmp = self.fresh_tmp();
                        let nl = self.register_string("");
                        writeln!(&mut self.output, "  {} = call i32 (ptr, ...) @printf(ptr @.fmt.sn, ptr {})", tmp, nl).unwrap();
                        return tmp;
                    }
                    "print_int" => {
                        let tmp = self.fresh_tmp();
                        writeln!(&mut self.output, "  {} = call i32 (ptr, ...) @printf(ptr @.fmt.d, {})", tmp, arg_strs[0]).unwrap();
                        return tmp;
                    }
                    _ => {}
                }

                let ret_ty = self.functions.get(&func_name).cloned().unwrap_or_else(|| "i32".into());
                let tmp = self.fresh_tmp();
                if ret_ty == "void" {
                    writeln!(&mut self.output, "  call void @{}({})", func_name, arg_strs.join(", ")).unwrap();
                    tmp
                } else {
                    writeln!(&mut self.output, "  {} = call {} @{}({})", tmp, ret_ty, func_name, arg_strs.join(", ")).unwrap();
                    tmp
                }
            }
            Expr::Path { namespace, member, .. } => {
                format!("{}_{}", namespace, member)
            }
            Expr::MemberAccess { base, member, .. } => {
                let lval = self.emit_lvalue(expr);
                let base_ty = self.infer_type(base);
                let mut field_ty = "i32".to_string(); // fallback
                if let Some(fields) = self.structs.get(base_ty.trim_start_matches('%')) {
                    for (fname, fty) in fields {
                        if fname == member {
                            field_ty = fty.clone();
                            break;
                        }
                    }
                }
                self.emit_load(&lval, &field_ty)
            }
            Expr::Index { .. } => {
                let lval = self.emit_lvalue(expr);
                self.emit_load(&lval, "i32") // Fallback
            }
            Expr::SelfLit(_) => "%self".into(),
            Expr::StructLit { name, fields, .. } => {
                let ty = format!("%{}", name);
                let mut current_val = "undef".to_string();
                
                for (fname, fexpr) in fields {
                    let mut field_idx = 0;
                    let mut field_ty = "i32".to_string();
                    if let Some(struct_fields) = self.structs.get(name).cloned() {
                        for (i, (sfname, sty)) in struct_fields.iter().enumerate() {
                            if sfname == fname { 
                                field_idx = i; 
                                field_ty = sty.clone();
                                break; 
                            }
                        }
                    }
                    let val = self.emit_expr(fexpr);
                    let val_ty = self.infer_type(fexpr);
                    let coerced = self.emit_coerce(&val, &val_ty, &field_ty);
                    let new_val = self.fresh_tmp();
                    writeln!(&mut self.output, "  {} = insertvalue {} {}, {} {}, {}", new_val, ty, current_val, field_ty, coerced, field_idx).unwrap();
                    current_val = new_val;
                }
                current_val
            }
            Expr::GenericCall { func, args, .. } => {
                // Generics are erased at IR level — emit as a regular call
                let func_name = self.emit_call_target(func);
                self.called_functions.push(func_name.clone());
                let mut arg_strs = Vec::new();
                for a in args {
                    let v = self.emit_expr(a);
                    let ty = self.infer_type(a);
                    arg_strs.push(format!("{} {}", ty, v));
                }
                let ret_ty = self.functions.get(&func_name).cloned().unwrap_or_else(|| "i32".into());
                let tmp = self.fresh_tmp();
                if ret_ty == "void" {
                    writeln!(&mut self.output, "  call void @{}({})", func_name, arg_strs.join(", ")).unwrap();
                    tmp
                } else {
                    writeln!(&mut self.output, "  {} = call {} @{}({})", tmp, ret_ty, func_name, arg_strs.join(", ")).unwrap();
                    tmp
                }
            }
            _ => {
                let tmp = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = add i32 0, 0 ; unhandled expr", tmp).unwrap();
                tmp
            }
        }
    }

    fn emit_call_target(&self, func: &Expr) -> String {
        match func {
            Expr::Ident(name, _) => name.clone(),
            Expr::Path { namespace, member, .. } => format!("{}_{}", namespace, member),
            Expr::MemberAccess { base, member, .. } => {
                if let Expr::Ident(base_name, _) = &**base {
                    format!("{}_{}", base_name, member)
                } else {
                    member.clone()
                }
            }
            _ => "unknown_func".into(),
        }
    }

    // ── Helpers ─────────────────────────────────────────────

    fn binop_to_llvm(&self, op: &BinaryOp, ty: &str) -> &'static str {
        let is_float = ty == "float" || ty == "double";
        match op {
            BinaryOp::Add => if is_float { "fadd" } else { "add" },
            BinaryOp::Sub => if is_float { "fsub" } else { "sub" },
            BinaryOp::Mul => if is_float { "fmul" } else { "mul" },
            BinaryOp::Div => if is_float { "fdiv" } else { "sdiv" },
            BinaryOp::Mod => if is_float { "frem" } else { "srem" },
            BinaryOp::Eq => if is_float { "fcmp oeq" } else { "icmp eq" },
            BinaryOp::NotEq => if is_float { "fcmp one" } else { "icmp ne" },
            BinaryOp::Lt => if is_float { "fcmp olt" } else { "icmp slt" },
            BinaryOp::Gt => if is_float { "fcmp ogt" } else { "icmp sgt" },
            BinaryOp::Le => if is_float { "fcmp ole" } else { "icmp sle" },
            BinaryOp::Ge => if is_float { "fcmp oge" } else { "icmp sge" },
            BinaryOp::And | BinaryOp::BitAnd => "and",
            BinaryOp::Or | BinaryOp::BitOr => "or",
            BinaryOp::BitXor => "xor",
            BinaryOp::Shl => "shl",
            BinaryOp::Shr => "ashr",
        }
    }

    fn infer_type(&self, expr: &Expr) -> String {
        match expr {
            Expr::IntLit(_, _) => "i32".into(),
            Expr::FloatLit(_, _) => "double".into(),
            Expr::BoolLit(_, _) => "i1".into(),
            Expr::CharLit(_, _) => "i8".into(),
            Expr::StringLit(_, _) => "ptr".into(),
            Expr::Ident(name, _) => self.locals.get(name).cloned().unwrap_or_else(|| "i32".into()),
            Expr::Call { func, .. } => {
                let func_name = self.emit_call_target(func);
                match func_name.as_str() {
                    "load" => {
                        // The load() intrinsic uses current_load_hint or defaults to double
                        self.current_load_hint.clone().unwrap_or_else(|| "double".into())
                    }
                    "println" | "print_int" => "i32".into(),
                    _ => self.functions.get(&func_name).cloned().unwrap_or_else(|| "i32".into()),
                }
            }
            Expr::GenericCall { func, .. } => {
                let func_name = self.emit_call_target(func);
                self.functions.get(&func_name).cloned().unwrap_or_else(|| "i32".into())
            }
            Expr::BinaryOp { op, left, .. } => {
                match op {
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge => "i1".into(),
                    _ => self.infer_type(left),
                }
            }
            Expr::MemberAccess { base, member, .. } => {
                let base_ty = self.infer_type(base);
                if let Some(fields) = self.structs.get(base_ty.trim_start_matches('%')) {
                    for (fname, fty) in fields {
                        if fname == member {
                            return fty.clone();
                        }
                    }
                }
                "i32".into()
            }
            Expr::StructLit { name, .. } => format!("%{}", name),
            _ => "i32".into(),
        }
    }
}

