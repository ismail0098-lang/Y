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

use crate::ast::*;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

pub struct LlvmEmitter {
    pub output: String,
    /// String constants collected during emission, emitted at module scope
    string_constants: Vec<String>,
    string_counter: usize,
    tmp_counter: usize,
    label_counter: usize,
    current_impl_target: Option<String>,
    /// Track local variables and their LLVM IR types
    locals: BTreeMap<String, String>,
    /// Map local variables to their AST type
    locals_ast_type: BTreeMap<String, String>,
    /// Track what struct type a pointer local variable points to
    pointee_types: BTreeMap<String, String>,
    /// Map function names to their LLVM parameter types and return type
    functions: BTreeMap<String, (Vec<String>, String)>,
    /// Track struct fields: StructName -> Vec<(FieldName, IRType)>
    structs: BTreeMap<String, Vec<(String, String)>>,
    /// Track struct fields AST Types: StructName -> Vec<(FieldName, ASTType)>
    ast_structs: HashMap<String, Vec<(String, String)>>,
    /// Track enums: EnumName -> has_data (true = tagged union, false = simple i32 tag)
    enums: BTreeMap<String, bool>,
    /// Track enum variant tags: EnumName_VariantName -> tag integer
    enum_variants: BTreeMap<String, i32>,
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

fn ast_type_to_string(ty: &Type) -> String {
    match ty {
        Type::Primitive(name, _) => name.clone(),
        Type::Ident(name, _) => name.clone(),
        Type::Reference { mutable, inner, .. } => {
            let mut_str = if *mutable { "mut " } else { "" };
            format!("&{}{}", mut_str, ast_type_to_string(inner))
        }
        Type::Generic { base, args: _, .. } => base.clone(),
        Type::Array {
            element, size: _, ..
        } => {
            format!("[{}]", ast_type_to_string(element))
        }
    }
}

impl LlvmEmitter {
    pub fn new() -> Self {
        let mut functions = BTreeMap::new();
        // Pre-populate runtime function return types
        functions.insert(
            "String_new".into(),
            (vec!["String".to_string()], "ptr".into()),
        );
        functions.insert(
            "File_read_to_string".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        functions.insert(
            "yfile_read_to_string".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        functions.insert(
            "ystr_new".into(),
            (vec!["String".to_string()], "ptr".into()),
        );
        functions.insert(
            "ystr_clone".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        functions.insert("yvec_new".into(), (vec!["i64".to_string()], "ptr".into()));
        functions.insert(
            "yvec_get".into(),
            (vec!["&Vec".to_string(), "usize".to_string()], "ptr".into()),
        );
        functions.insert("malloc".into(), (vec!["usize".to_string()], "ptr".into()));
        functions.insert(
            "File_write".into(),
            (
                vec!["&String".to_string(), "&String".to_string()],
                "void".into(),
            ),
        );
        functions.insert(
            "yfile_write".into(),
            (
                vec!["&String".to_string(), "&String".to_string()],
                "void".into(),
            ),
        );
        functions.insert(
            "println".into(),
            (vec!["&String".to_string()], "void".into()),
        );
        functions.insert("print".into(), (vec!["&String".to_string()], "void".into()));
        functions.insert("print_int".into(), (vec!["i64".to_string()], "void".into()));

        Self {
            output: String::new(),
            string_constants: Vec::new(),
            string_counter: 0,
            tmp_counter: 0,
            label_counter: 0,
            current_impl_target: None,
            locals: BTreeMap::new(),
            locals_ast_type: BTreeMap::new(),
            pointee_types: BTreeMap::new(),
            functions,
            structs: BTreeMap::new(),
            ast_structs: HashMap::new(),
            enums: BTreeMap::new(),
            enum_variants: BTreeMap::new(),
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
            writeln!(
                &mut self.output,
                "  ; NOTE: struct type coerce pass-through {} -> {}",
                src_ty, dst_ty
            )
            .unwrap();
            return val.to_string();
        }

        let tmp = self.fresh_tmp();
        let src_float = src_ty == "float" || src_ty == "double" || src_ty == "half";
        let dst_float = dst_ty == "float" || dst_ty == "double" || dst_ty == "half";
        let src_ptr = src_ty == "ptr";
        let dst_ptr = dst_ty == "ptr";
        let src_int = !src_float && !src_ptr;
        let dst_int = !dst_float && !dst_ptr;

        if src_ptr && dst_int {
            // ptr -> integer
            writeln!(
                &mut self.output,
                "  {} = ptrtoint ptr {} to {}",
                tmp, val, dst_ty
            )
            .unwrap();
        } else if src_int && dst_ptr {
            // integer -> ptr
            writeln!(
                &mut self.output,
                "  {} = inttoptr {} {} to ptr",
                tmp, src_ty, val
            )
            .unwrap();
        } else if src_float && dst_int {
            // float -> integer (signed)
            writeln!(
                &mut self.output,
                "  {} = fptosi {} {} to {}",
                tmp, src_ty, val, dst_ty
            )
            .unwrap();
        } else if src_int && dst_float {
            // integer -> float (signed)
            writeln!(
                &mut self.output,
                "  {} = sitofp {} {} to {}",
                tmp, src_ty, val, dst_ty
            )
            .unwrap();
        } else if src_float && dst_float {
            // float <-> float (truncate or extend)
            let src_bits: u32 = if src_ty == "double" {
                64
            } else if src_ty == "float" {
                32
            } else {
                16
            };
            let dst_bits: u32 = if dst_ty == "double" {
                64
            } else if dst_ty == "float" {
                32
            } else {
                16
            };
            if src_bits > dst_bits {
                writeln!(
                    &mut self.output,
                    "  {} = fptrunc {} {} to {}",
                    tmp, src_ty, val, dst_ty
                )
                .unwrap();
            } else {
                writeln!(
                    &mut self.output,
                    "  {} = fpext {} {} to {}",
                    tmp, src_ty, val, dst_ty
                )
                .unwrap();
            }
        } else if src_int && dst_int {
            // integer <-> integer (different widths)
            let src_bits = Self::int_bits(src_ty);
            let dst_bits = Self::int_bits(dst_ty);
            if src_bits > dst_bits {
                writeln!(
                    &mut self.output,
                    "  {} = trunc {} {} to {}",
                    tmp, src_ty, val, dst_ty
                )
                .unwrap();
            } else if src_bits < dst_bits {
                writeln!(
                    &mut self.output,
                    "  {} = sext {} {} to {}",
                    tmp, src_ty, val, dst_ty
                )
                .unwrap();
            } else {
                return val.to_string();
            }
        } else if src_ptr && dst_ptr {
            return val.to_string(); // ptr -> ptr, no conversion needed in opaque-ptr mode
        } else if src_float && dst_ptr {
            // float -> ptr via intermediate int
            let int_tmp = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {} = fptosi {} {} to i64",
                int_tmp, src_ty, val
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  {} = inttoptr i64 {} to ptr",
                tmp, int_tmp
            )
            .unwrap();
        } else if src_ptr && dst_float {
            // ptr -> float via intermediate int (PRESERVING BITS using bitcast)
            let int_tmp = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {} = ptrtoint ptr {} to i64",
                int_tmp, val
            )
            .unwrap();

            if dst_ty == "double" {
                // 64-bit pointer fits perfectly into 64-bit double
                writeln!(
                    &mut self.output,
                    "  {} = bitcast i64 {} to double",
                    tmp, int_tmp
                )
                .unwrap();
            } else {
                // For 32-bit float, we must truncate the 64-bit pointer first
                let trunc_tmp = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {} = trunc i64 {} to i32",
                    trunc_tmp, int_tmp
                )
                .unwrap();
                writeln!(
                    &mut self.output,
                    "  {} = bitcast i32 {} to float",
                    tmp, trunc_tmp
                )
                .unwrap();
            }
        } else {
            // Unknown conversion — pass through without conversion
            writeln!(
                &mut self.output,
                "  ; WARN: unhandled coerce {} -> {}",
                src_ty, dst_ty
            )
            .unwrap();
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
        let escaped = s
            .replace('\\', "\\5C")
            .replace('\n', "\\0A")
            .replace('"', "\\22");
        let len = s.len() + 1; // +1 for null terminator
        let decl = format!(
            "@.str.{} = private unnamed_addr constant [{} x i8] c\"{}\\00\"",
            id, len, escaped
        );
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
        let res: String = match ty {
            Type::Primitive(name, _) => match name.as_str() {
                "I32" | "u32" | "i32" => "i32".into(),
                "I64" | "usize" | "i64" => "i64".into(),
                "F16" | "f16" => "half".into(),
                "F32" | "f32" => "float".into(),
                "F64" | "f64" => "double".into(),
                "bool" => "i1".into(),
                "char" | "i8" | "u8" => "i8".into(),
                "I16" | "u16" | "i16" => "i16".into(),
                "String" | "Vec" | "ptr" => "ptr".into(),
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
                "String" | "Vec" | "ptr" => "ptr".into(),
                other => {
                    if other == "ptr" {
                        "ptr".into()
                    } else if let Some(has_data) = self.enums.get(other) {
                        if *has_data {
                            format!("%{}", other)
                        } else {
                            "i32".into()
                        }
                    } else {
                        format!("%{}", other)
                    }
                }
            },
            Type::Reference { .. } => "ptr".into(),
            Type::Generic { base, .. } => match base.as_str() {
                "Vec" | "Option" | "Box" | "GlobalMemory" | "SharedMemory" => "ptr".into(),
                _ => "ptr".into(),
            },
            Type::Array { .. } => "ptr".into(),
        };
        if res == "%ptr" {
            "ptr".into()
        } else {
            res
        }
    }

    // ── Entry Point ─────────────────────────────────────────

    pub fn emit_program(
        &mut self,
        prog: &Program,
        profile: &crate::sentinel::HardwareProfile,
    ) -> String {
        // Phase 0: Collect struct layouts and function signatures
        self.functions.insert(
            "ystr_new".into(),
            (vec!["String".to_string()], "ptr".into()),
        );
        self.functions.insert(
            "ystr_len".into(),
            (vec!["&String".to_string()], "i64".into()),
        );
        self.functions.insert(
            "ystr_eq".into(),
            (
                vec!["String".to_string(), "String".to_string()],
                "i1".into(),
            ),
        );
        self.functions.insert(
            "ystr_eq_cstr".into(),
            (vec!["String".to_string(), "ptr".to_string()], "i1".into()),
        );
        self.functions.insert(
            "ystr_push".into(),
            (
                vec!["String".to_string(), "char".to_string()],
                "void".into(),
            ),
        );
        self.functions.insert(
            "ystr_push_str".into(),
            (
                vec!["String".to_string(), "String".to_string()],
                "void".into(),
            ),
        );
        self.functions.insert(
            "ystr_free".into(),
            (vec!["String".to_string()], "void".into()),
        );
        self.functions.insert(
            "ystr_char_at".into(),
            (
                vec!["&String".to_string(), "usize".to_string()],
                "i8".into(),
            ),
        );
        self.functions.insert(
            "ystr_clone".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        self.functions
            .insert("yvec_new".into(), (vec!["i64".to_string()], "ptr".into()));
        self.functions.insert(
            "yvec_push".into(),
            (vec!["ptr".to_string(), "ptr".to_string()], "void".into()),
        );
        self.functions
            .insert("yvec_free".into(), (vec!["ptr".to_string()], "void".into()));
        self.functions
            .insert("yvec_len".into(), (vec!["&Vec".to_string()], "i64".into()));
        self.functions.insert(
            "yvec_get".into(),
            (vec!["&Vec".to_string(), "usize".to_string()], "ptr".into()),
        );
        self.functions.insert(
            "yvec_get_char".into(),
            (vec!["&Vec".to_string(), "usize".to_string()], "i8".into()),
        );
        self.functions.insert(
            "yfile_read_to_string".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        self.functions.insert(
            "yfile_write".into(),
            (
                vec!["&String".to_string(), "&String".to_string()],
                "void".into(),
            ),
        );
        self.functions
            .insert("printf".into(), (vec!["ptr".to_string()], "i32".into())); // variadic
        self.functions
            .insert("malloc".into(), (vec!["usize".to_string()], "ptr".into()));
        self.functions
            .insert("free".into(), (vec!["ptr".to_string()], "void".into()));
        self.functions
            .insert("exit".into(), (vec!["i32".to_string()], "void".into()));
        self.functions.insert(
            "ylexer_log".into(),
            (vec!["usize".to_string(), "char".to_string()], "void".into()),
        );
        self.functions.insert(
            "println".into(),
            (vec!["&String".to_string()], "void".into()),
        );
        self.functions
            .insert("print_int".into(), (vec!["i64".to_string()], "void".into()));

        for item in &prog.items {
            match item {
                Item::Struct(s) => {
                    let mut fields = Vec::new();
                    let mut ast_fields = Vec::new();
                    for f in &s.fields {
                        fields.push((f.name.clone(), self.emit_type(&f.ty)));
                        ast_fields.push((f.name.clone(), ast_type_to_string(&f.ty)));
                    }
                    self.structs.insert(s.name.clone(), fields);
                    self.ast_structs.insert(s.name.clone(), ast_fields);
                }
                Item::Func(f) => {
                    let ret_ty = f
                        .ret_ty
                        .as_ref()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "void".into());
                    let param_tys: Vec<String> =
                        f.params.iter().map(|p| ast_type_to_string(&p.ty)).collect();
                    self.functions.insert(f.name.clone(), (param_tys, ret_ty));
                }
                Item::Impl(imp) => {
                    for m in &imp.methods {
                        let ret_ty = m
                            .ret_ty
                            .as_ref()
                            .map(|t| self.emit_type(t))
                            .unwrap_or_else(|| "void".into());
                        let param_tys: Vec<String> =
                            m.params.iter().map(|p| ast_type_to_string(&p.ty)).collect();
                        self.functions.insert(
                            format!("{}_{}", imp.target_type, m.name),
                            (param_tys, ret_ty),
                        );
                    }
                }
                Item::Kernel(k) => {
                    let param_tys: Vec<String> =
                        k.params.iter().map(|p| ast_type_to_string(&p.ty)).collect();
                    self.functions
                        .insert(k.name.clone(), (param_tys, "void".into()));
                }
                Item::Enum(e) => {
                    let has_data = e.variants.iter().any(|v| v.fields.is_some());
                    self.enums.insert(e.name.clone(), has_data);
                    for (i, v) in e.variants.iter().enumerate() {
                        self.enum_variants
                            .insert(format!("{}_{}", e.name, v.name), i as i32);
                    }
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
                self.wln(&format!(
                    "%{} = type {{ {} }}",
                    s.name,
                    field_tys.join(", ")
                ));
            }
        }
        self.wln("");

        // Emit Enum definitions (tagged union layout)
        self.wln("; --- Enum Definitions ---");
        for item in &prog.items {
            if let Item::Enum(e) = item {
                let has_data = e.variants.iter().any(|v| v.fields.is_some());
                if has_data {
                    // LLVM represents tagged unions as { i32, [8 x i64] }
                    self.wln(&format!("%{} = type {{ i32, [8 x i64] }}", e.name));
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
        self.wln("declare void @print_int(i64)");
        self.wln("declare void @llvm.prefetch.p0(ptr nocapture readonly, i32, i32, i32)");
        self.wln("declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)");
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
        self.wln("@.fmt.s = private unnamed_addr constant [3 x i8] c\"%s\\00\"");
        self.wln("@.fmt.d = private unnamed_addr constant [4 x i8] c\"%ld\\00\"");
        self.wln("");

        // Append function bodies
        self.output.push_str(&func_output);

        // Auto-declare any called functions that are not defined or already declared
        let runtime_set: std::collections::HashSet<&str> = [
            "ystr_new",
            "ystr_push",
            "ystr_push_str",
            "ystr_eq_cstr",
            "ystr_len",
            "ystr_char_at",
            "ystr_clone",
            "yvec_new",
            "yvec_push",
            "yvec_get",
            "yvec_len",
            "yfile_read_to_string",
            "yfile_write",
            "printf",
            "malloc",
            "free",
            "exit",
            "println",
            "print_int",
            "llvm.prefetch.p0",
            "load",
        ]
        .iter()
        .cloned()
        .collect();

        let defined_set: std::collections::HashSet<String> =
            self.defined_functions.iter().cloned().collect();
        let mut auto_declared: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut extern_decls = String::new();

        for fname in &self.called_functions {
            if !runtime_set.contains(fname.as_str())
                && !defined_set.contains(fname)
                && !auto_declared.contains(fname)
            {
                // Look up the return type from the functions table, or use hardcoded built-ins
                let ret_ty = match fname.as_str() {
                    "println" | "print" | "print_int" | "File_write" | "yfile_write"
                    | "yvec_push" | "ystr_push" | "ystr_push_str" => "void".into(),
                    "String_new"
                    | "File_read_to_string"
                    | "yfile_read_to_string"
                    | "ystr_new"
                    | "ystr_clone"
                    | "yvec_new"
                    | "yvec_get"
                    | "malloc" => "ptr".into(),
                    _ => self
                        .functions
                        .get(fname)
                        .map(|(_, r)| r.clone())
                        .unwrap_or_else(|| "i32".into()),
                };

                if ret_ty.starts_with('%') {
                    writeln!(&mut extern_decls, "declare void @{}(...)", fname).unwrap();
                } else {
                    writeln!(&mut extern_decls, "declare {} @{}(...)", ret_ty, fname).unwrap();
                }
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
        self.wln(&format!(
            ";  Hardware Profile: AVX={}, AVX512={}, L2 Line={}B",
            profile.has_avx, profile.has_avx512, profile.l2_line_size
        ));
        self.wln("; ================================================");
        self.wln("");
        self.wln("target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"");
        self.wln("target triple = \"x86_64-pc-windows-msvc\"");
        self.wln("");

        // Dynamically inject LLVM function attributes based on Sentinel Probe
        if profile.has_avx512 {
            self.wln("attributes #0 = { \"target-cpu\"=\"skylake-avx512\" \"target-features\"=\"+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl\" }");
        } else if profile.has_avx {
            self.wln(
                "attributes #0 = { \"target-cpu\"=\"haswell\" \"target-features\"=\"+avx2,+avx\" }",
            );
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

        let params: Vec<String> = f
            .params
            .iter()
            .map(|p| {
                let ty = self.emit_type(&p.ty);
                format!("{} %{}.arg", ty, p.name)
            })
            .collect();
        let params_str = params.join(", ");

        writeln!(
            &mut self.output,
            "define {} @{}({}) #0 {{",
            ret_type, func_name, params_str
        )
        .unwrap();
        self.wln("entry:");

        // Alloca for all params so we can store/load them by name
        for p in &f.params {
            let ty = self.emit_type(&p.ty);
            self.locals.insert(p.name.clone(), ty.clone());
            self.locals_ast_type
                .insert(p.name.clone(), ast_type_to_string(&p.ty));
            if let Some(pty) = self.get_pointee_type(&p.ty) {
                self.pointee_types.insert(p.name.clone(), pty);
            }
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
            } else if ret_type == "ptr" {
                self.wln("  ret ptr null");
            } else if ret_type == "i1" {
                self.wln("  ret i1 0");
            } else if ret_type == "i8" {
                self.wln("  ret i8 0");
            } else if ret_type == "i64" {
                self.wln("  ret i64 0");
            } else if ret_type.starts_with('%') {
                writeln!(&mut self.output, "  ret {} zeroinitializer", ret_type).unwrap();
            } else {
                writeln!(&mut self.output, "  ret {} 0", ret_type).unwrap();
            }
        }

        self.wln("}");
        self.wln("");
    }

    fn emit_alloca_for_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { name, ty, init, .. } => {
                    if !self.locals.contains_key(name) {
                        let ir_ty = match ty {
                            Some(t) => {
                                if let Some(pty) = self.get_pointee_type(t) {
                                    self.pointee_types.insert(name.clone(), pty);
                                }
                                self.emit_type(t)
                            }
                            None => {
                                if let Some(init_expr) = init {
                                    let init_ty = self.infer_type(init_expr);
                                    let pty = self.infer_struct_type(init_expr);
                                    if pty != "i32" {
                                        self.pointee_types.insert(name.clone(), pty);
                                    }
                                    init_ty
                                } else {
                                    "i32".into()
                                }
                            }
                        };
                        self.locals.insert(name.clone(), ir_ty.clone());
                        match ty {
                            Some(t) => {
                                self.locals_ast_type
                                    .insert(name.clone(), ast_type_to_string(t));
                            }
                            None => {
                                if let Some(init_expr) = init {
                                    let inferred_ast_ty = self.infer_ast_type(init_expr);
                                    if inferred_ast_ty != "Unknown" {
                                        self.locals_ast_type.insert(name.clone(), inferred_ast_ty);
                                    }
                                }
                            }
                        }
                        // Track struct/enum-typed locals for GEP base type inference
                        if ir_ty.starts_with('%') {
                            self.pointee_types.insert(name.clone(), ir_ty.clone());
                        }
                        writeln!(&mut self.output, "  %{} = alloca {}", name, ir_ty).unwrap();
                    }
                }
                Stmt::For { loop_var, body, .. } => {
                    self.locals.insert(loop_var.clone(), "i32".into());
                    writeln!(&mut self.output, "  %{} = alloca i32", loop_var).unwrap();
                    self.emit_alloca_for_block(body);
                }
                Stmt::If {
                    then_block,
                    else_block,
                    ..
                } => {
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
                Stmt::SafeBlock(b, _) => {
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

        let params: Vec<String> = k
            .params
            .iter()
            .map(|p| {
                let ty = self.emit_type(&p.ty);
                format!("{} %{}.arg", ty, p.name)
            })
            .collect();

        writeln!(
            &mut self.output,
            "define void @{}({}) #0 {{",
            k.name,
            params.join(", ")
        )
        .unwrap();
        self.wln("entry:");
        self.defined_functions.push(k.name.clone());

        for p in &k.params {
            let ty = self.emit_type(&p.ty);
            self.locals.insert(p.name.clone(), ty.clone());
            self.locals_ast_type
                .insert(p.name.clone(), ast_type_to_string(&p.ty));
            if let Some(pty) = self.get_pointee_type(&p.ty) {
                self.pointee_types.insert(p.name.clone(), pty);
            }
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
            Stmt::Let {
                name,
                init,
                cache_policy,
                ..
            } => {
                if let Some(cp) = cache_policy {
                    self.current_cache_policy = Some(cp.policy.clone());
                }

                // alloca is already done in entry
                if let Some(init_expr) = init {
                    // Set load hint so `load()` intrinsic uses the LHS type
                    let dst_ty = self
                        .locals
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| "i32".into());
                    self.current_load_hint = Some(dst_ty.clone());
                    let target_ptr = format!("%{}", name);
                    let val =
                        self.emit_expr(init_expr, Some(target_ptr.clone()), Some(dst_ty.clone()));
                    let val_ty = self.infer_type(init_expr);
                    self.current_load_hint = None;

                    if matches!(init_expr, Expr::ZeroInit(_)) {
                        // For ZeroInit, the target pointer has already been memset. No further store needed.
                    } else {
                        let coerced = self.emit_coerce(&val, &val_ty, &dst_ty);

                        // ==========================================
                        // ARCHITECTURAL NOTE: Aggregate Memory Handling
                        // ==========================================
                        // LLVM differentiates between primitive (scalar) types and aggregate types (structs/arrays).
                        // While scalar variables can be directly assigned via `store`, aggregate types are essentially
                        // memory blocks. Assigning an aggregate requires explicitly copying its memory footprint.
                        //
                        // Direct Store vs Memcpy Decision:
                        // 1. If the target is a primitive type (i32, ptr, double), we emit a direct `store` instruction.
                        // 2. If the target is an aggregate type (starts with `%` for structs or `[` for arrays), we calculate
                        //    its byte size via GEP/ptrtoint and emit an `@llvm.memcpy` to bulk-copy the data. If the source
                        //    value was returned directly in a register rather than memory, we first dump it to a temporary alloca
                        //    so memcpy has a valid source pointer.
                        // ==========================================

                        if dst_ty.starts_with('%') || dst_ty.starts_with('[') {
                            let size_tmp_ptr = self.fresh_tmp();
                            let size_tmp = self.fresh_tmp();
                            writeln!(
                                &mut self.output,
                                "  {} = getelementptr {}, ptr null, i32 1",
                                size_tmp_ptr, dst_ty
                            )
                            .unwrap();
                            writeln!(
                                &mut self.output,
                                "  {} = ptrtoint ptr {} to i64",
                                size_tmp, size_tmp_ptr
                            )
                            .unwrap();

                            let is_aggregate_type = val_ty.starts_with('%') || val_ty.starts_with('[');
                            let is_registered_type = self.structs.contains_key(dst_ty.trim_start_matches('%'))
                                || self.enums.contains_key(dst_ty.trim_start_matches('%'));

                            let src_ptr = if is_aggregate_type && is_registered_type {
                                let tmp_ptr = self.fresh_tmp();
                                writeln!(&mut self.output, "  {} = alloca {}", tmp_ptr, dst_ty)
                                    .unwrap();
                                writeln!(
                                    &mut self.output,
                                    "  store {} {}, ptr {}",
                                    dst_ty, coerced, tmp_ptr
                                )
                                .unwrap();
                                tmp_ptr
                            } else {
                                coerced.clone()
                            };
                            writeln!(&mut self.output, "  call void @llvm.memcpy.p0.p0.i64(ptr align 8 {}, ptr align 8 {}, i64 {}, i1 false)", target_ptr, src_ptr, size_tmp).unwrap();
                        } else {
                            self.emit_store(&coerced, &target_ptr, &dst_ty);
                        }
                    }
                }

                self.current_cache_policy = None;
            }
            Stmt::Assign { target, value, .. } => {
                let target_addr = self.emit_lvalue(target);
                let dst_ty = self.infer_type(target);
                let val = self.emit_expr(value, Some(target_addr.clone()), Some(dst_ty.clone()));
                let val_ty = self.infer_type(value);

                if matches!(value, Expr::ZeroInit(_)) {
                    // ZeroInit handles memset directly into target_addr.
                } else {
                    let coerced = self.emit_coerce(&val, &val_ty, &dst_ty);

                    // See ARCHITECTURAL NOTE in Stmt::Let for aggregate vs primitive logic.
                    if dst_ty.starts_with('%') || dst_ty.starts_with('[') {
                        let size_tmp_ptr = self.fresh_tmp();
                        let size_tmp = self.fresh_tmp();
                        writeln!(
                            &mut self.output,
                            "  {} = getelementptr {}, ptr null, i32 1",
                            size_tmp_ptr, dst_ty
                        )
                        .unwrap();
                        writeln!(
                            &mut self.output,
                            "  {} = ptrtoint ptr {} to i64",
                            size_tmp, size_tmp_ptr
                        )
                        .unwrap();

                        let is_aggregate_type = val_ty.starts_with('%') || val_ty.starts_with('[');
                        let is_registered_type = self.structs.contains_key(dst_ty.trim_start_matches('%'))
                            || self.enums.contains_key(dst_ty.trim_start_matches('%'));

                        let src_ptr = if is_aggregate_type && is_registered_type {
                            let tmp_ptr = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = alloca {}", tmp_ptr, dst_ty)
                                .unwrap();
                            writeln!(
                                &mut self.output,
                                "  store {} {}, ptr {}",
                                dst_ty, coerced, tmp_ptr
                            )
                            .unwrap();
                            tmp_ptr
                        } else {
                            coerced.clone()
                        };
                        writeln!(&mut self.output, "  call void @llvm.memcpy.p0.p0.i64(ptr align 8 {}, ptr align 8 {}, i64 {}, i1 false)", target_addr, src_ptr, size_tmp).unwrap();
                    } else {
                        self.emit_store(&coerced, &target_addr, &dst_ty);
                    }
                }
            }
            Stmt::Return(expr, _) => {
                if let Some(e) = expr {
                    let val = self.emit_expr(e, None, None);
                    let val_ty = self.infer_type(e);
                    let coerced = self.emit_coerce(&val, &val_ty, ret_type);
                    writeln!(&mut self.output, "  ret {} {}", ret_type, coerced).unwrap();
                } else {
                    self.wln("  ret void");
                }
                self.block_terminated = true;
            }
            Stmt::Expr(e) => {
                self.emit_expr(e, None, None);
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let cond = self.emit_expr(condition, None, None);
                let then_lbl = self.fresh_label("then");
                let else_lbl = self.fresh_label("else");
                let merge_lbl = self.fresh_label("merge");

                writeln!(
                    &mut self.output,
                    "  br i1 {}, label %{}, label %{}",
                    cond,
                    then_lbl,
                    if else_block.is_some() {
                        &else_lbl
                    } else {
                        &merge_lbl
                    }
                )
                .unwrap();

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
            Stmt::While {
                condition, body, ..
            } => {
                let cond_lbl = self.fresh_label("while.cond");
                let body_lbl = self.fresh_label("while.body");
                let end_lbl = self.fresh_label("while.end");

                writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();
                writeln!(&mut self.output, "{}:", cond_lbl).unwrap();
                let cond = self.emit_expr(condition, None, None);
                writeln!(
                    &mut self.output,
                    "  br i1 {}, label %{}, label %{}",
                    cond, body_lbl, end_lbl
                )
                .unwrap();

                writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                self.block_terminated = false;
                self.emit_block_body(body, ret_type);
                if !self.block_terminated {
                    writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();
                }

                writeln!(&mut self.output, "{}:", end_lbl).unwrap();
                self.block_terminated = false;
            }
            Stmt::For {
                loop_var,
                start,
                end,
                step,
                body,
                ..
            } => {
                let s = self.emit_expr(start, None, None);
                let e = self.emit_expr(end, None, None);
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
                writeln!(
                    &mut self.output,
                    "  br i1 {}, label %{}, label %{}",
                    cmp, body_lbl, end_lbl
                )
                .unwrap();

                writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                self.block_terminated = false;
                self.emit_block_body(body, ret_type);

                // Increment
                let step_val = if let Some(st) = step {
                    self.emit_expr(st, None, None)
                } else {
                    "1".into()
                };
                let loaded = self.emit_load(&format!("%{}", loop_var), "i32");
                let incremented = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {} = add i32 {}, {}",
                    incremented, loaded, step_val
                )
                .unwrap();
                self.emit_store(&incremented, &format!("%{}", loop_var), "i32");
                writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();

                writeln!(&mut self.output, "{}:", end_lbl).unwrap();
                self.block_terminated = false;
            }
            Stmt::CompoundAssign {
                target, op, value, ..
            } => {
                let addr = self.emit_lvalue(target);
                let rhs = self.emit_expr(value, None, None);
                let ty = self.infer_type(target);
                let loaded = self.emit_load(&addr, &ty);
                let result = self.fresh_tmp();
                let op_str = self.binop_to_llvm(op, &ty);
                writeln!(
                    &mut self.output,
                    "  {} = {} {} {}, {}",
                    result, op_str, ty, loaded, rhs
                )
                .unwrap();
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
            Stmt::Match {
                scrutinee, arms, ..
            } => {
                let scrut_val = self.emit_expr(scrutinee, None, None);
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
                            let lit_val = self.emit_expr(lit, None, None);
                            let cmp = self.fresh_tmp();
                            let cmp_instr = if scrut_ty == "float" || scrut_ty == "double" {
                                "fcmp oeq"
                            } else {
                                "icmp eq"
                            };
                            writeln!(
                                &mut self.output,
                                "  {} = {} {} {}, {}",
                                cmp, cmp_instr, scrut_ty, scrut_val, lit_val
                            )
                            .unwrap();
                            writeln!(
                                &mut self.output,
                                "  br i1 {}, label %{}, label %{}",
                                cmp, body_lbl, next_test
                            )
                            .unwrap();
                        }
                        MatchPattern::Ident(name, _) => {
                            // Bind variable then always match
                            let cmp = self.fresh_tmp();
                            writeln!(
                                &mut self.output,
                                "  {} = icmp eq {} {}, {}",
                                cmp, scrut_ty, scrut_val, name
                            )
                            .unwrap();
                            writeln!(
                                &mut self.output,
                                "  br i1 {}, label %{}, label %{}",
                                cmp, body_lbl, next_test
                            )
                            .unwrap();
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
                            writeln!(
                                &mut self.output,
                                "  {} = icmp eq {} {}, {} ; enum {}",
                                cmp, scrut_ty, scrut_val, tag_name, variant
                            )
                            .unwrap();
                            writeln!(
                                &mut self.output,
                                "  br i1 {}, label %{}, label %{}",
                                cmp, body_lbl, next_test
                            )
                            .unwrap();
                        }
                    }

                    writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                    self.block_terminated = false;
                    self.emit_expr(&arm.body, None, None);
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
            Stmt::SafeBlock(block, _) => {
                self.wln("  ; --- @safe verified block ---");
                self.emit_block_body(block, ret_type);
            }
        }
    }

    // ── Expression Emission ─────────────────────────────────

    /// Emit an lvalue (address) for assignment targets — returns ptr
    fn emit_lvalue(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => format!("%{}", name),
            Expr::MemberAccess { base, member, .. } => {
                let (base_val, base_ty) = if let Expr::UnaryOp { op: UnaryOp::Deref, operand: inner, .. } = &**base {
                    (self.emit_expr(inner, None, None), self.infer_struct_type(inner))
                } else {
                    (self.emit_lvalue(base), self.infer_struct_type(base))
                };
                let tmp = self.fresh_tmp();

                // Handle tagged union synthetic fields
                let base_name = base_ty.trim_start_matches('%');
                if let Some(&has_data) = self.enums.get(base_name) {
                    if has_data {
                        writeln!(&mut self.output, "  ; lvalue .{}", member).unwrap();
                        if member == "tag" {
                            // .tag -> index 0 (i32 discriminator)
                            writeln!(
                                &mut self.output,
                                "  {} = getelementptr {}, ptr {}, i32 0, i32 0",
                                tmp, base_ty, base_val
                            )
                            .unwrap();
                            return tmp;
                        } else if member == "data" {
                            // .data -> index 1 (payload: [8 x i64])
                            writeln!(
                                &mut self.output,
                                "  {} = getelementptr {}, ptr {}, i32 0, i32 1",
                                tmp, base_ty, base_val
                            )
                            .unwrap();
                            return tmp;
                        }
                    }
                }

                if base_ty == "[8 x i64]" {
                    writeln!(&mut self.output, "  ; lvalue payload overlay .{}", member).unwrap();
                    if member.starts_with('_') {
                        // ._N -> index N into the [8 x i64] payload
                        let idx: usize = member[1..].parse().unwrap_or(0);
                        writeln!(
                            &mut self.output,
                            "  {} = getelementptr [8 x i64], ptr {}, i32 0, i32 {}",
                            tmp, base_val, idx
                        )
                        .unwrap();
                        return tmp;
                    } else {
                        // .VariantName -> pass-through (overlay on the data payload)
                        return base_val;
                    }
                }

                let mut field_index = 0;
                if let Some(fields) = self.structs.get(base_name) {
                    for (i, (fname, _)) in fields.iter().enumerate() {
                        if fname == member {
                            field_index = i;
                            break;
                        }
                    }
                }

                writeln!(&mut self.output, "  ; lvalue .{}", member).unwrap();
                writeln!(
                    &mut self.output,
                    "  {} = getelementptr {}, ptr {}, i32 0, i32 {}",
                    tmp, base_ty, base_val, field_index
                )
                .unwrap();
                tmp
            }
            Expr::Index { base, index, .. } => {
                let base_val = self.emit_expr(base, None, None);
                let idx_val = self.emit_expr(index, None, None);
                let base_ty = self.infer_type(base);
                let elem_ty = if base_ty == "ptr" {
                    "i64"
                } else {
                    base_ty.as_str()
                };
                let tmp = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {} = getelementptr {}, ptr {}, i64 {}",
                    tmp, elem_ty, base_val, idx_val
                )
                .unwrap();
                tmp
            }
            Expr::UnaryOp {
                op: UnaryOp::Deref,
                operand,
                ..
            } => self.emit_expr(operand, None, None),
            _ => self.emit_expr(expr, None, None),
        }
    }

    fn emit_expr(
        &mut self,
        expr: &Expr,
        target: Option<String>,
        expected_ty: Option<String>,
    ) -> String {
        match expr {
            Expr::IntLit(val, _) => format!("{}", val),
            Expr::FloatLit(val, _) => format!("{:.6e}", val),
            Expr::BoolLit(b, _) => {
                if *b {
                    "1".into()
                } else {
                    "0".into()
                }
            }
            Expr::CharLit(c, _) => format!("{}", *c as u32),
            Expr::Ident(name, _) => {
                // If it's a known enum variant, replace with integer
                if let Some(&tag) = self.enum_variants.get(name) {
                    return tag.to_string();
                }
                let mut tag_name = name.clone();
                if name.contains("_TAG_") {
                    tag_name = name.replace("_TAG_", "_");
                }
                if let Some(&tag) = self.enum_variants.get(&tag_name) {
                    return tag.to_string();
                }

                let ty = self
                    .locals
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| "i32".into());
                self.emit_load(&format!("%{}", name), &ty)
            }
            Expr::StringLit(s, _) => {
                let global_name = self.register_string(s);
                let tmp = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {} = call ptr @ystr_new(ptr {})",
                    tmp, global_name
                )
                .unwrap();
                tmp
            }
            Expr::BinaryOp {
                left, op, right, ..
            } => {
                let mut l = self.emit_expr(left, None, None);
                let mut r = self.emit_expr(right, None, None);
                let mut l_ty = self.infer_type(left);
                let r_ty = self.infer_type(right);

                // ==========================================
                // ARCHITECTURAL NOTE: BinaryOp Type Promotion
                // ==========================================
                // When executing binary operations (e.g. A + B), LLVM strictly requires both operands
                // to share the exact same type. If the frontend allows mixed-type expressions (like `int + float`),
                // we must automatically promote one of the scalars to match the wider type.
                //
                // Scalar Gating Logic:
                // 1. If both are floats, promote to the larger float precision.
                // 2. If one is float and the other is int, promote the int to the float type.
                // 3. If both are ints, promote to the larger integer bitwidth.
                // ==========================================

                // Promote types if there's a mismatch
                if l_ty != r_ty {
                    let l_is_float = l_ty == "float" || l_ty == "double" || l_ty == "half";
                    let r_is_float = r_ty == "float" || r_ty == "double" || r_ty == "half";

                    let common_ty = if l_is_float && r_is_float {
                        // Both floats, pick the larger one
                        let l_bits = if l_ty == "double" {
                            64
                        } else if l_ty == "float" {
                            32
                        } else {
                            16
                        };
                        let r_bits = if r_ty == "double" {
                            64
                        } else if r_ty == "float" {
                            32
                        } else {
                            16
                        };
                        if l_bits >= r_bits {
                            l_ty.clone()
                        } else {
                            r_ty.clone()
                        }
                    } else if l_is_float {
                        l_ty.clone()
                    } else if r_is_float {
                        r_ty.clone()
                    } else {
                        // Both ints, pick the larger one
                        let l_bits = Self::int_bits(&l_ty);
                        let r_bits = Self::int_bits(&r_ty);
                        if l_bits >= r_bits {
                            l_ty.clone()
                        } else {
                            r_ty.clone()
                        }
                    };

                    if l_ty != common_ty {
                        l = self.emit_coerce(&l, &l_ty, &common_ty);
                        l_ty = common_ty.clone();
                    }
                    if r_ty != common_ty {
                        r = self.emit_coerce(&r, &r_ty, &common_ty);
                        // r_ty = common_ty.clone(); // Not needed anymore
                    }
                }

                let ty = l_ty;
                let tmp = self.fresh_tmp();

                // Special case: Enum comparison (compare tags)
                let base_name = ty.trim_start_matches('%');
                if self.enums.contains_key(base_name)
                    && (op == &BinaryOp::Eq || op == &BinaryOp::NotEq)
                {
                    let l_tag = self.fresh_tmp();
                    let r_tag = self.fresh_tmp();
                    writeln!(
                        &mut self.output,
                        "  {} = extractvalue {} {}, 0",
                        l_tag, ty, l
                    )
                    .unwrap();
                    writeln!(
                        &mut self.output,
                        "  {} = extractvalue {} {}, 0",
                        r_tag, ty, r
                    )
                    .unwrap();
                    let instr = if op == &BinaryOp::Eq {
                        "icmp eq"
                    } else {
                        "icmp ne"
                    };
                    writeln!(
                        &mut self.output,
                        "  {} = {} i32 {}, {}",
                        tmp, instr, l_tag, r_tag
                    )
                    .unwrap();
                    return tmp;
                }

                let instr = self.binop_to_llvm(op, &ty);
                writeln!(
                    &mut self.output,
                    "  {} = {} {} {}, {}",
                    tmp, instr, ty, l, r
                )
                .unwrap();
                tmp
            }
            Expr::UnaryOp { op, operand, .. } => {
                let val = self.emit_expr(operand, None, None);
                let tmp = self.fresh_tmp();
                let ty = self.infer_type(operand);
                match op {
                    UnaryOp::Neg => {
                        if ty == "float" || ty == "double" {
                            writeln!(&mut self.output, "  {} = fneg {} {}", tmp, ty, val).unwrap();
                        } else {
                            writeln!(&mut self.output, "  {} = sub {} 0, {}", tmp, ty, val)
                                .unwrap();
                        }
                    }
                    UnaryOp::Not => {
                        writeln!(&mut self.output, "  {} = xor {} {}, 1", tmp, ty, val).unwrap();
                    }
                    UnaryOp::Ref => {
                        return self.emit_lvalue(operand);
                    }
                    UnaryOp::Deref => {
                        let inner_ty = self.infer_type(operand);
                        let load_ty = if inner_ty == "ptr" {
                            "i64"
                        } else {
                            inner_ty.as_str()
                        };
                        writeln!(
                            &mut self.output,
                            "  {} = load {}, ptr {}",
                            tmp, load_ty, val
                        )
                        .unwrap();
                    }
                }
                tmp
            }
            Expr::Call { func, args, .. } => {
                let func_name = self.emit_call_target(func);
                self.called_functions.push(func_name.clone());

                if (func_name.starts_with("String_")
                    || func_name.starts_with("Vec_")
                    || func_name.starts_with("File_")
                    || func_name.starts_with("yfile_")
                    || func_name.starts_with("ystr_")
                    || func_name.starts_with("yvec_"))
                    && args.len() >= 1
                {
                    if func_name == "Vec_push" && args.len() == 2 {
                        let vec_val = self.emit_expr(&args[0], None, None);
                        let elem_addr = self.emit_lvalue(&args[1]);
                        writeln!(
                            &mut self.output,
                            "  call void @Vec_push(ptr {}, ptr {})",
                            vec_val, elem_addr
                        )
                        .unwrap();
                        return self.fresh_tmp().replace("%t", "%_void");
                    }

                    let mut new_arg_strs = Vec::new();

                    let expected_params = self
                        .functions
                        .get(&func_name)
                        .map(|(p, _)| p.clone())
                        .unwrap_or_default();

                    for (i, arg) in args.iter().enumerate() {
                        let mut arg_val = self.emit_expr(arg, None, None);
                        let arg_ty = self.infer_type(arg);
                        let arg_ast = self.infer_ast_type(arg);

                        let param_ty = expected_params.get(i).map(|s| s.as_str()).unwrap_or("i32");

                        if arg_ast.starts_with('&') && arg_ast[1..] == *param_ty {
                            let tmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = load ptr, ptr {}", tmp, arg_val)
                                .unwrap();
                            arg_val = tmp;
                        }

                        let llvm_param_ty = match param_ty {
                            "String" | "&String" | "Vec" | "&Vec" | "ptr" => "ptr".to_string(),
                            "usize" | "i64" | "I64" => "i64".to_string(),
                            "i32" | "I32" => "i32".to_string(),
                            "bool" => "i1".to_string(),
                            "char" | "i8" => "i8".to_string(),
                            _ => {
                                if param_ty.starts_with('&') {
                                    "ptr".to_string()
                                } else {
                                    format!("%{}", param_ty)
                                }
                            }
                        };

                        if !arg_ty.starts_with('%')
                            && !llvm_param_ty.starts_with('%')
                            && arg_ty != "ptr"
                            && llvm_param_ty != "ptr"
                        {
                            arg_val = self.emit_coerce(&arg_val, &arg_ty, &llvm_param_ty);
                        }

                        if llvm_param_ty.starts_with('%') && arg_ty == "ptr" {
                            let tmp = self.fresh_tmp();
                            writeln!(
                                &mut self.output,
                                "  {} = load {}, ptr {}",
                                tmp, llvm_param_ty, arg_val
                            )
                            .unwrap();
                            new_arg_strs.push(format!("{} {}", llvm_param_ty, tmp));
                        } else if llvm_param_ty == "ptr" && arg_ty.starts_with('%') {
                            let tmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = alloca {}", tmp, arg_ty).unwrap();
                            writeln!(
                                &mut self.output,
                                "  store {} {}, ptr {}",
                                arg_ty, arg_val, tmp
                            )
                            .unwrap();
                            new_arg_strs.push(format!("ptr {}", tmp));
                        } else {
                            if llvm_param_ty == "ptr" {
                                new_arg_strs.push(format!("ptr {}", arg_val));
                            } else {
                                new_arg_strs.push(format!("{} {}", llvm_param_ty, arg_val));
                            }
                        }
                    }

                    if func_name.starts_with("Vec_get_") && args.len() == 2 {
                        let vec_val = &new_arg_strs[0].split_whitespace().last().unwrap();
                        let idx_val = &new_arg_strs[1].split_whitespace().last().unwrap();
                        let elem_ptr = self.fresh_tmp();
                        writeln!(
                            &mut self.output,
                            "  {} = call ptr @yvec_get(ptr {}, i64 {})",
                            elem_ptr, vec_val, idx_val
                        )
                        .unwrap();

                        let ret_type_name = &func_name[8..];
                        let llvm_ret_ty = match ret_type_name {
                            "usize" | "I64" | "i64" => "i64".to_string(),
                            "I32" | "i32" | "int" => "i32".to_string(),
                            "bool" => "i1".to_string(),
                            "char" => "i8".to_string(),
                            "String" | "Vec" | "ptr" => "ptr".to_string(),
                            _ => format!("%{}", ret_type_name),
                        };
                        let tmp = self.fresh_tmp();
                        writeln!(
                            &mut self.output,
                            "  {} = load {}, ptr {}",
                            tmp, llvm_ret_ty, elem_ptr
                        )
                        .unwrap();
                        return tmp;
                    }

                    let ret_ty: String = match func_name.as_str() {
                        "String_new"
                        | "String_clone"
                        | "Vec_new"
                        | "Vec_get"
                        | "File_read_to_string"
                        | "yfile_read_to_string"
                        | "ystr_new"
                        | "ystr_clone"
                        | "yvec_new"
                        | "yvec_get"
                        | "malloc" => "ptr".into(),
                        "String_len" | "ystr_len" | "Vec_len" | "yvec_len" => "i64".into(),
                        "String_eq" | "String_eq_cstr" | "ystr_eq" | "ystr_eq_cstr" => "i1".into(),
                        "String_char_at" | "ystr_char_at" | "yvec_get_char" => "i8".into(),
                        _ => "void".into(),
                    };

                    let tmp = self.fresh_tmp();
                    let args_joined = new_arg_strs.join(", ");
                    if ret_ty == "void" {
                        writeln!(
                            &mut self.output,
                            "  call void @{}({})",
                            func_name, args_joined
                        )
                        .unwrap();
                        return tmp.replace("%t", "%_void");
                    } else {
                        writeln!(
                            &mut self.output,
                            "  {} = call {} @{}({})",
                            tmp, ret_ty, func_name, args_joined
                        )
                        .unwrap();
                        return tmp;
                    }
                }

                let mut arg_strs = Vec::new();

                let expected_params = self
                    .functions
                    .get(&func_name)
                    .map(|(p, _)| p.clone())
                    .unwrap_or_default();

                for (i, a) in args.iter().enumerate() {
                    let param_ty = expected_params.get(i).map(|s| s.as_str()).unwrap_or("i32");

                    let mut arg_val = self.emit_expr(a, None, None);
                    let arg_ty = self.infer_type(a);
                    let arg_ast = self.infer_ast_type(a);

                    if arg_ast.starts_with('&') && arg_ast[1..] == *param_ty {
                        let tmp = self.fresh_tmp();
                        writeln!(&mut self.output, "  {} = load ptr, ptr {}", tmp, arg_val)
                            .unwrap();
                        arg_val = tmp;
                    }

                    let llvm_param_ty = match param_ty {
                        "String" | "&String" | "Vec" | "&Vec" | "ptr" => "ptr".to_string(),
                        "usize" | "i64" | "I64" => "i64".to_string(),
                        "i32" | "I32" => "i32".to_string(),
                        "bool" => "i1".to_string(),
                        "char" | "i8" => "i8".to_string(),
                        _ => {
                            if param_ty.starts_with('&') {
                                "ptr".to_string()
                            } else {
                                format!("%{}", param_ty)
                            }
                        }
                    };

                    if llvm_param_ty != "ptr" && !llvm_param_ty.starts_with('%') {
                        arg_val = self.emit_coerce(&arg_val, &arg_ty, &llvm_param_ty);
                    }

                    if llvm_param_ty.starts_with('%') && arg_ty == "ptr" {
                        let tmp = self.fresh_tmp();
                        writeln!(
                            &mut self.output,
                            "  {} = load {}, ptr {}",
                            tmp, llvm_param_ty, arg_val
                        )
                        .unwrap();
                        arg_strs.push(format!("{} {}", llvm_param_ty, tmp));
                    } else if llvm_param_ty == "ptr" && arg_ty.starts_with('%') {
                        let tmp = self.fresh_tmp();
                        writeln!(&mut self.output, "  {} = alloca {}", tmp, arg_ty).unwrap();
                        writeln!(
                            &mut self.output,
                            "  store {} {}, ptr {}",
                            arg_ty, arg_val, tmp
                        )
                        .unwrap();
                        arg_strs.push(format!("ptr {}", tmp));
                    } else {
                        if llvm_param_ty == "ptr" {
                            arg_strs.push(format!("ptr {}", arg_val));
                        } else {
                            arg_strs.push(format!("{} {}", llvm_param_ty, arg_val));
                        }
                    }
                }

                match func_name.as_str() {
                    "load" => {
                        let ptr_val = self.emit_expr(&args[0], None, None);
                        let tmp = self.fresh_tmp();
                        let mut metadata = String::new();

                        if let Some(policy) = &self.current_cache_policy.clone() {
                            if policy == "L2_EVICT_FIRST" {
                                metadata = ", !nontemporal !0".to_string();
                            } else if policy == "L2_PERSIST" {
                                // 0 = Read, 3 = High temporal locality, 1 = Data cache
                                writeln!(
                                    &mut self.output,
                                    "  call void @llvm.prefetch.p0(ptr {}, i32 0, i32 3, i32 1)",
                                    ptr_val
                                )
                                .unwrap();
                            }
                        }

                        // Infer load type from the LHS variable's alloca type.
                        // The caller (emit_stmt for Let) will coerce if needed.
                        // We use the type annotation from `self.current_let_type` if
                        // available, otherwise fall back to the pointer element type.
                        let load_ty = self.current_load_hint.clone().unwrap_or_else(|| {
                            // Infer from args: if loading from a typed pointer, use that type
                            let arg_ty = self.infer_type(&args[0]);
                            if arg_ty == "ptr" {
                                "double".into()
                            } else {
                                arg_ty
                            }
                        });
                        writeln!(
                            &mut self.output,
                            "  {} = load {}, ptr {}{}",
                            tmp, load_ty, ptr_val, metadata
                        )
                        .unwrap();
                        return tmp;
                    }
                    _ => {}
                }

                if let Some(&tag) = self.enum_variants.get(&func_name) {
                    let enum_name = func_name.split('_').next().unwrap();
                    let struct_name = format!("%{}", enum_name);

                    let alloc_tmp = self.fresh_tmp();
                    writeln!(&mut self.output, "  {} = alloca {}", alloc_tmp, struct_name).unwrap();

                    let tag_tmp = self.fresh_tmp();
                    writeln!(
                        &mut self.output,
                        "  {} = getelementptr {}, ptr {}, i32 0, i32 0",
                        tag_tmp, struct_name, alloc_tmp
                    )
                    .unwrap();
                    writeln!(&mut self.output, "  store i32 {}, ptr {}", tag, tag_tmp).unwrap();

                    let data_tmp = self.fresh_tmp();
                    writeln!(
                        &mut self.output,
                        "  {} = getelementptr {}, ptr {}, i32 0, i32 1",
                        data_tmp, struct_name, alloc_tmp
                    )
                    .unwrap();

                    if !args.is_empty() {
                        let val_val = self.emit_expr(&args[0], None, None);
                        let val_ty = self.infer_type(&args[0]);
                        writeln!(
                            &mut self.output,
                            "  store {} {}, ptr {}",
                            val_ty, val_val, data_tmp
                        )
                        .unwrap();
                    }

                    let res_tmp = self.fresh_tmp();
                    writeln!(
                        &mut self.output,
                        "  {} = load {}, ptr {}",
                        res_tmp, struct_name, alloc_tmp
                    )
                    .unwrap();
                    return res_tmp;
                }

                let ret_ty = match func_name.as_str() {
                    "println" | "print" | "print_int" | "File_write" | "yfile_write"
                    | "yvec_push" | "ystr_push" | "ystr_push_str" => "void".into(),
                    "String_new"
                    | "File_read_to_string"
                    | "yfile_read_to_string"
                    | "ystr_new"
                    | "ystr_clone"
                    | "yvec_new"
                    | "yvec_get"
                    | "malloc" => "ptr".into(),
                    _ => self
                        .functions
                        .get(&func_name)
                        .map(|(_, r)| r.clone())
                        .unwrap_or_else(|| "i32".into()),
                };
                let tmp = self.fresh_tmp();
                if ret_ty.starts_with('%') {
                    writeln!(
                        &mut self.output,
                        "  {} = call {} @{}({})",
                        tmp,
                        ret_ty,
                        func_name,
                        arg_strs.join(", ")
                    )
                    .unwrap();
                    tmp
                } else if ret_ty == "void" {
                    writeln!(
                        &mut self.output,
                        "  call void @{}({})",
                        func_name,
                        arg_strs.join(", ")
                    )
                    .unwrap();
                    tmp.replace("%t", "%_void")
                } else {
                    writeln!(
                        &mut self.output,
                        "  {} = call {} @{}({})",
                        tmp,
                        ret_ty,
                        func_name,
                        arg_strs.join(", ")
                    )
                    .unwrap();
                    tmp
                }
            }
            Expr::Path {
                namespace, member, ..
            } => {
                let full_name = format!("{}_{}", namespace, member);
                if let Some(&tag) = self.enum_variants.get(&full_name) {
                    if let Some(&has_data) = self.enums.get(namespace) {
                        if has_data {
                            return format!("{{ i32 {}, [8 x i64] zeroinitializer }}", tag);
                        }
                    }
                    tag.to_string()
                } else {
                    full_name
                }
            }
            Expr::MemberAccess { .. } => {
                let lval = self.emit_lvalue(expr);
                let field_ty = self.infer_type(expr);
                self.emit_load(&lval, &field_ty)
            }
            Expr::Index { .. } => {
                let lval = self.emit_lvalue(expr);
                let ty = self.infer_type(expr);
                self.emit_load(&lval, &ty)
            }
            Expr::SelfLit(_) => "%self".into(),
            Expr::ZeroInit(_) => {
                let ty = expected_ty
                    .or_else(|| self.current_load_hint.clone())
                    .unwrap_or_else(|| "i32".into());

                let target_ptr = target.unwrap_or_else(|| {
                    let tmp = self.fresh_tmp();
                    writeln!(&mut self.output, "  {} = alloca {}", tmp, ty).unwrap();
                    tmp
                });

                let size_tmp_ptr = self.fresh_tmp();
                let size_tmp = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {} = getelementptr {}, ptr null, i32 1",
                    size_tmp_ptr, ty
                )
                .unwrap();
                writeln!(
                    &mut self.output,
                    "  {} = ptrtoint ptr {} to i64",
                    size_tmp, size_tmp_ptr
                )
                .unwrap();
                writeln!(
                    &mut self.output,
                    "  call void @llvm.memset.p0.i64(ptr {}, i8 0, i64 {}, i1 false)",
                    target_ptr, size_tmp
                )
                .unwrap();

                return target_ptr;
            }
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
                    let val = self.emit_expr(fexpr, None, None);
                    let val_ty = self.infer_type(fexpr);
                    let coerced = self.emit_coerce(&val, &val_ty, &field_ty);
                    let new_val = self.fresh_tmp();
                    writeln!(
                        &mut self.output,
                        "  {} = insertvalue {} {}, {} {}, {}",
                        new_val, ty, current_val, field_ty, coerced, field_idx
                    )
                    .unwrap();
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
                    let v = self.emit_expr(a, None, None);
                    let ty = self.infer_type(a);
                    arg_strs.push(format!("{} {}", ty, v));
                }
                let ret_ty = self
                    .functions
                    .get(&func_name)
                    .map(|(_, r)| r.clone())
                    .unwrap_or_else(|| "i32".into());
                let tmp = self.fresh_tmp();
                if ret_ty.starts_with('%') {
                    let sret_alloc = self.fresh_tmp();
                    writeln!(&mut self.output, "  {} = alloca {}", sret_alloc, ret_ty).unwrap();
                    let mut sret_arg_strs = vec![format!("ptr {}", sret_alloc)];
                    sret_arg_strs.extend(arg_strs);
                    writeln!(
                        &mut self.output,
                        "  call void @{}({})",
                        func_name,
                        sret_arg_strs.join(", ")
                    )
                    .unwrap();
                    let res_tmp = self.fresh_tmp();
                    writeln!(
                        &mut self.output,
                        "  {} = load {}, ptr {}",
                        res_tmp, ret_ty, sret_alloc
                    )
                    .unwrap();
                    res_tmp
                } else if ret_ty == "void" {
                    writeln!(
                        &mut self.output,
                        "  call void @{}({})",
                        func_name,
                        arg_strs.join(", ")
                    )
                    .unwrap();
                    tmp.replace("%t", "%_void")
                } else {
                    writeln!(
                        &mut self.output,
                        "  {} = call {} @{}({})",
                        tmp,
                        ret_ty,
                        func_name,
                        arg_strs.join(", ")
                    )
                    .unwrap();
                    tmp
                }
            }
            _ => {
                let tmp = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {} = add i32 0, 0 ; unhandled expr",
                    tmp
                )
                .unwrap();
                tmp
            }
        }
    }

    fn emit_call_target(&self, func: &Expr) -> String {
        match func {
            Expr::Ident(name, _) => name.clone(),
            Expr::Path {
                namespace, member, ..
            } => format!("{}_{}", namespace, member),
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
        let is_float = ty == "float" || ty == "double" || ty == "half";
        match op {
            BinaryOp::Add => {
                if is_float {
                    "fadd"
                } else {
                    "add"
                }
            }
            BinaryOp::Sub => {
                if is_float {
                    "fsub"
                } else {
                    "sub"
                }
            }
            BinaryOp::Mul => {
                if is_float {
                    "fmul"
                } else {
                    "mul"
                }
            }
            BinaryOp::Div => {
                if is_float {
                    "fdiv"
                } else {
                    "sdiv"
                }
            }
            BinaryOp::Mod => {
                if is_float {
                    "frem"
                } else {
                    "srem"
                }
            }
            BinaryOp::Eq => {
                if is_float {
                    "fcmp oeq"
                } else {
                    "icmp eq"
                }
            }
            BinaryOp::NotEq => {
                if is_float {
                    "fcmp one"
                } else {
                    "icmp ne"
                }
            }
            BinaryOp::Lt => {
                if is_float {
                    "fcmp olt"
                } else {
                    "icmp slt"
                }
            }
            BinaryOp::Gt => {
                if is_float {
                    "fcmp ogt"
                } else {
                    "icmp sgt"
                }
            }
            BinaryOp::Le => {
                if is_float {
                    "fcmp ole"
                } else {
                    "icmp sle"
                }
            }
            BinaryOp::Ge => {
                if is_float {
                    "fcmp oge"
                } else {
                    "icmp sge"
                }
            }
            BinaryOp::And | BinaryOp::BitAnd => "and",
            BinaryOp::Or | BinaryOp::BitOr => "or",
            BinaryOp::BitXor => "xor",
            BinaryOp::Shl => "shl",
            BinaryOp::Shr => "ashr",
        }
    }

    fn infer_ast_type(&self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => {
                if let Some(ast_ty) = self.locals_ast_type.get(name) {
                    return ast_ty.clone();
                }
                "Unknown".into()
            }
            Expr::UnaryOp {
                op: UnaryOp::Ref,
                operand,
                ..
            } => {
                let inner = self.infer_ast_type(operand);
                format!("&{}", inner)
            }
            Expr::UnaryOp {
                op: UnaryOp::Deref,
                operand,
                ..
            } => {
                let inner = self.infer_ast_type(operand);
                if let Some(stripped) = inner.strip_prefix("&mut ") {
                    stripped.to_string()
                } else if let Some(stripped) = inner.strip_prefix('&') {
                    stripped.to_string()
                } else {
                    inner
                }
            }
            Expr::MemberAccess { base, member, .. } => {
                // Approximate base ty
                let base_ty = if let Expr::UnaryOp { op: UnaryOp::Deref, operand, .. } = &**base {
                    self.infer_ast_type(operand)
                } else {
                    self.infer_ast_type(base)
                };
                let struct_name = base_ty.trim_start_matches('&');

                if let Some(fields) = self.ast_structs.get(struct_name) {
                    for (fname, fty) in fields {
                        if fname == member {
                            return fty.clone();
                        }
                    }
                }
                "Unknown".into()
            }
            Expr::Call { func, .. } => {
                let func_name = self.emit_call_target(func);
                if let Some((_, ret_ast_ty)) = self.functions.get(&func_name) {
                    ret_ast_ty.clone()
                } else {
                    "Unknown".into()
                }
            }
            Expr::StringLit(_, _) => "String".into(),
            Expr::IntLit(_, _) => "i64".into(),
            Expr::FloatLit(_, _) => "f64".into(),
            Expr::BoolLit(_, _) => "bool".into(),
            Expr::CharLit(_, _) => "char".into(),
            Expr::StructLit { name, .. } => name.clone(),
            _ => "Unknown".into(),
        }
    }

    fn infer_type(&self, expr: &Expr) -> String {
        match expr {
            Expr::IntLit(_, _) => "i32".into(),
            Expr::FloatLit(_, _) => "double".into(),
            Expr::BoolLit(_, _) => "i1".into(),
            Expr::CharLit(_, _) => "i8".into(),
            Expr::StringLit(_, _) => "ptr".into(),
            Expr::Ident(name, _) => {
                if self.enum_variants.contains_key(name) {
                    return "i32".into();
                }
                let mut tag_name = name.clone();
                if name.contains("_TAG_") {
                    tag_name = name.replace("_TAG_", "_");
                }
                if self.enum_variants.contains_key(&tag_name) {
                    return "i32".into();
                }
                self.locals
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| "i32".into())
            }
            Expr::Call { func, .. } => {
                let func_name = self.emit_call_target(func);
                // Fix 2: enum constructor calls return the enum struct type
                if self.enum_variants.contains_key(&func_name) {
                    let enum_name = func_name.split('_').next().unwrap();
                    return format!("%{}", enum_name);
                }
                match func_name.as_str() {
                    "load" => {
                        // The load() intrinsic uses current_load_hint or defaults to double
                        self.current_load_hint
                            .clone()
                            .unwrap_or_else(|| "double".into())
                    }
                    "println" | "print" | "print_int" | "File_write" => "void".into(),
                    "String_new" | "File_read_to_string" => "ptr".into(),
                    _ => self
                        .functions
                        .get(&func_name)
                        .map(|(_, r)| r.clone())
                        .unwrap_or_else(|| "i32".into()),
                }
            }
            Expr::GenericCall { func, .. } => {
                let func_name = self.emit_call_target(func);
                self.functions
                    .get(&func_name)
                    .map(|(_, r)| r.clone())
                    .unwrap_or_else(|| "i32".into())
            }
            Expr::BinaryOp { op, left, .. } => match op {
                BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::Le
                | BinaryOp::Ge => "i1".into(),
                _ => self.infer_type(left),
            },
            Expr::MemberAccess { base, member, .. } => {
                let base_ty = if let Expr::UnaryOp { op: UnaryOp::Deref, operand, .. } = &**base {
                    self.infer_struct_type(operand)
                } else {
                    self.infer_struct_type(base)
                };
                let base_name = base_ty.trim_start_matches('%');

                if let Some(&has_data) = self.enums.get(base_name) {
                    if has_data {
                        if member == "tag" {
                            return "i32".into();
                        } else if member == "data" {
                            return "[8 x i64]".into();
                        }
                    }
                }

                if base_ty == "[8 x i64]" {
                    if member.starts_with('_') {
                        return "i64".into(); // The payload elements are i64
                    } else {
                        return "[8 x i64]".into(); // e.g. `.Let` overlays the payload
                    }
                }

                if let Some(fields) = self.structs.get(base_name) {
                    for (fname, fty) in fields {
                        if fname == member {
                            return fty.clone();
                        }
                    }
                }
                "i32".into()
            }
            Expr::ZeroInit(_) => self
                .current_load_hint
                .clone()
                .unwrap_or_else(|| "i32".into()),
            Expr::StructLit { name, .. } => format!("%{}", name),
            // Fix 3: Expr::Path on enum variants returns the enum struct type
            Expr::Path { namespace, .. } => {
                if let Some(&has_data) = self.enums.get(namespace) {
                    if has_data {
                        format!("%{}", namespace)
                    } else {
                        "i32".into() // simple enum = integer tag
                    }
                } else {
                    "i32".into()
                }
            }
            Expr::UnaryOp { op, operand, .. } => match op {
                UnaryOp::Ref => "ptr".into(),
                UnaryOp::Deref => {
                    let inner_ty = self.infer_type(operand);
                    if inner_ty == "ptr" {
                        "i32".into()
                    } else {
                        inner_ty
                    }
                }
                UnaryOp::Neg | UnaryOp::Not => self.infer_type(operand),
            },
            _ => "i32".into(),
        }
    }

    fn get_pointee_type(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Reference { inner, .. } => {
                if let Type::Ident(name, _) = &**inner {
                    if self.structs.contains_key(name.as_str()) {
                        return Some(format!("%{}", name));
                    }
                }
                None
            }
            Type::Ident(name, _) => {
                if self.structs.contains_key(name.as_str()) {
                    return Some(format!("%{}", name));
                }
                None
            }
            _ => None,
        }
    }

    fn infer_struct_type(&self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => {
                if let Some(t) = self.locals_ast_type.get(name) {
                    let cleaned = t.trim_start_matches('&').trim_start_matches("mut ");
                    if self.ast_structs.contains_key(cleaned) {
                        return format!("%{}", cleaned);
                    }
                }
                self.pointee_types
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| {
                        if let Some(t) = self.locals_ast_type.get(name) {
                            let cleaned = t.trim_start_matches('&').trim_start_matches("mut ");
                            if self.ast_structs.contains_key(cleaned) {
                                format!("%{}", cleaned)
                            } else {
                                "i32".into()
                            }
                        } else {
                            "i32".into()
                        }
                    })
            }
            Expr::MemberAccess { base, member, .. } => {
                let base_ty = if let Expr::UnaryOp { op: UnaryOp::Deref, operand, .. } = &**base {
                    self.infer_struct_type(operand)
                } else {
                    self.infer_struct_type(base)
                };
                let base_name = base_ty.trim_start_matches('%');

                if let Some(&has_data) = self.enums.get(base_name) {
                    if has_data {
                        if member == "data" || member.starts_with('_') {
                            return "[8 x i64]".into();
                        }
                        if member == "tag" {
                            return "i32".into();
                        }
                        return base_ty.clone();
                    }
                }

                if base_ty == "[8 x i64]" {
                    if member.starts_with('_') {
                        return "i64".into();
                    } else {
                        return "[8 x i64]".into();
                    }
                }

                if let Some(fields) = self.structs.get(base_name) {
                    for (fname, fty) in fields {
                        if fname == member {
                            if fty.starts_with('%') {
                                return fty.clone();
                            }
                            return "i32".into();
                        }
                    }
                }
                "i32".into()
            }
            Expr::Call { func, .. } => {
                let func_name = self.emit_call_target(func);
                // Enum constructor calls return the enum struct type
                if self.enum_variants.contains_key(&func_name) {
                    let enum_name = func_name.split('_').next().unwrap();
                    return format!("%{}", enum_name);
                }
                self.functions
                    .get(&func_name)
                    .map(|(_, r)| r.clone())
                    .unwrap_or_else(|| "i32".into())
            }
            Expr::UnaryOp {
                op: UnaryOp::Deref,
                operand,
                ..
            } => self.infer_struct_type(operand),
            _ => "i32".into(),
        }
    }
}
