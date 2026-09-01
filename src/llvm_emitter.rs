// ============================================================
//  Y — LLVM IR Backend Emitter
//  llvm_emitter.rs
//
//  Translates Y AST into LLVM IR textual representation.
//  The generated .ll file can be compiled by llc/clang to
//  produce native code for any LLVM-supported target.
//
//  Type mapping:
//    Y         LLVM IR
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

/// A literal's value, for reading `@bounds` at compile time.
fn const_f64_of(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::IntLit(v, _) => Some(*v as f64),
        Expr::FloatLit(v, _) => Some(*v),
        Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => const_f64_of(operand).map(|v| -v),
        _ => None,
    }
}

/// Symbols this module `declare`s in its own prelude, so a call to one needs no
/// extra declaration.
const PRELUDE_DECLARED: &[&str] = &[
            "exit",
            "free",
            "llvm.memset.p0.i64",
            "llvm.prefetch.p0",
            "load",
            "malloc",
            "print_int",
            "printf",
            "println",
            "yfile_read_to_string",
            "yfile_write",
            "ystr_char_at",
            "ystr_clone",
            "ystr_eq_cstr",
            "ystr_len",
            "ystr_new",
            "ystr_push",
            "ystr_push_str",
            "yvec_get",
            "yvec_len",
            "yvec_new",
            "yvec_push",
];

/// Symbols libc provides, which every host link already resolves.
///
/// Kept apart from `RUNTIME_SYMBOLS` because that list is asserted against
/// `c_src/runtime.c`, and a libc name is not defined there. Deliberately
/// short: each entry is a promise that the symbol exists on every host this
/// backend targets, so it is not a place to park a name that merely ought to.
pub const LIBC_SYMBOLS: &[&str] = &[
            "usleep",
];

/// Six names were removed from this list in the same change that added the
/// ShadowPlay surface, for the opposite reason: `init_allocator`,
/// `is_valid_ystr`, `make_enum`, `register_ystr`, `resolve_ystr` and
/// `ystr_hash_fn` are `static` helpers INSIDE the runtime. They emit no
/// symbol, so listing them meant `init_allocator();` in a Y program compiled
/// clean and then died at link with `undefined reference to 'init_allocator'`
/// - which reads as a broken toolchain. They are refused by name now.
///
/// Symbols that linking `c_src/runtime.c` PROVIDES, which the LLVM path links
/// against. That includes what the headers it pulls in define - the ShadowPlay
/// GUI surface comes from `c_src/shadowplay_gui.h`, and leaving it off this
/// list is what made `shadowplay.ysu`, the repo's only end-user application,
/// stop compiling: the backend refused nine of its calls by name.
///
/// A call to one of these needs a `declare` emitted; a call to anything in
/// NEITHER list and not defined in this module does not exist, and is refused.
///
/// These are two different questions and the first version of the refusal
/// conflated them - which suppressed the declaration for `String_new` and
/// turned two valid modules into invalid ones. The clang oracle caught it;
/// review did not.
///
/// `runtime_symbols_match_the_runtime` asserts this list against
/// `c_src/runtime.c` rather than re-deriving it, so the two cannot drift apart
/// silently - the same "assert the producers AGREE" device the `.version` gate
/// uses.
pub const RUNTIME_SYMBOLS: &[&str] = &[
            "Expr_BinaryExpr",
            "Expr_BoolLit",
            "Expr_Call",
            "Expr_CharLit",
            "Expr_FloatLit",
            "Expr_Ident",
            "Expr_Index",
            "Expr_IntLit",
            "Expr_MemberAccess",
            "Expr_Path",
            "Expr_StringLit",
            "Expr_StructLit",
            "Expr_UnaryExpr",
            "MatchPattern_EnumVariant",
            "MatchPattern_Ident",
            "MatchPattern_Literal",
            "Stmt_Assign",
            "Stmt_CompoundAssign",
            "Stmt_ExprStmt",
            "Stmt_For",
            "Stmt_If",
            "Stmt_Let",
            "Stmt_Match",
            "Stmt_Return",
            "Stmt_SafeBlock",
            "Stmt_While",
            "String_new",
            "TokenKind_AtUnknown",
            "TokenKind_CharLit",
            "TokenKind_FloatLit",
            "TokenKind_HardwareTarget",
            "TokenKind_Ident",
            "TokenKind_IntLit",
            "TokenKind_MmaMod",
            "TokenKind_StringLit",
            "TokenKind_Unknown",
            "cleanup_shadowplay_gui",
            "get_broadcast_state",
            "get_codec_state",
            "get_file_format_state",
            "get_indicator_state",
            "get_instant_replay_state",
            "get_microphone_index",
            "get_microphone_name",
            "get_quality_state",
            "get_recording_state",
            "get_replay_duration",
            "get_replay_duration_idx",
            "init_shadowplay_gui",
            "is_overlay_visible",
            "print",
            "print_int",
            "println",
            "str_to_i64",
            "update_shadowplay_gui",
            "ychar_to_ascii",
            "yfile_read_to_string",
            "yfile_write",
            "ymalloc",
            "yrealloc",
            "ystr_char_at",
            "ystr_clone",
            "ystr_eq",
            "ystr_eq_cstr",
            "ystr_free",
            "ystr_len",
            "ystr_new",
            "ystr_push",
            "ystr_push_str",
            "yvec_free",
            "yvec_get",
            "yvec_get_char",
            "yvec_len",
            "yvec_new",
            "yvec_push",
];

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
    /// Element LLVM type behind a `GlobalMemory<T>` / `SharedMemory<T>` binding.
    /// `ast_type_to_string` folds `Generic { base, args }` down to `base`, so the
    /// `T` is not recoverable from `locals_ast_type` — the block-pointer
    /// intrinsics need it to pick a load/store type, and guessing gives the
    /// silent-wrong-answer failure the `_ =>` rule exists to prevent.
    mem_elem_types: BTreeMap<String, String>,
    /// Map function names to their LLVM parameter types and return type
    functions: BTreeMap<String, (Vec<String>, String)>,
    /// Track struct fields: StructName -> Vec<(FieldName, IRType)>
    structs: BTreeMap<String, Vec<(String, String)>>,
    /// Track struct fields AST Types: StructName -> Vec<(FieldName, ASTType)>
    ast_structs: HashMap<String, Vec<(String, String)>>,
    /// Track struct field attributes: StructName -> HashMap<FieldName, Vec<FieldAttrKind>>
    struct_field_attrs: HashMap<String, HashMap<String, Vec<FieldAttrKind>>>,
    /// Track enums: EnumName -> has_data (true = tagged union, false = simple i32 tag)
    enums: BTreeMap<String, bool>,
    /// Track enum variant tags: EnumName_VariantName -> tag integer
    enum_variants: BTreeMap<String, i32>,
    /// Track whether the current block already has a terminator
    block_terminated: bool,
    /// Store current cache policy during let bindings
    current_cache_policy: Option<String>,
    /// Accumulators declared `@ZeroDrift`, with the exact representation chosen
    /// for each. These are stored as integers; conversion happens on write and
    /// on read, and the accumulation between is exact.
    zero_drift: BTreeMap<String, crate::zero_drift::DriftRepr>,
    /// Measured accumulate costs from the device, driving the choice.
    drift_costs: crate::zero_drift::CostTable,
    /// Constructs this backend refuses to emit: `@ZeroDrift` bindings it
    /// cannot honour, and block-pointer intrinsics whose shape or element
    /// type it cannot determine. Emitting something plausible instead is the
    /// silent-wrong-answer failure the repo's design rule forbids.
    pub emit_errors: Vec<String>,
    /// One line per `@ZeroDrift` binding: what was chosen, and on what basis.
    pub drift_report: Vec<String>,
    /// One entry per exact `vpdpwssd` GEMM this compilation SUBSTITUTED.
    ///
    /// The certificate is only meaningful where a kernel was actually swapped
    /// in: a nest left on the scalar exact path is already the naive nest, so
    /// there is nothing to certify equal to it. `main.rs` renders and writes
    /// these; the emitter does no file I/O.
    pub exact_gemm_certificates: Vec<crate::exact_gemm_certificate::Certificate>,
    /// Hint for the load() intrinsic: the declared LHS type of the current let
    current_load_hint: Option<String>,
    /// Track all function names called during emission
    called_functions: Vec<String>,
    /// Track all function names defined in this module
    defined_functions: Vec<String>,
    /// Whether we are currently inside a @ptx_emit function
    in_ptx_emit: bool,
    /// Stack of labels to jump to for break statements
    loop_exit_stack: Vec<String>,
    /// Set when a kernel was replaced by the packed AVX-512 GEMM, so the
    /// supporting module (packing routines, micro-kernel, driver) is emitted.
    needs_gemm_module: bool,
    /// The flush interval of the exact VNNI GEMM, when one was substituted.
    needs_exact_gemm_module: Option<u32>,
}

/// Entry-block stack slot that masked-off block-pointer stores are redirected
/// into. Declared in every kernel and function; it is a dead alloca whose
/// address never escapes, so it costs nothing when unused.
const Y_OOB_SINK: &str = "%.y_oob_sink";

/// LLVM element type behind a `GlobalMemory<T>` / `SharedMemory<T>` parameter.
/// Returns `None` for anything else, so callers can refuse rather than guess.
fn memory_element_llvm_type(ty: &Type) -> Option<String> {
    let Type::Generic { base, args, .. } = ty else {
        return None;
    };
    if base != "GlobalMemory" && base != "SharedMemory" {
        return None;
    }
    let GenericArg::Type(inner) = args.first()? else {
        return None;
    };
    let name = match inner {
        Type::Primitive(n, _) | Type::Ident(n, _) => n.as_str(),
        _ => return None,
    };
    match name {
        "F16" | "f16" | "half" => Some("half".into()),
        "F32" | "f32" | "float" => Some("float".into()),
        "F64" | "f64" | "double" => Some("double".into()),
        "I8" | "i8" | "u8" => Some("i8".into()),
        "I16" | "i16" | "u16" => Some("i16".into()),
        "I32" | "i32" | "u32" => Some("i32".into()),
        "I64" | "i64" | "u64" | "usize" => Some("i64".into()),
        _ => None,
    }
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
        Type::BlockTile { element, .. } => {
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
        functions.insert("sqrtf".into(), (vec!["F32".to_string()], "float".into()));
        functions.insert("math_sqrt".into(), (vec!["F32".to_string()], "float".into()));
        functions.insert("math_fmin".into(), (vec!["F32".to_string(), "F32".to_string()], "float".into()));
        functions.insert("math_fmax".into(), (vec!["F32".to_string(), "F32".to_string()], "float".into()));

        // --- Standard Library namespaced methods ---
        functions.insert("Vec_new".into(), (vec!["I32".to_string()], "ptr".into()));
        functions.insert("Vec_push".into(), (vec!["&mut Vec".to_string(), "&char".to_string()], "void".into()));
        functions.insert("Vec_free".into(), (vec!["&mut Vec".to_string()], "void".into()));
        functions.insert("Vec_len".into(), (vec!["&Vec".to_string()], "i64".into()));
        functions.insert("Vec_get_char".into(), (vec!["&Vec".to_string(), "usize".to_string()], "i8".into()));

        functions.insert("String_len".into(), (vec!["&String".to_string()], "i64".into()));
        functions.insert("String_clone".into(), (vec!["&String".to_string()], "ptr".into()));
        functions.insert("String_push".into(), (vec!["&mut String".to_string(), "char".to_string()], "void".into()));
        functions.insert("String_push_str".into(), (vec!["&mut String".to_string(), "&String".to_string()], "void".into()));
        functions.insert("String_eq".into(), (vec!["&String".to_string(), "&String".to_string()], "i1".into()));
        functions.insert("String_eq_cstr".into(), (vec!["&String".to_string(), "&char".to_string()], "i1".into()));
        functions.insert("String_char_at".into(), (vec!["&String".to_string(), "usize".to_string()], "i8".into()));
        functions.insert("String_free".into(), (vec!["&mut String".to_string()], "void".into()));

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
            mem_elem_types: BTreeMap::new(),
            functions,
            structs: BTreeMap::new(),
            ast_structs: HashMap::new(),
            struct_field_attrs: HashMap::new(),
            enums: BTreeMap::new(),
            enum_variants: BTreeMap::new(),
            block_terminated: false,
            current_cache_policy: None,
            zero_drift: BTreeMap::new(),
            drift_costs: crate::zero_drift::CostTable::new(),
            emit_errors: Vec::new(),
            drift_report: Vec::new(),
            exact_gemm_certificates: Vec::new(),
            current_load_hint: None,
            called_functions: Vec::new(),
            defined_functions: Vec::new(),
            in_ptx_emit: false,
            loop_exit_stack: Vec::new(),
            needs_gemm_module: false,
            needs_exact_gemm_module: None,
        }
    }

    fn fresh_tmp(&mut self) -> String {
        self.tmp_counter += 1;
        format!("%_t{}", self.tmp_counter)
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        format!("{}.{}", prefix, self.label_counter)
    }

    fn get_expr_attrs(&self, expr: &Expr) -> Option<Vec<FieldAttrKind>> {
        match expr {
            Expr::MemberAccess { base, member, .. } => {
                let base_ty = self.infer_struct_type(base);
                let base_name = base_ty.trim_start_matches('%');
                if let Some(field_map) = self.struct_field_attrs.get(base_name) {
                    return field_map.get(member).cloned();
                }
            }
            Expr::Index { base, .. } => {
                return self.get_expr_attrs(base);
            }
            _ => {}
        }
        None
    }

    fn emit_load(&mut self, ptr: &str, ty: &str) -> String {
        self.emit_load_with_attrs(ptr, ty, None)
    }

    fn emit_load_with_attrs(&mut self, ptr: &str, ty: &str, attrs: Option<Vec<FieldAttrKind>>) -> String {
        let tmp = self.fresh_tmp();
        let mut is_atomic = false;
        let mut atomic_ordering = "seq_cst".to_string();
        let mut is_volatile = false;
        let mut align_val = None;

        if let Some(attrs_list) = attrs {
            for attr in attrs_list {
                match attr {
                    FieldAttrKind::Atomic(ref ord) => {
                        is_atomic = true;
                        if let Some(ref o) = ord {
                            let mapped = match o.as_str() {
                                "relaxed" => "monotonic",
                                "release" => "monotonic", // load cannot release
                                "acq_rel" => "acquire",   // load cannot release
                                other => other,
                            };
                            atomic_ordering = mapped.to_string();
                        }
                    }
                    FieldAttrKind::GpuUncached => is_volatile = true,
                    FieldAttrKind::Align(expr) => {
                        if let Expr::IntLit(val, _) = expr {
                            align_val = Some(val);
                        }
                    }
                }
            }
        }

        let align_str = if let Some(a) = align_val {
            format!(", align {}", a)
        } else if is_atomic {
            let default_align = match ty {
                "i8" | "i1" => "1",
                "i16" => "2",
                "i32" | "float" => "4",
                _ => "8",
            };
            format!(", align {}", default_align)
        } else {
            "".to_string()
        };

        if is_atomic {
            writeln!(
                &mut self.output,
                "  {} = load atomic {}, ptr {}{} {}{}",
                tmp, ty, ptr, if is_volatile { " volatile" } else { "" }, atomic_ordering, align_str
            )
            .unwrap();
        } else {
            let volatile_str = if is_volatile { " volatile" } else { "" };
            let nontemporal_str = if is_volatile { ", !nontemporal !0" } else { "" };
            writeln!(
                &mut self.output,
                "  {} = load{} {}, ptr {}{}{}",
                tmp, volatile_str, ty, ptr, align_str, nontemporal_str
            )
            .unwrap();
        }
        tmp
    }

    fn emit_store(&mut self, val: &str, ptr: &str, ty: &str) {
        self.emit_store_with_attrs(val, ptr, ty, None)
    }

    fn emit_store_with_attrs(&mut self, val: &str, ptr: &str, ty: &str, attrs: Option<Vec<FieldAttrKind>>) {
        let mut is_atomic = false;
        let mut atomic_ordering = "seq_cst".to_string();
        let mut is_volatile = false;
        let mut align_val = None;

        if let Some(attrs_list) = attrs {
            for attr in attrs_list {
                match attr {
                    FieldAttrKind::Atomic(ref ord) => {
                        is_atomic = true;
                        if let Some(ref o) = ord {
                            let mapped = match o.as_str() {
                                "relaxed" => "monotonic",
                                "acquire" => "monotonic", // store cannot acquire
                                "acq_rel" => "release",   // store cannot acquire
                                other => other,
                            };
                            atomic_ordering = mapped.to_string();
                        }
                    }
                    FieldAttrKind::GpuUncached => is_volatile = true,
                    FieldAttrKind::Align(expr) => {
                        if let Expr::IntLit(val, _) = expr {
                            align_val = Some(val);
                        }
                    }
                }
            }
        }

        let align_str = if let Some(a) = align_val {
            format!(", align {}", a)
        } else if is_atomic {
            let default_align = match ty {
                "i8" | "i1" => "1",
                "i16" => "2",
                "i32" | "float" => "4",
                _ => "8",
            };
            format!(", align {}", default_align)
        } else {
            "".to_string()
        };

        if is_atomic {
            writeln!(
                &mut self.output,
                "  store atomic {} {}, ptr {}{} {}{}",
                ty, val, ptr, if is_volatile { " volatile" } else { "" }, atomic_ordering, align_str
            )
            .unwrap();
        } else {
            let volatile_str = if is_volatile { " volatile" } else { "" };
            let nontemporal_str = if is_volatile { ", !nontemporal !0" } else { "" };
            writeln!(
                &mut self.output,
                "  store{} {} {}, ptr {}{}{}",
                volatile_str, ty, val, ptr, align_str, nontemporal_str
            )
            .unwrap();
        }
    }

    /// Insert an LLVM conversion instruction when src_ty != dst_ty.
    /// Returns the new SSA name holding the converted value, or the
    /// original `val` if no conversion is needed.
    /// Supplies measured accumulate costs so `@ZeroDrift` chooses on evidence.
    /// Without this the selector falls back to narrowest-sufficient, which is
    /// deterministic but not informed.
    pub fn set_drift_costs(&mut self, costs: crate::zero_drift::CostTable) {
        self.drift_costs = costs;
    }

    /// Converts a `double` into `repr`'s integer domain.
    ///
    /// Rounds half away from zero rather than truncating. Truncation is also
    /// deterministic - so the accumulation would still be reorder-invariant -
    /// but it biases every term toward zero, and a long reduction turns that
    /// bias into a visible systematic error. The rounding is done with
    /// `fcmp`/`select` rather than `llvm.round` so no intrinsic declaration is
    /// needed.
    /// `acc = acc + rhs` / `acc = acc - rhs`, the running-sum form, or `None`.
    ///
    /// Only the accumulator on the LEFT of the `+` is accepted. `acc = rhs +
    /// acc` is the same value for addition and NOT for subtraction, and
    /// accepting one shape and silently treating it as the other is how the
    /// sign gets lost; the commuted form simply is not matched.
    /// Delegates to `zero_drift::running_sum`, which both backends share.
    ///
    /// This used to be the rule's only copy. The PTX backend had no equivalent
    /// at all, so `acc = acc + e` was an f32 accumulation there long after it
    /// was fixed here - `feedback-gotchas-apply-to-every-backend`, found again.
    fn drift_running_sum<'e>(
        target: &Expr,
        value: &'e Expr,
    ) -> Option<(BinaryOp, &'e Expr)> {
        crate::zero_drift::running_sum(target, value)
    }

    fn emit_to_fixed(&mut self, val: &str, repr: crate::zero_drift::DriftRepr) -> String {
        let ity = repr.llvm_type();
        if repr.frac_bits() == 0 {
            let out = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = fptosi double {} to {}", out, val, ity).unwrap();
            return out;
        }
        let scaled = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = fmul double {}, {:.1}",
            scaled,
            val,
            repr.scale()
        )
        .unwrap();
        let is_neg = self.fresh_tmp();
        writeln!(&mut self.output, "  {} = fcmp olt double {}, 0.0", is_neg, scaled).unwrap();
        let bias = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = select i1 {}, double -5.000000e-01, double 5.000000e-01",
            bias, is_neg
        )
        .unwrap();
        let rounded = self.fresh_tmp();
        writeln!(&mut self.output, "  {} = fadd double {}, {}", rounded, scaled, bias).unwrap();
        let out = self.fresh_tmp();
        writeln!(&mut self.output, "  {} = fptosi double {} to {}", out, rounded, ity).unwrap();
        out
    }

    /// Converts a value out of `repr`'s integer domain back to `double`.
    fn emit_from_fixed(&mut self, val: &str, repr: crate::zero_drift::DriftRepr) -> String {
        let ity = repr.llvm_type();
        let as_f = self.fresh_tmp();
        writeln!(&mut self.output, "  {} = sitofp {} {} to double", as_f, ity, val).unwrap();
        if repr.frac_bits() == 0 {
            return as_f;
        }
        let out = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = fdiv double {}, {:.1}",
            out,
            as_f,
            repr.scale()
        )
        .unwrap();
        out
    }

    /// Emits an expression and coerces the result to `double`, which is the
    /// domain every `@ZeroDrift` conversion works in.
    fn emit_expr_as_double(&mut self, expr: &Expr) -> String {
        let v = self.emit_expr(expr, None, None);
        let t = self.infer_type(expr);
        self.emit_coerce(&v, &t, "double")
    }

    /// Is this expression's Y-level type unsigned?
    ///
    /// LLVM has no unsigned integer types -- `U32` and `I32` are both `i32` --
    /// so `emit_type` erases the one bit that decides between `zext` and
    /// `sext`. `locals_ast_type` still has it for a binding, which is enough
    /// for the shapes that matter: a named value, a unary or binary
    /// expression over named values, and a parenthesised form of either.
    ///
    /// Defaults to `false` (signed), which is the previous behaviour, so an
    /// expression this cannot classify is no worse off than before.
    fn expr_is_unsigned(&self, e: &Expr) -> bool {
        match e {
            Expr::Ident(name, _) => self
                .locals_ast_type
                .get(name)
                .is_some_and(|t| matches!(t.as_str(), "U8" | "U16" | "U32" | "U64" | "usize")),
            // If either side is unsigned the value is being computed in
            // unsigned terms; widening it must not invent a sign bit.
            Expr::BinaryOp { left, right, .. } => {
                self.expr_is_unsigned(left) || self.expr_is_unsigned(right)
            }
            Expr::UnaryOp { operand, .. } => self.expr_is_unsigned(operand),
            _ => false,
        }
    }

    fn emit_coerce(&mut self, val: &str, src_ty: &str, dst_ty: &str) -> String {
        self.emit_coerce_from(val, src_ty, dst_ty, false)
    }

    fn emit_coerce_from(
        &mut self,
        val: &str,
        src_ty: &str,
        dst_ty: &str,
        src_unsigned: bool,
    ) -> String {
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
                // **`i1` must ZERO-extend.** A boolean is 0 or 1; sign-extending
                // it makes `true` into -1, and the comparison operators are the
                // only producers of `i1` in this backend. So
                //
                //     let t: I32 = a > b;   // 5 > 3
                //
                // evaluated to **-1**, and `t * 5` to -5. It is invisible in a
                // condition (`if t` tests non-zero either way) and wrong
                // wherever a comparison is used as a VALUE. Found by
                // `tests/backend_differential.rs` on its first run: the native
                // backend answers 1, and so do the ZK backend (whose condition
                // carries a booleanity constraint) and `cpu_emitter` (which
                // emits a Rust `bool`), so LLVM was the only one disagreeing.
                // `i1` is a boolean and is ALWAYS zero-extended -- see the
                // comparison bug below. An unsigned Y type is zero-extended
                // too: `let x: U32 = 3000000000; let y: U64 = x;` used to
                // emit `sext i32 to i64` and produce 0xFFFFFFFF_B2D05E00.
                // That is gotcha #7's bug, which CLAUDE.md documents as fixed
                // in the PTX backend and which was still live here -- the
                // third backend to have it.
                let how = if src_ty == "i1" || src_unsigned {
                    "zext"
                } else {
                    "sext"
                };
                writeln!(
                    &mut self.output,
                    "  {} = {} {} {} to {}",
                    tmp, how, src_ty, val, dst_ty
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

    fn emit_type(&mut self, ty: &Type) -> String {
        let res: String = match ty {
            Type::Primitive(name, _) => match name.as_str() {
                // `U64`/`u64` were absent and fell to the `_ => "i32"` arm
                // below, so `let x: U64 = ...` allocated an **i32**. The PTX
                // backend takes these types seriously (gotcha #7); this one
                // silently halved their width.
                "I32" | "U32" | "u32" | "i32" => "i32".into(),
                "I64" | "U64" | "u64" | "usize" | "isize" | "i64" => "i64".into(),
                "U8" | "I8" => "i8".into(),
                "U16" => "i16".into(),
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
                "I32" | "U32" | "u32" | "i32" => "i32".into(),
                "I64" | "U64" | "u64" | "usize" | "isize" | "i64" => "i64".into(),
                "U8" | "I8" => "i8".into(),
                "U16" => "i16".into(),
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
                    } else if self.structs.contains_key(other) {
                        format!("%{}", other)
                    } else {
                        // Not a primitive, not a registered struct, not an
                        // enum: there is nothing to lower this to. Emitting
                        // `%Name` names an LLVM struct type the module never
                        // defines, and `alloca %Name` on an undefined type is
                        // "Cannot allocate unsized type" - INVALID IR, written
                        // out under "Compilation Successful!" and exit 0.
                        //
                        // Every instance in the corpus was `U32x4`, the PTX
                        // backend's 16-byte vector type, reaching the host
                        // backend: 11 of the 76 programs this backend accepted
                        // emitted a module clang refuses.
                        self.emit_errors.push(format!(
                            "[LLVM host backend] type `{}` has no host lowering - it \
                             would name an LLVM struct this module never defines. \
                             GPU-only types such as `U32x4` belong to --emit-ptx.",
                            other
                        ));
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
            Type::BlockTile { .. } => "ptr".into(),
        };
        if res == "%ptr" {
            "ptr".into()
        } else {
            res
        }
    }

    fn emit_field_type(&mut self, ty: &Type) -> String {
        match ty {
            Type::Array { element, size, .. } => {
                let elem_llvm_ty = self.emit_type(element);
                if let Expr::IntLit(val, _) = &**size {
                    format!("[{} x {}]", val, elem_llvm_ty)
                } else {
                    "ptr".into()
                }
            }
            _ => self.emit_type(ty),
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

        // --- Standard Library namespaced methods ---
        self.functions.insert("Vec_new".into(), (vec!["I32".to_string()], "ptr".into()));
        self.functions.insert("Vec_push".into(), (vec!["&mut Vec".to_string(), "&char".to_string()], "void".into()));
        self.functions.insert("Vec_free".into(), (vec!["&mut Vec".to_string()], "void".into()));
        self.functions.insert("Vec_len".into(), (vec!["&Vec".to_string()], "i64".into()));
        self.functions.insert("Vec_get_char".into(), (vec!["&Vec".to_string(), "usize".to_string()], "i8".into()));

        self.functions.insert("String_len".into(), (vec!["&String".to_string()], "i64".into()));
        self.functions.insert("String_clone".into(), (vec!["&String".to_string()], "ptr".into()));
        self.functions.insert("String_push".into(), (vec!["&mut String".to_string(), "char".to_string()], "void".into()));
        self.functions.insert("String_push_str".into(), (vec!["&mut String".to_string(), "&String".to_string()], "void".into()));
        self.functions.insert("String_eq".into(), (vec!["&String".to_string(), "&String".to_string()], "i1".into()));
        self.functions.insert("String_eq_cstr".into(), (vec!["&String".to_string(), "&char".to_string()], "i1".into()));
        self.functions.insert("String_char_at".into(), (vec!["&String".to_string(), "usize".to_string()], "i8".into()));
        self.functions.insert("String_free".into(), (vec!["&mut String".to_string()], "void".into()));

        self.functions.insert("File_read_to_string".into(), (vec!["&String".to_string()], "ptr".into()));
        self.functions.insert("File_write".into(), (vec!["&String".to_string(), "&String".to_string()], "void".into()));

        // Phase 0a: register every struct and enum FIRST.
        //
        // This used to be one loop that registered structs and resolved
        // function signatures together, so `emit_type` was asked about a
        // struct declared later in the file and the struct table did not have
        // it yet. That was harmless while the fallback silently emitted
        // `%Name` anyway; the moment that fallback became a refusal, four
        // corpus programs with a perfectly ordinary `struct` in them were
        // refused. A name resolver must see all the names before it answers
        // any question.
        for item in &prog.items {
            match item {
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
        for item in &prog.items {
            match item {
                Item::Struct(s) => {
                    let mut fields = Vec::new();
                    let mut ast_fields = Vec::new();
                    let mut field_attrs = HashMap::new();
                    for f in &s.fields {
                        fields.push((f.name.clone(), self.emit_field_type(&f.ty)));
                        ast_fields.push((f.name.clone(), ast_type_to_string(&f.ty)));
                        let attrs: Vec<FieldAttrKind> = f.attrs.iter().map(|attr| attr.kind.clone()).collect();
                        field_attrs.insert(f.name.clone(), attrs);
                    }
                    self.structs.insert(s.name.clone(), fields);
                    self.ast_structs.insert(s.name.clone(), ast_fields);
                    self.struct_field_attrs.insert(s.name.clone(), field_attrs);
                }
                _ => {}
            }
        }

        // Phase 0b: resolve function signatures, with every type name known.
        for item in &prog.items {
            match item {
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
                    field_tys.push(self.emit_field_type(&f.ty));
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
        self.wln("@.str.bounds_err = private unnamed_addr constant [54 x i8] c\"Index out of bounds panic: index %ld, array size %ld\\0A\\00\"");
        self.wln("");

        // Append function bodies
        self.output.push_str(&func_output);

        // Auto-declare any called functions that are not defined or already declared
        let prelude_set: std::collections::HashSet<&str> =
            PRELUDE_DECLARED.iter().copied().collect();
        let runtime_set: std::collections::HashSet<&str> = RUNTIME_SYMBOLS
            .iter()
            .chain(LIBC_SYMBOLS.iter())
            .copied()
            .collect();

        let defined_set: std::collections::HashSet<String> =
            self.defined_functions.iter().cloned().collect();
        let mut auto_declared: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut extern_decls = String::new();

        for fname in &self.called_functions {
            if !prelude_set.contains(fname.as_str())
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

                if runtime_set.contains(fname.as_str()) {
                    if ret_ty.starts_with('%') {
                        writeln!(&mut extern_decls, "declare void @{}(...)", fname).unwrap();
                    } else {
                        writeln!(&mut extern_decls, "declare {} @{}(...)", ret_ty, fname)
                            .unwrap();
                    }
                } else {
                    // Neither declared above, nor defined here, nor present in
                    // the runtime: this symbol does not exist. Declaring it
                    // anyway produced a module that ASSEMBLES and then fails at
                    // link with `undefined reference to 'thread_idx_x'` - which
                    // reads as a broken toolchain rather than as a program
                    // using a construct this backend cannot lower. Every such
                    // name in the corpus was a GPU intrinsic: thread_idx_x,
                    // block_idx_x/y/z, the carry-chain intrinsics, the v4
                    // vector loads, mma_sync, ldmatrix, bvh_traverse.
                    //
                    // This is the exact check `cpu_emitter` already made
                    // (`a_gpu_intrinsic_is_refused_rather_than_transcribed`);
                    // the LLVM backend never got it.
                    self.emit_errors.push(format!(
                        "[LLVM host backend] `{}(...)` has no host lowering - it would be \
                         declared as an external symbol that does not exist, and the link \
                         would fail. This backend targets host code; GPU intrinsics belong \
                         to --emit-ptx.",
                        fname
                    ));
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

        if self.needs_gemm_module {
            self.wln("");
            let m = crate::cpu_gemm::emit_kernel_module();
            self.output.push_str(&m);
        }

        if let Some(flush) = self.needs_exact_gemm_module {
            self.wln("");
            let m = crate::cpu_gemm::emit_vnni_gemm_module(flush);
            self.output.push_str(&m);
            // The f32 module declares the same libc entry points, and a
            // duplicate `declare` is an INVALID REDEFINITION in LLVM rather
            // than a duplicate that gets merged - so they are emitted here only
            // when that module is absent.
            let t = crate::cpu_gemm::emit_vnni_threaded_module(!self.needs_gemm_module);
            self.output.push_str(&t);
        }

        // Nontemporal metadata definition
        self.wln("!0 = !{i32 1}");

        self.output.clone()
    }

    /// `(triple, datalayout mangling spec)` for the machine Y is running on.
    /// Y's LLVM backend compiles for the host, so the host is the target.
    fn host_triple() -> (&'static str, &'static str) {
        if cfg!(target_os = "windows") {
            ("x86_64-pc-windows-msvc", "m:w")
        } else if cfg!(target_os = "macos") {
            ("x86_64-apple-darwin", "m:o")
        } else {
            ("x86_64-unknown-linux-gnu", "m:e")
        }
    }

    /// `(target-cpu, target-features)` for the host.
    ///
    /// AVX-512 used to mean `skylake-avx512` unconditionally. On an AMD Zen 4/5
    /// that is a correct but pessimistic model — wrong port counts, wrong
    /// latencies, and it hides `avx512_bf16` / `avx512vnni`, which those parts
    /// have and Skylake-X does not. The vendor comes from CPUID, so this stays
    /// a probe rather than an assumption.
    fn host_cpu_attrs(profile: &crate::sentinel::HardwareProfile) -> (String, String) {
        if !profile.has_avx512 {
            return if profile.has_avx {
                ("haswell".into(), "+avx2,+avx,+fma".into())
            } else {
                ("x86-64".into(), String::new())
            };
        }
        let base = "+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl,+fma";
        match crate::sentinel::host_x86_uarch() {
            Some(uarch) => (uarch, format!("{},+avx512vnni,+avx512bf16", base)),
            None => ("skylake-avx512".into(), base.into()),
        }
    }

    fn emit_prelude(&mut self, profile: &crate::sentinel::HardwareProfile) {
        self.wln("; ================================================");
        self.wln(";  Generated by Y Compiler — LLVM IR Backend");
        self.wln(&format!(
            ";  Hardware Profile: AVX={}, AVX512={}, L2 Line={}B",
            profile.has_avx, profile.has_avx512, profile.l2_line_size
        ));
        self.wln("; ================================================");
        self.wln("");
        // The triple and datalayout used to be hardcoded to Windows/MSVC
        // (`m:w` mangling) regardless of host, so every Linux and macOS build
        // handed clang a module describing a platform it was not compiling for.
        let (triple, mangling) = Self::host_triple();
        self.wln(&format!(
            "target datalayout = \"e-{}-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"",
            mangling
        ));
        self.wln(&format!("target triple = \"{}\"", triple));
        self.wln("");

        // Dynamically inject LLVM function attributes based on Sentinel Probe
        let (cpu, features) = Self::host_cpu_attrs(profile);
        if features.is_empty() {
            self.wln(&format!("attributes #0 = {{ \"target-cpu\"=\"{}\" }}", cpu));
        } else {
            self.wln(&format!(
                "attributes #0 = {{ \"target-cpu\"=\"{}\" \"target-features\"=\"{}\" }}",
                cpu, features
            ));
        }
        self.wln("");
    }

    // ── Functions ───────────────────────────────────────────

    fn emit_func(&mut self, f: &FuncDecl) {
        self.tmp_counter = 0;
        self.locals.clear();
        self.block_terminated = false;
        let prev_ptx = self.in_ptx_emit;
        self.in_ptx_emit = f.is_ptx_emit;

        let ret_type = match &f.ret_ty {
            Some(ty) => self.emit_type(ty),
            None => "void".into(),
        };

        let func_name = if let Some(ref target) = self.current_impl_target {
            format!("{}_{}", target, f.name)
        } else if f.name == "main" {
            "ysu_main".to_string()
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
            if let Some(ety) = memory_element_llvm_type(&p.ty) {
                self.mem_elem_types.insert(p.name.clone(), ety);
            }
            writeln!(&mut self.output, "  %{} = alloca {}", p.name, ty).unwrap();
            self.emit_store(&format!("%{}.arg", p.name), &format!("%{}", p.name), &ty);
        }

        writeln!(&mut self.output, "  {} = alloca [8 x i8], align 8", Y_OOB_SINK).unwrap();

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
        self.in_ptx_emit = prev_ptx;
    }

    fn emit_alloca_for_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            match stmt {
                // `@ZeroDrift` accumulators live in an integer register, not a
                // float one - that is the entire mechanism. The representation
                // is chosen here, before the alloca, because the alloca's type
                // is what everything downstream keys off.
                Stmt::Let { name, ty, zero_drift: Some(_), bounds, span, .. }
                    if !self.locals.contains_key(name) =>
                {
                    let ty_name = match ty {
                        Some(Type::Primitive(n, _)) | Some(Type::Ident(n, _)) => n.clone(),
                        _ => "F32".to_string(),
                    };
                    let range = bounds.as_ref().and_then(|b| {
                        match (const_f64_of(&b.min), const_f64_of(&b.max)) {
                            (Some(lo), Some(hi)) => Some((lo, hi)),
                            _ => None,
                        }
                    });
                    let req = crate::zero_drift::Requirement::for_type_with_bounds(&ty_name, range);
                    match crate::zero_drift::select_repr(&req, &self.drift_costs) {
                        Ok(decision) => {
                            self.drift_report.push(crate::zero_drift::report_line(
                                name,
                                &ty_name,
                                &decision,
                                crate::zero_drift::explain_requested(),
                            ));
                            self.locals.insert(name.clone(), decision.repr.llvm_type().to_string());
                            self.locals_ast_type.insert(name.clone(), ty_name.clone());
                            self.zero_drift.insert(name.clone(), decision.repr);
                            writeln!(
                                &mut self.output,
                                "  %{} = alloca {}",
                                name,
                                decision.repr.llvm_type()
                            )
                            .unwrap();
                        }
                        Err(why) => {
                            self.emit_errors.push(format!(
                                "Line {}: @ZeroDrift on `{}: {}` cannot be honoured. No exact \
representation holds that range at that resolution, and only exact (integer or fixed-point) \
accumulation is drift-free - f64 is the same non-associative arithmetic with more mantissa. \
Add @bounds(min, max) to state the accumulator's real range, or declare it as a Q format.\n{}",
                                span.line,
                                name,
                                ty_name,
                                crate::zero_drift::explain_rejections(&why)
                            ));
                            // Fall back to the declared type so the rest of the
                            // function still emits; the error above fails the build.
                            let ir_ty = match ty {
                                Some(t) => self.emit_type(t),
                                None => "double".into(),
                            };
                            self.locals.insert(name.clone(), ir_ty.clone());
                            writeln!(&mut self.output, "  %{} = alloca {}", name, ir_ty).unwrap();
                        }
                    }
                }
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
                Stmt::GhostBlock(b, _) => {
                    self.emit_alloca_for_block(b);
                }
                Stmt::HintBlock { body, .. } => {
                    self.emit_alloca_for_block(body);
                }
                Stmt::ClockDomainBlock { body, .. } => {
                    self.emit_alloca_for_block(body);
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
            if let Some(ety) = memory_element_llvm_type(&p.ty) {
                self.mem_elem_types.insert(p.name.clone(), ety);
            }
            writeln!(&mut self.output, "  %{} = alloca {}", p.name, ty).unwrap();
            self.emit_store(&format!("%{}.arg", p.name), &format!("%{}", p.name), &ty);
        }

        writeln!(&mut self.output, "  {} = alloca [8 x i8], align 8", Y_OOB_SINK).unwrap();

        // A kernel whose whole body is the canonical matmul nest is replaced by
        // the packed AVX-512 kernel. The recogniser is strict and the scalar
        // lowering below is correct, so a near-miss costs speed, not an answer.
        if let Some(shape) = self.try_emit_gemm_kernel(k) {
            self.needs_gemm_module = true;
            writeln!(&mut self.output, "  ; [Y CPU GEMM] {:?}", shape).unwrap();
            self.wln("  ret void");
            self.wln("}");
            self.wln("");
            return;
        }

        self.emit_alloca_for_block(&k.body);

        self.emit_block_body(&k.body, "void");
        if !self.block_terminated {
            self.wln("  ret void");
        }
        self.wln("}");
        self.wln("");
    }

    /// Emit a call to the exact `vpdpwssd` GEMM for a recognised, licensed nest.
    ///
    /// The kernel's contract, from `emit_vnni_gemm_module`:
    ///
    ///   `__y_gemm_exact_vnni(A: i16*, B: i16*, C: i64*, M, N, K,
    ///                        lda, ldb, ldc, Ap: i16*, Bp: i16*, Ct: i64*)`
    ///
    /// Two parts of it are easy to get wrong and are handled explicitly here.
    ///
    /// **`C` is accumulated INTO, not overwritten.** That is deliberate - it is
    /// what lets a caller split the K range across threads and sum the pieces,
    /// which is the order-independence the exact path exists to sell. But the
    /// nest being replaced STORES its sum, so `C` has to be zeroed first or a
    /// second call over the same buffer would double it.
    ///
    /// **The three scratch buffers are the caller's.** Sizes come from the
    /// packers' layouts: `Ap` is one `i16` per (row-tile, k-pair, MR, 2),
    /// `Bp` one per (k-pair, NR, 2), and `Ct` is a single `MR x NR` `i64`
    /// micro-tile. They are heap-allocated rather than `alloca`d because `Ap`
    /// grows with M*K and a dynamic `alloca` of that size is a stack overflow
    /// on any real shape.
    fn emit_exact_gemm_call(
        &mut self,
        shape: &crate::cpu_gemm::GemmShape,
        flush_k_pairs: u32,
    ) -> Option<()> {
        use crate::cpu_gemm::{VNNI_MR, VNNI_NR};

        // Extents and strides, widened to i64 exactly as the f32 path does.
        let mut ext = Vec::new();
        for name in [
            &shape.m,
            &shape.n,
            &shape.k,
            &shape.lda,
            &shape.ldb,
            &shape.ldc,
        ] {
            let ty = self.locals.get(name)?.clone();
            let tmp = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = load {}, ptr %{}", tmp, ty, name).unwrap();
            ext.push(if ty == "i64" {
                tmp
            } else {
                let w = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = sext {} {} to i64", w, ty, tmp).unwrap();
                w
            });
        }

        let mut ptrs = Vec::new();
        for name in [&shape.a, &shape.b, &shape.c] {
            let tmp = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = load ptr, ptr %{}", tmp, name).unwrap();
            ptrs.push(tmp);
        }
        let (m, n, k) = (ext[0].clone(), ext[1].clone(), ext[2].clone());

        let mut bin = |op: &str, a: &str, b: &str, out: &mut String| {
            let t = format!("%_t{}", {
                self.tmp_counter += 1;
                self.tmp_counter
            });
            writeln!(out, "  {} = {} i64 {}, {}", t, op, a, b).unwrap();
            t
        };
        let mut ir = String::new();

        // kpairs = (K + 1) / 2 - the packers count k-PAIRS, and an odd K leaves
        // the final high half zero.
        let k1 = bin("add", &k, "1", &mut ir);
        let kpairs = bin("sdiv", &k1, "2", &mut ir);

        // Ap: ceil(M / MR) row tiles, each kpairs * MR * 2 i16.
        let m1 = bin("add", &m, &(VNNI_MR - 1).to_string(), &mut ir);
        let mtiles = bin("sdiv", &m1, &VNNI_MR.to_string(), &mut ir);
        let ap_e = bin("mul", &mtiles, &kpairs, &mut ir);
        let ap_e = bin("mul", &ap_e, &(VNNI_MR * 2).to_string(), &mut ir);
        let ap_b = bin("mul", &ap_e, "2", &mut ir);

        // Bp: kpairs * NR * 2 i16.
        let bp_e = bin("mul", &kpairs, &(VNNI_NR * 2).to_string(), &mut ir);
        let bp_b = bin("mul", &bp_e, "2", &mut ir);

        // C: M * N i64, zeroed because the kernel accumulates into it.
        let c_e = bin("mul", &m, &n, &mut ir);
        let c_b = bin("mul", &c_e, "8", &mut ir);
        self.output.push_str(&ir);

        // The threaded entry owns the scratch, the K-split and the zeroing of
        // `C`, because all three depend on the thread count it chooses. It
        // falls back to a single direct call when one thread is enough, so
        // there is no separate serial path to keep in step.
        let _ = (&ap_b, &bp_b, &c_b);
        writeln!(
            &mut self.output,
            "  call void @{}(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
             i64 {}, i64 {}, i64 {})",
            crate::cpu_gemm::VNNI_THREADED_NAME,
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ext[0],
            ext[1],
            ext[2],
            ext[3],
            ext[4],
            ext[5]
        )
        .unwrap();

        self.needs_exact_gemm_module = Some(flush_k_pairs);
        Some(())
    }

    /// If `k`'s body is the canonical `C = A * B` nest over `F32` buffers,
    /// emit a call to the packed AVX-512 kernel and report the shape.
    ///
    /// The element-type check is not a formality: the recogniser matches on
    /// loop structure, which is identical for `F64` or `F16` buffers, and the
    /// emitted kernel is `<16 x float>` throughout. Running it over a `F64`
    /// buffer would reinterpret the data rather than fail.
    fn try_emit_gemm_kernel(&mut self, k: &KernelDecl) -> Option<crate::cpu_gemm::GemmShape> {
        // `Y_NO_GEMM_RECOGNISER=1` lowers the nest as written instead of
        // substituting the packed kernel.
        //
        // This exists so the compiler can be asked for BOTH readings of one
        // source: the optimized kernel, and the naive loop nest it claims to
        // be equal to. That pair is the differential in
        // `tests/gemm_substitution_differential.rs`, and it is the cheapest
        // honest form of the claim `docs/proof_carrying_kernels.md` eventually
        // wants to PROVE - the spec is the user's own source lowered by the
        // same compiler, so unlike a reference written inside a test it cannot
        // drift from the language's semantics.
        //
        // Deliberately an escape hatch and not a tuning knob: it makes Y slow,
        // never wrong.
        if crate::cpu_gemm::recogniser_disabled() {
            return None;
        }
        let shape = crate::cpu_gemm::recognize_gemm(&k.body)?;

        let elem = |e: &Self, n: &String| e.mem_elem_types.get(n).cloned().unwrap_or_default();
        let (ea, eb, ec) = (
            elem(self, &shape.a),
            elem(self, &shape.b),
            elem(self, &shape.c),
        );

        // The EXACT path. `i16` operands accumulating into `i64` is precisely
        // `__y_gemm_exact_vnni`'s contract, so the source's own types state the
        // operand domain and there is nothing to convert.
        //
        // That is what makes the substitution legal at all. `VnniExact::license`
        // is stated over "the int16 operand values actually fed to `vpdpwssd`",
        // and an `F32` nest would need a quantization scale to reach that domain
        // - at which point the licence would have been granted against the
        // source's magnitude and not the kernel's. Declaring the operands `I16`
        // removes the question instead of answering it.
        if ea == "i16" && eb == "i16" && ec == "i64" {
            // THE PRODUCT MUST NOT TRUNCATE, and this is the check that makes
            // the whole substitution legal rather than merely fast.
            //
            // `let a_val: I16 = ...` makes `a_val * b_val` an i16 multiply.
            // 1024 * 1024 is 2^20, so it overflows, and the naive nest
            // accumulates the TRUNCATED product - the emitted IR is
            // `mul i16` followed by `sext i16 ... to i64`. `vpdpwssd` widens
            // internally, so substituting it there replaces a truncating
            // reduction with a widening one: a different function, computed
            // faster, under a certificate claiming exactness.
            //
            // Declaring the operands `I64` sign-extends at the load and makes
            // the multiply `i64`, which is what the kernel computes. Verified
            // by running both: the widened nest is bit-identical to an integer
            // reference and the truncating one is not.
            let widened = matches!(shape.operand_ty.as_deref(), Some("I64") | Some("I32"));
            if !widened {
                if shape.drift.is_some() {
                    self.drift_report.push(format!(
                        "matmul {}x{}: using scalar lowering. The exact vpdpwssd kernel is \
                         unavailable because the operands are declared `{}`, so `a * b` is a \
                         {}-bit multiply that truncates before it is accumulated - the kernel \
                         widens, so substituting it would compute a DIFFERENT function. \
                         Declare the operand `let`s as `I64` to state the widening.",
                        shape.m,
                        shape.n,
                        shape.operand_ty.as_deref().unwrap_or("?"),
                        16
                    ));
                }
                return None;
            }
            if let Some(drift) = &shape.drift {
                match crate::cpu_gemm::plan_exact_gemm(drift) {
                    crate::cpu_gemm::ExactGemmPlan::Vnni {
                        scheme,
                        operand_magnitude,
                    } => {
                        self.emit_exact_gemm_call(&shape, scheme.flush_k_pairs)?;
                        // The certificate is recorded HERE, at the one site
                        // where the substitution actually happens, so it can
                        // neither be emitted for a nest that stayed on the
                        // scalar path nor forgotten for one that did not.
                        self.exact_gemm_certificates.push(
                            crate::exact_gemm_certificate::Certificate {
                                operand_magnitude,
                                flush_k_pairs: scheme.flush_k_pairs,
                                extent_m: shape.m.clone(),
                                extent_n: shape.n.clone(),
                            },
                        );
                        self.drift_report.push(format!(
                            "matmul {}x{}: EXACT vpdpwssd kernel substituted (operands \
                             |x| <= {}, flush every {} k-pairs). Integer addition is \
                             associative, so the tiled, K-split result is bit-identical to \
                             the naive nest rather than merely close to it.",
                            shape.m, shape.n, operand_magnitude, scheme.flush_k_pairs
                        ));
                        return Some(shape);
                    }
                    crate::cpu_gemm::ExactGemmPlan::Unavailable(reason) => {
                        // Still exact, just not fast - the scalar lowering
                        // honours `@ZeroDrift` on its own. An advisory, not an
                        // error; see `ExactGemmPlan`.
                        self.drift_report.push(format!(
                            "matmul {}x{}: using scalar lowering, which is still EXACT. The \
                             fast vpdpwssd kernel is unavailable because {}",
                            shape.m, shape.n, reason
                        ));
                        return None;
                    }
                }
            }
            // Integer buffers with no `@ZeroDrift`: the nest is an ordinary
            // integer matmul and the f32 kernel below cannot serve it.
            return None;
        }

        // The f32 path. The element-type check is not a formality: the
        // recogniser matches on loop STRUCTURE, which is identical for `F64` or
        // `F16` buffers, and the emitted kernel is `<16 x float>` throughout.
        // Running it over an `F64` buffer would reinterpret the data rather
        // than fail.
        if ea != "float" || eb != "float" || ec != "float" {
            return None;
        }

        // A `@ZeroDrift` accumulator demands an EXACT reduction. The packed
        // kernel below accumulates in f32, which is not exact, so substituting
        // it here would hand back a fast kernel that quietly fails the
        // guarantee the source asked for — the exact failure mode this
        // repository's design rule exists to prevent.
        //
        // `recognize_gemm` used to refuse such a nest outright; it now records
        // the request so an exact kernel can be selected. Until that kernel
        // exists, returning None falls through to ordinary scalar lowering,
        // which honours `@ZeroDrift` correctly (see `Stmt::Let` with
        // `zero_drift` in `emit_alloca_for_block`). Slow and right, rather than
        // fast and wrong. See `docs/proof_carrying_kernels.md`, Phase 0, and
        // `docs/deterministic_inference.md`, M0.
        //
        // The licence is consulted and REPORTED even though no exact kernel is
        // emitted yet, because the two outcomes are already distinguishable and
        // the user can act on the difference: an unlicensed nest is on the slow
        // path for a stated reason it can fix (tighter `@bounds` on the
        // operands), while a licensed one is merely waiting on the kernel. A
        // silent `return None` tells them neither. Both are `drift_report`
        // advisories rather than `emit_errors` - see `ExactGemmPlan`, where the
        // distinction between "cannot be exact" and "cannot be exact AND fast"
        // is written down.
        if let Some(drift) = &shape.drift {
            match crate::cpu_gemm::plan_exact_gemm(drift) {
                crate::cpu_gemm::ExactGemmPlan::Vnni { scheme, operand_magnitude } => {
                    self.drift_report.push(format!(
                        "matmul {}x{}: exact vpdpwssd kernel is LICENSED (operands |x| <= {}, \
                         flush every {} k-pairs) but not yet implemented - using scalar lowering, \
                         which is exact and slow",
                        shape.m, shape.n, operand_magnitude, scheme.flush_k_pairs
                    ));
                }
                crate::cpu_gemm::ExactGemmPlan::Unavailable(reason) => {
                    self.drift_report.push(format!(
                        "matmul {}x{}: using scalar lowering, which is still EXACT. The fast \
                         vpdpwssd kernel is unavailable because {}",
                        shape.m, shape.n, reason
                    ));
                }
            }
            return None;
        }

        // The extents and the three leading dimensions arrive as i32
        // parameters; the kernel indexes in i64.
        //
        // The strides are loaded SEPARATELY even when they name the same
        // variables as the extents (the packed case, where `lda` is `K`).
        // Reusing the extent's register would be correct today and would
        // silently stop being correct the moment the recogniser accepts a
        // stride the extent does not equal — which is now the whole point.
        let mut ext = Vec::new();
        for name in [
            &shape.m,
            &shape.n,
            &shape.k,
            &shape.lda,
            &shape.ldb,
            &shape.ldc,
        ] {
            let ty = self.locals.get(name)?.clone();
            let tmp = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = load {}, ptr %{}", tmp, ty, name).unwrap();
            ext.push(if ty == "i64" {
                tmp
            } else {
                let w = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = sext {} {} to i64", w, ty, tmp).unwrap();
                w
            });
        }

        let mut ptrs = Vec::new();
        for name in [&shape.a, &shape.b, &shape.c] {
            let tmp = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = load ptr, ptr %{}", tmp, name).unwrap();
            ptrs.push(tmp);
        }

        writeln!(
            &mut self.output,
            "  call void @{}(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
             i64 {}, i64 {}, i64 {})",
            crate::cpu_gemm::KERNEL_NAME,
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ext[0],
            ext[1],
            ext[2],
            ext[3],
            ext[4],
            ext[5]
        )
        .unwrap();
        Some(shape)
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
            Stmt::Let { name, init, .. } if self.zero_drift.contains_key(name) => {
                let repr = self.zero_drift[name];
                let as_double = match init {
                    Some(e) => self.emit_expr_as_double(e),
                    None => "0.0".to_string(),
                };
                let fixed = self.emit_to_fixed(&as_double, repr);
                self.emit_store(&fixed, &format!("%{}", name), repr.llvm_type());
            }
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
                        let src_unsigned = self.expr_is_unsigned(init_expr);
                        let coerced =
                            self.emit_coerce_from(&val, &val_ty, &dst_ty, src_unsigned);

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
            // `sum = sum + rhs` on a @ZeroDrift accumulator.
            //
            // Only `+=` had a drift-aware arm, so this form - the one the GEMM
            // recogniser matches, and the one every reduction in `tests/` is
            // written with - fell through to the ordinary assignment path. That
            // path reads the accumulator through `sitofp`, adds in `double`, and
            // then emits `store double` into an `alloca i64`. LLVM allows it,
            // because pointers are untyped, so `clang` says nothing and the
            // next read interprets the double's BIT PATTERN as an integer:
            // `sum` came back as -4337501956902952448 where the answer was
            // 2896931.
            //
            // Same shape as the ZK emitter's `x += 5` versus `x = x + 5`,
            // recorded in the design-rule table - one spelling handled, its
            // sibling silently wrong - found here in the other direction.
            //
            // Routed into the same exact integer path as `+=`: the term is
            // quantised once and every addition after that is integer, which is
            // what makes the total independent of the order the terms arrived
            // in.
            Stmt::Assign { target, value, span }
                if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n))
                    && Self::drift_running_sum(target, value).is_some() =>
            {
                let name = match target {
                    Expr::Ident(n, _) => n.clone(),
                    _ => unreachable!(),
                };
                let (op, rhs) = Self::drift_running_sum(target, value).unwrap();
                let repr = self.zero_drift[&name];
                let rhs_double = self.emit_expr_as_double(rhs);
                let rhs_fixed = self.emit_to_fixed(&rhs_double, repr);
                let ity = repr.llvm_type();
                let addr = format!("%{}", name);
                let loaded = self.emit_load(&addr, ity);
                let result = self.fresh_tmp();
                let instr = if matches!(op, BinaryOp::Sub) { "sub" } else { "add" };
                writeln!(
                    &mut self.output,
                    "  {} = {} {} {}, {}",
                    result, instr, ity, loaded, rhs_fixed
                )
                .unwrap();
                writeln!(&mut self.output, "  store {} {}, ptr {}", ity, result, addr).unwrap();
                let _ = span;
            }
            // Any OTHER assignment to a drift accumulator is refused rather
            // than converted. `sum = <expr>` that is not a running sum would
            // have to round `<expr>` into the fixed domain, and whether that is
            // exact depends on the expression - which is precisely the
            // judgement the design rule forbids a backend from making silently.
            Stmt::Assign { target, span, .. }
                if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n)) =>
            {
                let name = match target {
                    Expr::Ident(n, _) => n.clone(),
                    _ => unreachable!(),
                };
                self.emit_errors.push(format!(
                    "Line {}: `{}` is a @ZeroDrift accumulator, so the only assignments that \
preserve drift-freedom are `{} = {} + <term>` and `{} = {} - <term>` (or `+=` / `-=`). \
Assigning anything else would have to round the value into the accumulator's exact \
representation, and whether that is lossless depends on the expression.",
                    span.line, name, name, name, name, name
                ));
            }
            Stmt::Assign { target, value, .. } => {
                let target_addr = self.emit_lvalue(target);
                let dst_ty = self.infer_type(target);
                let val = self.emit_expr(value, Some(target_addr.clone()), Some(dst_ty.clone()));
                let val_ty = self.infer_type(value);

                if matches!(value, Expr::ZeroInit(_)) {
                    // ZeroInit handles memset directly into target_addr.
                } else {
                    let src_unsigned = self.expr_is_unsigned(value);
                    let coerced = self.emit_coerce_from(&val, &val_ty, &dst_ty, src_unsigned);

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
                        let attrs = self.get_expr_attrs(target);
                        self.emit_store_with_attrs(&coerced, &target_addr, &dst_ty, attrs);
                    }
                }
            }
            Stmt::Return(expr, _) => {
                if let Some(e) = expr {
                    let val = self.emit_expr(e, None, None);
                    let val_ty = self.infer_type(e);
                    let src_unsigned = self.expr_is_unsigned(e);
                    let coerced = self.emit_coerce_from(&val, &val_ty, ret_type, src_unsigned);
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
                is_uniform_branch,
                ..
            } => {
                let cond = self.emit_expr(condition, None, None);
                let then_lbl = self.fresh_label("then");
                let else_lbl = self.fresh_label("else");
                let merge_lbl = self.fresh_label("merge");

                let metadata = if *is_uniform_branch {
                    ", !uniform_branch !0 ; Maps to BRANCH_UNIFORM_CYCLES scheduling baseline"
                } else {
                    ""
                };

                writeln!(
                    &mut self.output,
                    "  br i1 {}, label %{}, label %{}{}",
                    cond,
                    then_lbl,
                    if else_block.is_some() {
                        &else_lbl
                    } else {
                        &merge_lbl
                    },
                    metadata
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
            Stmt::Break { .. } => {
                if let Some(end_lbl) = self.loop_exit_stack.last() {
                    writeln!(&mut self.output, "  br label %{}", end_lbl).unwrap();
                    self.block_terminated = true;
                } else {
                    panic!("'break' statement outside of loop");
                }
            }
            Stmt::While {
                condition, body, is_uniform_branch, ..
            } => {
                let cond_lbl = self.fresh_label("while.cond");
                let body_lbl = self.fresh_label("while.body");
                let end_lbl = self.fresh_label("while.end");

                let metadata = if *is_uniform_branch {
                    ", !uniform_branch !0 ; Maps to BRANCH_UNIFORM_CYCLES scheduling baseline"
                } else {
                    ""
                };

                writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();
                writeln!(&mut self.output, "{}:", cond_lbl).unwrap();
                let cond = self.emit_expr(condition, None, None);
                writeln!(
                    &mut self.output,
                    "  br i1 {}, label %{}, label %{}{}",
                    cond, body_lbl, end_lbl, metadata
                )
                .unwrap();

                writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                self.block_terminated = false;
                self.loop_exit_stack.push(end_lbl.clone());
                self.emit_block_body(body, ret_type);
                self.loop_exit_stack.pop();
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
                is_uniform_branch,
                tile,
                prefetch_stride,
                ..
            } => {
                let s = self.emit_expr(start, None, None);
                let e = self.emit_expr(end, None, None);
                let cond_lbl = self.fresh_label("for.cond");
                let body_lbl = self.fresh_label("for.body");
                let end_lbl = self.fresh_label("for.end");

                if let Some(t) = tile {
                    writeln!(&mut self.output, "  ; [Y TILE OPTIMIZATION] Tiled loop dimensions: M={:?}, N={:?}, K={:?}", t.block_m, t.block_n, t.block_k).unwrap();
                }

                if let Some(pf) = prefetch_stride {
                    if let Some(ref stride_expr) = pf.stride {
                        writeln!(&mut self.output, "  ; [Y PREFETCH] @prefetch_stride({:?}) -- solver-guided cache warming", stride_expr).unwrap();
                    } else {
                        writeln!(&mut self.output, "  ; [Y PREFETCH] @prefetch_stride(auto) -- compiler deduces optimal prefetch distance").unwrap();
                    }
                }

                let metadata = if *is_uniform_branch {
                    ", !uniform_branch !0 ; Maps to BRANCH_UNIFORM_CYCLES scheduling baseline"
                } else {
                    ""
                };

                // alloca is in entry
                self.emit_store(&s, &format!("%{}", loop_var), "i32");
                writeln!(&mut self.output, "  br label %{}", cond_lbl).unwrap();

                writeln!(&mut self.output, "{}:", cond_lbl).unwrap();
                let cur = self.emit_load(&format!("%{}", loop_var), "i32");
                let cmp = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = icmp slt i32 {}, {}", cmp, cur, e).unwrap();
                writeln!(
                    &mut self.output,
                    "  br i1 {}, label %{}, label %{}{}",
                    cmp, body_lbl, end_lbl, metadata
                )
                .unwrap();

                writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                self.block_terminated = false;
                self.loop_exit_stack.push(end_lbl.clone());
                self.emit_block_body(body, ret_type);
                self.loop_exit_stack.pop();

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
            Stmt::CompoundAssign { target, op, value, span }
                if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n)) =>
            {
                let name = match target {
                    Expr::Ident(n, _) => n.clone(),
                    _ => unreachable!(),
                };
                let repr = self.zero_drift[&name];
                // Only `+=` and `-=` are exact here. Scaling a product or a
                // quotient would reintroduce rounding into the accumulation
                // itself, which is the single thing @ZeroDrift exists to
                // prevent, so it is refused rather than silently approximated.
                if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                    self.emit_errors.push(format!(
                        "Line {}: `{:?}=` is not exact on the @ZeroDrift accumulator `{}`. Only \
`+=` and `-=` preserve drift-freedom.",
                        span.line, op, name
                    ));
                }
                // Each term is quantised once, deterministically; every
                // addition after that is exact integer arithmetic, so the total
                // does not depend on the order the terms arrived in.
                let rhs_double = self.emit_expr_as_double(value);
                let rhs_fixed = self.emit_to_fixed(&rhs_double, repr);
                let ity = repr.llvm_type();
                let addr = format!("%{}", name);
                let loaded = self.emit_load(&addr, ity);
                let result = self.fresh_tmp();
                let instr = if matches!(op, BinaryOp::Sub) { "sub" } else { "add" };
                writeln!(
                    &mut self.output,
                    "  {} = {} {} {}, {}",
                    result, instr, ity, loaded, rhs_fixed
                )
                .unwrap();
                self.emit_store(&result, &addr, ity);
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
                if self.in_ptx_emit {
                    // PTX-targeted: emit chisel string literals as NVPTX inline asm
                    self.wln("  ; --- CHISEL INLINE PTX (nvptx target) ---");
                    for stmt in &block.stmts {
                        if let Stmt::Expr(Expr::StringLit(s, _)) = stmt {
                            // Emit PTX instruction as side-effecting inline asm
                            // On nvptx backend, constraints differ from x86
                            self.wln(&format!(
                                "  call void asm sideeffect \"{}\", \"\"()",
                                s.replace('\\', "\\\\").replace('"', "\\\"")
                            ));
                        } else {
                            self.emit_stmt(stmt, ret_type);
                        }
                    }
                    self.wln("  ; --- END CHISEL PTX ---");
                } else {
                    self.wln("  ; --- CHISEL INLINE ASM ---");
                    for stmt in &block.stmts {
                        if let Stmt::Expr(Expr::StringLit(s, _)) = stmt {
                            self.wln(&format!("  call void asm sideeffect \"{}\", \"~{{memory}},~{{dirflag}},~{{fpsr}},~{{flags}}\"()", s));
                        } else {
                            self.emit_stmt(stmt, ret_type);
                        }
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
            Stmt::GhostBlock(block, _) => {
                self.wln("  ; --- @ghost speculative block ---");
                self.emit_block_body(block, ret_type);
            }
            Stmt::HintBlock { body, .. } => {
                self.wln("  ; --- @hint unconstrained block ---");
                self.emit_block_body(body, ret_type);
            }
            Stmt::ClockDomainBlock { body, .. } => {
                self.wln("  ; --- @clock_domain block ---");
                self.emit_block_body(body, ret_type);
            }
            Stmt::CompileTimeAssert { .. } => {
                // compile_time::assert! is verified at compile time and stripped
                // from the final binary -- zero runtime cost.
                self.wln("  ; [compile_time::assert! verified and stripped]");
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
                    let raw_base_val = self.emit_lvalue(base);
                    let base_ast_ty = self.infer_ast_type(base);
                    if base_ast_ty.starts_with('&') {
                        let loaded = self.emit_load(&raw_base_val, "ptr");
                        (loaded, self.infer_struct_type(base))
                    } else {
                        (raw_base_val, self.infer_struct_type(base))
                    }
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
            Expr::Index { base, index, span } => {
                let base_val = self.emit_expr(base, None, None);
                let idx_val = self.emit_expr(index, None, None);
                let base_ty = self.infer_type(base);
                let idx_ty = self.infer_type(index);
                
                let is_safe = crate::type_checker::SAFE_INDICES.with(|set| {
                    set.borrow().contains(&(span.line, span.col))
                });
                let array_size = crate::type_checker::INDEX_ARRAY_SIZES.with(|map| {
                    map.borrow().get(&(span.line, span.col)).cloned()
                });
                
                if !is_safe {
                    if let Some(size) = array_size {
                        let cmp_ty = if idx_ty == "i32" { "i32" } else { "i64" };
                        let ok_lbl = self.fresh_label("bounds_ok");
                        let fail_lbl = self.fresh_label("bounds_fail");
                        let cond = self.fresh_tmp();
                        writeln!(
                            &mut self.output,
                            "  {} = icmp uge {} {}, {}",
                            cond, cmp_ty, idx_val, size
                        ).unwrap();
                        writeln!(
                            &mut self.output,
                            "  br i1 {}, label %{}, label %{}",
                            cond, fail_lbl, ok_lbl
                        ).unwrap();
                        
                        // Fail block
                        writeln!(&mut self.output, "{}:", fail_lbl).unwrap();
                        let idx_i64 = if cmp_ty == "i32" {
                            let tmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = sext i32 {} to i64", tmp, idx_val).unwrap();
                            tmp
                        } else {
                            idx_val.clone()
                        };
                        let print_tmp = self.fresh_tmp();
                        writeln!(
                            &mut self.output,
                            "  {} = call i32 (ptr, ...) @printf(ptr @.str.bounds_err, i64 {}, i64 {})",
                            print_tmp, idx_i64, size
                        ).unwrap();
                        writeln!(&mut self.output, "  call void @exit(i32 1)").unwrap();
                        writeln!(&mut self.output, "  unreachable").unwrap();
                        
                        // Ok block
                        writeln!(&mut self.output, "{}:", ok_lbl).unwrap();
                        self.block_terminated = false;
                    }
                }

                let elem_ty = if base_ty == "ptr" {
                    "i64".to_string()
                } else if base_ty.starts_with('[') && base_ty.ends_with(']') {
                    if let Some(pos) = base_ty.rfind(' ') {
                        base_ty[pos + 1..base_ty.len() - 1].to_string()
                    } else {
                        "i64".to_string()
                    }
                } else {
                    base_ty.clone()
                };
                let tmp = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {} = getelementptr {}, ptr {}, {} {}",
                    tmp, elem_ty, base_val, idx_ty, idx_val
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

                // Reading a @ZeroDrift accumulator converts back out of its
                // integer domain. The stored value stays exact; only this
                // observation is a float, which is the type the source declared.
                if let Some(repr) = self.zero_drift.get(name).copied() {
                    let raw = self.emit_load(&format!("%{}", name), repr.llvm_type());
                    return self.emit_from_fixed(&raw, repr);
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
                let mut r_ty = self.infer_type(right);

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

                // 4. `i1` is NEVER an operand width. A comparison produces
                //    `i1`, and the promotion below only runs when the two
                //    types DIFFER -- so two comparisons combined with a third
                //    operator were left at `i1`, where LLVM's signed
                //    interpretation of `1` is **-1**:
                //
                //        ((v == v) < (v > v))   ->  icmp slt i1 1, 0  ->  TRUE
                //
                //    Arithmetic is no better: `add i1` wraps mod 2, so
                //    `(a > b) + (c > d)` could only ever be 0 or 1. Both are
                //    fixed by widening a boolean to a real integer first, and
                //    zero-extension is what makes `true` 1 rather than -1.
                //
                //    Found by `tests/backend_differential.rs` once its
                //    generator produced NESTED expressions -- flat ones cannot
                //    put a comparison in an operand position, so 400 programs
                //    had already passed clean.
                if l_ty == "i1" {
                    l = self.emit_coerce_from(&l, "i1", "i32", true);
                    l_ty = "i32".to_string();
                }
                if r_ty == "i1" {
                    r = self.emit_coerce_from(&r, "i1", "i32", true);
                    r_ty = "i32".to_string();
                }

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
                if let UnaryOp::Ref { .. } = op {
                    return self.emit_lvalue(operand);
                }

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
                    UnaryOp::Deref => {
                        let inner_ty = self.infer_type(operand);
                        let load_ty = if inner_ty == "ptr" {
                            self.pointee_llvm_type(expr)
                        } else {
                            inner_ty
                        };
                        writeln!(
                            &mut self.output,
                            "  {} = load {}, ptr {}",
                            tmp, load_ty, val
                        )
                        .unwrap();
                    }
                    UnaryOp::Ref { .. } => unreachable!(),
                }
                tmp
            }
            Expr::Call { func, args, .. } => {
                let func_name = self.emit_call_target(func);

                // Block-pointer intrinsics lower to native address arithmetic.
                // Routing them through the generic call path instead declared
                // them `i32 (...)`, which truncated the 64-bit base pointer and
                // then `sitofp`'d a float bit pattern — wrong answers, not just
                // slow ones. It also planted an opaque call in the innermost
                // loop, which blocks every loop and vector transform LLVM has.
                if let Some(v) = self.try_emit_block_ptr_intrinsic(&func_name, args) {
                    return v;
                }

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
                            "I16" | "u16" | "i16" => "i16".to_string(),
                            "F16" | "f16" => "half".to_string(),
                            "F32" | "f32" => "float".to_string(),
                            "F64" | "f64" => "double".to_string(),
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
                            let u = self.expr_is_unsigned(arg);
                            arg_val =
                                self.emit_coerce_from(&arg_val, &arg_ty, &llvm_param_ty, u);
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
                        return tmp.replace("%_t", "%_void");
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
                        "I16" | "u16" | "i16" => "i16".to_string(),
                        "F16" | "f16" => "half".to_string(),
                        "F32" | "f32" => "float".to_string(),
                        "F64" | "f64" => "double".to_string(),
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
                        let u = self.expr_is_unsigned(a);
                        arg_val = self.emit_coerce_from(&arg_val, &arg_ty, &llvm_param_ty, u);
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
                let attrs = self.get_expr_attrs(expr);
                if field_ty.starts_with('[') {
                    lval
                } else {
                    self.emit_load_with_attrs(&lval, &field_ty, attrs)
                }
            }
            Expr::Index { .. } => {
                let lval = self.emit_lvalue(expr);
                let ty = self.infer_type(expr);
                let attrs = self.get_expr_attrs(expr);
                self.emit_load_with_attrs(&lval, &ty, attrs)
            }
            Expr::SelfLit(_) => "%self".into(),
            Expr::ZeroInit(_) => {
                let ty = expected_ty
                    .or_else(|| self.current_load_hint.clone())
                    .unwrap_or_else(|| "i32".into());

                if target.is_none() && (ty.starts_with('[') || ty.starts_with('%')) {
                    return "zeroinitializer".into();
                }

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

                if ty.starts_with('[') || ty.starts_with('%') {
                    target_ptr
                } else {
                    self.emit_load(&target_ptr, &ty)
                }
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
                    let val = self.emit_expr(fexpr, None, Some(field_ty.clone()));
                    let mut val_ty = self.infer_type(fexpr);
                    if val == "zeroinitializer" {
                        val_ty = field_ty.clone();
                    }
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

    /// Element type for the buffer `expr` names, or `None` if it is not a
    /// `GlobalMemory<T>` / `SharedMemory<T>` binding we tracked.
    fn block_ptr_elem_ty(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident(name, _) => self.mem_elem_types.get(name).cloned(),
            _ => None,
        }
    }

    /// Emit `val` widened to i64, for address arithmetic and bounds tests.
    ///
    /// A 32-bit index is sign-extended, so a negative index becomes a large
    /// unsigned value and fails the `ult` bound test — matching the PTX
    /// backend's `setp.lt.u32`. Keeping the two backends agreeing on
    /// out-of-range behaviour is the point; a CPU run that silently accepted
    /// what the GPU masks off would be a portability trap.
    fn emit_as_i64(&mut self, e: &Expr) -> String {
        let v = self.emit_expr(e, None, None);
        let t = self.infer_type(e);
        if t == "i64" {
            v
        } else {
            self.emit_coerce(&v, &t, "i64")
        }
    }

    /// Lowers the `block_ptr2d_*` / `block_ptr3d_*` family to GEP + typed
    /// load/store. Returns `None` for anything that is not one of them, so the
    /// caller falls through to the ordinary call path.
    fn try_emit_block_ptr_intrinsic(&mut self, name: &str, args: &[Expr]) -> Option<String> {
        let (is_load, dims) = match name {
            "block_ptr2d_load" | "make_block_ptr2d" => (true, 2usize),
            "block_ptr2d_store" => (false, 2),
            "block_ptr3d_load" => (true, 3),
            "block_ptr3d_store" => (false, 3),
            _ => return None,
        };

        // 2D: (base, row, col, stride, max_r, max_c [, val])
        // 3D: (base, d0, d1, d2, s0, s1, D0, D1, D2 [, val])
        let want = if dims == 2 { 6 } else { 9 } + if is_load { 0 } else { 1 };
        if args.len() != want {
            self.emit_errors.push(format!(
                "`{}` takes {} arguments, got {}. Refusing rather than \
                 guessing an address — a wrong stride here is a silent \
                 wrong answer, not a crash.",
                name,
                want,
                args.len()
            ));
            return Some("0".into());
        }

        let Some(elem_ty) = self.block_ptr_elem_ty(&args[0]) else {
            self.emit_errors.push(format!(
                "`{}` needs its first argument to be a `GlobalMemory<T>` or \
                 `SharedMemory<T>` binding so the element type is known. \
                 Refusing rather than assuming F32.",
                name
            ));
            return Some("0".into());
        };

        let base = self.emit_expr(&args[0], None, None);

        // Linear element offset, and the in-bounds predicate, per dimension.
        // 2D: off = row*stride + col
        // 3D: off = d0*s0 + d1*s1 + d2
        let (idx_lo, idx_hi, stride_lo, stride_hi) = if dims == 2 {
            (1, 3, 3, 4)
        } else {
            (1, 4, 4, 6)
        };
        let idxs: Vec<String> = (idx_lo..idx_hi)
            .map(|i| self.emit_as_i64(&args[i]))
            .collect();
        let strides: Vec<String> = (stride_lo..stride_hi)
            .map(|i| self.emit_as_i64(&args[i]))
            .collect();
        let bounds: Vec<String> = (want - dims - usize::from(!is_load)
            ..want - usize::from(!is_load))
            .map(|i| self.emit_as_i64(&args[i]))
            .collect();

        // offset = sum(idx[i] * stride[i]) + idx[last]
        let mut off = String::new();
        for (i, s) in strides.iter().enumerate() {
            let m = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = mul nsw i64 {}, {}", m, idxs[i], s).unwrap();
            if off.is_empty() {
                off = m;
            } else {
                let a = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = add nsw i64 {}, {}", a, off, m).unwrap();
                off = a;
            }
        }
        let last = idxs.last().unwrap().clone();
        let off = if off.is_empty() {
            last
        } else {
            let a = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = add nsw i64 {}, {}", a, off, last).unwrap();
            a
        };

        // Bounds predicate: every index unsigned-less-than its extent.
        let mut ok = String::new();
        for (i, b) in bounds.iter().enumerate() {
            let c = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = icmp ult i64 {}, {}", c, idxs[i], b).unwrap();
            if ok.is_empty() {
                ok = c;
            } else {
                let a = self.fresh_tmp();
                writeln!(&mut self.output, "  {} = and i1 {}, {}", a, ok, c).unwrap();
                ok = a;
            }
        }

        if is_load {
            // Redirect an out-of-range load to element 0 so the access is
            // always defined, then select the masked result. Selecting the
            // offset rather than branching keeps the loop vectorizable.
            let soff = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {} = select i1 {}, i64 {}, i64 0",
                soff, ok, off
            )
            .unwrap();
            let p = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {} = getelementptr inbounds {}, ptr {}, i64 {}",
                p, elem_ty, base, soff
            )
            .unwrap();
            let raw = self.fresh_tmp();
            writeln!(&mut self.output, "  {} = load {}, ptr {}", raw, elem_ty, p).unwrap();
            let zero = if elem_ty.starts_with('f') || elem_ty == "double" || elem_ty == "half" {
                "0.0"
            } else {
                "0"
            };
            let out = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {} = select i1 {}, {} {}, {} {}",
                out, ok, elem_ty, raw, elem_ty, zero
            )
            .unwrap();
            return Some(out);
        }

        // Store: an out-of-range write must not land anywhere real, so the
        // *pointer* is selected rather than the offset.
        let val_expr = &args[want - 1];
        let val = self.emit_expr(val_expr, None, None);
        let val_ty = self.infer_type(val_expr);
        let val = if val_ty == elem_ty {
            val
        } else {
            self.emit_coerce(&val, &val_ty, &elem_ty)
        };
        let p = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = getelementptr inbounds {}, ptr {}, i64 {}",
            p, elem_ty, base, off
        )
        .unwrap();
        // An out-of-range write is redirected into a dead stack slot rather
        // than branched around, so the loop stays vectorizable. The slot's
        // address never escapes, so LLVM folds both it and the select away
        // whenever it can prove the index is in range — the common case inside
        // `for i in 0..M` with `max_r = M`.
        let sink = Y_OOB_SINK;
        let dst = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = select i1 {}, ptr {}, ptr {}",
            dst, ok, p, sink
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  store {} {}, ptr {}",
            elem_ty, val, dst
        )
        .unwrap();
        Some("0".into())
    }

    fn emit_call_target(&self, func: &Expr) -> String {
        match func {
            Expr::Ident(name, _) => {
                if name == "main" {
                    "ysu_main".to_string()
                } else {
                    name.clone()
                }
            }
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
                op: UnaryOp::Ref { mutable },
                operand,
                ..
            } => {
                let inner = self.infer_ast_type(operand);
                format!("&{}{}", if *mutable { "mut " } else { "" }, inner)
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
            // **Typed by VALUE, not fixed at i32.** This is the same bug the PTX
            // backend had and fixed (see CLAUDE.md gotcha #7): a literal above
            // `i32::MAX` typed as i32 is NEGATIVE, and widening it sign-extends.
            // Measured before the fix, both compiling cleanly and running:
            //   `let a: I64 = 4294967296; if a > 0 { 1 } else { 0 }`  ->  0
            //   `let a: I64 = 3000000000; if a > 0 { 1 } else { 0 }`  ->  0
            // The first truncates to zero, the second sign-extends to a negative
            // i64. Nothing in the pipeline rejects `store i32 4294967296` or
            // `sext i32 4294967296 to i64` - clang accepts both and the program
            // runs, which is why this survived.
            Expr::IntLit(v, _) => {
                if *v > i32::MAX as i64 || *v < i32::MIN as i64 {
                    "i64".into()
                } else {
                    "i32".into()
                }
            }
            Expr::FloatLit(_, _) => "double".into(),
            Expr::BoolLit(_, _) => "i1".into(),
            Expr::CharLit(_, _) => "i8".into(),
            Expr::StringLit(_, _) => "ptr".into(),
            Expr::Ident(name, _) => {
                // A @ZeroDrift accumulator is STORED as an integer but READS as
                // a double. Reporting the storage type here would make every
                // downstream operation emit integer arithmetic against a value
                // that arrives as a float.
                if self.zero_drift.contains_key(name) {
                    return "double".into();
                }
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
                // A block-pointer load yields the buffer's element type. Falling
                // through to the `i32` default here is what put a `sitofp` on
                // an f32 *bit pattern* — an integer conversion of a value that
                // was already the right type, producing garbage silently.
                if let Expr::Call { args, .. } = expr {
                    if matches!(
                        func_name.as_str(),
                        "block_ptr2d_load" | "make_block_ptr2d" | "block_ptr3d_load"
                    ) {
                        if let Some(t) = args.first().and_then(|a| self.block_ptr_elem_ty(a)) {
                            return t;
                        }
                    }
                    if matches!(func_name.as_str(), "block_ptr2d_store" | "block_ptr3d_store") {
                        return "void".into();
                    }
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
                    _ => {
                        if func_name.starts_with("Vec_get_") {
                            let ret_type_name = &func_name[8..];
                            match ret_type_name {
                                "usize" | "I64" | "i64" => "i64".to_string(),
                                "I32" | "i32" | "int" => "i32".to_string(),
                                "bool" => "i1".to_string(),
                                "char" => "i8".to_string(),
                                "String" | "Vec" | "ptr" => "ptr".to_string(),
                                _ => format!("%{}", ret_type_name),
                            }
                        } else {
                            self.functions
                                .get(&func_name)
                                .map(|(_, r)| r.clone())
                                .unwrap_or_else(|| "i32".into())
                        }
                    }
                }
            }
            Expr::GenericCall { func, .. } => {
                let func_name = self.emit_call_target(func);
                self.functions
                    .get(&func_name)
                    .map(|(_, r)| r.clone())
                    .unwrap_or_else(|| "i32".into())
            }
            Expr::BinaryOp { op, left, right, .. } => match op {
                BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::Le
                | BinaryOp::Ge => "i1".into(),
                _ => {
                    // `i1` is never an operand width -- see the matching
                    // promotion in the emission path. This must agree with it
                    // exactly: the emitter widens the operands and emits
                    // `add i32`, so reporting `i1` here makes the CALLER emit
                    // `zext i1 %t` on an i32 register and clang rejects the
                    // module. `(5 > 3) + (9 > 1)` was the case that caught it,
                    // and it is the reason the fix is two sites and not one.
                    let widen = |t: String| if t == "i1" { "i32".to_string() } else { t };
                    let l_ty = widen(self.infer_type(left));
                    let r_ty = widen(self.infer_type(right));
                    if l_ty == r_ty {
                        l_ty
                    } else {
                        let l_is_float = l_ty == "float" || l_ty == "double" || l_ty == "half";
                        let r_is_float = r_ty == "float" || r_ty == "double" || r_ty == "half";
                        if l_is_float && r_is_float {
                            let l_bits = if l_ty == "double" { 64 } else if l_ty == "float" { 32 } else { 16 };
                            let r_bits = if r_ty == "double" { 64 } else if r_ty == "float" { 32 } else { 16 };
                            if l_bits >= r_bits { l_ty } else { r_ty }
                        } else if l_is_float {
                            l_ty
                        } else if r_is_float {
                            r_ty
                        } else {
                            let l_bits = Self::int_bits(&l_ty);
                            let r_bits = Self::int_bits(&r_ty);
                            if l_bits >= r_bits { l_ty } else { r_ty }
                        }
                    }
                }
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
                UnaryOp::Ref { .. } => "ptr".into(),
                UnaryOp::Deref => {
                    let inner_ty = self.infer_type(operand);
                    if inner_ty == "ptr" {
                        self.pointee_llvm_type(expr)
                    } else {
                        inner_ty
                    }
                }
                UnaryOp::Neg | UnaryOp::Not => self.infer_type(operand),
            },
            Expr::Index { base, .. } => {
                let base_ty = self.infer_type(base);
                if base_ty == "ptr" {
                    self.pointee_llvm_type(expr)
                } else if base_ty.starts_with('[') {
                    if let Some(pos) = base_ty.find('x') {
                        base_ty[pos + 1..].trim().trim_end_matches(']').to_string()
                    } else {
                        "i64".into()
                    }
                } else {
                    base_ty
                }
            }
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

    /// The LLVM type that a `ptr`-typed expression points at.
    ///
    /// `ast_type_to_llvm_type` answers `"i32"` for FIVE different reasons: a
    /// genuine `I32`, an `Unknown` ast type, an empty one, an unregistered
    /// type name, and a data-less enum. So its callers could not tell success
    /// from failure and used `resolved != "i32"` as a stand-in for "resolution
    /// succeeded", substituting `i64` whenever it came back `i32`.
    ///
    /// That discarded the CORRECT answer for the commonest pointer in the
    /// language. `fn g(r: &mut I32) { *r = 7; }` emitted
    ///
    /// ```text
    /// %_t2 = sext i32 7 to i64
    /// store i64 %_t2, ptr %_t1
    /// ```
    ///
    /// an EIGHT-byte store through a pointer to four bytes. That is valid IR,
    /// so `clang` accepts it without a word and the compiler printed
    /// "Compilation Successful!"; it overwrites whatever sits next to the
    /// target. With `struct Pair { a: I32, b: I32 }`, writing through
    /// `&mut p.a` set `p.b` to zero.
    ///
    /// The sentinel belongs on the AST type, which *does* distinguish the
    /// cases. `i64` is kept as the fallback for a genuinely unresolvable
    /// pointee - it is what this code has always guessed, and narrowing it is
    /// a separate question from not discarding a known answer.
    fn pointee_llvm_type(&self, expr: &Expr) -> String {
        let ast_ty = self.infer_ast_type(expr);
        if ast_ty == "Unknown" || ast_ty.is_empty() {
            return "i64".into();
        }
        let resolved = self.ast_type_to_llvm_type(&ast_ty);
        if resolved.is_empty() {
            "i64".into()
        } else {
            resolved
        }
    }

    fn ast_type_to_llvm_type(&self, ast_ty: &str) -> String {
        if ast_ty == "Unknown" || ast_ty.is_empty() {
            return "i32".into();
        }
        if ast_ty.starts_with('&') || ast_ty.starts_with('*') {
            return "ptr".into();
        }
        let clean = ast_ty.trim_start_matches("mut ").trim();
        if clean == "Vec" || clean.starts_with("Vec<") || clean == "String" || clean.starts_with("String<") || clean == "Option" || clean.starts_with("Option<") {
            return "ptr".into();
        }
        match clean {
            "I32" | "u32" | "i32" => "i32".into(),
            "I64" | "usize" | "i64" => "i64".into(),
            "F16" | "f16" => "half".into(),
            "F32" | "f32" => "float".into(),
            "F64" | "f64" => "double".into(),
            "bool" => "i1".into(),
            "char" | "i8" | "u8" => "i8".into(),
            "I16" | "u16" | "i16" => "i16".into(),
            "ptr" => "ptr".into(),
            other => {
                if other.is_empty() {
                    "i32".into()
                } else if let Some(has_data) = self.enums.get(other) {
                    if *has_data {
                        format!("%{}", other)
                    } else {
                        "i32".into()
                    }
                } else if self.structs.contains_key(other) {
                    format!("%{}", other)
                } else {
                    if other.contains('<') {
                        "ptr".into()
                    } else {
                        "i32".into()
                    }
                }
            }
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

